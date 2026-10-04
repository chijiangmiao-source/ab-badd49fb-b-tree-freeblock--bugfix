"""Unit tests for the snapshot page-ownership auditor."""

import unittest

from app import fixtures
from app.fixtures import (
    PAGE_SIZE,
    SnapshotBuilder,
    _base_builder,
    encode_varint,
    valid_snapshot,
)
from app.sqlite_audit import audit_snapshot


class ValidSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.data, self.root = valid_snapshot()
        self.res = audit_snapshot(self.data, self.root)

    def test_accepted(self):
        self.assertEqual(self.res["verdict"], "accepted")
        self.assertIsNone(self.res["error"])
        self.assertEqual(self.res["page_size"], PAGE_SIZE)
        self.assertEqual(self.res["page_count"], 12)

    def test_unique_ownership(self):
        pages = [p["page"] for p in self.res["pages"]]
        self.assertEqual(len(pages), len(set(pages)))
        self.assertEqual(pages, sorted(pages))

    def test_page_kinds(self):
        kinds = {p["page"]: p["kind"] for p in self.res["pages"]}
        self.assertEqual(
            kinds,
            {
                2: "btree_root",
                11: "btree_interior",
                3: "btree_leaf",
                4: "btree_leaf",
                5: "btree_leaf",
                6: "overflow",
                7: "overflow",
                8: "freelist_trunk",
                9: "freelist_leaf",
                10: "freelist_leaf",
            },
        )

    def test_rowid_ranges(self):
        pages = {p["page"]: p for p in self.res["pages"]}
        self.assertEqual(pages[3]["rowid_range"], [1, 4])
        self.assertEqual(pages[4]["rowid_range"], [5, 8])
        self.assertEqual(pages[5]["rowid_range"], [9, 12])
        self.assertEqual(pages[11]["rowid_range"], [1, 8])
        self.assertEqual(pages[2]["rowid_range"], [1, 12])
        self.assertEqual(self.res["summary"]["rowid_min"], 1)
        self.assertEqual(self.res["summary"]["rowid_max"], 12)

    def test_overflow_chain(self):
        pages = {p["page"]: p for p in self.res["pages"]}
        self.assertEqual(self.res["summary"]["overflow_chains"], 1)
        self.assertEqual(self.res["summary"]["overflow_pages"], 2)
        self.assertIn("page 4 cell 2 overflow pointer", pages[6]["referenced_by"])
        self.assertIn("page 6 overflow next pointer", pages[7]["referenced_by"])

    def test_reference_sources(self):
        pages = {p["page"]: p for p in self.res["pages"]}
        self.assertEqual(pages[2]["referenced_by"], "requested table root")
        self.assertIn("page 2 cell 0 child pointer", pages[11]["referenced_by"])
        self.assertIn("page 2 right-most child pointer", pages[5]["referenced_by"])
        self.assertEqual(pages[8]["referenced_by"], "database header freelist head")
        self.assertIn("page 8 freelist leaf slot 1", pages[10]["referenced_by"])

    def test_freelist_accounted(self):
        self.assertEqual(self.res["summary"]["freelist_pages"], 3)
        self.assertEqual(self.res["summary"]["freelist_declared"], 3)

    def test_deterministic(self):
        self.assertEqual(self.res, audit_snapshot(self.data, self.root))


class InvalidScenarioTests(unittest.TestCase):
    """Every crafted violation must be rejected with the exact first evidence."""

    def test_scenarios(self):
        scenarios = fixtures.invalid_scenarios()
        expected_names = {
            "shared_overflow_page",
            "ancestor_backpointer",
            "key_bound_conflict",
            "live_page_on_freelist",
            "truncated_cell",
            "leaf_cell_overlap",
            "cell_pointer_into_freeblock",
            "root_out_of_range",
            "overflow_chain_overrun",
            "overflow_chain_truncated",
            "freelist_count_mismatch",
            "auto_vacuum",
            "reserved_bytes",
            "page_size_unsupported",
            "bad_magic",
        }
        self.assertEqual(set(scenarios), expected_names)
        for name, (data, root, code, page, offset) in scenarios.items():
            with self.subTest(name=name):
                res = audit_snapshot(data, root)
                self.assertEqual(res["verdict"], "rejected")
                err = res["error"]
                self.assertEqual(err["code"], code, err)
                self.assertEqual(err["page"], page, err)
                self.assertEqual(err["offset"], offset, err)
                if 0 <= offset < len(data):
                    self.assertEqual(
                        err["bytes_hex"], data[offset : offset + 16].hex(), err
                    )
                # Deterministic: same input, same first evidence.
                self.assertEqual(res, audit_snapshot(data, root))

    def test_shared_overflow_detail_names_both_references(self):
        data, root, *_ = fixtures.invalid_scenarios()["shared_overflow_page"]
        err = audit_snapshot(data, root)["error"]
        self.assertEqual(err["detail"]["first_kind"], "overflow")
        self.assertIn("page 4 cell 2", err["detail"]["first_referenced_by"])
        self.assertIn("page 5 cell 1", err["detail"]["second_referenced_by"])

    def test_ancestor_backpointer_detail_names_root(self):
        data, root, *_ = fixtures.invalid_scenarios()["ancestor_backpointer"]
        err = audit_snapshot(data, root)["error"]
        self.assertEqual(err["detail"]["first_kind"], "btree_root")
        self.assertEqual(err["detail"]["first_referenced_by"], "requested table root")


class HeaderTests(unittest.TestCase):
    def build(self, **tweaks):
        b, _ = _base_builder()
        for offset, blob in tweaks.get("patches", {}).items():
            b.patch_header(offset, blob)
        return b.build()

    def test_short_file(self):
        res = audit_snapshot(b"SQLite format 3\x00" + b"\x00" * 20, 1)
        self.assertEqual(res["error"]["code"], "HEADER_TOO_SHORT")
        self.assertEqual(res["error"]["offset"], 0)

    def test_page_size_256_rejected(self):
        data = self.build(patches={16: (256).to_bytes(2, "big")})
        res = audit_snapshot(data, 2)
        self.assertEqual(res["error"]["code"], "PAGE_SIZE_UNSUPPORTED")
        self.assertEqual(res["error"]["offset"], 16)

    def test_page_size_not_power_of_two_rejected(self):
        data = self.build(patches={16: (768).to_bytes(2, "big")})
        res = audit_snapshot(data, 2)
        self.assertEqual(res["error"]["code"], "PAGE_SIZE_UNSUPPORTED")

    def test_incremental_vacuum_flag_rejected(self):
        data = self.build(patches={64: (1).to_bytes(4, "big")})
        res = audit_snapshot(data, 2)
        self.assertEqual(res["error"]["code"], "AUTO_VACUUM_ENABLED")
        self.assertEqual(res["error"]["offset"], 64)

    def test_truncated_last_page(self):
        data, _ = valid_snapshot()
        res = audit_snapshot(data[:-17], 2)
        self.assertEqual(res["error"]["code"], "TRUNCATED_PAGE")
        self.assertEqual(res["error"]["page"], 12)
        self.assertEqual(res["error"]["offset"], 11 * PAGE_SIZE)

    def test_root_page_zero_rejected_by_bounds(self):
        data, _ = valid_snapshot()
        res = audit_snapshot(data, 0)
        self.assertEqual(res["error"]["code"], "ROOT_PAGE_OUT_OF_RANGE")


class BtreeStructureTests(unittest.TestCase):
    def test_index_page_rejected_as_table_root(self):
        b, _ = _base_builder()
        b.page(2)[0] = 0x02  # index interior page
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "NOT_A_TABLE_BTREE_PAGE")
        self.assertEqual(res["error"]["page"], 2)
        self.assertEqual(res["error"]["offset"], PAGE_SIZE)

    def test_pointer_array_overflow(self):
        b, _ = _base_builder()
        b.page(3)[3:5] = (4000).to_bytes(2, "big")  # impossible cell count
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "CELL_POINTER_ARRAY_OVERFLOW")
        self.assertEqual(res["error"]["page"], 3)
        self.assertEqual(res["error"]["offset"], 2 * PAGE_SIZE + 3)

    def test_cell_pointer_out_of_bounds(self):
        b, _ = _base_builder()
        b.page(3)[8:10] = (5).to_bytes(2, "big")  # points into the page header
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "CELL_POINTER_OUT_OF_BOUNDS")
        self.assertEqual(res["error"]["page"], 3)
        self.assertEqual(res["error"]["offset"], 2 * PAGE_SIZE + 8)

    def test_child_page_out_of_range(self):
        b, _ = _base_builder()
        root = b.add_table_interior(2, [8], [11, 99])
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "CHILD_PAGE_OUT_OF_RANGE")
        self.assertEqual(res["error"]["page"], 99)
        self.assertEqual(res["error"]["offset"], PAGE_SIZE + root["right_ptr_offset"])

    def test_rowid_not_increasing_within_leaf(self):
        b, _ = _base_builder()
        info = b.add_table_leaf(3, [(1, b"a"), (3, b"b"), (2, b"c")])
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "ROWID_NOT_INCREASING")
        self.assertEqual(res["error"]["page"], 3)
        self.assertEqual(res["error"]["offset"], 2 * PAGE_SIZE + info[2]["ptr"])

    def test_overlapping_cells_rejected_with_stable_pointer_location(self):
        # Rowids are strictly increasing; only the cell byte ranges overlap.
        data, root, *_ = fixtures.invalid_scenarios()["leaf_cell_overlap"]
        res = audit_snapshot(data, root)
        self.assertEqual(res["verdict"], "rejected")
        err = res["error"]
        self.assertEqual(err["code"], "CELL_OVERLAP")
        self.assertEqual(err["page"], 3)
        # Evidence is the conflicting cell pointer (cell 1), not the cell body.
        self.assertEqual(err["offset"], 2 * PAGE_SIZE + 10)
        self.assertEqual(err["detail"]["cell_index"], 1)
        self.assertEqual(err["detail"]["other_cell_index"], 0)
        self.assertEqual(err["detail"]["cell_start"], err["detail"]["other_start"] + 2)
        # Stable: repeated audits report the same first evidence.
        self.assertEqual(res, audit_snapshot(data, root))

    def test_overlapping_cells_conflict_independent_of_pointer_order(self):
        # Even when the later (higher-address, smaller-range) cell is listed
        # first, the overlap must be rejected at the conflicting pointer.
        b = SnapshotBuilder(page_size=PAGE_SIZE, page_count=2)
        page = b.page(2)
        page[0] = 0x0D
        page[3:5] = (2).to_bytes(2, "big")
        page[5:7] = (900).to_bytes(2, "big")
        # Big cell (rowid 1) at 900..908; small cell (rowid 2) nested at 902.
        page[900:909] = (
            encode_varint(7) + encode_varint(1)
            + encode_varint(1) + encode_varint(2) + b"\xaa\x00\x00\x00\x00"
        )
        page[8:10] = (900).to_bytes(2, "big")   # cell 0 -> rowid 1, spans 900..910
        page[10:12] = (902).to_bytes(2, "big")  # cell 1 -> rowid 2, nested
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "CELL_OVERLAP")
        self.assertEqual(res["error"]["offset"], PAGE_SIZE + 10)

    def test_cell_pointer_into_declared_freeblock_rejected(self):
        # The freeblock bytes happen to parse as a short cell; they are still
        # declared free space and must not be accepted as a live cell.
        data, root, *_ = fixtures.invalid_scenarios()[
            "cell_pointer_into_freeblock"
        ]
        res = audit_snapshot(data, root)
        self.assertEqual(res["verdict"], "rejected")
        err = res["error"]
        self.assertEqual(err["code"], "CELL_POINTER_IN_FREEBLOCK")
        self.assertEqual(err["page"], 2)
        # Evidence is the cell pointer slot that references the freeblock.
        self.assertEqual(err["offset"], PAGE_SIZE + 8)
        self.assertEqual(err["detail"]["cell_index"], 0)
        self.assertEqual(err["detail"]["pointer"], err["detail"]["freeblock"])
        self.assertEqual(res, audit_snapshot(data, root))

    def test_declared_freeblocks_accepted_when_no_cell_uses_them(self):
        # Same freeblock layout as the corrupted scenario, but no cell pointer
        # targets either freeblock: declared free space is legal.
        data, root, *_ = fixtures.invalid_scenarios()[
            "cell_pointer_into_freeblock"
        ]
        repaired = bytearray(data)
        page_off = PAGE_SIZE
        # Slot 0 referenced the freeblock: repoint it at a disjoint short cell
        # placed just below the freeblock (900), and bump the real rowid-3 cell
        # in slot 1 so pointer order matches increasing rowids 2 then 3.
        repaired[page_off + 897 : page_off + 900] = (
            encode_varint(1) + encode_varint(2) + b"y"
        )
        repaired[page_off + 8 : page_off + 10] = (897).to_bytes(2, "big")
        repaired[page_off + 10 : page_off + 12] = (1021).to_bytes(2, "big")
        res = audit_snapshot(bytes(repaired), root)
        self.assertEqual(res["verdict"], "accepted", res["error"])

    def test_physically_reordered_but_disjoint_cells_accepted(self):
        # Cells need not be laid out in pointer order on the page.
        b = SnapshotBuilder(page_size=PAGE_SIZE, page_count=2)
        page = b.page(2)
        page[0] = 0x0D
        page[3:5] = (2).to_bytes(2, "big")
        page[5:7] = (900).to_bytes(2, "big")
        cell1 = encode_varint(1) + encode_varint(1) + b"a"  # rowid 1 at 920
        cell2 = encode_varint(1) + encode_varint(2) + b"b"  # rowid 2 at 900
        page[920 : 920 + len(cell1)] = cell1
        page[900 : 900 + len(cell2)] = cell2
        page[8:10] = (920).to_bytes(2, "big")
        page[10:12] = (900).to_bytes(2, "big")
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["verdict"], "accepted", res["error"])

    def test_divider_key_reaches_right_subtree(self):
        b, _ = _base_builder()
        root = b.add_table_interior(2, [9], [11, 5])  # right subtree starts at 9
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "KEY_BOUND_CONFLICT")
        self.assertEqual(res["error"]["page"], 2)
        self.assertEqual(res["error"]["offset"], PAGE_SIZE + root["ptrs"][0] + 4)
        self.assertEqual(res["error"]["detail"]["right_subtree_min"], 9)

    def test_duplicate_rowid_across_leaves(self):
        b, _ = _base_builder()
        b.add_table_leaf(4, [(4, b"dup"), (5, b"e"), (6, b"f"), (8, b"g")])
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "KEY_BOUND_CONFLICT")
        self.assertEqual(res["error"]["page"], 11)

    def test_deep_left_bound_still_checked(self):
        # Divider key below the max of a deeper left subtree.
        b, _ = _base_builder()
        b.add_table_interior(11, [2], [3, 4])  # leaf 3 holds rowids up to 4
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "KEY_BOUND_CONFLICT")
        self.assertEqual(res["error"]["page"], 11)
        self.assertEqual(res["error"]["detail"]["left_subtree_max"], 4)


class OverflowTests(unittest.TestCase):
    def test_overflow_page_out_of_range(self):
        b, meta = _base_builder()
        page4 = b.page(4)
        off = meta["leaf4"][fixtures.BLOB_ROWID]["overflow_ptr"]
        page4[off : off + 4] = (77).to_bytes(4, "big")
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "OVERFLOW_PAGE_OUT_OF_RANGE")
        self.assertEqual(res["error"]["page"], 77)
        self.assertEqual(res["error"]["offset"], 3 * PAGE_SIZE + off)

    def test_overflow_points_at_btree_page(self):
        b, meta = _base_builder()
        page4 = b.page(4)
        off = meta["leaf4"][fixtures.BLOB_ROWID]["overflow_ptr"]
        page4[off : off + 4] = (5).to_bytes(4, "big")  # leaf page 5
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "PAGE_OWNERSHIP_CONFLICT")
        self.assertEqual(res["error"]["page"], 5)

    def test_overflow_cycle(self):
        # Declared payload needs three pages, but the chain loops: 3 -> 4 -> 3.
        b = SnapshotBuilder(page_size=PAGE_SIZE, page_count=6)
        per_page = PAGE_SIZE - 4
        min_local = ((PAGE_SIZE - 12) * 32) // 255 - 23
        payload_len = min_local + 3 * per_page
        payload = bytes(payload_len)
        b.add_table_leaf(2, [(1, payload)], overflow_heads={1: 3})
        b.add_overflow_chain(3, payload[min_local:], next_overrides={4: 3})
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "PAGE_OWNERSHIP_CONFLICT")
        self.assertEqual(res["error"]["page"], 3)

    def test_exact_coverage_boundary(self):
        # Payload whose overflow part fills exactly one page must pass.
        b = SnapshotBuilder(page_size=PAGE_SIZE, page_count=4)
        per_page = PAGE_SIZE - 4
        # local = min_local when (P - min_local) % (U - 4) pushes it over max_local
        min_local = ((PAGE_SIZE - 12) * 32) // 255 - 23
        payload_len = min_local + per_page
        payload = bytes(payload_len)
        b.add_table_leaf(2, [(1, payload)], overflow_heads={1: 3})
        b.add_overflow_chain(3, payload[min_local:])
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["verdict"], "accepted", res["error"])
        self.assertEqual(res["summary"]["overflow_pages"], 1)


class FreelistTests(unittest.TestCase):
    def test_freelist_duplicate_leaf(self):
        b, _ = _base_builder()
        b.add_freelist_trunk(8, 0, [9, 9])
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "FREELIST_DUPLICATE")
        self.assertEqual(res["error"]["page"], 9)
        self.assertEqual(res["error"]["offset"], 7 * PAGE_SIZE + 8 + 4)

    def test_freelist_trunk_cycle(self):
        b, _ = _base_builder()
        b.add_freelist_trunk(8, 8, [9, 10])  # trunk points at itself
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "FREELIST_DUPLICATE")
        self.assertEqual(res["error"]["page"], 8)

    def test_freelist_trunk_out_of_range(self):
        b, _ = _base_builder()
        b.freelist_head = 55
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "FREELIST_PAGE_OUT_OF_RANGE")
        self.assertEqual(res["error"]["page"], 55)
        self.assertEqual(res["error"]["offset"], 32)

    def test_freelist_leaf_count_overflow(self):
        b, _ = _base_builder()
        page8 = b.page(8)
        page8[4:8] = (PAGE_SIZE // 4 - 1).to_bytes(4, "big")  # one over the max
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "FREELIST_TRUNK_LEAF_COUNT")
        self.assertEqual(res["error"]["page"], 8)

    def test_freelist_head_on_live_page(self):
        b, _ = _base_builder()
        b.freelist_head = 5  # b-tree leaf page
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "LIVE_PAGE_ON_FREELIST")
        self.assertEqual(res["error"]["page"], 5)
        self.assertEqual(res["error"]["offset"], 32)

    def test_overflow_page_landed_on_freelist(self):
        b, _ = _base_builder()
        b.add_freelist_trunk(8, 0, [9, 7])  # page 7 is an overflow page
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "LIVE_PAGE_ON_FREELIST")
        self.assertEqual(res["error"]["page"], 7)

    def test_empty_freelist_ok(self):
        b, _ = _base_builder()
        b.freelist_head = 0
        b.freelist_count = 0
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["verdict"], "accepted", res["error"])

    def test_head_zero_but_count_nonzero(self):
        b, _ = _base_builder()
        b.freelist_head = 0
        b.freelist_count = 3
        res = audit_snapshot(b.build(), 2)
        self.assertEqual(res["error"]["code"], "FREELIST_COUNT_MISMATCH")


class PageSizeTests(unittest.TestCase):
    def test_all_reviewed_page_sizes_accepted(self):
        for page_size in (512, 1024, 2048, 4096):
            with self.subTest(page_size=page_size):
                b = SnapshotBuilder(page_size=page_size, page_count=3)
                b.add_table_leaf(2, [(1, b"x"), (2, b"y")])
                res = audit_snapshot(b.build(), 2)
                self.assertEqual(res["verdict"], "accepted", res["error"])
                self.assertEqual(res["page_size"], page_size)

    def test_root_on_page_one(self):
        b = SnapshotBuilder(page_size=512, page_count=2)
        b.add_table_leaf(1, [(1, b"x")], first_page=True)
        res = audit_snapshot(b.build(), 1)
        self.assertEqual(res["verdict"], "accepted", res["error"])
        kinds = {p["page"]: p["kind"] for p in res["pages"]}
        self.assertEqual(kinds, {1: "btree_root"})


if __name__ == "__main__":
    unittest.main()
