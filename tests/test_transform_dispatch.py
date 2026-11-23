"""Check callback order and metadata alignment within and across locations."""

from typing import List
from unittest import TestCase

import pandas as pd

import Data_process as dp


def group_add(group: pd.DataFrame, heights: List[str]) -> pd.DataFrame:
    """Add one to the height blocks and reset their row index."""
    return (group.loc[:, pd.IndexSlice[heights, :]] + 1).reset_index(drop=True)


def group_double(group: pd.DataFrame, heights: List[str]) -> pd.DataFrame:
    """Double the height blocks and reset their row index."""
    return (group.loc[:, pd.IndexSlice[heights, :]] * 2).reset_index(drop=True)


def row_add(row: pd.Series, heights: List[str]) -> List[pd.Series]:
    """Add one to each height block in a row."""
    return [row.loc[pd.IndexSlice[[height], :]] + 1 for height in heights]


def row_double(row: pd.Series, heights: List[str]) -> List[pd.Series]:
    """Double each height block in a row."""
    return [row.loc[pd.IndexSlice[[height], :]] * 2 for height in heights]


class TransformDispatchTests(TestCase):
    """Check height ordering, location isolation, and target alignment."""

    def setUp(self) -> None:
        """Build two interleaved locations in an unsorted height layout."""
        columns = pd.MultiIndex.from_tuples(
            [("height_2", "x"), ("height_1", "x"), ("meta", "location"), ("y", "target")]
        )
        self.frame = pd.DataFrame(
            [[50., 10., "B", 1], [7., 2., "A", 2],
             [90., 16., "B", 3], [11., 4., "A", 4]],
            columns=columns,
        )

    def test_direct_height_dispatchers_preserve_metadata(self) -> None:
        expected = self.frame[[("height_1", "x"), ("height_2", "x"),
                               ("meta", "location"), ("y", "target")]].copy()
        expected[("height_1", "x")] += 1
        expected[("height_2", "x")] += 1
        for dispatcher, transform in ((dp.transform_height_columns, group_add),
                                      (dp.transform_height_columns_row_wise, row_add)):
            with self.subTest(dispatcher=dispatcher.__name__):
                pd.testing.assert_frame_equal(dispatcher(self.frame, transform), expected,
                                              check_dtype=False)

    def test_column_dispatch_centers_within_each_location_then_doubles(self) -> None:
        result = dp.transform_location_wise_height_group(
            self.frame, [dp.column_center_by_loc, group_double]
        )
        self.assertEqual(list(result.columns),
                         [("height_1", "centered_x"), ("height_2", "centered_x"),
                          ("meta", "location"), ("y", "target")])
        self.assertEqual(result[("height_1", "centered_x")].tolist(), [-2., 2., -6., 6.])
        self.assertEqual(result[("height_2", "centered_x")].tolist(), [-4., 4., -40., 40.])
        self.assertEqual(result[("meta", "location")].tolist(), ["A", "A", "B", "B"])
        self.assertEqual(result[("y", "target")].tolist(), [2, 4, 1, 3])
        pd.testing.assert_index_equal(result.index, pd.RangeIndex(4))

    def test_row_dispatch_applies_callbacks_in_list_order(self) -> None:
        result = dp.transform_location_wise_height_row_group(self.frame, [row_add, row_double])
        # Joining by target avoids relying on pandas' groupby.apply ordering.
        by_target = result.set_index(("y", "target")).sort_index()
        self.assertEqual(by_target[("height_1", "x")].tolist(), [22., 6., 34., 10.])
        self.assertEqual(by_target[("height_2", "x")].tolist(), [102., 16., 182., 24.])
        self.assertEqual(by_target[("meta", "location")].tolist(), ["B", "A", "B", "A"])
        pd.testing.assert_index_equal(result.index, pd.RangeIndex(4))
        self.assertIs(dp.transform_location_wise_height_row_group(self.frame, []), self.frame)
