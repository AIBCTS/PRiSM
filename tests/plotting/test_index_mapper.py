"""
Tests for IndexMapper class.

Tests cover:
- Dense to collapsed conversions
- Collapsed to dense conversions
- Collapsed to original conversions
- Edge cases and error handling
- With and without collapse mode
"""

import pytest

from prism.plotting.index_mapper import IndexMapper


class TestIndexMapperWithoutCollapse:
    """Test IndexMapper when no collapse is active (1:1:1 mapping)."""

    @pytest.fixture
    def simple_mapper(self):
        """Mapper with 5 features, 3 selected, no collapse."""
        return IndexMapper(
            original_names=['age', 'bmi', 'glucose', 'bp', 'cholesterol'],
            collapsed_names=['age', 'bmi', 'glucose', 'bp', 'cholesterol'],
            selected_indices=[1, 2, 4],  # Selected: bmi, glucose, cholesterol
            group_manager=None,
        )

    def test_properties(self, simple_mapper):
        """Test dimension properties."""
        assert simple_mapper.n_original == 5
        assert simple_mapper.n_collapsed == 5
        assert simple_mapper.n_dense == 3
        assert simple_mapper.is_collapse_mode is False

    def test_dense_to_collapsed(self, simple_mapper):
        """Test dense -> collapsed conversions."""
        # Dense 0 -> Collapsed 1 (bmi)
        assert simple_mapper.dense_to_collapsed(0) == 1
        # Dense 1 -> Collapsed 2 (glucose)
        assert simple_mapper.dense_to_collapsed(1) == 2
        # Dense 2 -> Collapsed 4 (cholesterol)
        assert simple_mapper.dense_to_collapsed(2) == 4

    def test_collapsed_to_dense(self, simple_mapper):
        """Test collapsed -> dense conversions."""
        # Collapsed 1 -> Dense 0 (bmi selected)
        assert simple_mapper.collapsed_to_dense(1) == 0
        # Collapsed 2 -> Dense 1 (glucose selected)
        assert simple_mapper.collapsed_to_dense(2) == 1
        # Collapsed 4 -> Dense 2 (cholesterol selected)
        assert simple_mapper.collapsed_to_dense(4) == 2
        # Collapsed 0 -> None (age not selected)
        assert simple_mapper.collapsed_to_dense(0) is None
        # Collapsed 3 -> None (bp not selected)
        assert simple_mapper.collapsed_to_dense(3) is None

    def test_collapsed_to_original(self, simple_mapper):
        """Test collapsed -> original conversions (1:1 mapping)."""
        assert simple_mapper.collapsed_to_original(0) == [0]  # age
        assert simple_mapper.collapsed_to_original(1) == [1]  # bmi
        assert simple_mapper.collapsed_to_original(4) == [4]  # cholesterol

    def test_dense_to_collapsed_out_of_range(self, simple_mapper):
        """Test error handling for out-of-range dense indices."""
        with pytest.raises(IndexError, match="Dense index 3 out of range"):
            simple_mapper.dense_to_collapsed(3)
        with pytest.raises(IndexError, match="Dense index -1 out of range"):
            simple_mapper.dense_to_collapsed(-1)

    def test_collapsed_to_original_out_of_range(self, simple_mapper):
        """Test error handling for out-of-range collapsed indices."""
        with pytest.raises(IndexError, match="Collapsed index 5 out of range"):
            simple_mapper.collapsed_to_original(5)

    def test_repr(self, simple_mapper):
        """Test string representation."""
        repr_str = repr(simple_mapper)
        assert "original=5" in repr_str
        assert "collapsed=5" in repr_str
        assert "dense=3" in repr_str
        assert "collapse_mode=False" in repr_str


class TestIndexMapperWithCollapse:
    """Test IndexMapper with one-hot group collapse."""

    @pytest.fixture
    def mock_group_manager(self):
        """Mock OneHotGroupManager."""

        class MockGroupManager:
            def __init__(self):
                self.groups_dict = {
                    'diagn': ['diagn_CAD', 'diagn_Valve', 'diagn_Other'],
                    'blood_type': ['blood_A', 'blood_B', 'blood_O'],
                }

        return MockGroupManager()

    @pytest.fixture
    def collapse_mapper(self, mock_group_manager):
        """Mapper with collapse: 8 original -> 4 collapsed -> 2 selected."""
        return IndexMapper(
            original_names=[
                'age',
                'bmi',
                'diagn_CAD',
                'diagn_Valve',
                'diagn_Other',
                'blood_A',
                'blood_B',
                'blood_O',
            ],
            collapsed_names=['age', 'bmi', 'diagn', 'blood_type'],
            selected_indices=[2, 3],  # Selected: diagn, blood_type
            group_manager=mock_group_manager,
        )

    def test_properties(self, collapse_mapper):
        """Test dimension properties with collapse."""
        assert collapse_mapper.n_original == 8
        assert collapse_mapper.n_collapsed == 4
        assert collapse_mapper.n_dense == 2
        assert collapse_mapper.is_collapse_mode is True

    def test_dense_to_collapsed(self, collapse_mapper):
        """Test dense -> collapsed with collapse."""
        assert collapse_mapper.dense_to_collapsed(0) == 2  # diagn
        assert collapse_mapper.dense_to_collapsed(1) == 3  # blood_type

    def test_collapsed_to_dense(self, collapse_mapper):
        """Test collapsed -> dense with collapse."""
        assert collapse_mapper.collapsed_to_dense(2) == 0  # diagn selected
        assert collapse_mapper.collapsed_to_dense(3) == 1  # blood_type selected
        assert collapse_mapper.collapsed_to_dense(0) is None  # age not selected
        assert collapse_mapper.collapsed_to_dense(1) is None  # bmi not selected

    def test_collapsed_to_original_group(self, collapse_mapper):
        """Test collapsed -> original for collapsed groups."""
        # diagn (collapsed idx 2) -> [diagn_CAD, diagn_Valve, diagn_Other]
        diagn_indices = collapse_mapper.collapsed_to_original(2)
        assert diagn_indices == [2, 3, 4]

        # blood_type (collapsed idx 3) -> [blood_A, blood_B, blood_O]
        blood_indices = collapse_mapper.collapsed_to_original(3)
        assert blood_indices == [5, 6, 7]

    def test_collapsed_to_original_single(self, collapse_mapper):
        """Test collapsed -> original for non-grouped features."""
        # age (collapsed idx 0) -> [age] (index 0)
        assert collapse_mapper.collapsed_to_original(0) == [0]
        # bmi (collapsed idx 1) -> [bmi] (index 1)
        assert collapse_mapper.collapsed_to_original(1) == [1]

    def test_repr(self, collapse_mapper):
        """Test string representation with collapse."""
        repr_str = repr(collapse_mapper)
        assert "original=8" in repr_str
        assert "collapsed=4" in repr_str
        assert "dense=2" in repr_str
        assert "collapse_mode=True" in repr_str


class TestIndexMapperEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_selection(self):
        """Test with no features selected."""
        mapper = IndexMapper(
            original_names=['a', 'b', 'c'],
            collapsed_names=['a', 'b', 'c'],
            selected_indices=[],
            group_manager=None,
        )
        assert mapper.n_dense == 0
        assert mapper.collapsed_to_dense(0) is None

    def test_all_selected(self):
        """Test with all features selected."""
        mapper = IndexMapper(
            original_names=['a', 'b', 'c'],
            collapsed_names=['a', 'b', 'c'],
            selected_indices=[0, 1, 2],
            group_manager=None,
        )
        assert mapper.n_dense == 3
        # Dense should map 1:1 to collapsed
        assert mapper.dense_to_collapsed(0) == 0
        assert mapper.dense_to_collapsed(1) == 1
        assert mapper.dense_to_collapsed(2) == 2

    def test_single_feature(self):
        """Test with only one feature."""
        mapper = IndexMapper(
            original_names=['x'],
            collapsed_names=['x'],
            selected_indices=[0],
            group_manager=None,
        )
        assert mapper.n_dense == 1
        assert mapper.dense_to_collapsed(0) == 0
        assert mapper.collapsed_to_dense(0) == 0
        assert mapper.collapsed_to_original(0) == [0]

    def test_noncontiguous_selection(self):
        """Test with non-contiguous selected indices."""
        mapper = IndexMapper(
            original_names=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            collapsed_names=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            selected_indices=[0, 3, 6],  # Non-contiguous
            group_manager=None,
        )
        assert mapper.dense_to_collapsed(0) == 0
        assert mapper.dense_to_collapsed(1) == 3
        assert mapper.dense_to_collapsed(2) == 6
        assert mapper.collapsed_to_dense(3) == 1
        assert mapper.collapsed_to_dense(1) is None  # Not selected


class TestIndexMapperBidirectional:
    """Test that conversions are bidirectional where applicable."""

    @pytest.fixture
    def mapper(self):
        return IndexMapper(
            original_names=['a', 'b', 'c', 'd', 'e'],
            collapsed_names=['a', 'b', 'c', 'd', 'e'],
            selected_indices=[1, 3, 4],
            group_manager=None,
        )

    def test_dense_collapsed_roundtrip(self, mapper):
        """Test dense -> collapsed -> dense roundtrip."""
        for dense_idx in range(mapper.n_dense):
            collapsed_idx = mapper.dense_to_collapsed(dense_idx)
            recovered_dense = mapper.collapsed_to_dense(collapsed_idx)
            assert recovered_dense == dense_idx

    def test_collapsed_dense_partial_roundtrip(self, mapper):
        """Test collapsed -> dense -> collapsed (only for selected)."""
        selected = mapper._selected_indices
        for collapsed_idx in selected:
            dense_idx = mapper.collapsed_to_dense(collapsed_idx)
            assert dense_idx is not None
            recovered_collapsed = mapper.dense_to_collapsed(dense_idx)
            assert recovered_collapsed == collapsed_idx
