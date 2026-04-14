"""
Tests for FeatureMetadataRegistry and FeatureMetadata.

Tests cover:
- Metadata initialization and storage
- Lookup by dense/collapsed/name
- Categorical detection (semantic and cardinality)
- Statistics computation (unique values, min/max)
- With and without label manager
- With and without group manager
- Edge cases and error handling
"""

import numpy as np
import pytest

from prism.plotting.index_mapper import IndexMapper
from prism.plotting.metadata import FeatureMetadata, FeatureMetadataRegistry


class TestFeatureMetadataBasics:
    """Test basic FeatureMetadata dataclass functionality."""

    def test_metadata_creation(self):
        """Test creating a FeatureMetadata instance."""
        metadata = FeatureMetadata(
            dense_idx=0,
            collapsed_idx=2,
            original_indices=[2, 3, 4],
            column_name='diagn',
            user_label='Diagnosis\n(categorical)',
            is_categorical=True,
            is_collapsed_group=True,
            unique_values=np.array([0, 1, 2]),
            min_value=None,
            max_value=None,
        )
        assert metadata.dense_idx == 0
        assert metadata.collapsed_idx == 2
        assert metadata.column_name == 'diagn'
        assert metadata.is_categorical is True
        assert metadata.is_collapsed_group is True
        assert len(metadata.unique_values) == 3

    def test_metadata_continuous_feature(self):
        """Test metadata for continuous feature."""
        metadata = FeatureMetadata(
            dense_idx=1,
            collapsed_idx=0,
            original_indices=[0],
            column_name='age',
            user_label='Age (years)',
            is_categorical=False,
            is_collapsed_group=False,
            unique_values=None,
            min_value=18.0,
            max_value=90.0,
        )
        assert metadata.is_categorical is False
        assert metadata.unique_values is None
        assert metadata.min_value == 18.0
        assert metadata.max_value == 90.0


class TestFeatureMetadataRegistryWithoutGroups:
    """Test registry without one-hot groups (simple case)."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data: 3 continuous, 1 categorical."""
        np.random.seed(42)
        n_samples = 100
        return np.column_stack(
            [
                np.random.uniform(18, 90, n_samples),  # age (continuous)
                np.random.uniform(15, 40, n_samples),  # bmi (continuous)
                np.random.choice([0, 1, 2], n_samples),  # status (categorical, 3 values)
                np.random.uniform(80, 180, n_samples),  # sbp (continuous)
            ]
        )

    @pytest.fixture
    def simple_mapper(self):
        """Mapper with 4 features, all selected, no groups."""
        return IndexMapper(
            original_names=['age', 'bmi', 'status', 'sbp'],
            collapsed_names=['age', 'bmi', 'status', 'sbp'],
            selected_indices=[0, 1, 2, 3],
            group_manager=None,
        )

    @pytest.fixture
    def simple_registry(self, simple_mapper, simple_data):
        """Registry without label manager or groups."""
        return FeatureMetadataRegistry(
            index_mapper=simple_mapper,
            all_feature_names=['age', 'bmi', 'status', 'sbp'],
            x_data=simple_data,
            categorical_threshold=10,
            group_manager=None,
            label_manager=None,
        )

    def test_length(self, simple_registry):
        """Test registry length."""
        assert len(simple_registry) == 4

    def test_repr(self, simple_registry):
        """Test string representation."""
        repr_str = repr(simple_registry)
        assert "FeatureMetadataRegistry" in repr_str
        assert "n_features=4" in repr_str

    def test_get_by_dense(self, simple_registry):
        """Test lookup by dense index."""
        # Dense 0 -> age
        metadata = simple_registry.get_by_dense(0)
        assert metadata.dense_idx == 0
        assert metadata.column_name == 'age'
        assert metadata.is_categorical is False

        # Dense 2 -> status (categorical)
        metadata = simple_registry.get_by_dense(2)
        assert metadata.dense_idx == 2
        assert metadata.column_name == 'status'
        assert metadata.is_categorical is True

    def test_get_by_dense_out_of_range(self, simple_registry):
        """Test error handling for out-of-range dense index."""
        with pytest.raises(IndexError, match="Dense index 4 out of range"):
            simple_registry.get_by_dense(4)
        with pytest.raises(IndexError, match="Dense index -1 out of range"):
            simple_registry.get_by_dense(-1)

    def test_get_by_collapsed(self, simple_registry):
        """Test lookup by collapsed index."""
        # Collapsed 1 -> bmi
        metadata = simple_registry.get_by_collapsed(1)
        assert metadata is not None
        assert metadata.collapsed_idx == 1
        assert metadata.column_name == 'bmi'

        # Collapsed 2 -> status
        metadata = simple_registry.get_by_collapsed(2)
        assert metadata is not None
        assert metadata.column_name == 'status'

    def test_get_by_name(self, simple_registry):
        """Test lookup by feature name."""
        metadata = simple_registry.get_by_name('age')
        assert metadata is not None
        assert metadata.column_name == 'age'

        metadata = simple_registry.get_by_name('status')
        assert metadata is not None
        assert metadata.column_name == 'status'

    def test_get_by_name_not_found(self, simple_registry):
        """Test lookup for non-existent name."""
        metadata = simple_registry.get_by_name('nonexistent')
        assert metadata is None

    def test_categorical_detection_cardinality(self, simple_registry):
        """Test cardinality-based categorical detection."""
        # status has 3 unique values -> categorical
        status_meta = simple_registry.get_by_name('status')
        assert status_meta.is_categorical is True
        assert len(status_meta.unique_values) == 3

        # age has many unique values -> continuous
        age_meta = simple_registry.get_by_name('age')
        assert age_meta.is_categorical is False
        assert age_meta.unique_values is None

    def test_continuous_statistics(self, simple_registry):
        """Test min/max computation for continuous features."""
        age_meta = simple_registry.get_by_name('age')
        assert age_meta.min_value is not None
        assert age_meta.max_value is not None
        assert age_meta.min_value < age_meta.max_value
        assert 18 <= age_meta.min_value <= 90
        assert 18 <= age_meta.max_value <= 90

    def test_categorical_statistics(self, simple_registry):
        """Test unique values for categorical features."""
        status_meta = simple_registry.get_by_name('status')
        assert status_meta.unique_values is not None
        assert status_meta.min_value is None
        assert status_meta.max_value is None
        assert set(status_meta.unique_values) == {0, 1, 2}

    def test_iteration(self, simple_registry):
        """Test iterating over registry."""
        names = [meta.column_name for meta in simple_registry]
        assert names == ['age', 'bmi', 'status', 'sbp']


class TestFeatureMetadataRegistryWithGroups:
    """Test registry with one-hot groups and collapse."""

    @pytest.fixture
    def mock_group_manager(self):
        """Mock OneHotGroupManager with semantic groups."""

        class MockGroupManager:
            def __init__(self):
                self.groups_dict = {
                    'diagn': ['diagn_CAD', 'diagn_Valve', 'diagn_Other'],
                    'blood_type': ['blood_A', 'blood_B', 'blood_O'],
                }

            def is_categorical_group(self, name):
                return name in self.groups_dict

        return MockGroupManager()

    @pytest.fixture
    def grouped_data(self):
        """Create test data with collapsed groups."""
        np.random.seed(42)
        n_samples = 100
        return np.column_stack(
            [
                np.random.uniform(18, 90, n_samples),  # age
                np.random.uniform(15, 40, n_samples),  # bmi
                np.random.choice([0, 1, 2], n_samples),  # diagn (collapsed categorical)
                np.random.choice([0, 1, 2], n_samples),  # blood_type (collapsed categorical)
            ]
        )

    @pytest.fixture
    def grouped_mapper(self, mock_group_manager):
        """Mapper with collapsed groups."""
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
            selected_indices=[2, 3],  # Select diagn and blood_type
            group_manager=mock_group_manager,
        )

    @pytest.fixture
    def grouped_registry(self, grouped_mapper, grouped_data, mock_group_manager):
        """Registry with groups."""
        return FeatureMetadataRegistry(
            index_mapper=grouped_mapper,
            all_feature_names=['age', 'bmi', 'diagn', 'blood_type'],
            x_data=grouped_data,
            categorical_threshold=10,
            group_manager=mock_group_manager,
            label_manager=None,
        )

    def test_semantic_categorical_detection(self, grouped_registry):
        """Test semantic categorical detection from group_manager."""
        # diagn is a collapsed group -> semantic categorical
        diagn_meta = grouped_registry.get_by_name('diagn')
        assert diagn_meta is not None
        assert diagn_meta.is_categorical is True
        assert diagn_meta.is_collapsed_group is True

        # blood_type is also a collapsed group
        blood_meta = grouped_registry.get_by_name('blood_type')
        assert blood_meta is not None
        assert blood_meta.is_categorical is True
        assert blood_meta.is_collapsed_group is True

    def test_original_indices_for_groups(self, grouped_registry):
        """Test that collapsed groups map to multiple original indices."""
        diagn_meta = grouped_registry.get_by_name('diagn')
        assert diagn_meta is not None
        # diagn -> [diagn_CAD, diagn_Valve, diagn_Other] at original indices [2, 3, 4]
        assert diagn_meta.original_indices == [2, 3, 4]

        blood_meta = grouped_registry.get_by_name('blood_type')
        assert blood_meta is not None
        # blood_type -> [blood_A, blood_B, blood_O] at original indices [5, 6, 7]
        assert blood_meta.original_indices == [5, 6, 7]

    def test_selected_features_only(self, grouped_registry):
        """Test that registry only contains selected features."""
        # Only diagn and blood_type were selected
        assert len(grouped_registry) == 2
        assert grouped_registry.get_by_name('diagn') is not None
        assert grouped_registry.get_by_name('blood_type') is not None
        assert grouped_registry.get_by_name('age') is None  # Not selected
        assert grouped_registry.get_by_name('bmi') is None  # Not selected


class TestFeatureMetadataRegistryWithLabels:
    """Test registry with FeatureLabelManager."""

    @pytest.fixture
    def mock_label_manager(self):
        """Mock FeatureLabelManager."""

        class MockLabelManager:
            def __init__(self):
                self._labels = {
                    'age': 'Age (years)',
                    'bmi': 'BMI (kg/m²)',
                    'status': 'Status\n(categorical)',
                }

            def get_label(self, name):
                return self._labels.get(name, name)

        return MockLabelManager()

    @pytest.fixture
    def simple_data(self):
        """Simple test data."""
        np.random.seed(42)
        n_samples = 100
        return np.column_stack(
            [
                np.random.uniform(18, 90, n_samples),
                np.random.uniform(15, 40, n_samples),
                np.random.choice([0, 1, 2], n_samples),
            ]
        )

    @pytest.fixture
    def labeled_registry(self, mock_label_manager, simple_data):
        """Registry with label manager."""
        mapper = IndexMapper(
            original_names=['age', 'bmi', 'status'],
            collapsed_names=['age', 'bmi', 'status'],
            selected_indices=[0, 1, 2],
            group_manager=None,
        )
        return FeatureMetadataRegistry(
            index_mapper=mapper,
            all_feature_names=['age', 'bmi', 'status'],
            x_data=simple_data,
            categorical_threshold=10,
            group_manager=None,
            label_manager=mock_label_manager,
        )

    def test_user_labels_from_manager(self, labeled_registry):
        """Test that user labels come from label manager."""
        age_meta = labeled_registry.get_by_name('age')
        assert age_meta.column_name == 'age'
        assert age_meta.user_label == 'Age (years)'

        bmi_meta = labeled_registry.get_by_name('bmi')
        assert bmi_meta.column_name == 'bmi'
        assert bmi_meta.user_label == 'BMI (kg/m²)'

        status_meta = labeled_registry.get_by_name('status')
        assert status_meta.column_name == 'status'
        assert status_meta.user_label == 'Status\n(categorical)'

    def test_fallback_to_column_name(self):
        """Test that column name is used when no label manager."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, (100, 2))
        mapper = IndexMapper(
            original_names=['feat1', 'feat2'],
            collapsed_names=['feat1', 'feat2'],
            selected_indices=[0, 1],
            group_manager=None,
        )
        registry = FeatureMetadataRegistry(
            index_mapper=mapper,
            all_feature_names=['feat1', 'feat2'],
            x_data=data,
            categorical_threshold=10,
            group_manager=None,
            label_manager=None,
        )
        feat1_meta = registry.get_by_name('feat1')
        assert feat1_meta.column_name == 'feat1'
        assert feat1_meta.user_label == 'feat1'  # Fallback


class TestFeatureMetadataRegistryEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_selection(self):
        """Test with no features selected."""
        mapper = IndexMapper(
            original_names=['a', 'b', 'c'],
            collapsed_names=['a', 'b', 'c'],
            selected_indices=[],
            group_manager=None,
        )
        data = np.random.uniform(0, 1, (10, 3))
        registry = FeatureMetadataRegistry(
            index_mapper=mapper,
            all_feature_names=['a', 'b', 'c'],
            x_data=data,
            categorical_threshold=10,
            group_manager=None,
            label_manager=None,
        )
        assert len(registry) == 0
        assert list(registry) == []

    def test_single_feature(self):
        """Test with only one feature selected."""
        mapper = IndexMapper(
            original_names=['x'],
            collapsed_names=['x'],
            selected_indices=[0],
            group_manager=None,
        )
        data = np.random.uniform(0, 1, (10, 1))
        registry = FeatureMetadataRegistry(
            index_mapper=mapper,
            all_feature_names=['x'],
            x_data=data,
            categorical_threshold=10,
            group_manager=None,
            label_manager=None,
        )
        assert len(registry) == 1
        meta = registry.get_by_dense(0)
        assert meta.column_name == 'x'

    def test_get_by_collapsed_not_selected(self):
        """Test lookup by collapsed index that wasn't selected."""
        mapper = IndexMapper(
            original_names=['a', 'b', 'c'],
            collapsed_names=['a', 'b', 'c'],
            selected_indices=[0, 2],  # Skip 'b'
            group_manager=None,
        )
        data = np.random.uniform(0, 1, (10, 3))
        registry = FeatureMetadataRegistry(
            index_mapper=mapper,
            all_feature_names=['a', 'b', 'c'],
            x_data=data,
            categorical_threshold=10,
            group_manager=None,
            label_manager=None,
        )
        # Collapsed index 1 (b) was not selected
        assert registry.get_by_collapsed(1) is None
        # Collapsed indices 0 and 2 were selected
        assert registry.get_by_collapsed(0) is not None
        assert registry.get_by_collapsed(2) is not None
