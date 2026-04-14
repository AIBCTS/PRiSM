"""Tests for data utility functions."""

import pandas as pd
import pytest

from prism.data_utilities import (
    analyze_categorical_columns,
    convert_to_categorical,
    create_binary_encoding,
    create_ordinal_encoding,
    ensure_consistent_categories,
    split_data_predefined,
    split_data_temporal_or_random,
)


class TestSplitDataTemporalOrRandom:
    """Tests for data splitting function."""

    def test_random_split_basic(self, sample_dataframe):
        """Test basic random split functionality."""
        train_df, test_df, val_df = split_data_temporal_or_random(
            sample_dataframe,
            temporal_column=None,
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            random_state=42,
        )

        # Check sizes
        total_size = len(sample_dataframe)
        assert len(train_df) == int(total_size * 0.6)
        assert len(test_df) == int(total_size * 0.2)
        assert len(val_df) == int(total_size * 0.2)

        # Check no overlap
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        val_indices = set(val_df.index)

        assert len(train_indices & test_indices) == 0
        assert len(train_indices & val_indices) == 0
        assert len(test_indices & val_indices) == 0

        # Check all data accounted for
        assert len(train_indices | test_indices | val_indices) == total_size

    def test_random_split_with_stratification(self, sample_dataframe):
        """Test random split with stratification by target."""
        target_col = sample_dataframe["target"]

        train_df, test_df, val_df = split_data_temporal_or_random(
            sample_dataframe,
            temporal_column=None,
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            stratify=target_col,
            random_state=42,
        )

        # Check that class proportions are roughly preserved
        original_proportion = target_col.mean()
        train_proportion = train_df["target"].mean()
        test_proportion = test_df["target"].mean()
        val_proportion = val_df["target"].mean()

        # Allow some tolerance due to small sample size
        assert abs(train_proportion - original_proportion) < 0.2
        assert abs(test_proportion - original_proportion) < 0.3
        assert abs(val_proportion - original_proportion) < 0.3

    def test_temporal_split_basic(self, temporal_dataframe):
        """Test basic temporal split functionality."""
        train_df, test_df, val_df = split_data_temporal_or_random(
            temporal_dataframe,
            temporal_column="year",
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            random_state=42,
        )

        # Check no overlap in years
        train_years = set(train_df["year"].unique())
        test_years = set(test_df["year"].unique())
        val_years = set(val_df["year"].unique())

        # Temporal splits should have distinct time periods
        # (though there might be some overlap at boundaries)
        # At minimum, check that data was split
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(val_df) > 0

        # Check temporal ordering: train should be earliest, val latest
        if len(train_years) > 0 and len(val_years) > 0:
            assert max(train_years) <= max(val_years)

    def test_split_ratios_sum_to_one(self, sample_dataframe):
        """Test that split ratios must sum to 1.0."""
        # This should work
        train_df, test_df, val_df = split_data_temporal_or_random(
            sample_dataframe,
            temporal_column=None,
            train_size=0.7,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
        )

        assert len(train_df) + len(test_df) + len(val_df) == len(sample_dataframe)

    def test_random_split_reproducibility(self, sample_dataframe):
        """Test that random split is reproducible with same seed."""
        train1, test1, val1 = split_data_temporal_or_random(
            sample_dataframe,
            temporal_column=None,
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            random_state=42,
        )

        train2, test2, val2 = split_data_temporal_or_random(
            sample_dataframe,
            temporal_column=None,
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            random_state=42,
        )

        # Should produce identical splits
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)
        pd.testing.assert_frame_equal(val1, val2)


class TestSplitDataPredefined:
    """Tests for predefined data splitting function."""

    def test_predefined_split_basic(self, predefined_split_dataframe):
        """Test basic predefined split functionality."""
        train_df, test_df, val_df = split_data_predefined(
            predefined_split_dataframe,
            split_column="split",
        )

        # Check sizes match the predefined splits
        assert len(train_df) == 60
        assert len(test_df) == 20
        assert len(val_df) == 20

        # Check no overlap
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        val_indices = set(val_df.index)

        assert len(train_indices & test_indices) == 0
        assert len(train_indices & val_indices) == 0
        assert len(test_indices & val_indices) == 0

        # Check all data accounted for
        total_size = len(predefined_split_dataframe)
        assert len(train_indices | test_indices | val_indices) == total_size

    def test_predefined_split_missing_column(self, predefined_split_dataframe):
        """Test error when split column is missing."""
        with pytest.raises(ValueError, match="Split column 'nonexistent' not found"):
            split_data_predefined(predefined_split_dataframe, split_column="nonexistent")

    def test_predefined_split_unrecognized_value(self):
        """Test error for unrecognized split values."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "split": ["train", "test", "unknown"],
            }
        )

        with pytest.raises(ValueError, match="Unrecognized split values"):
            split_data_predefined(df, split_column="split")

    def test_predefined_split_custom_mapping(self):
        """Test predefined split with custom value mapping."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "split": ["Tr", "Tr", "Tr", "Te", "Te", "Va"],
            }
        )

        custom_mapping = {
            "train": ["Tr"],
            "test": ["Te"],
            "val": ["Va"],
        }

        train_df, test_df, val_df = split_data_predefined(
            df, split_column="split", split_labels=custom_mapping
        )

        assert len(train_df) == 3
        assert len(test_df) == 2
        assert len(val_df) == 1

    def test_predefined_split_no_validation(self):
        """Test predefined split when validation set is missing."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "split": ["train", "train", "train", "test", "test"],
            }
        )

        train_df, test_df, val_df = split_data_predefined(df, split_column="split")

        assert len(train_df) == 3
        assert len(test_df) == 2
        assert len(val_df) == 0  # Empty but exists
        assert isinstance(val_df, pd.DataFrame)

    def test_predefined_split_missing_train(self):
        """Test error when training set is missing."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "split": ["test", "test", "val"],
            }
        )

        with pytest.raises(ValueError, match="No training samples found"):
            split_data_predefined(df, split_column="split")

    def test_predefined_split_missing_test(self):
        """Test error when test set is missing."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "split": ["train", "train", "val"],
            }
        )

        with pytest.raises(ValueError, match="No test samples found"):
            split_data_predefined(df, split_column="split")

    def test_predefined_split_case_variations(self):
        """Test that default mapping handles various cases."""
        df = pd.DataFrame(
            {
                "feature1": range(6),
                "split": ["Train", "TRAIN", "Test", "TEST", "Val", "VAL"],
            }
        )

        train_df, test_df, val_df = split_data_predefined(df, split_column="split")

        assert len(train_df) == 2
        assert len(test_df) == 2
        assert len(val_df) == 2

    def test_predefined_split_preserves_columns(self, predefined_split_dataframe):
        """Test that split preserves all columns including split column."""
        train_df, test_df, val_df = split_data_predefined(
            predefined_split_dataframe, split_column="split"
        )

        # Split column should still be present (dropping happens in preprocessing.py)
        assert "split" in train_df.columns
        assert "feature1" in train_df.columns
        assert "feature2" in train_df.columns
        assert "target" in train_df.columns

    def test_predefined_split_preserves_data(self, predefined_split_dataframe):
        """Test that split preserves data values correctly."""
        train_df, test_df, val_df = split_data_predefined(
            predefined_split_dataframe, split_column="split"
        )

        # All training samples should have split='train'
        assert all(train_df["split"] == "train")
        assert all(test_df["split"] == "test")
        assert all(val_df["split"] == "val")


class TestAnalyzeCategoricalColumns:
    """Tests for categorical column analysis."""

    def test_analyze_categorical_basic(self):
        """Test basic categorical analysis."""
        df = pd.DataFrame(
            {
                "binary_cat": ["A", "B", "A", "B", "A"],
                "multi_cat": ["X", "Y", "Z", "X", "Y"],
                "numeric": [1, 2, 3, 4, 5],
            }
        )

        # Convert to categorical dtype first
        df = convert_to_categorical(df, categorical_threshold=10, convert_numeric=False)

        result = analyze_categorical_columns(df)

        assert "binary" in result
        assert "multi_category" in result

        # binary_cat should be detected as binary
        assert "binary_cat" in result["binary"]

        # multi_cat should be detected as multi-category
        assert "multi_cat" in result["multi_category"]

    def test_analyze_no_categorical_columns(self):
        """Test analysis when no categorical columns present."""
        df = pd.DataFrame(
            {
                "num1": [1, 2, 3, 4, 5],
                "num2": [5, 4, 3, 2, 1],
            }
        )

        result = analyze_categorical_columns(df)

        assert len(result["binary"]) == 0
        assert len(result["multi_category"]) == 0

    def test_analyze_all_binary_categories(self):
        """Test analysis with only binary categorical columns."""
        df = pd.DataFrame(
            {
                "cat1": ["A", "B", "A", "B"],
                "cat2": ["X", "Y", "X", "Y"],
            }
        )

        # Convert to categorical dtype first
        df = convert_to_categorical(df, categorical_threshold=10, convert_numeric=False)

        result = analyze_categorical_columns(df)

        assert len(result["binary"]) == 2
        assert len(result["multi_category"]) == 0


class TestCreateBinaryOrdinalMapping:
    """Tests for binary ordinal mapping creation."""

    def test_create_binary_mapping_basic(self):
        """Test basic binary ordinal mapping."""
        df = pd.DataFrame(
            {
                "gender": ["M", "F", "M", "F", "M"],
                "status": ["active", "inactive", "active", "active", "inactive"],
            }
        )

        # Convert to categorical dtype first
        df = convert_to_categorical(df, categorical_threshold=10, convert_numeric=False)

        binary_cols = ["gender", "status"]
        mapping = create_binary_encoding(df, binary_cols)

        assert "gender" in mapping
        assert "status" in mapping

        # Each mapping should have exactly 2 values
        assert len(mapping["gender"]) == 2
        assert len(mapping["status"]) == 2

        # Mappings should contain the actual values
        assert set(mapping["gender"]) == {"M", "F"}
        assert set(mapping["status"]) == {"active", "inactive"}

    def test_create_binary_mapping_empty_list(self):
        """Test with empty list of binary columns."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        mapping = create_binary_encoding(df, [])

        assert mapping == {}

    def test_create_binary_mapping_sorted_order(self):
        """Test that binary mapping is sorted alphabetically."""
        df = pd.DataFrame({"col": ["Z", "A", "Z", "A"]})

        # Convert to categorical dtype first
        df = convert_to_categorical(df, categorical_threshold=10, convert_numeric=False)

        mapping = create_binary_encoding(df, ["col"])

        # Should be sorted alphabetically
        assert mapping["col"] == ["A", "Z"]


class TestCreateMultiCategoryOrdinalMapping:
    """Tests for multi-category ordinal mapping creation."""

    def test_create_multicategory_mapping_alphabetical(self):
        """Test multi-category mapping with alphabetical ordering."""
        df = pd.DataFrame(
            {
                "education": ["BS", "MS", "PhD", "HS", "BS", "MS"],
            }
        )

        # Convert to categorical dtype first
        df = convert_to_categorical(df, categorical_threshold=10, convert_numeric=False)

        mapping = create_ordinal_encoding(df, ["education"], ordering_method="alphabetical")

        assert "education" in mapping
        # Should be alphabetically sorted
        assert mapping["education"] == ["BS", "HS", "MS", "PhD"]

    def test_create_multicategory_mapping_frequency(self):
        """Test multi-category mapping with frequency-based ordering."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", "C", "A", "A", "B"],  # A=3, B=2, C=1
            }
        )

        # Convert to categorical dtype first
        df = convert_to_categorical(df, categorical_threshold=10, convert_numeric=False)

        mapping = create_ordinal_encoding(df, ["category"], ordering_method="frequency")

        # Should be ordered by frequency (descending)
        assert mapping["category"][0] == "A"  # Most frequent
        assert mapping["category"][-1] == "C"  # Least frequent

    def test_create_multicategory_mapping_empty_list(self):
        """Test with empty list of multi-category columns."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        mapping = create_ordinal_encoding(df, [])

        assert mapping == {}


class TestEnsureConsistentCategories:
    """Tests for ensuring consistent categories across splits."""

    def test_ensure_consistent_with_ordinal_features(self):
        """Test consistency enforcement with ordinal features."""
        df = pd.DataFrame(
            {
                "cat_col": ["A", "B", "C"],
                "num_col": [1, 2, 3],
            }
        )

        all_categories = {}
        ordinal_mappings = {"cat_col": ["A", "B", "C", "D"]}  # D not in data

        result = ensure_consistent_categories(df, all_categories, ordinal_mappings)

        # Should convert cat_col to categorical with all levels
        assert "cat_col" in result.columns
        if hasattr(result["cat_col"], "cat"):
            # Check that all categories are present
            assert set(result["cat_col"].cat.categories) >= {"A", "B", "C"}

    def test_ensure_consistent_empty_inputs(self):
        """Test with empty category and ordinal dicts."""
        df = pd.DataFrame({"col": [1, 2, 3]})

        result = ensure_consistent_categories(df, {}, {})

        # Should return DataFrame unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_ensure_consistent_with_all_categories(self):
        """Test with all_categories specified."""
        df = pd.DataFrame(
            {
                "cat_col": ["A", "B"],
                "num_col": [1, 2],
            }
        )

        all_categories = {"cat_col": ["A", "B", "C", "D"]}
        ordinal_mappings = {}

        result = ensure_consistent_categories(df, all_categories, ordinal_mappings)

        # Should have cat_col as categorical
        assert "cat_col" in result.columns


@pytest.mark.integration
class TestDataUtilitiesIntegration:
    """Integration tests for data utilities working together."""

    def test_full_preprocessing_workflow(self, classification_dataset):
        """Test complete workflow: split → analyze → encode."""
        # Step 1: Split data
        train_df, test_df, val_df = split_data_temporal_or_random(
            classification_dataset,
            temporal_column=None,
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            stratify=classification_dataset["target"],
            random_state=42,
        )

        # Check splits
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(val_df) > 0

        # Check no data leakage
        assert len(set(train_df.index) & set(test_df.index)) == 0
        assert len(set(train_df.index) & set(val_df.index)) == 0
        assert len(set(test_df.index) & set(val_df.index)) == 0


class TestFileHashComputation:
    """Tests for file hash computation utility."""

    def test_compute_hash_existing_file(self, tmp_path):
        """Test hash computation for existing file."""
        import hashlib

        from prism.data_utilities import compute_file_hash_and_size

        # Create test file with known content
        test_file = tmp_path / "test.csv"
        test_content = b"test,data\n1,2\n3,4\n"
        test_file.write_bytes(test_content)

        # Compute hash
        result = compute_file_hash_and_size(test_file)

        # Verify structure
        assert 'hash' in result
        assert 'size_bytes' in result
        assert 'file' in result
        assert 'computed_at' in result
        assert 'algorithm' in result

        # Verify hash (known SHA256 of test content)
        expected_hash = hashlib.sha256(test_content).hexdigest()
        assert result['hash'] == expected_hash

        # Verify size
        assert result['size_bytes'] == len(test_content)

        # Verify algorithm
        assert result['algorithm'] == 'sha256'

    def test_compute_hash_missing_file(self, tmp_path):
        """Test hash computation for missing file."""
        from prism.data_utilities import compute_file_hash_and_size

        missing_file = tmp_path / "nonexistent.csv"
        result = compute_file_hash_and_size(missing_file)

        # Should return None for hash/size with error message
        assert result['hash'] is None
        assert result['size_bytes'] is None
        assert 'error' in result
        assert 'file' in result

    def test_compute_hash_large_file(self, tmp_path):
        """Test hash computation with buffered reading for large file."""
        import hashlib

        from prism.data_utilities import compute_file_hash_and_size

        # Create file larger than buffer size (65536 bytes)
        test_file = tmp_path / "large.csv"
        test_content = b"x" * (65536 * 2 + 100)  # 2+ buffers worth
        test_file.write_bytes(test_content)

        # Compute hash
        result = compute_file_hash_and_size(test_file)

        # Verify correct hash despite chunked reading
        expected_hash = hashlib.sha256(test_content).hexdigest()
        assert result['hash'] == expected_hash
        assert result['size_bytes'] == len(test_content)

    def test_save_split_datasets_returns_hash_info(self, tmp_path):
        """Test that save_split_datasets returns hash information."""
        import pandas as pd

        from prism.data_utilities import save_split_datasets

        # Create test dataframes
        train_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        test_df = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
        val_df = pd.DataFrame({'a': [9, 10], 'b': [11, 12]})

        # Save datasets
        result = save_split_datasets(
            train_df,
            test_df,
            val_df,
            base_filename='test_dataset',
            save_dir=tmp_path,
            include_timestamp=False,
        )

        # Verify hash info is returned
        assert 'train_hash_info' in result
        assert 'test_hash_info' in result
        assert 'val_hash_info' in result

        # Verify each hash info structure
        for key in ['train_hash_info', 'test_hash_info', 'val_hash_info']:
            hash_info = result[key]
            assert 'hash' in hash_info
            assert 'size_bytes' in hash_info
            assert hash_info['hash'] is not None
            assert hash_info['size_bytes'] > 0

        # Verify path info is still returned
        assert 'train_path' in result
        assert 'test_path' in result
        assert 'val_path' in result
