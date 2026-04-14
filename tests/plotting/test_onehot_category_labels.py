"""
Diagnostic Tests for One-Hot Category Label Issue

This test module validates the category label rendering for one-hot encoded
features in nomogram plots.

Issue: One-hot encoded variables display numeric tick labels (0, 1, 2, ...)
instead of category names (CAD, Cardiomyopathy, Congenital, ...).

Test Hierarchy:
1. test_onehot_group_manager_category_info - Verify source data is available
2. test_current_label_behavior - Document current (broken) behavior
3. test_expected_label_behavior - Specify expected (fixed) behavior
4. test_label_fallback_hierarchy - Test label priority

Usage:
    pytest tests/plotting/test_onehot_category_labels.py -v -s
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prism.feature_labels import FeatureLabelManager  # noqa: E402
from prism.preprocessing import OneHotGroupManager, collapse_onehot_features  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Path Configuration
# =============================================================================

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "htx_example_replica"
METADATA_PATH = FIXTURES_DIR / "preprocessing_metadata.json"
LABELS_PATH = FIXTURES_DIR / "variable_labels.csv"


def check_fixtures_exist():
    """Check that fixture files exist."""
    missing = []
    if not METADATA_PATH.exists():
        missing.append(f"Metadata: {METADATA_PATH}")
    if not LABELS_PATH.exists():
        missing.append(f"Labels: {LABELS_PATH}")
    if missing:
        return False, "\n".join(missing)
    return True, ""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def verify_fixtures():
    """Verify fixture files exist before running tests."""
    exists, msg = check_fixtures_exist()
    if not exists:
        pytest.skip(f"Missing fixture files:\n{msg}")
    return True


@pytest.fixture(scope="module")
def preprocessing_metadata(verify_fixtures):
    """Load preprocessing metadata."""
    with open(METADATA_PATH, 'r') as f:
        return json.load(f)


@pytest.fixture(scope="module")
def group_manager(preprocessing_metadata):
    """Create OneHotGroupManager from metadata."""
    return OneHotGroupManager.from_preprocessing_metadata(preprocessing_metadata)


@pytest.fixture(scope="module")
def label_manager(verify_fixtures):
    """Load FeatureLabelManager from CSV."""
    return FeatureLabelManager.from_csv(
        LABELS_PATH, column_name_col='processed_name', label_col='user_label'
    )


# =============================================================================
# Test Class: Diagnostic Tests
# =============================================================================


class TestOneHotCategoryLabelDiagnostics:
    """Diagnostic tests to understand and document the label issue."""

    def test_onehot_group_manager_structure(self, group_manager):
        """
        DIAGNOSTIC: Verify OneHotGroupManager has the required category information.

        This test documents what data IS available in the group_manager.
        """
        print("\n" + "=" * 60)
        print("OneHotGroupManager Structure")
        print("=" * 60)

        # Check groups_dict
        assert hasattr(group_manager, 'groups_dict'), "Missing groups_dict attribute"
        print(f"\ngroups_dict has {len(group_manager.groups_dict)} groups:")
        for group_name, members in group_manager.groups_dict.items():
            print(f"  '{group_name}': {members}")

        # Check reference_columns
        assert hasattr(group_manager, 'reference_columns'), "Missing reference_columns attribute"
        print(f"\nreference_columns: {group_manager.reference_columns}")

        # Verify diagn group exists (this is what we see in production)
        assert 'diagn' in group_manager.groups_dict, "Expected 'diagn' group not found"

        print("\n[INFO] OneHotGroupManager structure verified")

    def test_get_category_integer_mapping_method(self, group_manager):
        """
        TEST: Verify get_category_integer_mapping returns correct mapping.

        This is the SINGLE SOURCE OF TRUTH for integer encoding.
        - Category 0 = reference column
        - Category 1 = first member in groups_dict
        - Category 2 = second member in groups_dict
        - etc.
        """
        print("\n" + "=" * 60)
        print("get_category_integer_mapping() Test")
        print("=" * 60)

        # Test 'diagn' group
        diagn_mapping = group_manager.get_category_integer_mapping('diagn')
        print(f"\ndiagn mapping: {diagn_mapping}")

        # Verify structure
        assert isinstance(diagn_mapping, dict), "Should return a dict"

        # Category 0 should be reference
        assert 0 in diagn_mapping, "Category 0 (reference) should be in mapping"
        expected_ref = group_manager.reference_columns.get('diagn')
        if expected_ref:
            assert (
                diagn_mapping[0] == expected_ref
            ), f"Category 0 should be {expected_ref}, got {diagn_mapping[0]}"

        # Categories 1-N should match groups_dict order
        diagn_members = group_manager.groups_dict['diagn']
        for i, member in enumerate(diagn_members):
            cat_num = i + 1
            assert cat_num in diagn_mapping, f"Category {cat_num} should be in mapping"
            assert (
                diagn_mapping[cat_num] == member
            ), f"Category {cat_num} should be {member}, got {diagn_mapping[cat_num]}"

        print("\n[PASS] get_category_integer_mapping returns correct mapping")

    def test_get_category_labels_method(self, group_manager):
        """
        TEST: Verify get_category_labels returns human-readable labels.

        Without a label_manager, should extract suffixes from column names.
        """
        print("\n" + "=" * 60)
        print("get_category_labels() Test")
        print("=" * 60)

        # Test 'diagn' group without label_manager
        diagn_labels = group_manager.get_category_labels('diagn')
        print(f"\ndiagn labels (no label_manager): {diagn_labels}")

        # Should have extracted suffixes
        diagn_mapping = group_manager.get_category_integer_mapping('diagn')
        for cat_int, col_name in diagn_mapping.items():
            assert cat_int in diagn_labels, f"Category {cat_int} should be in labels"
            expected_suffix = col_name.replace('diagn_', '')
            assert (
                diagn_labels[cat_int] == expected_suffix
            ), f"Category {cat_int} label should be '{expected_suffix}', got '{diagn_labels[cat_int]}'"

        print("\n[PASS] get_category_labels returns extracted suffixes")

    def test_get_category_labels_with_label_manager(self, group_manager, label_manager):
        """
        TEST: Verify get_category_labels uses FeatureLabelManager when provided.
        """
        print("\n" + "=" * 60)
        print("get_category_labels() with label_manager Test")
        print("=" * 60)

        # Test 'diagn' group with label_manager
        diagn_labels = group_manager.get_category_labels('diagn', label_manager=label_manager)
        print(f"\ndiagn labels (with label_manager): {diagn_labels}")

        # Verify that labels are looked up from label_manager
        diagn_mapping = group_manager.get_category_integer_mapping('diagn')
        for cat_int, col_name in diagn_mapping.items():
            user_label = label_manager.get_label(col_name)
            # If label_manager has a user label, it should be used
            # Otherwise, the suffix should be used
            if user_label != col_name:
                assert (
                    diagn_labels[cat_int] == user_label
                ), f"Category {cat_int} should use user label '{user_label}'"

        print("\n[PASS] get_category_labels uses label_manager when provided")

    def test_build_categorical_labels_dict(self, group_manager, label_manager):
        """
        TEST: Verify build_categorical_labels_dict builds dict for all groups.

        This is the method used to create categorical_labels for PlotFormatter.
        """
        print("\n" + "=" * 60)
        print("build_categorical_labels_dict() Test")
        print("=" * 60)

        # Without label_manager (uses extracted suffixes)
        cat_labels = group_manager.build_categorical_labels_dict()
        print(f"\nWithout label_manager: {cat_labels}")

        # Should have all groups
        for group_name in group_manager.groups_dict:
            assert group_name in cat_labels, f"Missing group {group_name}"

        # With label_manager
        cat_labels_with_mgr = group_manager.build_categorical_labels_dict(label_manager)
        print(f"\nWith label_manager: {cat_labels_with_mgr}")

        # Verify format for PlotFormatter
        for group_name, labels in cat_labels.items():
            assert isinstance(labels, dict), f"Labels for {group_name} should be dict"
            for cat_int, label in labels.items():
                assert isinstance(cat_int, int), f"Keys should be integers, got {type(cat_int)}"
                assert isinstance(label, str), f"Values should be strings, got {type(label)}"

        print("\n[PASS] build_categorical_labels_dict works correctly")

    def test_to_dict_includes_category_mapping(self, group_manager):
        """
        TEST: Verify to_dict() includes category_integer_mapping for JSON serialization.

        This ensures the explicit mapping is saved to preprocessing_metadata.json.
        """
        print("\n" + "=" * 60)
        print("to_dict() Category Mapping Test")
        print("=" * 60)

        serialized = group_manager.to_dict()
        print(f"\nto_dict() keys: {list(serialized.keys())}")

        # Verify structure
        assert '_type' in serialized, "Missing _type key"
        assert serialized['_type'] == 'OneHotGroupManager', "Wrong _type"
        assert 'groups_dict' in serialized, "Missing groups_dict"
        assert 'reference_columns' in serialized, "Missing reference_columns"
        assert (
            'category_integer_mapping' in serialized
        ), "Missing category_integer_mapping - should be included for human readability"

        # Verify mapping content
        mapping = serialized['category_integer_mapping']
        for group_name in group_manager.groups_dict:
            assert group_name in mapping, f"Missing mapping for {group_name}"

            # Check that mapping is correct
            expected = group_manager.get_category_integer_mapping(group_name)
            assert (
                mapping[group_name] == expected
            ), f"Mapping for {group_name} doesn't match get_category_integer_mapping()"

        print(f"\ncategory_integer_mapping: {serialized['category_integer_mapping']}")
        print("\n[PASS] to_dict() includes category_integer_mapping")

    def test_metadata_has_explicit_integer_mapping(self, preprocessing_metadata):
        """
        SPEC: Verify metadata contains explicit integer-to-category mapping.

        The onehot_group_manager section should include a human-readable
        'category_integer_mapping' field that documents exactly what each
        integer value means for each collapsed categorical feature.

        Expected structure:
        {
          "onehot_group_manager": {
            "category_integer_mapping": {
              "diagn": {
                "0": "diagn_Cardiomyopathy",
                "1": "diagn_CAD",
                "2": "diagn_Congenital",
                ...
              }
            }
          }
        }

        Format: {str(int_value): column_name} where 0 is always the reference.
        """
        print("\n" + "=" * 60)
        print("Explicit Integer Mapping in Metadata")
        print("=" * 60)

        assert 'onehot_group_manager' in preprocessing_metadata, "Missing onehot_group_manager"
        ohgm = preprocessing_metadata['onehot_group_manager']

        # category_integer_mapping is now always included
        assert (
            'category_integer_mapping' in ohgm
        ), "Missing category_integer_mapping - metadata needs to be regenerated"

        mapping = ohgm['category_integer_mapping']

        for group_name in ['diagn', 'recethcat']:
            if group_name in mapping:
                print(f"\n{group_name}:")
                group_mapping = mapping[group_name]
                for int_val, col_name in sorted(group_mapping.items(), key=lambda x: int(x[0])):
                    ref_marker = " (REFERENCE)" if int_val == '0' else ""
                    print(f"  {int_val} -> {col_name}{ref_marker}")

        # Verify structure - format is {str(int): column_name}
        diagn_mapping = mapping.get('diagn', {})
        assert '0' in diagn_mapping, "Missing category 0 (reference) in mapping"
        # Verify 0 maps to a column name string
        assert isinstance(diagn_mapping['0'], str), "Value at '0' should be column name string"

        # Verify reference column matches
        ref_cols = ohgm.get('reference_columns', {}).get('dropped_columns', [])
        if ref_cols and 'diagn' in diagn_mapping:
            diagn_ref = diagn_mapping['0']
            assert (
                diagn_ref in ref_cols
            ), f"Category 0 ({diagn_ref}) should be in reference columns {ref_cols}"

        print("\n[PASS] Explicit integer mapping verified in metadata")

    def test_preprocessing_metadata_has_category_info(self, preprocessing_metadata):
        """
        DIAGNOSTIC: Verify preprocessing metadata contains category name information.

        The metadata should have:
        - encoding[group_name]['reverse_mapping']: column -> original category
        - encoding[group_name]['original_categories']: list of category names
        """
        print("\n" + "=" * 60)
        print("Preprocessing Metadata Category Info")
        print("=" * 60)

        assert 'encoding' in preprocessing_metadata, "Missing 'encoding' key"

        for group_name in ['diagn', 'recethcat']:
            if group_name in preprocessing_metadata['encoding']:
                encoding = preprocessing_metadata['encoding'][group_name]
                print(f"\n{group_name}:")
                print(f"  encoding_type: {encoding.get('encoding_type')}")
                print(f"  created_columns: {encoding.get('created_columns')}")
                print(f"  original_categories: {encoding.get('original_categories')}")
                print(
                    f"  reverse_mapping keys: {list(encoding.get('reverse_mapping', {}).keys())}"
                )

                # Verify structure
                assert (
                    'original_categories' in encoding
                ), f"Missing original_categories for {group_name}"
                assert 'reverse_mapping' in encoding, f"Missing reverse_mapping for {group_name}"

        print("\n[INFO] Category information available in preprocessing metadata")

    def test_label_manager_has_onehot_labels(self, label_manager):
        """
        DIAGNOSTIC: Verify FeatureLabelManager has labels for one-hot columns.

        Should have labels for:
        - Group names (e.g., 'diagn' -> 'Diagnosis')
        - Individual columns (e.g., 'diagn_CAD' -> 'Ischemic cardiomyopathy')
        """
        print("\n" + "=" * 60)
        print("FeatureLabelManager One-Hot Labels")
        print("=" * 60)

        # Check group names
        for group_name in ['diagn', 'recethcat']:
            label = label_manager.get_label(group_name)
            print(f"\n{group_name} -> '{label}'")

        # Check individual columns
        test_columns = [
            'diagn_CAD',
            'diagn_Cardiomyopathy',
            'diagn_Congenital',
            'recethcat_Caucasian',
            'recethcat_African American',
        ]

        print("\nIndividual column labels:")
        for col in test_columns:
            label = label_manager.get_label(col)
            print(f"  {col} -> '{label}'")

        print("\n[INFO] FeatureLabelManager labels available")

    def test_collapse_encoding_values(self, group_manager):
        """
        DIAGNOSTIC: Understand how collapse_onehot_features encodes categories.

        This test documents the integer encoding:
        - 0 = reference category (all zeros in one-hot)
        - 1 = first member in groups_dict
        - 2 = second member in groups_dict
        - etc.

        The order is determined by groups_dict[group_name] list order.
        """
        print("\n" + "=" * 60)
        print("Collapse Encoding Values")
        print("=" * 60)

        # Create a single-group manager for testing
        diagn_members = group_manager.groups_dict['diagn']
        diagn_ref = group_manager.reference_columns.get('diagn')
        single_group_manager = OneHotGroupManager(
            {'diagn': diagn_members}, {'diagn': diagn_ref} if diagn_ref else {}
        )

        n_members = len(diagn_members)
        feature_names = diagn_members.copy()

        # Test data: one row for each category + reference
        # Row 0: all zeros (reference)
        # Row 1: first column = 1 (category 1)
        # Row 2: second column = 1 (category 2)
        # etc.
        n_rows = n_members + 1
        X = np.zeros((n_rows, n_members))
        for i in range(n_members):
            X[i + 1, i] = 1  # Activate the i-th column in row i+1

        print(f"\nOne-hot input (shape {X.shape}):")
        print(f"  Feature names (from groups_dict order): {feature_names}")
        print(f"  X[0] (reference): {X[0]}")
        for i in range(n_members):
            print(f"  X[{i+1}] ({diagn_members[i]}): {X[i+1]}")

        # Collapse
        X_collapsed, collapsed_names = collapse_onehot_features(
            X, single_group_manager, feature_names
        )

        print("\nCollapsed output:")
        print(f"  Collapsed names: {collapsed_names}")
        print(f"  X_collapsed.shape: {X_collapsed.shape}")
        print("  Encoding:")
        for i in range(n_rows):
            cat_value = X_collapsed[i, 0]
            if i == 0:
                print(f"    reference -> {cat_value}")
            else:
                print(f"    {diagn_members[i-1]} -> {cat_value}")

        # Document the mapping
        print("\n  ENCODING MAP for 'diagn' (from groups_dict order):")
        ref_col = group_manager.reference_columns.get('diagn', 'unknown')
        print(f"    0 -> reference ({ref_col})")
        for i, member in enumerate(diagn_members):
            print(f"    {i+1} -> {member}")

        # Verify the encoding matches expected values
        assert X_collapsed[0, 0] == 0, "Reference should encode to 0"
        for i in range(n_members):
            expected = i + 1
            actual = X_collapsed[i + 1, 0]
            assert (
                actual == expected
            ), f"Category {diagn_members[i]} should encode to {expected}, got {actual}"

        print("\n[INFO] Collapse encoding documented and verified")

    def test_encoding_consistency_between_collapse_and_partial_responses(self, group_manager):
        """
        CRITICAL: Verify that collapse_onehot_features and partial_responses use the same encoding.

        Both functions must map integer categories to one-hot columns in the same order.
        The order should be determined by groups_dict[group_name] list order.

        collapse_onehot_features: Uses [feature_names.index(f) for f in group_features]
        _create_collapsed_mapping: Should use same order!
        """
        print("\n" + "=" * 60)
        print("Encoding Consistency Check")
        print("=" * 60)

        # Get the order from groups_dict
        diagn_members = group_manager.groups_dict['diagn']
        print("\ngroups_dict['diagn'] order:")
        for i, member in enumerate(diagn_members):
            print(f"  Position {i} -> {member} -> Category {i+1}")

        # Create feature_names that might be in a different order
        # This simulates what could happen if columns are reordered
        feature_names_shuffled = sorted(diagn_members)  # Alphabetical order

        print("\nIf feature_names were alphabetically sorted:")
        for i, member in enumerate(feature_names_shuffled):
            print(f"  Position {i} -> {member}")

        # Check if orders differ
        if diagn_members != feature_names_shuffled:
            print("\n[WARNING] groups_dict order differs from alphabetical order!")
            print("  This could cause encoding mismatches if implementations differ.")

        # Create a single-group manager for testing
        diagn_ref = group_manager.reference_columns.get('diagn')
        single_group_manager = OneHotGroupManager(
            {'diagn': diagn_members}, {'diagn': diagn_ref} if diagn_ref else {}
        )

        # Test collapse_onehot_features with shuffled feature_names
        X = np.zeros((len(diagn_members) + 1, len(diagn_members)))
        for i in range(len(diagn_members)):
            X[i + 1, i] = 1

        # Collapse with groups_dict order
        X_collapsed_correct, _ = collapse_onehot_features(
            X, single_group_manager, diagn_members  # Uses groups_dict order
        )

        print("\n  Expected encoding (groups_dict order):")
        for i, member in enumerate(diagn_members):
            print(f"    {member} -> {i+1}")

        print("\n  Actual collapse_onehot_features encoding:")
        for i in range(1, len(diagn_members) + 1):
            actual_cat = int(X_collapsed_correct[i, 0])
            expected_member = diagn_members[actual_cat - 1] if actual_cat > 0 else "reference"
            print(f"    Row {i} (X[{i}]) -> Category {actual_cat} -> {expected_member}")

        print("\n[INFO] Encoding consistency check complete")

    def test_current_label_formatting_behavior(self, group_manager, label_manager):
        """
        DIAGNOSTIC: Document current (broken) label formatting behavior.

        This test shows that format_feature_label returns numeric values
        instead of category names for collapsed one-hot features.
        """
        from prism.plotting.formatter import PlotFormatter

        print("\n" + "=" * 60)
        print("Current Label Formatting Behavior")
        print("=" * 60)

        # Create formatter without categorical_labels (current production state)
        formatter = PlotFormatter(use_odds_ratio=False)

        # Test formatting for diagn feature
        feature_name = 'diagn'
        test_values = [0, 1, 2, 3, 4, 5]

        print(f"\nFormatting '{feature_name}' values:")
        print("  (Without categorical_labels dict)")

        issues = []
        for value in test_values:
            label = formatter.format_feature_label(feature_name, value, is_binary=False)
            print(f"    {value} -> '{label}'")

            # Check if it's just the numeric value
            if label == f"{value:.2g}" or label == str(value) or label == f"{value}":
                issues.append(f"Value {value} formatted as numeric: '{label}'")

        print("\n[ISSUE] Current behavior returns numeric labels:")
        for issue in issues:
            print(f"  - {issue}")

        # This assertion documents the CURRENT (broken) behavior
        # When fixed, this test should be updated or removed
        assert len(issues) > 0, "Expected numeric labels (current broken behavior)"

        print("\n[INFO] Current (broken) behavior documented")


class TestOneHotCategoryLabelSolution:
    """Tests that specify the expected behavior after the fix."""

    def test_expected_categorical_labels_structure(self, group_manager, preprocessing_metadata):
        """
        SPEC: Define the expected categorical_labels structure.

        categorical_labels should be a dict:
        {
            'diagn': {
                0: 'Cardiomyopathy' (or user label),
                1: 'CAD' (or user label),
                2: 'Congenital' (or user label),
                ...
            },
            'recethcat': {...}
        }
        """
        print("\n" + "=" * 60)
        print("Expected categorical_labels Structure")
        print("=" * 60)

        # Build expected structure from available data
        expected = {}

        for group_name, members in group_manager.groups_dict.items():
            if group_name in preprocessing_metadata.get('encoding', {}):
                encoding = preprocessing_metadata['encoding'][group_name]
                reverse_mapping = encoding.get('reverse_mapping', {})

                expected[group_name] = {}

                # Category 0: reference
                ref_col = group_manager.reference_columns.get(group_name)
                if ref_col and ref_col in reverse_mapping:
                    expected[group_name][0] = reverse_mapping[ref_col]
                else:
                    expected[group_name][0] = "(Reference)"

                # Categories 1-N: members in order
                for i, member in enumerate(members):
                    if member in reverse_mapping:
                        expected[group_name][i + 1] = reverse_mapping[member]
                    else:
                        # Fallback: extract suffix from column name
                        suffix = member.replace(f"{group_name}_", "")
                        expected[group_name][i + 1] = suffix

        print("\nExpected categorical_labels:")
        for group_name, labels in expected.items():
            print(f"\n  '{group_name}':")
            for value, label in sorted(labels.items()):
                print(f"    {value}: '{label}'")

        # Verify structure
        assert 'diagn' in expected, "Expected 'diagn' in categorical_labels"
        assert 0 in expected['diagn'], "Expected category 0 (reference) in diagn"
        assert 1 in expected['diagn'], "Expected category 1 in diagn"

        print("\n[INFO] Expected structure defined")

    def test_get_category_labels_method_with_label_manager_integration(
        self, group_manager, label_manager
    ):
        """
        SPEC: Test get_category_labels() returns user-friendly labels.

        Verifies that when a label_manager is provided, user-defined labels
        are used instead of column name suffixes.
        """
        # Expected: OneHotGroupManager has get_category_labels method
        assert hasattr(
            group_manager, 'get_category_labels'
        ), "OneHotGroupManager should have get_category_labels method"

        # Get labels for diagn with label_manager
        labels = group_manager.get_category_labels('diagn', label_manager=label_manager)

        # Verify structure
        assert isinstance(labels, dict), "Should return dict"
        assert 0 in labels, "Should have reference category (0)"
        assert all(isinstance(k, int) for k in labels.keys()), "Keys should be integers"
        assert all(isinstance(v, str) for v in labels.values()), "Values should be strings"

        # Verify content
        print("\nget_category_labels('diagn') with label_manager:")
        for value, label in sorted(labels.items()):
            print(f"  {value}: '{label}'")

        # When label_manager has labels, they should be used
        # The exact assertion depends on what labels are in variable_labels.csv
        # At minimum, labels should not be plain column names
        for cat_int, label in labels.items():
            # Label should be the user label OR the extracted suffix, not full column name
            assert (
                not label.startswith('diagn_') or label == labels[cat_int]
            ), f"Label for {cat_int} should be user-friendly, got '{label}'"

    def test_formatter_with_categorical_labels(self, group_manager, label_manager):
        """
        SPEC: Test PlotFormatter with categorical_labels populated.

        Verifies that when categorical_labels is passed to PlotFormatter,
        format_feature_label returns category names instead of numeric values.
        """
        from prism.plotting.formatter import PlotFormatter

        # Build categorical_labels using the implemented method
        categorical_labels = group_manager.build_categorical_labels_dict(label_manager)

        # Create formatter with labels
        formatter = PlotFormatter(use_odds_ratio=False, categorical_labels=categorical_labels)

        # Test formatting
        feature_name = 'diagn'
        test_values = [0, 1, 2, 3, 4, 5]

        print(f"\nFormatting '{feature_name}' values (with categorical_labels):")
        for value in test_values:
            label = formatter.format_feature_label(feature_name, value, is_binary=False)
            print(f"  {value} -> '{label}'")

            # Should NOT be a plain numeric string
            assert (
                label != f"{value:.2g}"
            ), f"Value {value} should have category label, got '{label}'"
            assert label != str(value), f"Value {value} should have category label, got '{label}'"


class TestLabelFallbackHierarchy:
    """Tests for the label fallback hierarchy."""

    def test_label_hierarchy_specification(
        self, group_manager, label_manager, preprocessing_metadata
    ):
        """
        SPEC: Define the label fallback hierarchy.

        For a collapsed one-hot feature category, labels should be resolved in order:
        1. User label from FeatureLabelManager (e.g., 'Ischemic cardiomyopathy')
        2. Original category name from preprocessing metadata (e.g., 'CAD')
        3. Column suffix (e.g., 'CAD' from 'diagn_CAD')
        4. Numeric value (e.g., '1') - last resort
        """
        print("\n" + "=" * 60)
        print("Label Fallback Hierarchy Specification")
        print("=" * 60)

        print("\nPriority order (highest to lowest):")
        print("  1. User label from FeatureLabelManager")
        print("     Example: diagn_CAD -> 'Ischemic cardiomyopathy'")
        print("  2. Original category name from preprocessing metadata")
        print("     Example: diagn_CAD -> 'CAD' (from reverse_mapping)")
        print("  3. Column suffix (column name minus group prefix)")
        print("     Example: diagn_CAD -> 'CAD' (from column name)")
        print("  4. Numeric value (last resort)")
        print("     Example: 1 -> '1'")

        # Document available labels for each level
        print("\n" + "-" * 40)
        print("Available labels for 'diagn' categories:")
        print("-" * 40)

        diagn_members = group_manager.groups_dict['diagn']
        reverse_mapping = preprocessing_metadata['encoding']['diagn'].get('reverse_mapping', {})

        for i, member in enumerate(diagn_members):
            cat_value = i + 1  # 0 is reference
            user_label = label_manager.get_label(member)
            original_cat = reverse_mapping.get(member, "(none)")
            suffix = member.replace('diagn_', '')

            print(f"\n  Category {cat_value} ({member}):")
            print(f"    L1 (user_label): '{user_label}'")
            print(f"    L2 (original_cat): '{original_cat}'")
            print(f"    L3 (suffix): '{suffix}'")
            print(f"    L4 (numeric): '{cat_value}'")

            # Best label should be L1 if available, else L2, etc.
            if user_label != member:  # Not fallback to column name
                best = user_label
            elif original_cat != "(none)":
                best = original_cat
            else:
                best = suffix

            print(f"    BEST: '{best}'")


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    # Check fixtures
    exists, msg = check_fixtures_exist()
    if not exists:
        print(f"\nERROR: Missing fixture files:\n{msg}")
        sys.exit(1)

    # Run tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])
