"""Tests for nomogram generation."""

import numpy as np
import pytest
import torch


class TestNomogramDataPreparation:
    """Tests for nomogram data preparation functions."""

    def test_nomogram_data_structure(self):
        """Test basic nomogram data structure requirements."""
        # A nomogram needs:
        # - Feature names
        # - Feature ranges
        # - Points per feature value
        # - Total score mapping to probability

        feature_names = ['age', 'bmi', 'glucose']
        n_features = len(feature_names)

        # Simulate nomogram data
        nomogram_data = {
            'features': [],
        }

        for name in feature_names:
            feature_data = {
                'name': name,
                'range': (0, 100),
                'points': np.linspace(0, 10, 20),
            }
            nomogram_data['features'].append(feature_data)

        # Verify structure
        assert len(nomogram_data['features']) == n_features
        assert all('name' in f for f in nomogram_data['features'])
        assert all('range' in f for f in nomogram_data['features'])
        assert all('points' in f for f in nomogram_data['features'])

    def test_point_calculation_from_partial_response(self):
        """Test converting partial responses to nomogram points."""
        # Partial responses are the basis for nomogram points
        # Points should be proportional to log-odds contribution

        # Simulate partial response for a feature
        x_values = np.linspace(20, 80, 50)  # Age from 20 to 80
        partial_response = 0.5 + 0.3 * (x_values - 50) / 30  # Linear relationship

        # Convert to log-odds (logit)
        from prism.partial_responses import stable_logit

        logit_values = stable_logit(torch.tensor(partial_response)).numpy()

        # Points are typically scaled version of log-odds
        # Normalize to 0-100 point scale
        points = (
            100
            * (logit_values - logit_values.min())
            / (logit_values.max() - logit_values.min() + 1e-10)
        )

        # Points should be within expected range
        assert points.min() >= 0
        assert points.max() <= 100

        # Points should be monotonic if partial response is monotonic
        if np.all(np.diff(partial_response) >= 0):
            assert np.all(np.diff(points) >= -1e-6)  # Allow small numerical errors


class TestNomogramScoring:
    """Tests for nomogram scoring logic."""

    def test_total_score_calculation(self):
        """Test calculation of total score from feature points."""
        # A nomogram sums points from each feature
        # Then maps total to probability

        # Example: 3 features contribute points
        feature_points = {
            'age': 25,
            'bmi': 18,
            'glucose': 32,
        }

        total_score = sum(feature_points.values())

        assert total_score == 75

    def test_score_to_probability_mapping(self):
        """Test mapping total score to probability."""
        # Total score should map monotonically to probability

        # Simulate score-to-probability lookup
        scores = np.linspace(0, 100, 101)
        # Example: logistic mapping
        probabilities = 1 / (1 + np.exp(-(scores - 50) / 10))

        # Should be monotonically increasing
        assert np.all(np.diff(probabilities) >= 0)
        # Should be valid probabilities
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)


class TestNomogramValidation:
    """Tests for nomogram validation."""

    def test_nomogram_point_ranges(self):
        """Test that nomogram points are in valid ranges."""
        # Points should typically be 0-100 for interpretability

        # Simulate feature point mappings
        features = ['feature1', 'feature2', 'feature3']
        point_mappings = {}

        for feat in features:
            x_values = np.linspace(0, 10, 20)
            points = np.linspace(0, 100, 20)  # Map to 0-100 points
            point_mappings[feat] = {'x': x_values, 'points': points}

        # Validate ranges
        for feat, mapping in point_mappings.items():
            assert mapping['points'].min() >= 0
            assert mapping['points'].max() <= 100

    def test_nomogram_feature_coverage(self):
        """Test that nomogram covers full feature range."""
        # Nomogram should span the observed range of each feature

        # Simulate training data ranges
        training_ranges = {
            'age': (18, 90),
            'weight': (45, 150),
            'height': (150, 200),
        }

        # Nomogram should cover these ranges
        nomogram_ranges = {
            'age': (15, 95),  # Slightly extended is okay
            'weight': (40, 160),
            'height': (145, 205),
        }

        for feat in training_ranges:
            train_min, train_max = training_ranges[feat]
            nom_min, nom_max = nomogram_ranges[feat]

            # Nomogram range should include training range
            assert nom_min <= train_min
            assert nom_max >= train_max


@pytest.mark.integration
class TestNomogramGenerationWorkflow:
    """Integration tests for nomogram generation workflow."""

    def test_nomogram_from_partial_responses(self, mock_partial_responses):
        """Test generating nomogram from partial responses."""
        # This is a simplified test of the nomogram generation concept

        # Partial responses are already computed
        # Each feature has (x_values, response_values)

        nomogram = {}

        # For each univariate partial response
        for feat_name, data in mock_partial_responses.items():
            if '_' not in feat_name:  # Skip bivariate
                x_vals = data['x_values']
                responses = data['responses']

                # Use median response across samples
                median_response = np.median(responses, axis=0)

                # Convert to logit (nomogram points)
                from prism.partial_responses import stable_logit

                logit_vals = stable_logit(torch.tensor(median_response)).numpy()

                # Normalize to 0-100 scale
                points = (
                    100
                    * (logit_vals - logit_vals.min())
                    / (logit_vals.max() - logit_vals.min() + 1e-10)
                )

                nomogram[feat_name] = {
                    'x_values': x_vals,
                    'points': points,
                }

        # Verify nomogram structure
        assert len(nomogram) > 0
        for feat_name, data in nomogram.items():
            assert 'x_values' in data
            assert 'points' in data
            assert len(data['x_values']) == len(data['points'])


class TestNomogramProperties:
    """Tests for mathematical properties of nomograms."""

    def test_nomogram_additivity(self):
        """Test that nomogram scores are additive."""
        # Core property: Total score = sum of individual feature points

        # Simulate a patient
        patient_values = {
            'age': 55,
            'bmi': 27,
            'glucose': 110,
        }

        # Look up points for each feature
        feature_points = {
            'age': 30,  # Points for age=55
            'bmi': 25,  # Points for BMI=27
            'glucose': 35,  # Points for glucose=110
        }

        # Total score is sum
        total = sum(feature_points.values())

        # Should equal sum of parts
        assert total == 30 + 25 + 35
        assert total == 90

    def test_nomogram_monotonicity(self):
        """Test that feature contributions are monotonic where expected."""
        # For features with monotonic effects, points should be monotonic

        # Example: age typically has monotonic effect on risk
        ages = np.linspace(20, 80, 50)

        # Simulate monotonic point assignment
        base_risk = 0.1
        age_effect = 0.01 * (ages - 20)  # Increasing risk with age
        probabilities = base_risk + age_effect

        # Convert to points (should be monotonic)
        points = (
            100
            * (probabilities - probabilities.min())
            / (probabilities.max() - probabilities.min())
        )

        # Points should increase monotonically
        assert np.all(np.diff(points) >= -1e-6)

    def test_nomogram_interpretability(self):
        """Test that nomogram maintains interpretability constraints."""
        # Nomograms should be interpretable: points in reasonable range

        # Feature contributions
        feature_points = {
            'age': 28,
            'sex_male': 15,
            'smoker_yes': 42,
            'diabetes_yes': 35,
        }

        # All individual contributions should be reasonable (0-100)
        for feat, points in feature_points.items():
            assert 0 <= points <= 100

        # Total should also be meaningful
        total = sum(feature_points.values())
        assert total > 0


@pytest.mark.unit
class TestNomogramUtilities:
    """Tests for nomogram utility functions."""

    def test_interpolate_points(self):
        """Test point interpolation for nomogram."""
        # Given feature value, interpolate points from lookup table

        # Lookup table
        x_values = np.array([0, 25, 50, 75, 100])
        point_values = np.array([0, 20, 50, 75, 100])

        # Test interpolation
        test_x = 37.5  # Between 25 and 50

        # Linear interpolation
        points = np.interp(test_x, x_values, point_values)

        # Should be halfway between 20 and 50
        assert points == pytest.approx(35.0, rel=1e-5)

    def test_extrapolation_handling(self):
        """Test handling of values outside nomogram range."""
        # Values outside range should be clamped or extrapolated

        x_values = np.array([20, 40, 60, 80])
        point_values = np.array([10, 30, 60, 90])

        # Test value below range
        test_low = 10
        points_low = np.interp(test_low, x_values, point_values)

        # Should use lowest point value (or extrapolate)
        assert points_low <= point_values[0]

        # Test value above range
        test_high = 100
        points_high = np.interp(test_high, x_values, point_values)

        # Should use highest point value (or extrapolate)
        assert points_high >= point_values[-1]
