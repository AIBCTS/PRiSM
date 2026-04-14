import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class IMPACTFeatureTransformer(nn.Module):
    """
    Transforms raw features into IMPACT score components using PyTorch operations.
    All transformations are vectorized and CUDA-compatible.

    Note: Bilirubin input expected in umol/L. Thresholds are pre-scaled from the
    original IMPACT paper (Weiss et al. 2011) which used mg/dL.
    """

    # Bilirubin thresholds in umol/L (converted from Weiss et al. 2011: 1, 2, 4 mg/dL)
    # Conversion factor: 1 mg/dL = 17.1 umol/L
    BILI_THRESHOLD_1 = 17.1  # 1 mg/dL
    BILI_THRESHOLD_2 = 34.2  # 2 mg/dL
    BILI_THRESHOLD_4 = 68.4  # 4 mg/dL

    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Transform input features into IMPACT score components.

        Args:
            features: Tensor of shape (batch_size, 18) with features in expected order
                     features[:, 1] = recbilirubin in umol/L

        Returns:
            Tensor of shape (batch_size, num_components) with IMPACT score components
        """
        # Pre-allocate output tensor for better performance
        batch_size = features.shape[0]
        components = torch.zeros(batch_size, 19, device=features.device, dtype=features.dtype)

        # Convert binary features to float once for efficiency
        binary_features = features[
            :, 3:
        ].float()  # All features from index 3 onwards are binary (15 features)
        binary_bool = features[:, 3:].bool()  # Boolean version for logical operations

        # Extract continuous features
        recageyear = features[:, 0]
        recbilirubin = features[:, 1]  # in umol/L
        creatinine_clearance = features[:, 2]

        # 1. Age > 60: 3 points
        components[:, 0] = (recageyear > 60).float() * 3

        # 2. Bilirubin scoring (thresholds in umol/L, converted from paper)
        # Original paper (mg/dL): 1-2: 1pt, 2-4: 3pts, >=4: 4pts
        # Converted (umol/L): 17.1-34.2: 1pt, 34.2-68.4: 3pts, >=68.4: 4pts
        components[:, 1] = (
            (recbilirubin >= self.BILI_THRESHOLD_1) & (recbilirubin < self.BILI_THRESHOLD_2)
        ).float()
        components[:, 2] = (
            (recbilirubin >= self.BILI_THRESHOLD_2) & (recbilirubin < self.BILI_THRESHOLD_4)
        ).float() * 3
        components[:, 3] = (recbilirubin >= self.BILI_THRESHOLD_4).float() * 4

        # 3. Creatinine clearance - vectorized
        components[:, 4] = ((creatinine_clearance >= 30) & (creatinine_clearance < 50)).float() * 2
        components[:, 5] = (creatinine_clearance < 30).float() * 5

        # 4-19. Binary features (multiply by their respective points)
        # Feature indices in 18-feature tensor: recsex(0), recdialysis(1), diagn_CAD(2), diagn_Congenital(3),
        # diagn_Valve(4), diagn_Graftfailure(5), diagn_Misc(6), recinfections2weeks(7), reciabp(8),
        # recventilator(9), recethcat_African American(10), recethcat_Hispanic(11), recethcat_Other(12),
        # rececmo(13), recvad(14)

        # Apply points to binary features
        components[:, 6] = binary_features[:, 1] * 4  # dialysis
        components[:, 7] = binary_features[:, 0] * 3  # female sex
        components[:, 8] = binary_features[:, 2] * 2  # CAD
        components[:, 9] = binary_features[:, 3] * 5  # congenital

        other_conditions = (
            binary_bool[:, 4]  # Valve
            | binary_bool[:, 5]  # Graftfailure
            | binary_bool[:, 6]  # Misc
        )
        components[:, 10] = other_conditions.float()

        components[:, 11] = binary_features[:, 7] * 3  # infection
        components[:, 12] = binary_features[:, 8] * 3  # iabp
        components[:, 13] = binary_features[:, 9] * 5  # ventilator
        components[:, 14] = binary_features[:, 10] * 3  # african_american
        components[:, 15] = binary_features[:, 11] * 0  # hispanic (0 points)
        components[:, 16] = binary_features[:, 12] * 0  # other_race (0 points)
        components[:, 17] = binary_features[:, 13] * 7  # ecmo
        components[:, 18] = binary_features[:, 14] * 3  # vad

        return components


class IMPACTModel(nn.Module):
    """
    Complete IMPACT model for heart transplant survival prediction (Weiss et al, 2011).
    Handles feature transformation and final prediction in PyTorch.

    Input Requirements:
    - Bilirubin (recbilirubin) expected in umol/L (thresholds pre-scaled internally)
    - Creatinine clearance (creatinine_clearance) expected in mL/min (pre-calculated)
    - All other features as specified in FEATURE_NAMES

    Note: This implementation has some known simplifications from the original paper
    (doi:10.1016/j.athoracsur.2011.04.030):

    1. VAD Scoring Simplification:
       - All VADs are assigned 3 points (older generation pulsatile score)
       - Original paper distinguishes:
         * Older generation pulsatile: 3 points
         * New generation continuous (excluding HMII): 5 points
         * Heartmate II: 0 points
       - Maximum possible score is 48 instead of original 50 due to this simplification

    2. Heart Failure Etiology Mapping:
       - CAD → Ischemic (2 points)
       - Congenital → Congenital (5 points)
       - Cardiomyopathy → Idiopathic (0 points) - BASELINE/REFERENCE CATEGORY
       - Other conditions (1 point) only count if no primary condition present
         (CAD, Cardiomyopathy, or Congenital)
    """

    # Define the expected feature order (18 features with precalculated creatinine_clearance)
    # Note: diagn_Cardiomyopathy is the reference category (baseline) and is not included
    # Note: recethcat_Caucasian is the reference category (baseline) and is not included
    FEATURE_NAMES = [
        'recageyear',
        'recbilirubin',
        'creatinine_clearance',
        'recsex',
        'recdialysis',
        'diagn_CAD',
        'diagn_Congenital',
        'diagn_Valve',
        'diagn_Graftfailure',
        'diagn_Misc',
        'recinfections2weeks',
        'reciabp',
        'recventilator',
        'recethcat_African American',
        'recethcat_Hispanic',
        'recethcat_Other',
        'rececmo',
        'recvad',
    ]

    # Logit coefficients from multivariable analysis (ln(OR))
    LOGIT_COEFFICIENTS = {
        'age_gt_60': 0.3001,  # ln(1.35)
        'bilirubin_1_2': 0.2469,  # ln(1.28)
        'bilirubin_2_4': 0.3988,  # ln(1.49)
        'bilirubin_4_plus': 0.6739,  # ln(1.96)
        'creat_30_50': 0.1906,  # ln(1.21)
        'creat_under_30': 0.8961,  # ln(2.45)
        'dialysis': 0.6575,  # ln(1.93)
        'female': 0.3293,  # ln(1.39)
        'ischemic': 0.2624,  # ln(1.30)
        'congenital': 1.0296,  # ln(2.80)
        'other_etiology': 0.1989,  # ln(1.22)
        'infection': 0.2852,  # ln(1.33)
        'iabp': 0.2311,  # ln(1.26)
        'ventilator': 0.7419,  # ln(2.10)
        'african_american': 0.3075,  # ln(1.36)
        'hispanic': 0.0677,  # ln(1.07)
        'other_race': -0.0202,  # ln(0.98)
        'ecmo': 1.1814,  # ln(3.26)
        'vad': 0.2624,  # ln(1.30) - older gen pulsatile
    }

    def __init__(self):
        super().__init__()
        self.feature_transformer = IMPACTFeatureTransformer()

        # IMPACT score coefficients (all 1s since we handle weighting in transformer)
        self.score_weights = nn.Parameter(
            torch.ones(19),
            requires_grad=False,  # 19 components from transformer (added 2 race categories)
        )

        # Final prediction parameters from the paper
        self.impact_coef = nn.Parameter(torch.tensor(0.13), requires_grad=False)
        self.impact_intercept = nn.Parameter(torch.tensor(-2.75), requires_grad=False)

    def _calculate_logit_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate logit contributions using multivariable OR coefficients.

        Args:
            x: Input tensor of shape (batch_size, 18)
               x[:, 1] = recbilirubin in umol/L

        Returns:
            Tensor of shape (batch_size, 19) with logit contributions for each component
        """
        # Pre-allocate output tensor
        batch_size = x.shape[0]
        logit_contributions = torch.zeros(batch_size, 19, device=x.device, dtype=x.dtype)

        # Convert binary features to float once
        binary_features = x[:, 3:].float()  # 15 binary features
        binary_bool = x[:, 3:].bool()  # Boolean version for logical operations

        # Extract continuous features
        recageyear = x[:, 0]
        recbilirubin = x[:, 1]  # in umol/L
        creatinine_clearance = x[:, 2]

        # Create coefficient tensors for vectorized operations
        coef = self.LOGIT_COEFFICIENTS

        # 1. Age > 60
        logit_contributions[:, 0] = (recageyear > 60).float() * coef['age_gt_60']

        # 2. Bilirubin - vectorized (thresholds in umol/L, converted from paper)
        # Original paper (mg/dL): 1-2, 2-4, >=4
        # Converted (umol/L): 17.1-34.2, 34.2-68.4, >=68.4
        logit_contributions[:, 1] = (
            (recbilirubin >= self.feature_transformer.BILI_THRESHOLD_1)
            & (recbilirubin < self.feature_transformer.BILI_THRESHOLD_2)
        ).float() * coef['bilirubin_1_2']
        logit_contributions[:, 2] = (
            (recbilirubin >= self.feature_transformer.BILI_THRESHOLD_2)
            & (recbilirubin < self.feature_transformer.BILI_THRESHOLD_4)
        ).float() * coef['bilirubin_2_4']
        logit_contributions[:, 3] = (
            recbilirubin >= self.feature_transformer.BILI_THRESHOLD_4
        ).float() * coef['bilirubin_4_plus']

        # 3. Creatinine clearance - vectorized
        logit_contributions[:, 4] = (
            (creatinine_clearance >= 30) & (creatinine_clearance < 50)
        ).float() * coef['creat_30_50']
        logit_contributions[:, 5] = (creatinine_clearance < 30).float() * coef['creat_under_30']

        # 4-19. Binary features with their coefficients
        # Feature indices in 18-feature tensor: recsex(0), recdialysis(1), diagn_CAD(2), diagn_Congenital(3),
        # diagn_Valve(4), diagn_Graftfailure(5), diagn_Misc(6), recinfections2weeks(7), reciabp(8),
        # recventilator(9), recethcat_African American(10), recethcat_Hispanic(11), recethcat_Other(12),
        # rececmo(13), recvad(14)

        logit_contributions[:, 6] = binary_features[:, 1] * coef['dialysis']  # dialysis
        logit_contributions[:, 7] = binary_features[:, 0] * coef['female']  # female sex
        logit_contributions[:, 8] = binary_features[:, 2] * coef['ischemic']  # CAD
        logit_contributions[:, 9] = binary_features[:, 3] * coef['congenital']  # congenital

        # Other etiology: only if not CAD, Cardiomyopathy, or Congenital
        other_conditions = (
            binary_bool[:, 4]  # Valve
            | binary_bool[:, 5]  # Graftfailure
            | binary_bool[:, 6]  # Misc
        )
        logit_contributions[:, 10] = other_conditions.float() * coef['other_etiology']

        logit_contributions[:, 11] = binary_features[:, 7] * coef['infection']  # infection
        logit_contributions[:, 12] = binary_features[:, 8] * coef['iabp']  # iabp
        logit_contributions[:, 13] = binary_features[:, 9] * coef['ventilator']  # ventilator
        logit_contributions[:, 14] = (
            binary_features[:, 10] * coef['african_american']
        )  # african_american
        logit_contributions[:, 15] = binary_features[:, 11] * coef['hispanic']  # hispanic
        logit_contributions[:, 16] = binary_features[:, 12] * coef['other_race']  # other_race
        logit_contributions[:, 17] = binary_features[:, 13] * coef['ecmo']  # ecmo
        logit_contributions[:, 18] = binary_features[:, 14] * coef['vad']  # vad

        return logit_contributions

    def forward(
        self, x: torch.Tensor, calculate_impact_score: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the IMPACT model.

        Args:
            x: Input tensor of shape (batch_size, 18) with features in expected order
            calculate_impact_score: If True, calculates and returns the impact score. Defaults to False for better performance.

        Returns:
            Dictionary with 'mortality_prob_points', 'mortality_prob_logit', and optionally 'impact_score' tensors
        """
        # Calculate mortality probability using logit coefficients from multivariable analysis
        logit_contributions = self._calculate_logit_contributions(x)
        total_logit = torch.sum(logit_contributions, dim=1) + self.impact_intercept
        mortality_prob_logit = torch.sigmoid(total_logit)

        # Initialize result dictionary
        result = {
            'impact_score': None,
            'mortality_prob_points': None,
            'mortality_prob_logit': mortality_prob_logit,
        }

        # Only calculate impact score and points-based mortality if requested
        if calculate_impact_score:
            # Transform features to score components
            score_components = self.feature_transformer(x)
            impact_score = torch.sum(score_components, dim=1)

            # Calculate mortality probability using original points-based approach
            logit_points = self.impact_coef * impact_score + self.impact_intercept

            result.update(
                {
                    'impact_score': impact_score,
                    'mortality_prob_points': torch.sigmoid(logit_points),
                }
            )

        return result

    @torch.no_grad()
    def predict(
        self,
        x: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        device: Optional[str] = None,
        calculate_impact_score: bool = False,
    ) -> torch.Tensor:
        """
        Make predictions using the trained model.

        This method sets the model to evaluation mode and performs a forward pass
        without computing gradients.

        Parameters
        ----------
        x : Union[np.ndarray, pd.DataFrame, torch.Tensor]
            The input data for prediction. Can be a NumPy array, Pandas DataFrame, or PyTorch tensor.
        device : str, optional
            The PyTorch device to use for computation. If None, uses the current model's device.
        calculate_impact_score : bool, optional
            If True, calculates the IMPACT score. Default is False for better performance.

        Returns
        -------
        torch.Tensor
            The model's mortality probability predictions.

        Notes
        -----
        - The input is automatically converted to a PyTorch tensor if it isn't already.
        - For DataFrames, features are reordered to match the expected IMPACT model input order.
        - The output is returned on the same device as the input.
        - TODO: this function currently returns the probability. It should be renamed to predict_proba() to indicate this, and another function for making binary predictions should be made.
        """
        self.eval()

        # Determine the device to use
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        # Move the model to the specified device
        self.to(device)

        # Convert input to tensor if necessary
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif isinstance(x, pd.DataFrame):
            # Use our specialized function to handle feature ordering for IMPACT model
            x = dataframe_to_tensor(x, device='cpu')  # Create on CPU first, then move
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a numpy array, pandas DataFrame, or PyTorch tensor")

        # Ensure input is on the correct device and dtype
        x = x.to(device=device, dtype=torch.float32)

        # Perform prediction
        outputs = self.forward(x, calculate_impact_score=calculate_impact_score)

        return outputs['mortality_prob_logit']  # Return logit-based mortality probabilities

    @classmethod
    def get_feature_column_names(cls) -> list:
        """
        Get the feature column names in the order expected by dataframe_to_tensor.

        This method returns the feature names that should be present in a DataFrame
        when using dataframe_to_tensor for model input. Note that this excludes
        the baseline/reference categories that are handled automatically.

        Returns:
            list: Feature column names in the expected order for model input
        """
        return cls.FEATURE_NAMES.copy()


def dataframe_to_tensor(df: pd.DataFrame, device: str = 'cpu') -> torch.Tensor:
    """
    Convert pandas DataFrame to PyTorch tensor for model input.

    Args:
        df: Input DataFrame with IMPACT features
        device: Device to place tensor on ('cpu' or 'cuda')

    Returns:
        Tensor of shape (len(df), 18) with features in the expected order

    Raises:
        ValueError: If any required features are missing from the DataFrame
        ValueError: If diagnosis or ethnicity categories are not properly one-hot encoded
    """
    # Log reference categories information
    logging.info(
        "IMPACT Model One-Hot Reference Categories: diagn_Cardiomyopathy (baseline), recethcat_Caucasian (baseline)"
    )

    # Work with a copy to avoid modifying the original DataFrame
    df_work = df.copy()

    # Handle diagnosis categories
    all_diagn_cols = [
        'diagn_CAD',
        'diagn_Congenital',
        'diagn_Cardiomyopathy',
        'diagn_Valve',
        'diagn_Graftfailure',
        'diagn_Misc',
    ]
    present_diagn_cols = [col for col in all_diagn_cols if col in df_work.columns]

    if present_diagn_cols:
        diagnosis_sum = df_work[present_diagn_cols].sum(axis=1)

        if 'diagn_Cardiomyopathy' in df_work.columns:
            # Reference column present: sum must be exactly 1
            invalid_rows = df_work.index[diagnosis_sum != 1].tolist()
            if invalid_rows:
                raise ValueError(
                    f"Diagnosis categories must be one-hot encoded (exactly one category per row). "
                    f"Found rows with invalid encoding at indices: {invalid_rows[:5]}"
                    + ("..." if len(invalid_rows) > 5 else "")
                )
        else:
            # Reference column missing: sum must be 0 or 1 (0 = baseline, 1 = other category)
            invalid_rows = df_work.index[diagnosis_sum > 1].tolist()
            if invalid_rows:
                raise ValueError(
                    f"Diagnosis categories must be one-hot encoded (at most one category per row). "
                    f"Found rows with multiple diagnoses at indices: {invalid_rows[:5]}"
                    + ("..." if len(invalid_rows) > 5 else "")
                )
            # Create baseline category for rows with sum = 0
            df_work['diagn_Cardiomyopathy'] = 1 - diagnosis_sum
    else:
        # No diagnosis columns present, assume all are baseline
        df_work['diagn_Cardiomyopathy'] = 1

    # Handle ethnicity categories
    all_eth_cols = [
        'recethcat_African American',
        'recethcat_Hispanic',
        'recethcat_Other',
        'recethcat_Caucasian',
    ]
    present_eth_cols = [col for col in all_eth_cols if col in df_work.columns]

    if present_eth_cols:
        ethnicity_sum = df_work[present_eth_cols].sum(axis=1)

        if 'recethcat_Caucasian' in df_work.columns:
            # Reference column present: sum must be exactly 1
            invalid_rows = df_work.index[ethnicity_sum != 1].tolist()
            if invalid_rows:
                raise ValueError(
                    f"Ethnicity categories must be one-hot encoded (exactly one category per row). "
                    f"Found rows with invalid encoding at indices: {invalid_rows[:5]}"
                    + ("..." if len(invalid_rows) > 5 else "")
                )
        else:
            # Reference column missing: sum must be 0 or 1 (0 = baseline, 1 = other category)
            invalid_rows = df_work.index[ethnicity_sum > 1].tolist()
            if invalid_rows:
                raise ValueError(
                    f"Ethnicity categories must be one-hot encoded (at most one category per row). "
                    f"Found rows with multiple ethnicities at indices: {invalid_rows[:5]}"
                    + ("..." if len(invalid_rows) > 5 else "")
                )
            # Create baseline category for rows with sum = 0
            df_work['recethcat_Caucasian'] = 1 - ethnicity_sum
    else:
        # No ethnicity columns present, assume all are baseline
        df_work['recethcat_Caucasian'] = 1

    # Check for missing required features (excluding baseline categories)
    missing_features = [
        feature for feature in IMPACTModel.FEATURE_NAMES if feature not in df_work.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features in DataFrame: {missing_features}. "
            f"Required features are: {IMPACTModel.FEATURE_NAMES}"
        )

    # Get feature values in the correct order (excluding baseline categories)
    feature_matrix = []

    for feature_name in IMPACTModel.FEATURE_NAMES:
        feature_matrix.append(df_work[feature_name].values)

    # Stack features and convert to tensor
    feature_array = np.column_stack(feature_matrix)
    return torch.tensor(feature_array, dtype=torch.float32, device=device)


def apply_impact_model_torch(
    df: pd.DataFrame, device: str = 'cpu', calculate_impact_score: bool = False
) -> pd.DataFrame:
    """
    Apply the PyTorch IMPACT model to a DataFrame.

    Args:
        df: Input DataFrame
        device: Device to run model on ('cpu' or 'cuda')
        calculate_impact_score: If True, calculates the IMPACT score and points-based mortality.
                              Defaults to False for better performance.

    Returns:
        DataFrame with added prediction columns
    """
    # Initialize model
    model = IMPACTModel()
    model.to(device)
    model.eval()

    # Convert DataFrame to tensor
    x = dataframe_to_tensor(df, device)

    # Run model
    with torch.no_grad():
        output = model.forward(x, calculate_impact_score=calculate_impact_score)

    # Add results to DataFrame
    df_result = df.copy()
    df_result['pred_impact'] = (
        output['mortality_prob_logit'].cpu().numpy()
    )  # Primary prediction uses logit

    # Add impact score and points-based predictions if calculated
    if calculate_impact_score:
        df_result['impact_score'] = output['impact_score'].cpu().numpy()
        df_result['pred_impact_points'] = output['mortality_prob_points'].cpu().numpy()

    return df_result


# Example usage
if __name__ == "__main__":
    # Example: Create model and test with dummy data
    model = IMPACTModel()

    # First test the one-hot encoding validation
    print("\nTesting one-hot encoding validation...")

    # Create a small test DataFrame with valid one-hot encoding
    valid_df = pd.DataFrame(
        {
            'recageyear': [60, 70],
            'recbilirubin': [1.5, 2.5],
            'creatinine_clearance': [45, 25],
            'recsex': [1, 0],
            'recdialysis': [0, 1],
            'diagn_CAD': [1, 0],
            'diagn_Congenital': [0, 0],
            'diagn_Cardiomyopathy': [0, 1],  # Valid: one diagnosis per row
            'diagn_Valve': [0, 0],
            'diagn_Graftfailure': [0, 0],
            'diagn_Misc': [0, 0],
            'recinfections2weeks': [0, 1],
            'reciabp': [0, 0],
            'recventilator': [0, 1],
            'recethcat_African American': [1, 0],
            'recethcat_Hispanic': [0, 1],  # Valid: one ethnicity per row
            'recethcat_Other': [0, 0],
            'recethcat_Caucasian': [0, 0],  # Include reference category for this test
            'rececmo': [0, 0],
            'recvad': [0, 1],
        }
    )

    try:
        print("Testing valid DataFrame conversion...")
        tensor = dataframe_to_tensor(valid_df)
        print("✓ Valid DataFrame passed validation")
    except ValueError as e:
        print("✗ Valid DataFrame failed validation:", str(e))

    # Test invalid diagnosis encoding
    invalid_diagnosis_df = valid_df.copy()
    invalid_diagnosis_df.loc[0, 'diagn_CAD'] = 1
    invalid_diagnosis_df.loc[0, 'diagn_Congenital'] = 1  # Two diagnoses for one row

    try:
        print("\nTesting invalid diagnosis encoding...")
        tensor = dataframe_to_tensor(invalid_diagnosis_df)
        print("✗ Invalid diagnosis DataFrame passed validation when it should have failed")
    except ValueError as e:
        print("✓ Caught invalid diagnosis:", str(e))

    # Test invalid ethnicity encoding - multiple ethnicities for one row
    invalid_ethnicity_df = valid_df.copy()
    invalid_ethnicity_df.loc[0, 'recethcat_African American'] = 1
    invalid_ethnicity_df.loc[0, 'recethcat_Hispanic'] = 1  # Two ethnicities for one row

    try:
        print("\nTesting invalid ethnicity encoding...")
        tensor = dataframe_to_tensor(invalid_ethnicity_df)
        print("✗ Invalid ethnicity DataFrame passed validation when it should have failed")
    except ValueError as e:
        print("✓ Caught invalid ethnicity:", str(e))

    # Continue with the rest of the example code
    print("\nContinuing with standard tests...")

    # Create dummy tensor input (batch_size=1000, 18 features) - Fixed size
    batch_size = 1000
    x = torch.randn(batch_size, 18)  # Fixed: changed from 19 to 18

    # Set realistic ranges for features
    x[:, 0] = torch.randint(20, 80, (batch_size,), dtype=torch.float32)  # age
    x[:, 1] = torch.rand(batch_size) * 5  # bilirubin
    x[:, 2] = torch.rand(batch_size) * 100 + 20  # creatinine_clearance (20-120 mL/min)
    x[:, 3:] = torch.randint(
        0, 2, (batch_size, 15), dtype=torch.float32
    )  # binary features (15 remaining)

    # Test forward method in performance mode (default)
    with torch.no_grad():
        print("\nTesting performance mode (no impact score)...")
        output_perf = model.forward(x)
        print(
            f"Logit-based mortality probabilities: {output_perf['mortality_prob_logit'].min():.3f} - {output_perf['mortality_prob_logit'].max():.3f}"
        )
        print(f"Impact score present: {output_perf['impact_score'] is not None}")

    # Test forward method with impact score calculation
    with torch.no_grad():
        print("\nTesting with impact score calculation...")
        output_full = model.forward(x, calculate_impact_score=True)
        print(
            f"IMPACT scores range: {output_full['impact_score'].min():.1f} - {output_full['impact_score'].max():.1f}"
        )
        print(
            f"Points-based mortality probabilities: {output_full['mortality_prob_points'].min():.3f} - {output_full['mortality_prob_points'].max():.3f}"
        )
        print(
            f"Logit-based mortality probabilities: {output_full['mortality_prob_logit'].min():.3f} - {output_full['mortality_prob_logit'].max():.3f}"
        )

    # Test predict method in performance mode
    mortality_probs = model.predict(x)
    print("\nPredict method (performance mode) mortality probabilities:")
    print(f"Range: {mortality_probs.min():.3f} - {mortality_probs.max():.3f}")

    # Compare approaches when using full calculation mode
    output_full = model.forward(x, calculate_impact_score=True)
    print(
        f"\nMean difference (logit - points): {(output_full['mortality_prob_logit'] - output_full['mortality_prob_points']).mean():.4f}"
    )

    # Test CUDA if available
    if torch.cuda.is_available():
        print("\nTesting CUDA compatibility...")
        model_cuda = model.cuda()
        x_cuda = x.cuda()

        mortality_probs_cuda = model_cuda.predict(x_cuda)
        print("CUDA test passed!")

    # Test DataFrame to tensor conversion and both modes
    print("\nTesting DataFrame conversion...")
    # Create fresh model instance to avoid device mismatch
    model_fresh = IMPACTModel()  # Create a larger dummy dataset with proper one-hot encoding
    n_samples = 100
    dummy_df = pd.DataFrame(
        {
            'recageyear': np.random.randint(20, 80, n_samples),
            'recbilirubin': np.random.rand(n_samples) * 5,
            'creatinine_clearance': np.random.rand(n_samples) * 100 + 20,  # 20-120 mL/min
            'recsex': np.random.randint(0, 2, n_samples),
            'recdialysis': np.random.randint(0, 2, n_samples),
            'recinfections2weeks': np.random.randint(0, 2, n_samples),
            'reciabp': np.random.randint(0, 2, n_samples),
            'recventilator': np.random.randint(0, 2, n_samples),
            'rececmo': np.random.randint(0, 2, n_samples),
            'recvad': np.random.randint(0, 2, n_samples),
        }
    )

    # Create proper one-hot encoded diagnosis categories
    diagnosis_categories = [
        'diagn_CAD',
        'diagn_Congenital',
        'diagn_Cardiomyopathy',
        'diagn_Valve',
        'diagn_Graftfailure',
        'diagn_Misc',
    ]
    diagnosis_choice = np.random.choice(len(diagnosis_categories), size=n_samples)
    for i, category in enumerate(diagnosis_categories):
        dummy_df[category] = (diagnosis_choice == i).astype(int)

    # Create proper one-hot encoded ethnicity categories
    ethnicity_categories = ['recethcat_African American', 'recethcat_Hispanic', 'recethcat_Other']
    ethnicity_choice = np.random.choice(len(ethnicity_categories), size=n_samples)
    for i, category in enumerate(ethnicity_categories):
        dummy_df[category] = (ethnicity_choice == i).astype(int)

    # Test performance mode
    print("\nTesting performance mode with DataFrame...")
    result_df_perf = apply_impact_model_torch(dummy_df, calculate_impact_score=False)
    print(f"Columns in performance mode: {list(result_df_perf.columns)}")

    # Test full mode
    print("\nTesting full mode with DataFrame...")
    result_df_full = apply_impact_model_torch(dummy_df, calculate_impact_score=True)
    print(f"Added columns in full mode: {['impact_score', 'pred_impact_points', 'pred_impact']}")
    print("Sample predictions from full mode:")
    print(f"  Points-based: {result_df_full['pred_impact_points'].head(3).values}")
    print(f"  Logit-based:  {result_df_full['pred_impact'].head(3).values}")
    print(
        f"  Mean difference: {(result_df_full['pred_impact'] - result_df_full['pred_impact_points']).mean():.4f}"
    )
