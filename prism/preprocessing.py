import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import Optional, Tuple, Union

def normalize(data: pd.DataFrame, test: Optional[pd.DataFrame] = None, sd_scale: float = 2.0) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Normalize the dataset to have 0 median and a standard deviation scaled by `sd_scale`.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to normalize.
    test : pd.DataFrame, optional
        The test dataset to normalize using the same parameters as `data`.
    sd_scale : float, optional
        The scaling factor for the standard deviation of the data.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
        Normalized training dataset or tuple of normalized training and testing datasets.
    """
    med_train = data.median()
    sd_train = data.std()

    x_train = (data - med_train) / (sd_scale * sd_train)

    if test is not None:
        x_test = (test - med_train) / (sd_scale * sd_train)
        return x_train, x_test
    else:
        return x_train

# Statistical overview of features
def feature_summary(data, categorical_threshold=15):
    summary = pd.DataFrame({
        'Data Type': data.dtypes,
        'Non-Null Count': data.count(),
        'Null Count': data.isnull().sum(),
        'Mean': data.mean(),
        'Median': data.median(),
        'Std Dev': data.std(),
        'IQR': data.quantile(0.75) - data.quantile(0.25),
        'Min': data.min(),
        'Max': data.max(),
        'Unique Values': data.nunique()
    })
    summary['Is Categorical'] = summary['Unique Values'] < categorical_threshold
    return summary

def plot_feature_histograms(X, feature_stats, feature_names=None, figsize=(20, 20)):
    """
    Plot histograms for all features in X, distinguishing between categorical and continuous features.
    
    Parameters:
    X (pd.DataFrame): The dataset containing the features
    feature_stats (pd.DataFrame): DataFrame containing feature statistics including 'Is Categorical'
    feature_names (list): Optional list of feature names to use as labels
    figsize (tuple): Figure size (width, height) in inches
    """
    num_features = X.shape[1]
    num_cols = min(3, num_features)  # Maximum 3 columns
    num_rows = math.ceil(num_features / num_cols)
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
    
    if num_features == 1:
        axes = np.array([axes])  # Ensure axes is always a 2D array
    axes = axes.flatten()  # Flatten axes array for easy indexing

    # Use feature_names if provided, otherwise use column names
    if feature_names is None:
        feature_names = X.columns
    
    for idx, (column, is_categorical, feature_name) in enumerate(zip(X.columns, feature_stats['Is Categorical'], feature_names)):
        ax = axes[idx]
        if is_categorical:
            sns.countplot(x=column, data=X, ax=ax, color='lightgreen')
            feature_name = f'{feature_name} (Categorical)'
        else:
            sns.histplot(X[column], ax=ax, kde=True, color='skyblue')
            feature_name = f'{feature_name} (Continuous)'
        
        ax.set_xlabel(feature_name, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Rotate x-axis labels if they are too long
        if max([len(str(item.get_text())) for item in ax.get_xticklabels()]) > 6:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Remove any unused subplots
    for idx in range(num_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()
    return plt