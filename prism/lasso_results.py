import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from itertools import combinations

class LassoResultsManager:
    def __init__(self, lambdas: np.ndarray, betas: np.ndarray, models: List[LogisticRegression], feature_names: List[str], train_losses: np.ndarray, test_losses: np.ndarray, train_aucs: np.ndarray, test_aucs: np.ndarray, train_devs: np.ndarray, test_devs: np.ndarray):
        self.lambdas = lambdas
        self.betas = betas
        self.models = models
        self.univariate_feature_names = feature_names
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.train_aucs = train_aucs
        self.test_aucs = test_aucs
        self.train_devs = train_devs
        self.test_devs = test_devs
        self.selected_lambda_index = None
        
        self.n_univ = len(feature_names)
        self.bivariate_inputs = list(combinations(range(self.n_univ), 2))
        self.n_biv = len(self.bivariate_inputs)
        self.num_features = self.n_univ + self.n_biv
        
        self._generate_all_feature_names()

    def _generate_all_feature_names(self):
        self.all_feature_names = self.univariate_feature_names.copy()
        for i, (f1, f2) in enumerate(self.bivariate_inputs):
            self.all_feature_names.append(f"{self.univariate_feature_names[f1]} : {self.univariate_feature_names[f2]}")

    def select_lambda(self, lambda_index: int):
        if lambda_index < 0 or lambda_index >= len(self.lambdas):
            raise ValueError("Invalid lambda index")
        self.selected_lambda_index = lambda_index

    def get_selected_beta(self) -> np.ndarray:
        if self.selected_lambda_index is None:
            raise ValueError("No lambda selected")
        return self.betas[:, self.selected_lambda_index]

    def get_selected_model(self) -> LogisticRegression:
        if self.selected_lambda_index is None:
            raise ValueError("No lambda selected")
        else:
            print(f"Logistic regression model for Lambda index {self.selected_lambda_index} ({self.lambdas[self.selected_lambda_index]:.4g}) selected.")
        return self.models[self.selected_lambda_index]

    def get_selected_feature_indicies(self, threshold: float = 0.1) -> List[int]:
        beta = self.get_selected_beta()
        return np.where(np.abs(beta) > threshold)[0]

    def get_selected_feature_names(self, threshold: float = 0.1) -> List[str]:
        return [self.all_feature_names[i] for i in self.get_selected_feature_indicies(threshold=threshold)]

    def get_selected_univariate_indices(self, threshold: float = 0.1) -> List[int]:
        beta = self.get_selected_beta()
        return [i for i in range(self.n_univ) if abs(beta[i]) > threshold]

    def get_selected_bivariate_indices(self, threshold: float = 0.1) -> List[int]:
        beta = self.get_selected_beta()
        return [i - self.n_univ for i in range(self.n_univ, len(beta)) if abs(beta[i]) > threshold]

    def get_selected_bivariate_index_pairs(self, threshold: float = 0.1) -> List[Tuple[int, int]]:
        return [self.bivariate_inputs[i] for i in self.get_selected_bivariate_indices()]

    def is_mixed_bivariate(self, feature1: int, feature2: int, x: np.ndarray, categorical_threshold: int) -> bool:
        is_categorical1 = len(np.unique(x[:, feature1])) < categorical_threshold
        is_categorical2 = len(np.unique(x[:, feature2])) < categorical_threshold
        return is_categorical1 != is_categorical2

    def plot_lambda_path(self):
        plt.figure(figsize=(12, 6))
        for i, name in enumerate(self.all_feature_names):
            plt.semilogx(self.lambdas, self.betas[i, :], label=name)
        plt.xlabel('Lambda')
        plt.ylabel('Coefficient value')
        plt.title('LASSO Path')
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        beta = self.get_selected_beta()
        feature_importance = list(zip(self.all_feature_names, np.abs(beta)))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        features, importance = zip(*feature_importance)

        plt.figure(figsize=(12, 6))
        plt.bar(features, importance)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Absolute Coefficient Value')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()

    def plot_lasso_loss_path(self):
        plt.figure(figsize=(12, 6))
        plt.semilogx(self.lambdas, self.train_losses, label='Train Loss')
        plt.semilogx(self.lambdas, self.test_losses, label='Test Loss')
        plt.xlabel('Lambda')
        plt.ylabel('Log Loss')
        plt.title('LASSO Path')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'lambdas': self.lambdas,
            'betas': self.betas,
            'univariate_feature_names': self.univariate_feature_names,
            'bivariate_inputs': self.bivariate_inputs,
            'selected_lambda_index': self.selected_lambda_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LassoResultsManager':
        manager = cls(
            data['lambdas'],
            data['betas'],
            [],  # models are not serialized
            data['univariate_feature_names'],
            data['bivariate_inputs']
        )
        manager.selected_lambda_index = data['selected_lambda_index']
        return manager

    def get_mask(self, threshold: float = 0.1, subnet_nodes: int = 5, 
                 bivariate_only_if_univariate: bool = False, 
                 include_bivariate_as_univariate: bool = True, 
                 verbose: bool = True) -> Tuple[np.ndarray, int]:
        """
        Generate a mask for selected features based on the selected beta.

        Parameters:
        -----------
        threshold : float, optional
            Threshold for considering a feature as selected (default is 0.1)
        subnet_nodes : int, optional
            Number of subnet nodes for each feature (default is 5)
        bivariate_only_if_univariate : bool, optional
            If True, include bivariate features only if both univariate features are selected (default is False)
        include_bivariate_as_univariate : bool, optional
            If True, include univariate features of selected bivariate features (default is True)
        verbose : bool, optional
            If True, print selected feature names and show heatmap of the mask (default is True)

        Returns:
        --------
        Tuple[np.ndarray, int]
            Mask array for selected features and total number of selected features
        """
        beta = self.get_selected_beta()
        selected_indices = np.where(np.abs(beta) > threshold)[0]
        
        univ_selected = [idx for idx in selected_indices if idx < self.n_univ]
        pr_names = [self.univariate_feature_names[idx] for idx in univ_selected]

        biv_selected_pairs = []
        for idx in selected_indices:
            if idx >= self.n_univ:
                first, second = self.bivariate_inputs[idx - self.n_univ]
                if bivariate_only_if_univariate:
                    if first in univ_selected and second in univ_selected:
                        biv_selected_pairs.append((first, second))
                        pr_names.append(f"{self.univariate_feature_names[first]} : {self.univariate_feature_names[second]}")
                elif include_bivariate_as_univariate:
                    univ_selected.extend(feature for feature in [first, second] if feature not in univ_selected)
                    biv_selected_pairs.append((first, second))
                    pr_names.append(f"{self.univariate_feature_names[first]} : {self.univariate_feature_names[second]}")
                else:
                    biv_selected_pairs.append((first, second))
                    pr_names.append(f"{self.univariate_feature_names[first]} : {self.univariate_feature_names[second]}")

        univ_selected = sorted(set(univ_selected))
        n_univ = len(univ_selected)
        n_biv = len(biv_selected_pairs)
        mask = np.zeros((self.n_univ, subnet_nodes * (n_univ + n_biv)))

        for i, idx in enumerate(univ_selected):
            mask[idx, i * subnet_nodes:(i + 1) * subnet_nodes] = 1

        biv_start = n_univ * subnet_nodes
        for i, (first, second) in enumerate(biv_selected_pairs):
            start_col = biv_start + i * subnet_nodes
            end_col = start_col + subnet_nodes
            mask[first, start_col:end_col] = 1
            mask[second, start_col:end_col] = 1

        if verbose:
            print("Selected features:", pr_names)
            fig, ax = plt.subplots(figsize=(6, 4))
            heatmap = sns.heatmap(mask, ax=ax)
            heatmap.set_xlabel('subnet node index')
            heatmap.set_ylabel('input features')
            heatmap.set_title('input mask')
            ax.set_yticklabels(self.univariate_feature_names, rotation=0)
            plt.show()

        return mask, n_univ + n_biv