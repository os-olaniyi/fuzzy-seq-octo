import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from io import StringIO
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import shap
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression

from utils import (
    setup_plot_directory, 
    save_plot,
    save_plot_with_timestamp,
    save_analysis_to_file
)

class FeatureCorrelationAnalyzer:
    def __init__(self, data_path):
        """
        Initialize the analyzer with data path and key features.
        
        Args:
            data_path (str or Path): Path to the CSV data file
        """
        self.data_path = Path(data_path)
        self.plot_dir = setup_plot_directory()
        self.key_features = [
            "Procada Current", "Procada Voltage", "Laser Power", 
            "Wire Current", "Wire Speed", "Wire Voltage", 
            "X", "Y", "Z"
        ]
        self.target_features = ["Procada Current", "Procada Voltage", "Laser Power"]
        
        # Read the data
        self.df = pd.read_csv(self.data_path)
        
        # # Setup output capturing
        # self.output_buffer = StringIO()
        # self.original_stdout = sys.stdout
        # sys.stdout = self.output_buffer

        # Suppress specific warning from SHAP
        warnings.filterwarnings(
            'ignore', 
            message='This figure includes Axes that are not compatible with tight_layout',
            category=UserWarning
        )


    def analyze_basic_statistics(self):
        """
        Calculate and return basic statistics of the dataset.
        
        Returns:
            stats (pd.DataFrame): Basic statistics of the dataset
            null_counts (pd.Series): Null counts of the dataset
        """
        stats = self.df.describe()
        null_counts = self.df.isnull().sum()
        return stats, null_counts
    
    def create_correlation_matrix(self):
        """Create and save correlation matrix plot."""
        plt.figure(figsize = (12, 10))
        correlation_matrix = self.df.select_dtypes(include = [np.number]).corr()
        sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm", center = 0)
        plt.title("Correlation Matrix of Features")
        plt.tight_layout()
        plt.show()
        save_plot(plt,
                  "correlation_matrix.png",
                  self.plot_dir
                  )
        plt.close()
        return correlation_matrix
    
    def create_pairwise_plots(self):
        """Create and save pairwise scatter plots."""
        sns.pairplot(self.df[self.key_features], diag_kind = "kde")
        plt.show()
        save_plot(plt, "pairwise_plots.png", self.plot_dir)
        plt.close()

    def analyze_time_series(self):
        """Create and save time series analysis plots."""
        plt.figure(figsize = (15, 10))
        for i, feature in enumerate(self.key_features, 1):
            plt.subplot(3, 3, i)
            plt.plot(self.df['Datetime'], self.df[feature])
            plt.title(f'{feature} over Time')
            plt.xticks(rotation = 45)
        plt.tight_layout()
        plt.show()
        save_plot(plt, "time_series.png", self.plot_dir)
        plt.close()

    def analyze_distributions(self):
        """Create and save feature distribution plots."""
        plt.figure(figsize = (15, 10))
        for i, feature in enumerate(self.key_features, 1):
            plt.subplot(3, 3, i)
            sns.histplot(self.df[feature], kde = True)
            plt.title(f'{feature} Distribution')
        plt.tight_layout()
        plt.show()
        save_plot(plt, "distributions.png", self.plot_dir)
        plt.close()

    def analyze_key_correlations(self, threshold=0.5):
        """Print key correlations above threshold."""
        print("\nKey Correlations:")
        for feature1 in self.key_features:
            for feature2 in self.key_features:
                if feature1 < feature2:  # To avoid printing duplicates
                    correlation = self.df[feature1].corr(self.df[feature2])
                    if abs(correlation) > threshold:
                        print(f"{feature1} vs {feature2}: {correlation:.3f}")
    
    def analyze_feature_statistics(self):
        """Print basic statistics for each feature."""
        print("\nFeature Statistics (in their respective units):")
        for feature in self.key_features:
            stats_dict = {
                'Mean': self.df[feature].mean(),
                'Std': self.df[feature].std(),
                'Min': self.df[feature].min(),
                'Max': self.df[feature].max(),
                'Median': self.df[feature].median()
            }
            print(f"\n{feature}:")
            for stat, value in stats_dict.items():
                print(f"{stat}: {value:.3f}")

    def analyze_feature_importance(self):
        """Analyze and plot feature importance for each target."""
        plt.figure(figsize = (15, 15))
        for idx, target in enumerate(self.target_features, 1):
            y = self.df[target]
            X = self.df[self.key_features].drop(target, axis = 1)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size = 0.2, random_state = 42
            )
            
            rf_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
            rf_model.fit(X_train, y_train)
            
            importance = rf_model.feature_importances_
            features = X_train.columns
            
            plt.subplot(len(self.target_features), 1, idx)
            importance_df = pd.DataFrame({'features': features, 'importance': importance})
            importance_df = importance_df.sort_values('importance', ascending = True)
            
            plt.barh(range(len(importance)), importance_df['importance'])
            plt.yticks(range(len(importance)), importance_df['features'])
            plt.xlabel("Importance")
            plt.title(f"Feature Importance for {target}")
            
            y_pred = rf_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"\nModel Performance for {target}:")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")

        plt.tight_layout()
        plt.show()
        save_plot_with_timestamp(plt, "feature_importance.png", self.plot_dir)
        plt.close()

    def comprehensive_feature_importance(self, X, y, model = None):
        """Perform comprehensive feature importance analysis."""
        if model is None:
            model = RandomForestRegressor(n_estimators = 100, random_state = 42)
        
        # Statistical Correlation
        corr = pd.DataFrame({
            'pearson': X.corrwith(y),
            'spearman': X.corrwith(y, method = 'spearman')
        })
        
        # Mutual Information
        mi = mutual_info_regression(X, y)
        
        # Model-based importance
        model.fit(X, y)
        model_importance = model.feature_importances_
        
        # Permutation Importance
        perm_importance = permutation_importance(model, X, y, n_repeats = 10)
        
        # SHAP Values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Combine results
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Pearson_Correlation': corr['pearson'],
            'Spearman_Correlation': corr['spearman'],
            'Mutual_Information': mi,
            'Model_Importance': model_importance,
            'Permutation_Importance': perm_importance.importances_mean
        })
        
        return importance_df, shap_values
    
    def analyze_comprehensive_importance(self):
        """Analyze and plot comprehensive feature importance."""
        for target in self.target_features:
            X = self.df[self.key_features].drop(target, axis=1)
            y = self.df[target]
            
            importance_df, shap_values = self.comprehensive_feature_importance(X, y)
            
            print(f"\nFeature Importance Analysis for {target}:")
            print("\nImportance Metrics:")
            print(importance_df.round(4))
            
            self._plot_comprehensive_importance(importance_df, shap_values, X, target)
            
            # Print detailed analysis
            print("\nFeature Ranking Summary:")
            rankings = importance_df.rank(ascending=False)
            mean_rank = rankings.mean(axis=1).sort_values()
            
            for feature in mean_rank.index:
                print(f"\n{feature}:")
                print(f"  Average Rank: {mean_rank[feature]:.2f}")
                print(f"  Pearson Correlation: {importance_df.loc[importance_df['Feature'] == feature, 'Pearson_Correlation'].values[0]:.4f}")
                print(f"  Model Importance: {importance_df.loc[importance_df['Feature'] == feature, 'Model_Importance'].values[0]:.4f}")

    
    def _plot_comprehensive_importance(self, importance_df, shap_values, X, target):
        """Helper method to create comprehensive importance plots."""
        fig = plt.figure(figsize=(15, 16))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=1.0)
        
        # Bar plot
        ax1 = fig.add_subplot(gs[0])
        metrics_plot = importance_df[['Pearson_Correlation', 'Spearman_Correlation', 
                                    'Model_Importance', 'Permutation_Importance']]
        
        colors = ['#2196F3', '#FFA726', '#66BB6A', '#EF5350']
        bar_width = 0.18
        x = np.arange(len(importance_df))
        
        for idx, (column, color) in enumerate(zip(metrics_plot.columns, colors)):
            ax1.bar(x + idx * bar_width, 
                   metrics_plot[column], 
                   bar_width, 
                   label=column.replace('_', ' '),
                   color=color)
        
        ax1.set_title(f'Feature Importance Metrics for {target}', 
                     pad=20, fontsize=14, fontweight="bold")
        ax1.set_xticks(x + bar_width * 1.5)
        ax1.set_xticklabels(importance_df['Feature'], rotation=30, ha="right", fontsize=10)
        ax1.grid(True, axis="y", linestyle='--', alpha=0.3)
        ax1.set_ylabel("Importance Score", fontsize=12)
        ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
        
        # SHAP plot
        ax2 = fig.add_subplot(gs[1])
        shap.summary_plot(
            shap_values, 
            X,
            plot_type="bar",
            show=False,
            max_display=10,
            plot_size=(12, 6)
        )
        
        ax2.set_title(f'SHAP Feature Importance for {target}', 
                     pad=40,
                     y=1.1,
                     fontsize=14, 
                     fontweight="bold")
        
        plt.subplots_adjust(
            top=0.95,     
            bottom=0.1,   
            hspace=1.0,  
            right=0.85,  
            left=0.1    
        )
        plt.show()
        save_plot(
            plt, 
            f'comprehensive_importance_{target.replace(" ", "_")}.png', 
            self.plot_dir, 
            dpi=300, 
            bbox_inches="tight"
        )
        plt.close()

    def save_analysis_results(self):
        """Save analysis results to file."""
        sys.stdout = self.original_stdout
        analysis_content = self.output_buffer.getvalue()
        save_analysis_to_file(
            analysis_content, 
            'feature_importance_analysis.txt', 
            'analysis_outputs'
        )
        self.output_buffer.close()

    