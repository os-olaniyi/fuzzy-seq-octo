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

# Suppress specific warning from SHAP
warnings.filterwarnings(
    'ignore', 
    message = 'This figure includes Axes that are not compatible with tight_layout',
    category = UserWarning
)

output_buffer = StringIO()
original_stdout = sys.stdout
sys.stdout = output_buffer

plot_dir = setup_plot_directory()

"""Read the data"""
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent  # Go up one level from ANALYTICS to CODES
csv_path = project_root /"DATA-CLEAN"/"PROCESSED"/"cleaned_proc_data_v1.csv"

df = pd.read_csv(csv_path)

""" Basic Statistics """
print(df.describe())

print(df.isnull().sum())

"""  Correlation Matrix """
plt.figure(figsize = (12, 10))
correlation_matrix = df.select_dtypes(include = [np.number]).corr()
sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm", center = 0)
plt.title("Correlation Matrix of Features")
plt.tight_layout()
save_plot(plt, "correlation_matrix.png", plot_dir)
plt.close()

""" Pairwise Scatter Plots for Key Features """
key_features = ["Procada Current", "Procada Voltage", "Laser Power", 
                "Wire Current", "Wire Speed", "Wire Voltage", 
                "X", "Y", "Z"]

sns.pairplot(df[key_features], diag_kind = 'kde')
save_plot(plt, "pairwise_plots.png", plot_dir)
plt.close()

""" Time Series Analysis """
plt.figure(figsize=(15, 10))
for i, feature in enumerate(key_features, 1):
    plt.subplot(3, 3, i)
    plt.plot(df['Datetime'], df[feature])
    plt.title(f'{feature} over Time')
    plt.xticks(rotation = 45)
plt.tight_layout()
save_plot(plt, "time_series.png", plot_dir)
plt.close()

""" Feature Distribution Analysis """
plt.figure(figsize = (15, 10))
for i, feature in enumerate(key_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[feature], kde = True)
    plt.title(f'{feature} Distribution')
plt.tight_layout()
save_plot(plt, "distributions.png", plot_dir)
plt.close()


""" Calculate and print key correlations """
print("\nKey Correlations:")
for feature1 in key_features:
    for feature2 in key_features:
        if feature1 < feature2:  # To avoid printing duplicates
            correlation = df[feature1].corr(df[feature2])
            if abs(correlation) > 0.5:  # Only show strong correlations
                print(f"{feature1} vs {feature2}: {correlation:.3f}")


""" Basic statistics for each feature """
print("\nFeature Statistics (in their respective units):")
for feature in key_features:
    stats_dict = {
        'Mean': df[feature].mean(),
        'Std': df[feature].std(),
        'Min': df[feature].min(),
        'Max': df[feature].max(),
        'Median': df[feature].median()
    }
    print(f"\n{feature}:")
    for stat, value in stats_dict.items():
        print(f"{stat}: {value:.3f}")


"""Feature Importance Analysis"""

# Prepare data for feature importance analysis
X = df[key_features]
target_features = ["Procada Current", "Procada Voltage", "Laser Power"]   # Multiple targets to analyze

# Create feature importance plots for each target
plt.figure(figsize = (15, 15))
for idx, target in enumerate(target_features, 1):
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(target, axis = 1), y, test_size = 0.2, random_state = 42
    )
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    rf_model.fit(X_train, y_train)
    
    # Calculate feature importance
    importance = rf_model.feature_importances_
    features = X_train.columns
    
    # Plot feature importance
    plt.subplot(len(target_features), 1, idx)
    importance_df = pd.DataFrame({'features': features, 'importance': importance})
    importance_df = importance_df.sort_values('importance', ascending = True)
    
    plt.barh(range(len(importance)), importance_df['importance'])
    plt.yticks(range(len(importance)), importance_df['features'])
    plt.xlabel("Importance")
    plt.title(f"Feature Importance for {target}")
    
    # Print model performance
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nModel Performance for {target}:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")

plt.tight_layout()
save_plot_with_timestamp(plt, "feature_importance.png", plot_dir)
plt.close()


""" Comprehensive Feature Importance Analysis """

def comprehensive_feature_importance(X, y, model=None):
    if model is None:
        # Explicitly use RandomForestRegressor as default
        model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    
    # 1. Statistical Correlation
    corr = pd.DataFrame({'pearson': X.corrwith(y),
                        'spearman': X.corrwith(y, method='spearman')})
    
    # 2. Mutual Information
    mi = mutual_info_regression(X, y)
    
    # 3. Model-based importance (Random Forest Regressor)
    model.fit(X, y)
    model_importance = model.feature_importances_
    
    # 4. Permutation Importance
    perm_importance = permutation_importance(model, X, y, n_repeats = 10)
    
    # 5. SHAP Values
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


""" Analyze feature importance for each target feature """
target_features = ["Procada Current", "Procada Voltage", "Laser Power"]

for target in target_features:
    # Prepare X and y
    X = df[key_features].drop(target, axis = 1)  # Remove target from features
    y = df[target]
    
    # Run comprehensive analysis
    importance_df, shap_values = comprehensive_feature_importance(X, y)
    
    print(f"\nFeature Importance Analysis for {target}:")
    print("\nImportance Metrics:")
    print(importance_df.round(4))
    
    # Create figure with ample space
    fig = plt.figure(figsize = (15, 16))
    
    # Create subplot layout with explicit spacing
    gs = fig.add_gridspec(
        2, 1, 
        height_ratios = [1, 1.2], 
        hspace = 1.0 
    )
    
    # 1. Bar plot of combined importance metrics
    ax1 = fig.add_subplot(gs[0])
    
    # Create a more organized bar plot
    metrics_plot = importance_df[['Pearson_Correlation', 'Spearman_Correlation', 
                                'Model_Importance', 'Permutation_Importance']]
    
    # Set consistent colors and style
    colors = ['#2196F3', '#FFA726', '#66BB6A', '#EF5350']
    bar_width = 0.18
    x = np.arange(len(importance_df))
    
    # Plot each metric as grouped bars
    for idx, (column, color) in enumerate(zip(metrics_plot.columns, colors)):
        ax1.bar(x + idx * bar_width, 
                metrics_plot[column], 
                bar_width, 
                label = column.replace('_', ' '),
                color = color)
    
    # Customize the first subplot
    ax1.set_title(f'Feature Importance Metrics for {target}', 
                  pad = 20, fontsize = 14, fontweight = "bold")
    ax1.set_xticks(x + bar_width * 1.5)
    ax1.set_xticklabels(importance_df['Feature'], rotation = 30, ha = "right", fontsize = 10)
    ax1.grid(True, axis = "y", linestyle = '--', alpha = 0.3)
    ax1.set_ylabel("Importance Score", fontsize = 12)
    ax1.legend(bbox_to_anchor = (1.02, 1), loc = "upper left", fontsize = 10)
    
    # 2. SHAP Summary Plot
    ax2 = fig.add_subplot(gs[1])
    

    shap.summary_plot(
        shap_values, 
        X,
        plot_type = "bar",
        show = False,
        max_display = 10,
        plot_size = (12, 6)
    )
    
    ax2.set_title(f'SHAP Feature Importance for {target}', 
                  pad = 40,
                  y = 1.1,    # Move title much higher
                  fontsize = 14, 
                  fontweight = "bold")
    
    # Manual spacing adjustment - no tight_layout
    plt.subplots_adjust(
        top = 0.95,     
        bottom = 0.1,   
        hspace = 1.0,  
        right = 0.85,  
        left = 0.1    
    )
    
    # Save the plot
    save_plot(
        plt, 
        f'comprehensive_importance_{target.replace(" ", "_")}.png', 
        plot_dir, 
        dpi = 300, 
        bbox_inches = "tight"
    )
    plt.close()
    
    # Print detailed analysis
    print("\nFeature Ranking Summary:")
    rankings = importance_df.rank(ascending = False)
    mean_rank = rankings.mean(axis = 1).sort_values()
    
    for feature in mean_rank.index:
        print(f"\n{feature}:")
        print(f"  Average Rank: {mean_rank[feature]:.2f}")
        print(f"  Pearson Correlation: {importance_df.loc[importance_df['Feature'] == feature, 'Pearson_Correlation'].values[0]:.4f}")
        print(f"  Model Importance: {importance_df.loc[importance_df['Feature'] == feature, 'Model_Importance'].values[0]:.4f}")

# Save detailed analysis to file
sys.stdout = original_stdout
analysis_content = output_buffer.getvalue()
save_analysis_to_file(analysis_content, f'feature_importance_analysis_{target.replace(" ", "_")}.txt', 'analysis_outputs')

# Restore original stdout
sys.stdout = original_stdout
output_buffer.close()