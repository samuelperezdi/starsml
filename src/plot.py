import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.utils import prepare_feature_subset, preprocess, create_new_columns, load_model, transform_features
import shap
import lightgbm as lgb
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
import lightgbm as lgb
from collections import defaultdict

# Global variables
FEATURE_NAMES = [
    'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'bp_rp', 'bp_g', 'g_rp', 'parallax', 'hard_hs', 'hard_hm', 
    'hard_ms', 'var_intra_prob_b', 'var_intra_index_b', 'var_inter_prob_b', 'var_inter_index_b', 'var_inter_sigma_b', 
    'extent_flag', 'photflux_aper_b', 'yangetal_gcs_class', 'yangetal_training_class', 'perezdiazetal_class', 'parallax_error', 
    'sqrt(pmra^2+pmdec^2)', 'phot_g_mean_flux', 'phot_bp_mean_flux', 'phot_rp_mean_flux', 'radial_velocity', 'vbroad', 
    'phot_variable_flag', 'classprob_dsc_combmod_quasar', 'classprob_dsc_combmod_galaxy', 'classprob_dsc_combmod_star', 
    'distance_gspphot'
]

LOG_TRANSFORM_FEATURES = [
    'parallax', 'parallax_error', 'photflux_aper_b', 'phot_g_mean_flux', 'phot_bp_mean_flux', 'phot_rp_mean_flux', 'radial_velocity', 'vbroad'
]

def plot_experiment_results(results, results_info, df_pos, df_neg, experiment_name, offaxis_interval):
    """plot experiment results and save as pdf"""
    save_dir = os.path.join('figures', 'experiments', experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{offaxis_interval}_experiment_results.pdf"
    file_path = os.path.join(save_dir, file_name)
    
    # activate sns formatting
    sns.set_theme(style="whitegrid")
    
    with PdfPages(file_path) as pdf:
        for subset_type, result in results.items():
            best_model = result['best_model']
            model_type = results_info['model_type']
            p_ind_name = f"p_ind_{model_type.lower()}"
            
            create_new_columns(df_pos)
            create_new_columns(df_neg)
           
            df_pos_subset = prepare_feature_subset(df_pos, subset_type)
            df_neg_subset = prepare_feature_subset(df_neg, subset_type)
            
            X_train, X_test, Y_train, Y_test, _, indices_test, categorical_features, scaler = preprocess(
                df_pos_subset, df_neg_subset,
                log_transform=results_info['log_transform'],
                normalization_method=results_info['normalization_method'],
                model_type=model_type,
                feature_names=df_pos_subset.columns,
                random_seed=results_info['random_seed']
            )
            
            y_test_prob = best_model.predict_proba(X_test)[:, 1]
            y_pred = best_model.predict(X_test)
           
            df_test = pd.concat([df_pos, df_neg]).iloc[indices_test].reset_index(drop=True)
           
            fig = plt.figure(figsize=(20, 18))  # Reduced height
            gs = fig.add_gridspec(7, 3)  # Changed to 7 rows
           
            # title and info
            ax_title = fig.add_subplot(gs[0:1, :])  # Use only one row for title
            ax_title.axis('off')
            title_text = f"Experiment: {experiment_name} (Subset {subset_type})\n"
            title_text += f"Model: {model_type} | Hyperparameters: {results_info.get('best_params', 'N/A')}\n"
            title_text += f"Features: {', '.join(X_test.columns)}"
            ax_title.text(0.5, 0.5, title_text, ha='center', va='center', wrap=True, fontsize=10)
           
            # scatter plot
            ax1 = fig.add_subplot(gs[1:3, 0])  
            cmap = 'coolwarm_r'
            norm = plt.Normalize(df_test['p_any'].min(), df_test['p_any'].max())
            sc = ax1.scatter(y_test_prob, df_test['p_i'],
                             c=df_test['p_any'], cmap=cmap, norm=norm, s=6)
            cbar = plt.colorbar(sc, ax=ax1)
            cbar.set_label('p_any', rotation=270, labelpad=15)
            ax1.set_title(f'p_abs_nway vs {p_ind_name} - Test Set')
            ax1.set_ylabel('p_abs_nway')
            ax1.set_xlabel(p_ind_name)
           
            # histograms
            ax2 = fig.add_subplot(gs[1:3, 1])
            sns.histplot(y_test_prob[Y_test == 1], bins=20, kde=False, ax=ax2)
            ax2.set_title(f'{p_ind_name} for true positive matches - Test Set')
            ax2.set_xlabel(p_ind_name)
            ax2.set_ylabel('Frequency')
           
            ax3 = fig.add_subplot(gs[1:3, 2]) 
            sns.histplot(y_test_prob[Y_test == 0], bins=20, kde=False, ax=ax3)
            ax3.set_title(f'{p_ind_name} for true negative matches - Test Set')
            ax3.set_xlabel(p_ind_name)
            ax3.set_ylabel('Frequency')
           
            # performance metrics
            ax4 = fig.add_subplot(gs[3:5, 0]) 
            metrics = {
                f"{model_type} train accuracy": best_model.score(X_train, Y_train),
                "Test accuracy": accuracy_score(Y_test, y_pred),
                "Precision": precision_score(Y_test, y_pred),
                "Recall": recall_score(Y_test, y_pred),
                "F1 Score": f1_score(Y_test, y_pred),
                "AUC-ROC": roc_auc_score(Y_test, y_test_prob)
            }
            
            cell_text = [[f"{v:.3f}"] for v in metrics.values()]
            ax4.axis('tight')
            ax4.axis('off')
            table = ax4.table(cellText=cell_text, rowLabels=list(metrics.keys()),
                              colWidths=[0.5], loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            ax4.set_title("Performance Metrics", pad=20)
           
            # confusion matrix
            ax5 = fig.add_subplot(gs[3:5, 1:]) 
            cm = confusion_matrix(Y_test, y_pred, normalize='true')
            cm_percentage = (cm * 100).round(1)
            class_labels = ['bad', 'good']
            
            sns.heatmap(cm_percentage, annot=True, fmt='', cmap='Blues', cbar=False,
                        xticklabels=class_labels, yticklabels=class_labels,
                        annot_kws={'va':'center'}, ax=ax5)
            
            ax5.set_xlabel('Predicted')
            ax5.set_ylabel('True')
            ax5.set_title(f'Confusion Matrix for {model_type} - Test Set')
            
            for t in ax5.texts:
                t.set_text(t.get_text() + " %")
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Finished {subset_type}")

    print(f"Results saved to {file_path}")

# feature importance

def plot_shap_analysis(model_path, df_pos, df_neg, output_dir):
    # PLACEHOLDER, DOESN'T WORK
    """
    Perform SHAP analysis on a given model and create plots.
   
    Parameters:
    - model_path: str, path to the saved model file
    - df_pos: DataFrame, positive samples
    - df_neg: DataFrame, negative samples
    - output_dir: str, directory to save the output plots
    """
    # Load the model and its configuration
    model_info = load_model(model_path)
    model = model_info['model']
    config = model_info['experiment_info']
    print(config['model_type'])

    create_new_columns(df_pos)
    create_new_columns(df_neg)
    # feature subset
    df_pos_subset = prepare_feature_subset(df_pos, 8)
    df_neg_subset = prepare_feature_subset(df_neg, 8)
    # Preprocess data
    X_train, X_test, Y_train, Y_test, _, _, categorical_features, scaler = preprocess(
        df_pos_subset, df_neg_subset,
        log_transform=False,
        normalization_method='none',
        model_type=config['model_type'],
        feature_names=df_pos_subset.columns,
        random_seed=42
    )

    for c in categorical_features:
        X_train[c] = X_train[c].astype('category')
        X_test[c] = X_test[c].astype('category')
    model.set_params(categorical_feature=categorical_features)
    print(X_test.shape, X_test.columns)

    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
   
    # Summary plot
    shap.summary_plot(shap_values[1], X_test, plot_type="bar", feature_names=X_test.columns)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'shap_summary_plot.png'))
    plt.close()
   
    # Force plot for first prediction
    #shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:],
    #                feature_names=X_test.columns, matplotlib=True, show=False)
    #plt.tight_layout()
    #plt.savefig(os.path.join(output_dir, f'shap_force_plot.png'))
    #plt.close()
   
    # Dependence plot for most important feature
    most_important_feature = X_test.columns[np.argmax(np.mean(np.abs(shap_values), axis=0))]
    shap.dependence_plot(most_important_feature, shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'shap_dependence_plot.png'))
    plt.close()

    print(f"SHAP analysis completed. Plots saved in {output_dir}")

    # Return SHAP values and feature names for potential further analysis
    return shap_values, X_test.columns.tolist()

# def plot_lgbm_importance(model_path, df_pos, df_neg, output_dir):
#     """
#     Plot feature importance for a LightGBM model using lightgbm.plot_importance.
   
#     Parameters:
#     - model_path: str, path to the saved model file
#     - df_pos: DataFrame, positive samples
#     - df_neg: DataFrame, negative samples
#     - output_dir: str, directory to save the output plot
#     """
#     # Load the model and its configuration
#     model_info = load_model(model_path)
#     model = model_info['model']
#     config = model_info['experiment_info']

#     if config['model_type'] != 'lgbm':
#         raise ValueError("This function is only for LightGBM models.")

#     create_new_columns(df_pos)
#     create_new_columns(df_neg)
    
#     # feature subset (using subset 8 as in the original function)
#     df_pos_subset = prepare_feature_subset(df_pos, 8)
#     df_neg_subset = prepare_feature_subset(df_neg, 8)
    
#     # Preprocess data
#     X_train, X_test, Y_train, Y_test, _, _, categorical_features, _ = preprocess(
#         df_pos_subset, df_neg_subset,
#         log_transform=False,
#         normalization_method='none',
#         model_type=config['model_type'],
#         feature_names=df_pos_subset.columns,
#         random_seed=42
#     )

#     for c in categorical_features:
#         X_train[c] = X_train[c].astype('category')
#         X_test[c] = X_test[c].astype('category')

#     # Plot feature importance
#     fig, ax = plt.subplots(figsize=(10, 10))
#     lgb.plot_importance(model, ax=ax, max_num_features=30, importance_type='gain')
#     plt.title("LightGBM Feature Importance")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'lgbm_feature_importance_gain.png'))
#     plt.close()

#     print(f"LightGBM feature importance plot saved in {output_dir}")

#     # Return feature importance as a dictionary for potential further analysis
#     feature_importance = {name: importance for name, importance in zip(model.feature_name_, model.feature_importances_)}
#     return feature_importance

def feature_importance_analysis(model, X_train, X_test, y_train, y_test, output_dir):
    plt.rcParams['font.family'] = 'Nimbus Sans'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    os.makedirs(output_dir, exist_ok=True)
    print(model)

    model_info = load_model(model)
    model = model_info['model']
    config = model_info['experiment_info']

    if config['model_type'] != 'lgbm':
        raise ValueError("This function is only for LightGBM models.")
    
    # lgbm importances
    plot_lgbm_importance(model, X_train, 'gain', os.path.join(output_dir, 'lgbm_importance_gain.pdf'))
    plot_lgbm_importance(model, X_train, 'split', os.path.join(output_dir, 'lgbm_importance_split.pdf'))
    
    # permutation importance
    #plot_permutation_importance(model, X_train, y_train, X_test, y_test, output_dir)
    
    # feature correlation analysis
    plot_feature_correlation(X_train, output_dir)

def plot_lgbm_importance(model, X, importance_type, output_file):
    fig, ax = plt.subplots(figsize=(10, 12))
    lgb.plot_importance(model, ax=ax, max_num_features=30, importance_type=importance_type)
    ax.set_title(f"LightGBM Feature Importance ({importance_type.capitalize()})")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_permutation_importance(model, X_train, y_train, X_test, y_test, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    for ax, X, y, title in [(ax1, X_train, y_train, "Train"), (ax2, X_test, y_test, "Test")]:
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()
        
        ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
        ax.set_title(f"Permutation Importance ({title} Set)")
        ax.set_xlabel("Decrease in accuracy score")
        ax.axvline(x=0, color="k", linestyle="--")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'permutation_importance.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_correlation(X, output_dir):

    # select only numeric columns
    X_numeric = X.select_dtypes(include=[np.number])
    
    # handle NaN values by filling them with the mean of each column
    X_numeric = X_numeric.fillna(X_numeric.mean())
    
    if X_numeric.empty:
        print("No numeric features found. Skipping correlation plot.")
        return
    
    corr, _ = spearmanr(X_numeric)
    
    # ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # dendogram
    dendro = hierarchy.dendrogram(dist_linkage, labels=X_numeric.columns.to_list(), ax=ax1, leaf_rotation=90)
    ax1.set_title("Feature Clustering Dendrogram")
    
    # heatmap
    dendro_idx = np.arange(0, len(dendro["ivl"]))
    sns.heatmap(corr[dendro["leaves"], :][:, dendro["leaves"]], ax=ax2, cmap="coolwarm", center=0)
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation=90)
    ax2.set_yticklabels(dendro["ivl"], rotation=0)
    ax2.set_title("Feature Correlation Heatmap")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def lgbm_feature_cluster_and_importance(X_train, X_test, y_train, y_test, output_dir, threshold=0.5):

    plt.rcParams['font.family'] = 'Nimbus Sans'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    os.makedirs(output_dir, exist_ok=True)
    
    # select only numeric columns
    X_numeric = X_train.select_dtypes(include=[np.number])
    
    # handle NaN values by filling them with the mean of each column
    X_numeric = X_numeric.fillna(X_numeric.mean())
    
    if X_numeric.empty:
        print("No numeric features found. Skipping analysis.")
        return
    
    # compute correlation and linkage
    corr, _ = spearmanr(X_numeric)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    
    # select features based on clustering
    cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_numeric_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_numeric_features_names = X_numeric.columns[selected_numeric_features]

    # include non-numeric features
    non_numeric_features = X_train.select_dtypes(exclude=[np.number]).columns
    selected_features_names = list(selected_numeric_features_names) + list(non_numeric_features)

    # prepare selected feature datasets
    X_train_sel = X_train[selected_features_names]
    X_test_sel = X_test[selected_features_names]
    
    # train LightGBM model
    default_params = {
    'num_leaves': 127,
    'max_depth': 15,
    'learning_rate': 0.01,
    'n_estimators': 500,
    'min_child_samples': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8
    }
    model = lgb.LGBMClassifier(random_state=42)
    model.set_params(**default_params)
    model.fit(X_train_sel, y_train)
    
    # accuracy
    accuracy = model.score(X_test_sel, y_test)
    print(f"Accuracy on test data with selected features: {accuracy:.4f}")
    
    # permutation importance
    result = permutation_importance(model, X_test_sel, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    
    plt.figure(figsize=(10, 8))
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test_sel.columns[sorted_idx])
    plt.title("Permutation Importances on Selected Features (Test Set)")
    plt.xlabel("Decrease in Accuracy Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'permutation_importance_selected.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, selected_features_names

def plot_benchmark_results(model, benchmark_df, config, experiment_name):
    """plot benchmark results and save as PDF"""

    def preprocess_benchmark_df(df):
        df, _ = transform_features(df, model_type='lgbm', log_transform=False)

        return df
    save_dir = os.path.join('figures', 'benchmarks', experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"benchmark_results.pdf")
    
    plt.rcParams['font.family'] = 'Nimbus Sans'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    sns.set_theme(style="whitegrid")
    
    with PdfPages(file_path) as pdf:
        model_type = config['model_type']
        p_ind_name = f"p_ind_{model_type.lower()}"

        create_new_columns(benchmark_df)
        X_benchmark = benchmark_df[FEATURE_NAMES]
        y_benchmark = benchmark_df['benchmark_label']

        X_benchmark = preprocess_benchmark_df(X_benchmark)
        
        y_benchmark_prob = model.predict_proba(X_benchmark)[:, 1]
        y_benchmark_pred = model.predict(X_benchmark)
        
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(7, 3)
        
        # title
        ax_title = fig.add_subplot(gs[0:1, :])
        ax_title.axis('off')
        title_text = f"Benchmark: {experiment_name}\n"
        title_text += f"Model: {model_type} | Hyperparameters: {config.get('best_params', 'N/A')}\n"
        title_text += f"Features: {', '.join(X_benchmark.columns)}"
        ax_title.text(0.5, 0.5, title_text, ha='center', va='center', wrap=True, fontsize=10)
        
        # scatter
        ax1 = fig.add_subplot(gs[1:3, 0])
        cmap = 'coolwarm_r'
        norm = plt.Normalize(benchmark_df['p_any'].min(), benchmark_df['p_any'].max())
        sc = ax1.scatter(y_benchmark_prob, benchmark_df['p_i'],
                         c=benchmark_df['p_any'], cmap=cmap, norm=norm, s=6, rasterized=True)
        cbar = plt.colorbar(sc, ax=ax1)
        cbar.set_label('p_any', rotation=270, labelpad=15)
        ax1.set_title(f'p_abs_nway vs {p_ind_name} - Benchmark Set')
        ax1.set_ylabel('p_abs_nway')
        ax1.set_xlabel(p_ind_name)
        
        # histograms
        ax2 = fig.add_subplot(gs[1:3, 1])
        sns.histplot(y_benchmark_prob[y_benchmark == 1], bins=20, kde=False, ax=ax2)
        ax2.set_title(f'{p_ind_name} for true positive matches - Benchmark Set')
        ax2.set_xlabel(p_ind_name)
        ax2.set_ylabel('Frequency')
        
        ax3 = fig.add_subplot(gs[1:3, 2])
        sns.histplot(y_benchmark_prob[y_benchmark == 0], bins=20, kde=False, ax=ax3)
        ax3.set_title(f'{p_ind_name} for true negative matches - Benchmark Set')
        ax3.set_xlabel(p_ind_name)
        ax3.set_ylabel('Frequency')
        
        # performance metrics
        ax4 = fig.add_subplot(gs[3:5, 0])
        metrics = {
            "Benchmark accuracy": accuracy_score(y_benchmark, y_benchmark_pred),
            "Precision": precision_score(y_benchmark, y_benchmark_pred),
            "Recall": recall_score(y_benchmark, y_benchmark_pred),
            "F1 Score": f1_score(y_benchmark, y_benchmark_pred),
            "AUC-ROC": roc_auc_score(y_benchmark, y_benchmark_prob)
        }
        
        cell_text = [[f"{v:.3f}"] for v in metrics.values()]
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=cell_text, rowLabels=list(metrics.keys()),
                          colWidths=[0.5], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax4.set_title("Performance Metrics", pad=20)
        
        # confusion
        ax5 = fig.add_subplot(gs[3:5, 1:])
        cm = confusion_matrix(y_benchmark, y_benchmark_pred, normalize='true')
        cm_percentage = (cm * 100).round(1)
        class_labels = ['bad', 'good']
        
        sns.heatmap(cm_percentage, annot=True, fmt='', cmap='Blues', cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels,
                    annot_kws={'va':'center'}, ax=ax5)
        
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('True')
        ax5.set_title(f'Confusion Matrix for {model_type} - Benchmark Set')
        
        for t in ax5.texts:
            t.set_text(t.get_text() + " %")
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"Benchmark results saved to {file_path}")