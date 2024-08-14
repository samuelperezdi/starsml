import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.utils import prepare_feature_subset, preprocess

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

def plot_experiment_results(results, df_pos, df_neg, experiment_name, offaxis_interval):
    """plot experiment results and save as pdf"""
    save_dir = os.path.join('figures', 'experiments', experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{offaxis_interval}_experiment_results.pdf"
    file_path = os.path.join(save_dir, file_name)
    
    with PdfPages(file_path) as pdf:
        for subset_type, result in results.items():
            best_model = result['best_model']
           
            df_pos_subset = prepare_feature_subset(df_pos, subset_type)
            df_neg_subset = prepare_feature_subset(df_neg, subset_type)
           
            X_train, X_test, Y_train, Y_test, _, indices_test = preprocess(
                df_pos_subset, df_neg_subset,
                log_transform=False,
                random_seed=42,
                feature_names=df_pos_subset.columns
            )
           
            y_test_rf_prob = best_model.predict_proba(X_test)[:, 1]
            y_pred_rf = best_model.predict(X_test)
           
            df_test = pd.concat([df_pos, df_neg]).iloc[indices_test].reset_index(drop=True)
           
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(2, 3)
           
            # scatter plot
            ax1 = fig.add_subplot(gs[0, 0])
            cmap = 'coolwarm_r'
            norm = plt.Normalize(df_test['p_any'].min(), df_test['p_any'].max())
            sc = ax1.scatter(y_test_rf_prob, df_test['p_i'],
                             c=df_test['p_any'], cmap=cmap, norm=norm, s=6)
            cbar = plt.colorbar(sc, ax=ax1)
            cbar.set_label('p_any', rotation=270, labelpad=15)
            ax1.set_title(f'$p_i^{{NWAY}}$ vs $p_i^{{RF}}$\n(Subset {subset_type}) - Test Set')
            ax1.set_ylabel('$p_i^{NWAY}$')
            ax1.set_xlabel('$p_i^{RF}$')
           
            # histograms
            ax2 = fig.add_subplot(gs[0, 1])
            sns.histplot(y_test_rf_prob[Y_test == 1], bins=20, kde=False, ax=ax2)
            ax2.set_title(f'$p_i^{{RF}}$ for true positive matches\n(Subset {subset_type}) - Test Set')
            ax2.set_xlabel('$p_i^{RF}$')
            ax2.set_ylabel('Frequency')
           
            ax3 = fig.add_subplot(gs[0, 2])
            sns.histplot(y_test_rf_prob[Y_test == 0], bins=20, kde=False, ax=ax3)
            ax3.set_title(f'$p_i^{{RF}}$ for true negative matches\n(Subset {subset_type}) - Test Set')
            ax3.set_xlabel('$p_i^{RF}$')
            ax3.set_ylabel('Frequency')
           
            # performance metrics
            ax4 = fig.add_subplot(gs[1, 0])
            metrics = {
                "RF train accuracy": best_model.score(X_train, Y_train),
                "Test accuracy": accuracy_score(Y_test, y_pred_rf),
                "Precision": precision_score(Y_test, y_pred_rf),
                "Recall": recall_score(Y_test, y_pred_rf),
                "F1 Score": f1_score(Y_test, y_pred_rf),
                "AUC-ROC": roc_auc_score(Y_test, y_test_rf_prob)
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
            ax5 = fig.add_subplot(gs[1, 1:])
            cm = confusion_matrix(Y_test, y_pred_rf, normalize='true')
            cm_percentage = (cm * 100).round(1)
            class_labels = ['bad', 'good']
            
            sns.heatmap(cm_percentage, annot=True, fmt='', cmap='Blues', cbar=False,
                        xticklabels=class_labels, yticklabels=class_labels,
                        annot_kws={'va':'center'}, ax=ax5)
            
            ax5.set_xlabel('Predicted')
            ax5.set_ylabel('True')
            ax5.set_title(f'Confusion Matrix for Random Forest\n(Subset {subset_type}) - Test Set')
            
            for t in ax5.texts:
                t.set_text(t.get_text() + " %")
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Results saved to {file_path}")