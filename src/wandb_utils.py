import wandb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from src.utils import prepare_feature_subset, preprocess, create_new_columns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import pandas as pd

def plot_subsets_experiment_wandb(results, results_info, df_pos, df_neg, experiment_name):
    wandb.init(project="starsml", name=experiment_name)

    for subset_type, result in results.items():
        best_model = result['best_model']
        
        create_new_columns(df_pos)
        create_new_columns(df_neg)
        
        df_pos_subset = prepare_feature_subset(df_pos, subset_type)
        df_neg_subset = prepare_feature_subset(df_neg, subset_type)

        X_train, X_test, Y_train, Y_test, _, indices_test, categorical_features, scaler = preprocess(
            df_pos_subset, df_neg_subset,
            log_transform=results_info['log_transform'],
            normalization_method=results_info['normalization_method'],
            model_type=results_info['model_type'],
            feature_names=df_pos_subset.columns,
            random_seed=results_info['random_seed']
        )

        wandb.log({f"features_{subset_type}": X_test.columns.tolist()})

        y_test_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)

        df_test = pd.concat([df_pos, df_neg]).iloc[indices_test].reset_index(drop=True)

        # log scatter plot
        fig, ax = plt.subplots()
        sc = ax.scatter(y_test_prob, df_test['p_i'], c=df_test['p_any'], cmap='coolwarm_r', s=6)
        plt.colorbar(sc).set_label('p_any', rotation=270, labelpad=15)
        ax.set_title(f'p_i_NWAY vs p_i_RF (Subset {subset_type}) - Test Set')
        ax.set_ylabel('p_i_NWAY')
        ax.set_xlabel('p_i_RF')
        wandb.log({f"scatter_plot_{subset_type}": wandb.Image(fig)})
        plt.close(fig)

        # log histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(y_test_prob[Y_test == 1], bins=20, kde=False, ax=ax1)
        ax1.set_title(f'p_i_RF for true positive matches (Subset {subset_type})')
        ax1.set_xlabel('p_i_RF')
        ax1.set_ylabel('Frequency')
        sns.histplot(y_test_prob[Y_test == 0], bins=20, kde=False, ax=ax2)
        ax2.set_title(f'p_i_RF for true negative matches (Subset {subset_type})')
        ax2.set_xlabel('p_i_RF')
        ax2.set_ylabel('Frequency')
        wandb.log({f"histograms_{subset_type}": wandb.Image(fig)})
        plt.close(fig)

        # log confusion matrix
        cm = confusion_matrix(Y_test, y_pred, normalize='true')
        fig, ax = plt.subplots()
        sns.heatmap(cm * 100, annot=True, fmt='.1f', cmap='Blues', cbar=False,
                    xticklabels=['bad', 'good'], yticklabels=['bad', 'good'],
                    annot_kws={'va':'center'}, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (Subset {subset_type})')
        for t in ax.texts:
            t.set_text(t.get_text() + " %")
        wandb.log({f"confusion_matrix_{subset_type}": wandb.Image(fig)})
        plt.close(fig)

        # log metrics
        metrics = {
            f"{subset_type}_train_accuracy": best_model.score(X_train, Y_train),
            f"{subset_type}_test_accuracy": accuracy_score(Y_test, y_pred),
            f"{subset_type}_precision": precision_score(Y_test, y_pred),
            f"{subset_type}_recall": recall_score(Y_test, y_pred),
            f"{subset_type}_f1_score": f1_score(Y_test, y_pred),
            f"{subset_type}_auc_roc": roc_auc_score(Y_test, y_test_prob)
        }
        wandb.log(metrics)

    wandb.finish()