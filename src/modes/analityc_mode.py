# Режим виконання
#from src.utils.logger import get_logger
from src.utils.file_utils import aggregate_statistics, aggregate_metric_over_epochs, save_aggregated_statistics, load_and_aggregate_confusion_matrices, combine_activity_stat_files, save_checkpoint, load_checkpoint, load_register, save_prepared_data, load_prepared_data, load_global_statistics_from_json, save2csv
from src.utils.file_utils_l import is_file_exist, join_path
from src.utils.visualizer import plot_metric_over_epochs, visualize_diff_conf_matrix, plot_architecture_radar_by_metric,plot_regression_logs_vs_bpmn, plot_class_bar_chart,visualize_aggregated_conf_matrix, visualize_confusion_matrix, plot_avg_epoch_time_bar, plot_regression_by_architecture
from src.config.config import LEARN_PR_DIAGRAMS_PATH, NN_PR_MODELS_CHECKPOINTS_PATH, NN_PR_MODELS_DATA_PATH, PROCESSED_DATA_PATH

#logger = get_logger(__name__)

def run_analitics_mode(args):
    """
    Аналітики резульатів.
    """
    seed=9467
    seed=None
   # logger.info("🚀 Запущено аналітичний режим.")
    print("⚙️ Виконується аналітичний режим ...")

    final_df = aggregate_statistics(LEARN_PR_DIAGRAMS_PATH) #комбінована статистика навчання
    final_df_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'final_df_statistics.xlsx'])
    save_aggregated_statistics(final_df, final_df_file)

    combined_df = combine_activity_stat_files(LEARN_PR_DIAGRAMS_PATH) #комбінована статистика розподілу
    final_df_ac_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'final_train_vs_val_accuracy_all.xlsx'])
    save_aggregated_statistics(combined_df, final_df_ac_file)

    cm_bpmn = load_and_aggregate_confusion_matrices(LEARN_PR_DIAGRAMS_PATH, data_type_filter="bpmn", reduction="avg", normalize=False)
    cm_norm_bpmn = load_and_aggregate_confusion_matrices(LEARN_PR_DIAGRAMS_PATH, data_type_filter="bpmn", reduction="avg", normalize=True)
    final_cm_bpmn_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'final_cm_bpmn.xlsx'])
    save_aggregated_statistics(cm_bpmn, final_cm_bpmn_file)
    cm_logs = load_and_aggregate_confusion_matrices(LEARN_PR_DIAGRAMS_PATH, data_type_filter="logs", reduction="avg", normalize=False)
    cm_norm_logs = load_and_aggregate_confusion_matrices(LEARN_PR_DIAGRAMS_PATH, data_type_filter="logs", reduction="avg", normalize=True)
    final_cm_logs_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'final_cm_log.xlsx'])
    save_aggregated_statistics(cm_logs, final_cm_logs_file)

    graf_cm_diff_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_cm_diff.png'])
    visualize_diff_conf_matrix(cm_bpmn,cm_logs,top_k=60, top_k_mode="diff",min_value=4.1, normalize=False, title="Δ Confusion Matrix (BPMN - Logs)", file_path=graf_cm_diff_file)
    graf_cm_norm_diff_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_cm_norm_diff.png'])
    visualize_diff_conf_matrix(cm_bpmn,cm_logs,top_k=60, top_k_mode="diff", min_value=0.01, normalize=True, title="Δ Confusion Matrix Normilized (BPMN - Logs)", file_path=graf_cm_norm_diff_file)

    graf_cm_bpmn_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_cm_bpmn.png'])
    visualize_aggregated_conf_matrix(
        cm_df=cm_bpmn,
        title="Confusion Matrix (BPMN)",
        top_k=40,
        use_log_scale=True,
        file_path=graf_cm_bpmn_file
    )
    graf_cm_bpmn_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_cm_norm_bpmn.png'])
    visualize_aggregated_conf_matrix(
        cm_df=cm_norm_bpmn,
        title="Normalized Confusion Matrix (BPMN)",
        top_k=40,
        use_log_scale=True,
        file_path=graf_cm_bpmn_file
    )

    graf_cm_logs_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_cm_logs.png'])
    visualize_aggregated_conf_matrix(
        cm_df=cm_logs,
        title="Confusion Matrix (Logs)",
        top_k=40,
        use_log_scale=True,
        file_path=graf_cm_logs_file
    )
    graf_cm_logs_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_cm_norm_logs.png'])
    visualize_aggregated_conf_matrix(
        cm_df=cm_norm_logs,
        title="Normalized Confusion Matrix (Logs)",
        top_k=40,
        use_log_scale=True,
        file_path=graf_cm_logs_file
    )

    graf_regression_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_regression_bpmn.png'])
    plot_regression_by_architecture(
        df=combined_df,
        chart_title="Average Regression Curve (BPMN)",
        data_type_filter="bpmn",
        arhitec_filter=None,
        seed_filter=None,
        group_arhitec=False,
        group_seed=True,
        poly_level=4,
        figsize=(18, 10),
        ylim=(-0.05, 1.2),
        xlim=(0, 3000),
        file_path = graf_regression_file
    )

    graf_regression_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_regression_logs.png'])
    plot_regression_by_architecture(
        df=combined_df,
        chart_title="Average Regression Curve (Logs)",
        data_type_filter="logs",
        arhitec_filter=None,
        seed_filter=None,
        group_arhitec=False,
        group_seed=True,
        poly_level=4,
        figsize=(18, 10),
        ylim=(-0.05, 1.2),
        xlim=(0, 3000),
        file_path = graf_regression_file
    )
    graf_regression_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_regression_grouped.png'])

    plot_regression_logs_vs_bpmn(
        df=combined_df,
        chart_title="Regression Curve: Logs vs BPMN (All Architectures)",
        group_seed=True,
        group_arhitec=True,
        poly_level=4,
        figsize=(20, 8),
        ylim=(-0.05, 1.2),
        xlim=(0, 3000),
        file_path=graf_regression_file
    )

    final_acc_df = aggregate_metric_over_epochs(LEARN_PR_DIAGRAMS_PATH, 'val_accuracy')
    final_acct3_df = aggregate_metric_over_epochs(LEARN_PR_DIAGRAMS_PATH, 'val_top_k_accuracy')
    final_train_loss_df = aggregate_metric_over_epochs(LEARN_PR_DIAGRAMS_PATH, 'train_loss')
    final_oos_df = aggregate_metric_over_epochs(LEARN_PR_DIAGRAMS_PATH, 'val_out_of_scope_rate')

    graf_radar_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_radar_bpmn.png'])
    plot_architecture_radar_by_metric(
        df=final_df,
        chart_title="Метрики по архітектурах (BPMN)",
        data_type_filter="bpmn",
        seed_filter=None,
        normalize=True,
        figsize=(10, 10),
        file_path = graf_radar_file
    )
    graf_radar_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_radar_log.png'])
    plot_architecture_radar_by_metric(
        df=final_df,
        chart_title="Метрики по архітектурах (Logs)",
        data_type_filter="logs",
        seed_filter=seed,
        normalize=True,
        figsize=(10, 10),
        file_path = graf_radar_file
    )

    avg_epoch_time_bar_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_bar_avg_epoch_time_bpmn.png'])
    plot_avg_epoch_time_bar(
        df=final_df,
        chart_title="Середній Час Навчання На Епоху (BPMN)",
        data_type_filter="bpmn",
        seed_filter=None,
        figsize=(18, 10),
        file_path=avg_epoch_time_bar_file
    )
    avg_epoch_time_bar_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_bar_avg_epoch_time_log.png'])
    plot_avg_epoch_time_bar(
        df=final_df,
        chart_title="Середній Час Навчання На Епоху (Logs)",
        data_type_filter="logs",
        seed_filter=None,
        figsize=(18, 10),
        file_path=avg_epoch_time_bar_file
    )

    bar_radar_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_bar_class_bpmn.png'])
    class_dict_dynamics  = {
        "Hybrid": ["GGNN", "MuseGNN", "TransformerMLP","GraphMixer", "Graphormer"],
        "Static": ["MLP", "GCN", "GraphSAGE", "APPNP", "GPRGNN","MixHop", "DeepGCN", "GATConv", "GATv2" ],
        "Discrete-dynamic": ["TGCN", "GRUGAT"],
        "Continuously-dynamic": ["TGAT", "MuseGNN", "TemporalGAT"]
    }
    metric_list = ["val_accuracy", "val_top_k_accuracy", "val_out_of_scope_rate"]
    metric_labels = ["Accuracy", "Top-3 Accuracy", "Out-of-Scope Rate"]
    plot_class_bar_chart(
        df=final_df,
        class_dict=class_dict_dynamics ,
        metric_list=metric_list,
        metric_labels=metric_labels,
        chart_title="Architectures Grouped by Dynamics Type",
        data_type_filter='bpmn',
        figsize=(16, 10),
        file_path=bar_radar_file
    )

    bar_radar_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_bar_atten_bpmn.png'])
    class_dict_attention  = {
        "Uses Attention": [
            "TGAT", "TemporalGAT", "DFAGNN", "GRUGAT", "MuseGNN", "TransformerMLP", "GraphMixer",
            "GATConv", "GATv2", "Graphormer", "SAN", "GPS", "Performer-GNN"
        ],
        "No Attention": [
            "MLP", "GCN", "GraphSAGE", "APPNP", "GPRGNN", "MixHop", "DeepGCN", "TGCN",
            "GGNN", "GIN", "MEIG", "MiTFM", "H2GCN"
        ]
    }
    metric_list = ["val_accuracy", "val_top_k_accuracy", "val_out_of_scope_rate"]
    metric_labels = ["Accuracy", "Top-3 Accuracy", "Out-of-Scope Rate"]
    plot_class_bar_chart(
        df=final_df,
        class_dict=class_dict_attention ,
        metric_list=metric_list,
        metric_labels=metric_labels,
        chart_title="Effect of Attention Mechanism",
        data_type_filter='bpmn',
        figsize=(16, 10),
        file_path=bar_radar_file
    )

    bar_radar_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_bar_aggreg_bpmn.png'])
    class_dict_aggregation  = {
        "Mean/Max/Sum Aggr": ["GCN", "GraphSAGE", "GIN", "GPRGNN", "APPNP"],
        "RNN-based (GRU/LSTM)": ["GGNN", "GRUGAT", "TGCN"],
        "Attention-based": ["GATConv", "GATv2", "TGAT", "TemporalGAT", "Graphormer", "SAN", "GPS", "Performer-GNN"],
        "MLP-based Mixing": ["MixHop", "TransformerMLP", "GraphMixer", "MEIG", "MiTFM"]
    }
    metric_list = ["val_accuracy", "val_top_k_accuracy", "val_out_of_scope_rate"]
    metric_labels = ["Accuracy", "Top-3 Accuracy", "Out-of-Scope Rate"]
    plot_class_bar_chart(
        df=final_df,
        class_dict=class_dict_aggregation ,
        metric_list=metric_list,
        metric_labels=metric_labels,
        data_type_filter='bpmn',
        chart_title="Aggregation type",
        figsize=(16, 10),
        file_path=bar_radar_file
    )
    
    bar_scope_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_bar_scope_bpmn.png'])
    class_dict_scope = {
        "Local Propagation": [
            "GCN", "GATConv", "GATv2", "GraphSAGE", "GIN", "TGAT", "TemporalGAT", "GRUGAT",
            "GGNN", "TGCN", "APPNP", "GPRGNN", "MixHop", "DeepGCN"
        ],
        "Global Attention": ["Graphormer", "SAN", "GPS", "Performer-GNN"],
        "Globalized MLP-based": ["GraphMixer", "TransformerMLP", "MuseGNN", "DFAGNN", "MEIG", "MiTFM"]
    }
    metric_list = ["val_accuracy", "val_top_k_accuracy", "val_out_of_scope_rate"]
    metric_labels = ["Accuracy", "Top-3 Accuracy", "Out-of-Scope Rate"]
    plot_class_bar_chart(
        df=final_df,
        class_dict=class_dict_scope,
        metric_list=metric_list,
        metric_labels=metric_labels,
        data_type_filter='bpmn',
        chart_title="Propagation Scope of GNN Architectures",
        figsize=(16, 10),
        file_path=bar_scope_file
    )
    bar_scope_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_bar_input_bpmn.png'])
    class_dict_input = {
        "Positional Encoding": ["Graphormer", "GPS", "SAN"],
        "Structural Input": ["GCN", "GIN", "GraphSAGE", "GATConv", "GATv2", "APPNP", "DeepGCN", "MixHop", "GPRGNN",
                             "H2GCN"],
        "Structural + Semantic": ["MuseGNN", "TransformerMLP", "GraphMixer", "MEIG", "DFAGNN", "MiTFM"]
    }
    metric_list = ["val_accuracy", "val_top_k_accuracy", "val_out_of_scope_rate"]
    metric_labels = ["Accuracy", "Top-3 Accuracy", "Out-of-Scope Rate"]
    plot_class_bar_chart(
        df=final_df,
        class_dict=class_dict_input,
        metric_list=metric_list,
        metric_labels=metric_labels,
        data_type_filter='bpmn',
        chart_title="Input Representation Strategy",
        figsize=(16, 10),
        file_path=bar_scope_file
    )
    bar_scope_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_bar_model_type_bpmn.png'])
    class_dict_model_type = {
        "Standard GNN": ["GCN", "GIN", "GraphSAGE", "GATConv", "GATv2", "APPNP", "GPRGNN"],
        "GNN + RNN": ["GGNN", "GRUGAT", "TGCN"],
        "GNN + Transformer": ["TGAT", "TemporalGAT", "Graphormer", "GPS", "SAN", "Performer-GNN"],
        "GNN + MLP/Neural Mixer": ["MixHop", "DeepGCN", "TransformerMLP", "GraphMixer", "MuseGNN", "DFAGNN", "MEIG",
                                   "MiTFM"]
    }
    metric_list = ["val_accuracy", "val_top_k_accuracy", "val_out_of_scope_rate"]
    metric_labels = ["Accuracy", "Top-3 Accuracy", "Out-of-Scope Rate"]
    plot_class_bar_chart(
        df=final_df,
        class_dict=class_dict_model_type,
        metric_list=metric_list,
        data_type_filter='bpmn',
        metric_labels=metric_labels,
        chart_title="Architectures Grouped by Model Type",
        figsize=(16, 10),
        file_path=bar_scope_file
    )

    #final_metric_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'final_df_val_accuracy.xlsx'])
    #save_aggregated_statistics(final_acc_df, final_metric_file)

    graf_acc_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_accuracy_file.png'])
    plot_metric_over_epochs(final_acc_df, "Accuracy Over Epochs (BPMN)",'bpmn',seed, max_epoch=80,figsize=(16, 10),loc='lower right',file_path=graf_acc_file )

    graf_acct3_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_acct3_file.png'])
    plot_metric_over_epochs(final_acct3_df, "TOP 3 Accuracy Over Epochs (BPMN)",'bpmn',seed, max_epoch=80,figsize=(16, 10),loc='lower right',file_path=graf_acct3_file )

    graf_loss_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_loss_file.png'])
    plot_metric_over_epochs(final_train_loss_df, "Loss Over Epochs (BPMN)",'bpmn',seed, ylim=(1,6), max_epoch=80,figsize=(16, 10),loc='upper right', file_path=graf_loss_file )

    graf_oos_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_oos_file.png'])
    plot_metric_over_epochs(final_oos_df, " Out of Scope Epochs (BPMN)",'bpmn',seed,ylim=(0,0.1), max_epoch=80,figsize=(16, 10),loc='upper right',file_path=graf_oos_file )

    graf_acc_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_accuracy_log_file.png'])
    plot_metric_over_epochs(final_acc_df, "Accuracy Over Epochs (Logs)",'logs',seed, max_epoch=80,figsize=(16, 10),loc='upper left',file_path=graf_acc_file )

    graf_acct3_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_acct3_log_file.png'])
    plot_metric_over_epochs(final_acct3_df, "TOP 3 Accuracy Over Epochs (Logs)",'logs',seed, max_epoch=80,figsize=(16, 10),loc='upper left',file_path=graf_acct3_file )

    graf_loss_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_loss_log_file.png'])
    plot_metric_over_epochs(final_train_loss_df, "Loss Over Epochs (Logs)",'logs',seed,ylim=(1,6), max_epoch=80,figsize=(16, 10),loc='upper right',file_path=graf_loss_file )

    graf_oos_file = join_path([LEARN_PR_DIAGRAMS_PATH, f'graf_oos_log_file.png'])
    plot_metric_over_epochs(final_oos_df, " Out of Scope Epochs (Logs)",'logs',seed,ylim=(0,0.1), max_epoch=80,figsize=(16, 10),loc='upper right',file_path=graf_oos_file )

