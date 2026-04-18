"""
analysis.py

Responsibility Shielding Analysis Pipeline
Author: Greg Stanley

Entry point. Run with:
    python analysis.py

All settings live in the general_settings dict below.
Default settings reproduce the manuscript results.
Use force_rebuild=True to regenerate outputs from scratch.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from config import (
    GeneralSettings,
    Filing,
    FileNames,
    FilePaths,
    TableNames,
    Visuals,
    MiscSettings,
    PunishSettings,
)
from preprocessing import (
    load_or_build_cleaned_dataframe,
)
from core import (
    compute_group_summaries,
    compute_twoafc_counts,
    compute_correlations,
    compute_individual_difference_regressions,
    compute_consistency_effects,
    compute_triangulation_results,
    run_confirmatory_and_exploratory_tests,
    compute_integrated_distal_blame_results,
    generate_processed_data_codebook,
)
from visualization import (
    plot_ratings_by_vignette_condition,
    plot_participant_level_shielding_heatmap,
    plot_within_subject_pairwise_comparisons,
    plot_shielding_effects_by_cognitive_load,
    plot_trial_order_effects_line_graph,
    plot_blameworthiness_wrongness_correlate,
    plot_triangulation_2afc_vs_rating_delta,
    plot_shielding_by_individual_difference,
    plot_response_distribution_histogram_by_condition,
)
from tables import generate_manuscript_and_supplementary_tables


"=========================================================================================="
"==================================== Common Variables ===================================="
"=========================================================================================="

ROOT = Path(__file__).resolve().parent 

file_names: FileNames = {
    "tests": "responsibility_shielding_tests.csv",
    "cleaned": "responsibility_shielding_cleaned.csv",
    "raw_data": "responsibility_shielding_raw_data.csv",
    "group_summaries": "responsibility_shielding_group_summaries.csv",
    "consistency_effects": "responsibility_shielding_consistency_effects.csv",
    "afc_counts_table": "responsibility_shielding_2afc_counts_table.csv",
    "afc_counts_long": "responsibility_shielding_2afc_counts_long.csv",
    "triangulation": "responsibility_shielding_triangulation.csv",
    "correlations": "responsibility_shielding_correlations.csv",
    "regressions": "responsibility_shielding_regressions.csv",
    "first_vignette": "responsibility_shielding_integrated_first_vignette_blame_models.csv",
    "within_subject": "responsibility_shielding_integrated_within_subject_blame_models.csv",
    "blame_models": "responsibility_shielding_integrated_blame_models.csv",
    "codebook": "responsibility_shielding_processed_codebook.csv",
}

table_names: dict[str, str] = {
    "table_1_participant_counts": "Table_1_Participant_Counts.csv",
    "table_2_means_by_dv_and_condition": "Table_2_Means_by_DV_and_Condition.csv",
    "table_3_primary_distal_blame_contrasts": "table_3_Primary_Distal_Blame_Contrasts.csv",
    "table_4_story_specific_distal_blame_contrasts": "table_4_Story_Specific_Distal_Blame_Contrasts.csv",
    "table_5_two_alternative_forced_choice_distribution": "table_5_Two_Alternative_Forced_Choice_Distribution.csv",
    "table_6_within_subject_pairwise_blame_matrix": "table_6_Within_Subject_Pairwise_Blame_Matrix.csv",
    "table_6_within_subject_pairwise_blame_long": "table_6_Within_Subject_Pairwise_Blame_Long.csv",
    "table_7_cognitive_load_blame_contrasts": "table_7_Cognitive_Load_Blame_Contrasts.csv",
    "table_8_order_effects_summary": "table_8_Order_Effects_Summary.csv",
    "table_9_secondary_dv_contrasts": "table_9_Secondary_DV_Contrasts.csv",
    "table_manifest": "Table_Manifest.csv",
}

file_paths: FilePaths = {
    "raw_data":    ROOT / "raw_data",
    "processed":   ROOT / "processed",
    "visuals":     ROOT / "visuals",
    "images":      ROOT / "images",
    "tables":      ROOT / "tables",
    "root":        ROOT
}

confirmatory_between_subjects_method = "pooled_ols"
freeze_timestamp_first = "2/19/2026 10:57:56 PM" 
freeze_timestamp_last =  "3/20/2026 10:00:09 AM"
rebuild_cleaned_dataframe = True
print_tables_to_terminal = True
use_integrated_models = False
force_rebuild = True
one_tailed = True

default_marker_size = 7
create_figures = True
export_figure = True
dark_mode = False
base_hue = 220

figure_layout = dict(
    template="plotly_dark" if dark_mode else "plotly_white",
    font=dict(
        family="Calibri",
        size=20,
        color="white" if dark_mode else "black",
    ),
    title_x=0.5,
    title_xanchor="center",
    margin=dict(
        l=200,
        r=200,
        t=90,
        b=90,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="center", x=0.5,
        bgcolor="hsla(0, 0%, 0%, 1.0)",
    ),
    xaxis=dict(
        showline=True,
        linewidth=2,
        linecolor="black",
        ticks="outside",
        tickwidth=2,
        ticklen=6,
        mirror=True,
        zeroline=False,
    ),
    yaxis=dict(
        showline=True,
        linewidth=2,
        linecolor="black",
        ticks="outside",
        tickwidth=2,
        ticklen=6,
        mirror=True,
        zeroline=False,
    ),
)

general_settings: GeneralSettings = {
    "filing": {
        "file_names": file_names,
        "file_paths": file_paths,
        "table_names": table_names
    },
    "visuals": {
        "figure_layout": figure_layout,
        "marker_size": default_marker_size,
        "create_figures": create_figures,
        "export_figure": export_figure,
        "dark_mode": dark_mode,
        "base_hue": base_hue
    },
    "punish": {
        "analysis_mode": "log1p_parametric", # Alt: "log1p_parametric", "raw_nonparametric", "raw_parametric"
        "bootstrap_iterations": 5000,
        "random_seed": 2026
    },
    "misc": {
        "confirmatory_between_subjects_method": confirmatory_between_subjects_method,
        "rebuild_cleaned_dataframe": rebuild_cleaned_dataframe,
        "print_tables_to_terminal": print_tables_to_terminal,
        "freeze_timestamp_first": freeze_timestamp_first,
        "freeze_timestamp_last": freeze_timestamp_last,
        "use_integrated_models": use_integrated_models,
        "force_rebuild": force_rebuild,
        "one_tailed": one_tailed
    },
}


def main() -> None:
    """
    Executes preprocessing + analysis and writes outputs to disk.
    """
    "Preprocessed dataframe"
    load_or_build_cleaned_dataframe(general_settings=general_settings, force_rebuild=general_settings["misc"]["rebuild_cleaned_dataframe"])

    "Group summaries"
    compute_group_summaries(general_settings=general_settings, force_rebuild=None)

    "2AFC counts"
    compute_twoafc_counts(general_settings=general_settings, force_rebuild=None, table_form=True)

    "Correlations"
    compute_correlations(general_settings=general_settings, force_rebuild=None)

    "Regressions"
    compute_individual_difference_regressions(general_settings=general_settings, force_rebuild=None)

    "Consistency effects"
    compute_consistency_effects(general_settings=general_settings, force_rebuild=None)

    "Triangulation"
    compute_triangulation_results(general_settings=general_settings, force_rebuild=None)

    "Confirmatory and exploratory tests"
    run_confirmatory_and_exploratory_tests(
        general_settings=general_settings,
        confirmatory_pooled_ols_covariance_type=None,
        force_rebuild=None
    )

    "Code book for preprocessed data"
    generate_processed_data_codebook(general_settings=general_settings, force_rebuild=None)

    "Integrated models"
    compute_integrated_distal_blame_results(
        general_settings=general_settings,
        force_rebuild=None,
    )

    if create_figures:
        "Figure 3"
        plot_ratings_by_vignette_condition(      dv="blame", base_hue=base_hue, general_settings=general_settings, figure_type="violin")

        "Figure 4"
        plot_participant_level_shielding_heatmap(dv="blame", base_hue=base_hue, general_settings=general_settings, include_marginals=True)

        "Table 5"
        plot_within_subject_pairwise_comparisons(dv="blame", base_hue=base_hue, general_settings=general_settings, include_proximate_agent=True)

        "Figure 6"
        plot_shielding_effects_by_cognitive_load(dv="blame", base_hue=base_hue, general_settings=general_settings, figure_type="violin", delta_type="CH_CC")

        "Figure 7"
        plot_trial_order_effects_line_graph(     dv="blame", base_hue=base_hue, general_settings=general_settings, order_analysis_mode="legacy")

        "Figure 8"
        plot_blameworthiness_wrongness_correlate(            base_hue=base_hue, general_settings=general_settings, all_ratings=True, jitter_strength=0.1)

        "Additional Figures"
        plot_triangulation_2afc_vs_rating_delta( dv="blame", base_hue=base_hue, general_settings=general_settings, comparison="CH_CC")
        plot_shielding_by_individual_difference( dv="blame", base_hue=base_hue, general_settings=general_settings, predictor="indcol")
        plot_shielding_by_individual_difference( dv="blame", base_hue=base_hue, general_settings=general_settings, predictor="crt")

        plot_response_distribution_histogram_by_condition(general_settings=general_settings, dv="blame")
        plot_response_distribution_histogram_by_condition(general_settings=general_settings, dv="wrong")
        plot_response_distribution_histogram_by_condition(general_settings=general_settings, dv="punish")


    "Tables in the order they appear in the paper"
    generate_manuscript_and_supplementary_tables(
        general_settings=general_settings,
        force_rebuild=None,
    )


if __name__ == "__main__":
    main()





