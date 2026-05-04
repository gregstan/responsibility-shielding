from __future__ import annotations
from scipy import stats
from pathlib import Path
from typing import Any
import pandas as pd, numpy as np
from config import GeneralSettings, Filing, FileNames, FilePaths, TableNames
from preprocessing import (
    load_or_build_cleaned_dataframe,
    load_analysis_dataframe,
    _save_analysis_dataframe_to_processed_folder,
)
from core import (
    run_confirmatory_and_exploratory_tests,
    run_welch_t_test_between_groups,
    run_one_sample_t_test_on_delta,
    compute_group_summaries,
    compute_twoafc_counts,
    compute_correlations,
    compute_individual_difference_regressions,
    compute_consistency_effects,
    compute_triangulation_results,
    compute_integrated_distal_blame_results,
)


"=========================================================================================="
"==================================== Table Generation ===================================="
"=========================================================================================="

def _load_table_dataframe_from_tables_folder(
    general_settings: GeneralSettings,
    table_file_name: str,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """
    Load a saved manuscript table from the tables folder.

    Arguments:
        • general_settings: GeneralSettings
            - Master project settings dictionary.
        • table_file_name: str
            - File name for the table CSV.
        • force_rebuild: bool
            - If True, ignore any saved table and force regeneration.

    Returns:
        • dict[str, Any]
            - Dictionary with keys:
                "success"
                "error"
                "message"
                "dataframe"
                "file_path"
    """
    file_path_table: Path = general_settings["filing"]["file_paths"]["tables"] / table_file_name

    if force_rebuild:
        return {
            "success": False,
            "error": False,
            "message": f"Forced rebuild requested for {table_file_name}.",
            "dataframe": None,
            "file_path": file_path_table,
        }

    if file_path_table.exists():
        try:
            dataframe_table = pd.read_csv(file_path_table, encoding="utf-8-sig", engine="python")
            return {
                "success": True,
                "error": False,
                "message": f"Loaded existing table: {file_path_table}",
                "dataframe": dataframe_table,
                "file_path": file_path_table,
            }
        except Exception as exception:
            return {
                "success": False,
                "error": True,
                "message": f"Failed to load existing table {file_path_table}: {exception}",
                "dataframe": None,
                "file_path": file_path_table,
            }

    return {
        "success": False,
        "error": False,
        "message": f"No existing table found at {file_path_table}.",
        "dataframe": None,
        "file_path": file_path_table,
    }


def _save_table_dataframe_to_tables_folder(
    dataframe_table: pd.DataFrame,
    general_settings: GeneralSettings,
    table_file_name: str,
) -> Path:
    """
    Save a manuscript-facing table CSV to the tables folder.

    Arguments:
        • dataframe_table: pd.DataFrame
            - Table dataframe to save.
        • general_settings: GeneralSettings
            - Master project settings dictionary.
        • table_file_name: str
            - Output file name for the table CSV.

    Returns:
        • Path
            - Full path to the saved CSV.
    """
    file_path_table: Path = general_settings["filing"]["file_paths"]["tables"] / table_file_name
    file_path_table.parent.mkdir(parents=True, exist_ok=True)
    dataframe_table.to_csv(file_path_table, index=False, encoding="utf-8-sig")
    return file_path_table


def _format_p_value_for_manuscript_table(
        p_value: float | int | None
) -> str:
    """
    Format a p-value for manuscript-facing display.

    Arguments:
        • p_value: float | int | None
            - Numeric p-value.

    Returns:
        • str
            - Nicely formatted p-value string.
    """
    if p_value is None or pd.isna(p_value):
        return ""

    p_value = float(p_value)

    if p_value < 0.001:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def _format_ci_for_manuscript_table(
        ci95_lower: float | int | None, 
        ci95_upper: float | int | None
) -> str:
    """
    Format a 95% confidence interval for manuscript-facing display.

    Arguments:
        • ci95_lower: float | int | None
        • ci95_upper: float | int | None

    Returns:
        • str
            - Confidence interval string.
    """
    if ci95_lower is None or ci95_upper is None or pd.isna(ci95_lower) or pd.isna(ci95_upper):
        return ""
    return f"[{float(ci95_lower):.2f}, {float(ci95_upper):.2f}]"


def _pretty_print_table_to_terminal(
    table_title: str,
    dataframe_table: pd.DataFrame,
) -> None:
    """
    Pretty-print a manuscript table to the terminal.

    Arguments:
        • table_title: str
            - Human-readable title.
        • dataframe_table: pd.DataFrame
            - Table dataframe to print.
    """
    "Create a view/copy with specific columns dropped without affecting the original"
    excluded_columns = {'p_value_two_tailed', 'p_value_one_tailed', 'p_value_holm', 
                        'p_value_displayed', 'p_value_correction', 'source_file', 'meaning'}
    printable_table = dataframe_table.drop(columns=excluded_columns, errors='ignore')    

    print("\n" + "=" * 110)
    print(table_title)
    print("=" * 110)
    print(printable_table.to_string(index=False))


def _extract_exact_test_row_from_test_csv(
    dataframe_tests: pd.DataFrame,
    inclusion_filter: str,
    story_family: str,
    load_condition: str,
    design: str,
    dv: str,
    agent_role: str,
    contrast_type: str,
    analysis_family: str | None = None,
) -> pd.Series:
    """
    Extract exactly one row from responsibility_shielding_tests.csv.

    Arguments:
        • dataframe_tests: pd.DataFrame
            - Dataframe loaded from responsibility_shielding_tests.csv.
        • inclusion_filter: str
            - "included_only" or "all_finishers"
        • story_family: str
            - "pooled", "firework", or "trolley"
        • load_condition: str
            - "pooled", "high", or "low"
        • design: str
            - "between_subjects_first_vignette" or "within_subjects_all_vignettes"
        • dv: str
            - "blame", "wrong", or "punish"
        • agent_role: str
            - "distal" or "proximate"
        • contrast_type: str
            - E.g. "CH - CC", "DIV - CC", "CH - DIV", "MIN(CH, DIV) - CC", "CC - CH"
        • analysis_family: str | None
            - "confirmatory", "exploratory", or None to ignore this column.

    Returns:
        • pd.Series
            - Exactly one matching row.

    Notes:
        • This helper is tolerant to minor schema drift while the codebase is being standardized.
        • It accepts the preferred logical names and maps them onto whichever concrete column names
          currently exist in tests.csv.
    """
    tests_dataframe = dataframe_tests.copy()

    participants_column_name = (
        "inclusion_filter" if "inclusion_filter" in tests_dataframe.columns else "participants"
    )
    story_family_column_name = (
        "story_condition" if "story_condition" in tests_dataframe.columns else "story_family"
    )
    contrast_type_column_name = (
        "contrast_or_condition" if "contrast_or_condition" in tests_dataframe.columns else "contrast_type"
    )

    def normalize_inclusion_filter_value(value: str) -> set[str]:
        if value in {"included_only", "included"}:
            return {"included_only", "included"}
        if value == "all_finishers":
            return {"all_finishers"}
        return {value}

    def normalize_story_family_value(value: str) -> set[str]:
        if value in {"pooled", "all"}:
            return {"pooled", "all"}
        return {value}

    def normalize_agent_role_value(value: str) -> set[str]:
        if value in {"distal", "clark"}:
            return {"distal", "clark"}
        return {value}

    def normalize_dv_value(value: str) -> set[str]:
        if value in {"wrong", "wrongness"}:
            return {"wrong", "wrongness"}
        if value in {"punish", "punishment"}:
            return {"punish", "punishment"}
        return {value}

    valid_inclusion_filter_values = normalize_inclusion_filter_value(inclusion_filter)
    valid_story_family_values = normalize_story_family_value(story_family)
    valid_agent_role_values = normalize_agent_role_value(agent_role)
    valid_dv_values = normalize_dv_value(dv)

    row_mask = (
        tests_dataframe[participants_column_name].isin(valid_inclusion_filter_values)
        & tests_dataframe[story_family_column_name].isin(valid_story_family_values)
        & (tests_dataframe["load_condition"] == load_condition)
        & (tests_dataframe["analysis_scope"] == design)
        & tests_dataframe["dv"].isin(valid_dv_values)
        & tests_dataframe["agent_role"].isin(valid_agent_role_values)
        & (tests_dataframe[contrast_type_column_name] == contrast_type)
    )

    if analysis_family is not None:
        row_mask = row_mask & (tests_dataframe["analysis_family"] == analysis_family)

    matching_rows = tests_dataframe.loc[row_mask].copy()

    if matching_rows.shape[0] != 1:
        raise Exception(
            "Expected exactly one row in tests.csv for "
            f"inclusion_filter={inclusion_filter!r}, "
            f"story_family={story_family!r}, "
            f"load_condition={load_condition!r}, "
            f"analysis_scope={design!r}, "
            f"dv={dv!r}, "
            f"agent_role={agent_role!r}, "
            f"contrast_type={contrast_type!r}, "
            f"analysis_family={analysis_family!r}, "
            f"but found {matching_rows.shape[0]} rows."
        )

    return matching_rows.iloc[0]


def _extract_estimate_from_test_row(row_series: pd.Series) -> float:
    """
    Extract the estimate that should be reported from one tests.csv row.

    Arguments:
        • row_series: pd.Series
            - One row from responsibility_shielding_tests.csv.

    Returns:
        • float
            - mean_difference_a_minus_b or median_difference_a_minus_b, depending on
              location_statistic_reported.
    """
    location_statistic_reported = str(row_series["location_statistic_reported"]).strip().lower()

    if location_statistic_reported == "median_difference":
        return float(row_series["median_difference_a_minus_b"])

    return float(row_series["mean_difference_a_minus_b"])


def _extract_reported_p_value_from_test_row(
    row_series: pd.Series,
    general_settings: GeneralSettings,
    use_holm_if_available: bool = False,
) -> float:
    """
    Extract the p-value that should be displayed from one tests.csv row.

    Arguments:
        • row_series: pd.Series
            - One row from responsibility_shielding_tests.csv.
        • general_settings: GeneralSettings
            - Master settings dictionary.
        • use_holm_if_available: bool
            - If True, use p_value_holm when present and non-missing.

    Returns:
        • float
            - Confirmatory rows: p_value_holm when requested and available
            - All other rows: one-tailed or two-tailed p depending on general_settings["misc"]["one_tailed"]
    """
    if use_holm_if_available and "p_value_holm" in row_series.index and pd.notna(row_series["p_value_holm"]):
        return float(row_series["p_value_holm"])

    if general_settings["misc"].get("one_tailed", False):
        return float(row_series["p_value_one_tailed"])

    return float(row_series["p_value_two_tailed"])


def _extract_exact_consistency_row_from_csv(
    dataframe_consistency_effects: pd.DataFrame,
    inclusion_filter: str,
    comparison: str,
) -> pd.Series:
    """
    Extract exactly one row from consistency_effects.csv.

    Arguments:
        • dataframe_consistency_effects: pd.DataFrame
            - Output of compute_consistency_effects(...).
        • inclusion_filter: str
            - included_only or all_finishers.
        • comparison: str
            - Exact comparison label from compute_consistency_effects(...).

    Returns:
        • pd.Series
            - The unique matching row.
    """
    row_mask = (
        (dataframe_consistency_effects["inclusion_filter"] == inclusion_filter)
        & (dataframe_consistency_effects["comparison"] == comparison)
    )

    matching_rows = dataframe_consistency_effects.loc[row_mask].copy()

    if matching_rows.shape[0] != 1:
        raise Exception(
            f"Expected exactly one row in consistency_effects.csv for "
            f"inclusion_filter={inclusion_filter!r}, comparison={comparison!r}, "
            f"but found {matching_rows.shape[0]} rows."
        )

    return matching_rows.iloc[0]


def _build_within_subject_pairwise_blame_matrix(
    cleaned_dataframe: pd.DataFrame,
    only_included_participants: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the manuscript-facing within-subject pairwise blame matrix and its long-form backing table.

    Arguments:
        • cleaned_dataframe: pd.DataFrame
            - Cleaned analysis dataframe.
        • only_included_participants: bool
            - If True, restrict to included participants.

    Returns:
        • tuple[pd.DataFrame, pd.DataFrame]
            - formatted_matrix_dataframe
            - pairwise_long_dataframe
    """
    analysis_dataframe = cleaned_dataframe.copy()
    if only_included_participants:
        analysis_dataframe = analysis_dataframe.loc[analysis_dataframe["included"] == True].copy()  # noqa: E712

    ordered_series_metadata: list[tuple[str, str]] = [
        ("CC Distal", "distal_blame_cc"),
        ("CH Distal", "distal_blame_ch"),
        ("DIV Distal", "distal_blame_div"),
        ("CC Proximate", "proximate_blame_cc"),
        ("CH Proximate", "proximate_blame_ch"),
    ]

    category_summary_rows: list[dict[str, Any]] = []
    for category_label, column_name in ordered_series_metadata:
        values = pd.to_numeric(analysis_dataframe[column_name], errors="coerce").dropna()
        category_summary_rows.append(
            {
                "category_label": category_label,
                "column_name": column_name,
                "n": int(values.shape[0]),
                "mean": float(values.mean()) if values.shape[0] > 0 else np.nan,
                "std": float(values.std(ddof=1)) if values.shape[0] > 1 else np.nan,
            }
        )
    category_summary_dataframe = pd.DataFrame(category_summary_rows)

    pairwise_rows: list[dict[str, Any]] = []
    formatted_matrix_dataframe = pd.DataFrame(
        "",
        index=[metadata[0] for metadata in ordered_series_metadata],
        columns=[metadata[0] for metadata in ordered_series_metadata],
    )

    for row_index, (row_label, row_column_name) in enumerate(ordered_series_metadata):
        for column_index, (column_label, column_column_name) in enumerate(ordered_series_metadata):
            if column_index < row_index:
                formatted_matrix_dataframe.loc[row_label, column_label] = ""
                continue

            if column_index == row_index:
                formatted_matrix_dataframe.loc[row_label, column_label] = "—"
                continue

            paired_dataframe = analysis_dataframe[[row_column_name, column_column_name]].copy()
            paired_dataframe[row_column_name] = pd.to_numeric(paired_dataframe[row_column_name], errors="coerce")
            paired_dataframe[column_column_name] = pd.to_numeric(paired_dataframe[column_column_name], errors="coerce")
            paired_dataframe = paired_dataframe.dropna()

            paired_difference_values = paired_dataframe[column_column_name] - paired_dataframe[row_column_name]
            n_pairs = int(paired_difference_values.shape[0])

            mean_difference = float(np.mean(paired_difference_values)) if n_pairs > 0 else np.nan
            ci95_lower = np.nan
            ci95_upper = np.nan
            t_statistic = np.nan
            p_value = np.nan
            cohen_dz = np.nan

            if n_pairs >= 2:
                t_statistic, p_value = stats.ttest_1samp(paired_difference_values, popmean=0.0)
                standard_deviation_of_difference = float(np.std(paired_difference_values, ddof=1))

                if standard_deviation_of_difference == 0:
                    cohen_dz = 0.0 if mean_difference == 0 else np.nan
                else:
                    cohen_dz = mean_difference / standard_deviation_of_difference

                standard_error_of_difference = standard_deviation_of_difference / np.sqrt(n_pairs)
                critical_t_value = stats.t.ppf(0.975, df=n_pairs - 1)
                ci95_lower = mean_difference - critical_t_value * standard_error_of_difference
                ci95_upper = mean_difference + critical_t_value * standard_error_of_difference

            pairwise_rows.append(
                {
                    "row_label": row_label,
                    "column_label": column_label,
                    "row_column_name": row_column_name,
                    "column_column_name": column_column_name,
                    "n_pairs": n_pairs,
                    "mean_difference_column_minus_row": mean_difference,
                    "ci95_lower": float(ci95_lower) if not pd.isna(ci95_lower) else np.nan,
                    "ci95_upper": float(ci95_upper) if not pd.isna(ci95_upper) else np.nan,
                    "t_statistic": float(t_statistic) if not pd.isna(t_statistic) else np.nan,
                    "p_value": float(p_value) if not pd.isna(p_value) else np.nan,
                    "cohen_dz": float(cohen_dz) if not pd.isna(cohen_dz) else np.nan,
                }
            )

            formatted_matrix_dataframe.loc[row_label, column_label] = (
                f"Δ={mean_difference:+.2f}\n"
                f"dz={cohen_dz:+.2f}\n"
                f"p={_format_p_value_for_manuscript_table(p_value)}"
            )

    formatted_header_labels = []
    for _, summary_row in category_summary_dataframe.iterrows():
        formatted_header_labels.append(
            f"{summary_row['category_label']}\n"
            f"μ={summary_row['mean']:.2f}"
        )

    formatted_matrix_dataframe.index = formatted_header_labels
    formatted_matrix_dataframe.columns = formatted_header_labels

    pairwise_long_dataframe = pd.DataFrame(pairwise_rows)

    return formatted_matrix_dataframe, pairwise_long_dataframe


def compute_manuscript_table_1_participant_counts(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Compute manuscript Table 1: participant counts by randomization condition.

    Arguments:
        • general_settings: GeneralSettings
        • cleaned_dataframe: pd.DataFrame | None
        • force_rebuild: bool | None

    Returns:
        • pd.DataFrame
            - Manuscript-facing participant count table.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    table_names = general_settings["filing"]["table_names"]

    table_extraction = _load_table_dataframe_from_tables_folder(
        general_settings=general_settings,
        table_file_name=table_names["table_1_participant_counts"],
        force_rebuild=force_rebuild,
    )
    if table_extraction["success"]:
        return table_extraction["dataframe"]
    if table_extraction["error"]:
        raise Exception(table_extraction["message"])

    if cleaned_dataframe is None:
        cleaned_dataframe = load_or_build_cleaned_dataframe(
            general_settings=general_settings,
            force_rebuild=False,
        )
    else:
        cleaned_dataframe = cleaned_dataframe.copy()

    def count_subset(subset_dataframe: pd.DataFrame) -> dict[str, Any]:
        return {
            "Subjects": "Included" if subset_dataframe["included"].eq(True).all() else "Everyone",
            "CC": int((subset_dataframe["vignette_condition_position_1"] == "CC").sum()),
            "CH": int((subset_dataframe["vignette_condition_position_1"] == "CH").sum()),
            "DIV": int((subset_dataframe["vignette_condition_position_1"] == "DIV").sum()),
            "Firework": int((subset_dataframe["story_condition"] == "firework").sum()),
            "Trolley": int((subset_dataframe["story_condition"] == "trolley").sum()),
            "High": int((subset_dataframe["load_condition"] == "high").sum()),
            "Low": int((subset_dataframe["load_condition"] == "low").sum()),
            "Total": int(subset_dataframe.shape[0]),
        }

    dataframe_included_only = cleaned_dataframe.loc[cleaned_dataframe["included"] == True].copy()  # noqa: E712
    dataframe_everyone = cleaned_dataframe.copy()

    dataframe_table_1 = pd.DataFrame(
        [
            count_subset(dataframe_included_only),
            count_subset(dataframe_everyone),
        ]
    )

    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table_1,
        general_settings=general_settings,
        table_file_name=table_names["table_1_participant_counts"],
    )

    return dataframe_table_1


def _compute_model_means_from_dataframe(cleaned_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Compute blame means per model × story condition × vignette condition directly from the cleaned dataframe.

    Row order mirrors Table 2: CC Proximate, CH Distal, DIV Distal, CC Distal, CH Proximate.
    Proximate rows show NaN for Blame First Vignette (no between-subjects first-vignette design for proximate).
    Returns two stacked blocks: Included participants first, then Everyone (all finishers).

    Arguments:
        • cleaned_dataframe: pd.DataFrame
            - Must contain race_compact, included, story_condition, vignette_condition_position_1,
              first_vignette_distal_blame, distal_blame_cc/ch/div, proximate_blame_cc/ch.

    Returns:
        • pd.DataFrame with columns: Inclusion, Model, Story Condition, Condition,
          Blame First Vignette, Blame All Vignettes.
    """
    row_specs = [
        {"condition_display": "CC Proximate", "agent_role": "proximate", "condition_code": "CC"},
        {"condition_display": "CH Distal",    "agent_role": "distal",    "condition_code": "CH"},
        {"condition_display": "DIV Distal",   "agent_role": "distal",    "condition_code": "DIV"},
        {"condition_display": "CC Distal",    "agent_role": "distal",    "condition_code": "CC"},
        {"condition_display": "CH Proximate", "agent_role": "proximate", "condition_code": "CH"},
    ]
    all_vignettes_column_map = {
        ("distal",    "CC"):  "distal_blame_cc",
        ("distal",    "CH"):  "distal_blame_ch",
        ("distal",    "DIV"): "distal_blame_div",
        ("proximate", "CC"):  "proximate_blame_cc",
        ("proximate", "CH"):  "proximate_blame_ch",
    }
    story_condition_display_map = {"pooled": "Pooled", "firework": "Firework", "trolley": "Trolley"}

    def _rows_for_subset(df: pd.DataFrame, inclusion_label: str) -> list[dict]:
        rows = []
        for model_name in sorted(df["race_compact"].dropna().unique()):
            model_df = df.loc[df["race_compact"] == model_name]
            for story_condition_key, story_display in story_condition_display_map.items():
                story_df = model_df if story_condition_key == "pooled" else model_df.loc[model_df["story_condition"] == story_condition_key]
                for spec in row_specs:
                    condition_code = spec["condition_code"]
                    agent_role = spec["agent_role"]
                    all_col = all_vignettes_column_map[(agent_role, condition_code)]
                    all_mean = float(story_df[all_col].mean()) if len(story_df) > 0 else np.nan
                    if agent_role == "distal":
                        first_df = story_df.loc[story_df["vignette_condition_position_1"] == condition_code]
                        first_mean = float(first_df["first_vignette_distal_blame"].mean()) if len(first_df) > 0 else np.nan
                    else:
                        first_mean = np.nan
                    rows.append({
                        "Inclusion": inclusion_label,
                        "Model": model_name,
                        "Story Condition": story_display,
                        "Condition": spec["condition_display"],
                        "Blame First Vignette": round(first_mean, 2) if pd.notna(first_mean) else np.nan,
                        "Blame All Vignettes": round(all_mean, 2) if pd.notna(all_mean) else np.nan,
                    })
        return rows

    included_df = cleaned_dataframe.loc[cleaned_dataframe["included"] == True].copy()  # noqa: E712
    all_rows = _rows_for_subset(included_df, "Included") + _rows_for_subset(cleaned_dataframe.copy(), "Everyone")
    return pd.DataFrame(all_rows, columns=["Inclusion", "Model", "Story Condition", "Condition", "Blame First Vignette", "Blame All Vignettes"])


def compute_manuscript_table_2_mean_scale_values_by_dv_and_condition(
    general_settings: GeneralSettings,
    force_rebuild: bool | None = None,
    inclusion_filter: str = "included_only",
    save_pretty_multilevel_version: bool = True,
    include_medians: bool = False,
    include_std: bool = False,
    group_by_model: bool = False,
    cleaned_dataframe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute a descriptive table of mean scale values by DV and condition.

    The table is designed to answer the editor's request that the paper report
    mean answers on the core rating items, not only contrasts.

    When include_medians=False and include_std=False, the table is pivoted wide
    so that each DV appears side-by-side for each story condition and condition.

    When include_medians=True or include_std=True, the table is long-format with
    leading Story Condition, Vignette, and Condition columns, followed by DV-level
    mean columns and optionally median/SD columns.

    Columns (wide format, include_medians=False and include_std=False):
        • Story Condition
        • Condition
        • Blame First Vignette
        • Wrong First Vignette
        • Punish First Vignette
        • Blame All Vignettes
        • Wrong All Vignettes
        • Punish All Vignettes

    Columns (long format, include_medians=True or include_std=True):
        • Story Condition
        • Vignette
        • Condition
        • Blame Mean
        • Wrong Mean
        • Punish Mean
        • [Blame Median, Wrong Median, Punish Median]   (if include_medians=True)
        • [Blame Std, Wrong Std, Punish Std]            (if include_std=True)

    Notes:
        • "First vignette" is only applicable to the distal-agent rows, because proximate
          ratings were collected later rather than as a first-impression first-vignette measure.
        • The function saves:
            1. a flat CSV suitable for code/reloading
            2. optionally, a pretty CSV with a two-row multilevel header for manuscript use

    Arguments:
        • general_settings: GeneralSettings
            - Master project settings dictionary.
        • force_rebuild: bool | None
            - Whether to rebuild even if the table already exists.
        • inclusion_filter: str
            - Usually "included_only" for the manuscript main text.
            - Also supports "all_finishers".
        • save_pretty_multilevel_version: bool
            - If True, also save a pretty CSV with a two-level header.
        • include_medians: bool
            - If True, include median columns in the long-format table.
        • include_std: bool
            - If True, include standard deviation columns in the long-format table.

    Returns:
        • pd.DataFrame
            - Flat version of the descriptive table.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    if group_by_model:
        resolved_dataframe = cleaned_dataframe if cleaned_dataframe is not None else load_or_build_cleaned_dataframe(general_settings)
        return _compute_model_means_from_dataframe(resolved_dataframe)

    valid_inclusion_filters = {"included_only", "all_finishers"}
    if inclusion_filter not in valid_inclusion_filters:
        raise ValueError(
            f"inclusion_filter must be one of {sorted(valid_inclusion_filters)}, not {inclusion_filter!r}."
        )

    table_names = general_settings["filing"]["table_names"]
    table_file_name = table_names["table_2_means_by_dv_and_condition"]

    table_extraction = _load_table_dataframe_from_tables_folder(
        general_settings=general_settings,
        table_file_name=table_file_name,
        force_rebuild=force_rebuild,
    )
    if table_extraction["success"]:
        return table_extraction["dataframe"]
    if table_extraction["error"]:
        raise Exception(table_extraction["message"])

    group_summaries_dataframe = compute_group_summaries(
        general_settings=general_settings,
        force_rebuild=force_rebuild,
    ).copy()

    inclusion_filter_column_name = (
        "participants" if "participants" in group_summaries_dataframe.columns else "inclusion_filter"
    )
    story_condition_column_name = (
        "story_family" if "story_family" in group_summaries_dataframe.columns else "story_condition"
    )
    analysis_scope_column_name = "analysis_scope"

    def normalize_inclusion_filter_for_group_summaries(value: str) -> set[str]:
        if value in {"included_only", "included"}:
            return {"included_only", "included"}
        if value == "all_finishers":
            return {"all_finishers"}
        return {value}

    group_summaries_dataframe = group_summaries_dataframe.loc[
        group_summaries_dataframe[inclusion_filter_column_name].isin(
            normalize_inclusion_filter_for_group_summaries(inclusion_filter)
        )
        & (group_summaries_dataframe["load_condition"] == "pooled")
    ].copy()

    story_condition_display_map = {
        "pooled": "Pooled",
        "firework": "Firework",
        "trolley": "Trolley",
    }

    dv_short_map = {
        "blame": "Blame",
        "wrong": "Wrong",
        "punish": "Punish",
    }

    row_specifications: list[dict[str, str]] = [
        {"dv": "blame", "condition_code": "CC", "condition_display": "CC Distal", "agent_role": "distal"},
        {"dv": "blame", "condition_code": "CH", "condition_display": "CH Distal", "agent_role": "distal"},
        {"dv": "blame", "condition_code": "DIV", "condition_display": "DIV Distal", "agent_role": "distal"},
        {"dv": "blame", "condition_code": "CC", "condition_display": "CC Proximate", "agent_role": "proximate"},
        {"dv": "blame", "condition_code": "CH", "condition_display": "CH Proximate", "agent_role": "proximate"},

        {"dv": "wrong", "condition_code": "CC", "condition_display": "CC Distal", "agent_role": "distal"},
        {"dv": "wrong", "condition_code": "CH", "condition_display": "CH Distal", "agent_role": "distal"},
        {"dv": "wrong", "condition_code": "DIV", "condition_display": "DIV Distal", "agent_role": "distal"},
        {"dv": "wrong", "condition_code": "CC", "condition_display": "CC Proximate", "agent_role": "proximate"},
        {"dv": "wrong", "condition_code": "CH", "condition_display": "CH Proximate", "agent_role": "proximate"},

        {"dv": "punish", "condition_code": "CC", "condition_display": "CC Distal", "agent_role": "distal"},
        {"dv": "punish", "condition_code": "CH", "condition_display": "CH Distal", "agent_role": "distal"},
        {"dv": "punish", "condition_code": "DIV", "condition_display": "DIV Distal", "agent_role": "distal"},
        {"dv": "punish", "condition_code": "CC", "condition_display": "CC Proximate", "agent_role": "proximate"},
        {"dv": "punish", "condition_code": "CH", "condition_display": "CH Proximate", "agent_role": "proximate"},
    ]

    story_condition_sort_order = {
        "Pooled": 0,
        "Firework": 1,
        "Trolley": 2,
    }

    condition_sort_order = {
        "CC Proximate": 0,
        "CH Distal": 1,
        "DIV Distal": 2,
        "CC Distal": 3,
        "CH Proximate": 4,
    }

    def extract_summary_row(
        analysis_scope: str,
        story_condition: str,
        agent_role: str,
        dv_value: str,
        condition_code: str,
    ) -> pd.Series | None:
        valid_analysis_scope_values = {analysis_scope}
        if analysis_scope == "within_subjects_all_vignettes":
            valid_analysis_scope_values.add("within_subjects")
        if analysis_scope == "between_subjects_first_vignette":
            valid_analysis_scope_values.add("between_subjects")

        matching_rows = group_summaries_dataframe.loc[
            group_summaries_dataframe[analysis_scope_column_name].isin(valid_analysis_scope_values)
            & (group_summaries_dataframe[story_condition_column_name] == story_condition)
            & (group_summaries_dataframe["agent_role"] == agent_role)
            & (group_summaries_dataframe["dv"] == dv_value)
            & (group_summaries_dataframe["condition"] == condition_code)
        ].copy()

        if matching_rows.shape[0] == 0:
            return None
        if matching_rows.shape[0] != 1:
            raise Exception(
                f"Expected exactly one group summary row for analysis_scope={analysis_scope!r}, "
                f"story_condition={story_condition!r}, agent_role={agent_role!r}, "
                f"dv={dv_value!r}, condition={condition_code!r}, "
                f"but found {matching_rows.shape[0]} rows."
            )

        return matching_rows.iloc[0]

    intermediate_rows: list[dict[str, Any]] = []

    for story_condition in ["pooled", "firework", "trolley"]:
        story_condition_display = story_condition_display_map[story_condition]

        for row_specification in row_specifications:
            dv_value = row_specification["dv"]
            agent_role = row_specification["agent_role"]
            condition_code = row_specification["condition_code"]
            dv_short = dv_short_map[dv_value]

            all_vignettes_summary_row = extract_summary_row(
                analysis_scope="within_subjects_all_vignettes",
                story_condition=story_condition,
                agent_role=agent_role,
                dv_value=dv_value,
                condition_code=condition_code,
            )

            if agent_role == "distal":
                first_vignette_summary_row = extract_summary_row(
                    analysis_scope="between_subjects_first_vignette",
                    story_condition=story_condition,
                    agent_role="distal",
                    dv_value=dv_value,
                    condition_code=condition_code,
                )
                first_vignette_mean = float(first_vignette_summary_row["mean"]) if first_vignette_summary_row is not None else np.nan
                first_vignette_median = float(first_vignette_summary_row["median"]) if first_vignette_summary_row is not None else np.nan
                first_vignette_std = float(first_vignette_summary_row["std"]) if first_vignette_summary_row is not None else np.nan
            else:
                first_vignette_mean = np.nan
                first_vignette_median = np.nan
                first_vignette_std = np.nan

            all_vignettes_mean = float(all_vignettes_summary_row["mean"]) if all_vignettes_summary_row is not None else np.nan
            all_vignettes_median = float(all_vignettes_summary_row["median"]) if all_vignettes_summary_row is not None else np.nan
            all_vignettes_std = float(all_vignettes_summary_row["std"]) if all_vignettes_summary_row is not None else np.nan

            intermediate_rows.append(
                {
                    "Story Condition": story_condition_display,
                    "story_condition_sort": story_condition_sort_order[story_condition_display],
                    "dv": dv_value,
                    "dv_short": dv_short,
                    "Condition": row_specification["condition_display"],
                    "condition_sort": condition_sort_order[row_specification["condition_display"]],
                    "first_mean": first_vignette_mean,
                    "first_median": first_vignette_median,
                    "first_std": first_vignette_std,
                    "all_mean": all_vignettes_mean,
                    "all_median": all_vignettes_median,
                    "all_std": all_vignettes_std,
                }
            )

    intermediate_dataframe = pd.DataFrame(intermediate_rows)

    use_wide_format = not include_medians and not include_std

    if use_wide_format:
        wide_rows: list[dict[str, Any]] = []

        for story_condition_display, _ in sorted(story_condition_sort_order.items(), key=lambda item: item[1]):
            for condition_display, _ in sorted(condition_sort_order.items(), key=lambda item: item[1]):
                condition_group = intermediate_dataframe.loc[
                    (intermediate_dataframe["Story Condition"] == story_condition_display)
                    & (intermediate_dataframe["Condition"] == condition_display)
                ].copy()

                wide_row: dict[str, Any] = {
                    "Story Condition": story_condition_display,
                    "Condition": condition_display,
                }

                for _, record in condition_group.iterrows():
                    dv_short = record["dv_short"]
                    wide_row[f"{dv_short} First Vignette"] = record["first_mean"]
                    wide_row[f"{dv_short} All Vignettes"] = record["all_mean"]

                wide_rows.append(wide_row)

        column_order = [
            "Story Condition",
            "Condition",
            "Blame First Vignette",
            "Wrong First Vignette",
            "Punish First Vignette",
            "Blame All Vignettes",
            "Wrong All Vignettes",
            "Punish All Vignettes",
        ]
        dataframe_table_flat = pd.DataFrame(wide_rows)[column_order]

    else:
        long_rows: list[dict[str, Any]] = []

        for story_condition_display, _ in sorted(story_condition_sort_order.items(), key=lambda item: item[1]):
            for vignette_label, mean_key, median_key, std_key in [
                ("First", "first_mean", "first_median", "first_std"),
                ("All", "all_mean", "all_median", "all_std"),
            ]:
                for condition_display, _ in sorted(condition_sort_order.items(), key=lambda item: item[1]):
                    condition_group = intermediate_dataframe.loc[
                        (intermediate_dataframe["Story Condition"] == story_condition_display)
                        & (intermediate_dataframe["Condition"] == condition_display)
                    ].copy()

                    long_row: dict[str, Any] = {
                        "Story Condition": story_condition_display,
                        "Vignette": vignette_label,
                        "Condition": condition_display,
                    }

                    for _, record in condition_group.iterrows():
                        dv_short = record["dv_short"]
                        long_row[f"{dv_short} Mean"] = record[mean_key]
                        if include_medians:
                            long_row[f"{dv_short} Median"] = record[median_key]
                        if include_std:
                            long_row[f"{dv_short} Std"] = record[std_key]

                    long_rows.append(long_row)

        mean_columns = ["Blame Mean", "Wrong Mean", "Punish Mean"]
        median_columns = ["Blame Median", "Wrong Median", "Punish Median"] if include_medians else []
        std_columns = ["Blame Std", "Wrong Std", "Punish Std"] if include_std else []

        column_order = ["Story Condition", "Vignette", "Condition"] + mean_columns + median_columns + std_columns
        dataframe_table_flat = pd.DataFrame(long_rows)[column_order]

    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table_flat,
        general_settings=general_settings,
        table_file_name=table_file_name,
    )

    if save_pretty_multilevel_version:
        pretty_dataframe = dataframe_table_flat.copy()

        float_columns = [column_name for column_name in pretty_dataframe.columns if column_name not in {"Story Condition", "Vignette", "Condition"}]
        for column_name in float_columns:
            pretty_dataframe[column_name] = pretty_dataframe[column_name].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.2f}"
            )

        if use_wide_format:
            pretty_dataframe.columns = pd.MultiIndex.from_tuples(
                [
                    ("", "Story Condition"),
                    ("", "Condition"),
                    ("First Vignette", "Blame"),
                    ("First Vignette", "Wrong"),
                    ("First Vignette", "Punish"),
                    ("All Vignettes", "Blame"),
                    ("All Vignettes", "Wrong"),
                    ("All Vignettes", "Punish"),
                ]
            )
        else:
            mean_tuples = [("Mean", dv) for dv in ["Blame", "Wrong", "Punish"]]
            median_tuples = [("Median", dv) for dv in ["Blame", "Wrong", "Punish"]] if include_medians else []
            std_tuples = [("Std", dv) for dv in ["Blame", "Wrong", "Punish"]] if include_std else []

            pretty_dataframe.columns = pd.MultiIndex.from_tuples(
                [("", "Story Condition"), ("", "Vignette"), ("", "Condition")] + mean_tuples + median_tuples + std_tuples
            )

        pretty_file_name = table_file_name.replace(".csv", "_Pretty.csv")
        pretty_file_path = general_settings["filing"]["file_paths"]["tables"] / pretty_file_name
        pretty_dataframe.to_csv(pretty_file_path, index=False, encoding="utf-8-sig")

    return dataframe_table_flat


def compute_manuscript_table_3_primary_distal_blame_contrasts(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Compute manuscript Table 3: primary distal blame contrasts.

    Arguments:
        • general_settings: GeneralSettings
        • cleaned_dataframe: pd.DataFrame | None
        • force_rebuild: bool | None

    Returns:
        • pd.DataFrame
            - Manuscript-facing primary contrast table.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    table_names = general_settings["filing"]["table_names"]
    table_name_key = (
        "table_3_primary_distal_blame_contrasts"
        if "table_3_primary_distal_blame_contrasts" in table_names
        else "table_3_primary_clark_blame_contrasts"
    )

    table_extraction = _load_table_dataframe_from_tables_folder(
        general_settings=general_settings,
        table_file_name=table_names[table_name_key],
        force_rebuild=force_rebuild,
    )
    if table_extraction["success"]:
        return table_extraction["dataframe"]
    if table_extraction["error"]:
        raise Exception(table_extraction["message"])

    if cleaned_dataframe is None:
        cleaned_dataframe = load_or_build_cleaned_dataframe(
            general_settings=general_settings,
            force_rebuild=False,
        )
    else:
        cleaned_dataframe = cleaned_dataframe.copy()

    dataframe_tests = run_confirmatory_and_exploratory_tests(
        general_settings=general_settings,
        force_rebuild=force_rebuild,
    )

    contrast_display_map = {
        "CH - CC": "BCH - BCC",
        "DIV - CC": "BDIV - BCC",
    }

    table_rows: list[dict[str, Any]] = []

    for inclusion_filter_value, inclusion_display_label in [
        ("included_only", "Included"),
        ("all_finishers", "Everyone"),
    ]:
        for contrast_type_value, contrast_display_label in contrast_display_map.items():
            between_row = _extract_exact_test_row_from_test_csv(
                dataframe_tests=dataframe_tests,
                inclusion_filter=inclusion_filter_value,
                story_family="pooled",
                load_condition="pooled",
                design="between_subjects_first_vignette",
                dv="blame",
                agent_role="distal",
                contrast_type=contrast_type_value,
                analysis_family=("confirmatory" if inclusion_filter_value == "included_only" else "exploratory"),
            )

            displayed_between_subject_p_value = _extract_reported_p_value_from_test_row(
                row_series=between_row,
                general_settings=general_settings,
                use_holm_if_available=(inclusion_filter_value == "included_only"),
            )

            table_rows.append(
                {
                    "Inclusion": inclusion_display_label,
                    "Design": "Between-subj",
                    "Contrast": contrast_display_label,
                    "Estimate": round(_extract_estimate_from_test_row(between_row), 2),
                    "95% CI": _format_ci_for_manuscript_table(
                        between_row["ci95_lower"],
                        between_row["ci95_upper"],
                    ),
                    "p-value": _format_p_value_for_manuscript_table(displayed_between_subject_p_value),
                    "p_value_two_tailed": float(between_row["p_value_two_tailed"]),
                    "p_value_one_tailed": float(between_row["p_value_one_tailed"]),
                    "p_value_holm": (
                        float(between_row["p_value_holm"])
                        if pd.notna(between_row["p_value_holm"])
                        else np.nan
                    ),
                    "p_value_displayed": float(displayed_between_subject_p_value),
                    "p_value_correction": (
                        "Holm corrected"
                        if inclusion_filter_value == "included_only" and pd.notna(between_row["p_value_holm"])
                        else "Unadjusted"
                    ),
                    "source_file": "tests.csv",
                    "meaning": between_row["notes"],
                }
            )

            within_row = _extract_exact_test_row_from_test_csv(
                dataframe_tests=dataframe_tests,
                inclusion_filter=inclusion_filter_value,
                story_family="pooled",
                load_condition="pooled",
                design="within_subjects_all_vignettes",
                dv="blame",
                agent_role="distal",
                contrast_type=contrast_type_value,
                analysis_family="exploratory",
            )

            displayed_within_subject_p_value = _extract_reported_p_value_from_test_row(
                row_series=within_row,
                general_settings=general_settings,
                use_holm_if_available=False,
            )

            table_rows.append(
                {
                    "Inclusion": inclusion_display_label,
                    "Design": "Within-subj",
                    "Contrast": contrast_display_label,
                    "Estimate": round(_extract_estimate_from_test_row(within_row), 2),
                    "95% CI": _format_ci_for_manuscript_table(
                        within_row["ci95_lower"],
                        within_row["ci95_upper"],
                    ),
                    "p-value": _format_p_value_for_manuscript_table(displayed_within_subject_p_value),
                    "p_value_two_tailed": float(within_row["p_value_two_tailed"]),
                    "p_value_one_tailed": float(within_row["p_value_one_tailed"]),
                    "p_value_holm": np.nan,
                    "p_value_displayed": float(displayed_within_subject_p_value),
                    "p_value_correction": "Unadjusted",
                    "source_file": "tests.csv",
                    "meaning": within_row["notes"],
                }
            )

    dataframe_table_3 = pd.DataFrame(table_rows)

    inclusion_sort_order = {"Included": 0, "Everyone": 1}
    design_sort_order = {"Between-subj": 0, "Within-subj": 1}
    contrast_sort_order = {"BCH - BCC": 0, "BDIV - BCC": 1}

    dataframe_table_3["inclusion_sort_order"] = dataframe_table_3["Inclusion"].map(inclusion_sort_order)
    dataframe_table_3["design_sort_order"] = dataframe_table_3["Design"].map(design_sort_order)
    dataframe_table_3["contrast_sort_order"] = dataframe_table_3["Contrast"].map(contrast_sort_order)

    dataframe_table_3 = dataframe_table_3.sort_values(
        by=["inclusion_sort_order", "design_sort_order", "contrast_sort_order"],
        kind="stable",
    ).drop(columns=["inclusion_sort_order", "design_sort_order", "contrast_sort_order"])

    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table_3,
        general_settings=general_settings,
        table_file_name=table_names[table_name_key],
    )

    return dataframe_table_3


def compute_manuscript_table_4_story_specific_distal_blame_contrasts(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Compute manuscript Table 4: story-specific distal blame contrasts.

    Arguments:
        • general_settings: GeneralSettings
        • cleaned_dataframe: pd.DataFrame | None
        • force_rebuild: bool | None

    Returns:
        • pd.DataFrame
            - Manuscript-facing story-specific contrast table.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    table_names = general_settings["filing"]["table_names"]
    table_name_key = (
        "table_4_story_specific_distal_blame_contrasts"
        if "table_4_story_specific_distal_blame_contrasts" in table_names
        else "table_4_story_specific_clark_blame_contrasts"
    )

    table_extraction = _load_table_dataframe_from_tables_folder(
        general_settings=general_settings,
        table_file_name=table_names[table_name_key],
        force_rebuild=force_rebuild,
    )
    if table_extraction["success"]:
        return table_extraction["dataframe"]
    if table_extraction["error"]:
        raise Exception(table_extraction["message"])

    if cleaned_dataframe is None:
        cleaned_dataframe = load_or_build_cleaned_dataframe(
            general_settings=general_settings,
            force_rebuild=False,
        )
    else:
        cleaned_dataframe = cleaned_dataframe.copy()

    dataframe_tests = run_confirmatory_and_exploratory_tests(
        general_settings=general_settings,
        force_rebuild=force_rebuild,
    )

    story_display_map = {
        "firework": "Firework",
        "trolley": "Trolley",
    }
    contrast_display_map = {
        "CH - CC": "BCH - BCC",
        "DIV - CC": "BDIV - BCC",
    }

    table_rows: list[dict[str, Any]] = []

    for story_family_value, story_display_label in story_display_map.items():
        for contrast_type_value, contrast_display_label in contrast_display_map.items():
            between_row = _extract_exact_test_row_from_test_csv(
                dataframe_tests=dataframe_tests,
                inclusion_filter="included_only",
                story_family=story_family_value,
                load_condition="pooled",
                design="between_subjects_first_vignette",
                dv="blame",
                agent_role="distal",
                contrast_type=contrast_type_value,
                analysis_family="exploratory",
            )

            displayed_between_subject_p_value = _extract_reported_p_value_from_test_row(
                row_series=between_row,
                general_settings=general_settings,
                use_holm_if_available=False,
            )

            table_rows.append(
                {
                    "Story": story_display_label,
                    "Design": "Between-subj",
                    "Contrast": contrast_display_label,
                    "Estimate": round(_extract_estimate_from_test_row(between_row), 2),
                    "95% CI": _format_ci_for_manuscript_table(
                        between_row["ci95_lower"],
                        between_row["ci95_upper"],
                    ),
                    "p-value": _format_p_value_for_manuscript_table(displayed_between_subject_p_value),
                    "p_value_two_tailed": float(between_row["p_value_two_tailed"]),
                    "p_value_one_tailed": float(between_row["p_value_one_tailed"]),
                    "p_value_displayed": float(displayed_between_subject_p_value),
                    "source_file": "tests.csv",
                    "meaning": between_row["notes"],
                }
            )

            within_row = _extract_exact_test_row_from_test_csv(
                dataframe_tests=dataframe_tests,
                inclusion_filter="included_only",
                story_family=story_family_value,
                load_condition="pooled",
                design="within_subjects_all_vignettes",
                dv="blame",
                agent_role="distal",
                contrast_type=contrast_type_value,
                analysis_family="exploratory",
            )

            displayed_within_subject_p_value = _extract_reported_p_value_from_test_row(
                row_series=within_row,
                general_settings=general_settings,
                use_holm_if_available=False,
            )

            table_rows.append(
                {
                    "Story": story_display_label,
                    "Design": "Within-subj",
                    "Contrast": contrast_display_label,
                    "Estimate": round(_extract_estimate_from_test_row(within_row), 2),
                    "95% CI": _format_ci_for_manuscript_table(
                        within_row["ci95_lower"],
                        within_row["ci95_upper"],
                    ),
                    "p-value": _format_p_value_for_manuscript_table(displayed_within_subject_p_value),
                    "p_value_two_tailed": float(within_row["p_value_two_tailed"]),
                    "p_value_one_tailed": float(within_row["p_value_one_tailed"]),
                    "p_value_displayed": float(displayed_within_subject_p_value),
                    "source_file": "tests.csv",
                    "meaning": within_row["notes"],
                }
            )

    dataframe_table_4 = pd.DataFrame(table_rows)

    story_sort_order = {"Firework": 0, "Trolley": 1}
    design_sort_order = {"Between-subj": 0, "Within-subj": 1}
    contrast_sort_order = {"BCH - BCC": 0, "BDIV - BCC": 1}

    dataframe_table_4["story_sort_order"] = dataframe_table_4["Story"].map(story_sort_order)
    dataframe_table_4["design_sort_order"] = dataframe_table_4["Design"].map(design_sort_order)
    dataframe_table_4["contrast_sort_order"] = dataframe_table_4["Contrast"].map(contrast_sort_order)

    dataframe_table_4 = dataframe_table_4.sort_values(
        by=["story_sort_order", "design_sort_order", "contrast_sort_order"],
        kind="stable",
    ).drop(columns=["story_sort_order", "design_sort_order", "contrast_sort_order"])

    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table_4,
        general_settings=general_settings,
        table_file_name=table_names[table_name_key],
    )

    return dataframe_table_4


def compute_manuscript_table_5_two_alternative_forced_choice_distribution(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Compute manuscript Table 5: the 2AFC response distribution table.

    Arguments:
        • general_settings: GeneralSettings
        • cleaned_dataframe: pd.DataFrame | None
        • force_rebuild: bool | None

    Returns:
        • pd.DataFrame
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    table_names = general_settings["filing"]["table_names"]

    table_extraction = _load_table_dataframe_from_tables_folder(
        general_settings=general_settings,
        table_file_name=table_names["table_5_two_alternative_forced_choice_distribution"],
        force_rebuild=force_rebuild,
    )
    if table_extraction["success"]:
        return table_extraction["dataframe"]
    if table_extraction["error"]:
        raise Exception(table_extraction["message"])

    "Load the pivoted 2AFC counts (table_form=True always returns the wide form)"
    dataframe_twoafc_counts = compute_twoafc_counts(
        general_settings=general_settings,
        force_rebuild=force_rebuild,
        table_form=True,
    )

    pivoted_columns = {"operator", "Bill ? Clark", "CH ? CC", "DIV ? CC"}
    long_columns = {"inclusion_filter", "story_condition", "comparison", "operator", "count"}

    if pivoted_columns.issubset(set(dataframe_twoafc_counts.columns)):
        "Already in pivoted form — filter to included_only / all and select the four columns"
        dataframe_table_5 = (
            dataframe_twoafc_counts.loc[
                (dataframe_twoafc_counts["inclusion_filter"] == "included_only")
                & (dataframe_twoafc_counts["story_condition"] == "pooled")
            ]
            [["operator", "Bill ? Clark", "CH ? CC", "DIV ? CC"]]
            .copy()
            .rename(columns={"operator": "Operator"})
        )

    elif long_columns.issubset(set(dataframe_twoafc_counts.columns)):
        "Fallback: received long form — pivot it manually"
        dataframe_twoafc_counts = dataframe_twoafc_counts.loc[
            (dataframe_twoafc_counts["inclusion_filter"] == "included_only")
            & (dataframe_twoafc_counts["story_condition"] == "pooled")
        ].copy()

        dataframe_table_5 = (
            dataframe_twoafc_counts
            .pivot(index="operator", columns="comparison", values="count")
            .reset_index()
            .rename(columns={"operator": "Operator"})
        )

        desired_column_order = ["Operator", "Bill ? Clark", "CH ? CC", "DIV ? CC"]
        dataframe_table_5 = dataframe_table_5[desired_column_order].copy()

    else:
        raise Exception(
            "compute_twoafc_counts returned a dataframe that does not look like either the "
            "long-form or the manuscript-ready pivoted form. "
            f"Columns found: {list(dataframe_twoafc_counts.columns)}"
        )

    operator_sort_order = {">": 0, "≥": 1, "≤": 2, "<": 3}
    dataframe_table_5["operator_sort_order"] = dataframe_table_5["Operator"].map(operator_sort_order)
    dataframe_table_5 = dataframe_table_5.sort_values(
        by="operator_sort_order",
        kind="stable",
    ).drop(columns=["operator_sort_order"])

    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table_5,
        general_settings=general_settings,
        table_file_name=table_names["table_5_two_alternative_forced_choice_distribution"],
    )

    return dataframe_table_5


def compute_supplementary_table_6_within_subject_pairwise_blame_matrix(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute supplementary Table 6: the within-subject pairwise blame matrix, plus its long-form backing table.

    Arguments:
        • general_settings: GeneralSettings
        • cleaned_dataframe: pd.DataFrame | None
        • force_rebuild: bool | None

    Returns:
        • dict[str, pd.DataFrame]
            - Dictionary containing:
                "formatted_matrix_dataframe"
                "pairwise_long_dataframe"
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    table_names = general_settings["filing"]["table_names"]

    table_extraction = _load_table_dataframe_from_tables_folder(
        general_settings=general_settings,
        table_file_name=table_names["table_6_within_subject_pairwise_blame_matrix"],
        force_rebuild=force_rebuild,
    )
    long_table_extraction = _load_table_dataframe_from_tables_folder(
        general_settings=general_settings,
        table_file_name=table_names["table_6_within_subject_pairwise_blame_long"],
        force_rebuild=force_rebuild,
    )

    if table_extraction["success"] and long_table_extraction["success"]:
        return {
            "formatted_matrix_dataframe": table_extraction["dataframe"],
            "pairwise_long_dataframe": long_table_extraction["dataframe"],
        }
    if table_extraction["error"]:
        raise Exception(table_extraction["message"])
    if long_table_extraction["error"]:
        raise Exception(long_table_extraction["message"])

    if cleaned_dataframe is None:
        cleaned_dataframe = load_or_build_cleaned_dataframe(
            general_settings=general_settings,
            force_rebuild=False,
        )
    else:
        cleaned_dataframe = cleaned_dataframe.copy()

    formatted_matrix_dataframe, pairwise_long_dataframe = _build_within_subject_pairwise_blame_matrix(
        cleaned_dataframe=cleaned_dataframe,
        only_included_participants=True,
    )

    _save_table_dataframe_to_tables_folder(
        dataframe_table=formatted_matrix_dataframe.reset_index().rename(columns={"index": "Row / Column"}),
        general_settings=general_settings,
        table_file_name=table_names["table_6_within_subject_pairwise_blame_matrix"],
    )
    _save_table_dataframe_to_tables_folder(
        dataframe_table=pairwise_long_dataframe,
        general_settings=general_settings,
        table_file_name=table_names["table_6_within_subject_pairwise_blame_long"],
    )

    return {
        "formatted_matrix_dataframe": formatted_matrix_dataframe,
        "pairwise_long_dataframe": pairwise_long_dataframe,
    }


def compute_supplementary_table_7_cognitive_load_blame_contrasts(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Compute supplementary Table 7: cognitive-load contrasts for the core blame deltas.

    Arguments:
        • general_settings: GeneralSettings
        • cleaned_dataframe: pd.DataFrame | None
        • force_rebuild: bool | None

    Returns:
        • pd.DataFrame
            - Manuscript-facing cognitive-load table.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    table_names = general_settings["filing"]["table_names"]

    table_extraction = _load_table_dataframe_from_tables_folder(
        general_settings=general_settings,
        table_file_name=table_names["table_7_cognitive_load_blame_contrasts"],
        force_rebuild=force_rebuild,
    )
    if table_extraction["success"]:
        return table_extraction["dataframe"]
    if table_extraction["error"]:
        raise Exception(table_extraction["message"])

    if cleaned_dataframe is None:
        cleaned_dataframe = load_or_build_cleaned_dataframe(
            general_settings=general_settings,
            force_rebuild=False,
        )
    else:
        cleaned_dataframe = cleaned_dataframe.copy()

    dataframe_tests = run_confirmatory_and_exploratory_tests(
        general_settings=general_settings,
        force_rebuild=force_rebuild,
    )

    load_display_map = {
        "pooled": "Pooled",
        "high": "High",
        "low": "Low",
    }
    contrast_display_map = {
        "CH - CC": "BCH - BCC",
        "DIV - CC": "BDIV - BCC",
    }

    table_rows: list[dict[str, Any]] = []

    for load_condition_value, load_display_label in load_display_map.items():
        for contrast_type_value, contrast_display_label in contrast_display_map.items():
            row_series = _extract_exact_test_row_from_test_csv(
                dataframe_tests=dataframe_tests,
                inclusion_filter="included_only",
                story_family="pooled",
                load_condition=load_condition_value,
                design="within_subjects_all_vignettes",
                dv="blame",
                agent_role="distal",
                contrast_type=contrast_type_value,
                analysis_family="exploratory",
            )

            displayed_p_value = _extract_reported_p_value_from_test_row(
                row_series=row_series,
                general_settings=general_settings,
                use_holm_if_available=False,
            )

            table_rows.append(
                {
                    "Cognitive Load": load_display_label,
                    "Contrast": contrast_display_label,
                    "Estimate": round(_extract_estimate_from_test_row(row_series), 2),
                    "95% CI": _format_ci_for_manuscript_table(
                        row_series["ci95_lower"],
                        row_series["ci95_upper"],
                    ),
                    "p-value": _format_p_value_for_manuscript_table(displayed_p_value),
                    "p_value_two_tailed": float(row_series["p_value_two_tailed"]),
                    "p_value_one_tailed": float(row_series["p_value_one_tailed"]),
                    "p_value_displayed": float(displayed_p_value),
                    "source_file": "tests.csv",
                    "meaning": row_series["notes"],
                }
            )

    dataframe_table_7 = pd.DataFrame(table_rows)

    load_sort_order = {"Pooled": 0, "High": 1, "Low": 2}
    contrast_sort_order = {"BCH - BCC": 0, "BDIV - BCC": 1}

    dataframe_table_7["load_sort_order"] = dataframe_table_7["Cognitive Load"].map(load_sort_order)
    dataframe_table_7["contrast_sort_order"] = dataframe_table_7["Contrast"].map(contrast_sort_order)

    dataframe_table_7 = dataframe_table_7.sort_values(
        by=["load_sort_order", "contrast_sort_order"],
        kind="stable",
    ).drop(columns=["load_sort_order", "contrast_sort_order"])

    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table_7,
        general_settings=general_settings,
        table_file_name=table_names["table_7_cognitive_load_blame_contrasts"],
    )

    return dataframe_table_7


def compute_supplementary_table_8_order_effects_summary(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Compute supplementary Table 8: order-effects summary.

    Arguments:
        • general_settings: GeneralSettings
        • cleaned_dataframe: pd.DataFrame | None
        • force_rebuild: bool | None

    Returns:
        • pd.DataFrame
            - Manuscript-facing order-effects summary table.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    use_integrated_models = general_settings["misc"]["use_integrated_models"]

    table_names = general_settings["filing"]["table_names"]

    base_table_file_name = table_names["table_8_order_effects_summary"]
    table_file_name = (
        base_table_file_name.replace(".csv", "_Integrated.csv")
        if use_integrated_models
        else base_table_file_name
    )

    table_extraction = _load_table_dataframe_from_tables_folder(
        general_settings=general_settings,
        table_file_name=table_file_name,
        force_rebuild=force_rebuild,
    )
    if table_extraction["success"]:
        return table_extraction["dataframe"]
    if table_extraction["error"]:
        raise Exception(table_extraction["message"])

    if cleaned_dataframe is None:
        cleaned_dataframe = load_or_build_cleaned_dataframe(
            general_settings=general_settings,
            force_rebuild=False,
        )
    else:
        cleaned_dataframe = cleaned_dataframe.copy()

    if use_integrated_models:
        dataframe_integrated_models = compute_integrated_distal_blame_results(
            general_settings=general_settings,
            cleaned_dataframe=cleaned_dataframe,
            force_rebuild=force_rebuild,
        )

        table_rows: list[dict[str, Any]] = []

        coefficient_rows = dataframe_integrated_models.loc[
            (dataframe_integrated_models["analysis_scope"] == "within_subjects_repeated_measures")
            & (dataframe_integrated_models["row_type"] == "coefficient")
            & (dataframe_integrated_models["inclusion_filter"] == "included_only")
            & (dataframe_integrated_models["term_name"].isin([
                "C(vignette_position, Treatment(reference=1))[T.2]",
                "C(vignette_position, Treatment(reference=1))[T.3]",
            ]))
        ].copy()

        coefficient_display_map = {
            "C(vignette_position, Treatment(reference=1))[T.2]": "Position 2 - Position 1 baseline shift",
            "C(vignette_position, Treatment(reference=1))[T.3]": "Position 3 - Position 1 baseline shift",
        }

        for _, coefficient_row in coefficient_rows.iterrows():
            table_rows.append(
                {
                    "Effect": coefficient_display_map[coefficient_row["term_name"]],
                    "Estimate": round(float(coefficient_row["estimate"]), 2),
                    "95% CI": _format_ci_for_manuscript_table(coefficient_row["ci95_lower"], coefficient_row["ci95_upper"]),
                    "p-value": _format_p_value_for_manuscript_table(coefficient_row["p_value"]),
                    "p_value_raw": float(coefficient_row["p_value"]),
                    "source_file": "integrated_blame_models.csv",
                    "meaning": coefficient_row["analysis_meaning"],
                }
            )

        omnibus_row_mask = (
            (dataframe_integrated_models["analysis_scope"] == "within_subjects_repeated_measures")
            & (dataframe_integrated_models["row_type"] == "omnibus_test")
            & (dataframe_integrated_models["inclusion_filter"] == "included_only")
            & (dataframe_integrated_models["contrast_label"] == "condition_by_position_interaction")
        )
        omnibus_rows = dataframe_integrated_models.loc[omnibus_row_mask].copy()

        if omnibus_rows.shape[0] == 1:
            omnibus_row = omnibus_rows.iloc[0]
            table_rows.append(
                {
                    "Effect": "Omnibus condition × position interaction",
                    "Estimate": "",
                    "95% CI": "",
                    "p-value": _format_p_value_for_manuscript_table(omnibus_row["p_value"]),
                    "p_value_raw": float(omnibus_row["p_value"]),
                    "source_file": "integrated_blame_models.csv",
                    "meaning": omnibus_row["analysis_meaning"],
                }
            )

        dataframe_table_8 = pd.DataFrame(table_rows)

        effect_sort_order = {
            "Position 2 - Position 1 baseline shift": 0,
            "Position 3 - Position 1 baseline shift": 1,
            "Omnibus condition × position interaction": 2,
        }
        dataframe_table_8["effect_sort_order"] = dataframe_table_8["Effect"].map(effect_sort_order)
        dataframe_table_8 = dataframe_table_8.sort_values(
            by="effect_sort_order",
            kind="stable",
        ).drop(columns=["effect_sort_order"])

    else:
        dataframe_consistency_effects = compute_consistency_effects(
            general_settings=general_settings,
            force_rebuild=force_rebuild,
        )

        table_rows: list[dict[str, Any]] = []

        dv_display_map = {
            "distal_blame": "Blame",
            "distal_wrong": "Wrongness",
            "distal_punish": "Punishment",
        }

        comparison_suffixes = [
            ("CH rating when CH-first vs when CC-first", "CH rating when CH first vs CC first"),
            ("CC rating when CH-first vs when CC-first", "CC rating when CH first vs CC first"),
        ]

        for dv_prefix, dv_display_label in dv_display_map.items():
            for raw_comparison_suffix, comparison_display_label in comparison_suffixes:
                row_series = _extract_exact_consistency_row_from_csv(
                    dataframe_consistency_effects=dataframe_consistency_effects,
                    inclusion_filter="included_only",
                    comparison=f"{dv_prefix}: {raw_comparison_suffix}",
                )

                table_rows.append(
                    {
                        "DV": dv_display_label,
                        "Comparison": comparison_display_label,
                        "Estimate": round(float(row_series["mean_difference_a_minus_b"]), 2),
                        "95% CI": _format_ci_for_manuscript_table(row_series["ci95_lower"], row_series["ci95_upper"]),
                        "p-value": _format_p_value_for_manuscript_table(row_series["p_value"]),
                        "p_value_raw": float(row_series["p_value"]),
                        "source_file": "consistency_effects.csv",
                        "meaning": row_series["comparison"],
                    }
                )

        dataframe_table_8 = pd.DataFrame(table_rows)

        dv_sort_order = {"Blame": 0, "Wrong": 1, "Punish": 2}
        comparison_sort_order = {
            "CH rating when CH first vs CC first": 0,
            "CC rating when CH first vs CC first": 1,
        }

        dataframe_table_8["dv_sort_order"] = dataframe_table_8["DV"].map(dv_sort_order)
        dataframe_table_8["comparison_sort_order"] = dataframe_table_8["Comparison"].map(comparison_sort_order)

        dataframe_table_8 = dataframe_table_8.sort_values(
            by=["dv_sort_order", "comparison_sort_order"],
            kind="stable",
        ).drop(columns=["dv_sort_order", "comparison_sort_order"])

    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table_8,
        general_settings=general_settings,
        table_file_name=table_file_name,
    )

    return dataframe_table_8


def _compute_model_contrasts_from_dataframe(
    cleaned_dataframe: pd.DataFrame,
    general_settings: GeneralSettings,
) -> pd.DataFrame:
    """
    Compute responsibility-shielding contrast statistics per model directly from the cleaned dataframe.

    Returns two stacked blocks: Included participants first, then Everyone (all finishers).

    Arguments:
        • cleaned_dataframe: pd.DataFrame
            - Must contain race_compact, included, story_condition, vignette_condition_position_1,
              first_vignette_distal_blame/wrong/punish, and delta columns.
        • general_settings: GeneralSettings

    Returns:
        • pd.DataFrame with columns: Inclusion, DV, Story Family, Model, Design, Contrast,
          Estimate, 95% CI, p-value.
    """
    dv_display_map = {"blame": "Blame", "wrong": "Wrong", "punish": "Punish"}
    story_display_map = {"pooled": "Pooled", "firework": "Firework", "trolley": "Trolley"}
    design_display_map = {
        "between_subjects_first_vignette": "Between-subj",
        "within_subjects_all_vignettes": "Within-subj",
    }
    within_delta_columns = {
        ("blame", "CH - CC"): "distal_blame_ch_minus_cc",
        ("blame", "DIV - CC"): "distal_blame_div_minus_cc",
        ("wrong", "CH - CC"): "distal_wrong_ch_minus_cc",
        ("wrong", "DIV - CC"): "distal_wrong_div_minus_cc",
        ("punish", "CH - CC"): "distal_punish_ch_minus_cc",
        ("punish", "DIV - CC"): "distal_punish_div_minus_cc",
    }
    first_vignette_columns = {
        "blame": "first_vignette_distal_blame",
        "wrong": "first_vignette_distal_wrong",
        "punish": "first_vignette_distal_punish",
    }

    def _rows_for_subset(df: pd.DataFrame, inclusion_label: str) -> list[dict]:
        rows = []
        for model_name in sorted(df["race_compact"].dropna().unique()):
            model_df = df.loc[df["race_compact"] == model_name]
            for dv_key, dv_display in dv_display_map.items():
                for story_key, story_display in story_display_map.items():
                    story_df = model_df if story_key == "pooled" else model_df.loc[model_df["story_condition"] == story_key]
                    for design_key, design_display in design_display_map.items():
                        for contrast in ["CH - CC", "DIV - CC"]:
                            try:
                                if design_key == "between_subjects_first_vignette":
                                    condition_a, condition_b = contrast.split(" - ")
                                    result = run_welch_t_test_between_groups(
                                        dataframe=story_df,
                                        dv_column_name=first_vignette_columns[dv_key],
                                        group_column_name="vignette_condition_position_1",
                                        group_a_value=condition_a,
                                        group_b_value=condition_b,
                                    )
                                else:
                                    result = run_one_sample_t_test_on_delta(
                                        dataframe=story_df,
                                        delta_column_name=within_delta_columns[(dv_key, contrast)],
                                    )
                                estimate = round(float(result["mean_difference_a_minus_b"]), 2)
                                ci_string = _format_ci_for_manuscript_table(result["ci95_lower"], result["ci95_upper"])
                                p_string = _format_p_value_for_manuscript_table(result["p_value"])
                            except Exception:
                                estimate = np.nan
                                ci_string = ""
                                p_string = ""
                            rows.append({
                                "Inclusion": inclusion_label,
                                "DV": dv_display,
                                "Story Family": story_display,
                                "Model": model_name,
                                "Design": design_display,
                                "Contrast": contrast,
                                "Estimate": estimate,
                                "95% CI": ci_string,
                                "p-value": p_string,
                            })
        return rows

    included_df = cleaned_dataframe.loc[cleaned_dataframe["included"] == True].copy()  # noqa: E712
    all_rows = _rows_for_subset(included_df, "Included") + _rows_for_subset(cleaned_dataframe.copy(), "Everyone")
    return pd.DataFrame(all_rows, columns=["Inclusion", "DV", "Story Family", "Model", "Design", "Contrast", "Estimate", "95% CI", "p-value"])


def compute_supplementary_table_9_secondary_dv_contrasts(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
    story_condition: bool = True,
    group_by_model: bool = False,
) -> pd.DataFrame:
    """
    Compute supplementary Table 9: included-only contrasts across blame, wrongness, and punishment.

    Arguments:
        • general_settings: GeneralSettings
        • cleaned_dataframe: pd.DataFrame | None
        • force_rebuild: bool | None
        • story_condition: bool
        • group_by_model: bool
            - If True, compute per-model contrasts directly from cleaned_dataframe and return
              without file I/O. Adds a Model column to the output.

    Returns:
        • pd.DataFrame
            - Manuscript-facing secondary-DV contrast table.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    if group_by_model:
        resolved_dataframe = cleaned_dataframe if cleaned_dataframe is not None else load_or_build_cleaned_dataframe(general_settings)
        return _compute_model_contrasts_from_dataframe(
            cleaned_dataframe=resolved_dataframe,
            general_settings=general_settings,
        )

    table_names = general_settings["filing"]["table_names"]
    table_file_name = table_names["table_9_secondary_dv_contrasts"]

    "Load existing table if it already matches the requested shape."
    table_extraction = _load_table_dataframe_from_tables_folder(
        general_settings=general_settings,
        table_file_name=table_file_name,
        force_rebuild=force_rebuild,
    )

    if table_extraction["success"]:
        cached_table_dataframe = table_extraction["dataframe"].copy()
        expected_row_count = 36 if story_condition else 12
        expected_story_values = {"Pooled", "Firework", "Trolley"} if story_condition else {"Pooled"}

        if (
            "Story Family" in cached_table_dataframe.columns
            and cached_table_dataframe.shape[0] == expected_row_count
            and set(cached_table_dataframe["Story Family"].dropna().unique()).issubset(expected_story_values)
        ):
            return cached_table_dataframe

    if table_extraction["error"] and "PathError" not in str(table_extraction["message"]):
        raise Exception(table_extraction["message"])

    "Load or rebuild preprocessed dataframe."
    if cleaned_dataframe is None:
        cleaned_dataframe = load_or_build_cleaned_dataframe(
            general_settings=general_settings,
            force_rebuild=False,
        )
    else:
        cleaned_dataframe = cleaned_dataframe.copy()

    dataframe_tests = run_confirmatory_and_exploratory_tests(
        general_settings=general_settings,
        force_rebuild=force_rebuild,
    )

    dv_display_map = {
        "blame": "Blame",
        "wrong": "Wrong",
        "punish": "Punish",
    }
    story_family_display_map = {
        "pooled": "Pooled",
        "firework": "Firework",
        "trolley": "Trolley",
    }
    design_display_map = {
        "between_subjects_first_vignette": "Between-subj",
        "within_subjects_all_vignettes": "Within-subj",
    }
    contrast_display_map = {
        "CH - CC": "CH - CC",
        "DIV - CC": "DIV - CC",
    }

    story_family_values = ["pooled", "firework", "trolley"] if story_condition else ["pooled"]

    table_rows = []

    for dv_value, dv_display_label in dv_display_map.items():
        for story_family_value in story_family_values:
            story_family_display_label = story_family_display_map[story_family_value]

            for design_value, design_display_label in design_display_map.items():
                for contrast_type_value, contrast_display_label in contrast_display_map.items():
                    row_series = _extract_exact_test_row_from_test_csv(
                        dataframe_tests=dataframe_tests,
                        inclusion_filter="included_only",
                        story_family=story_family_value,
                        load_condition="pooled",
                        design=design_value,
                        dv=dv_value,
                        agent_role="distal",
                        contrast_type=contrast_type_value,
                        analysis_family=(
                            "confirmatory"
                            if (
                                dv_value == "blame"
                                and design_value == "between_subjects_first_vignette"
                                and story_family_value == "pooled"
                            )
                            else "exploratory"
                        ),
                    )

                    displayed_p_value = _extract_reported_p_value_from_test_row(
                        row_series=row_series,
                        general_settings=general_settings,
                        use_holm_if_available=(
                            dv_value == "blame"
                            and design_value == "between_subjects_first_vignette"
                            and story_family_value == "pooled"
                        ),
                    )

                    table_rows.append(
                        {
                            "DV": dv_display_label,
                            "Story Family": story_family_display_label,
                            "Design": design_display_label,
                            "Contrast": contrast_display_label,
                            "Estimate": round(_extract_estimate_from_test_row(row_series), 2),
                            "95% CI": _format_ci_for_manuscript_table(
                                row_series["ci95_lower"],
                                row_series["ci95_upper"],
                            ),
                            "p_value_one_tailed": float(row_series["p_value_one_tailed"]),
                            "p_value_two_tailed": float(row_series["p_value_two_tailed"]),
                            "p-value": _format_p_value_for_manuscript_table(displayed_p_value),
                            "p_value_holm": (
                                float(row_series["p_value_holm"])
                                if "p_value_holm" in row_series.index and pd.notna(row_series["p_value_holm"])
                                else np.nan
                            ),
                            "p_value_displayed": float(displayed_p_value),
                            "source_file": "tests.csv",
                            "meaning": row_series["notes"],
                        }
                    )

    dataframe_table_9 = pd.DataFrame(table_rows)

    dv_sort_order = {
        "Blame": 0,
        "Wrong": 1,
        "Punish": 2,
    }
    story_family_sort_order = {
        "Pooled": 0,
        "Firework": 1,
        "Trolley": 2,
    }
    design_sort_order = {
        "Between-subj": 0,
        "Within-subj": 1,
    }
    contrast_sort_order = {
        "CH - CC": 0,
        "DIV - CC": 1,
    }

    dataframe_table_9["dv_sort_order"] = dataframe_table_9["DV"].map(dv_sort_order)
    dataframe_table_9["story_family_sort_order"] = dataframe_table_9["Story Family"].map(story_family_sort_order)
    dataframe_table_9["design_sort_order"] = dataframe_table_9["Design"].map(design_sort_order)
    dataframe_table_9["contrast_sort_order"] = dataframe_table_9["Contrast"].map(contrast_sort_order)

    dataframe_table_9 = dataframe_table_9.sort_values(
        by=["dv_sort_order", "story_family_sort_order", "design_sort_order", "contrast_sort_order"],
        kind="stable",
    ).drop(
        columns=["dv_sort_order", "story_family_sort_order", "design_sort_order", "contrast_sort_order"]
    )

    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table_9,
        general_settings=general_settings,
        table_file_name=table_file_name,
    )

    return dataframe_table_9


def compute_robot_table_10_model_means(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Robot-only Table 10: distal blame means broken down by model × story condition × vignette condition.

    Arguments:
        • general_settings: GeneralSettings
        • cleaned_dataframe: pd.DataFrame
            - Robot cleaned dataframe (must have race_compact column).

    Returns:
        • pd.DataFrame with columns: Model, Story Condition, Condition, Blame First Vignette, Blame All Vignettes.
    """
    dataframe_table = compute_manuscript_table_2_mean_scale_values_by_dv_and_condition(
        general_settings=general_settings,
        group_by_model=True,
        cleaned_dataframe=cleaned_dataframe,
    )
    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table,
        general_settings=general_settings,
        table_file_name=general_settings["filing"]["table_names"]["table_10_model_means"],
    )
    return dataframe_table


def compute_robot_table_11_model_contrasts(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Robot-only Table 11: responsibility-shielding contrast statistics broken down by model.

    Arguments:
        • general_settings: GeneralSettings
        • cleaned_dataframe: pd.DataFrame
            - Robot cleaned dataframe (must have race_compact column).

    Returns:
        • pd.DataFrame with columns: DV, Story Family, Model, Design, Contrast, Estimate, 95% CI, p-value.
    """
    dataframe_table = compute_supplementary_table_9_secondary_dv_contrasts(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        group_by_model=True,
    )
    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table,
        general_settings=general_settings,
        table_file_name=general_settings["filing"]["table_names"]["table_11_model_contrasts"],
    )
    return dataframe_table


def _compute_extra_terminal_statistics_for_manuscript(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame,
    print_: bool = True
) -> dict[str, Any]:
    """
    Compute a few one-line statistics that are useful to print to the terminal but do not need their own tables.

    Correlations are reported in three pairs: blame–wrong, wrong–punish, and
    punish–blame.  Each pair is computed two ways:
        1. Between-subjects first-vignette (one row per participant).
        2. Within-subjects participant means pooled across the three vignette
           conditions (CC, CH, DIV).

    Punishment is log1p-transformed before computing any punishment-involved
    correlation when ``general_settings > punish > analysis_mode`` is
    ``"log1p_parametric"``.  For the participant-means variants the
    transformation is applied to the per-participant mean of the three raw
    condition values (mean first, then log1p), which mirrors the approach used
    in ``compute_group_summaries``.

    Arguments:
        • general_settings: GeneralSettings
        • cleaned_dataframe: pd.DataFrame

    Returns:
        • dict[str, Any]
            - Dictionary of summary statistics.
    """
    included_dataframe = cleaned_dataframe.loc[cleaned_dataframe["included"] == True].copy()  # noqa: E712

    "Determine whether punishment should be log1p-transformed for correlations."
    punishment_analysis_mode = (
        str(general_settings.get("punish", {}).get("analysis_mode", "raw_nonparametric"))
        .strip()
        .lower()
    )
    punishment_is_log1p_transformed = punishment_analysis_mode == "log1p_parametric"

    "-------------------------------------------------------------------------"
    "---- Helper: apply log1p to a punishment Series when requested. ---------" 
    "---- For blame and wrong the raw values are always returned unchanged. --"
    "-------------------------------------------------------------------------"
    def _apply_punishment_transformation_if_requested(raw_punish_series: pd.Series) -> pd.Series:
        """Return log1p-transformed punishment values if analysis_mode is log1p_parametric, else raw."""
        if punishment_is_log1p_transformed:
            return np.log1p(raw_punish_series)
        return raw_punish_series

    "========================================================================="
    " FIRST-VIGNETTE CORRELATIONS  (between-subjects; one row per participant)"
    "========================================================================="

    "blame × wrong (first vignette)"
    first_vignette_blame_wrong_dataframe = included_dataframe[
        ["first_vignette_distal_blame", "first_vignette_distal_wrong"]
    ].dropna().copy()
    first_vignette_blame_wrong_correlation = stats.pearsonr(
        first_vignette_blame_wrong_dataframe["first_vignette_distal_blame"],
        first_vignette_blame_wrong_dataframe["first_vignette_distal_wrong"],
    )

    "wrong × punish (first vignette)"
    "Punishment values are transformed before correlating when log1p_parametric."
    first_vignette_wrong_punish_dataframe = included_dataframe[
        ["first_vignette_distal_wrong", "first_vignette_distal_punish"]
    ].dropna().copy()
    first_vignette_wrong_punish_dataframe = first_vignette_wrong_punish_dataframe.assign(
        first_vignette_distal_punish_analysis=_apply_punishment_transformation_if_requested(
            first_vignette_wrong_punish_dataframe["first_vignette_distal_punish"]
        )
    )
    first_vignette_wrong_punish_correlation = stats.pearsonr(
        first_vignette_wrong_punish_dataframe["first_vignette_distal_wrong"],
        first_vignette_wrong_punish_dataframe["first_vignette_distal_punish_analysis"],
    )

    "punish × blame (first vignette)"
    first_vignette_punish_blame_dataframe = included_dataframe[
        ["first_vignette_distal_punish", "first_vignette_distal_blame"]
    ].dropna().copy()
    first_vignette_punish_blame_dataframe = first_vignette_punish_blame_dataframe.assign(
        first_vignette_distal_punish_analysis=_apply_punishment_transformation_if_requested(
            first_vignette_punish_blame_dataframe["first_vignette_distal_punish"]
        )
    )
    first_vignette_punish_blame_correlation = stats.pearsonr(
        first_vignette_punish_blame_dataframe["first_vignette_distal_punish_analysis"],
        first_vignette_punish_blame_dataframe["first_vignette_distal_blame"],
    )

    "========================================================================="
    "PARTICIPANT-MEANS CORRELATIONS  (within-subjects; pooled across vignettes)"
    "========================================================================="
    participant_means_raw_blame_series = included_dataframe[
        ["distal_blame_cc", "distal_blame_ch", "distal_blame_div"]
    ].mean(axis=1)
    participant_means_raw_wrong_series = included_dataframe[
        ["distal_wrong_cc", "distal_wrong_ch", "distal_wrong_div"]
    ].mean(axis=1)
    participant_means_raw_punish_series = included_dataframe[
        ["distal_punish_cc", "distal_punish_ch", "distal_punish_div"]
    ].mean(axis=1)

    "Apply punishment transformation to the participant-level mean punish values."
    participant_means_punish_analysis_series = _apply_punishment_transformation_if_requested(
        participant_means_raw_punish_series
    )

    "blame × wrong (participant means)"
    participant_means_blame_wrong_dataframe = pd.DataFrame(
        {
            "mean_distal_blame": participant_means_raw_blame_series,
            "mean_distal_wrong": participant_means_raw_wrong_series,
        }
    ).dropna()
    participant_means_blame_wrong_correlation = stats.pearsonr(
        participant_means_blame_wrong_dataframe["mean_distal_blame"],
        participant_means_blame_wrong_dataframe["mean_distal_wrong"],
    )

    "wrong × punish (participant means)"
    participant_means_wrong_punish_dataframe = pd.DataFrame(
        {
            "mean_distal_wrong": participant_means_raw_wrong_series,
            "mean_distal_punish_analysis": participant_means_punish_analysis_series,
        }
    ).dropna()
    participant_means_wrong_punish_correlation = stats.pearsonr(
        participant_means_wrong_punish_dataframe["mean_distal_wrong"],
        participant_means_wrong_punish_dataframe["mean_distal_punish_analysis"],
    )

    "punish × blame (participant means)" 
    participant_means_punish_blame_dataframe = pd.DataFrame(
        {
            "mean_distal_punish_analysis": participant_means_punish_analysis_series,
            "mean_distal_blame": participant_means_raw_blame_series,
        }
    ).dropna()
    participant_means_punish_blame_correlation = stats.pearsonr(
        participant_means_punish_blame_dataframe["mean_distal_punish_analysis"],
        participant_means_punish_blame_dataframe["mean_distal_blame"],
    )

    "========================================================================="
    "==================== COGNITIVE-LOAD BLAME DIFFERENCE ===================="
    "========================================================================="
    cognitive_load_blame_difference_ch_minus_cc = run_welch_t_test_between_groups(
        dataframe=included_dataframe,
        dv_column_name="distal_blame_ch_minus_cc",
        group_column_name="load_condition",
        group_a_value="high",
        group_b_value="low",
    )
    cognitive_load_blame_difference_div_minus_cc = run_welch_t_test_between_groups(
        dataframe=included_dataframe,
        dv_column_name="distal_blame_div_minus_cc",
        group_column_name="load_condition",
        group_a_value="high",
        group_b_value="low",
    )

    extra_statistics = {
        "first_vignette_blame_wrong_r": float(first_vignette_blame_wrong_correlation.statistic),
        "first_vignette_blame_wrong_p": float(first_vignette_blame_wrong_correlation.pvalue),
        "first_vignette_wrong_punish_r": float(first_vignette_wrong_punish_correlation.statistic),
        "first_vignette_wrong_punish_p": float(first_vignette_wrong_punish_correlation.pvalue),
        "first_vignette_punish_blame_r": float(first_vignette_punish_blame_correlation.statistic),
        "first_vignette_punish_blame_p": float(first_vignette_punish_blame_correlation.pvalue),
        "participant_means_blame_wrong_r": float(participant_means_blame_wrong_correlation.statistic),
        "participant_means_blame_wrong_p": float(participant_means_blame_wrong_correlation.pvalue),
        "participant_means_wrong_punish_r": float(participant_means_wrong_punish_correlation.statistic),
        "participant_means_wrong_punish_p": float(participant_means_wrong_punish_correlation.pvalue),
        "participant_means_punish_blame_r": float(participant_means_punish_blame_correlation.statistic),
        "participant_means_punish_blame_p": float(participant_means_punish_blame_correlation.pvalue),
        "high_minus_low_ch_cc_delta_difference": float(cognitive_load_blame_difference_ch_minus_cc["mean_difference_a_minus_b"]),
        "high_minus_low_ch_cc_delta_ci95_lower": float(cognitive_load_blame_difference_ch_minus_cc["ci95_lower"]),
        "high_minus_low_ch_cc_delta_ci95_upper": float(cognitive_load_blame_difference_ch_minus_cc["ci95_upper"]),
        "high_minus_low_ch_cc_delta_p": float(cognitive_load_blame_difference_ch_minus_cc["p_value"]),
        "high_minus_low_div_cc_delta_difference": float(cognitive_load_blame_difference_div_minus_cc["mean_difference_a_minus_b"]),
        "high_minus_low_div_cc_delta_ci95_lower": float(cognitive_load_blame_difference_div_minus_cc["ci95_lower"]),
        "high_minus_low_div_cc_delta_ci95_upper": float(cognitive_load_blame_difference_div_minus_cc["ci95_upper"]),
        "high_minus_low_div_cc_delta_p": float(cognitive_load_blame_difference_div_minus_cc["p_value"]),
    }

    if print_:
        print("\n" + "-" * 110)
        print("EXTRA ONE-LINE STATISTICS")
        print("-" * 110)
        print(
            "First-vignette blame-wrongness correlation: "
            f"r = {extra_statistics['first_vignette_blame_wrong_r']:.2f}, "
            f"p = {_format_p_value_for_manuscript_table(extra_statistics['first_vignette_blame_wrong_p'])}"
        )
        print(
            "First-vignette wrong-punishment correlation: "
            f"r = {extra_statistics['first_vignette_wrong_punish_r']:.2f}, "
            f"p = {_format_p_value_for_manuscript_table(extra_statistics['first_vignette_wrong_punish_p'])}"
        )
        print(
            "First-vignette punishment-blame correlation: "
            f"r = {extra_statistics['first_vignette_punish_blame_r']:.2f}, "
            f"p = {_format_p_value_for_manuscript_table(extra_statistics['first_vignette_punish_blame_p'])}"
        )
        print(
            "Participant-means blame-wrongness correlation: "
            f"r = {extra_statistics['participant_means_blame_wrong_r']:.2f}, "
            f"p = {_format_p_value_for_manuscript_table(extra_statistics['participant_means_blame_wrong_p'])}"
        )
        print(
            "Participant-means wrong-punishment correlation: "
            f"r = {extra_statistics['participant_means_wrong_punish_r']:.2f}, "
            f"p = {_format_p_value_for_manuscript_table(extra_statistics['participant_means_wrong_punish_p'])}"
        )
        print(
            "Participant-means punishment-blame correlation: "
            f"r = {extra_statistics['participant_means_punish_blame_r']:.2f}, "
            f"p = {_format_p_value_for_manuscript_table(extra_statistics['participant_means_punish_blame_p'])}"
        )
        print(
            "Cognitive-load difference in CH - CC shielding (High - Low): "
            f"Δ = {extra_statistics['high_minus_low_ch_cc_delta_difference']:+.2f}, "
            f"95% CI = {_format_ci_for_manuscript_table(extra_statistics['high_minus_low_ch_cc_delta_ci95_lower'], extra_statistics['high_minus_low_ch_cc_delta_ci95_upper'])}, "
            f"p = {_format_p_value_for_manuscript_table(extra_statistics['high_minus_low_ch_cc_delta_p'])}"
        )
        print(
            "Cognitive-load difference in DIV - CC shielding (High - Low): "
            f"Δ = {extra_statistics['high_minus_low_div_cc_delta_difference']:+.2f}, "
            f"95% CI = {_format_ci_for_manuscript_table(extra_statistics['high_minus_low_div_cc_delta_ci95_lower'], extra_statistics['high_minus_low_div_cc_delta_ci95_upper'])}, "
            f"p = {_format_p_value_for_manuscript_table(extra_statistics['high_minus_low_div_cc_delta_p'])}"
        )

    return extra_statistics


def generate_manuscript_and_supplementary_tables(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> dict[str, Any]:
    """
    Generate all manuscript and supplementary tables in one call.

    Arguments:
        • general_settings: GeneralSettings
            - Master project settings dictionary.
        • cleaned_dataframe: pd.DataFrame | None
            - Optional cleaned dataframe. If None, load or build it.
        • force_rebuild: bool | None
            - Whether to force rebuilding of the table CSVs.

    Returns:
        • dict[str, Any]
            - Dictionary with keys:
                "table_1"     — participant counts by condition
                "table_2"     — mean scale values by DV and condition
                "table_3"     — primary distal-blame contrasts
                "table_4"     — story-specific distal-blame contrasts
                "table_5"     — 2AFC response distribution
                "table_6"     — within-subject pairwise blame matrix (wide format)
                "table_6_long"— within-subject pairwise blame matrix (long format)
                "table_7"     — cognitive-load blame contrasts
                "table_8"     — order-effects summary
                "table_9"     — secondary DV contrasts
                "table_manifest" — one-row-per-table manifest of all saved files
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    table_names = general_settings["filing"]["table_names"]
    print_tables_to_terminal = general_settings["misc"]["print_tables_to_terminal"]

    "Load or rebuild preprocessed dataframe."
    if cleaned_dataframe is None:
        cleaned_dataframe = load_or_build_cleaned_dataframe(
            general_settings=general_settings,
            force_rebuild=False,
        )
    else:
        cleaned_dataframe = cleaned_dataframe.copy()

    "Build all tables."
    dataframe_table_1 = compute_manuscript_table_1_participant_counts(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )
    dataframe_table_2 = compute_manuscript_table_2_mean_scale_values_by_dv_and_condition(
        general_settings=general_settings,
        force_rebuild=force_rebuild,
    )
    dataframe_table_3 = compute_manuscript_table_3_primary_distal_blame_contrasts(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )
    dataframe_table_4 = compute_manuscript_table_4_story_specific_distal_blame_contrasts(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )
    dataframe_table_5 = compute_manuscript_table_5_two_alternative_forced_choice_distribution(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )
    table_6_outputs = compute_supplementary_table_6_within_subject_pairwise_blame_matrix(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )
    dataframe_table_7 = compute_supplementary_table_7_cognitive_load_blame_contrasts(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )
    dataframe_table_8 = compute_supplementary_table_8_order_effects_summary(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )
    dataframe_table_9 = compute_supplementary_table_9_secondary_dv_contrasts(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )

    "Build and save a simple manifest."
    table_manifest_rows = [
        {
            "table_key": "table_1",
            "table_title": "Table 1. Participant counts by randomization condition",
            "file_name": table_names["table_1_participant_counts"],
            "table_meaning": "Main-text participant counts for included participants and everyone.",
        },
        {
            "table_key": "table_2",
            "table_title": "Table 2. Mean values by story type, dependent variable, and vignette condition",
            "file_name": table_names["table_2_means_by_dv_and_condition"],
            "table_meaning": "Main-text mean scale values by DV and condition.",
        },        
        {
            "table_key": "table_3",
            "table_title": "Table 3. Primary Clark-blame contrasts",
            "file_name": table_names["table_3_primary_distal_blame_contrasts"],
            "table_meaning": "Main-text primary contrast table showing blame contrasts between- and within-subjects.",
        },
        {
            "table_key": "table_4",
            "table_title": "Table 4. Story-specific Clark-blame contrasts",
            "file_name": table_names["table_4_story_specific_distal_blame_contrasts"],
            "table_meaning": "Main-text story-specific decomposition of Clark blame contrasts.",
        },
        {
            "table_key": "table_5",
            "table_title": "Table 5. 2AFC response distribution",
            "file_name": table_names["table_5_two_alternative_forced_choice_distribution"],
            "table_meaning": "Main-text forced-choice response table.",
        },
        {
            "table_key": "table_6",
            "table_title": "Table 6. Within-subject pairwise blame matrix",
            "file_name": table_names["table_6_within_subject_pairwise_blame_matrix"],
            "table_meaning": "Supplementary formatted upper-triangular pairwise matrix for blame.",
        },
        {
            "table_key": "table_6_long",
            "table_title": "Table 6 long-form backing table",
            "file_name": table_names["table_6_within_subject_pairwise_blame_long"],
            "table_meaning": "Supplementary long-form exact statistics for the pairwise blame matrix.",
        },
        {
            "table_key": "table_7",
            "table_title": "Table 7. Cognitive-load blame contrasts",
            "file_name": table_names["table_7_cognitive_load_blame_contrasts"],
            "table_meaning": "Supplementary cognitive-load table for the core blame deltas.",
        },
        {
            "table_key": "table_8",
            "table_title": "Table 8. Order-effects summary",
            "file_name": table_names["table_8_order_effects_summary"],
            "table_meaning": "Supplementary compact summary of baseline position effects.",
        },
        {
            "table_key": "table_9",
            "table_title": "Table 9. Secondary DV contrasts",
            "file_name": table_names["table_9_secondary_dv_contrasts"],
            "table_meaning": "Supplementary table spanning blame, wrongness, and punishment.",
        },
    ]
    dataframe_table_manifest = pd.DataFrame(table_manifest_rows)

    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table_manifest,
        general_settings=general_settings,
        table_file_name=table_names["table_manifest"],
    )

    if print_tables_to_terminal:
        print("\n" + "=" * 110)
        print("Responsibility Shielding: When Causally Proximate Agents Reduce Blame for Distal Enablers")
        print("Greg Stanley")
        print("=" * 110)

        _pretty_print_table_to_terminal("TABLE 1. Participant Counts by Randomization Condition", dataframe_table_1)
        _pretty_print_table_to_terminal("TABLE 2. Mean Scale Values by DV and Condition", dataframe_table_2)
        _pretty_print_table_to_terminal("TABLE 3. Primary Clark-Blame Contrasts", dataframe_table_3)
        _pretty_print_table_to_terminal("TABLE 4. Story-Specific Clark-Blame Contrasts", dataframe_table_4)
        _pretty_print_table_to_terminal("TABLE 5. 2AFC Distribution", dataframe_table_5)
        _pretty_print_table_to_terminal(
            "TABLE 6. Within-Subject Pairwise Blame Matrix",
            table_6_outputs["formatted_matrix_dataframe"].reset_index().rename(columns={"index": "Row / Column"}),
        )
        _pretty_print_table_to_terminal("TABLE 7. Cognitive-Load Blame Contrasts", dataframe_table_7)
        _pretty_print_table_to_terminal("TABLE 8. Order-Effects Summary", dataframe_table_8)
        _pretty_print_table_to_terminal("TABLE 9. Secondary DV Contrasts", dataframe_table_9)

        _compute_extra_terminal_statistics_for_manuscript(
            general_settings=general_settings,
            cleaned_dataframe=cleaned_dataframe,
            print_=True
        )

    return {
        "table_1": dataframe_table_1,
        "table_2": dataframe_table_2,
        "table_3": dataframe_table_3,
        "table_4": dataframe_table_4,
        "table_5": dataframe_table_5,
        "table_6": table_6_outputs["formatted_matrix_dataframe"],
        "table_6_long": table_6_outputs["pairwise_long_dataframe"],
        "table_7": dataframe_table_7,
        "table_8": dataframe_table_8,
        "table_9": dataframe_table_9,
        "table_manifest": dataframe_table_manifest,
    }


