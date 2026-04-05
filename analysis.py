
"""
analysis.py

Responsibility Shielding Analysis Pipeline
Author: Greg Stanley

This script reproduces the analyses, tables, and figures 
for the paper on responsibility shielding in moral judgment.

The pipeline:
1. Loads or rebuilds the cleaned dataset from raw Qualtrics data
2. Applies the preregistered data-freeze window
3. Runs confirmatory and exploratory analyses
4. Runs integrated model analyses
5. Generates all manuscript and supplementary tables
6. Generates all Plotly figures 
7. Saves outputs to processed/, tables/, and visuals/

Default settings reproduce the manuscript results.

Use `force_rebuild=True` to regenerate outputs from scratch.
Set to False after first run to save compute resources.
"""
from __future__ import annotations
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import stats

from typing import Sequence, TypedDict, Dict, List, Tuple, Any
import plotly.graph_objects as go, statsmodels.api as sm, \
    statsmodels.formula.api as smf, pandas as pd, numpy as np, copy, os, re


"=========================================================================================="
"====================================== Type Aliases ======================================"
"=========================================================================================="

class FileNames(TypedDict):
    tests: Path
    cleaned: Path
    raw_data: Path
    group_summaries: Path
    consistency_effects: Path
    afc_counts_table: Path
    afc_counts_long: Path
    triangulation: Path
    correlations: Path
    regressions: Path
    first_vignette: Path
    within_subject: Path
    blame_models: Path

class TableNames(TypedDict):
    table_1_participant_counts: Path
    table_2_means_by_dv_and_condition: Path
    table_3_primary_clark_blame_contrasts: Path
    table_4_story_specific_clark_blame_contrasts: Path
    table_5_two_alternative_forced_choice_distribution: Path
    table_6_within_subject_pairwise_blame_matrix: Path
    table_6_within_subject_pairwise_blame_long: Path
    table_7_cognitive_load_blame_contrasts: Path
    table_8_order_effects_summary: Path
    table_9_secondary_dv_contrasts: Path
    table_manifest: Path

class FilePaths(TypedDict):
    raw_data: Path
    processed: Path
    visuals: Path
    images: Path
    tables: Path
    root: Path

class Filing(TypedDict):
    file_paths: FilePaths
    file_names: FileNames
    table_names: TableNames

class Visuals(TypedDict):
    figure_layout: dict[str, Any]
    create_figures: bool
    export_figure: bool
    marker_size: int
    dark_mode: bool
    base_hue: int

class MiscSettings(TypedDict):
    confirmatory_between_subjects_method: str
    rebuild_cleaned_dataframe: bool
    print_tables_to_terminal: bool
    freeze_timestamp_first: str
    freeze_timestamp_last: str
    use_integrated_models: bool
    force_rebuild: bool
    
class GeneralSettings(TypedDict):
    filing: Filing
    visuals: Visuals
    misc: MiscSettings


"=========================================================================================="
"===================================== Preprocessing ======================================"
"=========================================================================================="

def convert_string_to_snake_case(original_string: str) -> str:
    """
    Converts a string into snake_case.

    Arguments:
        • original_string:
            Any column name string.

    Returns:
        A snake_case version of the input string.
    """
    if original_string is None:
        return ""

    string_value = str(original_string).strip()

    "Replace non-alphanumeric characters with underscores"
    snake = []
    for character_value in string_value:
        if character_value.isalnum():
            snake.append(character_value)
        else:
            snake.append("_")

    snake_string = "".join(snake)

    "Insert underscores before capitals"
    with_underscores = []
    for idx, character_value in enumerate(snake_string):
        if idx > 0 and character_value.isupper() and snake_string[idx - 1].isalnum() and snake_string[idx - 1].islower():
            with_underscores.append("_")
        with_underscores.append(character_value)

    snake_string = "".join(with_underscores)

    "Collapse repeated underscores and lower-case"
    snake_string = "_".join([chunk for chunk in snake_string.split("_") if chunk != ""]).lower()

    return snake_string


def parse_likert_numeric_value(likert_value) -> float:
    """
    Parses a Likert response exported by Qualtrics into a numeric value.

    Handles cases like:
        - '(9) Extremely blameworthy'
        - '9 - Extremely blameworthy'
        - '7'
        - 7

    Returns:
        float (np.nan if parsing fails)
    """
    if pd.isna(likert_value):
        return np.nan

    if isinstance(likert_value, (int, np.integer)):
        return float(int(likert_value))

    if isinstance(likert_value, float):
        if np.isnan(likert_value):
            return np.nan
        return float(likert_value)

    likert_string = str(likert_value).strip()

    "Extract the first integer substring found"
    integer_match = re.search(r"-?\d+", likert_string)

    if integer_match is None:
        return np.nan

    try:
        return float(integer_match.group())
    except Exception:
        return np.nan


def parse_boolean_value(boolean_value) -> bool:
    """
    Parses a Qualtrics-exported boolean-like value into a Python bool.

    Accepts:
        - True/False (bool)
        - 'True'/'False' (strings)
        - '1'/'0', 'yes'/'no', 'y'/'n'

    Returns:
        bool (defaults to False when ambiguous)
    """
    if pd.isna(boolean_value):
        return np.nan

    if isinstance(boolean_value, bool):
        return boolean_value

    boolean_string = str(boolean_value).strip().lower()

    if boolean_string in ["true", "t", "1", "yes", "y"]:
        return True

    if boolean_string in ["false", "f", "0", "no", "n"]:
        return False

    return False


def coalesce_series(primary_series: pd.Series, secondary_series: pd.Series) -> pd.Series:
    """
    Returns a series equal to primary_series where non-null, otherwise secondary_series.

    Arguments:
        • primary_series:
            Preferred values.
        • secondary_series:
            Fallback values.

    Returns:
        Combined series.
    """
    return primary_series.combine_first(secondary_series)


def compute_crt_score(crt_response_1, crt_response_2, crt_response_3) -> float:
    """
    Computes CRT total score (0-3) using answers:
        Q1 = 5, Q2 = 5, Q3 = 47

    Returns:
        float score in {0,1,2,3} or np.nan if all responses missing.
    """
    responses = [crt_response_1, crt_response_2, crt_response_3]
    numeric_responses = [parse_likert_numeric_value(value) for value in responses]

    if all([np.isnan(value) for value in numeric_responses]):
        return np.nan

    answer_key = [5.0, 5.0, 47.0]

    correct_count = 0
    for given_value, correct_value in zip(numeric_responses, answer_key):
        if np.isnan(given_value):
            continue
        if float(given_value) == float(correct_value):
            correct_count += 1

    return float(correct_count)


def compute_indcol_scores(indcol_item_series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Computes INDCOL-derived individualism scores.

    I treat collectivism as the opposite pole of individualism and compute:

        individualism_horizontal = scaled( mean(HI) - mean(HC) ) in [0,1]
        individualism_vertical   = scaled( mean(VI) - mean(VC) ) in [0,1]
        individualism_score      = mean(individualism_horizontal, individualism_vertical)

    Notes:
        - This uses only whichever items are present in the dataset.
        - All items are assumed to be on the same 1-9 scale.

    Returns:
        DataFrame with columns:
            - individualism_horizontal
            - individualism_vertical
            - individualism_score
    """
    hi_columns = [column_name for column_name in indcol_item_series_dict.keys() if column_name.startswith("indcol_hi_")]
    vi_columns = [column_name for column_name in indcol_item_series_dict.keys() if column_name.startswith("indcol_vi_")]
    hc_columns = [column_name for column_name in indcol_item_series_dict.keys() if column_name.startswith("indcol_hc_")]
    vc_columns = [column_name for column_name in indcol_item_series_dict.keys() if column_name.startswith("indcol_vc_")]

    "Convert all INDCOL items to numeric"
    indcol_numeric = {}
    for column_name, column_series in indcol_item_series_dict.items():
        indcol_numeric[column_name] = column_series.apply(parse_likert_numeric_value)

    indcol_numeric_df = pd.DataFrame(indcol_numeric)

    hi_mean = indcol_numeric_df[hi_columns].mean(axis=1, skipna=True) if len(hi_columns) > 0 else np.nan
    vi_mean = indcol_numeric_df[vi_columns].mean(axis=1, skipna=True) if len(vi_columns) > 0 else np.nan
    hc_mean = indcol_numeric_df[hc_columns].mean(axis=1, skipna=True) if len(hc_columns) > 0 else np.nan
    vc_mean = indcol_numeric_df[vc_columns].mean(axis=1, skipna=True) if len(vc_columns) > 0 else np.nan

    "Scale mean differences from [-8, +8] into [0, 1] on a 1-9 scale"
    horizontal_difference = hi_mean - hc_mean
    vertical_difference = vi_mean - vc_mean

    individualism_horizontal = (horizontal_difference + 8) / 16
    individualism_vertical = (vertical_difference + 8) / 16

    individualism_score = pd.concat([individualism_horizontal, individualism_vertical], axis=1).mean(axis=1, skipna=True)

    return pd.DataFrame(
        {
            "individualism_horizontal": individualism_horizontal,
            "individualism_vertical": individualism_vertical,
            "individualism_score": individualism_score,
        }
    )


def compute_cognitive_load_digits_correct_bool(load_condition_value: str, digits_response_value) -> bool:
    """
    Computes whether the cognitive load digits were recalled correctly.

    Expected responses:
        - High load: 8403259
        - Low load: 63
    """
    if pd.isna(load_condition_value):
        return False

    load_condition_string = str(load_condition_value).strip().lower()

    if pd.isna(digits_response_value):
        return False

    digits_string = str(digits_response_value).strip()

    digits_string = digits_string.replace(" ", "")

    if load_condition_string == "high":
        return digits_string == "8403259"

    if load_condition_string == "low":
        return digits_string == "63"

    return False


def encode_two_afc_compact_response(primary_choice_value, followup_choice_value, left_label: str, 
                                    right_label: str, left_prefix: str, right_prefix: str) -> str:
    """
    Encodes a 2AFC + follow-up into one of four compact strings.

    If the participant selects the left option:
        - If they later say roughly equal: left_prefix ≥ right_prefix
        - Else: left_prefix > right_prefix

    If the participant selects the right option:
        - If they later say roughly equal: left_prefix ≤ right_prefix
        - Else: left_prefix < right_prefix

    Returns:
        One of:
            f"{left_prefix} > {right_prefix}"
            f"{left_prefix} ≥ {right_prefix}"
            f"{left_prefix} ≤ {right_prefix}"
            f"{left_prefix} < {right_prefix}"
        or np.nan if mapping fails.
    """
    if pd.isna(primary_choice_value) or pd.isna(followup_choice_value):
        return np.nan

    primary_string = str(primary_choice_value).strip()
    followup_string = str(followup_choice_value).strip().lower()

    is_roughly_equal = "roughly equally" in followup_string

    if primary_string == left_label:
        return f"{left_prefix} ≥ {right_prefix}" if is_roughly_equal else f"{left_prefix} > {right_prefix}"

    if primary_string == right_label:
        return f"{left_prefix} ≤ {right_prefix}" if is_roughly_equal else f"{left_prefix} < {right_prefix}"

    return np.nan


def parse_datetime_series(datetime_series: pd.Series) -> pd.Series:
    """
    Parses a Qualtrics StartDate/EndDate column into pandas datetime.

    Returns:
        pandas Series of dtype datetime64[ns] with NaT for unparseable values.
    """
    return pd.to_datetime(datetime_series, format="%m/%d/%Y %H:%M", errors="coerce")


def apply_collection_window_freeze(raw_dataframe: pd.DataFrame, general_settings: dict[str, Any]) -> pd.DataFrame:
    """
    Restrict the raw Qualtrics dataframe to the final collection window used in the paper.

    Arguments:
        • raw_dataframe: pd.DataFrame
            - Raw Qualtrics export dataframe.
        • general_settings: dict[str, Any]
            - Master settings dictionary. Expects:
                general_settings["misc"]["freeze_timestamp_first"]
                general_settings["misc"]["freeze_timestamp_last"]

    Returns:
        • pd.DataFrame
            - Raw dataframe filtered to the frozen collection window.
    """
    freeze_timestamp_first: pd.Timestamp = pd.to_datetime(
        general_settings["misc"]["freeze_timestamp_first"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="raise",
    )
    freeze_timestamp_last: pd.Timestamp = pd.to_datetime(
        general_settings["misc"]["freeze_timestamp_last"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="raise",
    )

    possible_timestamp_column_names: list[str] = [
        "RecordedDate",
        "StartDate",
        "EndDate",
        "recorded_date",
        "start_date",
        "end_date",
    ]

    timestamp_column_name: str | None = None
    for column_name in possible_timestamp_column_names:
        if column_name in raw_dataframe.columns:
            timestamp_column_name = column_name
            break

    if timestamp_column_name is None:
        raise Exception(
            "Could not find a Qualtrics timestamp column for freeze-window filtering. "
            f"Tried: {possible_timestamp_column_names}"
        )

    filtered_dataframe = raw_dataframe.copy()
    filtered_dataframe[timestamp_column_name] = pd.to_datetime(
        filtered_dataframe[timestamp_column_name],
        errors="coerce",
    )

    filtered_dataframe = filtered_dataframe.loc[
        filtered_dataframe[timestamp_column_name].between(
            freeze_timestamp_first,
            freeze_timestamp_last,
            inclusive="both",
        )
    ].copy()

    return filtered_dataframe


def drop_identifying_columns_from_cleaned_dataframe(cleaned_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary identifying Qualtrics columns from the cleaned dataframe before saving it.

    Arguments:
        • cleaned_dataframe: pd.DataFrame
            - Cleaned participant-level dataframe.

    Notes:
        • I already removed these columns from the raw data file before pushing the code, making this function pro forma.
        • The only columns that could have possibly identified the participants were their latitude, longitude, and IP address.        

    Returns:
        • pd.DataFrame
            - De-identified cleaned dataframe.
    """
    column_names_to_drop_if_present: list[str] = [
        "IPAddress",
        "RecipientLast Name",
        "RecipientFirst Name",
        "RecipientEmail",
        "ExternalDataReference",
        "LocationLatitude",
        "LocationLongitude",
        "DistributionChannel",
        "Status"
    ]

    cleaned_dataframe = cleaned_dataframe.drop(
        columns=[column_name for column_name in column_names_to_drop_if_present if column_name in cleaned_dataframe.columns],
        errors="ignore",
    ).copy()

    return cleaned_dataframe


def preprocess_raw_qualtrics_export(general_settings: GeneralSettings) -> pd.DataFrame:
    """
    Preprocesses the raw Qualtrics export into an analysis-ready dataframe.

    Returns:
        Cleaned dataframe (completed participants only).
    """
    file_path_raw_data = general_settings["filing"]["file_paths"]["raw_data"] / \
        general_settings["filing"]["file_names"]["raw_data"]

    freeze_timestamp_first = general_settings["misc"]["freeze_timestamp_first"]

    f"Loading raw Qualtrics export: {file_path_raw_data}"
    raw_dataframe = pd.read_csv(file_path_raw_data, low_memory=False)

    "Filter to actual participant responses (exclude preview rows and Qualtrics header rows)"
    if "Status" in raw_dataframe.columns and "IP Address" in raw_dataframe.columns:
        raw_dataframe = raw_dataframe[raw_dataframe["Status"] == "IP Address"].copy()

    "StartDate cutoff to remove pilot / pre-freeze runs"
    raw_dataframe["start_date"] = parse_datetime_series(raw_dataframe["StartDate"])

    raw_dataframe["end_date"] = parse_datetime_series(raw_dataframe["EndDate"])

    raw_dataframe["duration_seconds"] = pd.to_numeric(raw_dataframe["Duration (in seconds)"], errors="coerce")

    freeze_timestamp = pd.to_datetime(freeze_timestamp_first)

    raw_dataframe = raw_dataframe[raw_dataframe["start_date"] > freeze_timestamp].copy()

    "Keep only completed surveys"
    raw_dataframe["finished_bool"] = raw_dataframe["Finished"].apply(parse_boolean_value)
    raw_dataframe["progress_numeric"] = pd.to_numeric(raw_dataframe["Progress"], errors="coerce")

    raw_dataframe = raw_dataframe[(raw_dataframe["finished_bool"]) & (raw_dataframe["progress_numeric"] >= 100)].copy()

    "Standardize core condition columns"
    raw_dataframe["story_condition"] = raw_dataframe["StoryCondition"].astype(str).str.strip().str.lower()
    raw_dataframe["story_condition"] = raw_dataframe["story_condition"].str.replace("parade", "firework", regex=False)
    raw_dataframe["load_condition"] = raw_dataframe["LoadCondition"].astype(str).str.strip().str.lower()

    raw_dataframe["case_order"] = raw_dataframe["CaseOrder"].astype(str)

    raw_dataframe["case_code_position_1"] = raw_dataframe["case_order"].str.split("-").str[0]
    raw_dataframe["case_code_position_2"] = raw_dataframe["case_order"].str.split("-").str[1]
    raw_dataframe["case_code_position_3"] = raw_dataframe["case_order"].str.split("-").str[2]

    raw_dataframe["cognitive_load_digits_response"] = raw_dataframe["cog_load_check"]
    raw_dataframe["cognitive_load_digits_correct_bool"] = raw_dataframe.apply(
        lambda row: compute_cognitive_load_digits_correct_bool(row["load_condition"], row["cognitive_load_digits_response"]),
        axis=1,
    )

    "Merge comprehension checks across story condition"
    raw_dataframe["comprehension_probability_same_bool"] = coalesce_series(
        raw_dataframe["comp_p_prob_harm"].apply(parse_boolean_value),
        raw_dataframe["comp_t_prob_harm"].apply(parse_boolean_value),
    )

    raw_dataframe["comprehension_clark_necessary_bool"] = coalesce_series(
        raw_dataframe["comp_p_agency_across"].apply(parse_boolean_value),
        raw_dataframe["comp_t_agency_across"].apply(parse_boolean_value),
    )

    raw_dataframe["comprehension_bill_necessary_bool"] = coalesce_series(
        raw_dataframe["comp_p_agency_within"].apply(parse_boolean_value),
        raw_dataframe["comp_t_agency_within"].apply(parse_boolean_value),
    )

    raw_dataframe["comprehension_all_correct_bool"] = (
        raw_dataframe["comprehension_probability_same_bool"]
        & raw_dataframe["comprehension_clark_necessary_bool"]
        & raw_dataframe["comprehension_bill_necessary_bool"]
    )

    "Merge core outcome measures across story condition (Clark = upstream actor; Bill/proximate = downstream node)"
    raw_dataframe["clark_blame_cc"] = coalesce_series(
        raw_dataframe["pucc_blame_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tucc_blame_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["clark_blame_ch"] = coalesce_series(
        raw_dataframe["puch_blame_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tuch_blame_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["clark_blame_div"] = coalesce_series(
        raw_dataframe["pudiv_blame_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tudiv_blame_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["clark_wrong_cc"] = coalesce_series(
        raw_dataframe["pucc_wrong_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tucc_wrong_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["clark_wrong_ch"] = coalesce_series(
        raw_dataframe["puch_wrong_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tuch_wrong_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["clark_wrong_div"] = coalesce_series(
        raw_dataframe["pudiv_wrong_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tudiv_wrong_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["clark_punish_cc"] = coalesce_series(
        pd.to_numeric(raw_dataframe["pucc_punish"], errors="coerce"),
        pd.to_numeric(raw_dataframe["tucc_punish"], errors="coerce"),
    )

    raw_dataframe["clark_punish_ch"] = coalesce_series(
        pd.to_numeric(raw_dataframe["puch_punish"], errors="coerce"),
        pd.to_numeric(raw_dataframe["tuch_punish"], errors="coerce"),
    )

    raw_dataframe["clark_punish_div"] = coalesce_series(
        pd.to_numeric(raw_dataframe["pudiv_punish"], errors="coerce"),
        pd.to_numeric(raw_dataframe["tudiv_punish"], errors="coerce"),
    )

    raw_dataframe["proximate_blame_cc"] = coalesce_series(
        raw_dataframe["ppcc_blame_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tpcc_blame_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["proximate_blame_ch"] = coalesce_series(
        raw_dataframe["ppch_blame_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tpch_blame_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["proximate_wrong_cc"] = coalesce_series(
        raw_dataframe["ppcc_wrong_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tpcc_wrong_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["proximate_wrong_ch"] = coalesce_series(
        raw_dataframe["ppch_wrong_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tpch_wrong_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["proximate_punish_cc"] = coalesce_series(
        pd.to_numeric(raw_dataframe["ppcc_punish"], errors="coerce"),
        pd.to_numeric(raw_dataframe["tpcc_punish"], errors="coerce"),
    )

    raw_dataframe["proximate_punish_ch"] = coalesce_series(
        pd.to_numeric(raw_dataframe["ppch_punish"], errors="coerce"),
        pd.to_numeric(raw_dataframe["tpch_punish"], errors="coerce"),
    )

    "Helper columns for between-subjects confirmatory analysis (first vignette only)"
    def choose_first_case_value(row, column_cc: str, column_ch: str, column_div: str) -> float:
        case_code_value = row["case_code_position_1"]
        if case_code_value == "CC":
            return row[column_cc]
        if case_code_value == "CH":
            return row[column_ch]
        if case_code_value == "DIV":
            return row[column_div]
        return np.nan

    raw_dataframe["first_vignette_clark_blame"] = raw_dataframe.apply(
        lambda row: choose_first_case_value(row, "clark_blame_cc", "clark_blame_ch", "clark_blame_div"),
        axis=1,
    )

    raw_dataframe["first_vignette_clark_wrong"] = raw_dataframe.apply(
        lambda row: choose_first_case_value(row, "clark_wrong_cc", "clark_wrong_ch", "clark_wrong_div"),
        axis=1,
    )

    raw_dataframe["first_vignette_clark_punish"] = raw_dataframe.apply(
        lambda row: choose_first_case_value(row, "clark_punish_cc", "clark_punish_ch", "clark_punish_div"),
        axis=1,
    )

    "Within-subject deltas (Clark: CH-CC and DIV-CC; Proximate: CC-CH) for blame, wrongness, punishment"
    raw_dataframe["clark_blame_ch_minus_cc"] = raw_dataframe["clark_blame_ch"] - raw_dataframe["clark_blame_cc"]
    raw_dataframe["clark_blame_div_minus_cc"] = raw_dataframe["clark_blame_div"] - raw_dataframe["clark_blame_cc"]
    raw_dataframe["proximate_blame_cc_minus_ch"] = raw_dataframe["proximate_blame_cc"] - raw_dataframe["proximate_blame_ch"]

    raw_dataframe["clark_wrong_ch_minus_cc"] = raw_dataframe["clark_wrong_ch"] - raw_dataframe["clark_wrong_cc"]
    raw_dataframe["clark_wrong_div_minus_cc"] = raw_dataframe["clark_wrong_div"] - raw_dataframe["clark_wrong_cc"]
    raw_dataframe["proximate_wrong_cc_minus_ch"] = raw_dataframe["proximate_wrong_cc"] - raw_dataframe["proximate_wrong_ch"]

    raw_dataframe["clark_punish_ch_minus_cc"] = raw_dataframe["clark_punish_ch"] - raw_dataframe["clark_punish_cc"]
    raw_dataframe["clark_punish_div_minus_cc"] = raw_dataframe["clark_punish_div"] - raw_dataframe["clark_punish_cc"]
    raw_dataframe["proximate_punish_cc_minus_ch"] = raw_dataframe["proximate_punish_cc"] - raw_dataframe["proximate_punish_ch"]

    "Responsibility Shielding Effect (per preregistration): min(CH, DIV) - CC (computed within-subjects per participant)"
    raw_dataframe["clark_blame_min_ch_div_minus_cc"] = np.minimum(raw_dataframe["clark_blame_ch"], raw_dataframe["clark_blame_div"]) - raw_dataframe["clark_blame_cc"]
    raw_dataframe["clark_wrong_min_ch_div_minus_cc"] = np.minimum(raw_dataframe["clark_wrong_ch"], raw_dataframe["clark_wrong_div"]) - raw_dataframe["clark_wrong_cc"]
    raw_dataframe["clark_punish_min_ch_div_minus_cc"] = np.minimum(raw_dataframe["clark_punish_ch"], raw_dataframe["clark_punish_div"]) - raw_dataframe["clark_punish_cc"]

    "Interpersonal blame disparity within a given vignette (diagnostic, not the definition of shielding)"
    raw_dataframe["bill_minus_clark_cc_blame"] = raw_dataframe["proximate_blame_cc"] - raw_dataframe["clark_blame_cc"]
    raw_dataframe["bill_minus_clark_ch_blame"] = raw_dataframe["proximate_blame_ch"] - raw_dataframe["clark_blame_ch"]

    raw_dataframe["bill_minus_clark_cc_wrong"] = raw_dataframe["proximate_wrong_cc"] - raw_dataframe["clark_wrong_cc"]
    raw_dataframe["bill_minus_clark_ch_wrong"] = raw_dataframe["proximate_wrong_ch"] - raw_dataframe["clark_wrong_ch"]

    raw_dataframe["bill_minus_clark_cc_punish"] = raw_dataframe["proximate_punish_cc"] - raw_dataframe["clark_punish_cc"]
    raw_dataframe["bill_minus_clark_ch_punish"] = raw_dataframe["proximate_punish_ch"] - raw_dataframe["clark_punish_ch"]

    "2AFC: encode compact responses (Bill vs Clark; CH vs CC; DIV vs CC)"
    firework_interpersonal_bill = "Bill (the buyer who detonated the dangerous fireworks) is more blameworthy."
    firework_interpersonal_clark = "Clark (the clerk who enabled and sold the explosive charge) is more blameworthy."
    trolley_interpersonal_bill = "Bill (the operator at the second fork) is more blameworthy."
    trolley_interpersonal_clark = "Clark (the operator at the first fork) is more blameworthy."

    firework_ch_more = "Clark is more blameworthy in the story where Bill has brain damage."
    firework_cc_more_ch = "Clark is more blameworthy in the story where Bill is a mentally competent adult."
    trolley_ch_more = "Clark is more blameworthy in the story where the second fork is operated by a computerized switching mechanism."
    trolley_cc_more_ch = "Clark is more blameworthy in the story where the second fork is operated by Bill, another operator."

    firework_div_more = "Clark is more blameworthy in the story where he armed the flare canister remotely at the same time that Bill ignited it."
    firework_cc_more_div = "Clark is more blameworthy in the story where he armed the flare canister before selling it to Bill."
    trolley_div_more = "Clark is more blameworthy in the story where he flips the switch at the same time as Bill."
    trolley_cc_more_div = "Clark is more blameworthy in the story where he flips the switch before Bill."

    raw_dataframe["twoafc_bill_vs_clark"] = np.where(
        raw_dataframe["story_condition"] == "firework",
        raw_dataframe.apply(
            lambda row: encode_two_afc_compact_response(
                row["2afc_p_interperson_1"],
                row["2afc_p_interperson_2"],
                firework_interpersonal_bill,
                firework_interpersonal_clark,
                "Bill",
                "Clark",
            ),
            axis=1,
        ),
        raw_dataframe.apply(
            lambda row: encode_two_afc_compact_response(
                row["2afc_t_interperson_1"],
                row["2afc_t_interperson_2"],
                trolley_interpersonal_bill,
                trolley_interpersonal_clark,
                "Bill",
                "Clark",
            ),
            axis=1,
        ),
    )

    raw_dataframe["twoafc_ch_vs_cc"] = np.where(
        raw_dataframe["story_condition"] == "firework",
        raw_dataframe.apply(
            lambda row: encode_two_afc_compact_response(
                row["2afc_p_intraperson_1"],
                row["2afc_p_intraperson_2"],
                firework_ch_more,
                firework_cc_more_ch,
                "CH",
                "CC",
            ),
            axis=1,
        ),
        raw_dataframe.apply(
            lambda row: encode_two_afc_compact_response(
                row["2afc_t_intraperson_1"],
                row["2afc_t_intraperson_2"],
                trolley_ch_more,
                trolley_cc_more_ch,
                "CH",
                "CC",
            ),
            axis=1,
        ),
    )

    raw_dataframe["twoafc_div_vs_cc"] = np.where(
        raw_dataframe["story_condition"] == "firework",
        raw_dataframe.apply(
            lambda row: encode_two_afc_compact_response(
                row["2afc_p_intraperson_3"],
                row["2afc_p_intraperson_4"],
                firework_div_more,
                firework_cc_more_div,
                "DIV",
                "CC",
            ),
            axis=1,
        ),
        raw_dataframe.apply(
            lambda row: encode_two_afc_compact_response(
                row["2afc_t_intraperson_3"],
                row["2afc_t_intraperson_4"],
                trolley_div_more,
                trolley_cc_more_div,
                "DIV",
                "CC",
            ),
            axis=1,
        ),
    )

    "CRT score"
    raw_dataframe["crt_score"] = raw_dataframe.apply(
        lambda row: compute_crt_score(row.get("crt_bat_ball"), row.get("crt_widgets"), row.get("crt_lilly_pads")),
        axis=1,
    )

    "INDCOL composite scores"
    indcol_item_columns = [column_name for column_name in raw_dataframe.columns if str(column_name).startswith("indcol_")]
    indcol_series_dict = {column_name: raw_dataframe[column_name] for column_name in indcol_item_columns}
    indcol_scores_df = compute_indcol_scores(indcol_series_dict)

    raw_dataframe = pd.concat([raw_dataframe, indcol_scores_df], axis=1)

    "Included column is True iff the participant completed and passed all comprehension checks"
    raw_dataframe["included"] = raw_dataframe["comprehension_all_correct_bool"]

    "Compress race/political columns"
    def compress_self_describe(base_value, self_describe_text_value) -> str:
        if pd.isna(base_value):
            return np.nan
        base_string = str(base_value).strip()
        if base_string.lower() == "self describe":
            if pd.isna(self_describe_text_value):
                return "Self describe: (blank)"
            text_string = str(self_describe_text_value).strip()
            return f"Self describe: {text_string}"
        return base_string

    raw_dataframe["race_compact"] = raw_dataframe.apply(lambda row: compress_self_describe(row.get("Race"), row.get("Race_7_TEXT")), axis=1)
    raw_dataframe["political_compact"] = raw_dataframe.apply(lambda row: compress_self_describe(row.get("Political"), row.get("Political_6_TEXT")), axis=1)

    "Timing columns: keep First/Last Click, merge across story types, drop Page Submit and Click Count"
    timing_blocks = [
        ("clark_cc", "pucc", "tucc"),
        ("clark_ch", "puch", "tuch"),
        ("clark_div", "pudiv", "tudiv"),
        ("proximate_cc", "ppcc", "tpcc"),
        ("proximate_ch", "ppch", "tpch"),
        ("comprehension", "comp_p", "comp_t"),
    ]

    for unified_prefix, firework_prefix, trolley_prefix in timing_blocks:
        first_click_column_firework = f"{firework_prefix}_timing_First Click"
        last_click_column_firework = f"{firework_prefix}_timing_Last Click"

        first_click_column_trolley = f"{trolley_prefix}_timing_First Click"
        last_click_column_trolley = f"{trolley_prefix}_timing_Last Click"

        unified_first_click = f"{unified_prefix}_timing_first_click_seconds"
        unified_last_click = f"{unified_prefix}_timing_last_click_seconds"
        unified_last_minus_first = f"{unified_prefix}_timing_last_minus_first_seconds"

        raw_dataframe[unified_first_click] = coalesce_series(
            pd.to_numeric(raw_dataframe.get(first_click_column_firework), errors="coerce"),
            pd.to_numeric(raw_dataframe.get(first_click_column_trolley), errors="coerce"),
        )

        raw_dataframe[unified_last_click] = coalesce_series(
            pd.to_numeric(raw_dataframe.get(last_click_column_firework), errors="coerce"),
            pd.to_numeric(raw_dataframe.get(last_click_column_trolley), errors="coerce"),
        )

        raw_dataframe[unified_last_minus_first] = raw_dataframe[unified_last_click] - raw_dataframe[unified_first_click]

    "Drop story-specific raw columns that are now harmonized, plus free response text fields"
    columns_to_drop = []

    columns_to_drop += [
        "StartDate",
        "EndDate",
        "Duration (in seconds)",
        "RecordedDate",
        "ExternalReference",
        "DistributionChannel",
        "UserLanguage",
        "finished_bool",
        "progress_numeric",
        "Finished",
        "Progress",
        "StoryCondition",
        "LoadCondition",
        "CaseOrder",
        "Status",
        "Race_7_TEXT",
        "Political_6_TEXT",
        "user_feedback",
        "RecipientLastName",
        "RecipientFirstName",
        "RecipientEmail",
    ]

    "Outcome columns (story-specific)"
    columns_to_drop += [
        "pucc_blame_1", "puch_blame_1", "pudiv_blame_1",
        "tucc_blame_1", "tuch_blame_1", "tudiv_blame_1",
        "pucc_wrong_1", "puch_wrong_1", "pudiv_wrong_1",
        "tucc_wrong_1", "tuch_wrong_1", "tudiv_wrong_1",
        "pucc_punish", "puch_punish", "pudiv_punish",
        "tucc_punish", "tuch_punish", "tudiv_punish",
        "ppcc_blame_1", "ppch_blame_1",
        "tpcc_blame_1", "tpch_blame_1",
        "ppcc_wrong_1", "ppch_wrong_1",
        "tpcc_wrong_1", "tpch_wrong_1",
        "ppcc_punish", "ppch_punish",
        "tpcc_punish", "tpch_punish",
    ]

    "Comprehension story-specific"
    columns_to_drop += [
        "comp_p_prob_harm", "comp_p_agency_across", "comp_p_agency_within",
        "comp_t_prob_harm", "comp_t_agency_across", "comp_t_agency_within",
    ]

    "Timing story-specific (drop First/Last Click, Page Submit, Click Count; keep only unified timing columns)"
    for column_name in raw_dataframe.columns:
        column_string = str(column_name)
        if (
            column_string.endswith("_timing_First Click")
            or column_string.endswith("_timing_Last Click")
            or column_string.endswith("_timing_Page Submit")
            or column_string.endswith("_timing_Click Count")
            or column_string.endswith(" Submit")
            or column_string.endswith(" Count")
            or column_string.endswith(" Click")
        ):
            columns_to_drop.append(column_name)

    columns_to_drop += [
        "comp_p_timing_First Click", "comp_p_timing_Last Click",
        "comp_t_timing_First Click", "comp_t_timing_Last Click",
    ]

    "2AFC raw columns"
    twoafc_raw_columns = [column_name for column_name in raw_dataframe.columns if str(column_name).startswith("2afc_")]
    columns_to_drop += twoafc_raw_columns

    "Race/Political raw columns"
    columns_to_drop += ["Race", "Political"]

    "De-duplicate drop list and drop safely"
    columns_to_drop = list(dict.fromkeys(columns_to_drop))

    cleaned_dataframe = raw_dataframe.drop(columns=[column_name for column_name in columns_to_drop if column_name in raw_dataframe.columns])

    "Restrict rows to the frozen manuscript collection window."
    cleaned_dataframe = apply_collection_window_freeze(
        raw_dataframe=cleaned_dataframe,
        general_settings=general_settings,
    )

    "Remove unnecessary identifying Qualtrics columns before saving."
    cleaned_dataframe = drop_identifying_columns_from_cleaned_dataframe(
        cleaned_dataframe=cleaned_dataframe,
    )

    "Rename remaining columns to snake_case"
    cleaned_dataframe = cleaned_dataframe.rename(columns={column_name: convert_string_to_snake_case(column_name) for column_name in cleaned_dataframe.columns})

    "Manual fixups for a few Qualtrics camel-case edge cases"
    if "ipaddress" in cleaned_dataframe.columns:
        cleaned_dataframe = cleaned_dataframe.rename(columns={"ipaddress": "ip_address"})

    "Reorder columns to match preferred layout"
    preferred_prefix_order = [
        "response_id",
        "start_date",
        "end_date",
        "story_condition",
        "load_condition",
        "cognitive_load_digits_response",
        "cognitive_load_digits_correct_bool",
        "case_order",
        "case_code_position_1",
        "case_code_position_2",
        "case_code_position_3",
        "first_vignette_clark_blame",
        "first_vignette_clark_wrong",
        "first_vignette_clark_punish",
        "clark_blame_ch_minus_cc",
        "clark_blame_div_minus_cc",
        "proximate_blame_cc_minus_ch",
        "clark_wrong_ch_minus_cc",
        "clark_wrong_div_minus_cc",
        "proximate_wrong_cc_minus_ch",
        "clark_punish_ch_minus_cc",
        "clark_punish_div_minus_cc",
        "proximate_punish_cc_minus_ch",
        "clark_blame_min_ch_div_minus_cc",
        "clark_wrong_min_ch_div_minus_cc",
        "clark_punish_min_ch_div_minus_cc",
        "twoafc_bill_vs_clark",
        "twoafc_ch_vs_cc",
        "twoafc_div_vs_cc",
        "clark_blame_cc",
        "clark_blame_ch",
        "clark_blame_div",
        "clark_wrong_cc",
        "clark_wrong_ch",
        "clark_wrong_div",
        "clark_punish_cc",
        "clark_punish_ch",
        "clark_punish_div",
        "proximate_blame_cc",
        "proximate_blame_ch",
        "proximate_wrong_cc",
        "proximate_wrong_ch",
        "proximate_punish_cc",
        "proximate_punish_ch",
        "comprehension_probability_same_bool",
        "comprehension_clark_necessary_bool",
        "comprehension_bill_necessary_bool",
        "comprehension_all_correct_bool",
        "crt_score",
        "individualism_horizontal",
        "individualism_vertical",
        "individualism_score",
        "included",
        "age",
        "gender",
        "race_compact",
        "political_compact",
        "ip_address",
        "location_latitude",
        "location_longitude",
        "duration_seconds",
    ]

    timing_suffixes = [column_name for column_name in cleaned_dataframe.columns if "timing_" in column_name]
    other_columns = [column_name for column_name in cleaned_dataframe.columns if column_name not in preferred_prefix_order and column_name not in timing_suffixes]

    reordered_columns = [column_name for column_name in preferred_prefix_order if column_name in cleaned_dataframe.columns]
    reordered_columns += other_columns
    reordered_columns += timing_suffixes

    cleaned_dataframe = cleaned_dataframe[reordered_columns].copy()

    return cleaned_dataframe


def load_or_build_cleaned_dataframe(general_settings: GeneralSettings, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Extracts the preprocessed data CSV or builds it from scratch.

    Arguments:
        • general_settings: GeneralSettings; Bundle of variables used throughout analysis.
            - Contains file_paths and file_names used to extract files.
        • force_rebuild: bool
            - If True, rebuild the cleaned CSV from the raw export even if the cleaned CSV exists.

    Returns:
        • pd.DataFrame: Cleaned dataframe used by the main analysis pipeline.

    Raises:
        • FileNotFoundError: If neither the cleaned CSV nor the raw export CSV can be found.
    """
    file_path_cleaned = general_settings["filing"]["file_paths"]["processed"] / \
        general_settings["filing"]["file_names"]["cleaned"]

    if file_path_cleaned.exists() and not force_rebuild:
        return pd.read_csv(file_path_cleaned)
    
    file_name_raw_data = general_settings["filing"]["file_names"]["raw_data"]
    file_path_raw_data = general_settings["filing"]["file_paths"]["raw_data"] / file_name_raw_data

    if not file_path_raw_data.exists():
        raise FileNotFoundError(
            "Could not find the cleaned analysis CSV: "
            f"{file_path_raw_data}."
        )
    
    cleaned_dataframe = preprocess_raw_qualtrics_export(general_settings=general_settings)
    cleaned_dataframe.to_csv(file_path_cleaned, index=False, encoding="utf-8-sig")
    return cleaned_dataframe


def load_analysis_dataframe(general_settings: GeneralSettings, file_name_key: str, force_rebuild: bool | None = None) -> dict[str, bool | str | pd.DataFrame | None]:
    """
    Loads a dataframe from the 'preprocess' folder if not force_rebuild and it can be found.

    Arguments:
        • general_settings: GeneralSettings
            - Long dataframe subset for one delta type.
        • file_name_key: str
            - Subplot column index.
        • force_rebuild: bool | None
            - Base hue for this panel's two load conditions.

    Returns:
        • dict[str, bool | str | pd.DataFrame | None]
            - {
                'success': bool; True if dataframe successfully extracted,
                'error': bool; True if extraction errored,
                'message': str; Explanation of success or error,
                'dataframe' pd.DataFrame; Dataframe if successful and None otherwise
            }
    """
    if force_rebuild:
        message = f"Skipping loading CSV."
        return {
            "success": False, 
            "error": False,
            "message": message, 
            "dataframe": None
        }

    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]
    if not isinstance(force_rebuild, bool):
        message = f"TypeError: force_rebuild must be type<bool>, not {type(force_rebuild)}."
        return {
            "success": False, 
            "error": True, 
            "message": message, 
            "dataframe": None
        }
    
    valid_file_name_keys = list(general_settings["filing"]["file_names"].keys())
    if file_name_key not in valid_file_name_keys:
        message = f"KeyError: file_name_key {file_name_key} not found in {valid_file_name_keys}."
        return {
            "success": False, 
            "error": True, 
            "message": message, 
            "dataframe": None
        }
    
    file_path_preprocessing = general_settings["filing"]["file_paths"]["processed"]
    file_name = general_settings["filing"]["file_names"][file_name_key]
    full_path = file_path_preprocessing / file_name

    if not os.path.exists(path=full_path):
        message = f"PathError: Cannot find file: {full_path}."
        return {
            "success": False, 
            "error": True, 
            "message": message, 
            "dataframe": None
        }
    
    dataframe = pd.read_csv(full_path, encoding="utf-8-sig", engine="python")
    if "Unnamed: 0" in dataframe.columns:
        dataframe = dataframe.drop(columns=["Unnamed: 0"])
    message = f"Successfully loaded dataframe {file_name}!"
    return {
        "success": True, 
        "error": False, 
        "message": message, 
        "dataframe": dataframe
    }    


"=========================================================================================="
"================================= Core Analysis Helpers =================================="
"=========================================================================================="

def compute_welch_degrees_of_freedom(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """
    Computes Welch-Satterthwaite degrees of freedom.

    Arguments:
        • sample_a:
            Numeric vector.
        • sample_b:
            Numeric vector.

    Returns:
        Welch df.
    """
    sample_a = sample_a[~np.isnan(sample_a)]
    sample_b = sample_b[~np.isnan(sample_b)]

    n_a = sample_a.shape[0]
    n_b = sample_b.shape[0]

    var_a = np.var(sample_a, ddof=1)
    var_b = np.var(sample_b, ddof=1)

    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)

    return float(numerator / denominator)


def hedges_g_for_two_independent_samples(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """
    Computes Hedges g for two independent samples.

    Arguments:
        • sample_a:
            Numeric vector.
        • sample_b:
            Numeric vector.

    Returns:
        Hedges g (sample_a mean - sample_b mean, standardized).
    """
    sample_a = sample_a[~np.isnan(sample_a)]
    sample_b = sample_b[~np.isnan(sample_b)]

    n_a = sample_a.shape[0]
    n_b = sample_b.shape[0]

    mean_a = np.mean(sample_a)
    mean_b = np.mean(sample_b)

    var_a = np.var(sample_a, ddof=1)
    var_b = np.var(sample_b, ddof=1)

    pooled_standard_deviation = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_standard_deviation == 0:
        return np.nan

    cohen_d = (mean_a - mean_b) / pooled_standard_deviation

    correction_factor = 1 - (3 / (4 * (n_a + n_b) - 9))

    return float(cohen_d * correction_factor)


def cohens_d_for_two_independent_samples(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """
    Computes Cohen's d for two independent samples.

    Arguments:
        • sample_a: np.ndarray
            First numeric sample.
        • sample_b: np.ndarray
            Second numeric sample.

    Notes:
        • This is the uncorrected standardized mean difference.
        • I am using this here because the preregistration explicitly named Cohen's d.

    Returns:
        • float
            Cohen's d for mean(sample_a) - mean(sample_b).
    """
    "======================================="
    "Drop missing values and compute inputs"
    "======================================="
    sample_a = sample_a[~np.isnan(sample_a)]
    sample_b = sample_b[~np.isnan(sample_b)]

    n_a = sample_a.shape[0]
    n_b = sample_b.shape[0]

    mean_a = np.mean(sample_a)
    mean_b = np.mean(sample_b)

    variance_a = np.var(sample_a, ddof=1)
    variance_b = np.var(sample_b, ddof=1)

    pooled_standard_deviation = np.sqrt(
        ((n_a - 1) * variance_a + (n_b - 1) * variance_b) / (n_a + n_b - 2)
    )

    if pooled_standard_deviation == 0:
        return np.nan

    return float((mean_a - mean_b) / pooled_standard_deviation)


def compute_welch_mean_difference_ci(sample_a: np.ndarray, sample_b: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float, float, float]:
    """
    Computes a confidence interval for mean(sample_a) - mean(sample_b) using Welch standard error.

    Returns:
        • difference, ci_lower, ci_upper, df
    """
    sample_a = sample_a[~np.isnan(sample_a)]
    sample_b = sample_b[~np.isnan(sample_b)]

    n_a = sample_a.shape[0]
    n_b = sample_b.shape[0]

    mean_a = np.mean(sample_a)
    mean_b = np.mean(sample_b)

    var_a = np.var(sample_a, ddof=1)
    var_b = np.var(sample_b, ddof=1)

    difference = float(mean_a - mean_b)

    standard_error = np.sqrt(var_a / n_a + var_b / n_b)

    welch_df = compute_welch_degrees_of_freedom(sample_a, sample_b)

    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha / 2, df=welch_df)

    ci_lower = float(difference - t_critical * standard_error)
    ci_upper = float(difference + t_critical * standard_error)

    return difference, ci_lower, ci_upper, float(welch_df)


def holm_bonferroni_correct_p_values(p_values: Sequence[float]) -> List[float]:
    """
    Applies Holm-Bonferroni correction to a list of p-values.

    Returns:
        • List of adjusted p-values, aligned with the input order.
    """
    p_values_array = np.asarray(p_values, dtype=float)

    sorted_indices = np.argsort(p_values_array)
    adjusted = np.empty_like(p_values_array)

    m = p_values_array.shape[0]

    for rank, index_value in enumerate(sorted_indices):
        adjusted[index_value] = min((m - rank) * p_values_array[index_value], 1.0)

    return adjusted.tolist()


def run_welch_t_test_between_groups(dataframe: pd.DataFrame, dv_column_name: str, group_column_name: str, group_a_value: str, group_b_value: str) -> Dict:
    """
    Runs a Welch two-sample t-test on dv_column_name comparing group_a_value vs group_b_value.

    Returns:
        • Dict with test summary (including Hedges g, CI, df).
    """
    sample_a = pd.to_numeric(dataframe[dataframe[group_column_name] == group_a_value][dv_column_name], errors="coerce").to_numpy(dtype=float)
    sample_b = pd.to_numeric(dataframe[dataframe[group_column_name] == group_b_value][dv_column_name], errors="coerce").to_numpy(dtype=float)

    sample_a = sample_a[~np.isnan(sample_a)]
    sample_b = sample_b[~np.isnan(sample_b)]

    t_statistic, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)

    mean_a = float(np.mean(sample_a)) if sample_a.shape[0] > 0 else np.nan
    mean_b = float(np.mean(sample_b)) if sample_b.shape[0] > 0 else np.nan

    difference, ci_lower, ci_upper, welch_df = compute_welch_mean_difference_ci(sample_a, sample_b)

    hedges_g = hedges_g_for_two_independent_samples(sample_a, sample_b)

    return {
        "test_type": "welch_t_test_independent",
        "dv": dv_column_name,
        "group_column": group_column_name,
        "group_a": group_a_value,
        "group_b": group_b_value,
        "n_a": int(sample_a.shape[0]),
        "n_b": int(sample_b.shape[0]),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_difference_a_minus_b": difference,
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "t_statistic": float(t_statistic),
        "df": float(welch_df),
        "p_value": float(p_value),
        "effect_size_name": "hedges_g",
        "effect_size": float(hedges_g),
    }


def run_pooled_ols_planned_contrasts(dataframe: pd.DataFrame, dv_column_name: str, group_column_name: str = "case_code_position_1", covariance_type: str | None = None) -> pd.DataFrame:
    """
    Fits the preregistered pooled OLS / one-way ANOVA style model and returns the two planned contrasts.

    Arguments:
        • dataframe: pd.DataFrame
            The dataframe containing the dependent variable and the three-level group factor.
        • dv_column_name: str
            The dependent variable column to analyze.
        • group_column_name: str = "case_code_position_1"
            The grouping factor with levels CC, CH, and DIV.
        • covariance_type: str | None = None
            Optional statsmodels covariance type. If None, uses the classical pooled-variance OLS fit.
            If you later want a robustness variant, you can set this to "HC3".

    Notes:
        • This is the closest implementation of what the preregistration says for the confirmatory
          between-subjects analysis: one pooled first-vignette model with planned contrasts.
        • The returned rows have the same basic structure as the Welch rows so they can drop into the
          existing pipeline with minimal disruption.

    Returns:
        • pd.DataFrame
            One row for CH vs CC and one row for DIV vs CC.
    """
    "======================================="
    "Prepare the pooled first-vignette data"
    "======================================="
    analysis_dataframe = dataframe[[dv_column_name, group_column_name]].copy()

    analysis_dataframe[dv_column_name] = pd.to_numeric(
        analysis_dataframe[dv_column_name],
        errors="coerce",
    )

    analysis_dataframe[group_column_name] = (
        analysis_dataframe[group_column_name]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    analysis_dataframe = analysis_dataframe.loc[
        analysis_dataframe[group_column_name].isin(["CC", "CH", "DIV"])
    ].dropna(subset=[dv_column_name, group_column_name]).copy()

    analysis_dataframe[group_column_name] = pd.Categorical(
        analysis_dataframe[group_column_name],
        categories=["CC", "CH", "DIV"],
        ordered=True,
    )

    if analysis_dataframe.shape[0] == 0:
        raise ValueError("No valid rows remain for the pooled OLS planned-contrast analysis.")

    "============================"
    "Fit the pooled OLS model"
    "============================"
    formula_string = f"{dv_column_name} ~ C({group_column_name}, Treatment(reference='CC'))"

    if covariance_type is None:
        fitted_pooled_ols_model = smf.ols(
            formula=formula_string,
            data=analysis_dataframe,
        ).fit()
        covariance_type_label = "classical"
    else:
        fitted_pooled_ols_model = smf.ols(
            formula=formula_string,
            data=analysis_dataframe,
        ).fit(cov_type=covariance_type)
        covariance_type_label = covariance_type

    "========================================="
    "Extract the two preregistered contrasts"
    "========================================="
    term_name_by_group = {
        "CH": f"C({group_column_name}, Treatment(reference='CC'))[T.CH]",
        "DIV": f"C({group_column_name}, Treatment(reference='CC'))[T.DIV]",
    }

    contrast_rows: list[dict[str, Any]] = []

    for group_a_value, group_b_value in [("CH", "CC"), ("DIV", "CC")]:
        term_name = term_name_by_group[group_a_value]

        sample_a = pd.to_numeric(
            analysis_dataframe.loc[analysis_dataframe[group_column_name] == group_a_value, dv_column_name],
            errors="coerce",
        ).to_numpy(dtype=float)

        sample_b = pd.to_numeric(
            analysis_dataframe.loc[analysis_dataframe[group_column_name] == group_b_value, dv_column_name],
            errors="coerce",
        ).to_numpy(dtype=float)

        sample_a = sample_a[~np.isnan(sample_a)]
        sample_b = sample_b[~np.isnan(sample_b)]

        contrast_rows.append(
            {
                "test_type": "pooled_ols_planned_contrast",
                "dv": dv_column_name,
                "group_column": group_column_name,
                "group_a": group_a_value,
                "group_b": group_b_value,
                "n_a": int(sample_a.shape[0]),
                "n_b": int(sample_b.shape[0]),
                "mean_a": float(np.mean(sample_a)),
                "mean_b": float(np.mean(sample_b)),
                "mean_difference_a_minus_b": float(fitted_pooled_ols_model.params[term_name]),
                "ci95_lower": float(fitted_pooled_ols_model.conf_int().loc[term_name, 0]),
                "ci95_upper": float(fitted_pooled_ols_model.conf_int().loc[term_name, 1]),
                "t_statistic": float(fitted_pooled_ols_model.tvalues[term_name]),
                "df": float(fitted_pooled_ols_model.df_resid),
                "p_value": float(fitted_pooled_ols_model.pvalues[term_name]),
                "effect_size_name": "cohens_d",
                "effect_size": float(cohens_d_for_two_independent_samples(sample_a, sample_b)),
                "model_formula": formula_string,
                "model_covariance_type": covariance_type_label,
            }
        )

    return pd.DataFrame(contrast_rows)


def run_one_sample_t_test_on_delta(dataframe: pd.DataFrame, delta_column_name: str) -> Dict:
    """
    One-sample t-test of a delta column against 0, equivalent to a paired test.

    Returns:
        • Dict with test summary and Cohen's dz.
    """
    delta_values = pd.to_numeric(dataframe[delta_column_name], errors="coerce").to_numpy(dtype=float)
    delta_values = delta_values[~np.isnan(delta_values)]

    t_statistic, p_value = stats.ttest_1samp(delta_values, popmean=0.0)

    n = int(delta_values.shape[0])
    mean_delta = float(np.mean(delta_values))
    std_delta = float(np.std(delta_values, ddof=1))

    dz = np.nan
    if std_delta > 0:
        dz = float(mean_delta / std_delta)

    standard_error = std_delta / np.sqrt(n) if n > 0 else np.nan
    df_value = float(n - 1)

    t_critical = stats.t.ppf(0.975, df=df_value) if n > 1 else np.nan
    ci_lower = float(mean_delta - t_critical * standard_error) if n > 1 else np.nan
    ci_upper = float(mean_delta + t_critical * standard_error) if n > 1 else np.nan

    return {
        "test_type": "t_test_onesample_delta",
        "dv": delta_column_name,
        "group_column": "(delta_vs_0)",
        "group_a": "delta",
        "group_b": "0",
        "n_a": n,
        "n_b": np.nan,
        "mean_a": mean_delta,
        "mean_b": 0.0,
        "mean_difference_a_minus_b": mean_delta,
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "t_statistic": float(t_statistic),
        "df": df_value,
        "p_value": float(p_value),
        "effect_size_name": "cohens_dz",
        "effect_size": dz,
    }


def _save_analysis_dataframe_to_processed_folder(dataframe_to_save: pd.DataFrame, general_settings: dict[str, Any], file_name_key: str) -> None:
    """
    Save one analysis dataframe to the processed folder.

    Arguments:
        • dataframe_to_save: pd.DataFrame
            - Dataframe to save.
        • general_settings: dict[str, Any]
            - Master project settings dictionary.
        • file_name_key: str
            - Key inside general_settings["filing"]["file_names"].
    """
    file_path_output: Path = (
        general_settings["filing"]["file_paths"]["processed"]
        / general_settings["filing"]["file_names"][file_name_key]
    )
    file_path_output.parent.mkdir(parents=True, exist_ok=True)
    dataframe_to_save.to_csv(file_path_output, index=False, encoding="utf-8-sig")


"=========================================================================================="
"================================ Core Analysis Functions ================================="
"=========================================================================================="

def compute_group_summaries(general_settings: GeneralSettings, force_rebuild: bool = None) -> pd.DataFrame:
    """
    Produces long-format descriptive statistics for Plotly-friendly grouping.

    Returns:
        • DataFrame with columns:
            - inclusion_filter
            - story_condition
            - load_condition
            - analysis_scope
            - agent_role
            - dv
            - condition
            - n
            - mean
            - median
            - std
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    "Load and return dataframe if one already exists and not directed to rebuild."
    group_summaries_extraction = load_analysis_dataframe(
        general_settings=general_settings, file_name_key="group_summaries", force_rebuild=force_rebuild)
    if group_summaries_extraction["success"]:
        group_summaries_dataframe: pd.DataFrame = group_summaries_extraction["dataframe"]
        return group_summaries_dataframe
    if group_summaries_extraction["error"]:
        raise Exception(group_summaries_extraction["message"])

    "Load of rebuild preprocessed dataframe"
    cleaned_dataframe = load_or_build_cleaned_dataframe(general_settings=general_settings, force_rebuild=False)

    dataframes = {}
    for inclusion_filter_value_and_analysis_dataframe in [
            ("included_only", cleaned_dataframe[cleaned_dataframe["included"]].copy()), 
            ("all_finishers", cleaned_dataframe.copy())
        ]:
        inclusion_filter_value, analysis_dataframe = inclusion_filter_value_and_analysis_dataframe

        summary_rows = []

        "Within-subject condition means (each participant contributes to each condition once)"
        dv_map = {
            "blame": ("clark_blame_cc", "clark_blame_ch", "clark_blame_div"),
            "wrongness": ("clark_wrong_cc", "clark_wrong_ch", "clark_wrong_div"),
            "punishment": ("clark_punish_cc", "clark_punish_ch", "clark_punish_div"),
        }

        for story_condition_value in ["all", "firework", "trolley"]:
            for load_condition_value in ["all", "high", "low"]:
                subset = analysis_dataframe.copy()

                if story_condition_value != "all":
                    subset = subset[subset["story_condition"] == story_condition_value]

                if load_condition_value != "all":
                    subset = subset[subset["load_condition"] == load_condition_value]

                for dv_name, (column_cc, column_ch, column_div) in dv_map.items():
                    for condition_code, column_name in [("CC", column_cc), ("CH", column_ch), ("DIV", column_div)]:
                        values = pd.to_numeric(subset[column_name], errors="coerce")
                        summary_rows.append(
                            {
                                "inclusion_filter": inclusion_filter_value,
                                "story_condition": story_condition_value,
                                "load_condition": load_condition_value,
                                "analysis_scope": "within_subjects_all_vignettes",
                                "agent_role": "clark",
                                "dv": dv_name,
                                "condition": condition_code,
                                "n": int(values.dropna().shape[0]),
                                "mean": float(values.mean()),
                                "median": float(values.median()),
                                "std": float(values.std(ddof=1)),
                            }
                        )

                "Proximate ratings exist for CC and CH"
                proximate_map = {
                    "blame": ("proximate_blame_cc", "proximate_blame_ch"),
                    "wrongness": ("proximate_wrong_cc", "proximate_wrong_ch"),
                    "punishment": ("proximate_punish_cc", "proximate_punish_ch"),
                }

                for dv_name, (column_cc, column_ch) in proximate_map.items():
                    for condition_code, column_name in [("CC", column_cc), ("CH", column_ch)]:
                        values = pd.to_numeric(subset[column_name], errors="coerce")
                        summary_rows.append(
                            {
                                "inclusion_filter": inclusion_filter_value,
                                "story_condition": story_condition_value,
                                "load_condition": load_condition_value,
                                "analysis_scope": "within_subjects_all_vignettes",
                                "agent_role": "proximate",
                                "dv": dv_name,
                                "condition": condition_code,
                                "n": int(values.dropna().shape[0]),
                                "mean": float(values.mean()),
                                "median": float(values.median()),
                                "std": float(values.std(ddof=1)),
                            }
                        )

                "Between-subject first-vignette summaries"
                for dv_name, first_column_name in [
                    ("blame", "first_vignette_clark_blame"),
                    ("wrongness", "first_vignette_clark_wrong"),
                    ("punishment", "first_vignette_clark_punish"),
                ]:
                    for condition_code in ["CC", "CH", "DIV"]:
                        subset_first = subset[subset["case_code_position_1"] == condition_code]
                        values = pd.to_numeric(subset_first[first_column_name], errors="coerce")
                        summary_rows.append(
                            {
                                "inclusion_filter": inclusion_filter_value,
                                "story_condition": story_condition_value,
                                "load_condition": load_condition_value,
                                "analysis_scope": "between_subjects_first_vignette",
                                "agent_role": "clark",
                                "dv": dv_name,
                                "condition": condition_code,
                                "n": int(values.dropna().shape[0]),
                                "mean": float(values.mean()),
                                "median": float(values.median()),
                                "std": float(values.std(ddof=1)),
                            }
                        )

                "Delta summaries (within-subject)"
                delta_map = {
                    "blame": ("clark_blame_ch_minus_cc", "clark_blame_div_minus_cc", "proximate_blame_cc_minus_ch", "clark_blame_min_ch_div_minus_cc"),
                    "wrongness": ("clark_wrong_ch_minus_cc", "clark_wrong_div_minus_cc", "proximate_wrong_cc_minus_ch", "clark_wrong_min_ch_div_minus_cc"),
                    "punishment": ("clark_punish_ch_minus_cc", "clark_punish_div_minus_cc", "proximate_punish_cc_minus_ch", "clark_punish_min_ch_div_minus_cc"),
                }

                for dv_name, (delta_ch, delta_div, delta_prox, delta_min) in delta_map.items():
                    for delta_name, column_name in [
                        ("CH-CC", delta_ch),
                        ("DIV-CC", delta_div),
                        ("MIN(CH,DIV)-CC", delta_min),
                        ("CC-CH_proximate", delta_prox),
                    ]:
                        values = pd.to_numeric(subset[column_name], errors="coerce")
                        summary_rows.append(
                            {
                                "inclusion_filter": inclusion_filter_value,
                                "story_condition": story_condition_value,
                                "load_condition": load_condition_value,
                                "analysis_scope": "within_subjects_deltas",
                                "agent_role": "clark" if "proximate" not in delta_name else "proximate",
                                "dv": dv_name,
                                "condition": delta_name,
                                "n": int(values.dropna().shape[0]),
                                "mean": float(values.mean()),
                                "median": float(values.median()),
                                "std": float(values.std(ddof=1)),
                            }
                        )

            dataframes[inclusion_filter_value] = pd.DataFrame(summary_rows)

    group_summaries = pd.concat(list(dataframe for dataframe in dataframes.values()), axis=0, ignore_index=True)
    _save_analysis_dataframe_to_processed_folder(dataframe_to_save=group_summaries, 
                                                 general_settings=general_settings, file_name_key="group_summaries")

    return group_summaries


def compute_twoafc_counts(general_settings: GeneralSettings, force_rebuild: bool | None = None, table_form: bool = False) -> pd.DataFrame:
    """
    Produces 2AFC frequency tables matching the manuscript-ready style.

    Returns:
        • DataFrame with columns:
            - inclusion_filter
            - story_condition
            - comparison
            - operator
            - count
          Or, if table_form=True:
            - inclusion_filter
            - story_condition
            - operator
            - Bill ? Clark
            - CH ? CC
            - DIV ? CC
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    def pivot_twoafc_table(df: pd.DataFrame) -> pd.DataFrame:
        """Pivots the long-form 2AFC counts into a wide table with comparisons as columns."""
        comparison_order = ["Bill ? Clark", "CH ? CC", "DIV ? CC"]
        operator_order = [">", "≥", "≤", "<"]

        pivoted = df.pivot_table(
            index=["inclusion_filter", "story_condition", "operator"],
            columns="comparison",
            values="count",
            aggfunc="sum",
        ).reset_index()

        pivoted.columns.name = None

        pivoted = pivoted[["inclusion_filter", "story_condition", "operator"] + comparison_order]
        pivoted["operator"] = pd.Categorical(pivoted["operator"], categories=operator_order, ordered=True)
        pivoted = pivoted.sort_values(["inclusion_filter", "story_condition", "operator"]).reset_index(drop=True)

        return pivoted

    "Determine which file_name_key to use based on table_form"
    file_name_key = "afc_counts_table" if table_form else "afc_counts_long"

    "Load and return dataframe if one already exists and not directed to rebuild."
    group_summaries_extraction = load_analysis_dataframe(
        general_settings=general_settings, file_name_key=file_name_key, force_rebuild=force_rebuild)
    if group_summaries_extraction["success"]:
        return group_summaries_extraction["dataframe"]
    if group_summaries_extraction["error"]:
        raise Exception(group_summaries_extraction["message"])

    "Load or rebuild preprocessed dataframe"
    cleaned_dataframe = load_or_build_cleaned_dataframe(general_settings=general_settings, force_rebuild=False)

    dataframes = {}
    for inclusion_filter_and_analysis_dataframe in [
            ("included_only", cleaned_dataframe[cleaned_dataframe["included"]].copy()),
            ("all_finishers", cleaned_dataframe.copy())
        ]:
        inclusion_filter, analysis_dataframe = inclusion_filter_and_analysis_dataframe

        summary_rows = []

        def add_counts_for_subset(subset: pd.DataFrame, story_condition_value: str):
            for comparison_column, left_prefix, right_prefix, comparison_label in [
                ("twoafc_bill_vs_clark", "Bill", "Clark", "Bill ? Clark"),
                ("twoafc_ch_vs_cc", "CH", "CC", "CH ? CC"),
                ("twoafc_div_vs_cc", "DIV", "CC", "DIV ? CC"),
            ]:
                counts = subset[comparison_column].value_counts(dropna=False)

                ordered_categories = [
                    f"{left_prefix} > {right_prefix}",
                    f"{left_prefix} ≥ {right_prefix}",
                    f"{left_prefix} ≤ {right_prefix}",
                    f"{left_prefix} < {right_prefix}",
                ]
                operators = [">", "≥", "≤", "<"]

                for operator_symbol, category in zip(operators, ordered_categories):
                    summary_rows.append(
                        {
                            "inclusion_filter": inclusion_filter,
                            "story_condition": story_condition_value,
                            "comparison": comparison_label,
                            "operator": operator_symbol,
                            "count": int(counts.get(category, 0)),
                        }
                    )

        add_counts_for_subset(analysis_dataframe, "all")
        add_counts_for_subset(analysis_dataframe[analysis_dataframe["story_condition"] == "firework"], "firework")
        add_counts_for_subset(analysis_dataframe[analysis_dataframe["story_condition"] == "trolley"], "trolley")

        dataframes[inclusion_filter] = pd.DataFrame(summary_rows)

    "Concatenate included_only and all_finishers dataframes"
    dataframe_twoafc_counts = pd.concat(list(dataframes.values()), axis=0, ignore_index=True)

    "Apply pivot if table_form, then save and return"
    if table_form:
        dataframe_twoafc_counts = pivot_twoafc_table(dataframe_twoafc_counts)

    _save_analysis_dataframe_to_processed_folder(dataframe_to_save=dataframe_twoafc_counts, 
                                                general_settings=general_settings, file_name_key=file_name_key)

    return dataframe_twoafc_counts


def compute_correlations(general_settings: GeneralSettings, force_rebuild: bool | None = None) -> pd.DataFrame:
    """
    Computes basic correlations among Clark blame/wrongness/punishment.

    I compute correlations in two ways:
        • Between-subjects first vignette (one row per participant)
        • Within-subject pooled across three vignettes by averaging per participant

    Returns:
        • DataFrame with correlation results.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    "Load and return dataframe if one already exists and not directed to rebuild."
    group_summaries_extraction = load_analysis_dataframe(
        general_settings=general_settings, file_name_key="correlations", force_rebuild=force_rebuild)
    if group_summaries_extraction["success"]:
        group_summaries_dataframe: pd.DataFrame = group_summaries_extraction["dataframe"]
        return group_summaries_dataframe
    if group_summaries_extraction["error"]:
        raise Exception(group_summaries_extraction["message"])

    "Load or rebuild preprocessed dataframe"
    cleaned_dataframe = load_or_build_cleaned_dataframe(general_settings=general_settings, force_rebuild=False)

    dataframes = {}
    for inclusion_filter_and_analysis_dataframe in [
            ("included_only", cleaned_dataframe[cleaned_dataframe["included"]].copy()),
            ("all_finishers", cleaned_dataframe.copy())
        ]:
        inclusion_filter, analysis_dataframe = inclusion_filter_and_analysis_dataframe

        rows = []

        "Between-subject first vignette correlations"
        first_blame = pd.to_numeric(analysis_dataframe["first_vignette_clark_blame"], errors="coerce")
        first_wrong = pd.to_numeric(analysis_dataframe["first_vignette_clark_wrong"], errors="coerce")
        first_punish = pd.to_numeric(analysis_dataframe["first_vignette_clark_punish"], errors="coerce")

        for label_a, series_a in [("blame", first_blame), ("wrongness", first_wrong), ("punishment", first_punish)]:
            for label_b, series_b in [("blame", first_blame), ("wrongness", first_wrong), ("punishment", first_punish)]:
                if label_a >= label_b:
                    continue
                valid_mask = (~series_a.isna()) & (~series_b.isna())
                if valid_mask.sum() < 3:
                    continue
                r_value, p_value = stats.pearsonr(series_a[valid_mask], series_b[valid_mask])
                rows.append(
                    {
                        "inclusion_filter": inclusion_filter,
                        "analysis_scope": "between_subjects_first_vignette",
                        "var_a": label_a,
                        "var_b": label_b,
                        "n": int(valid_mask.sum()),
                        "correlation_type": "pearson_r",
                        "r": float(r_value),
                        "p_value": float(p_value),
                    }
                )

        "Within-subject participant means across all three vignettes"
        analysis_dataframe["clark_blame_mean"] = analysis_dataframe[["clark_blame_cc", "clark_blame_ch", "clark_blame_div"]].mean(axis=1, skipna=True)
        analysis_dataframe["clark_wrong_mean"] = analysis_dataframe[["clark_wrong_cc", "clark_wrong_ch", "clark_wrong_div"]].mean(axis=1, skipna=True)
        analysis_dataframe["clark_punish_mean"] = analysis_dataframe[["clark_punish_cc", "clark_punish_ch", "clark_punish_div"]].mean(axis=1, skipna=True)

        mean_blame = pd.to_numeric(analysis_dataframe["clark_blame_mean"], errors="coerce")
        mean_wrong = pd.to_numeric(analysis_dataframe["clark_wrong_mean"], errors="coerce")
        mean_punish = pd.to_numeric(analysis_dataframe["clark_punish_mean"], errors="coerce")

        for label_a, series_a in [("blame_mean", mean_blame), ("wrongness_mean", mean_wrong), ("punishment_mean", mean_punish)]:
            for label_b, series_b in [("blame_mean", mean_blame), ("wrongness_mean", mean_wrong), ("punishment_mean", mean_punish)]:
                if label_a >= label_b:
                    continue
                valid_mask = (~series_a.isna()) & (~series_b.isna())
                if valid_mask.sum() < 3:
                    continue
                r_value, p_value = stats.pearsonr(series_a[valid_mask], series_b[valid_mask])
                rows.append(
                    {
                        "inclusion_filter": inclusion_filter,
                        "analysis_scope": "within_subjects_participant_means",
                        "var_a": label_a,
                        "var_b": label_b,
                        "n": int(valid_mask.sum()),
                        "correlation_type": "pearson_r",
                        "r": float(r_value),
                        "p_value": float(p_value),
                    }
                )

        dataframes[inclusion_filter] = pd.DataFrame(rows)

    "Concatenate included_only and all_finishers dataframes, save, and return"
    dataframe_correlations = pd.concat(list(dataframes.values()), axis=0, ignore_index=True)
    _save_analysis_dataframe_to_processed_folder(dataframe_to_save=dataframe_correlations, 
                                                 general_settings=general_settings, file_name_key="correlations")

    return dataframe_correlations


def compute_individual_difference_regressions(general_settings: GeneralSettings, force_rebuild: bool | None = None) -> pd.DataFrame:
    """
    Runs exploratory regressions predicting shielding deltas from individual differences.

    Models (OLS):
        • delta ~ load_condition + story_condition + crt_score + individualism_score

    Returns:
        • A tidy table with coefficient estimates and p-values.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    "Load and return dataframe if one already exists and not directed to rebuild."
    group_summaries_extraction = load_analysis_dataframe(
        general_settings=general_settings, file_name_key="regressions", force_rebuild=force_rebuild)
    if group_summaries_extraction["success"]:
        group_summaries_dataframe: pd.DataFrame = group_summaries_extraction["dataframe"]
        return group_summaries_dataframe
    if group_summaries_extraction["error"]:
        raise Exception(group_summaries_extraction["message"])

    "Load or rebuild preprocessed dataframe"
    cleaned_dataframe = load_or_build_cleaned_dataframe(general_settings=general_settings, force_rebuild=False)

    dataframes = {}
    for inclusion_filter_and_analysis_dataframe in [
            ("included_only", cleaned_dataframe[cleaned_dataframe["included"]].copy()),
            ("all_finishers", cleaned_dataframe.copy())
        ]:
        inclusion_filter, analysis_dataframe = inclusion_filter_and_analysis_dataframe

        "Ensure numeric"
        analysis_dataframe["crt_score"] = pd.to_numeric(analysis_dataframe["crt_score"], errors="coerce")
        analysis_dataframe["individualism_score"] = pd.to_numeric(analysis_dataframe["individualism_score"], errors="coerce")

        rows = []

        for outcome_column in ["clark_blame_ch_minus_cc", "clark_blame_div_minus_cc", "clark_blame_min_ch_div_minus_cc"]:
            "Model A: overall individualism_score"
            formula_a = f"{outcome_column} ~ C(load_condition) + C(story_condition) + crt_score + individualism_score"
            model_a = smf.ols(formula=formula_a, data=analysis_dataframe).fit()

            for term_name, coefficient_value in model_a.params.items():
                rows.append(
                    {
                        "inclusion_filter": inclusion_filter,
                        "model": "A_overall_individualism",
                        "outcome": outcome_column,
                        "term": term_name,
                        "estimate": float(coefficient_value),
                        "std_error": float(model_a.bse[term_name]),
                        "t_value": float(model_a.tvalues[term_name]),
                        "p_value": float(model_a.pvalues[term_name]),
                        "n": int(model_a.nobs),
                        "r_squared": float(model_a.rsquared),
                    }
                )

            "Model B: horizontal + vertical individualism components"
            formula_b = f"{outcome_column} ~ C(load_condition) + C(story_condition) + crt_score + individualism_horizontal + individualism_vertical"
            model_b = smf.ols(formula=formula_b, data=analysis_dataframe).fit()

            for term_name, coefficient_value in model_b.params.items():
                rows.append(
                    {
                        "inclusion_filter": inclusion_filter,
                        "model": "B_horizontal_vertical",
                        "outcome": outcome_column,
                        "term": term_name,
                        "estimate": float(coefficient_value),
                        "std_error": float(model_b.bse[term_name]),
                        "t_value": float(model_b.tvalues[term_name]),
                        "p_value": float(model_b.pvalues[term_name]),
                        "n": int(model_b.nobs),
                        "r_squared": float(model_b.rsquared),
                    }
                )

        dataframes[inclusion_filter] = pd.DataFrame(rows)

    "Concatenate included_only and all_finishers dataframes, save, and return"
    dataframe_regressions = pd.concat(list(dataframes.values()), axis=0, ignore_index=True)
    _save_analysis_dataframe_to_processed_folder(dataframe_to_save=dataframe_regressions, 
                                                 general_settings=general_settings, file_name_key="regressions")

    return dataframe_regressions


def compute_consistency_effects(general_settings: GeneralSettings, force_rebuild: bool | None = None) -> pd.DataFrame:
    """
    Explores order/consistency effects suggested in the design notes.

    Primary consistency contrast:
        Compare CH ratings depending on whether CH was encountered first vs CC encountered first.

    I implement:
        • CH rating among CH-first vs CC-first participants.
        • CC rating among CH-first vs CC-first participants.

    Returns:
        • DataFrame of exploratory Welch tests.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    "Load and return dataframe if one already exists and not directed to rebuild."
    group_summaries_extraction = load_analysis_dataframe(
        general_settings=general_settings, file_name_key="consistency_effects", force_rebuild=force_rebuild)
    if group_summaries_extraction["success"]:
        group_summaries_dataframe: pd.DataFrame = group_summaries_extraction["dataframe"]
        return group_summaries_dataframe
    if group_summaries_extraction["error"]:
        raise Exception(group_summaries_extraction["message"])

    "Load or rebuild preprocessed dataframe"
    cleaned_dataframe = load_or_build_cleaned_dataframe(general_settings=general_settings, force_rebuild=False)

    dataframes = {}
    for inclusion_filter_and_analysis_dataframe in [
            ("included_only", cleaned_dataframe[cleaned_dataframe["included"]].copy()),
            ("all_finishers", cleaned_dataframe.copy())
        ]:
        inclusion_filter, analysis_dataframe = inclusion_filter_and_analysis_dataframe

        rows = []

        for dv_prefix in ["clark_blame", "clark_wrong", "clark_punish"]:
            column_cc = f"{dv_prefix}_cc"
            column_ch = f"{dv_prefix}_ch"

            subset_cc_first = analysis_dataframe[analysis_dataframe["case_code_position_1"] == "CC"].copy()
            subset_ch_first = analysis_dataframe[analysis_dataframe["case_code_position_1"] == "CH"].copy()

            "CH rating: CH-first vs CC-first"
            test_ch_rating = run_welch_t_test_between_groups(
                pd.concat([subset_ch_first, subset_cc_first], axis=0),
                dv_column_name=column_ch,
                group_column_name="case_code_position_1",
                group_a_value="CH",
                group_b_value="CC",
            )
            test_ch_rating["inclusion_filter"] = inclusion_filter
            test_ch_rating["comparison"] = f"{dv_prefix}: CH rating when CH-first vs when CC-first"
            rows.append(test_ch_rating)

            "CC rating: CH-first vs CC-first"
            test_cc_rating = run_welch_t_test_between_groups(
                pd.concat([subset_ch_first, subset_cc_first], axis=0),
                dv_column_name=column_cc,
                group_column_name="case_code_position_1",
                group_a_value="CH",
                group_b_value="CC",
            )
            test_cc_rating["inclusion_filter"] = inclusion_filter
            test_cc_rating["comparison"] = f"{dv_prefix}: CC rating when CH-first vs when CC-first"
            rows.append(test_cc_rating)

        dataframes[inclusion_filter] = pd.DataFrame(rows)

    "Concatenate included_only and all_finishers dataframes, save, and return"
    dataframe_consistency_effects = pd.concat(list(dataframes.values()), axis=0, ignore_index=True)
    _save_analysis_dataframe_to_processed_folder(dataframe_to_save=dataframe_consistency_effects, 
                                                 general_settings=general_settings, file_name_key="consistency_effects")

    return dataframe_consistency_effects


def compute_triangulation_results(general_settings: GeneralSettings, force_rebuild: bool | None = None) -> pd.DataFrame:
    """
    Triangulation analyses linking:
        • within-subject numeric deltas (e.g., Clark blame CH-CC)
        • 2AFC categorical judgments (e.g., "CH > CC", "CH ≥ CC", "CH ≤ CC", "CH < CC")

    Outputs:
        • Pearson and Spearman correlations between delta and a 0-3 coding of the 2AFC response
        • Logistic regressions predicting the probability of a pro-shielding 2AFC response from the numeric delta
        • Simple concordance rates

    Coding (as suggested in the analysis notes):
        strict greater: 3
        weak greater:   2
        weak less:      1
        strict less:    0
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    "Load and return dataframe if one already exists and not directed to rebuild."
    group_summaries_extraction = load_analysis_dataframe(
        general_settings=general_settings, file_name_key="triangulation", force_rebuild=force_rebuild)
    if group_summaries_extraction["success"]:
        group_summaries_dataframe: pd.DataFrame = group_summaries_extraction["dataframe"]
        return group_summaries_dataframe
    if group_summaries_extraction["error"]:
        raise Exception(group_summaries_extraction["message"])

    "Load or rebuild preprocessed dataframe"
    cleaned_dataframe = load_or_build_cleaned_dataframe(general_settings=general_settings, force_rebuild=False)

    dataframes = {}
    for inclusion_filter_and_analysis_dataframe in [
            ("included_only", cleaned_dataframe[cleaned_dataframe["included"]].copy()),
            ("all_finishers", cleaned_dataframe.copy())
        ]:
        inclusion_filter, analysis_dataframe = inclusion_filter_and_analysis_dataframe

        rows = []

        def twoafc_to_numeric(twoafc_string_value: str) -> float:
            if pd.isna(twoafc_string_value):
                return np.nan
            value_string = str(twoafc_string_value)
            if " > " in value_string:
                return 3.0
            if " ≥ " in value_string:
                return 2.0
            if " ≤ " in value_string:
                return 1.0
            if " < " in value_string:
                return 0.0
            return np.nan

        for comparison_name, twoafc_column, delta_column, left_prefix, right_prefix in [
            ("CH_vs_CC", "twoafc_ch_vs_cc", "clark_blame_ch_minus_cc", "CH", "CC"),
            ("DIV_vs_CC", "twoafc_div_vs_cc", "clark_blame_div_minus_cc", "DIV", "CC"),
        ]:
            numeric_code = analysis_dataframe[twoafc_column].apply(twoafc_to_numeric)
            delta_values = pd.to_numeric(analysis_dataframe[delta_column], errors="coerce")

            valid_mask = (~numeric_code.isna()) & (~delta_values.isna())

            if valid_mask.sum() < 5:
                continue

            pearson_r, pearson_p = stats.pearsonr(delta_values[valid_mask], numeric_code[valid_mask])
            spearman_r, spearman_p = stats.spearmanr(delta_values[valid_mask], numeric_code[valid_mask])

            rows.append(
                {
                    "inclusion_filter": inclusion_filter,
                    "comparison": comparison_name,
                    "analysis_type": "correlation",
                    "correlation_type": "pearson_r",
                    "n": int(valid_mask.sum()),
                    "statistic": float(pearson_r),
                    "p_value": float(pearson_p),
                    "notes": f"Correlation between {delta_column} and 2AFC 0-3 code",
                }
            )

            rows.append(
                {
                    "inclusion_filter": inclusion_filter,
                    "comparison": comparison_name,
                    "analysis_type": "correlation",
                    "correlation_type": "spearman_r",
                    "n": int(valid_mask.sum()),
                    "statistic": float(spearman_r),
                    "p_value": float(spearman_p),
                    "notes": f"Spearman correlation between {delta_column} and 2AFC 0-3 code",
                }
            )

            "Concordance: sign(delta) vs direction implied by 2AFC (treat ≥ as pro-left; ≤ as pro-right)"
            pro_left_mask = analysis_dataframe[twoafc_column].isin([f"{left_prefix} > {right_prefix}", f"{left_prefix} ≥ {right_prefix}"])
            pro_right_mask = analysis_dataframe[twoafc_column].isin([f"{left_prefix} < {right_prefix}", f"{left_prefix} ≤ {right_prefix}"])

            concordant = (
                ((delta_values > 0) & pro_left_mask)
                | ((delta_values < 0) & pro_right_mask)
                | ((delta_values == 0) & analysis_dataframe[twoafc_column].isin([f"{left_prefix} ≥ {right_prefix}", f"{left_prefix} ≤ {right_prefix}"]))
            )
            concordance_rate = float(concordant[valid_mask].mean())

            rows.append(
                {
                    "inclusion_filter": inclusion_filter,
                    "comparison": comparison_name,
                    "analysis_type": "concordance",
                    "correlation_type": "directional_match_rate",
                    "n": int(valid_mask.sum()),
                    "statistic": concordance_rate,
                    "p_value": np.nan,
                    "notes": "Directional concordance between sign(delta) and 2AFC direction (≥ treated as pro-left)",
                }
            )

            "Logistic regression: predict probability of pro-left response from numeric delta"
            y = pro_left_mask.astype(int)
            x = sm.add_constant(delta_values)

            logistic_valid = (~y.isna()) & (~x[delta_column].isna())

            if logistic_valid.sum() >= 20:
                try:
                    model = sm.Logit(y[logistic_valid], x[logistic_valid]).fit(disp=False)
                    rows.append(
                        {
                            "inclusion_filter": inclusion_filter,
                            "comparison": comparison_name,
                            "analysis_type": "logistic_regression",
                            "correlation_type": "logit_pro_left",
                            "n": int(logistic_valid.sum()),
                            "statistic": float(model.params[delta_column]),
                            "p_value": float(model.pvalues[delta_column]),
                            "notes": "Logit coefficient for delta predicting pro-left (>,≥) vs pro-right (<,≤)",
                        }
                    )
                except Exception:
                    pass

        dataframes[inclusion_filter] = pd.DataFrame(rows)

    "Concatenate included_only and all_finishers dataframes, save, and return"
    dataframe_triangulation = pd.concat(list(dataframes.values()), axis=0, ignore_index=True)
    _save_analysis_dataframe_to_processed_folder(dataframe_to_save=dataframe_triangulation, 
                                                 general_settings=general_settings, file_name_key="triangulation")

    return dataframe_triangulation


def run_confirmatory_and_exploratory_tests(general_settings: GeneralSettings, confirmatory_pooled_ols_covariance_type: str | None = None, force_rebuild: bool | None = None) -> pd.DataFrame:
    """
    Runs the primary confirmatory tests plus a structured set of exploratory tests.

    Confirmatory (preregistered):
        • Between-subjects, included only:
            H1: first_vignette_clark_blame CH vs CC (two-sided)
            H2: first_vignette_clark_blame DIV vs CC (two-sided)
        • Holm correction across these two p-values.

    Returns:
        • Long-format table of test results with metadata columns.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    "Load and return dataframe if one already exists and not directed to rebuild."
    group_summaries_extraction = load_analysis_dataframe(
        general_settings=general_settings, file_name_key="tests", force_rebuild=force_rebuild)
    if group_summaries_extraction["success"]:
        group_summaries_dataframe: pd.DataFrame = group_summaries_extraction["dataframe"]
        return group_summaries_dataframe
    if group_summaries_extraction["error"]:
        raise Exception(group_summaries_extraction["message"])

    "Load of rebuild preprocessed dataframe"
    cleaned_dataframe = load_or_build_cleaned_dataframe(general_settings=general_settings, force_rebuild=False)

    dataframes = {}
    for inclusion_filter_and_analysis_dataframe in [
            ("included_only", cleaned_dataframe[cleaned_dataframe["included"]].copy()), 
            ("all_finishers", cleaned_dataframe.copy())
        ]:
        inclusion_filter, analysis_dataframe = inclusion_filter_and_analysis_dataframe

        tests_rows = []

        "Convenience helper"
        def append_test_row(row_dict: Dict, analysis_family: str, inclusion_filter: str, story_condition_value: str, load_condition_value: str, notes: str):
            row_dict = dict(row_dict)
            row_dict["analysis_family"] = analysis_family
            row_dict["inclusion_filter"] = inclusion_filter
            row_dict["story_condition"] = story_condition_value
            row_dict["load_condition"] = load_condition_value
            row_dict["notes"] = notes
            tests_rows.append(row_dict)

        "Confirmatory: between-subject first vignette, Clark blame"
        confirmatory_subset = analysis_dataframe.copy()

        confirmatory_between_subjects_method_normalized = str(
            general_settings["misc"]["confirmatory_between_subjects_method"],
        ).strip().lower()

        if confirmatory_between_subjects_method_normalized == "welch":
            confirmatory_contrast_rows_dataframe = pd.DataFrame(
                [
                    run_welch_t_test_between_groups(
                        confirmatory_subset,
                        dv_column_name="first_vignette_clark_blame",
                        group_column_name="case_code_position_1",
                        group_a_value="CH",
                        group_b_value="CC",
                    ),
                    run_welch_t_test_between_groups(
                        confirmatory_subset,
                        dv_column_name="first_vignette_clark_blame",
                        group_column_name="case_code_position_1",
                        group_a_value="DIV",
                        group_b_value="CC",
                    ),
                ]
            )

        elif confirmatory_between_subjects_method_normalized in {"pooled_ols", "ols", "anova"}:
            confirmatory_contrast_rows_dataframe = run_pooled_ols_planned_contrasts(
                dataframe=confirmatory_subset,
                dv_column_name="first_vignette_clark_blame",
                group_column_name="case_code_position_1",
                covariance_type=confirmatory_pooled_ols_covariance_type,
            )

        else:
            raise ValueError(
                "confirmatory_between_subjects_method must be 'welch' or 'pooled_ols'. "
                f"Got: {confirmatory_between_subjects_method!r}"
            )

        "Holm correction across the two preregistered contrasts"
        adjusted_p_values = holm_bonferroni_correct_p_values(
            confirmatory_contrast_rows_dataframe["p_value"].tolist()
        )
        confirmatory_contrast_rows_dataframe["p_value_holm"] = adjusted_p_values

        confirmatory_test_ch_vs_cc = confirmatory_contrast_rows_dataframe.loc[
            confirmatory_contrast_rows_dataframe["group_a"] == "CH"
        ].iloc[0].to_dict()

        confirmatory_test_div_vs_cc = confirmatory_contrast_rows_dataframe.loc[
            confirmatory_contrast_rows_dataframe["group_a"] == "DIV"
        ].iloc[0].to_dict()

        append_test_row(
            confirmatory_test_ch_vs_cc,
            "confirmatory",
            inclusion_filter,
            "all",
            "all",
            "H1: CH vs CC on first vignette (Clark blame)",
        )
        append_test_row(
            confirmatory_test_div_vs_cc,
            "confirmatory",
            inclusion_filter,
            "all",
            "all",
            "H2: DIV vs CC on first vignette (Clark blame)",
        )

        "Exploratory: parallel between-subject tests for wrongness and punishment"
        for dv_column_name, label in [
            ("first_vignette_clark_wrong", "Clark wrongness"),
            ("first_vignette_clark_punish", "Clark punishment"),
        ]:
            test_ch_vs_cc = run_welch_t_test_between_groups(analysis_dataframe, dv_column_name, "case_code_position_1", "CH", "CC")
            test_div_vs_cc = run_welch_t_test_between_groups(analysis_dataframe, dv_column_name, "case_code_position_1", "DIV", "CC")
            append_test_row(test_ch_vs_cc, "exploratory", inclusion_filter, "all", "all", f"Between-subject first vignette (included): CH vs CC on {label}")
            append_test_row(test_div_vs_cc, "exploratory", inclusion_filter, "all", "all", f"Between-subject first vignette (included): DIV vs CC on {label}")

        "Exploratory: within-subject deltas (included only)"
        delta_columns = [
            ("clark_blame_ch_minus_cc", "Clark blame CH-CC"),
            ("clark_blame_div_minus_cc", "Clark blame DIV-CC"),
            ("clark_blame_min_ch_div_minus_cc", "Clark blame MIN(CH,DIV)-CC"),
            ("proximate_blame_cc_minus_ch", "Proximate blame CC-CH (responsibility manipulation check)"),
            ("clark_wrong_ch_minus_cc", "Clark wrongness CH-CC"),
            ("clark_wrong_div_minus_cc", "Clark wrongness DIV-CC"),
            ("clark_wrong_min_ch_div_minus_cc", "Clark wrongness MIN(CH,DIV)-CC"),
            ("proximate_wrong_cc_minus_ch", "Proximate wrongness CC-CH"),
            ("clark_punish_ch_minus_cc", "Clark punishment CH-CC"),
            ("clark_punish_div_minus_cc", "Clark punishment DIV-CC"),
            ("clark_punish_min_ch_div_minus_cc", "Clark punishment MIN(CH,DIV)-CC"),
            ("proximate_punish_cc_minus_ch", "Proximate punishment CC-CH"),
        ]

        for delta_column_name, label in delta_columns:
            delta_test = run_one_sample_t_test_on_delta(analysis_dataframe, delta_column_name)
            append_test_row(delta_test, "exploratory", inclusion_filter, "all", "all", f"Within-subject delta vs 0 ({inclusion_filter}): {label}")

        "Exploratory: subgroup tests by story_condition and load_condition (included only)"
        for story_condition_value in ["firework", "trolley"]:
            subset_story = analysis_dataframe[analysis_dataframe["story_condition"] == story_condition_value]

            for dv_column_name, label in [("first_vignette_clark_blame", "Clark blame"), ("first_vignette_clark_wrong", "Clark wrongness"), ("first_vignette_clark_punish", "Clark punishment")]:
                test_ch_vs_cc = run_welch_t_test_between_groups(subset_story, dv_column_name, "case_code_position_1", "CH", "CC")
                test_div_vs_cc = run_welch_t_test_between_groups(subset_story, dv_column_name, "case_code_position_1", "DIV", "CC")
                append_test_row(test_ch_vs_cc, "exploratory", inclusion_filter, story_condition_value, "all", f"Between-subject first vignette (included): CH vs CC on {label}")
                append_test_row(test_div_vs_cc, "exploratory", inclusion_filter, story_condition_value, "all", f"Between-subject first vignette (included): DIV vs CC on {label}")

            "Within-subject deltas by story"
            for delta_column_name, label in [
                ("clark_blame_ch_minus_cc", "Clark blame CH-CC"),
                ("clark_blame_div_minus_cc", "Clark blame DIV-CC"),
                ("clark_blame_min_ch_div_minus_cc", "Clark blame MIN(CH,DIV)-CC"),
                ("proximate_blame_cc_minus_ch", "Proximate blame CC-CH"),
                ("clark_wrong_ch_minus_cc", "Clark wrongness CH-CC"),
                ("clark_wrong_div_minus_cc", "Clark wrongness DIV-CC"),
                ("clark_wrong_min_ch_div_minus_cc", "Clark wrongness MIN(CH,DIV)-CC"),
                ("proximate_wrong_cc_minus_ch", "Proximate wrongness CC-CH"),
                ("clark_punish_ch_minus_cc", "Clark punishment CH-CC"),
                ("clark_punish_div_minus_cc", "Clark punishment DIV-CC"),
                ("clark_punish_min_ch_div_minus_cc", "Clark punishment MIN(CH,DIV)-CC"),
                ("proximate_punish_cc_minus_ch", "Proximate punishment CC-CH"),
            ]:
                delta_test = run_one_sample_t_test_on_delta(subset_story, delta_column_name)
                append_test_row(delta_test, "exploratory", inclusion_filter, story_condition_value, "all", f"Within-subject delta vs 0 ({inclusion_filter}): {label}")

        for load_condition_value in ["high", "low"]:
            subset_load = analysis_dataframe[analysis_dataframe["load_condition"] == load_condition_value]

            for dv_column_name, label in [("first_vignette_clark_blame", "Clark blame"), ("first_vignette_clark_wrong", "Clark wrongness"), ("first_vignette_clark_punish", "Clark punishment")]:
                test_ch_vs_cc = run_welch_t_test_between_groups(subset_load, dv_column_name, "case_code_position_1", "CH", "CC")
                test_div_vs_cc = run_welch_t_test_between_groups(subset_load, dv_column_name, "case_code_position_1", "DIV", "CC")
                append_test_row(test_ch_vs_cc, "exploratory", inclusion_filter, "all", load_condition_value, f"Between-subject first vignette (included): CH vs CC on {label}")
                append_test_row(test_div_vs_cc, "exploratory", inclusion_filter, "all", load_condition_value, f"Between-subject first vignette (included): DIV vs CC on {label}")

            for delta_column_name, label in [
                ("clark_blame_ch_minus_cc", "Clark blame CH-CC"),
                ("clark_blame_div_minus_cc", "Clark blame DIV-CC"),
                ("clark_blame_min_ch_div_minus_cc", "Clark blame MIN(CH,DIV)-CC"),
                ("clark_wrong_ch_minus_cc", "Clark wrongness CH-CC"),
                ("clark_wrong_div_minus_cc", "Clark wrongness DIV-CC"),
                ("clark_wrong_min_ch_div_minus_cc", "Clark wrongness MIN(CH,DIV)-CC"),
                ("clark_punish_ch_minus_cc", "Clark punishment CH-CC"),
                ("clark_punish_div_minus_cc", "Clark punishment DIV-CC"),
                ("clark_punish_min_ch_div_minus_cc", "Clark punishment MIN(CH,DIV)-CC"),
            ]:
                delta_test = run_one_sample_t_test_on_delta(subset_load, delta_column_name)
                append_test_row(delta_test, "exploratory", inclusion_filter, "all", load_condition_value, f"Within-subject delta vs 0 ({inclusion_filter}): {label}")

        "Exploratory moderation: does load predict delta magnitude?"
        for delta_column_name, label in [
            ("clark_blame_ch_minus_cc", "Clark blame CH-CC"),
            ("clark_blame_div_minus_cc", "Clark blame DIV-CC"),
            ("clark_blame_min_ch_div_minus_cc", "Clark blame MIN(CH,DIV)-CC"),
            ("clark_wrong_ch_minus_cc", "Clark wrongness CH-CC"),
            ("clark_wrong_div_minus_cc", "Clark wrongness DIV-CC"),
            ("clark_wrong_min_ch_div_minus_cc", "Clark wrongness MIN(CH,DIV)-CC"),
            ("clark_punish_ch_minus_cc", "Clark punishment CH-CC"),
            ("clark_punish_div_minus_cc", "Clark punishment DIV-CC"),
            ("clark_punish_min_ch_div_minus_cc", "Clark punishment MIN(CH,DIV)-CC"),
            ("proximate_blame_cc_minus_ch", "Proximate blame CC-CH"),
        ]:
            test_high_vs_low = run_welch_t_test_between_groups(
                analysis_dataframe,
                dv_column_name=delta_column_name,
                group_column_name="load_condition",
                group_a_value="high",
                group_b_value="low",
            )
            append_test_row(test_high_vs_low, "exploratory", inclusion_filter, "all", "all", f"Moderation: High vs Low load on delta ({label})")

        "Exploratory: compare CH vs DIV on first vignette (not preregistered direction)"
        ch_vs_div_test = run_welch_t_test_between_groups(
            analysis_dataframe,
            dv_column_name="first_vignette_clark_blame",
            group_column_name="case_code_position_1",
            group_a_value="CH",
            group_b_value="DIV",
        )
        append_test_row(ch_vs_div_test, "exploratory", inclusion_filter, "all", "all", "Between-subject first vignette (included): CH vs DIV on Clark blame")

        dataframes[inclusion_filter] = pd.DataFrame(tests_rows)

    "Concatenate included_only and all_finishers dataframes, save, and return"
    dataframe_tests = pd.concat(list(dataframe for dataframe in dataframes.values()), axis=0, ignore_index=True) 
    _save_analysis_dataframe_to_processed_folder(dataframe_to_save=dataframe_tests, 
                                                 general_settings=general_settings, file_name_key="tests")

    return dataframe_tests


"=========================================================================================="
"=================================== Integrated Models ===================================="
"=========================================================================================="

def _compute_model_based_linear_contrast_row(
    fitted_model_result: Any,
    new_data_for_group_a: pd.DataFrame,
    new_data_for_group_b: pd.DataFrame,
    model_name: str,
    model_variant: str,
    analysis_scope: str,
    inclusion_filter: str,
    story_condition: str,
    contrast_label: str,
    display_label: str,
    formula_string: str,
    n_model_rows: int,
    n_unique_participants: int,
    analysis_meaning: str,
    likely_manuscript_use: str,
) -> dict[str, Any]:
    """
    Compute one model-based linear contrast by averaging predictions over two design grids.

    Arguments:
        • fitted_model_result: Any
            - Fitted statsmodels model result object.
        • new_data_for_group_a: pd.DataFrame
            - Prediction grid for group A.
        • new_data_for_group_b: pd.DataFrame
            - Prediction grid for group B.
        • model_name: str
            - Human-readable name of the model family.
        • model_variant: str
            - Specific model variant within the family.
        • analysis_scope: str
            - High-level scope such as between_subjects_first_vignette or within_subjects_repeated_measures.
        • inclusion_filter: str
            - included_only or all_finishers.
        • story_condition: str
            - all, firework, or trolley.
        • contrast_label: str
            - CH - CC, DIV - CC, or CH - DIV.
        • display_label: str
            - Reader-facing row label.
        • formula_string: str
            - Formula used for the fitted model.
        • n_model_rows: int
            - Number of rows fed into the model.
        • n_unique_participants: int
            - Number of unique participants contributing rows.
        • analysis_meaning: str
            - Plain-language explanation of what this row means.
        • likely_manuscript_use: str
            - Where this row is most likely to be used in the manuscript.

    Returns:
        • dict[str, Any]
            - One tidy contrast row.
    """
    from patsy import build_design_matrices

    design_info = fitted_model_result.model.data.design_info

    design_matrix_group_a = build_design_matrices(
        [design_info], new_data_for_group_a, return_type="dataframe"
    )[0]
    design_matrix_group_b = build_design_matrices(
        [design_info], new_data_for_group_b, return_type="dataframe"
    )[0]

    mean_design_vector_group_a = np.asarray(
        design_matrix_group_a.mean(axis=0),
        dtype=float,
    )
    mean_design_vector_group_b = np.asarray(
        design_matrix_group_b.mean(axis=0),
        dtype=float,
    )
    contrast_vector = mean_design_vector_group_a - mean_design_vector_group_b

    contrast_test = fitted_model_result.t_test(contrast_vector.reshape(1, -1))

    contrast_estimate = float(np.asarray(contrast_test.effect).reshape(-1)[0])
    contrast_standard_error = float(np.asarray(contrast_test.sd).reshape(-1)[0])
    contrast_p_value = float(np.asarray(contrast_test.pvalue).reshape(-1)[0])

    confidence_interval_array = np.asarray(contrast_test.conf_int())
    ci95_lower = float(confidence_interval_array[0, 0])
    ci95_upper = float(confidence_interval_array[0, 1])

    test_statistic = np.nan
    if contrast_standard_error > 0 and not np.isnan(contrast_standard_error):
        test_statistic = float(contrast_estimate / contrast_standard_error)

    return {
        "model_name": model_name,
        "model_variant": model_variant,
        "analysis_scope": analysis_scope,
        "row_type": "contrast",
        "inclusion_filter": inclusion_filter,
        "story_condition": story_condition,
        "contrast_label": contrast_label,
        "display_label": display_label,
        "estimate": contrast_estimate,
        "standard_error": contrast_standard_error,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
        "test_statistic": test_statistic,
        "p_value": contrast_p_value,
        "term_name": np.nan,
        "observed_standard_deviation": np.nan,
        "observed_median": np.nan,
        "n_model_rows": int(n_model_rows),
        "n_unique_participants": int(n_unique_participants),
        "formula_string": formula_string,
        "analysis_meaning": analysis_meaning,
        "likely_manuscript_use": likely_manuscript_use,
    }


def _extract_model_coefficient_rows_from_fitted_results(
    fitted_model_result: Any,
    model_name: str,
    model_variant: str,
    analysis_scope: str,
    inclusion_filter: str,
    formula_string: str,
    n_model_rows: int,
    n_unique_participants: int,
    term_meaning_dictionary: dict[str, str],
    likely_manuscript_use: str = "Supplement",
) -> list[dict[str, Any]]:
    """
    Convert raw model coefficients into tidy output rows.

    Arguments:
        • fitted_model_result: Any
            - Fitted statsmodels model result object.
        • model_name: str
            - Human-readable name of the model family.
        • model_variant: str
            - Specific model variant within the family.
        • analysis_scope: str
            - High-level scope of the model.
        • inclusion_filter: str
            - included_only or all_finishers.
        • formula_string: str
            - Formula used for the fitted model.
        • n_model_rows: int
            - Number of rows fed into the model.
        • n_unique_participants: int
            - Number of unique participants contributing rows.
        • term_meaning_dictionary: dict[str, str]
            - Mapping from coefficient names to plain-language interpretations.
        • likely_manuscript_use: str
            - Where these rows are most likely to appear.

    Returns:
        • list[dict[str, Any]]
            - One tidy row per coefficient.
    """
    confidence_intervals_dataframe = fitted_model_result.conf_int()

    coefficient_rows: list[dict[str, Any]] = []

    for term_name in fitted_model_result.params.index:
        coefficient_rows.append(
            {
                "model_name": model_name,
                "model_variant": model_variant,
                "analysis_scope": analysis_scope,
                "row_type": "coefficient",
                "inclusion_filter": inclusion_filter,
                "story_condition": "all",
                "contrast_label": np.nan,
                "display_label": term_name,
                "estimate": float(fitted_model_result.params[term_name]),
                "standard_error": float(fitted_model_result.bse[term_name]),
                "ci95_lower": float(confidence_intervals_dataframe.loc[term_name, 0]),
                "ci95_upper": float(confidence_intervals_dataframe.loc[term_name, 1]),
                "test_statistic": float(fitted_model_result.tvalues[term_name]),
                "p_value": float(fitted_model_result.pvalues[term_name]),
                "term_name": term_name,
                "observed_standard_deviation": np.nan,
                "observed_median": np.nan,
                "n_model_rows": int(n_model_rows),
                "n_unique_participants": int(n_unique_participants),
                "formula_string": formula_string,
                "analysis_meaning": term_meaning_dictionary.get(term_name, "Raw model coefficient."),
                "likely_manuscript_use": likely_manuscript_use,
            }
        )

    return coefficient_rows


def compute_first_vignette_clark_blame_integrated_models(
    general_settings: dict[str, Any],
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Compute the first-vignette integrated model results for Clark blame.

    Arguments:
        • general_settings: dict[str, Any]
            - Master project settings dictionary.
        • cleaned_dataframe: pd.DataFrame | None
            - Optional cleaned dataframe. If None, load or build it.
        • force_rebuild: bool | None
            - Whether to ignore existing CSV output and rebuild.

    Returns:
        • pd.DataFrame
            - Tidy long-format dataframe containing observed descriptives, raw coefficients,
              model-based contrasts, and omnibus tests for both included_only and all_finishers.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    "Load and return dataframe if one already exists and not directed to rebuild."
    integrated_model_extraction = load_analysis_dataframe(
        general_settings=general_settings,
        file_name_key="first_vignette",
        force_rebuild=force_rebuild,
    )
    if integrated_model_extraction["success"]:
        dataframe_integrated_first_vignette_models: pd.DataFrame = integrated_model_extraction["dataframe"]
        return dataframe_integrated_first_vignette_models
    if integrated_model_extraction["error"]:
        raise Exception(integrated_model_extraction["message"])

    "Load or rebuild preprocessed dataframe."
    if cleaned_dataframe is None:
        cleaned_dataframe = load_or_build_cleaned_dataframe(
            general_settings=general_settings,
            force_rebuild=False,
        )
    else:
        cleaned_dataframe = cleaned_dataframe.copy()

    dataframes_by_inclusion_filter: dict[str, pd.DataFrame] = {}

    for inclusion_filter_value, include_only_passers in [
        ("included_only", True),
        ("all_finishers", False),
    ]:
        analysis_dataframe = cleaned_dataframe[
            [
                "response_id",
                "included",
                "story_condition",
                "case_code_position_1",
                "first_vignette_clark_blame",
            ]
        ].copy()

        if include_only_passers:
            analysis_dataframe = analysis_dataframe.loc[analysis_dataframe["included"] == True].copy()  # noqa: E712

        analysis_dataframe["story_condition"] = pd.Categorical(
            analysis_dataframe["story_condition"],
            categories=["firework", "trolley"],
            ordered=True,
        )
        analysis_dataframe["case_code_position_1"] = pd.Categorical(
            analysis_dataframe["case_code_position_1"],
            categories=["CC", "CH", "DIV"],
            ordered=True,
        )
        analysis_dataframe["first_vignette_clark_blame"] = pd.to_numeric(
            analysis_dataframe["first_vignette_clark_blame"],
            errors="coerce",
        )

        analysis_dataframe = analysis_dataframe.dropna(
            subset=["first_vignette_clark_blame", "story_condition", "case_code_position_1"]
        ).copy()

        model_formula_string = (
            "first_vignette_clark_blame ~ "
            "C(case_code_position_1, Treatment(reference='CC')) * "
            "C(story_condition, Treatment(reference='firework'))"
        )

        fitted_model_result = smf.ols(
            formula=model_formula_string,
            data=analysis_dataframe,
        ).fit(cov_type="HC3")

        n_model_rows: int = int(analysis_dataframe.shape[0])
        n_unique_participants: int = int(analysis_dataframe["response_id"].nunique())

        output_rows: list[dict[str, Any]] = []

        "Observed cell descriptives"
        observed_descriptives = (
            analysis_dataframe
            .groupby(["story_condition", "case_code_position_1"], observed=True)["first_vignette_clark_blame"]
            .agg(["count", "mean", "std", "median"])
            .reset_index()
        )

        for _, descriptive_row in observed_descriptives.iterrows():
            story_condition_value: str = str(descriptive_row["story_condition"])
            vignette_condition_value: str = str(descriptive_row["case_code_position_1"])

            output_rows.append(
                {
                    "model_name": "first_vignette_condition_story_model",
                    "model_variant": "primary_hc3_ols",
                    "analysis_scope": "between_subjects_first_vignette",
                    "row_type": "observed_descriptive",
                    "inclusion_filter": inclusion_filter_value,
                    "story_condition": story_condition_value,
                    "contrast_label": np.nan,
                    "display_label": f"{story_condition_value.title()} | {vignette_condition_value} observed mean",
                    "estimate": float(descriptive_row["mean"]),
                    "standard_error": np.nan,
                    "ci95_lower": np.nan,
                    "ci95_upper": np.nan,
                    "test_statistic": np.nan,
                    "p_value": np.nan,
                    "term_name": np.nan,
                    "observed_standard_deviation": float(descriptive_row["std"]) if pd.notna(descriptive_row["std"]) else np.nan,
                    "observed_median": float(descriptive_row["median"]) if pd.notna(descriptive_row["median"]) else np.nan,
                    "n_model_rows": int(descriptive_row["count"]),
                    "n_unique_participants": n_unique_participants,
                    "formula_string": model_formula_string,
                    "analysis_meaning": "Observed first-vignette Clark-blame descriptive statistic for one story-family × condition cell.",
                    "likely_manuscript_use": "Figure support or supplement",
                }
            )

        "Coefficient rows"
        coefficient_meaning_dictionary = {
            "Intercept": "Predicted mean for CC in the firework story family.",
            "C(case_code_position_1, Treatment(reference='CC'))[T.CH]": "CH - CC in the firework story family.",
            "C(case_code_position_1, Treatment(reference='CC'))[T.DIV]": "DIV - CC in the firework story family.",
            "C(story_condition, Treatment(reference='firework'))[T.trolley]": "How the CC baseline changes in trolley relative to firework.",
            "C(case_code_position_1, Treatment(reference='CC'))[T.CH]:C(story_condition, Treatment(reference='firework'))[T.trolley]": "How the CH - CC contrast changes in trolley relative to firework.",
            "C(case_code_position_1, Treatment(reference='CC'))[T.DIV]:C(story_condition, Treatment(reference='firework'))[T.trolley]": "How the DIV - CC contrast changes in trolley relative to firework.",
        }

        output_rows.extend(
            _extract_model_coefficient_rows_from_fitted_results(
                fitted_model_result=fitted_model_result,
                model_name="first_vignette_condition_story_model",
                model_variant="primary_hc3_ols",
                analysis_scope="between_subjects_first_vignette",
                inclusion_filter=inclusion_filter_value,
                formula_string=model_formula_string,
                n_model_rows=n_model_rows,
                n_unique_participants=n_unique_participants,
                term_meaning_dictionary=coefficient_meaning_dictionary,
                likely_manuscript_use="Supplement",
            )
        )

        "Model-based pooled and story-specific contrasts"
        for contrast_group_a_value, contrast_group_b_value, contrast_label in [
            ("CH", "CC", "CH - CC"),
            ("DIV", "CC", "DIV - CC"),
            ("CH", "DIV", "CH - DIV"),
        ]:
            pooled_grid_group_a = pd.DataFrame(
                {
                    "case_code_position_1": [contrast_group_a_value, contrast_group_a_value],
                    "story_condition": ["firework", "trolley"],
                }
            )
            pooled_grid_group_b = pd.DataFrame(
                {
                    "case_code_position_1": [contrast_group_b_value, contrast_group_b_value],
                    "story_condition": ["firework", "trolley"],
                }
            )

            output_rows.append(
                _compute_model_based_linear_contrast_row(
                    fitted_model_result=fitted_model_result,
                    new_data_for_group_a=pooled_grid_group_a,
                    new_data_for_group_b=pooled_grid_group_b,
                    model_name="first_vignette_condition_story_model",
                    model_variant="primary_hc3_ols",
                    analysis_scope="between_subjects_first_vignette",
                    inclusion_filter=inclusion_filter_value,
                    story_condition="all",
                    contrast_label=contrast_label,
                    display_label=f"Between-subj | {inclusion_filter_value} | Pooled | {contrast_label}",
                    formula_string=model_formula_string,
                    n_model_rows=n_model_rows,
                    n_unique_participants=n_unique_participants,
                    analysis_meaning=f"Pooled first-vignette model-based contrast for {contrast_label}. Positive values mean more Clark blame in the left-hand condition.",
                    likely_manuscript_use="Main table or story summary",
                )
            )

            for story_condition_value in ["firework", "trolley"]:
                story_grid_group_a = pd.DataFrame(
                    {
                        "case_code_position_1": [contrast_group_a_value],
                        "story_condition": [story_condition_value],
                    }
                )
                story_grid_group_b = pd.DataFrame(
                    {
                        "case_code_position_1": [contrast_group_b_value],
                        "story_condition": [story_condition_value],
                    }
                )

                output_rows.append(
                    _compute_model_based_linear_contrast_row(
                        fitted_model_result=fitted_model_result,
                        new_data_for_group_a=story_grid_group_a,
                        new_data_for_group_b=story_grid_group_b,
                        model_name="first_vignette_condition_story_model",
                        model_variant="primary_hc3_ols",
                        analysis_scope="between_subjects_first_vignette",
                        inclusion_filter=inclusion_filter_value,
                        story_condition=story_condition_value,
                        contrast_label=contrast_label,
                        display_label=f"Between-subj | {inclusion_filter_value} | {story_condition_value.title()} | {contrast_label}",
                        formula_string=model_formula_string,
                        n_model_rows=n_model_rows,
                        n_unique_participants=n_unique_participants,
                        analysis_meaning=f"Story-specific first-vignette model-based contrast for {contrast_label} in the {story_condition_value} story family.",
                        likely_manuscript_use="Story-specific table",
                    )
                )

        "Omnibus story-interaction test"
        story_interaction_term_names = [
            term_name
            for term_name in fitted_model_result.params.index
            if ":C(story_condition" in term_name
        ]

        if len(story_interaction_term_names) > 0:
            constraint_matrix = np.zeros((len(story_interaction_term_names), len(fitted_model_result.params.index)))
            for row_index, term_name in enumerate(story_interaction_term_names):
                column_index = list(fitted_model_result.params.index).index(term_name)
                constraint_matrix[row_index, column_index] = 1.0

            wald_test_result = fitted_model_result.wald_test(constraint_matrix, scalar=True)
            omnibus_statistic = float(np.asarray(wald_test_result.statistic).reshape(-1)[0])
            omnibus_df = getattr(wald_test_result, "df_denom", np.nan)

            output_rows.append(
                {
                    "model_name": "first_vignette_condition_story_model",
                    "model_variant": "primary_hc3_ols",
                    "analysis_scope": "between_subjects_first_vignette",
                    "row_type": "omnibus_test",
                    "inclusion_filter": inclusion_filter_value,
                    "story_condition": "all",
                    "contrast_label": "condition_by_story_interaction",
                    "display_label": f"Between-subj | {inclusion_filter_value} | Omnibus story interaction",
                    "estimate": np.nan,
                    "standard_error": np.nan,
                    "ci95_lower": np.nan,
                    "ci95_upper": np.nan,
                    "test_statistic": omnibus_statistic,
                    "p_value": float(wald_test_result.pvalue),
                    "term_name": "condition_by_story_interaction",
                    "observed_standard_deviation": np.nan,
                    "observed_median": np.nan,
                    "n_model_rows": n_model_rows,
                    "n_unique_participants": n_unique_participants,
                    "formula_string": model_formula_string,
                    "analysis_meaning": "Omnibus test of whether the condition contrasts differ across the firework and trolley story families.",
                    "likely_manuscript_use": "Supplement or story-robustness paragraph",
                }
            )

        dataframe_integrated_first_vignette_models = pd.DataFrame(output_rows)
        dataframe_integrated_first_vignette_models["inclusion_sort_order"] = np.where(
            dataframe_integrated_first_vignette_models["inclusion_filter"] == "included_only",
            0,
            1,
        )
        dataframe_integrated_first_vignette_models["row_type_sort_order"] = dataframe_integrated_first_vignette_models["row_type"].map(
            {
                "contrast": 0,
                "omnibus_test": 1,
                "coefficient": 2,
                "observed_descriptive": 3,
            }
        ).fillna(9)

        dataframes_by_inclusion_filter[inclusion_filter_value] = dataframe_integrated_first_vignette_models

    "Concatenate included_only and all_finishers dataframes, save, and return."
    dataframe_integrated_first_vignette_models = pd.concat(
        list(dataframe for dataframe in dataframes_by_inclusion_filter.values()),
        axis=0,
        ignore_index=True,
    ).sort_values(
        by=["inclusion_sort_order", "row_type_sort_order", "story_condition", "display_label"],
        kind="stable",
    ).drop(
        columns=["inclusion_sort_order", "row_type_sort_order"],
    )

    _save_analysis_dataframe_to_processed_folder(
        dataframe_to_save=dataframe_integrated_first_vignette_models,
        general_settings=general_settings,
        file_name_key="first_vignette",
    )

    return dataframe_integrated_first_vignette_models


def compute_within_subject_clark_blame_integrated_models(
    general_settings: dict[str, Any],
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Compute the within-subject integrated model results for Clark blame.

    Arguments:
        • general_settings: dict[str, Any]
            - Master project settings dictionary.
        • cleaned_dataframe: pd.DataFrame | None
            - Optional cleaned dataframe. If None, load or build it.
        • force_rebuild: bool | None
            - Whether to ignore existing CSV output and rebuild.

    Returns:
        • pd.DataFrame
            - Tidy long-format dataframe containing observed descriptives, raw coefficients,
              model-based contrasts, and omnibus tests for both included_only and all_finishers.

    Notes:
        • The primary repeated-measures model is:
              condition × story + vignette_position
          because that is the model from which the pooled and story-specific contrasts are easiest to read.
        • A secondary position-sensitivity model is also fit:
              condition × vignette_position + story to test whether the 
              condition pattern changes across absolute vignette position.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    "Load and return dataframe if one already exists and not directed to rebuild."
    integrated_model_extraction = load_analysis_dataframe(
        general_settings=general_settings,
        file_name_key="within_subject",
        force_rebuild=force_rebuild,
    )
    if integrated_model_extraction["success"]:
        dataframe_integrated_within_subject_models: pd.DataFrame = integrated_model_extraction["dataframe"]
        return dataframe_integrated_within_subject_models
    if integrated_model_extraction["error"]:
        raise Exception(integrated_model_extraction["message"])

    "Load or rebuild preprocessed dataframe."
    if cleaned_dataframe is None:
        cleaned_dataframe = load_or_build_cleaned_dataframe(
            general_settings=general_settings,
            force_rebuild=False,
        )
    else:
        cleaned_dataframe = cleaned_dataframe.copy()

    dataframes_by_inclusion_filter: dict[str, pd.DataFrame] = {}

    for inclusion_filter_value, include_only_passers in [
        ("included_only", True),
        ("all_finishers", False),
    ]:
        wide_analysis_dataframe = cleaned_dataframe[
            [
                "response_id",
                "included",
                "story_condition",
                "case_code_position_1",
                "case_code_position_2",
                "case_code_position_3",
                "clark_blame_cc",
                "clark_blame_ch",
                "clark_blame_div",
            ]
        ].copy()

        if include_only_passers:
            wide_analysis_dataframe = wide_analysis_dataframe.loc[
                wide_analysis_dataframe["included"] == True
            ].copy()  # noqa: E712

        long_rows: list[dict[str, Any]] = []

        for _, participant_row in wide_analysis_dataframe.iterrows():
            vignette_position_by_condition = {
                participant_row["case_code_position_1"]: 1,
                participant_row["case_code_position_2"]: 2,
                participant_row["case_code_position_3"]: 3,
            }

            for vignette_condition_value in ["CC", "CH", "DIV"]:
                long_rows.append(
                    {
                        "response_id": participant_row["response_id"],
                        "story_condition": participant_row["story_condition"],
                        "vignette_condition": vignette_condition_value,
                        "vignette_position": vignette_position_by_condition[vignette_condition_value],
                        "clark_rating_value": participant_row[f"clark_blame_{vignette_condition_value.lower()}"],
                    }
                )

        analysis_dataframe_long = pd.DataFrame(long_rows)

        analysis_dataframe_long["clark_rating_value"] = pd.to_numeric(
            analysis_dataframe_long["clark_rating_value"],
            errors="coerce",
        )
        analysis_dataframe_long = analysis_dataframe_long.dropna(
            subset=["clark_rating_value"]
        ).copy()

        analysis_dataframe_long["story_condition"] = pd.Categorical(
            analysis_dataframe_long["story_condition"],
            categories=["firework", "trolley"],
            ordered=True,
        )
        analysis_dataframe_long["vignette_condition"] = pd.Categorical(
            analysis_dataframe_long["vignette_condition"],
            categories=["CC", "CH", "DIV"],
            ordered=True,
        )
        analysis_dataframe_long["vignette_position"] = pd.Categorical(
            analysis_dataframe_long["vignette_position"],
            categories=[1, 2, 3],
            ordered=True,
        )

        primary_formula_string = (
            "clark_rating_value ~ "
            "C(vignette_condition, Treatment(reference='CC')) * "
            "C(story_condition, Treatment(reference='firework')) + "
            "C(vignette_position, Treatment(reference=1))"
        )

        fitted_primary_model_result = smf.gee(
            formula=primary_formula_string,
            groups="response_id",
            data=analysis_dataframe_long,
            cov_struct=sm.cov_struct.Exchangeable(),
            family=sm.families.Gaussian(),
        ).fit()

        position_sensitivity_formula_string = (
            "clark_rating_value ~ "
            "C(vignette_condition, Treatment(reference='CC')) * "
            "C(vignette_position, Treatment(reference=1)) + "
            "C(story_condition, Treatment(reference='firework'))"
        )

        fitted_position_sensitivity_model_result = smf.gee(
            formula=position_sensitivity_formula_string,
            groups="response_id",
            data=analysis_dataframe_long,
            cov_struct=sm.cov_struct.Exchangeable(),
            family=sm.families.Gaussian(),
        ).fit()

        n_model_rows: int = int(analysis_dataframe_long.shape[0])
        n_unique_participants: int = int(analysis_dataframe_long["response_id"].nunique())

        output_rows: list[dict[str, Any]] = []

        "Observed descriptives pooled across story family and position"
        observed_condition_descriptives = (
            analysis_dataframe_long
            .groupby(["vignette_condition"], observed=True)["clark_rating_value"]
            .agg(["count", "mean", "std", "median"])
            .reset_index()
        )
        for _, descriptive_row in observed_condition_descriptives.iterrows():
            vignette_condition_value: str = str(descriptive_row["vignette_condition"])
            output_rows.append(
                {
                    "model_name": "within_subject_condition_position_model",
                    "model_variant": "primary_gee_condition_story_plus_position",
                    "analysis_scope": "within_subjects_repeated_measures",
                    "row_type": "observed_descriptive",
                    "inclusion_filter": inclusion_filter_value,
                    "story_condition": "all",
                    "contrast_label": np.nan,
                    "display_label": f"Within-subj | {inclusion_filter_value} | {vignette_condition_value} observed mean",
                    "estimate": float(descriptive_row["mean"]),
                    "standard_error": np.nan,
                    "ci95_lower": np.nan,
                    "ci95_upper": np.nan,
                    "test_statistic": np.nan,
                    "p_value": np.nan,
                    "term_name": np.nan,
                    "observed_standard_deviation": float(descriptive_row["std"]) if pd.notna(descriptive_row["std"]) else np.nan,
                    "observed_median": float(descriptive_row["median"]) if pd.notna(descriptive_row["median"]) else np.nan,
                    "n_model_rows": int(descriptive_row["count"]),
                    "n_unique_participants": n_unique_participants,
                    "formula_string": primary_formula_string,
                    "analysis_meaning": "Observed repeated-measures Clark-blame descriptive statistic pooled across story family and position.",
                    "likely_manuscript_use": "Figure support or supplement",
                }
            )

        "Primary-model coefficient rows"
        primary_coefficient_meaning_dictionary = {
            "Intercept": "Predicted mean for CC when it appears first in the firework story family.",
            "C(vignette_condition, Treatment(reference='CC'))[T.CH]": "CH - CC when the vignette is first in the firework story family.",
            "C(vignette_condition, Treatment(reference='CC'))[T.DIV]": "DIV - CC when the vignette is first in the firework story family.",
            "C(story_condition, Treatment(reference='firework'))[T.trolley]": "How the CC baseline shifts in trolley relative to firework.",
            "C(vignette_position, Treatment(reference=1))[T.2]": "How appearing second changes the overall baseline relative to first.",
            "C(vignette_position, Treatment(reference=1))[T.3]": "How appearing third changes the overall baseline relative to first.",
            "C(vignette_condition, Treatment(reference='CC'))[T.CH]:C(story_condition, Treatment(reference='firework'))[T.trolley]": "How the CH - CC contrast changes in trolley relative to firework.",
            "C(vignette_condition, Treatment(reference='CC'))[T.DIV]:C(story_condition, Treatment(reference='firework'))[T.trolley]": "How the DIV - CC contrast changes in trolley relative to firework.",
        }

        output_rows.extend(
            _extract_model_coefficient_rows_from_fitted_results(
                fitted_model_result=fitted_primary_model_result,
                model_name="within_subject_condition_position_model",
                model_variant="primary_gee_condition_story_plus_position",
                analysis_scope="within_subjects_repeated_measures",
                inclusion_filter=inclusion_filter_value,
                formula_string=primary_formula_string,
                n_model_rows=n_model_rows,
                n_unique_participants=n_unique_participants,
                term_meaning_dictionary=primary_coefficient_meaning_dictionary,
                likely_manuscript_use="Supplement",
            )
        )

        "Model-based pooled and story-specific contrasts from the primary model"
        for contrast_group_a_value, contrast_group_b_value, contrast_label in [
            ("CH", "CC", "CH - CC"),
            ("DIV", "CC", "DIV - CC"),
            ("CH", "DIV", "CH - DIV"),
        ]:
            pooled_grid_group_a = pd.DataFrame(
                [
                    {
                        "vignette_condition": contrast_group_a_value,
                        "story_condition": story_condition_value,
                        "vignette_position": vignette_position_value,
                    }
                    for story_condition_value in ["firework", "trolley"]
                    for vignette_position_value in [1, 2, 3]
                ]
            )
            pooled_grid_group_b = pd.DataFrame(
                [
                    {
                        "vignette_condition": contrast_group_b_value,
                        "story_condition": story_condition_value,
                        "vignette_position": vignette_position_value,
                    }
                    for story_condition_value in ["firework", "trolley"]
                    for vignette_position_value in [1, 2, 3]
                ]
            )

            output_rows.append(
                _compute_model_based_linear_contrast_row(
                    fitted_model_result=fitted_primary_model_result,
                    new_data_for_group_a=pooled_grid_group_a,
                    new_data_for_group_b=pooled_grid_group_b,
                    model_name="within_subject_condition_position_model",
                    model_variant="primary_gee_condition_story_plus_position",
                    analysis_scope="within_subjects_repeated_measures",
                    inclusion_filter=inclusion_filter_value,
                    story_condition="all",
                    contrast_label=contrast_label,
                    display_label=f"Within-subj | {inclusion_filter_value} | Pooled | {contrast_label}",
                    formula_string=primary_formula_string,
                    n_model_rows=n_model_rows,
                    n_unique_participants=n_unique_participants,
                    analysis_meaning=f"Pooled repeated-measures model-based contrast for {contrast_label}, averaging equally across story family and vignette position.",
                    likely_manuscript_use="Main table or story summary",
                )
            )

            for story_condition_value in ["firework", "trolley"]:
                story_grid_group_a = pd.DataFrame(
                    [
                        {
                            "vignette_condition": contrast_group_a_value,
                            "story_condition": story_condition_value,
                            "vignette_position": vignette_position_value,
                        }
                        for vignette_position_value in [1, 2, 3]
                    ]
                )
                story_grid_group_b = pd.DataFrame(
                    [
                        {
                            "vignette_condition": contrast_group_b_value,
                            "story_condition": story_condition_value,
                            "vignette_position": vignette_position_value,
                        }
                        for vignette_position_value in [1, 2, 3]
                    ]
                )

                output_rows.append(
                    _compute_model_based_linear_contrast_row(
                        fitted_model_result=fitted_primary_model_result,
                        new_data_for_group_a=story_grid_group_a,
                        new_data_for_group_b=story_grid_group_b,
                        model_name="within_subject_condition_position_model",
                        model_variant="primary_gee_condition_story_plus_position",
                        analysis_scope="within_subjects_repeated_measures",
                        inclusion_filter=inclusion_filter_value,
                        story_condition=story_condition_value,
                        contrast_label=contrast_label,
                        display_label=f"Within-subj | {inclusion_filter_value} | {story_condition_value.title()} | {contrast_label}",
                        formula_string=primary_formula_string,
                        n_model_rows=n_model_rows,
                        n_unique_participants=n_unique_participants,
                        analysis_meaning=f"Story-specific repeated-measures model-based contrast for {contrast_label} in the {story_condition_value} story family.",
                        likely_manuscript_use="Story-specific table",
                    )
                )

        "Omnibus story-interaction test from the primary model"
        story_interaction_term_names = [
            term_name
            for term_name in fitted_primary_model_result.params.index
            if ":C(story_condition" in term_name
        ]

        if len(story_interaction_term_names) > 0:
            story_constraint_matrix = np.zeros((len(story_interaction_term_names), len(fitted_primary_model_result.params.index)))
            for row_index, term_name in enumerate(story_interaction_term_names):
                column_index = list(fitted_primary_model_result.params.index).index(term_name)
                story_constraint_matrix[row_index, column_index] = 1.0

            story_wald_test_result = fitted_primary_model_result.wald_test(story_constraint_matrix, scalar=True)

            output_rows.append(
                {
                    "model_name": "within_subject_condition_position_model",
                    "model_variant": "primary_gee_condition_story_plus_position",
                    "analysis_scope": "within_subjects_repeated_measures",
                    "row_type": "omnibus_test",
                    "inclusion_filter": inclusion_filter_value,
                    "story_condition": "all",
                    "contrast_label": "condition_by_story_interaction",
                    "display_label": f"Within-subj | {inclusion_filter_value} | Omnibus story interaction",
                    "estimate": np.nan,
                    "standard_error": np.nan,
                    "ci95_lower": np.nan,
                    "ci95_upper": np.nan,
                    "test_statistic": float(np.asarray(story_wald_test_result.statistic).reshape(-1)[0]),
                    "p_value": float(story_wald_test_result.pvalue),
                    "term_name": "condition_by_story_interaction",
                    "observed_standard_deviation": np.nan,
                    "observed_median": np.nan,
                    "n_model_rows": n_model_rows,
                    "n_unique_participants": n_unique_participants,
                    "formula_string": primary_formula_string,
                    "analysis_meaning": "Omnibus test of whether the repeated-measures condition contrasts differ across story family.",
                    "likely_manuscript_use": "Supplement or story-robustness paragraph",
                }
            )

        "Omnibus condition-by-position test from the secondary position-sensitivity model"
        position_interaction_term_names = [
            term_name
            for term_name in fitted_position_sensitivity_model_result.params.index
            if ":C(vignette_position" in term_name
        ]

        if len(position_interaction_term_names) > 0:
            position_constraint_matrix = np.zeros((len(position_interaction_term_names), len(fitted_position_sensitivity_model_result.params.index)))
            for row_index, term_name in enumerate(position_interaction_term_names):
                column_index = list(fitted_position_sensitivity_model_result.params.index).index(term_name)
                position_constraint_matrix[row_index, column_index] = 1.0

            position_wald_test_result = fitted_position_sensitivity_model_result.wald_test(
                position_constraint_matrix,
                scalar=True,
            )

            output_rows.append(
                {
                    "model_name": "within_subject_condition_position_model",
                    "model_variant": "position_sensitivity_gee_condition_by_position",
                    "analysis_scope": "within_subjects_repeated_measures",
                    "row_type": "omnibus_test",
                    "inclusion_filter": inclusion_filter_value,
                    "story_condition": "all",
                    "contrast_label": "condition_by_position_interaction",
                    "display_label": f"Within-subj | {inclusion_filter_value} | Omnibus condition-by-position interaction",
                    "estimate": np.nan,
                    "standard_error": np.nan,
                    "ci95_lower": np.nan,
                    "ci95_upper": np.nan,
                    "test_statistic": float(np.asarray(position_wald_test_result.statistic).reshape(-1)[0]),
                    "p_value": float(position_wald_test_result.pvalue),
                    "term_name": "condition_by_position_interaction",
                    "observed_standard_deviation": np.nan,
                    "observed_median": np.nan,
                    "n_model_rows": n_model_rows,
                    "n_unique_participants": n_unique_participants,
                    "formula_string": position_sensitivity_formula_string,
                    "analysis_meaning": "Omnibus test of whether the repeated-measures condition contrasts depend on vignette position.",
                    "likely_manuscript_use": "Supplement or order-sensitivity paragraph",
                }
            )

        dataframe_integrated_within_subject_models = pd.DataFrame(output_rows)
        dataframe_integrated_within_subject_models["inclusion_sort_order"] = np.where(
            dataframe_integrated_within_subject_models["inclusion_filter"] == "included_only",
            0,
            1,
        )
        dataframe_integrated_within_subject_models["row_type_sort_order"] = dataframe_integrated_within_subject_models["row_type"].map(
            {
                "contrast": 0,
                "omnibus_test": 1,
                "coefficient": 2,
                "observed_descriptive": 3,
            }
        ).fillna(9)

        dataframes_by_inclusion_filter[inclusion_filter_value] = dataframe_integrated_within_subject_models

    "Concatenate included_only and all_finishers dataframes, save, and return."
    dataframe_integrated_within_subject_models = pd.concat(
        list(dataframe for dataframe in dataframes_by_inclusion_filter.values()),
        axis=0,
        ignore_index=True,
    ).sort_values(
        by=["inclusion_sort_order", "row_type_sort_order", "story_condition", "display_label"],
        kind="stable",
    ).drop(
        columns=["inclusion_sort_order", "row_type_sort_order"],
    )

    _save_analysis_dataframe_to_processed_folder(
        dataframe_to_save=dataframe_integrated_within_subject_models,
        general_settings=general_settings,
        file_name_key="within_subject",
    )

    return dataframe_integrated_within_subject_models


def compute_integrated_clark_blame_results(
    general_settings: dict[str, Any],
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Build a single master CSV containing the integrated-model results for Clark blame.

    Arguments:
        • general_settings: dict[str, Any]
            - Master project settings dictionary.
        • cleaned_dataframe: pd.DataFrame | None
            - Optional cleaned dataframe. If None, load or build it.
        • force_rebuild: bool | None
            - Whether to ignore existing CSV output and rebuild.

    Returns:
        • pd.DataFrame
            - Concatenated integrated-model dataframe spanning both the first-vignette
              and within-subject model families.

    Notes:
        • This function keeps the integrated models separate from tests.csv.
        • That is deliberate. tests.csv remains the home for direct preregistered and exploratory test rows.
          This integrated CSV becomes the home for model-based rows.
        • Later, manuscript-table functions can merge the two after the fact.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    "Load and return dataframe if one already exists and not directed to rebuild."
    integrated_model_extraction = load_analysis_dataframe(
        general_settings=general_settings,
        file_name_key="blame_models",
        force_rebuild=force_rebuild,
    )
    if integrated_model_extraction["success"]:
        dataframe_integrated_models: pd.DataFrame = integrated_model_extraction["dataframe"]
        return dataframe_integrated_models
    if integrated_model_extraction["error"]:
        raise Exception(integrated_model_extraction["message"])

    "Load or rebuild preprocessed dataframe."
    if cleaned_dataframe is None:
        cleaned_dataframe = load_or_build_cleaned_dataframe(
            general_settings=general_settings,
            force_rebuild=False,
        )
    else:
        cleaned_dataframe = cleaned_dataframe.copy()

    dataframe_first_vignette_models = compute_first_vignette_clark_blame_integrated_models(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )
    dataframe_within_subject_models = compute_within_subject_clark_blame_integrated_models(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )

    dataframe_integrated_models = pd.concat(
        [dataframe_first_vignette_models, dataframe_within_subject_models],
        axis=0,
        ignore_index=True,
    )

    _save_analysis_dataframe_to_processed_folder(
        dataframe_to_save=dataframe_integrated_models,
        general_settings=general_settings,
        file_name_key="blame_models",
    )

    return dataframe_integrated_models


def fit_first_vignette_condition_story_model(
    general_settings: dict[str, Any],
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper.

    Arguments:
        • general_settings: dict[str, Any]
        • cleaned_dataframe: pd.DataFrame | None
        • force_rebuild: bool | None

    Returns:
        • pd.DataFrame
            - Same dataframe returned by compute_first_vignette_clark_blame_integrated_models.
    """
    return compute_first_vignette_clark_blame_integrated_models(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )


def fit_within_subject_condition_position_model(
    general_settings: dict[str, Any],
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper.

    Arguments:
        • general_settings: dict[str, Any]
        • cleaned_dataframe: pd.DataFrame | None
        • force_rebuild: bool | None

    Returns:
        • pd.DataFrame
            - Same dataframe returned by compute_within_subject_clark_blame_integrated_models.
    """
    return compute_within_subject_clark_blame_integrated_models(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )


"=========================================================================================="
"================================== Visualization Helpers ================================="
"=========================================================================================="

def _normalize_story_condition_input(story_condition: str | Any = None) -> str | None:
    """
    Normalizes story-condition filters to the canonical values used in the cleaned dataframe.
    """

    if story_condition is None:
        return None

    story_condition_normalized = str(story_condition).strip().lower()
    if story_condition_normalized in {"all", "pooled", "both", "either", "none"}:
        return None
    if story_condition_normalized in {"trolley", "trolleys"}:
        return "trolley"
    if story_condition_normalized in {
        "fireworks",
        "firework",
        "firework show",
        "firework_show",
        "parade",
    }:
        return "firework"

    raise ValueError(
        "Unrecognized story_condition. Expected one of: None (pooled), 'trolley', or 'firework'. "
        f"Got: {story_condition!r}"
    )


def _normalize_load_condition_input(cognitive_load: str | Any = None) -> str | None:
    """
    Normalizes cognitive-load filters to the canonical values used in the cleaned dataframe.
    """

    if cognitive_load is None:
        return None

    cognitive_load_normalized = str(cognitive_load).strip().lower()
    if cognitive_load_normalized in {"all", "pooled", "both", "either", "none"}:
        return None
    if cognitive_load_normalized in {"high", "hi"}:
        return "high"
    if cognitive_load_normalized in {"low", "lo"}:
        return "low"

    raise ValueError(
        "Unrecognized cognitive_load. Expected one of: None (pooled), 'high', or 'low'. "
        f"Got: {cognitive_load!r}"
    )


def _normalize_subjects_input(subjects: str | Any = None) -> str:
    """
    Normalizes the subjects-display option for plotting.
    """

    if subjects is None:
        return "both"

    subjects_normalized = str(subjects).strip().lower()
    if subjects_normalized in {"both", "all", "combined", "subplots"}:
        return "both"
    if subjects_normalized in {"between", "between-subjects", "between_subjects", "bs"}:
        return "between"
    if subjects_normalized in {"within", "within-subjects", "within_subjects", "ws"}:
        return "within"

    raise ValueError(
        "Unrecognized subjects option. Expected one of: None/'both', 'between', or 'within'. "
        f"Got: {subjects!r}"
    )


def _normalize_dependent_variable_input(dependent_variable: str | Any = "blame") -> tuple[str, str, tuple[float, float] | None]:
    """
    Normalizes a DV string to: (column_suffix, display_label, bounded_y_range_or_None).
    """

    dv_normalized = str(dependent_variable).strip().lower()
    if dv_normalized in {"blame", "blameworthiness"}:
        return "blame", "Blameworthiness", [0.8, 9.2]
    if dv_normalized in {"wrong", "wrongness"}:
        return "wrong", "Wrongness", [0.8, 9.2]
    if dv_normalized in {"punish", "punishment", "years", "prison"}:
        "Punishment is not bounded in the same way; I compute a range later."
        return "punish", "Punishment (years)", None

    raise ValueError(
        "Unrecognized dependent_variable. Expected one of: 'blame', 'wrong', or 'punishment'. "
        f"Got: {dependent_variable!r}"
    )


def _hsla_color(hue: int, saturation_percent: int = 100, lightness_percent: int = 50, alpha: float = 0.9) -> str:
    """
    Creates an hsla() color string compatible with Plotly.
    """

    return f"hsla({hue}, {saturation_percent}%, {lightness_percent}%, {alpha})"


def _normalize_condition_subset_input(conditions: Sequence[str] | str | Any | None = None) -> list[str]:
    """
    Normalize a condition subset input into canonical condition codes.

    Arguments:
        • conditions: Sequence[str] | str | Any | None
            - If None, returns all three conditions: ["CC", "CH", "DIV"].
            - If a string, accepts values like:
                "CC"
                "CH"
                "DIV"
                "CC,CH"
                "CC/CH"
                "Choice-Choice"
                "Choice-Chance"
                "Division"
                "all"
            - If a sequence, each element is normalized individually.

    Returns:
        • list[str]
            - Ordered list of canonical condition codes selected from {"CC", "CH", "DIV"}.

    Raises:
        • ValueError
            - If one or more condition labels cannot be recognized.
    """
    if conditions is None:
        return ["CC", "CH", "DIV"]

    if isinstance(conditions, str):
        condition_tokens = [
            token.strip() for token in re.split(r"[,/|+]", conditions) if token.strip() != ""
        ]
    else:
        condition_tokens = [str(token).strip() for token in conditions]

    if len(condition_tokens) == 0:
        return ["CC", "CH", "DIV"]

    normalized_condition_codes: list[str] = []

    for raw_condition_token in condition_tokens:
        token_normalized = raw_condition_token.strip().lower()

        if token_normalized in {"all", "both", "all_conditions", "all conditions"}:
            return ["CC", "CH", "DIV"]
        if token_normalized in {"cc", "choice-choice", "choice_choice", "choice choice"}:
            normalized_condition_codes.append("CC")
            continue
        if token_normalized in {"ch", "choice-chance", "choice_chance", "choice chance"}:
            normalized_condition_codes.append("CH")
            continue
        if token_normalized in {"div", "division"}:
            normalized_condition_codes.append("DIV")
            continue

        raise ValueError(
            "Unrecognized condition label. Expected one or more of: "
            "'CC', 'CH', 'DIV', 'Choice-Choice', 'Choice-Chance', 'Division', or None/'all'. "
            f"Got: {raw_condition_token!r}"
        )

    deduplicated_condition_codes: list[str] = []
    for normalized_condition_code in normalized_condition_codes:
        if normalized_condition_code not in deduplicated_condition_codes:
            deduplicated_condition_codes.append(normalized_condition_code)

    return deduplicated_condition_codes


def _normalize_delta_type_input(delta_type: str | Sequence[str] | Any | None = None) -> list[str]:
    """
    Normalize a delta-type input into canonical delta labels.

    Arguments:
        • delta_type: str | Sequence[str] | Any | None
            - If None or "both", returns ["CH_CC", "DIV_CC"].
            - Also accepts:
                "CH_CC", "CH-CC", "shielding"
                "DIV_CC", "DIV-CC", "beyond_division"

    Returns:
        • list[str]
            - Ordered list selected from {"CH_CC", "DIV_CC"}.

    Raises:
        • ValueError
            - If one or more delta labels cannot be recognized.
    """
    if delta_type is None:
        return ["CH_CC", "DIV_CC"]

    if isinstance(delta_type, str):
        delta_tokens = [
            token.strip() for token in re.split(r"[,/|+]", delta_type) if token.strip() != ""
        ]
    else:
        delta_tokens = [str(token).strip() for token in delta_type]

    if len(delta_tokens) == 0:
        return ["CH_CC", "DIV_CC"]

    normalized_delta_types: list[str] = []

    for raw_delta_token in delta_tokens:
        token_normalized = raw_delta_token.strip().lower()

        if token_normalized in {"both", "all", "all_deltas", "all deltas"}:
            return ["CH_CC", "DIV_CC"]
        if token_normalized in {"ch_cc", "ch-cc", "ch − cc", "shielding", "chminuscc"}:
            normalized_delta_types.append("CH_CC")
            continue
        if token_normalized in {"div_cc", "div-cc", "div − cc", "beyond_division", "divminuscc"}:
            normalized_delta_types.append("DIV_CC")
            continue

        raise ValueError(
            "Unrecognized delta_type. Expected one or more of: "
            "'CH_CC', 'DIV_CC', 'both', 'shielding', 'beyond_division'. "
            f"Got: {raw_delta_token!r}"
        )

    deduplicated_delta_types: list[str] = []
    for normalized_delta_type in normalized_delta_types:
        if normalized_delta_type not in deduplicated_delta_types:
            deduplicated_delta_types.append(normalized_delta_type)

    return deduplicated_delta_types


def _get_delta_metadata(dv_suffix: str) -> dict[str, dict[str, str]]:
    """
    Return column-name and display-label metadata for shielding delta variables.

    Arguments:
        • dv_suffix: str
            - Normalized dependent-variable suffix from `_normalize_dependent_variable_input`.

    Returns:
        • dict[str, dict[str, str]]
            - Metadata keyed by canonical delta type:
                "CH_CC"
                "DIV_CC"
    """
    return {
        "CH_CC": {
            "column": f"clark_{dv_suffix}_ch_minus_cc",
            "label": "CH - CC",
            "long_label": "Shielding (CH - CC)",
        },
        "DIV_CC": {
            "column": f"clark_{dv_suffix}_div_minus_cc",
            "label": "DIV - CC",
            "long_label": "Shielding beyond division (DIV - CC)",
        },
    }


def _build_delta_long_dataframe(analysis_dataframe: pd.DataFrame, dv_suffix: str, delta_types: Sequence[str]) -> pd.DataFrame:
    """
    Convert selected shielding-delta columns into a long-form dataframe.

    Arguments:
        • analysis_dataframe: pd.DataFrame
            - Already-filtered cleaned dataframe.
        • dv_suffix: str
            - Normalized dependent-variable suffix.
        • delta_types: Sequence[str]
            - Canonical delta types, e.g., ["CH_CC", "DIV_CC"].

    Returns:
        • pd.DataFrame
            - Long dataframe with columns:
                response_id
                load_condition
                story_condition
                crt_score
                individualism_score
                individualism_horizontal
                individualism_vertical
                delta_type
                delta_label
                delta_long_label
                delta_value

    Raises:
        • KeyError
            - If one or more delta columns are missing.
    """
    delta_metadata = _get_delta_metadata(dv_suffix=dv_suffix)

    required_delta_columns = [delta_metadata[delta_type]["column"] for delta_type in delta_types]
    missing_delta_columns = [
        column_name for column_name in required_delta_columns if column_name not in analysis_dataframe.columns
    ]
    if missing_delta_columns:
        raise KeyError(
            "Missing one or more expected shielding-delta columns: "
            + ", ".join(repr(column_name) for column_name in missing_delta_columns)
        )

    id_and_covariate_columns = [
        column_name
        for column_name in [
            "response_id",
            "load_condition",
            "story_condition",
            "crt_score",
            "individualism_score",
            "individualism_horizontal",
            "individualism_vertical",
        ]
        if column_name in analysis_dataframe.columns
    ]

    delta_long_rows: list[pd.DataFrame] = []

    for delta_type in delta_types:
        delta_column_name = delta_metadata[delta_type]["column"]

        temp_dataframe = analysis_dataframe[id_and_covariate_columns + [delta_column_name]].copy()
        temp_dataframe["delta_type"] = delta_type
        temp_dataframe["delta_label"] = delta_metadata[delta_type]["label"]
        temp_dataframe["delta_long_label"] = delta_metadata[delta_type]["long_label"]
        temp_dataframe["delta_value"] = pd.to_numeric(temp_dataframe[delta_column_name], errors="coerce")

        delta_long_rows.append(
            temp_dataframe.drop(columns=[delta_column_name])
        )

    delta_long_dataframe = pd.concat(delta_long_rows, ignore_index=True)
    delta_long_dataframe = delta_long_dataframe.dropna(subset=["delta_value"]).copy()

    return delta_long_dataframe


def _format_p_value_for_display(p_value: float | int | None) -> str:
    """
    Format a p-value compactly for figure/table display.

    Arguments:
        • p_value: float | int | None
            - Raw p-value.

    Returns:
        • str
            - Compact string such as:
                "< .001"
                ".013"
                ".42"
    """
    if p_value is None or pd.isna(p_value):
        return ""

    p_value = float(p_value)

    if p_value < 0.001:
        return "< .001"

    formatted_string = f"{p_value:.3f}"

    if formatted_string.startswith("0"):
        formatted_string = formatted_string[1:]

    return formatted_string


def _get_filtered_plotting_dataframe(general_settings: GeneralSettings, story_condition: str | Any = None, 
                                     cognitive_load: str | Any = None, only_included_participants: bool = True) -> pd.DataFrame:
    """
    Load the cleaned dataframe and apply the standard plotting filters.

    Arguments:
        • general_settings: GeneralSettings;
            - Bundle of settings, including file paths and names.
        • story_condition: str | Any
            - If None, pool story families.
            - If 'firework' or 'trolley', filter to that story condition.
        • cognitive_load: str | Any
            - If None, pool load conditions.
            - If 'high' or 'low', filter to that load condition.
        • only_included_participants: bool
            - If True, restrict to participants who passed all comprehension checks.

    Returns:
        • pd.DataFrame
            - Filtered plotting dataframe.

    Raises:
        • ValueError
            - If no rows remain after filtering.
    """
    story_condition_normalized = _normalize_story_condition_input(story_condition)
    cognitive_load_normalized = _normalize_load_condition_input(cognitive_load)

    cleaned_dataframe = load_or_build_cleaned_dataframe(general_settings=general_settings, force_rebuild=False)
    plotting_dataframe = cleaned_dataframe.copy()

    if only_included_participants:
        plotting_dataframe = plotting_dataframe.loc[plotting_dataframe["included"] == True].copy()  # noqa: E712

    if story_condition_normalized is not None:
        plotting_dataframe = plotting_dataframe.loc[
            plotting_dataframe["story_condition"] == story_condition_normalized
        ].copy()

    if cognitive_load_normalized is not None:
        plotting_dataframe = plotting_dataframe.loc[
            plotting_dataframe["load_condition"] == cognitive_load_normalized
        ].copy()

    if plotting_dataframe.shape[0] == 0:
        raise ValueError(
            "No rows remain after applying the requested filters. "
            "Try relaxing story_condition / cognitive_load / only_included_participants."
        )

    return plotting_dataframe


def _export_plotly_figure_html(fig: "object", general_settings: GeneralSettings, file_name: str, show_figure: bool = False) -> None:
    """
    Export a Plotly figure to HTML.

    Arguments:
        • fig: object
            - Plotly figure to export.
        • general_settings: GeneralSettings;
            - Bundle of settings, including file paths and names.
        • file_name: str;
            - Name of the figure html file.

    Returns:
        • None
    """
    if not isinstance(file_name, str):
        raise TypeError(f"file_name must be a string, not {type(file_name)}.")

    "Strip .html suffix before sanitizing so it doesn't get mangled"
    if file_name.endswith(".html"):
        file_name = file_name[:-5]

    r"Remove illegal Windows filename characters: \ / : * ? \" < > |"
    file_name = re.sub(r'[\\/:*?"<>|]', "_", file_name)

    "Collapse any runs of whitespace or underscores introduced by substitution"
    file_name = re.sub(r"_+", "_", file_name).strip("_")

    "Windows MAX_PATH is 260; reserve space for the directory path and .html extension"
    file_path_figures = general_settings["filing"]["file_paths"]["visuals"]
    max_stem_length = 255 - len(str(file_path_figures)) - len(".html") - 1  # -1 for path separator
    if max_stem_length < 1:
        raise ValueError(f"The figures directory path is too long to accommodate any filename: {file_path_figures}")
    file_name = file_name[:max_stem_length]

    file_name += ".html"

    full_path = file_path_figures / file_name
    fig.write_html(str(full_path), include_plotlyjs="cdn")

    if show_figure:
        fig.show()


"=========================================================================================="
"=================================== Data Visualization ==================================="
"=========================================================================================="

def plot_ratings_by_vignette_condition(
    general_settings: GeneralSettings,
    dv: str | Any = "blame",
    figure_type: str | Any = None,
    subjects: str | Any = None,
    include_proximate_agent: bool = True,
    story_condition: str | Any = None,
    cognitive_load: str | Any = None,
    only_included_participants: bool = True,
    export_html: bool = True,
    base_hue: int | None = 200,
) -> "object":
    """Create the headline distribution plot for CC/CH/DIV (between- and/or within-subjects).

    This function is designed to produce the primary distribution visualization described in your
    message: boxplots (or violins) of blame/wrongness/punishment ratings by vignette condition, with
    between-subjects results shown above within-subjects results.

    Arguments:
        • dv: str | Any
            - Which dependent variable to plot. Supported: 'blame', 'wrong', 'punishment'.
        • figure_type: str | Any
            - If 'box', render box plots.
            - If 'violin', render violin plots.
            - Otherwise, include a dropdown that toggles between box and violin.
        • subjects: str | Any
            - If 'between', plot only the between-subjects (first vignette) distribution.
            - If 'within', plot only the within-subjects distribution.
            - Otherwise (default), show both as stacked subplots (between on top, within on bottom).
        • include_proximate_agent: bool
            - If True, include the proximate-agent (Bill) ratings for CC and CH in the within-subjects
              panel as additional categories.
        • story_condition: str | Any
            - If None, pool stories.
            - If 'firework' or 'trolley', filter to that story condition.
        • cognitive_load: str | Any
            - If None, pool cognitive load.
            - If 'high' or 'low', filter to that load condition.
        • only_included_participants: bool
            - If True, restrict to participants who passed all comprehension checks.
        • export_html: bool
            - If True, export the Plotly figure to an .html file.
        • base_hue: int
            - Starting hue for hsla colors. Subsequent categories increment hue by +20.

    Returns:
        • plotly.graph_objects.Figure
    """
    dv_suffix, dv_label, bounded_y_range = _normalize_dependent_variable_input(dv)
    if figure_type == "violin" and isinstance(bounded_y_range, (list, tuple)):
        bounded_y_range = (-0.5, 10.5)
    subjects_setting = _normalize_subjects_input(subjects)
    story_condition_normalized = _normalize_story_condition_input(story_condition)
    cognitive_load_normalized = _normalize_load_condition_input(cognitive_load)

    figure_type_normalized = None if figure_type is None else str(figure_type).strip().lower()
    include_toggle_dropdown = figure_type_normalized not in {"box", "violin"}
    initial_view = "box" if include_toggle_dropdown else figure_type_normalized

    if include_proximate_agent and subjects_setting == "between":
        raise ValueError(
            "include_proximate_agent=True requires within-subject data, but subjects='between' was requested. "
            "Use subjects=None/'both' or subjects='within'."
        )

    cleaned_dataframe = load_or_build_cleaned_dataframe(general_settings=general_settings)
    analysis_dataframe = cleaned_dataframe.copy()

    if only_included_participants:
        analysis_dataframe = analysis_dataframe.loc[analysis_dataframe["included"] == True].copy()  # noqa: E712

    if story_condition_normalized is not None:
        analysis_dataframe = analysis_dataframe.loc[
            analysis_dataframe["story_condition"] == story_condition_normalized
        ].copy()

    if cognitive_load_normalized is not None:
        analysis_dataframe = analysis_dataframe.loc[
            analysis_dataframe["load_condition"] == cognitive_load_normalized
        ].copy()

    "------------------------------"
    "---- Long-form dataframes ----"
    "------------------------------"

    explicit_box_width = 0.46

    if include_proximate_agent:
        distal_category_labels_in_order = ["CC Distal", "CH Distal", "DIV Distal"]
        proximate_category_labels_in_order = ["CC Proximate", "CH Proximate"]

        between_category_labels_in_order = distal_category_labels_in_order
        within_category_labels_in_order = distal_category_labels_in_order + proximate_category_labels_in_order

        "Reserve five x slots in BOTH rows so the top and bottom panels line up perfectly."
        between_category_positions = {
            "CC Distal": 0.0,
            "CH Distal": 1.0,
            "DIV Distal": 2.0,
        }
        within_category_positions = {
            "CC Distal": 0.0,
            "CH Distal": 1.0,
            "DIV Distal": 2.0,
            "CC Proximate": 3.0,
            "CH Proximate": 4.0,
        }

        between_axis_tick_values = [0.0, 1.0, 2.0, 3.0, 4.0]
        between_axis_tick_text = ["CC Distal", "CH Distal", "DIV Distal", "", ""]
        within_axis_tick_values = [0.0, 1.0, 2.0, 3.0, 4.0]
        within_axis_tick_text = ["CC Distal", "CH Distal", "DIV Distal", "CC Proximate", "CH Proximate"]

    else:
        distal_category_labels_in_order = ["Choice-Choice", "Choice-Chance", "Division"]
        between_category_labels_in_order = distal_category_labels_in_order
        within_category_labels_in_order = distal_category_labels_in_order

        between_category_positions = {
            "Choice-Choice": 0.0,
            "Choice-Chance": 1.0,
            "Division": 2.0,
        }
        within_category_positions = {
            "Choice-Choice": 0.0,
            "Choice-Chance": 1.0,
            "Division": 2.0,
        }

        between_axis_tick_values = [0.0, 1.0, 2.0]
        between_axis_tick_text = ["Choice-Choice", "Choice-Chance", "Division"]
        within_axis_tick_values = [0.0, 1.0, 2.0]
        within_axis_tick_text = ["Choice-Choice", "Choice-Chance", "Division"]

    condition_code_to_distal_label = {
        "CC": distal_category_labels_in_order[0],
        "CH": distal_category_labels_in_order[1],
        "DIV": distal_category_labels_in_order[2],
    }

    explicit_box_width = 0.46

    between_category_positions = {
        category_label: float(category_index)
        for category_index, category_label in enumerate(between_category_labels_in_order)
    }

    within_category_positions = {
        category_label: float(category_index)
        for category_index, category_label in enumerate(within_category_labels_in_order)
    }

    first_vignette_value_column = f"first_vignette_clark_{dv_suffix}"
    if first_vignette_value_column not in analysis_dataframe.columns:
        raise KeyError(
            f"Expected column {first_vignette_value_column!r} in the cleaned dataframe, but it was not found."
        )

    between_long = pd.DataFrame(
        {
            "condition_code": analysis_dataframe["case_code_position_1"].astype(str).str.upper(),
            "rating_value": analysis_dataframe[first_vignette_value_column],
        }
    )
    between_long["condition_label"] = between_long["condition_code"].map(condition_code_to_distal_label)
    between_long = between_long.dropna(subset=["condition_label", "rating_value"]).copy()

    within_value_columns = {
        distal_category_labels_in_order[0]: f"clark_{dv_suffix}_cc",
        distal_category_labels_in_order[1]: f"clark_{dv_suffix}_ch",
        distal_category_labels_in_order[2]: f"clark_{dv_suffix}_div",
    }
    if include_proximate_agent:
        within_value_columns.update(
            {
                "CC Proximate": f"proximate_{dv_suffix}_cc",
                "CH Proximate": f"proximate_{dv_suffix}_ch",
            }
        )

    missing_within_columns = [
        col_name for col_name in within_value_columns.values() if col_name not in analysis_dataframe.columns
    ]
    if missing_within_columns:
        raise KeyError(
            "Missing one or more expected within-subject columns in the cleaned dataframe: "
            + ", ".join(repr(col) for col in missing_within_columns)
        )

    within_long = (
        analysis_dataframe[list(within_value_columns.values())]
        .rename(columns={column_name: label for label, column_name in within_value_columns.items()})
        .melt(var_name="condition_label", value_name="rating_value")
        .dropna(subset=["rating_value"])
        .copy()
    )

    show_between_panel = subjects_setting in {"both", "between"}
    show_within_panel = subjects_setting in {"both", "within"}
    n_rows = 2 if (show_between_panel and show_within_panel) else 1

    subplot_titles = (
        ["Between-subjects", "Within-subjects"]
        if n_rows == 2
        else (["Between-subjects"] if show_between_panel else ["Within-subjects"])
    )

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
    )

    initial_subplot_title_count = len(fig.layout.annotations)
    for annotation_index in range(initial_subplot_title_count):
        fig.layout.annotations[annotation_index].update(
            x=0.5, xanchor="center", font=dict(size=24, family="Calibri", color="black"),
        )

    trace_type_labels: list[str] = []

    def add_panel_traces(
        panel_long_dataframe: pd.DataFrame,
        category_labels_in_order: list[str],
        category_positions: dict[str, float],
        row: int,
    ) -> None:
        """Adds box and/or violin traces for a panel."""

        nonlocal trace_type_labels

        for category_index, category_label in enumerate(category_labels_in_order):
            category_values = panel_long_dataframe.loc[
                panel_long_dataframe["condition_label"] == category_label, "rating_value"
            ]

            category_center_x = category_positions[category_label]

            fill_color = _hsla_color(hue=base_hue + 20 * category_index, alpha=0.55)
            line_color = _hsla_color(hue=base_hue + 20 * category_index, alpha=1.00)
            point_color = _hsla_color(hue=base_hue + 20 * category_index, alpha=0.60)

            if include_toggle_dropdown or figure_type_normalized in {None, "box"}:
                fig.add_trace(
                    go.Box(
                        x=[category_center_x] * len(category_values),
                        y=category_values,
                        name=category_label,
                        width=explicit_box_width,
                        boxpoints="all",
                        jitter=0.45,
                        pointpos=0,
                        boxmean=True,
                        fillcolor=fill_color,
                        line=dict(color=line_color, width=4),
                        marker=dict(color=point_color, size=default_marker_size),
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
                trace_type_labels.append("box")

            if include_toggle_dropdown or figure_type_normalized == "violin":
                fig.add_trace(
                    go.Violin(
                        x=[category_center_x] * len(category_values),
                        y=category_values,
                        name=category_label,
                        width=explicit_box_width,
                        points="all",
                        jitter=0.45,
                        pointpos=0,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=fill_color,
                        line=dict(color=line_color, width=4),
                        marker=dict(color=point_color, size=default_marker_size),
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
                trace_type_labels.append("violin")

    current_row = 1
    if show_between_panel:
        add_panel_traces(
            panel_long_dataframe=between_long,
            category_labels_in_order=between_category_labels_in_order,
            category_positions=between_category_positions,
            row=current_row,
        )
        current_row += 1

    if show_within_panel:
        add_panel_traces(
            panel_long_dataframe=within_long,
            category_labels_in_order=within_category_labels_in_order,
            category_positions=within_category_positions,
            row=current_row,
        )

    "------------------------------"
    "------ Toggle visibility -----"
    "------------------------------"

    if include_toggle_dropdown:
        box_visible_mask = [trace_type == "box" for trace_type in trace_type_labels]
        violin_visible_mask = [trace_type == "violin" for trace_type in trace_type_labels]
        visible_mask_to_apply = box_visible_mask if initial_view == "box" else violin_visible_mask
        for trace, visible in zip(fig.data, visible_mask_to_apply):
            trace.visible = visible

    "------------------------------"
    "-- Mean-difference brackets --"
    "------------------------------"

    def add_mean_difference_brackets(
        panel_long_dataframe: pd.DataFrame,
        category_positions: dict[str, float],
        xaxis_ref: str,
        yaxis_ref: str,
    ) -> None:
        """
        Add horizontal mean-reference lines and compact annotations showing CH−CC and DIV−CC.

        The x positions are computed directly from the centers of the plotted boxes and the
        explicit box width, so the dashed lines begin/end at box borders rather than at guessed
        paper coordinates.

        Arguments:
            • panel_long_dataframe: pd.DataFrame
                - Long-form dataframe for one panel only.
            • category_positions: dict[str, float]
                - Mapping from condition labels to numeric x-axis centers for this panel.
            • xaxis_ref: str
                - Plotly x-axis reference for this panel (e.g., "x", "x2").
            • yaxis_ref: str
                - Plotly y-axis reference for this panel (e.g., "y", "y2").

        Returns:
            • None
        """
        cc_label = distal_category_labels_in_order[0]
        ch_label = distal_category_labels_in_order[1]
        div_label = distal_category_labels_in_order[2]

        mean_cc = float(
            panel_long_dataframe.loc[
                panel_long_dataframe["condition_label"] == cc_label,
                "rating_value"
            ].mean()
        )
        mean_ch = float(
            panel_long_dataframe.loc[
                panel_long_dataframe["condition_label"] == ch_label,
                "rating_value"
            ].mean()
        )
        mean_div = float(
            panel_long_dataframe.loc[
                panel_long_dataframe["condition_label"] == div_label,
                "rating_value"
            ].mean()
        )

        half_box_width = explicit_box_width / 2

        cc_center_x = category_positions[cc_label]
        ch_center_x = category_positions[ch_label]
        div_center_x = category_positions[div_label]

        cc_right_edge_x = cc_center_x + half_box_width
        ch_left_edge_x = ch_center_x - half_box_width
        ch_right_edge_x = ch_center_x + half_box_width
        div_left_edge_x = div_center_x - half_box_width

        first_gap_midpoint_x = (cc_right_edge_x + ch_left_edge_x) / 2
        second_gap_midpoint_x = (ch_right_edge_x + div_left_edge_x) / 2

        new_shapes = [
            # "CC mean line: from right edge of CC box to left edge of DIV box"
            dict(
                type="line",
                xref=xaxis_ref,
                yref=yaxis_ref,
                x0=cc_right_edge_x,
                x1=div_left_edge_x,
                y0=mean_cc,
                y1=mean_cc,
                line=dict(color="hsla(0, 0%, 0%, 0.65)", width=3.5, dash="dash"),
                layer="below",
            ),
            # "CH mean line: from right edge of CC box to left edge of CH box"
            dict(
                type="line",
                xref=xaxis_ref,
                yref=yaxis_ref,
                x0=cc_right_edge_x,
                x1=ch_left_edge_x,
                y0=mean_ch,
                y1=mean_ch,
                line=dict(color="hsla(0, 0%, 0%, 0.65)", width=3.5, dash="dash"),
                layer="below",
            ),
            # "DIV mean line: from right edge of CH box to left edge of DIV box"
            dict(
                type="line",
                xref=xaxis_ref,
                yref=yaxis_ref,
                x0=ch_right_edge_x,
                x1=div_left_edge_x,
                y0=mean_div,
                y1=mean_div,
                line=dict(color="hsla(0, 0%, 0%, 0.65)", width=3.5, dash="dash"),
                layer="below",
            ),
        ]

        existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        fig.update_layout(shapes=existing_shapes + new_shapes)

        fig.add_annotation(
            x=first_gap_midpoint_x,
            y=(mean_cc + mean_ch) / 2,
            xref=xaxis_ref,
            yref=yaxis_ref,
            text=f"Shielding<br>(CH - CC)<br>Δ={mean_ch - mean_cc:+.2f}",
            showarrow=False,
            align="center",
            xanchor="center",
            yanchor="middle",
            font=dict(size=22, family="Calibri", color="black"),
            bgcolor="hsla(0, 0%, 100%, 0.80)",
        )

        fig.add_annotation(
            x=second_gap_midpoint_x,
            y=(mean_cc + mean_div) / 2,
            xref=xaxis_ref,
            yref=yaxis_ref,
            text=f"Shielding<br>beyond<br>division<br>(DIV - CC)<br>Δ={mean_div - mean_cc:+.2f}",
            showarrow=False,
            align="center",
            xanchor="center",
            yanchor="middle",
            font=dict(size=22, family="Calibri", color="black"),
            bgcolor="hsla(0, 0%, 100%, 0.80)",
        )

    def add_proximate_manipulation_check_annotation(
        panel_long_dataframe: pd.DataFrame,
        category_positions: dict[str, float],
        xaxis_ref: str,
        yaxis_ref: str,
        left_label: str = "CC Proximate",
        right_label: str = "CH Proximate",
    ) -> None:
        """
        Add a two-line mean comparison and annotation for the proximate-agent manipulation check.

        Arguments:
            • panel_long_dataframe: pd.DataFrame
                - Long-form dataframe for one panel only.
            • category_positions: dict[str, float]
                - Mapping from condition labels to numeric x-axis centers for this panel.
            • xaxis_ref: str
                - Plotly x-axis reference for this panel (e.g., "x", "x2").
            • yaxis_ref: str
                - Plotly y-axis reference for this panel (e.g., "y", "y2").
            • left_label: str
                - Left-hand proximate category label. Defaults to "CC Proximate".
            • right_label: str
                - Right-hand proximate category label. Defaults to "CH Proximate".

        Returns:
            • None
        """

        mean_left = float(
            panel_long_dataframe.loc[
                panel_long_dataframe["condition_label"] == left_label,
                "rating_value"
            ].mean()
        )
        mean_right = float(
            panel_long_dataframe.loc[
                panel_long_dataframe["condition_label"] == right_label,
                "rating_value"
            ].mean()
        )

        half_box_width = explicit_box_width / 2

        left_center_x = category_positions[left_label]
        right_center_x = category_positions[right_label]

        left_right_edge_x = left_center_x + half_box_width
        right_left_edge_x = right_center_x - half_box_width
        midpoint_between_proximate_boxes_x = (left_right_edge_x + right_left_edge_x) / 2

        new_shapes = [
            dict(
                type="line",
                xref=xaxis_ref,
                yref=yaxis_ref,
                x0=left_right_edge_x,
                x1=right_left_edge_x,
                y0=mean_left,
                y1=mean_left,
                line=dict(color="hsla(0, 0%, 0%, 0.65)", width=3.5, dash="dash"),
                layer="below",
            ),
            dict(
                type="line",
                xref=xaxis_ref,
                yref=yaxis_ref,
                x0=left_right_edge_x,
                x1=right_left_edge_x,
                y0=mean_right,
                y1=mean_right,
                line=dict(color="hsla(0, 0%, 0%, 0.65)", width=3.5, dash="dash"),
                layer="below",
            ),
        ]

        existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        fig.update_layout(shapes=existing_shapes + new_shapes)

        fig.add_annotation(
            x=midpoint_between_proximate_boxes_x,
            y=(mean_left + mean_right) / 2,
            xref=xaxis_ref,
            yref=yaxis_ref,
            text=f"Manipulation<br>Check:<br>(CC - CH)<br>Δ={mean_left - mean_right:+.2f}",
            showarrow=False,
            align="center",
            xanchor="center",
            yanchor="middle",
            font=dict(size=22, family="Calibri", color="black"),
            bgcolor="hsla(0, 0%, 100%, 0.80)",
        )

    if show_between_panel:
        add_mean_difference_brackets(
            panel_long_dataframe=between_long,
            category_positions=between_category_positions,
            xaxis_ref="x",
            yaxis_ref="y",
        )

    if show_within_panel:
        xaxis_ref_within = "x" if n_rows == 1 else "x2"
        yaxis_ref_within = "y" if n_rows == 1 else "y2"
        add_mean_difference_brackets(
            panel_long_dataframe=within_long,
            category_positions=within_category_positions,
            xaxis_ref=xaxis_ref_within,
            yaxis_ref=yaxis_ref_within,
        )

        if include_proximate_agent:
            add_proximate_manipulation_check_annotation(
                panel_long_dataframe=within_long,
                category_positions=within_category_positions,
                xaxis_ref=xaxis_ref_within,
                yaxis_ref=yaxis_ref_within,
            )

    "------------------------------"
    "------- Layout and axes ------"
    "------------------------------"

    title_prefix = {
        "blame": "Blame Ratings",
        "wrong": "Wrongness Ratings",
        "punish": "Punishment Ratings",
    }[dv_suffix]

    fig_title = f"{title_prefix} by Vignette Condition"
    if story_condition == "firework":
        fig_title += " - Firework Story"
    elif story_condition == "trolley":
        fig_title += " - Trolley Story"
    if cognitive_load == "high":
        fig_title += " - High Load"
    elif cognitive_load == "low":
        fig_title += " - Low Load"

    fig.update_layout(**figure_layout, title=fig_title)

    if bounded_y_range is None:
        combined_values = pd.concat(
            [between_long["rating_value"], within_long["rating_value"]],
            ignore_index=True,
        )
        max_value = float(np.nanmax(combined_values)) if len(combined_values) else 1.0
        bounded_y_range = (0.0, max(1.0, max_value * 1.10))

    if dv_suffix == "blame":
        y_axis_title = "Blameworthiness Ratings" if include_proximate_agent else "Blameworthiness of Distal Agent"
    elif dv_suffix == "wrong":
        y_axis_title = "Wrongness Ratings" if include_proximate_agent else "Wrongness of Distal Agent"
    else:
        y_axis_title = "Punishment (years)" if include_proximate_agent else "Punishment of Distal Agent (years)"

    for row_index in range(1, n_rows + 1):
        if n_rows == 1:
            if show_between_panel:
                tick_values_for_row = between_axis_tick_values
                tick_text_for_row = between_axis_tick_text
            else:
                tick_values_for_row = within_axis_tick_values
                tick_text_for_row = within_axis_tick_text
        else:
            if row_index == 1:
                tick_values_for_row = between_axis_tick_values
                tick_text_for_row = between_axis_tick_text
            else:
                tick_values_for_row = within_axis_tick_values
                tick_text_for_row = within_axis_tick_text

        x_axis_min = min(tick_values_for_row) - 0.55
        x_axis_max = max(tick_values_for_row) + 0.55

        fig.update_xaxes(
            row=row_index,
            title_text="Vignette Condition" if row_index == n_rows else None,
            tickmode="array",
            tickvals=tick_values_for_row,
            ticktext=tick_text_for_row,
            range=[x_axis_min, x_axis_max],
            zeroline=False,
            showline=False,
            mirror=False,
            tickwidth=0,
            ticklen=0,
            ticks="",
            col=1,
        )

        fig.update_yaxes(
            row=row_index,
            title_text=y_axis_title,
            range=list(bounded_y_range),
            zeroline=False,
            showline=False,
            mirror=False,
            tickwidth=0,
            ticklen=0,
            ticks="",
            col=1,
        )

        if dv in ("blame", "wrong"):
            fig.update_yaxes(
                tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                ticktext=['1', '2', '3', '4', '5', '6', '7', '8', '9']
            )

        if include_toggle_dropdown:
            fig.update_layout(
                updatemenus=[
                    dict(
                        buttons=[
                            dict(
                                label="Box", method="update",
                                args=[{"visible": [t == "box" for t in trace_type_labels]}],
                            ),
                            dict(
                                label="Violin", method="update",
                                args=[{"visible": [t == "violin" for t in trace_type_labels]}],
                            ),
                        ],
                        direction="down",
                        x=1.02, xanchor="left",
                        y=1.02, yanchor="top",
                    )
                ]
            )

    if include_proximate_agent:
        if figure_type in ("box", "violin"):
            margins = dict(l=90, r=40, t=90, b=90)
        else:
            margins = dict(l=120, r=120, t=90, b=90)
    else:
        if figure_type in ("box", "violin"):
            margins = dict(l=160, r=140, t=90, b=90)
        else:
            margins = dict(l=200, r=200, t=90, b=90)

    fig.update_layout(title_x=0.5, title_xanchor="center", title_font_size=44, margin=margins)

    "------------------------------"
    "----------- Export -----------"
    "------------------------------"

    if export_html:
        story_tag = "pooled" if story_condition_normalized is None else story_condition_normalized
        load_tag = "pooled" if cognitive_load_normalized is None else cognitive_load_normalized
        proximate_tag = "with_proximate" if include_proximate_agent else "distal_only"
        file_name_figure = f"figure_3_ratings_{dv_suffix}_{subjects_setting}_{story_tag}_{load_tag}_{proximate_tag}"
        _export_plotly_figure_html(
            fig=fig,
            general_settings=general_settings,
            file_name=file_name_figure
        )

    return fig


def plot_participant_level_shielding_heatmap(
    general_settings: GeneralSettings,
    dv: str | Any = "blame",
    include_marginals: bool = True,
    annotate_regions: bool = True,
    story_condition: str | Any = None,
    cognitive_load: str | Any = None,
    only_included_participants: bool = True,
    export_html: bool = True,
    base_hue: int = 215,
) -> "object":
    """
    Plot a participant-level joint shielding map with CH - CC on one axis and DIV - CC on the other.

    For bounded integer-valued DVs (blame, wrongness), the heat map uses one-cell-per-integer-difference.
    For punishment, it falls back to evenly spaced bins.

    Optional marginal histograms summarize the one-dimensional distributions on the top and right.

    Arguments:
        • dv: str | Any
            - Which dependent variable to use. Supported: 'blame', 'wrong', 'punishment'.
        • include_marginals: bool
            - If True, show top and right marginal histograms.
        • annotate_regions: bool
            - If True, annotate psychologically relevant regions of the heat map.
        • story_condition: str | Any
            - If None, pool stories.
            - If 'firework' or 'trolley', filter to that story condition.
        • cognitive_load: str | Any
            - If None, pool load conditions.
            - If 'high' or 'low', filter to that load condition.
        • only_included_participants: bool
            - If True, restrict to participants who passed all comprehension checks.
        • export_html: bool
            - If True, export the Plotly figure to an .html file.
        • base_hue: int
            - Base hue for hsla colors.

    Returns:
        • plotly.graph_objects.Figure
    """
    dv_suffix, dv_label, _ = _normalize_dependent_variable_input(dv)

    analysis_dataframe = _get_filtered_plotting_dataframe(
        general_settings=general_settings,
        story_condition=story_condition,
        cognitive_load=cognitive_load,
        only_included_participants=only_included_participants,
    )

    x_column_name = f"clark_{dv_suffix}_ch_minus_cc"
    y_column_name = f"clark_{dv_suffix}_div_minus_cc"

    missing_required_columns = [
        column_name
        for column_name in [x_column_name, y_column_name]
        if column_name not in analysis_dataframe.columns
    ]
    if missing_required_columns:
        raise KeyError(
            "Missing one or more expected delta columns required for the shielding heat map: "
            + ", ".join(repr(column_name) for column_name in missing_required_columns)
        )

    joint_dataframe = analysis_dataframe[[x_column_name, y_column_name]].copy()
    joint_dataframe["x_delta"] = pd.to_numeric(joint_dataframe[x_column_name], errors="coerce")
    joint_dataframe["y_delta"] = pd.to_numeric(joint_dataframe[y_column_name], errors="coerce")
    joint_dataframe = joint_dataframe.dropna(subset=["x_delta", "y_delta"]).copy()

    if joint_dataframe.shape[0] == 0:
        raise ValueError("No valid paired CH - CC and DIV - CC deltas remain after filtering.")

    if dv_suffix in {"blame", "wrong"}:
        x_bin_edges = np.arange(-8.5, 9.5, 1.0)
        y_bin_edges = np.arange(-8.5, 9.5, 1.0)
        x_bin_centers = np.arange(-8.0, 9.0, 1.0)
        y_bin_centers = np.arange(-8.0, 9.0, 1.0)
        x_axis_range = [-8.5, 8.5]
        y_axis_range = [-8.5, 8.5]
    else:
        x_min = float(np.floor(joint_dataframe["x_delta"].min()))
        x_max = float(np.ceil(joint_dataframe["x_delta"].max()))
        y_min = float(np.floor(joint_dataframe["y_delta"].min()))
        y_max = float(np.ceil(joint_dataframe["y_delta"].max()))

        x_bin_edges = np.linspace(x_min - 0.5, x_max + 0.5, 21)
        y_bin_edges = np.linspace(y_min - 0.5, y_max + 0.5, 21)
        x_bin_centers = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2
        y_bin_centers = (y_bin_edges[:-1] + y_bin_edges[1:]) / 2
        x_axis_range = [float(x_bin_edges[0]), float(x_bin_edges[-1])]
        y_axis_range = [float(y_bin_edges[0]), float(y_bin_edges[-1])]

    heatmap_counts, _, _ = np.histogram2d(
        joint_dataframe["x_delta"],
        joint_dataframe["y_delta"],
        bins=[x_bin_edges, y_bin_edges],
    )
    heatmap_counts = heatmap_counts.T

    x_marginal_counts, _ = np.histogram(joint_dataframe["x_delta"], bins=x_bin_edges)
    y_marginal_counts, _ = np.histogram(joint_dataframe["y_delta"], bins=y_bin_edges)

    custom_heatmap_colorscale = [
        [0.00, _hsla_color(hue=base_hue, saturation_percent=100, lightness_percent=50, alpha=1.0)],
        [0.25, _hsla_color(hue=base_hue, saturation_percent=100, lightness_percent=50, alpha=1.0)],
        [0.50, _hsla_color(hue=base_hue, saturation_percent=100, lightness_percent=50, alpha=1.0)],
        [0.75, _hsla_color(hue=base_hue, saturation_percent=100, lightness_percent=50, alpha=1.0)],
        [1.00, _hsla_color(hue=base_hue, saturation_percent=100, lightness_percent=50, alpha=1.0)],
    ]

    custom_heatmap_colorscale = [
        [0.000, _hsla_color(hue=base_hue, saturation_percent= 0, lightness_percent=100, alpha=1.0)],
        [0.125, _hsla_color(hue=base_hue, saturation_percent=60, lightness_percent=50, alpha=1.0)],
        [0.250, _hsla_color(hue=base_hue, saturation_percent=70, lightness_percent=40, alpha=1.0)],
        [0.370, _hsla_color(hue=base_hue, saturation_percent=80, lightness_percent=30, alpha=1.0)],
        [0.500, _hsla_color(hue=base_hue, saturation_percent=90, lightness_percent=20, alpha=1.0)],
        [1.000, _hsla_color(hue=base_hue, saturation_percent=100, lightness_percent=10, alpha=1.0)],
    ]

    if include_marginals:
        fig = make_subplots(
            rows=2,
            cols=2,
            row_heights=[0.22, 0.78],
            column_widths=[0.78, 0.22],
            horizontal_spacing=0.00,
            vertical_spacing=0.00,
            specs=[
                [{"type": "bar"}, {"type": "xy"}],
                [{"type": "heatmap"}, {"type": "bar"}],
            ],
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "heatmap"}]],
        )

    if include_marginals:
        marginal_bar_color = _hsla_color(hue=base_hue, saturation_percent=100, lightness_percent=30, alpha=0.85)
        marginal_line_color = _hsla_color(hue=base_hue, saturation_percent=100, lightness_percent=20, alpha=0.95)

        fig.add_trace(
            go.Bar(
                x=x_bin_centers,
                y=x_marginal_counts,
                marker=dict(
                    color=marginal_bar_color, 
                    line=dict(color=[                                  
                            marginal_line_color if col > 0
                            else "hsla(0, 0%, 100%, 0.0)"
                            for col in x_marginal_counts
                        ], width=1.5)
                ),
                hovertemplate="CH - CC bin center: %{x}<br>Count: %{y}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=y_marginal_counts,
                y=y_bin_centers,
                orientation="h",
                marker=dict(
                    color=marginal_bar_color, 
                    line=dict(color=[                                  
                            marginal_line_color if col > 0
                            else "hsla(0, 0%, 100%, 0.0)"
                            for col in y_marginal_counts
                        ], width=1.5)
                ),
                hovertemplate="DIV - CC bin center: %{y}<br>Count: %{x}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    heatmap_trace = go.Heatmap(
        x=x_bin_centers,
        y=y_bin_centers,
        z=heatmap_counts,
        colorscale=custom_heatmap_colorscale,
        colorbar=dict(
            title=dict(text="Count", font=dict(size=24, family="Calibri", color="black")),
            tickfont=dict(size=20, family="Calibri", color="black"),
            thickness=34,
        ),
        hovertemplate=(
            "CH - CC: %{x}<br>"
            "DIV - CC: %{y}<br>"
            "Count: %{z}<extra></extra>"
        ),
    )

    if include_marginals:
        fig.add_trace(heatmap_trace, row=2, col=1)
    else:
        fig.add_trace(heatmap_trace, row=1, col=1)

    figure_title = f"Participant-Level Shielding Map ({dv_label})"
    fig.update_layout(**figure_layout, title=figure_title)
    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_font_size=32,
        margin=dict(l=590, r=590, t=90, b=90),
        showlegend=False,
        bargap=0.0
    )

    heat_tickvals=[-8, -6, -4, -2, 0, 2, 4, 6, 8]
    heat_ticktext=['-8', '-6', '-4', '-2', '0', '2', '4', '6', '8']

    marginal_tickvals=[0, 20, 40, 60, 80]
    marginal_ticktext=['', '20', '40', '60', '80']

    if include_marginals:
        "Top histogram"
        fig.update_xaxes(
            row=1,
            col=1,
            range=x_axis_range,
            tickvals=marginal_tickvals,
            ticktext=marginal_ticktext,
            matches="x3",
            ticks="",
            ticklen=0,
            tickwidth=0,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            showline=False,
            mirror=False,
        )
        fig.update_yaxes(
            row=1,
            col=1,
            title_text="Count",
            tickvals=marginal_tickvals,
            ticktext=marginal_ticktext,
            ticks="",
            ticklen=0,
            tickwidth=0,
            zeroline=False,
            showgrid=False,
            showline=False,
            mirror=False,
        )

        "Right histogram"
        fig.update_xaxes(
            row=2,
            col=2,
            title_text="Count",
            tickvals=marginal_tickvals,
            ticktext=marginal_ticktext,
            ticks="",
            ticklen=0,
            tickwidth=0,
            zeroline=False,
            showgrid=False,
            showline=False,
            mirror=False,
        )
        fig.update_yaxes(
            row=2,
            col=2,
            range=y_axis_range,
            tickvals=marginal_tickvals,
            ticktext=marginal_ticktext,
            matches="y3",
            ticks="",
            ticklen=0,
            tickwidth=0,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            showline=False,
            mirror=False,
        )

        "Blank top-right panel"
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)

        "Heat map panel"
        fig.update_xaxes(
            row=2,
            col=1,
            title_text="Choice-Chance - Choice-Choice (CH - CC)",
            range=x_axis_range,
            tickvals=heat_tickvals, 
            ticktext=heat_ticktext, 
            ticks="",
            ticklen=0,
            tickwidth=0,
            zeroline=False,
            scaleanchor="y3",
            scaleratio=1,
            constrain="domain"
        )
        fig.update_yaxes(
            row=2,
            col=1,
            title_text="Division - Choice-Choice (DIV - CC)",
            range=y_axis_range,
            tickvals=heat_tickvals, 
            ticktext=heat_ticktext, 
            ticks="",
            ticklen=0,
            tickwidth=0,
            zeroline=False,
            constrain="domain",
            scaleanchor="x3",
            scaleratio=1
        )

        heatmap_xref = "x3"
        heatmap_yref = "y3"
    else:
        fig.update_xaxes(
            scaleratio=1,
            scaleanchor="y",
            title_text="CH - CC",
            tickvals=heat_tickvals, 
            ticktext=heat_ticktext, 
            range=x_axis_range,
            zeroline=False,
            showline=False,
            mirror=False,
            tickwidth=0,
            ticklen=0,
            ticks="",
            row=1,
            col=1,

        )
        fig.update_yaxes(
            title_text="DIV - CC",
            tickvals=heat_tickvals, 
            ticktext=heat_ticktext, 
            range=y_axis_range,
            zeroline=False,
            showline=False,
            mirror=False,
            tickwidth=0,
            ticklen=0,
            ticks="",
            row=1,
            col=1,
        )

        heatmap_xref = "x"
        heatmap_yref = "y"

    existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    new_shapes = [
        dict(
            type="line",
            xref=heatmap_xref,
            yref=heatmap_yref,
            x0=0.0,
            x1=0.0,
            y0=y_axis_range[0],
            y1=y_axis_range[1],
            line=dict(color="hsla(0, 0%, 0%, 0.55)", width=3, dash="dash"),
            layer="above",
        ),
        dict(
            type="line",
            xref=heatmap_xref,
            yref=heatmap_yref,
            x0=x_axis_range[0],
            x1=x_axis_range[1],
            y0=0.0,
            y1=0.0,
            line=dict(color="hsla(0, 0%, 0%, 0.55)", width=3, dash="dash"),
            layer="above",
        ),
        dict(
            type="line",
            xref=heatmap_xref,
            yref=heatmap_yref,
            x0=max(x_axis_range[0], y_axis_range[0]),
            x1=min(x_axis_range[1], y_axis_range[1]),
            y0=max(x_axis_range[0], y_axis_range[0]),
            y1=min(x_axis_range[1], y_axis_range[1]),
            line=dict(color="hsla(0, 0%, 0%, 0.45)", width=3, dash="dot"),
            layer="above",
        ),
    ]
    fig.update_layout(shapes=existing_shapes + new_shapes)

    if annotate_regions:
        x_values = joint_dataframe["x_delta"].to_numpy(dtype=float)
        y_values = joint_dataframe["y_delta"].to_numpy(dtype=float)
        total_n = joint_dataframe.shape[0]

        "Fractional region counts"
        "Border classification helpers"
        on_x_axis   = y_values == 0   # lies on y=0 line
        on_y_axis   = x_values == 0   # lies on x=0 line
        on_diagonal = x_values == y_values  # lies on y=x diagonal

        "Strict interior masks (no boundary points included)"
        q_strict  = (x_values > 0) & (y_values > 0) & (x_values > y_values)    # quintessential
        dd_strict = (x_values > 0) & (y_values > 0) & (y_values > x_values)    # division-dominant
        ch_strict = (x_values > 0) & (y_values < 0)                            # CH-only
        dv_strict = (x_values < 0) & (y_values > 0)                            # DIV-only
        rn_strict = (x_values < 0) & (y_values < 0)                            # reverse/null

        "(x=0, y<0) and (x<0, y=0) also belong strictly to reverse/null"
        rn_strict = rn_strict | ((x_values == 0) & (y_values < 0)) | ((x_values < 0) & (y_values == 0))

        "Boundary points and their fractional allocations"
        "1. Origin (0, 0): Q=0.125, DD=0.125, CH=0.25, DIV=0.25, RN=0.25"
        at_origin = (x_values == 0) & (y_values == 0)

        "2. Positive x-axis border (x>0, y=0): shared Q / CH  → 0.5 each"
        pos_x_border = (x_values > 0) & on_x_axis & ~at_origin

        "3. Positive y-axis border (x=0, y>0): shared DD / DIV → 0.5 each"
        pos_y_border = (x_values == 0) & (y_values > 0) & ~at_origin

        "4. Positive diagonal (x>0, y>0, x==y): shared Q / DD → 0.5 each"
        pos_diag_border = (x_values > 0) & (y_values > 0) & on_diagonal

        counts = {}
        counts["q"]  = (q_strict.sum()
                        + 0.5 * pos_x_border.sum()
                        + 0.5 * pos_diag_border.sum()
                        + 0.125 * at_origin.sum())
        counts["dd"] = (dd_strict.sum()
                        + 0.5 * pos_y_border.sum()
                        + 0.5 * pos_diag_border.sum()
                        + 0.125 * at_origin.sum())
        counts["ch"] = (ch_strict.sum()
                        + 0.5 * pos_x_border.sum()
                        + 0.25 * at_origin.sum())
        counts["dv"] = (dv_strict.sum()
                        + 0.5 * pos_y_border.sum()
                        + 0.25 * at_origin.sum())
        counts["rn"] = (rn_strict.sum()
                        + 0.25 * at_origin.sum())

        "Sanity check: fractional counts should sum to total_n"
        assert abs(sum(counts.values()) - total_n) < 1e-9, sum(counts.values())

        region_summary = [
            ("Quintessential<br>shielders", counts["q"],  6.5,  2.5),
            ("Division<br>-dominant",       counts["dd"], 3.0,  6.5),
            ("CH-only",                     counts["ch"], 5.5, -5.5),
            ("DIV-only",                    counts["dv"],-5.5,  5.5),
            ("Reverse/null",                counts["rn"],-5.5, -5.5),
        ]


        for region_label, region_count, annotation_x_value, annotation_y_value in region_summary:
            fig.add_annotation(
                x=annotation_x_value,
                y=annotation_y_value,
                xref=heatmap_xref,
                yref=heatmap_yref,
                text=f"{region_label}<br>𝑛={int(region_count)}<br>{region_count / total_n:.0%}",
                showarrow=False,
                align="center",
                xanchor="center",
                yanchor="middle",
                font=dict(size=18, family="Calibri", color="black"),
                bgcolor="hsla(0, 0%, 100%, 0.78)",
            )

    if export_html:
        story_condition_normalized = _normalize_story_condition_input(story_condition)
        cognitive_load_normalized = _normalize_load_condition_input(cognitive_load)
        story_tag = "pooled" if story_condition_normalized is None else story_condition_normalized
        load_tag = "pooled" if cognitive_load_normalized is None else cognitive_load_normalized
        marginals_tag = "with_marginals" if include_marginals else "heat_only"

        file_name_figure = f"figure_4_shielding_heatmap_{dv_suffix}_{story_tag}_{load_tag}_{marginals_tag}"
        _export_plotly_figure_html(
            fig=fig,
            general_settings=general_settings,
            file_name=file_name_figure
        )

    return fig


def plot_within_subject_pairwise_comparisons(
    general_settings: GeneralSettings,
    dv: str | Any = "blame",
    include_proximate_agent: bool = True,
    story_condition: str | Any = None,
    cognitive_load: str | Any = None,
    only_included_participants: bool = True,
    export_html: bool = True,
    export_csv: bool = False,
    base_hue: int = 210,
) -> tuple["object", pd.DataFrame]:
    """
    Render a Plotly table summarizing the within-subject pairwise comparisons among the key rating series.

    Cell entries report:
        • mean difference (column - row)
        • Cohen's dz
        • p-value from a paired t-test

    Arguments:
        • dv: str | Any
            - Which dependent variable to use. Supported: 'blame', 'wrong', 'punishment'.
        • include_proximate_agent: bool
            - If True, include proximate ratings (CC Proximate, CH Proximate).
        • story_condition: str | Any
            - If None, pool stories.
            - If 'firework' or 'trolley', filter to that story condition.
        • cognitive_load: str | Any
            - If None, pool load conditions.
            - If 'high' or 'low', filter to that load condition.
        • only_included_participants: bool
            - If True, restrict to participants who passed all comprehension checks.
        • export_html: bool
            - If True, export the Plotly figure to an .html file.
        • export_csv: bool
            - If True, export the long statistics and formatted matrix as CSV files.
        • base_hue: int
            - Base hue for hsla table styling.

    Returns:
        • tuple[plotly.graph_objects.Figure, pd.DataFrame]
            - Plotly table figure
            - long-format pairwise statistics dataframe
    """
    def plot_within_subject_pairwise_comparison_matrix(
        general_settings: GeneralSettings,
        dv: str | Any = "blame",
        include_proximate_agent: bool = True,
        story_condition: str | Any = None,
        cognitive_load: str | Any = None,
        only_included_participants: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build a compact upper-triangular within-subject comparison matrix.

        Each non-empty cell reports the paired comparison corresponding to:
            column minus row

        and includes:
            • mean difference
            • Cohen's dz
            • p-value from a paired t-test

        Arguments:
            • dv: str | Any
                - Which dependent variable to use. Supported: 'blame', 'wrong', 'punishment'.
            • include_proximate_agent: bool
                - If True, include CC Proximate and CH Proximate in addition to the three distal categories.
            • story_condition: str | Any
                - If None, pool stories.
                - If 'firework' or 'trolley', filter to that story condition.
            • cognitive_load: str | Any
                - If None, pool load conditions.
                - If 'high' or 'low', filter to that load condition.
            • only_included_participants: bool
                - If True, restrict to participants who passed all comprehension checks.

        Returns:
            • tuple[pd.DataFrame, pd.DataFrame]
                - pairwise_long_dataframe:
                    tidy row-per-comparison statistics
                - formatted_matrix_dataframe:
                    upper-triangular matrix of formatted cell strings
        """
        dv_suffix, _, _ = _normalize_dependent_variable_input(dv)

        analysis_dataframe = _get_filtered_plotting_dataframe(
            general_settings=general_settings,
            story_condition=story_condition,
            cognitive_load=cognitive_load,
            only_included_participants=only_included_participants,
        )

        if dv_suffix == "blame":
            ordered_category_metadata = [
                ("CC Distal", "clark_blame_cc"),
                ("CH Distal", "clark_blame_ch"),
                ("DIV Distal", "clark_blame_div"),
            ]
            if include_proximate_agent:
                ordered_category_metadata += [
                    ("CC Proximate", "proximate_blame_cc"),
                    ("CH Proximate", "proximate_blame_ch"),
                ]
        elif dv_suffix == "wrong":
            ordered_category_metadata = [
                ("CC Distal", "clark_wrong_cc"),
                ("CH Distal", "clark_wrong_ch"),
                ("DIV Distal", "clark_wrong_div"),
            ]
            if include_proximate_agent:
                ordered_category_metadata += [
                    ("CC Proximate", "proximate_wrong_cc"),
                    ("CH Proximate", "proximate_wrong_ch"),
                ]
        else:
            ordered_category_metadata = [
                ("CC Distal", "clark_punish_cc"),
                ("CH Distal", "clark_punish_ch"),
                ("DIV Distal", "clark_punish_div"),
            ]
            if include_proximate_agent:
                ordered_category_metadata += [
                    ("CC Proximate", "proximate_punish_cc"),
                    ("CH Proximate", "proximate_punish_ch"),
                ]

        missing_required_columns = [
            column_name
            for _, column_name in ordered_category_metadata
            if column_name not in analysis_dataframe.columns
        ]
        if missing_required_columns:
            raise KeyError(
                "Missing one or more expected rating columns for the pairwise comparison matrix: "
                + ", ".join(repr(column_name) for column_name in missing_required_columns)
            )

        category_summary_rows: list[dict[str, Any]] = []
        for category_label, column_name in ordered_category_metadata:
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
            index=[category_label for category_label, _ in ordered_category_metadata],
            columns=[category_label for category_label, _ in ordered_category_metadata],
            dtype=object,
        )

        for row_index, (row_label, row_column_name) in enumerate(ordered_category_metadata):
            for column_index, (column_label, column_column_name) in enumerate(ordered_category_metadata):
                if row_index == column_index:
                    formatted_matrix_dataframe.loc[row_label, column_label] = "—"
                    continue

                if column_index < row_index:
                    formatted_matrix_dataframe.loc[row_label, column_label] = ""
                    continue

                paired_dataframe = analysis_dataframe[[row_column_name, column_column_name]].copy()
                paired_dataframe[row_column_name] = pd.to_numeric(paired_dataframe[row_column_name], errors="coerce")
                paired_dataframe[column_column_name] = pd.to_numeric(paired_dataframe[column_column_name], errors="coerce")
                paired_dataframe = paired_dataframe.dropna()

                paired_difference_values = (
                    paired_dataframe[column_column_name] - paired_dataframe[row_column_name]
                ).to_numpy(dtype=float)

                n_pairs = int(paired_difference_values.shape[0])
                mean_difference = float(np.mean(paired_difference_values)) if n_pairs > 0 else np.nan

                if n_pairs > 1:
                    t_statistic, p_value = stats.ttest_1samp(paired_difference_values, popmean=0.0)
                    standard_deviation_of_difference = float(np.std(paired_difference_values, ddof=1))
                    if standard_deviation_of_difference == 0:
                        cohen_dz = 0.0 if mean_difference == 0 else np.nan
                    else:
                        cohen_dz = mean_difference / standard_deviation_of_difference
                else:
                    t_statistic = np.nan
                    p_value = np.nan
                    cohen_dz = np.nan

                pairwise_rows.append(
                    {
                        "row_label": row_label,
                        "column_label": column_label,
                        "row_column_name": row_column_name,
                        "column_column_name": column_column_name,
                        "n_pairs": n_pairs,
                        "mean_difference_column_minus_row": mean_difference,
                        "t_statistic": float(t_statistic) if not pd.isna(t_statistic) else np.nan,
                        "p_value": float(p_value) if not pd.isna(p_value) else np.nan,
                        "cohen_dz": float(cohen_dz) if not pd.isna(cohen_dz) else np.nan,
                    }
                )

                formatted_matrix_dataframe.loc[row_label, column_label] = (
                    f"Δ={mean_difference:+.2f}<br>"
                    f"dz={cohen_dz:+.2f}<br>"
                    f"p={_format_p_value_for_display(p_value)}"
                )

        pairwise_long_dataframe = pd.DataFrame(pairwise_rows)

        formatted_header_labels = []
        for _, category_summary_row in category_summary_dataframe.iterrows():
            formatted_header_labels.append(
                f"{category_summary_row['category_label']}<br>"
                f"M={category_summary_row['mean']:.2f}, "
                f"SD={category_summary_row['std']:.2f}"
            )

        formatted_matrix_dataframe.index = formatted_header_labels
        formatted_matrix_dataframe.columns = formatted_header_labels

        return pairwise_long_dataframe, formatted_matrix_dataframe

    dv_suffix, dv_label, _ = _normalize_dependent_variable_input(dv)

    pairwise_long_dataframe, formatted_matrix_dataframe = plot_within_subject_pairwise_comparison_matrix(
        dv=dv,
        include_proximate_agent=include_proximate_agent,
        story_condition=story_condition,
        cognitive_load=cognitive_load,
        only_included_participants=only_included_participants,
        general_settings=general_settings,
    )

    header_values = ["Row / Column<br>(cell = column - row)"] + list(formatted_matrix_dataframe.columns)
    cell_values = [list(formatted_matrix_dataframe.index)] + [
        formatted_matrix_dataframe[column_name].tolist()
        for column_name in formatted_matrix_dataframe.columns
    ]

    header_fill_color = _hsla_color(hue=base_hue, saturation_percent=70, lightness_percent=90, alpha=1.0)
    cells_fill_color = _hsla_color(hue=base_hue + 20, saturation_percent=25, lightness_percent=98, alpha=1.0)
    line_color = _hsla_color(hue=0, saturation_percent=0, lightness_percent=60, alpha=0.85)

    n_columns = len(header_values)
    first_column_width = 260
    other_column_width = 170
    column_widths = [first_column_width] + [other_column_width] * (n_columns - 1)

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=column_widths,
                header=dict(
                    values=header_values,
                    align="center",
                    fill_color=header_fill_color,
                    font=dict(family="Calibri", size=18, color="black"),
                    line=dict(color=line_color, width=1.5),
                    height=44,
                ),
                cells=dict(
                    values=cell_values,
                    align="center",
                    fill_color=cells_fill_color,
                    font=dict(family="Calibri", size=16, color="black"),
                    line=dict(color=line_color, width=1.0),
                    height=58,
                ),
            )
        ]
    )

    figure_title = f"Within-Subject Pairwise Comparison Matrix ({dv_label})"
    fig.update_layout(**figure_layout, title=figure_title)
    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_font_size=36,
        margin=dict(l=40, r=40, t=90, b=40),
    )

    fig.add_annotation(
        x=0.5,
        y=1.02,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(size=18, family="Calibri", color="black"),
        bgcolor="hsla(0, 0%, 100%, 0.0)",
        text="Each non-empty cell reports column - row: mean difference (Δ), Cohen's dz, and paired-test p-value.",
    )

    story_condition_normalized = _normalize_story_condition_input(story_condition)
    cognitive_load_normalized = _normalize_load_condition_input(cognitive_load)
    story_tag = "pooled" if story_condition_normalized is None else story_condition_normalized
    load_tag = "pooled" if cognitive_load_normalized is None else cognitive_load_normalized
    proximate_tag = "with_proximate" if include_proximate_agent else "distal_only"

    file_name_figure = f"figure_x_pairwise_matrix_{dv_suffix}_{story_tag}_{load_tag}_{proximate_tag}"
    file_path_tables = general_settings["filing"]["file_paths"]["tables"]

    if export_html:
        _export_plotly_figure_html(
            fig=fig,
            general_settings=general_settings,
            file_name=file_name_figure
        )

    if export_csv:
        pairwise_long_dataframe.to_csv(
            file_path_tables / f"{file_name_figure}_long.csv",
            index=False,
        )
        formatted_matrix_dataframe.to_csv(
            file_path_tables / f"{file_name_figure}_matrix.csv",
            index=True,
        )

    return fig, pairwise_long_dataframe


def plot_shielding_effects_by_cognitive_load(
    general_settings: GeneralSettings,
    dv: str | Any = "blame",
    delta_type: str | Sequence[str] | Any | None = "both",
    figure_type: str | Any = None,
    story_condition: str | Any = None,
    only_included_participants: bool = True,
    export_html: bool = True,
    base_hue: int = 200,
) -> "object":
    """
    Plot shielding-effect deltas by cognitive load using boxplots or violin plots.

    When both deltas are selected, the figure is displayed as two side-by-side subplots:
        • CH - CC
        • DIV - CC

    Each subplot compares the low-load and high-load distributions for that delta and includes:
        • a strong horizontal reference line at y = 0
        • a dashed horizontal mean line for low load
        • a dashed horizontal mean line for high load
        • an annotation between the two boxes reporting High - Low

    Arguments:
        • dv: str | Any
            - Which dependent variable to use. Supported: 'blame', 'wrong', 'punishment'.
        • delta_type: str | Sequence[str] | Any | None
            - Which delta(s) to include. Supports:
                "both"
                "CH_CC"
                "DIV_CC"
        • figure_type: str | Any
            - If 'box', render box plots.
            - If 'violin', render violin plots.
            - Otherwise, include a dropdown that toggles between box and violin.
        • story_condition: str | Any
            - If None, pool stories.
            - If 'firework' or 'trolley', filter to that story condition.
        • only_included_participants: bool
            - If True, restrict to participants who passed all comprehension checks.
        • export_html: bool
            - If True, export the Plotly figure to an .html file.
        • base_hue: int
            - Starting hue for hsla colors.

    Returns:
        • plotly.graph_objects.Figure
    """
    dv_suffix, dv_label, _ = _normalize_dependent_variable_input(dv)

    selected_delta_types = _normalize_delta_type_input(delta_type)
    selected_delta_types = [delta_name for delta_name in ["CH_CC", "DIV_CC"] if delta_name in selected_delta_types]

    figure_type_normalized = None if figure_type is None else str(figure_type).strip().lower()
    include_toggle_dropdown = figure_type_normalized not in {"box", "violin"}
    initial_view = "violin" if include_toggle_dropdown else figure_type_normalized

    analysis_dataframe = _get_filtered_plotting_dataframe(
        general_settings=general_settings,
        story_condition=story_condition,
        cognitive_load=None,
        only_included_participants=only_included_participants,
    )

    delta_long_dataframe = _build_delta_long_dataframe(
        analysis_dataframe=analysis_dataframe,
        dv_suffix=dv_suffix,
        delta_types=selected_delta_types,
    )

    delta_metadata = _get_delta_metadata(dv_suffix=dv_suffix)
    display_label_by_delta_type = {
        delta_name: delta_metadata[delta_name]["label"].replace("−", "-")
        for delta_name in selected_delta_types
    }

    n_columns = len(selected_delta_types)
    explicit_box_width = 0.46
    half_box_width = explicit_box_width / 2
    x_axis_min = -0.55
    x_axis_max = 1.55
    y_axis_min = -8.0
    y_axis_max = 8.0
    y_axis_span = y_axis_max - y_axis_min
    x_axis_span = x_axis_max - x_axis_min
    panel_scaleratio = y_axis_span / x_axis_span

    load_label_order = ["low", "high"]
    load_display_labels = {
        "low": "Low load",
        "high": "High load",
    }
    load_x_positions = {
        "low": 0.0,
        "high": 1.0,
    }

    subplot_titles = [display_label_by_delta_type[delta_name] for delta_name in selected_delta_types] if n_columns > 1 else None

    fig = make_subplots(
        rows=1, cols=n_columns,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12 if n_columns > 1 else 0.0,
        shared_yaxes=False,
    )

    if subplot_titles is not None:
        initial_subplot_title_count = len(fig.layout.annotations)
        for annotation_index in range(initial_subplot_title_count):
            fig.layout.annotations[annotation_index].update(
                xanchor="center",
                font=dict(size=30, family="Calibri", color="black"),
            )

    trace_type_labels: list[str] = []

    def add_delta_panel_traces(
        delta_subset_dataframe: pd.DataFrame,
        subplot_column: int,
        panel_base_hue: int,
    ) -> None:
        """
        Add low-load and high-load box/violin traces for one delta panel.

        Arguments:
            • delta_subset_dataframe: pd.DataFrame
                - Long dataframe subset for one delta type.
            • subplot_column: int
                - Subplot column index.
            • panel_base_hue: int
                - Base hue for this panel's two load conditions.

        Returns:
            • None
        """
        nonlocal trace_type_labels

        for load_index, load_condition_label in enumerate(load_label_order):
            load_values = pd.to_numeric(
                delta_subset_dataframe.loc[
                    delta_subset_dataframe["load_condition"] == load_condition_label,
                    "delta_value"
                ],
                errors="coerce",
            ).dropna()

            category_center_x = load_x_positions[load_condition_label]
            fill_color = _hsla_color(hue=panel_base_hue + 20 * load_index, alpha=0.55)
            line_color = _hsla_color(hue=panel_base_hue + 20 * load_index, alpha=1.00)
            point_color = _hsla_color(hue=panel_base_hue + 20 * load_index, alpha=0.60)

            if include_toggle_dropdown or figure_type_normalized in {None, "box"}:
                fig.add_trace(
                    go.Box(
                        x=[category_center_x] * len(load_values),
                        y=load_values,
                        name=load_display_labels[load_condition_label],
                        width=explicit_box_width,
                        boxpoints="all",
                        jitter=0.45,
                        pointpos=0,
                        boxmean=True,
                        fillcolor=fill_color,
                        line=dict(color=line_color, width=4),
                        marker=dict(color=point_color, size=default_marker_size),
                        showlegend=False,
                    ),
                    row=1,
                    col=subplot_column,
                )
                trace_type_labels.append("box")

            if include_toggle_dropdown or figure_type_normalized == "violin":
                fig.add_trace(
                    go.Violin(
                        x=[category_center_x] * len(load_values),
                        y=load_values,
                        name=load_display_labels[load_condition_label],
                        width=explicit_box_width,
                        points="all",
                        jitter=0.45,
                        pointpos=0,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=fill_color,
                        line=dict(color=line_color, width=4),
                        marker=dict(color=point_color, size=default_marker_size),
                        showlegend=False,
                    ),
                    row=1,
                    col=subplot_column,
                )
                trace_type_labels.append("violin")

    def add_panel_comparison_lines_and_annotation(
        delta_subset_dataframe: pd.DataFrame,
        subplot_column: int,
        delta_display_label: str,
        xaxis_ref: str,
        yaxis_ref: str,
    ) -> None:
        """
        Add:
            • strong horizontal reference line at 0
            • low-load and high-load mean comparison lines
            • annotation reporting High - Low

        Arguments:
            • delta_subset_dataframe: pd.DataFrame
                - Long dataframe subset for one delta type.
            • subplot_column: int
                - Subplot column index.
            • delta_display_label: str
                - Display label for the delta, e.g., "CH - CC".
            • xaxis_ref: str
                - Plotly x-axis reference for this subplot.
            • yaxis_ref: str
                - Plotly y-axis reference for this subplot.

        Returns:
            • None
        """
        low_values = pd.to_numeric(
            delta_subset_dataframe.loc[
                delta_subset_dataframe["load_condition"] == "low",
                "delta_value"
            ],
            errors="coerce",
        ).dropna()

        high_values = pd.to_numeric(
            delta_subset_dataframe.loc[
                delta_subset_dataframe["load_condition"] == "high",
                "delta_value"
            ],
            errors="coerce",
        ).dropna()

        if low_values.shape[0] == 0 or high_values.shape[0] == 0:
            return

        mean_low = float(low_values.mean())
        mean_high = float(high_values.mean())

        low_center_x = load_x_positions["low"]
        high_center_x = load_x_positions["high"]

        low_right_edge_x = low_center_x + half_box_width
        high_left_edge_x = high_center_x - half_box_width
        midpoint_between_boxes_x = (low_right_edge_x + high_left_edge_x) / 2

        new_shapes = [
            dict(
                type="line",
                xref=xaxis_ref,
                yref=yaxis_ref,
                x0=x_axis_min,
                x1=x_axis_max,
                y0=0.0, y1=0.0,
                line=dict(color="hsla(0, 0%, 40%, 0.80)", width=2.5, dash="dash"),
                layer="above",
            ),
            dict(
                type="line",
                xref=xaxis_ref,
                yref=yaxis_ref,
                x0=low_right_edge_x,
                x1=high_left_edge_x,
                y0=mean_low, y1=mean_low,
                line=dict(color="hsla(0, 0%, 0%, 0.65)", width=3.2, dash="dash"),
                layer="above",
            ),
            dict(
                type="line",
                xref=xaxis_ref,
                yref=yaxis_ref,
                x0=low_right_edge_x,
                x1=high_left_edge_x,
                y0=mean_high,
                y1=mean_high,
                line=dict(color="hsla(0, 0%, 0%, 0.65)", width=3.2, dash="dash"),
                layer="above",
            ),
        ]

        existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        fig.update_layout(shapes=existing_shapes + new_shapes)

        fig.add_annotation(
            x=midpoint_between_boxes_x,
            y=(mean_low + mean_high) / 2,
            xref=xaxis_ref,
            yref=yaxis_ref,
            text=f"High - Low<br>Δ={mean_high - mean_low:+.2f}",
            showarrow=False,
            align="center",
            xanchor="center",
            yanchor="middle",
            font=dict(size=22, family="Calibri", color="black"),
            bgcolor="hsla(0, 0%, 100%, 0.80)",
        )

    panel_base_hues = {
        "CH_CC": base_hue,
        "DIV_CC": base_hue + 60,
    }

    for subplot_index, selected_delta_type in enumerate(selected_delta_types, start=1):
        delta_subset = delta_long_dataframe.loc[
            delta_long_dataframe["delta_type"] == selected_delta_type
        ].copy()

        add_delta_panel_traces(
            delta_subset_dataframe=delta_subset,
            subplot_column=subplot_index,
            panel_base_hue=panel_base_hues[selected_delta_type],
        )

    if include_toggle_dropdown:
        violin_visible_mask = [trace_type_label == "violin" for trace_type_label in trace_type_labels]
        box_visible_mask = [trace_type_label == "box" for trace_type_label in trace_type_labels]
        visible_mask_to_apply = violin_visible_mask if initial_view == "violin" else box_visible_mask

        for trace, visible in zip(fig.data, visible_mask_to_apply):
            trace.visible = visible

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label="Violin",
                            method="update",
                            args=[{"visible": violin_visible_mask}],
                        ),
                        dict(
                            label="Box",
                            method="update",
                            args=[{"visible": box_visible_mask}],
                        ),
                    ],
                    direction="down",
                    x=0.8,
                    xanchor="left",
                    y=1.02,
                    yanchor="top",
                )
            ]
        )

    for subplot_index, selected_delta_type in enumerate(selected_delta_types, start=1):
        xaxis_ref = "x" if subplot_index == 1 else f"x{subplot_index}"
        yaxis_ref = "y" if subplot_index == 1 else f"y{subplot_index}"

        add_panel_comparison_lines_and_annotation(
            delta_subset_dataframe=delta_long_dataframe.loc[
                delta_long_dataframe["delta_type"] == selected_delta_type
            ].copy(),
            subplot_column=subplot_index,
            delta_display_label=display_label_by_delta_type[selected_delta_type],
            xaxis_ref=xaxis_ref,
            yaxis_ref=yaxis_ref,
        )

    if selected_delta_types == ["CH_CC"]:
        figure_title = f"Shielding Effect by Cognitive Load: CH - CC"
    elif selected_delta_types == ["DIV_CC"]:
        figure_title = f"Shielding Beyond Division<br>by Cognitive Load: DIV - CC"
    else:
        figure_title = f"Responsibility Shielding Effects by Cognitive Load ({dv_label})"

    if n_columns == 1:
        figure_width = 760
        figure_height = 760
    else:
        figure_width = 1280
        figure_height = 760

    if delta_type in ("CH_CC", "DIV_CC") or isinstance(delta_type, (list, tuple)):
        title_font_size = 24
    else:
        title_font_size = 40

    fig.update_layout(**figure_layout, title=figure_title)
    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_font_size=title_font_size,
        margin=dict(l=80, r=80, t=90, b=80),
        showlegend=False,
        width=figure_width,
        height=figure_height,
    )

    for subplot_index in range(1, n_columns + 1):
        xaxis_name = "xaxis" if subplot_index == 1 else f"xaxis{subplot_index}"
        yaxis_name = "yaxis" if subplot_index == 1 else f"yaxis{subplot_index}"
        xaxis_ref = "y" if subplot_index == 1 else f"y{subplot_index}"

        fig.update_xaxes(
            
            col=subplot_index,
            title_text="Cognitive Load" if n_columns == 1 else None,
            tickmode="array",
            tickvals=[0.0, 1.0],
            ticktext=["Low load", "High load"],
            range=[x_axis_min, x_axis_max],
            zeroline=False,
            showgrid=False,
            showline=False,
            mirror=False,
            tickwidth=0,
            ticklen=0,
            ticks="",
            row=1,
        )

        fig.update_yaxes(
            col=subplot_index,
            title_text=f"{dv_label} Delta" if subplot_index == 1 else None,
            range=[y_axis_min, y_axis_max],
            tickvals=[-8, -6, -4, -2, 0, 2, 4, 6, 8],
            ticktext=["-8", "-6", "-4", "-2", "0", "2", "4", "6", "8"],
            zeroline=False,
            showline=False,
            mirror=False,
            tickwidth=0,
            ticklen=0,
            ticks="",
            row=1,
        )

        getattr(fig.layout, xaxis_name).update(
            scaleanchor=xaxis_ref,
            scaleratio=panel_scaleratio,
        )

    if export_html:
        story_condition_normalized = _normalize_story_condition_input(story_condition)
        story_tag = "pooled" if story_condition_normalized is None else story_condition_normalized
        delta_tag = "_".join(delta_name.lower() for delta_name in selected_delta_types)
        figure_type_tag = initial_view if include_toggle_dropdown else figure_type_normalized

        file_name_figure = f"figure_6_shielding_by_load_{dv_suffix}_{delta_tag}_{figure_type_tag}_{story_tag}"
        _export_plotly_figure_html(
            fig=fig,
            general_settings=general_settings,
            file_name=file_name_figure
        )

    return fig


def plot_trial_order_effects_line_graph(
    general_settings: GeneralSettings,
    dv: str | Any = "blame",
    conditions: Sequence[str] | str | Any | None = None,
    average_late_positions: bool = False,
    story_condition: str | Any = None,
    cognitive_load: str | Any = None,
    only_included_participants: bool = True,
    export_html: bool = True,
    base_hue: int = 220,
    confidence_level: float = 0.95,
    order_analysis_mode: str | Any = "legacy",
) -> "object":
    """
    Create a line chart showing how mean distal-agent ratings vary by vignette position.

    Arguments:
        • dv: str | Any
            - Which dependent variable to plot. Supported: 'blame', 'wrong', 'punishment'.
        • conditions: Sequence[str] | str | Any | None
            - Which vignette conditions to plot.
            - Defaults to all three conditions: CC, CH, DIV.
            - Can also be a subset such as ("CC", "CH") or "CC,CH".
        • average_late_positions: bool
            - If False, show separate points for 1st, 2nd, and 3rd position.
            - If True, collapse 2nd and 3rd into a single "Later (2–3)" group.
        • story_condition: str | Any
            - If None, pool stories.
            - If 'firework' or 'trolley', filter to that story condition.
        • cognitive_load: str | Any
            - If None, pool load.
            - If 'high' or 'low', filter to that load condition.
        • only_included_participants: bool
            - If True, restrict to participants who passed all comprehension checks.
        • export_html: bool
            - If True, export the Plotly figure to an .html file.
        • base_hue: int
            - Starting hue for hsla colors. Subsequent conditions increment hue by +20.
        • confidence_level: float
            - Confidence level for error bars around the means.
        • order_analysis_mode: str | Any
            - If None (default), use the new pairwise CC-vs-CH relative-order analysis.
            - Supported aliases:
                "relative_cc_ch"
                "pairwise"
                "cc_ch"
                "relative"
            - If "legacy", "position", or "old", preserve the original analysis that plots
              mean ratings by absolute vignette position (1st / 2nd / 3rd) across selected conditions.
            - Note: `conditions` and `average_late_positions` only apply in the legacy mode.            

    Returns:
        • plotly.graph_objects.Figure
    """
    def compute_mean_and_confidence_interval(values: pd.Series) -> tuple[float, float, float, int]:
        """
        Compute the mean and a t-based confidence interval for one condition-position cell.

        Arguments:
            • values: pd.Series
                - Numeric values for one cell.

        Returns:
            • tuple[float, float, float, int]
                - mean_value
                - ci_lower
                - ci_upper
                - n
        """
        numeric_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        n_values = int(numeric_values.shape[0])

        if n_values == 0:
            return np.nan, np.nan, np.nan, 0

        mean_value = float(np.mean(numeric_values))

        if n_values == 1:
            return mean_value, mean_value, mean_value, 1

        standard_deviation = float(np.std(numeric_values, ddof=1))
        standard_error = standard_deviation / np.sqrt(n_values)

        alpha_value = 1 - confidence_level
        t_critical_value = stats.t.ppf(1 - alpha_value / 2, df=n_values - 1)

        ci_lower = float(mean_value - t_critical_value * standard_error)
        ci_upper = float(mean_value + t_critical_value * standard_error)

        return mean_value, ci_lower, ci_upper, n_values

    dv_suffix, dv_label, bounded_y_range = _normalize_dependent_variable_input(dv)

    order_analysis_mode_normalized = (
        "relative_cc_ch"
        if order_analysis_mode is None
        else str(order_analysis_mode).strip().lower()
    )

    if order_analysis_mode_normalized in {"relative_cc_ch", "pairwise", "cc_ch", "relative", "new"}:
        analysis_dataframe = _get_filtered_plotting_dataframe(
            general_settings=general_settings,
            story_condition=story_condition,
            cognitive_load=cognitive_load,
            only_included_participants=only_included_participants,
        )

        cc_value_column_name = f"clark_{dv_suffix}_cc"
        ch_value_column_name = f"clark_{dv_suffix}_ch"

        required_columns = [
            "response_id",
            "case_code_position_1",
            "case_code_position_2",
            "case_code_position_3",
            cc_value_column_name,
            ch_value_column_name,
        ]
        missing_required_columns = [
            column_name for column_name in required_columns if column_name not in analysis_dataframe.columns
        ]
        if missing_required_columns:
            raise KeyError(
                "Missing one or more expected columns required for the relative-order analysis: "
                + ", ".join(repr(column_name) for column_name in missing_required_columns)
            )

        pairwise_order_dataframe = analysis_dataframe[
            [
                "response_id",
                "case_code_position_1",
                "case_code_position_2",
                "case_code_position_3",
                cc_value_column_name,
                ch_value_column_name,
            ]
        ].copy()

        pairwise_order_dataframe["cc_position"] = np.select(
            condlist=[
                pairwise_order_dataframe["case_code_position_1"] == "CC",
                pairwise_order_dataframe["case_code_position_2"] == "CC",
                pairwise_order_dataframe["case_code_position_3"] == "CC",
            ],
            choicelist=[1, 2, 3],
            default=np.nan,
        )

        pairwise_order_dataframe["ch_position"] = np.select(
            condlist=[
                pairwise_order_dataframe["case_code_position_1"] == "CH",
                pairwise_order_dataframe["case_code_position_2"] == "CH",
                pairwise_order_dataframe["case_code_position_3"] == "CH",
            ],
            choicelist=[1, 2, 3],
            default=np.nan,
        )

        pairwise_order_dataframe["cc_rating"] = pd.to_numeric(
            pairwise_order_dataframe[cc_value_column_name],
            errors="coerce",
        )
        pairwise_order_dataframe["ch_rating"] = pd.to_numeric(
            pairwise_order_dataframe[ch_value_column_name],
            errors="coerce",
        )

        pairwise_order_dataframe = pairwise_order_dataframe.dropna(
            subset=["cc_position", "ch_position", "cc_rating", "ch_rating"]
        ).copy()

        if pairwise_order_dataframe.shape[0] == 0:
            raise ValueError(
                "No valid CC/CH paired observations remain after filtering for the relative-order analysis."
            )

        pairwise_order_dataframe["relative_order_group"] = np.where(
            pairwise_order_dataframe["cc_position"] < pairwise_order_dataframe["ch_position"],
            "CC before CH",
            "CH before CC",
        )

        cc_long_dataframe = pd.DataFrame(
            {
                "response_id": pairwise_order_dataframe["response_id"],
                "condition_code": "CC",
                "condition_label": "Choice-Choice",
                "relative_position_label": np.where(
                    pairwise_order_dataframe["cc_position"] < pairwise_order_dataframe["ch_position"],
                    "Presented first",
                    "Presented second",
                ),
                "relative_order_group": pairwise_order_dataframe["relative_order_group"],
                "rating_value": pairwise_order_dataframe["cc_rating"],
            }
        )

        ch_long_dataframe = pd.DataFrame(
            {
                "response_id": pairwise_order_dataframe["response_id"],
                "condition_code": "CH",
                "condition_label": "Choice-Chance",
                "relative_position_label": np.where(
                    pairwise_order_dataframe["ch_position"] < pairwise_order_dataframe["cc_position"],
                    "Presented first",
                    "Presented second",
                ),
                "relative_order_group": pairwise_order_dataframe["relative_order_group"],
                "rating_value": pairwise_order_dataframe["ch_rating"],
            }
        )

        relative_order_long_dataframe = pd.concat(
            [cc_long_dataframe, ch_long_dataframe],
            ignore_index=True,
        )

        ordered_relative_position_labels = ["Presented first", "Presented second"]
        relative_position_to_x_value = {
            "Presented first": 1.0,
            "Presented second": 2.0,
        }

        summary_rows: list[dict[str, Any]] = []

        for selected_condition_code, selected_condition_label in [
            ("CC", "Choice-Choice"),
            ("CH", "Choice-Chance"),
        ]:
            condition_subset = relative_order_long_dataframe.loc[
                relative_order_long_dataframe["condition_code"] == selected_condition_code
            ].copy()

            for ordered_relative_position_label in ordered_relative_position_labels:
                position_subset = condition_subset.loc[
                    condition_subset["relative_position_label"] == ordered_relative_position_label,
                    "rating_value",
                ]

                mean_value, ci_lower, ci_upper, n_values = compute_mean_and_confidence_interval(position_subset)

                summary_rows.append(
                    {
                        "condition_code": selected_condition_code,
                        "condition_label": selected_condition_label,
                        "relative_position_label": ordered_relative_position_label,
                        "position_x_value": relative_position_to_x_value[ordered_relative_position_label],
                        "mean_value": mean_value,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "error_plus": ci_upper - mean_value if not np.isnan(ci_upper) else np.nan,
                        "error_minus": mean_value - ci_lower if not np.isnan(ci_lower) else np.nan,
                        "n": n_values,
                    }
                )

        summary_dataframe = pd.DataFrame(summary_rows)

        fig = go.Figure()

        plotted_condition_order = [
            ("CC", "Choice-Choice"),
            ("CH", "Choice-Chance"),
        ]

        for condition_index, (selected_condition_code, selected_condition_label) in enumerate(plotted_condition_order):
            line_color = _hsla_color(hue=base_hue + 20 * condition_index, alpha=1.0)

            condition_summary = (
                summary_dataframe.loc[summary_dataframe["condition_code"] == selected_condition_code]
                .set_index("relative_position_label")
                .loc[ordered_relative_position_labels]
                .reset_index()
            )

            fig.add_trace(
                go.Scatter(
                    x=condition_summary["position_x_value"],
                    y=condition_summary["mean_value"],
                    mode="lines+markers",
                    name=selected_condition_label,
                    line=dict(color=line_color, width=8),
                    marker=dict(color=line_color, size=18),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=condition_summary["error_plus"],
                        arrayminus=condition_summary["error_minus"],
                        visible=True,
                        color=line_color,
                        thickness=4,
                        width=6,
                    ),
                    customdata=np.stack(
                        [
                            condition_summary["relative_position_label"],
                            condition_summary["n"],
                            condition_summary["ci_lower"],
                            condition_summary["ci_upper"],
                        ],
                        axis=1,
                    ),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Relative order: %{customdata[0]}<br>"
                        f"Mean {dv_label.lower()}: "
                        "%{y:.2f}<br>"
                        "95% CI: [%{customdata[2]:.2f}, %{customdata[3]:.2f}]<br>"
                        "n: %{customdata[1]}<extra></extra>"
                    ),
                )
            )

        title_prefix = {
            "blame": "Distal Blame",
            "wrong": "Distal Wrongness",
            "punish": "Distal Punishment",
        }[dv_suffix]

        fig.update_layout(**figure_layout, title=f"{title_prefix} by Relative CC-CH Order")
        fig.update_layout(
            title_x=0.5,
            title_xanchor="center",
            title_font_size=44,
            margin=dict(l=160, r=160, t=100, b=100),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.90,
                xanchor="center",
                x=0.5,
                bgcolor="hsla(0, 0%, 100%, 0.0)",
            ),
        )

        x_tick_values = [relative_position_to_x_value[position_label] for position_label in ordered_relative_position_labels]

        fig.update_xaxes(
            tickmode="array",
            tickvals=x_tick_values,
            ticktext=ordered_relative_position_labels,
            title_text="Relative Order within the CC-CH Pair",
            range=[min(x_tick_values) - 0.2, max(x_tick_values) + 0.2],
            zeroline=False,
            showline=False,
            mirror=False,
            tickwidth=0,
            ticklen=0,
            ticks="",
        )

        if bounded_y_range is None:
            plotted_values = summary_dataframe["mean_value"].dropna().to_numpy(dtype=float)
            bounded_y_range = [0.0, max(1.0, float(np.nanmax(plotted_values) * 1.10))] if plotted_values.shape[0] > 0 else [0.0, 1.0]

        y_axis_title = {
            "blame": "Mean Blameworthiness of Distal Agent",
            "wrong": "Mean Wrongness of Distal Agent",
            "punish": "Mean Punishment of Distal Agent (years)",
        }[dv_suffix]

        fig.update_yaxes(
            title_text=y_axis_title,
            range=list(bounded_y_range),
            zeroline=False,
            showline=False,
            mirror=False,
            tickwidth=0,
            ticklen=0,
            ticks="",
        )

        if dv_suffix in {"blame", "wrong"}:
            fig.update_yaxes(
                tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                ticktext=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            )

        if export_html:
            story_condition_normalized = _normalize_story_condition_input(story_condition)
            cognitive_load_normalized = _normalize_load_condition_input(cognitive_load)
            story_tag = "pooled" if story_condition_normalized is None else story_condition_normalized
            load_tag = "pooled" if cognitive_load_normalized is None else cognitive_load_normalized

            file_name_figure = f"figure_7_trial_order_relative_cc_ch_{dv_suffix}_{story_tag}_{load_tag}"
            if export_html:
                _export_plotly_figure_html(
                    fig=fig,
                    general_settings=general_settings,
                    file_name=file_name_figure
                )

        return fig

    selected_condition_codes = _normalize_condition_subset_input(conditions)

    analysis_dataframe = _get_filtered_plotting_dataframe(
        general_settings=general_settings,
        story_condition=story_condition,
        cognitive_load=cognitive_load,
        only_included_participants=only_included_participants,
    )

    condition_code_to_value_column = {
        "CC": f"clark_{dv_suffix}_cc",
        "CH": f"clark_{dv_suffix}_ch",
        "DIV": f"clark_{dv_suffix}_div",
    }

    condition_code_to_display_label = {
        "CC": "Choice-Choice",
        "CH": "Choice-Chance",
        "DIV": "Division",
    }

    missing_required_columns = [
        column_name
        for column_name in [
            "response_id",
            "case_code_position_1",
            "case_code_position_2",
            "case_code_position_3",
            *[condition_code_to_value_column[condition_code] for condition_code in selected_condition_codes],
        ]
        if column_name not in analysis_dataframe.columns
    ]
    if missing_required_columns:
        raise KeyError(
            "Missing one or more expected columns required for the trial-order figure: "
            + ", ".join(repr(column_name) for column_name in missing_required_columns)
        )

    trial_order_rows: list[dict[str, Any]] = []

    for selected_condition_code in selected_condition_codes:
        value_column_name = condition_code_to_value_column[selected_condition_code]

        temp_dataframe = analysis_dataframe[
            [
                "response_id",
                "case_code_position_1",
                "case_code_position_2",
                "case_code_position_3",
                value_column_name,
            ]
        ].copy()

        temp_dataframe["position_numeric"] = np.select(
            condlist=[
                temp_dataframe["case_code_position_1"] == selected_condition_code,
                temp_dataframe["case_code_position_2"] == selected_condition_code,
                temp_dataframe["case_code_position_3"] == selected_condition_code,
            ],
            choicelist=[1, 2, 3],
            default=np.nan,
        )

        temp_dataframe["rating_value"] = pd.to_numeric(temp_dataframe[value_column_name], errors="coerce")
        temp_dataframe["condition_code"] = selected_condition_code
        temp_dataframe["condition_label"] = condition_code_to_display_label[selected_condition_code]

        trial_order_rows.extend(
            temp_dataframe[
                ["response_id", "condition_code", "condition_label", "position_numeric", "rating_value"]
            ].dropna(subset=["position_numeric", "rating_value"]).to_dict("records")
        )

    trial_order_long_dataframe = pd.DataFrame(trial_order_rows)

    if average_late_positions:
        trial_order_long_dataframe["position_group_label"] = np.where(
            trial_order_long_dataframe["position_numeric"] == 1,
            "First",
            "Later (2–3)",
        )
        ordered_position_labels = ["First", "Later (2–3)"]
        position_label_to_x_value = {"First": 1.0, "Later (2–3)": 2.0}
        x_axis_title = "Vignette Position Group"
    else:
        trial_order_long_dataframe["position_group_label"] = trial_order_long_dataframe["position_numeric"].map(
            {
                1.0: "1st",
                2.0: "2nd",
                3.0: "3rd",
            }
        )
        ordered_position_labels = ["1st", "2nd", "3rd"]
        position_label_to_x_value = {"1st": 1.0, "2nd": 2.0, "3rd": 3.0}
        x_axis_title = "Vignette Position"

    summary_rows: list[dict[str, Any]] = []

    for selected_condition_code in selected_condition_codes:
        selected_condition_label = condition_code_to_display_label[selected_condition_code]
        condition_subset = trial_order_long_dataframe.loc[
            trial_order_long_dataframe["condition_code"] == selected_condition_code
        ].copy()

        for ordered_position_label in ordered_position_labels:
            position_subset = condition_subset.loc[
                condition_subset["position_group_label"] == ordered_position_label,
                "rating_value"
            ]

            mean_value, ci_lower, ci_upper, n_values = compute_mean_and_confidence_interval(position_subset)

            summary_rows.append(
                {
                    "condition_code": selected_condition_code,
                    "condition_label": selected_condition_label,
                    "position_group_label": ordered_position_label,
                    "position_x_value": position_label_to_x_value[ordered_position_label],
                    "mean_value": mean_value,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "error_plus": ci_upper - mean_value if not np.isnan(ci_upper) else np.nan,
                    "error_minus": mean_value - ci_lower if not np.isnan(ci_lower) else np.nan,
                    "n": n_values,
                }
            )

    summary_dataframe = pd.DataFrame(summary_rows)
    fig = go.Figure()

    for condition_index, selected_condition_code in enumerate(selected_condition_codes):
        condition_label = condition_code_to_display_label[selected_condition_code]
        line_color = _hsla_color(hue=base_hue + 20 * condition_index, alpha=1.0)
        error_color = _hsla_color(hue=base_hue + 20 * condition_index, alpha=8.0)

        condition_summary = (
            summary_dataframe.loc[summary_dataframe["condition_code"] == selected_condition_code]
            .set_index("position_group_label")
            .loc[ordered_position_labels]
            .reset_index()
        )

        fig.add_trace(
            go.Scatter(
                x=condition_summary["position_x_value"],
                y=condition_summary["mean_value"],
                mode="lines+markers",
                name=condition_label,
                line=dict(color=line_color, width=8),
                marker=dict(color=line_color, size=18),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=condition_summary["error_plus"],
                    arrayminus=condition_summary["error_minus"],
                    visible=True,
                    color=line_color,
                    thickness=4,
                    width=6,
                ),
                customdata=np.stack(
                    [
                        condition_summary["position_group_label"],
                        condition_summary["n"],
                        condition_summary["ci_lower"],
                        condition_summary["ci_upper"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Position: %{customdata[0]}<br>"
                    f"Mean {dv_label.lower()}: "
                    "%{y:.2f}<br>"
                    "95% CI: [%{customdata[2]:.2f}, %{customdata[3]:.2f}]<br>"
                    "n: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    title_prefix = {
        "blame": "Distal Blame",
        "wrong": "Distal Wrongness",
        "punish": "Distal Punishment",
    }[dv_suffix]

    title_suffix = " by Vignette Position" if not average_late_positions else " by First vs. Later Vignette Position"
    fig.update_layout(**figure_layout, title=f"{title_prefix}{title_suffix}")

    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_font_size=44,
        margin=dict(l=440, r=440, t=100, b=100),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=0.935,
            xanchor="center", x=0.5,
            bgcolor="hsla(0, 0%, 100%, 0.0)",
        ),
    )

    x_tick_values = [position_label_to_x_value[position_label] for position_label in ordered_position_labels]

    fig.update_xaxes(
        tickmode="array",
        tickvals=x_tick_values,
        ticktext=ordered_position_labels,
        title_text=x_axis_title,
        range=[min(x_tick_values) - 0.2, max(x_tick_values) + 0.2],
        scaleanchor="y", scaleratio=6,
        zeroline=False,
        showline=False,
        mirror=False,
        tickwidth=0,
        ticklen=0,
        ticks="",
    )

    if bounded_y_range is None:
        plotted_values = summary_dataframe["mean_value"].dropna().to_numpy(dtype=float)
        bounded_y_range = [0.0, max(1.0, float(np.nanmax(plotted_values) * 1.10))] if plotted_values.shape[0] > 0 else [0.0, 1.0]

    y_axis_title = {
        "blame": "Mean Blameworthiness of Distal Agent",
        "wrong": "Mean Wrongness of Distal Agent",
        "punish": "Mean Punishment of Distal Agent (years)",
    }[dv_suffix]

    fig.update_yaxes(
        title_text=y_axis_title,
        range=list(bounded_y_range),
        zeroline=False,
        showline=False,
        mirror=False,
        tickwidth=0,
        ticklen=0,
        ticks="",
    )

    if dv_suffix in {"blame", "wrong"}:
        fig.update_yaxes(
            tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            ticktext=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        )

    if export_html:
        story_condition_normalized = _normalize_story_condition_input(story_condition)
        cognitive_load_normalized = _normalize_load_condition_input(cognitive_load)
        story_tag = "pooled" if story_condition_normalized is None else story_condition_normalized
        load_tag = "pooled" if cognitive_load_normalized is None else cognitive_load_normalized
        positions_tag = "first_vs_later" if average_late_positions else "all_positions"
        conditions_tag = "_".join(condition_code.lower() for condition_code in selected_condition_codes)

        file_name_figure = f"figure_7_trial_order_{dv_suffix}_{conditions_tag}_{story_tag}_{load_tag}_{positions_tag}"
        _export_plotly_figure_html(
            fig=fig,
            general_settings=general_settings,
            file_name=file_name_figure
        )

    return fig


def plot_blameworthiness_wrongness_correlate(
    general_settings: GeneralSettings,
    aggregation_level: str | Any = "participant_mean",
    condition: str | Any = None,
    story_condition: str | Any = None,
    cognitive_load: str | Any = None,
    only_included_participants: bool = True,
    export_html: bool = True,
    base_hue: int = 260,
    all_ratings: bool = False,
    include_proximate_agent: bool = True,
    jitter_strength: float = 0.15,
) -> "object":
    """
    Plot the relationship between blame and wrongness.

    Arguments:
        • aggregation_level: str | Any
            - If 'participant_mean', average each participant across the three distal conditions.
            - If 'condition_rows', create one row per participant × condition (CC, CH, DIV).
            - Only used when all_ratings=False.
        • condition: str | Any
            - Optional condition restriction. Supports: 'CC', 'CH', 'DIV', or None.
            - Only used when aggregation_level='condition_rows' and all_ratings=False.
        • story_condition: str | Any
            - If None, pool stories.
            - If 'firework' or 'trolley', filter to that story condition.
        • cognitive_load: str | Any
            - If None, pool load conditions.
            - If 'high' or 'low', filter to that load condition.
        • only_included_participants: bool
            - If True, restrict to participants who passed all comprehension checks.
        • export_html: bool
            - If True, export the Plotly figure to an .html file.
        • base_hue: int
            - Starting hue for hsla colors.
        • all_ratings: bool
            - If True, plot all individual rating pairs across vignettes with jitter instead of
              aggregated participant means or condition rows. Ignores aggregation_level and condition.
        • include_proximate_agent: bool
            - If True, include CC Proximate and CH Proximate rating pairs.
            - Only used when all_ratings=True.
        • jitter_strength: float
            - Standard deviation of Gaussian jitter applied to x and y when all_ratings=True.
              Jitter is purely aesthetic and does not affect the correlation or trendline.

    Returns:
        • plotly.graph_objects.Figure
    """
    analysis_dataframe = _get_filtered_plotting_dataframe(
        general_settings=general_settings,
        story_condition=story_condition,
        cognitive_load=cognitive_load,
        only_included_participants=only_included_participants,
    )

    if all_ratings:
        ordered_rating_pairs = [
            ("CC Distal", "clark_blame_cc", "clark_wrong_cc"),
            ("CH Distal", "clark_blame_ch", "clark_wrong_ch"),
            ("DIV Distal", "clark_blame_div", "clark_wrong_div"),
        ]
        if include_proximate_agent:
            ordered_rating_pairs += [
                ("CC Proximate", "proximate_blame_cc", "proximate_wrong_cc"),
                ("CH Proximate", "proximate_blame_ch", "proximate_wrong_ch"),
            ]

        missing_required_columns = [
            column_name
            for _, blame_column_name, wrong_column_name in ordered_rating_pairs
            for column_name in [blame_column_name, wrong_column_name]
            if column_name not in analysis_dataframe.columns
        ]
        if missing_required_columns:
            raise KeyError(
                "Missing one or more expected columns required for the all-ratings blame–wrongness plot: "
                + ", ".join(repr(column_name) for column_name in missing_required_columns)
            )

        pooled_rows: list[dict[str, Any]] = []
        for series_label, blame_column_name, wrong_column_name in ordered_rating_pairs:
            temp_dataframe = pd.DataFrame(
                {
                    "series_label": series_label,
                    "blame_value": pd.to_numeric(analysis_dataframe[blame_column_name], errors="coerce"),
                    "wrong_value": pd.to_numeric(analysis_dataframe[wrong_column_name], errors="coerce"),
                }
            ).dropna(subset=["blame_value", "wrong_value"])
            pooled_rows.extend(temp_dataframe.to_dict("records"))

        plotting_dataframe = pd.DataFrame(pooled_rows)

        figure_title = "Blame vs. Wrongness Across All Ratings"
        if include_proximate_agent:
            figure_title += " (Distal + Proximate)"
        else:
            figure_title += " (Distal Only)"

    else:
        aggregation_level_normalized = str(aggregation_level).strip().lower()

        if aggregation_level_normalized == "participant_mean":
            plotting_dataframe = pd.DataFrame(
                {
                    "blame_value": analysis_dataframe[
                        ["clark_blame_cc", "clark_blame_ch", "clark_blame_div"]
                    ].mean(axis=1, skipna=True),
                    "wrong_value": analysis_dataframe[
                        ["clark_wrong_cc", "clark_wrong_ch", "clark_wrong_div"]
                    ].mean(axis=1, skipna=True),
                }
            ).dropna()

            figure_title = "Blame vs. Wrongness (Participant Means)"
        else:
            selected_condition_codes = _normalize_condition_subset_input(condition)

            condition_rows = []
            for selected_condition_code in selected_condition_codes:
                blame_column_name = f"clark_blame_{selected_condition_code.lower()}"
                wrong_column_name = f"clark_wrong_{selected_condition_code.lower()}"
                condition_label = {
                    "CC": "Choice-Choice",
                    "CH": "Choice-Chance",
                    "DIV": "Division",
                }[selected_condition_code]

                temp_dataframe = pd.DataFrame(
                    {
                        "condition_label": condition_label,
                        "blame_value": pd.to_numeric(analysis_dataframe[blame_column_name], errors="coerce"),
                        "wrong_value": pd.to_numeric(analysis_dataframe[wrong_column_name], errors="coerce"),
                    }
                )
                condition_rows.append(temp_dataframe)

            plotting_dataframe = pd.concat(condition_rows, ignore_index=True).dropna()
            figure_title = "Blame vs. Wrongness (Participant × Condition Rows)"

    if plotting_dataframe.shape[0] < 5:
        raise ValueError("Not enough valid observations remain to create the blame–wrongness plot.")

    "Compute correlation and trendline on clean data before any jitter is applied"
    slope_value, intercept_value, pearson_r_value, pearson_p_value, _ = stats.linregress(
        plotting_dataframe["blame_value"],
        plotting_dataframe["wrong_value"],
    )

    x_grid = np.linspace(
        float(plotting_dataframe["blame_value"].min()),
        float(plotting_dataframe["blame_value"].max()),
        200,
    )
    fitted_y_values = intercept_value + slope_value * x_grid

    "Apply jitter to plot coordinates only, after statistics are computed"
    if jitter_strength > 0:
        rng = np.random.default_rng(seed=42)
        plot_blame = plotting_dataframe["blame_value"] + rng.normal(0, jitter_strength, size=len(plotting_dataframe))
        plot_wrong = plotting_dataframe["wrong_value"] + rng.normal(0, jitter_strength, size=len(plotting_dataframe))
    else:
        plot_blame = plotting_dataframe["blame_value"]
        plot_wrong = plotting_dataframe["wrong_value"]

    point_color = _hsla_color(hue=base_hue - 20, alpha=0.45)
    line_color = _hsla_color(hue=base_hue, alpha=1.0)

    fig = go.Figure()

    if all_ratings:
        fig.add_trace(
            go.Scatter(
                x=plot_blame,
                y=plot_wrong,
                mode="markers",
                marker=dict(color=point_color, size=8),
                customdata=np.stack([plotting_dataframe["series_label"]], axis=1),
                showlegend=False,
                hovertemplate=(
                    "Series: %{customdata[0]}<br>"
                    "Blame: %{x:.2f}<br>"
                    "Wrongness: %{y:.2f}<extra></extra>"
                ),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=plot_blame,
                y=plot_wrong,
                mode="markers",
                marker=dict(color=point_color, size=10),
                showlegend=False,
                hovertemplate=(
                    "Blame: %{x:.2f}<br>"
                    "Wrongness: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=fitted_y_values,
            mode="lines",
            line=dict(color=line_color, width=8),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(**figure_layout, title=figure_title)
    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_font_size=30 if all_ratings else 40,
        margin=dict(l=585, r=585, t=80, b=90),
    )

    axes_ranges = [0.8, 9.2]
    tickvals = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ticktext = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    fig.update_xaxes(
        title_text="Blameworthiness",
        range=axes_ranges,
        tickvals=tickvals,
        ticktext=ticktext,
        scaleanchor="y",
        scaleratio=1,
        zeroline=False,
        showline=True,
        mirror=True,
        tickwidth=0,
        ticklen=0,
        ticks="",
    )
    fig.update_yaxes(
        title_text="Wrongness",
        range=axes_ranges,
        tickvals=tickvals,
        ticktext=ticktext,
        zeroline=False,
        showline=True,
        mirror=True,
        tickwidth=0,
        ticklen=0,
        ticks="",
    )

    fig.add_annotation(
        x=5, y=1.5, xref='x', yref='y',
        xanchor="center", yanchor="middle",
        align="center", showarrow=False,
        font=dict(size=26, family="Calibri", color="black"),
        bgcolor="hsla(0, 0%, 100%, 0.75)",
        text=f"𝑟 = {pearson_r_value:+.2f}<br>𝑝 = {pearson_p_value:.3g}",
    )

    if export_html:
        story_condition_normalized = _normalize_story_condition_input(story_condition)
        cognitive_load_normalized = _normalize_load_condition_input(cognitive_load)
        story_tag = "pooled" if story_condition_normalized is None else story_condition_normalized
        load_tag = "pooled" if cognitive_load_normalized is None else cognitive_load_normalized

        if all_ratings:
            proximate_tag = "with_proximate" if include_proximate_agent else "distal_only"
            file_name_figure = f"figure_8_blame_wrongness_all_ratings_{story_tag}_{load_tag}_{proximate_tag}"
        else:
            file_name_figure = f"figure_8_blame_wrongness_{aggregation_level_normalized}_{story_tag}_{load_tag}"

        _export_plotly_figure_html(
            fig=fig,
            general_settings=general_settings,
            file_name=file_name_figure
        )

    return fig


def plot_triangulation_2afc_vs_rating_delta(
    general_settings: GeneralSettings,
    comparison: str | Any = "CH_CC",
    dv: str | Any = "blame",
    story_condition: str | Any = None,
    cognitive_load: str | Any = None,
    only_included_participants: bool = True,
    export_html: bool = True,
    base_hue: int = 220,
) -> "object":
    """
    Plot 2AFC endorsement against the corresponding participant-level Likert delta with a logistic fit.

    Arguments:
        • comparison: str | Any
            - Which comparison to plot. Supported:
                "CH_CC"
                "DIV_CC"
        • dv: str | Any
            - Which dependent variable to use for the numeric delta. Supported:
                'blame', 'wrong', 'punishment'
            - Note: the current 2AFC columns correspond most directly to blame comparisons.
        • story_condition: str | Any
            - If None, pool stories.
            - If 'firework' or 'trolley', filter to that story condition.
        • cognitive_load: str | Any
            - If None, pool load conditions.
            - If 'high' or 'low', filter to that load condition.
        • only_included_participants: bool
            - If True, restrict to participants who passed all comprehension checks.
        • export_html: bool
            - If True, export the Plotly figure to an .html file.
        • base_hue: int
            - Starting hue for hsla colors.

    Returns:
        • plotly.graph_objects.Figure
    """
    dv_suffix, dv_label, _ = _normalize_dependent_variable_input(dv)

    comparison_normalized = str(comparison).strip().upper()
    if comparison_normalized not in {"CH_CC", "DIV_CC"}:
        raise ValueError(
            "comparison must be one of: 'CH_CC' or 'DIV_CC'. "
            f"Got: {comparison!r}"
        )

    analysis_dataframe = _get_filtered_plotting_dataframe(
        general_settings=general_settings,
        story_condition=story_condition,
        cognitive_load=cognitive_load,
        only_included_participants=only_included_participants,
    )

    if comparison_normalized == "CH_CC":
        twoafc_column_name = "twoafc_ch_vs_cc"
        delta_column_name = f"clark_{dv_suffix}_ch_minus_cc"
        left_prefix = "CH"
        right_prefix = "CC"
        comparison_label = "CH vs. CC"
    else:
        twoafc_column_name = "twoafc_div_vs_cc"
        delta_column_name = f"clark_{dv_suffix}_div_minus_cc"
        left_prefix = "DIV"
        right_prefix = "CC"
        comparison_label = "DIV vs. CC"

    if twoafc_column_name not in analysis_dataframe.columns:
        raise KeyError(f"Expected 2AFC column {twoafc_column_name!r} but it was not found.")
    if delta_column_name not in analysis_dataframe.columns:
        raise KeyError(f"Expected delta column {delta_column_name!r} but it was not found.")

    def twoafc_to_binary_pro_left(twoafc_string_value: str) -> float:
        if pd.isna(twoafc_string_value):
            return np.nan

        value_string = str(twoafc_string_value)

        if value_string in {f"{left_prefix} > {right_prefix}", f"{left_prefix} ≥ {right_prefix}"}:
            return 1.0
        if value_string in {f"{left_prefix} < {right_prefix}", f"{left_prefix} ≤ {right_prefix}"}:
            return 0.0

        return np.nan

    triangulation_dataframe = analysis_dataframe[[twoafc_column_name, delta_column_name]].copy()
    triangulation_dataframe["delta_value"] = pd.to_numeric(triangulation_dataframe[delta_column_name], errors="coerce")
    triangulation_dataframe["binary_twoafc"] = triangulation_dataframe[twoafc_column_name].apply(twoafc_to_binary_pro_left)
    triangulation_dataframe = triangulation_dataframe.dropna(subset=["delta_value", "binary_twoafc"]).copy()

    if triangulation_dataframe.shape[0] < 10:
        raise ValueError(
            "Not enough valid observations remain to fit the logistic triangulation plot."
        )

    logistic_model = smf.glm(
        formula="binary_twoafc ~ delta_value",
        data=triangulation_dataframe,
        family=sm.families.Binomial(),
    ).fit()

    x_grid = np.linspace(
        float(triangulation_dataframe["delta_value"].min()),
        float(triangulation_dataframe["delta_value"].max()),
        200,
    )
    prediction_dataframe = pd.DataFrame({"delta_value": x_grid})
    predicted_probabilities = logistic_model.predict(prediction_dataframe)

    jitter_rng = np.random.default_rng(seed=1337)
    jittered_binary_values = triangulation_dataframe["binary_twoafc"] + jitter_rng.normal(loc=0.0, scale=0.035, size=triangulation_dataframe.shape[0])
    jittered_binary_values = np.clip(jittered_binary_values, -0.05, 1.05)

    point_color = _hsla_color(hue=base_hue, alpha=0.45)
    line_color = _hsla_color(hue=base_hue + 20, alpha=1.0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=triangulation_dataframe["delta_value"],
            y=jittered_binary_values,
            mode="markers",
            name="Participants",
            marker=dict(color=point_color, size=8),
            customdata=np.stack(
                [
                    triangulation_dataframe["binary_twoafc"],
                ],
                axis=1,
            ),
            hovertemplate=(
                "Delta: %{x:.2f}<br>"
                "2AFC endorsement (unjittered): %{customdata[0]:.0f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=predicted_probabilities,
            mode="lines",
            name="Logistic fit",
            line=dict(color=line_color, width=5),
            showlegend=False,
        )
    )

    fig.update_layout(**figure_layout, title=f"Triangulation: 2AFC vs. {dv_label} Delta ({comparison_label})")
    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_font_size=40,
        margin=dict(l=90, r=90, t=90, b=90),
    )

    fig.update_xaxes(
        title_text=f"{comparison_label} {dv_label} Delta",
        ticks="",
        ticklen=0,
        tickwidth=0,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text=f"P(Endorse {left_prefix} ≥ {right_prefix})",
        range=[-0.05, 1.05],
        tickvals=[0.0, 0.5, 1.0],
        ticktext=["0", "0.5", "1"],
        ticks="",
        ticklen=0,
        tickwidth=0,
        zeroline=False,
    )

    fig.add_hline(
        y=0.0,
        line=dict(color="hsla(0, 0%, 0%, 0.35)", width=1.5, dash="dash"),
        layer="below",
    )
    fig.add_hline(
        y=1.0,
        line=dict(color="hsla(0, 0%, 0%, 0.35)", width=1.5, dash="dash"),
        layer="below",
    )
    fig.add_vline(
        x=0.0,
        line=dict(color="hsla(0, 0%, 0%, 0.35)", width=1.5, dash="dash"),
        layer="below",
    )

    if export_html:
        story_condition_normalized = _normalize_story_condition_input(story_condition)
        cognitive_load_normalized = _normalize_load_condition_input(cognitive_load)
        story_tag = "pooled" if story_condition_normalized is None else story_condition_normalized
        load_tag = "pooled" if cognitive_load_normalized is None else cognitive_load_normalized

        file_name_figure = f"figure_x_triangulation_{comparison_normalized.lower()}_{dv_suffix}_{story_tag}_{load_tag}"
        _export_plotly_figure_html(
            fig=fig,
            general_settings=general_settings,
            file_name=file_name_figure
        )

    return fig


def plot_shielding_by_individual_difference(
    general_settings: GeneralSettings,
    predictor: str | Any = "crt",
    dv: str | Any = "blame",
    delta_type: str | Sequence[str] | Any | None = "CH_CC",
    figure_type: str | Any = None,
    individualism_dimension: str | Any = "overall",
    binning: str | Any = "quartile",
    story_condition: str | Any = None,
    cognitive_load: str | Any = None,
    only_included_participants: bool = True,
    export_html: bool = True,
    base_hue: int = 240,
) -> "object":
    """
    Plot shielding deltas as a function of an individual-difference predictor (CRT or individualism),
    with either scatter+trendline or box/violin distributions.

    Arguments:
        • predictor: str | Any
            - Which individual-difference predictor to use. Supported: 'crt', 'indcol'.
        • dv: str | Any
            - Which dependent variable to use. Supported: 'blame', 'wrong', 'punishment'.
        • delta_type: str | Sequence[str] | Any | None
            - Which delta(s) to include. Supports:
                "CH_CC"
                "DIV_CC"
                "both"
        • figure_type: str | Any
            - If 'box' or 'violin', show distributions binned by predictor value.
            - If 'scatter', show scatterplots with trendlines.
            - Otherwise, defaults to 'scatter'.
        • individualism_dimension: str | Any
            - Which individualism score to use. Only used when predictor='indcol'.
                "overall"
                "horizontal"
                "vertical"
        • binning: str | Any
            - If using box/violin mode with predictor='indcol', currently supports:
                "quartile"
            - Ignored when predictor='crt' (CRT scores 0–3 are used directly as categories).
        • story_condition: str | Any
            - If None, pool stories.
            - If 'firework' or 'trolley', filter to that story condition.
        • cognitive_load: str | Any
            - If None, pool load conditions.
            - If 'high' or 'low', filter to that load condition.
        • only_included_participants: bool
            - If True, restrict to participants who passed all comprehension checks.
        • export_html: bool
            - If True, export the Plotly figure to an .html file.
        • base_hue: int
            - Starting hue for hsla colors.

    Returns:
        • plotly.graph_objects.Figure
    """
    dv_suffix, dv_label, _ = _normalize_dependent_variable_input(dv)
    selected_delta_types = _normalize_delta_type_input(delta_type)
    figure_type_normalized = "scatter" if figure_type is None else str(figure_type).strip().lower()
    predictor_normalized = str(predictor).strip().lower()

    "Resolve predictor column name and axis label"
    if predictor_normalized == "crt":
        predictor_column_name = "crt_score"
        predictor_axis_label = "CRT Score"
        predictor_file_tag = "crt"
    elif predictor_normalized == "indcol":
        individualism_dimension_normalized = str(individualism_dimension).strip().lower()
        if individualism_dimension_normalized == "overall":
            predictor_column_name = "individualism_score"
            predictor_axis_label = "Overall Individualism"
        elif individualism_dimension_normalized == "horizontal":
            predictor_column_name = "individualism_horizontal"
            predictor_axis_label = "Horizontal Individualism"
        elif individualism_dimension_normalized == "vertical":
            predictor_column_name = "individualism_vertical"
            predictor_axis_label = "Vertical Individualism"
        else:
            raise ValueError(
                "individualism_dimension must be one of: 'overall', 'horizontal', 'vertical'. "
                f"Got: {individualism_dimension!r}"
            )
        predictor_file_tag = f"indcol_{individualism_dimension_normalized}"
    else:
        raise ValueError(
            "predictor must be one of: 'crt', 'indcol'. "
            f"Got: {predictor!r}"
        )

    analysis_dataframe = _get_filtered_plotting_dataframe(
        general_settings=general_settings,
        story_condition=story_condition,
        cognitive_load=cognitive_load,
        only_included_participants=only_included_participants,
    )

    delta_long_dataframe = _build_delta_long_dataframe(
        analysis_dataframe=analysis_dataframe,
        dv_suffix=dv_suffix,
        delta_types=selected_delta_types,
    )

    if predictor_column_name not in delta_long_dataframe.columns:
        raise KeyError(
            f"Expected predictor column {predictor_column_name!r} but it was not found in the cleaned dataframe."
        )

    delta_long_dataframe[predictor_column_name] = pd.to_numeric(
        delta_long_dataframe[predictor_column_name], errors="coerce"
    )
    delta_long_dataframe = delta_long_dataframe.dropna(
        subset=[predictor_column_name, "delta_value"]
    ).copy()

    n_columns = len(selected_delta_types)
    fig = make_subplots(
        rows=1,
        cols=n_columns,
        subplot_titles=[
            _get_delta_metadata(dv_suffix=dv_suffix)[selected_delta_type]["label"]
            for selected_delta_type in selected_delta_types
        ],
        horizontal_spacing=0.10,
    )

    initial_subplot_title_count = len(fig.layout.annotations)
    for annotation_index in range(initial_subplot_title_count):
        fig.layout.annotations[annotation_index].update(
            font=dict(size=30, family="Calibri", color="black"),
            xanchor="center",
        )

    for subplot_index, selected_delta_type in enumerate(selected_delta_types, start=1):
        delta_subset = delta_long_dataframe.loc[
            delta_long_dataframe["delta_type"] == selected_delta_type
        ].copy()

        color_value = _hsla_color(hue=base_hue + 20 * (subplot_index - 1), alpha=0.70)
        line_color = _hsla_color(hue=base_hue + 20 * (subplot_index - 1) + 8, alpha=1.00)

        if figure_type_normalized in {"box", "violin"}:
            "Resolve x-axis categories: CRT uses integer scores directly; INDCOL uses quartile bins"
            if predictor_normalized == "crt":
                delta_subset["x_category"] = delta_subset[predictor_column_name].apply(
                    lambda v: str(int(v))
                )
                x_category_order = ["0", "1", "2", "3"]
                x_axis_title = predictor_axis_label
            else:
                if str(binning).strip().lower() != "quartile":
                    raise ValueError("Currently only quartile binning is supported for box/violin INDCOL plots.")
                delta_subset["x_category"] = pd.qcut(
                    delta_subset[predictor_column_name],
                    q=4,
                    labels=["Q1", "Q2", "Q3", "Q4"],
                    duplicates="drop",
                ).astype(str)
                x_category_order = ["Q1", "Q2", "Q3", "Q4"]
                x_axis_title = f"{predictor_axis_label} Quartile"

            for x_category_value in x_category_order:
                category_subset = delta_subset.loc[
                    delta_subset["x_category"] == x_category_value,
                    "delta_value"
                ]

                if figure_type_normalized == "box":
                    fig.add_trace(
                        go.Box(
                            x=[x_category_value] * len(category_subset),
                            y=category_subset,
                            boxpoints="all",
                            jitter=0.45,
                            pointpos=0,
                            boxmean=True,
                            fillcolor=color_value,
                            line=dict(color=line_color, width=4),
                            marker=dict(color=color_value, size=default_marker_size),
                            showlegend=False,
                        ),
                        row=1,
                        col=subplot_index,
                    )
                else:
                    fig.add_trace(
                        go.Violin(
                            x=[x_category_value] * len(category_subset),
                            y=category_subset,
                            points="all",
                            jitter=0.45,
                            pointpos=0,
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor=color_value,
                            line=dict(color=line_color, width=4),
                            marker=dict(color=color_value, size=default_marker_size),
                            showlegend=False,
                        ),
                        row=1,
                        col=subplot_index,
                    )

            fig.update_xaxes(
                row=1,
                col=subplot_index,
                title_text=x_axis_title,
                categoryorder="array",
                categoryarray=x_category_order,
                ticks="",
                ticklen=0,
                tickwidth=0,
                zeroline=False,
            )

        else:
            "Compute correlation and trendline on clean data before any jitter is applied"
            slope_value, intercept_value, pearson_r_value, pearson_p_value, _ = stats.linregress(
                delta_subset[predictor_column_name],
                delta_subset["delta_value"],
            )

            x_grid = np.linspace(
                float(delta_subset[predictor_column_name].min()),
                float(delta_subset[predictor_column_name].max()),
                200,
            )
            fitted_y_values = intercept_value + slope_value * x_grid

            "Apply jitter to plot coordinates only, after statistics are computed"
            rng = np.random.default_rng(seed=42)
            x_jitter = 0.03 if predictor == "crt" else 0.01
            plot_x = delta_subset[predictor_column_name] + rng.normal(0, x_jitter, size=len(delta_subset))
            plot_y = delta_subset["delta_value"] + rng.normal(0, 0.15, size=len(delta_subset))

            fig.add_trace(
                go.Scatter(
                    x=plot_x,
                    y=plot_y,
                    mode="markers",
                    marker=dict(color=color_value, size=10),
                    showlegend=False,
                    hovertemplate=(
                        f"{predictor_axis_label}: " + "%{x:.2f}<br>"
                        "Delta: %{y:.2f}<extra></extra>"
                    ),
                ),
                row=1,
                col=subplot_index,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=fitted_y_values,
                    mode="lines",
                    line=dict(color=line_color, width=8),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=subplot_index,
            )

            fig.add_annotation(
                row=1,
                col=subplot_index,
                x=0.98,
                y=0.02,
                xref=f"x{subplot_index} domain" if subplot_index > 1 else "x domain",
                yref=f"y{subplot_index} domain" if subplot_index > 1 else "y domain",
                xanchor="right",
                yanchor="bottom",
                align="center",
                showarrow=False,
                font=dict(size=26, family="Calibri", color="black"),
                bgcolor="hsla(0, 0%, 100%, 0.75)",
                text=f"𝑟 = {pearson_r_value:+.2f}<br>𝑝 = {pearson_p_value:.3g}",
            )

            fig.update_xaxes(
                row=1,
                col=subplot_index,
                title_text=predictor_axis_label,
                zeroline=False,
                showline=True,
                mirror=True,
                ticks="",
                ticklen=0,
                tickwidth=0,
                range=[-0.12, 3.12] if predictor == "crt" else [-0.01, 1.01],
                tickvals=[0, 1, 2, 3] if predictor == "crt" else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['0', '1', '2', '3'] if predictor == "crt" else ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
            )

        fig.update_yaxes(
            row=1,
            col=subplot_index,
            title_text=f"{dv_label} Delta" if subplot_index == 1 else None,
            zeroline=False,
            showline=True,
            mirror=True,
            ticks="",
            ticklen=0,
            tickwidth=0,
        )

        fig.add_hline(
            y=0.0,
            row=1,
            col=subplot_index,
            line=dict(color="hsla(0, 0%, 0%, 0.50)", width=2, dash="dash"),
            layer="below",
        )

    fig.update_layout(**figure_layout, title=f"Shielding by {predictor_axis_label} ({dv_label})")
    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_font_size=38 if predictor == "crt" else 30,
        margin=dict(l=585, r=585, t=80, b=90),
        showlegend=False,
        width=1940,
        height=900,
    )

    if export_html:
        story_condition_normalized = _normalize_story_condition_input(story_condition)
        cognitive_load_normalized = _normalize_load_condition_input(cognitive_load)
        story_tag = "pooled" if story_condition_normalized is None else story_condition_normalized
        load_tag = "pooled" if cognitive_load_normalized is None else cognitive_load_normalized
        delta_tag = "_".join(delta_type_value.lower() for delta_type_value in selected_delta_types)

        file_name_figure = (
            f"figure_x_shielding_by_{predictor_file_tag}_{dv_suffix}_{delta_tag}_{figure_type_normalized}_{story_tag}_{load_tag}"
        )
        _export_plotly_figure_html(
            fig=fig,
            general_settings=general_settings,
            file_name=file_name_figure
        )

    return fig


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
    print("\n" + "=" * 110)
    print(table_title)
    print("=" * 110)
    print(dataframe_table.to_string(index=False))


def _extract_exact_test_row_from_test_csv(
    dataframe_tests: pd.DataFrame,
    dv: str,
    group_a: str,
    group_b: str,
    inclusion_filter: str,
    story_condition: str = "all",
    load_condition: str = "all",
    analysis_family: str | None = None,
) -> pd.Series:
    """
    Extract exactly one row from tests.csv.

    Arguments:
        • dataframe_tests: pd.DataFrame
            - Output of run_confirmatory_and_exploratory_tests.
        • dv: str
            - Dependent-variable row label.
        • group_a: str
            - Left-hand group label.
        • group_b: str
            - Right-hand group label.
        • inclusion_filter: str
            - included_only or all_finishers.
        • story_condition: str
            - all, firework, trolley.
        • load_condition: str
            - all, high, low.
        • analysis_family: str | None
            - Optional confirmatory/exploratory family filter.

    Returns:
        • pd.Series
            - The unique matching row.
    """
    row_mask = (
        (dataframe_tests["dv"] == dv)
        & (dataframe_tests["group_a"] == group_a)
        & (dataframe_tests["group_b"] == group_b)
        & (dataframe_tests["inclusion_filter"] == inclusion_filter)
        & (dataframe_tests["story_condition"] == story_condition)
        & (dataframe_tests["load_condition"] == load_condition)
    )

    if analysis_family is not None:
        row_mask = row_mask & (dataframe_tests["analysis_family"] == analysis_family)

    matching_rows = dataframe_tests.loc[row_mask].copy()

    if matching_rows.shape[0] != 1:
        raise Exception(
            f"Expected exactly one row in tests.csv for dv={dv!r}, group_a={group_a!r}, "
            f"group_b={group_b!r}, inclusion_filter={inclusion_filter!r}, "
            f"story_condition={story_condition!r}, load_condition={load_condition!r}, "
            f"analysis_family={analysis_family!r}, but found {matching_rows.shape[0]} rows."
        )

    return matching_rows.iloc[0]


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


def _extract_exact_integrated_contrast_row(
    dataframe_integrated_models: pd.DataFrame,
    analysis_scope: str,
    inclusion_filter: str,
    story_condition: str,
    contrast_label: str,
) -> pd.Series:
    """
    Extract exactly one contrast row from the integrated-model CSV.

    Arguments:
        • dataframe_integrated_models: pd.DataFrame
            - Output of compute_integrated_clark_blame_results.
        • analysis_scope: str
            - between_subjects_first_vignette or within_subjects_repeated_measures.
        • inclusion_filter: str
            - included_only or all_finishers.
        • story_condition: str
            - all, firework, trolley.
        • contrast_label: str
            - CH - CC, DIV - CC, or CH - DIV.

    Returns:
        • pd.Series
            - The unique matching contrast row.
    """
    row_mask = (
        (dataframe_integrated_models["analysis_scope"] == analysis_scope)
        & (dataframe_integrated_models["row_type"] == "contrast")
        & (dataframe_integrated_models["inclusion_filter"] == inclusion_filter)
        & (dataframe_integrated_models["story_condition"] == story_condition)
        & (dataframe_integrated_models["contrast_label"] == contrast_label)
    )

    matching_rows = dataframe_integrated_models.loc[row_mask].copy()

    if matching_rows.shape[0] != 1:
        raise Exception(
            f"Expected exactly one integrated-model contrast row for analysis_scope={analysis_scope!r}, "
            f"inclusion_filter={inclusion_filter!r}, story_condition={story_condition!r}, "
            f"contrast_label={contrast_label!r}, but found {matching_rows.shape[0]} rows."
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
        ("CC Distal", "clark_blame_cc"),
        ("CH Distal", "clark_blame_ch"),
        ("DIV Distal", "clark_blame_div"),
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
            "CC": int((subset_dataframe["case_code_position_1"] == "CC").sum()),
            "CH": int((subset_dataframe["case_code_position_1"] == "CH").sum()),
            "DIV": int((subset_dataframe["case_code_position_1"] == "DIV").sum()),
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


def compute_manuscript_table_2_mean_scale_values_by_dv_and_condition(
    general_settings: GeneralSettings,
    force_rebuild: bool | None = None,
    inclusion_filter: str = "included_only",
    save_pretty_multilevel_version: bool = True,
    include_medians: bool = False,
    include_std: bool = False,
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

    group_summaries_dataframe = group_summaries_dataframe.loc[
        (group_summaries_dataframe["inclusion_filter"] == inclusion_filter)
        & (group_summaries_dataframe["load_condition"] == "all")
    ].copy()

    story_condition_display_map = {
        "all": "Pooled",
        "firework": "Firework",
        "trolley": "Trolley",
    }

    dv_short_map = {
        "blame": "Blame",
        "wrongness": "Wrong",
        "punishment": "Punish",
    }

    row_specifications: list[dict[str, str]] = [
        {
            "dv": "blame",
            "condition_code": "CC",
            "condition_display": "CC Distal",
            "agent_role": "clark",
        },
        {
            "dv": "blame",
            "condition_code": "CH",
            "condition_display": "CH Distal",
            "agent_role": "clark",
        },
        {
            "dv": "blame",
            "condition_code": "DIV",
            "condition_display": "DIV Distal",
            "agent_role": "clark",
        },
        {
            "dv": "blame",
            "condition_code": "CC",
            "condition_display": "CC Proximate",
            "agent_role": "proximate",
        },
        {
            "dv": "blame",
            "condition_code": "CH",
            "condition_display": "CH Proximate",
            "agent_role": "proximate",
        },

        {
            "dv": "wrongness",
            "condition_code": "CC",
            "condition_display": "CC Distal",
            "agent_role": "clark",
        },
        {
            "dv": "wrongness",
            "condition_code": "CH",
            "condition_display": "CH Distal",
            "agent_role": "clark",
        },
        {
            "dv": "wrongness",
            "condition_code": "DIV",
            "condition_display": "DIV Distal",
            "agent_role": "clark",
        },
        {
            "dv": "wrongness",
            "condition_code": "CC",
            "condition_display": "CC Proximate",
            "agent_role": "proximate",
        },
        {
            "dv": "wrongness",
            "condition_code": "CH",
            "condition_display": "CH Proximate",
            "agent_role": "proximate",
        },

        {
            "dv": "punishment",
            "condition_code": "CC",
            "condition_display": "CC Distal",
            "agent_role": "clark",
        },
        {
            "dv": "punishment",
            "condition_code": "CH",
            "condition_display": "CH Distal",
            "agent_role": "clark",
        },
        {
            "dv": "punishment",
            "condition_code": "DIV",
            "condition_display": "DIV Distal",
            "agent_role": "clark",
        },
        {
            "dv": "punishment",
            "condition_code": "CC",
            "condition_display": "CC Proximate",
            "agent_role": "proximate",
        },
        {
            "dv": "punishment",
            "condition_code": "CH",
            "condition_display": "CH Proximate",
            "agent_role": "proximate",
        },
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
        matching_rows = group_summaries_dataframe.loc[
            (group_summaries_dataframe["analysis_scope"] == analysis_scope)
            & (group_summaries_dataframe["story_condition"] == story_condition)
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

    "Build a flat intermediate record per story condition, DV, and condition with all statistics."

    intermediate_rows: list[dict[str, Any]] = []

    for story_condition in ["all", "firework", "trolley"]:
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

            if agent_role == "clark":
                first_vignette_summary_row = extract_summary_row(
                    analysis_scope="between_subjects_first_vignette",
                    story_condition=story_condition,
                    agent_role="clark",
                    dv_value=dv_value,
                    condition_code=condition_code,
                )
                first_vignette_mean = (
                    float(first_vignette_summary_row["mean"])
                    if first_vignette_summary_row is not None
                    else np.nan
                )
                first_vignette_median = (
                    float(first_vignette_summary_row["median"])
                    if first_vignette_summary_row is not None
                    else np.nan
                )
                first_vignette_std = (
                    float(first_vignette_summary_row["std"])
                    if first_vignette_summary_row is not None
                    else np.nan
                )
            else:
                first_vignette_mean = np.nan
                first_vignette_median = np.nan
                first_vignette_std = np.nan

            all_vignettes_mean = (
                float(all_vignettes_summary_row["mean"])
                if all_vignettes_summary_row is not None
                else np.nan
            )
            all_vignettes_median = (
                float(all_vignettes_summary_row["median"])
                if all_vignettes_summary_row is not None
                else np.nan
            )
            all_vignettes_std = (
                float(all_vignettes_summary_row["std"])
                if all_vignettes_summary_row is not None
                else np.nan
            )

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

    "Shape the output table according to include_medians / include_std."

    use_wide_format = not include_medians and not include_std

    if use_wide_format:
        "One row per Story Condition and Condition; DVs spread across columns."

        wide_rows: list[dict[str, Any]] = []

        for story_condition_display, _ in sorted(
            story_condition_sort_order.items(),
            key=lambda item: item[1],
        ):
            for condition_display, _ in sorted(
                condition_sort_order.items(),
                key=lambda item: item[1],
            ):
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
        "One row per Story Condition, Vignette, and Condition; DVs as column groups."

        long_rows: list[dict[str, Any]] = []

        for story_condition_display, _ in sorted(
            story_condition_sort_order.items(),
            key=lambda item: item[1],
        ):
            for vignette_label, mean_key, median_key, std_key in [
                ("First", "first_mean", "first_median", "first_std"),
                ("All", "all_mean", "all_median", "all_std"),
            ]:
                for condition_display, _ in sorted(
                    condition_sort_order.items(),
                    key=lambda item: item[1],
                ):
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
        median_columns = (
            ["Blame Median", "Wrong Median", "Punish Median"]
            if include_medians
            else []
        )
        std_columns = (
            ["Blame Std", "Wrong Std", "Punish Std"]
            if include_std
            else []
        )

        column_order = (
            ["Story Condition", "Vignette", "Condition"]
            + mean_columns
            + median_columns
            + std_columns
        )
        dataframe_table_flat = pd.DataFrame(long_rows)[column_order]

    "Save flat CSV."

    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table_flat,
        general_settings=general_settings,
        table_file_name=table_file_name,
    )

    "Optionally save a pretty multilevel CSV."

    if save_pretty_multilevel_version:
        pretty_dataframe = dataframe_table_flat.copy()

        float_columns = [
            col
            for col in pretty_dataframe.columns
            if col not in {"Story Condition", "Vignette", "Condition"}
        ]
        for col in float_columns:
            pretty_dataframe[col] = pretty_dataframe[col].map(
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
            median_tuples = (
                [("Median", dv) for dv in ["Blame", "Wrong", "Punish"]]
                if include_medians
                else []
            )
            std_tuples = (
                [("Std", dv) for dv in ["Blame", "Wrong", "Punish"]]
                if include_std
                else []
            )

            pretty_dataframe.columns = pd.MultiIndex.from_tuples(
                [
                    ("", "Story Condition"),
                    ("", "Vignette"),
                    ("", "Condition"),
                ]
                + mean_tuples
                + median_tuples
                + std_tuples
            )

        pretty_table_file_path = (
            general_settings["filing"]["file_paths"]["tables"]
            / table_file_name.replace(".csv", "_Pretty.csv")
        )
        pretty_table_file_path.parent.mkdir(parents=True, exist_ok=True)
        pretty_dataframe.to_csv(pretty_table_file_path, index=False, encoding="utf-8-sig")

    return dataframe_table_flat


def compute_manuscript_table_3_primary_clark_blame_contrasts(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Compute manuscript Table 3: primary Clark-blame contrasts.

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

    use_integrated_models = general_settings["misc"]["use_integrated_models"]

    table_names = general_settings["filing"]["table_names"]

    base_table_file_name = table_names["table_3_primary_clark_blame_contrasts"]
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

    dataframe_tests = run_confirmatory_and_exploratory_tests(
        general_settings=general_settings,
        force_rebuild=force_rebuild,
    )

    if use_integrated_models:
        dataframe_integrated_models = compute_integrated_clark_blame_results(
            general_settings=general_settings,
            cleaned_dataframe=cleaned_dataframe,
            force_rebuild=force_rebuild,
        )

    table_rows: list[dict[str, Any]] = []

    contrast_metadata = [
        ("CH", "BCH - BCC", "clark_blame_ch_minus_cc", "CH - CC"),
        ("DIV", "BDIV - BCC", "clark_blame_div_minus_cc", "DIV - CC"),
    ]

    for inclusion_filter_value, inclusion_display_label in [
        ("included_only", "Included"),
        ("all_finishers", "Everyone"),
    ]:
        "Between-subject rows always come from tests.csv"
        for between_group_a_value, contrast_display_label, _, _ in contrast_metadata:
            if inclusion_filter_value == "included_only":
                between_row = _extract_exact_test_row_from_test_csv(
                    dataframe_tests=dataframe_tests,
                    dv="first_vignette_clark_blame",
                    group_a=between_group_a_value,
                    group_b="CC",
                    inclusion_filter="included_only",
                    story_condition="all",
                    load_condition="all",
                    analysis_family="confirmatory",
                )
            else:
                between_row = _extract_exact_test_row_from_test_csv(
                    dataframe_tests=dataframe_tests,
                    dv="first_vignette_clark_blame",
                    group_a=between_group_a_value,
                    group_b="CC",
                    inclusion_filter="all_finishers",
                    story_condition="all",
                    load_condition="all",
                    analysis_family=None,
                )

            reported_p_value = (
                between_row["p_value_holm"]
                if inclusion_filter_value == "included_only" and pd.notna(between_row["p_value_holm"])
                else between_row["p_value"]
            )

            table_rows.append(
                {
                    "Inclusion": inclusion_display_label,
                    "Design": "Between-subj",
                    "Contrast": contrast_display_label,
                    "Estimate": round(float(between_row["mean_difference_a_minus_b"]), 2),
                    "95% CI": _format_ci_for_manuscript_table(between_row["ci95_lower"], between_row["ci95_upper"]),
                    "p-value": _format_p_value_for_manuscript_table(reported_p_value),
                    "p_value_raw": float(between_row["p_value"]),
                    "p_value_holm": (
                        float(between_row["p_value_holm"])
                        if pd.notna(between_row["p_value_holm"])
                        else np.nan
                    ),
                    "p_value_reported": float(reported_p_value),
                    "p_value_correction": (
                        "Holm corrected"
                        if inclusion_filter_value == "included_only" and pd.notna(between_row["p_value_holm"])
                        else "Unadjusted"
                    ),
                    "source_file": "tests.csv",
                    "source_row_note": between_row["notes"],
                }
            )

        "Within-subject rows: either integrated-model rows or direct tests from tests.csv"
        for _, contrast_display_label, within_dv_name, integrated_contrast_label in contrast_metadata:
            if use_integrated_models:
                within_row = _extract_exact_integrated_contrast_row(
                    dataframe_integrated_models=dataframe_integrated_models,
                    analysis_scope="within_subjects_repeated_measures",
                    inclusion_filter=inclusion_filter_value,
                    story_condition="all",
                    contrast_label=integrated_contrast_label,
                )

                estimate_value = float(within_row["estimate"])
                ci95_lower = float(within_row["ci95_lower"])
                ci95_upper = float(within_row["ci95_upper"])
                p_value_value = float(within_row["p_value"])
                source_file_value = "integrated_blame_models.csv"
                source_note_value = within_row["analysis_meaning"]

            else:
                within_row = _extract_exact_test_row_from_test_csv(
                    dataframe_tests=dataframe_tests,
                    dv=within_dv_name,
                    group_a="delta",
                    group_b="0",
                    inclusion_filter=inclusion_filter_value,
                    story_condition="all",
                    load_condition="all",
                    analysis_family=None,
                )

                estimate_value = float(within_row["mean_difference_a_minus_b"])
                ci95_lower = float(within_row["ci95_lower"])
                ci95_upper = float(within_row["ci95_upper"])
                p_value_value = float(within_row["p_value"])
                source_file_value = "tests.csv"
                source_note_value = within_row["notes"]

            table_rows.append(
                {
                    "Inclusion": inclusion_display_label,
                    "Design": "Within-subj",
                    "Contrast": contrast_display_label,
                    "Estimate": round(estimate_value, 2),
                    "95% CI": _format_ci_for_manuscript_table(ci95_lower, ci95_upper),
                    "p-value": _format_p_value_for_manuscript_table(p_value_value),
                    "p_value_raw": p_value_value,
                    "p_value_holm": np.nan,
                    "p_value_reported": p_value_value,
                    "p_value_correction": "Unadjusted",
                    "source_file": source_file_value,
                    "source_row_note": source_note_value,
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
    ).drop(
        columns=["inclusion_sort_order", "design_sort_order", "contrast_sort_order"],
    )

    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table_3,
        general_settings=general_settings,
        table_file_name=table_file_name,
    )

    return dataframe_table_3


def compute_manuscript_table_4_story_specific_clark_blame_contrasts(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Compute manuscript Table 4: story-specific Clark-blame contrasts.

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

    use_integrated_models = general_settings["misc"]["use_integrated_models"]

    table_names = general_settings["filing"]["table_names"]

    base_table_file_name = table_names["table_4_story_specific_clark_blame_contrasts"]
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
        dataframe_integrated_models = compute_integrated_clark_blame_results(
            general_settings=general_settings,
            cleaned_dataframe=cleaned_dataframe,
            force_rebuild=force_rebuild,
        )
    else:
        dataframe_tests = run_confirmatory_and_exploratory_tests(
            general_settings=general_settings,
            force_rebuild=force_rebuild,
        )

    table_rows: list[dict[str, Any]] = []

    story_display_map = {"firework": "Firework", "trolley": "Trolley"}
    contrast_metadata = [
        ("CH", "BCH - BCC", "clark_blame_ch_minus_cc", "CH - CC"),
        ("DIV", "BDIV - BCC", "clark_blame_div_minus_cc", "DIV - CC"),
    ]

    for story_condition_value in ["firework", "trolley"]:
        if use_integrated_models:
            for analysis_scope_value, design_display_label in [
                ("between_subjects_first_vignette", "Between-subj"),
                ("within_subjects_repeated_measures", "Within-subj"),
            ]:
                for _, contrast_display_label, _, integrated_contrast_label in contrast_metadata:
                    integrated_row = _extract_exact_integrated_contrast_row(
                        dataframe_integrated_models=dataframe_integrated_models,
                        analysis_scope=analysis_scope_value,
                        inclusion_filter="included_only",
                        story_condition=story_condition_value,
                        contrast_label=integrated_contrast_label,
                    )

                    table_rows.append(
                        {
                            "Story": story_display_map[story_condition_value],
                            "Design": design_display_label,
                            "Contrast": contrast_display_label,
                            "Estimate": round(float(integrated_row["estimate"]), 2),
                            "95% CI": _format_ci_for_manuscript_table(integrated_row["ci95_lower"], integrated_row["ci95_upper"]),
                            "p-value": _format_p_value_for_manuscript_table(integrated_row["p_value"]),
                            "p_value_raw": float(integrated_row["p_value"]),
                            "source_file": "integrated_blame_models.csv",
                            "source_row_note": integrated_row["analysis_meaning"],
                        }
                    )

        else:
            "Between-subject story-specific rows from tests.csv"
            for between_group_a_value, contrast_display_label, within_dv_name, _ in contrast_metadata:
                between_row = _extract_exact_test_row_from_test_csv(
                    dataframe_tests=dataframe_tests,
                    dv="first_vignette_clark_blame",
                    group_a=between_group_a_value,
                    group_b="CC",
                    inclusion_filter="included_only",
                    story_condition=story_condition_value,
                    load_condition="all",
                    analysis_family="exploratory",
                )

                table_rows.append(
                    {
                        "Story": story_display_map[story_condition_value],
                        "Design": "Between-subj",
                        "Contrast": contrast_display_label,
                        "Estimate": round(float(between_row["mean_difference_a_minus_b"]), 2),
                        "95% CI": _format_ci_for_manuscript_table(between_row["ci95_lower"], between_row["ci95_upper"]),
                        "p-value": _format_p_value_for_manuscript_table(between_row["p_value"]),
                        "p_value_raw": float(between_row["p_value"]),
                        "source_file": "tests.csv",
                        "source_row_note": between_row["notes"],
                    }
                )

                within_row = _extract_exact_test_row_from_test_csv(
                    dataframe_tests=dataframe_tests,
                    dv=within_dv_name,
                    group_a="delta",
                    group_b="0",
                    inclusion_filter="included_only",
                    story_condition=story_condition_value,
                    load_condition="all",
                    analysis_family=None,
                )

                table_rows.append(
                    {
                        "Story": story_display_map[story_condition_value],
                        "Design": "Within-subj",
                        "Contrast": contrast_display_label,
                        "Estimate": round(float(within_row["mean_difference_a_minus_b"]), 2),
                        "95% CI": _format_ci_for_manuscript_table(within_row["ci95_lower"], within_row["ci95_upper"]),
                        "p-value": _format_p_value_for_manuscript_table(within_row["p_value"]),
                        "p_value_raw": float(within_row["p_value"]),
                        "source_file": "tests.csv",
                        "source_row_note": within_row["notes"],
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
    ).drop(
        columns=["story_sort_order", "design_sort_order", "contrast_sort_order"],
    )

    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table_4,
        general_settings=general_settings,
        table_file_name=table_file_name,
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
                & (dataframe_twoafc_counts["story_condition"] == "all")
            ]
            [["operator", "Bill ? Clark", "CH ? CC", "DIV ? CC"]]
            .copy()
            .rename(columns={"operator": "Operator"})
        )

    elif long_columns.issubset(set(dataframe_twoafc_counts.columns)):
        "Fallback: received long form — pivot it manually"
        dataframe_twoafc_counts = dataframe_twoafc_counts.loc[
            (dataframe_twoafc_counts["inclusion_filter"] == "included_only")
            & (dataframe_twoafc_counts["story_condition"] == "all")
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
    force_rebuild: bool | None = None
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

    load_display_map = {"all": "Pooled", "high": "High", "low": "Low"}
    dv_map = {
        "clark_blame_ch_minus_cc": "BCH - BCC",
        "clark_blame_div_minus_cc": "BDIV - BCC",
    }

    table_rows: list[dict[str, Any]] = []

    for load_condition_value in ["all", "high", "low"]:
        for dv_value, contrast_display_label in dv_map.items():
            row_series = _extract_exact_test_row_from_test_csv(
                dataframe_tests=dataframe_tests,
                dv=dv_value,
                group_a="delta",
                group_b="0",
                inclusion_filter="included_only",
                story_condition="all",
                load_condition=load_condition_value,
                analysis_family=None,
            )

            table_rows.append(
                {
                    "Cognitive Load": load_display_map[load_condition_value],
                    "Contrast": contrast_display_label,
                    "Estimate": round(float(row_series["mean_difference_a_minus_b"]), 2),
                    "95% CI": _format_ci_for_manuscript_table(row_series["ci95_lower"], row_series["ci95_upper"]),
                    "p-value": _format_p_value_for_manuscript_table(row_series["p_value"]),
                    "p_value_raw": float(row_series["p_value"]),
                    "source_file": "tests.csv",
                    "source_row_note": row_series["notes"],
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
        dataframe_integrated_models = compute_integrated_clark_blame_results(
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
                    "source_row_note": coefficient_row["analysis_meaning"],
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
                    "source_row_note": omnibus_row["analysis_meaning"],
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
            "clark_blame": "Blame",
            "clark_wrong": "Wrongness",
            "clark_punish": "Punishment",
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
                        "source_row_note": row_series["comparison"],
                    }
                )

        dataframe_table_8 = pd.DataFrame(table_rows)

        dv_sort_order = {"Blame": 0, "Wrongness": 1, "Punishment": 2}
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


def compute_supplementary_table_9_secondary_dv_contrasts(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame | None = None,
    force_rebuild: bool | None = None,
) -> pd.DataFrame:
    """
    Compute supplementary Table 9: included-only contrasts across blame, wrongness, and punishment.

    Arguments:
        • general_settings: GeneralSettings
        • cleaned_dataframe: pd.DataFrame | None
        • force_rebuild: bool | None

    Returns:
        • pd.DataFrame
            - Manuscript-facing secondary-DV contrast table.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    table_names = general_settings["filing"]["table_names"]

    table_extraction = _load_table_dataframe_from_tables_folder(
        general_settings=general_settings,
        table_file_name=table_names["table_9_secondary_dv_contrasts"],
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

    dv_display_map = {
        "blame": ("first_vignette_clark_blame", "clark_blame", "Blame"),
        "wrongness": ("first_vignette_clark_wrong", "clark_wrong", "Wrongness"),
        "punishment": ("first_vignette_clark_punish", "clark_punish", "Punishment"),
    }
    contrast_map = {
        "CH - CC": ("CH", "CC", "ch_minus_cc"),
        "DIV - CC": ("DIV", "CC", "div_minus_cc"),
    }

    table_rows: list[dict[str, Any]] = []

    for dv_short_label, (between_dv_name, within_dv_prefix, dv_display_label) in dv_display_map.items():
        for contrast_display_label, (between_group_a_value, between_group_b_value, within_dv_suffix) in contrast_map.items():
            between_row = _extract_exact_test_row_from_test_csv(
                dataframe_tests=dataframe_tests,
                dv=between_dv_name,
                group_a=between_group_a_value,
                group_b=between_group_b_value,
                inclusion_filter="included_only",
                story_condition="all",
                load_condition="all",
                analysis_family="confirmatory" if dv_short_label == "blame" else None,
            )

            between_reported_p_value = (
                between_row["p_value_holm"]
                if dv_short_label == "blame" and pd.notna(between_row["p_value_holm"])
                else between_row["p_value"]
            )

            table_rows.append(
                {
                    "DV": dv_display_label,
                    "Design": "Between-subj",
                    "Contrast": contrast_display_label,
                    "Estimate": round(float(between_row["mean_difference_a_minus_b"]), 2),
                    "95% CI": _format_ci_for_manuscript_table(between_row["ci95_lower"], between_row["ci95_upper"]),
                    "p-value": _format_p_value_for_manuscript_table(between_reported_p_value),
                    "p_value_raw": float(between_row["p_value"]),
                    "p_value_holm": (
                        float(between_row["p_value_holm"])
                        if pd.notna(between_row["p_value_holm"])
                        else np.nan
                    ),
                    "p_value_reported": float(between_reported_p_value),
                    "source_file": "tests.csv",
                    "source_row_note": between_row["notes"],
                }
            )

            within_row = _extract_exact_test_row_from_test_csv(
                dataframe_tests=dataframe_tests,
                dv=f"{within_dv_prefix}_{within_dv_suffix}",
                group_a="delta",
                group_b="0",
                inclusion_filter="included_only",
                story_condition="all",
                load_condition="all",
                analysis_family=None,
            )

            table_rows.append(
                {
                    "DV": dv_display_label,
                    "Design": "Within-subj",
                    "Contrast": contrast_display_label,
                    "Estimate": round(float(within_row["mean_difference_a_minus_b"]), 2),
                    "95% CI": _format_ci_for_manuscript_table(within_row["ci95_lower"], within_row["ci95_upper"]),
                    "p-value": _format_p_value_for_manuscript_table(within_row["p_value"]),
                    "p_value_raw": float(within_row["p_value"]),
                    "p_value_holm": np.nan,
                    "p_value_reported": float(within_row["p_value"]),
                    "source_file": "tests.csv",
                    "source_row_note": within_row["notes"],
                }
            )

    dataframe_table_9 = pd.DataFrame(table_rows)

    dv_sort_order = {"Blame": 0, "Wrongness": 1, "Punishment": 2}
    design_sort_order = {"Between-subj": 0, "Within-subj": 1}
    contrast_sort_order = {"CH - CC": 0, "DIV - CC": 1}

    dataframe_table_9["dv_sort_order"] = dataframe_table_9["DV"].map(dv_sort_order)
    dataframe_table_9["design_sort_order"] = dataframe_table_9["Design"].map(design_sort_order)
    dataframe_table_9["contrast_sort_order"] = dataframe_table_9["Contrast"].map(contrast_sort_order)

    dataframe_table_9 = dataframe_table_9.sort_values(
        by=["dv_sort_order", "design_sort_order", "contrast_sort_order"],
        kind="stable",
    ).drop(columns=["dv_sort_order", "design_sort_order", "contrast_sort_order"])

    _save_table_dataframe_to_tables_folder(
        dataframe_table=dataframe_table_9,
        general_settings=general_settings,
        table_file_name=table_names["table_9_secondary_dv_contrasts"],
    )

    return dataframe_table_9


def _compute_extra_terminal_statistics_for_manuscript(
    general_settings: GeneralSettings,
    cleaned_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute a few one-line statistics that are useful to print to the terminal but do not need their own tables.

    Arguments:
        • general_settings: GeneralSettings
        • cleaned_dataframe: pd.DataFrame

    Returns:
        • dict[str, Any]
            - Dictionary of summary statistics.
    """
    included_dataframe = cleaned_dataframe.loc[cleaned_dataframe["included"] == True].copy()  # noqa: E712

    first_vignette_blame_wrong_dataframe = included_dataframe[
        ["first_vignette_clark_blame", "first_vignette_clark_wrong"]
    ].dropna().copy()
    first_vignette_blame_wrong_correlation = stats.pearsonr(
        first_vignette_blame_wrong_dataframe["first_vignette_clark_blame"],
        first_vignette_blame_wrong_dataframe["first_vignette_clark_wrong"],
    )

    participant_means_dataframe = pd.DataFrame(
        {
            "mean_clark_blame": included_dataframe[["clark_blame_cc", "clark_blame_ch", "clark_blame_div"]].mean(axis=1),
            "mean_clark_wrong": included_dataframe[["clark_wrong_cc", "clark_wrong_ch", "clark_wrong_div"]].mean(axis=1),
        }
    ).dropna()
    participant_means_blame_wrong_correlation = stats.pearsonr(
        participant_means_dataframe["mean_clark_blame"],
        participant_means_dataframe["mean_clark_wrong"],
    )

    cognitive_load_blame_difference_row = run_welch_t_test_between_groups(
        dataframe=included_dataframe,
        dv_column_name="clark_blame_ch_minus_cc",
        group_column_name="load_condition",
        group_a_value="high",
        group_b_value="low",
    )

    return {
        "first_vignette_blame_wrong_r": float(first_vignette_blame_wrong_correlation.statistic),
        "first_vignette_blame_wrong_p": float(first_vignette_blame_wrong_correlation.pvalue),
        "participant_means_blame_wrong_r": float(participant_means_blame_wrong_correlation.statistic),
        "participant_means_blame_wrong_p": float(participant_means_blame_wrong_correlation.pvalue),
        "high_minus_low_ch_cc_delta_difference": float(cognitive_load_blame_difference_row["mean_difference_a_minus_b"]),
        "high_minus_low_ch_cc_delta_ci95_lower": float(cognitive_load_blame_difference_row["ci95_lower"]),
        "high_minus_low_ch_cc_delta_ci95_upper": float(cognitive_load_blame_difference_row["ci95_upper"]),
        "high_minus_low_ch_cc_delta_p": float(cognitive_load_blame_difference_row["p_value"]),
    }


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
            - Dictionary containing the saved table dataframes and their manifest.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

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
    dataframe_table_3 = compute_manuscript_table_3_primary_clark_blame_contrasts(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )
    dataframe_table_4 = compute_manuscript_table_4_story_specific_clark_blame_contrasts(
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
            "table_title": "Table 2. Participant counts by randomization condition",
            "file_name": table_names["table_2_means_by_dv_and_condition"],
            "table_meaning": "Main-text mean scale values by DV and condition.",
        },        
        {
            "table_key": "table_3",
            "table_title": "Table 3. Primary Clark-blame contrasts",
            "file_name": table_names["table_3_primary_clark_blame_contrasts"],
            "table_meaning": "Main-text primary contrast table showing blame contrasts between- and within-subjects.",
        },
        {
            "table_key": "table_4",
            "table_title": "Table 4. Story-specific Clark-blame contrasts",
            "file_name": table_names["table_4_story_specific_clark_blame_contrasts"],
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

        extra_statistics = _compute_extra_terminal_statistics_for_manuscript(
            general_settings=general_settings,
            cleaned_dataframe=cleaned_dataframe,
        )

        print("\n" + "-" * 110)
        print("EXTRA ONE-LINE STATISTICS")
        print("-" * 110)
        print(
            "First-vignette blame-wrongness correlation: "
            f"r = {extra_statistics['first_vignette_blame_wrong_r']:.2f}, "
            f"p = {_format_p_value_for_manuscript_table(extra_statistics['first_vignette_blame_wrong_p'])}"
        )
        print(
            "Participant-means blame-wrongness correlation: "
            f"r = {extra_statistics['participant_means_blame_wrong_r']:.2f}, "
            f"p = {_format_p_value_for_manuscript_table(extra_statistics['participant_means_blame_wrong_p'])}"
        )
        print(
            "Cognitive-load difference in CH - CC shielding (High - Low): "
            f"Δ = {extra_statistics['high_minus_low_ch_cc_delta_difference']:+.2f}, "
            f"95% CI = {_format_ci_for_manuscript_table(extra_statistics['high_minus_low_ch_cc_delta_ci95_lower'], extra_statistics['high_minus_low_ch_cc_delta_ci95_upper'])}, "
            f"p = {_format_p_value_for_manuscript_table(extra_statistics['high_minus_low_ch_cc_delta_p'])}"
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
}

table_names: dict[str, str] = {
    "table_1_participant_counts": "Table_1_Participant_Counts.csv",
    "table_2_means_by_dv_and_condition": "Table_2_Means_by_DV_and_Condition.csv",
    "table_3_primary_clark_blame_contrasts": "table_3_Primary_Clark_Blame_Contrasts.csv",
    "table_4_story_specific_clark_blame_contrasts": "table_4_Story_Specific_Clark_Blame_Contrasts.csv",
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
    "misc": {
        "confirmatory_between_subjects_method": confirmatory_between_subjects_method,
        "rebuild_cleaned_dataframe": rebuild_cleaned_dataframe,
        "print_tables_to_terminal": print_tables_to_terminal,
        "freeze_timestamp_first": freeze_timestamp_first,
        "freeze_timestamp_last": freeze_timestamp_last,
        "use_integrated_models": use_integrated_models,
        "force_rebuild": force_rebuild
    }
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
    
    "Integrated models"
    compute_integrated_clark_blame_results(
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
        plot_shielding_effects_by_cognitive_load(dv="blame", base_hue=base_hue, general_settings=general_settings, delta_type="CH_CC", figure_type="violin")

        "Figure 7"
        plot_trial_order_effects_line_graph(     dv="blame", base_hue=base_hue, general_settings=general_settings, order_analysis_mode="legacy")

        "Figure 8"
        plot_blameworthiness_wrongness_correlate(            base_hue=base_hue, general_settings=general_settings, all_ratings=True, jitter_strength=0.1)

        "Additional Figures"
        plot_triangulation_2afc_vs_rating_delta( dv="blame", base_hue=base_hue, general_settings=general_settings, comparison="CH_CC")
        plot_shielding_by_individual_difference( dv="blame", base_hue=base_hue, general_settings=general_settings, predictor="indcol")
        plot_shielding_by_individual_difference( dv="blame", base_hue=base_hue, general_settings=general_settings, predictor="crt")

    "Tables in the order they appear in the paper"
    generate_manuscript_and_supplementary_tables(
        general_settings=general_settings,
        force_rebuild=None,
    )


if __name__ == "__main__":
    main()





