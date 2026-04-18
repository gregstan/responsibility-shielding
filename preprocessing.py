from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import pandas as pd, numpy as np, re, os, copy
from config import (
    GeneralSettings,
    Filing,
    FileNames,
    FilePaths,
    TableNames,
    MiscSettings,
    PunishSettings,
    Visuals,
)


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

    raw_dataframe["vignette_condition_order"] = raw_dataframe["CaseOrder"].astype(str)

    raw_dataframe["vignette_condition_position_1"] = raw_dataframe["vignette_condition_order"].str.split("-").str[0]
    raw_dataframe["vignette_condition_position_2"] = raw_dataframe["vignette_condition_order"].str.split("-").str[1]
    raw_dataframe["vignette_condition_position_3"] = raw_dataframe["vignette_condition_order"].str.split("-").str[2]

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

    raw_dataframe["comprehension_distal_necessary_bool"] = coalesce_series(
        raw_dataframe["comp_p_agency_across"].apply(parse_boolean_value),
        raw_dataframe["comp_t_agency_across"].apply(parse_boolean_value),
    )

    raw_dataframe["comprehension_bill_necessary_bool"] = coalesce_series(
        raw_dataframe["comp_p_agency_within"].apply(parse_boolean_value),
        raw_dataframe["comp_t_agency_within"].apply(parse_boolean_value),
    )

    raw_dataframe["comprehension_all_correct_bool"] = (
        raw_dataframe["comprehension_probability_same_bool"]
        & raw_dataframe["comprehension_distal_necessary_bool"]
        & raw_dataframe["comprehension_bill_necessary_bool"]
    )

    "Merge core outcome measures across story condition (Clark = upstream actor; Bill/proximate = downstream node)"
    raw_dataframe["distal_blame_cc"] = coalesce_series(
        raw_dataframe["pucc_blame_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tucc_blame_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["distal_blame_ch"] = coalesce_series(
        raw_dataframe["puch_blame_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tuch_blame_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["distal_blame_div"] = coalesce_series(
        raw_dataframe["pudiv_blame_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tudiv_blame_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["distal_wrong_cc"] = coalesce_series(
        raw_dataframe["pucc_wrong_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tucc_wrong_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["distal_wrong_ch"] = coalesce_series(
        raw_dataframe["puch_wrong_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tuch_wrong_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["distal_wrong_div"] = coalesce_series(
        raw_dataframe["pudiv_wrong_1"].apply(parse_likert_numeric_value),
        raw_dataframe["tudiv_wrong_1"].apply(parse_likert_numeric_value),
    )

    raw_dataframe["distal_punish_cc"] = coalesce_series(
        pd.to_numeric(raw_dataframe["pucc_punish"], errors="coerce"),
        pd.to_numeric(raw_dataframe["tucc_punish"], errors="coerce"),
    )

    raw_dataframe["distal_punish_ch"] = coalesce_series(
        pd.to_numeric(raw_dataframe["puch_punish"], errors="coerce"),
        pd.to_numeric(raw_dataframe["tuch_punish"], errors="coerce"),
    )

    raw_dataframe["distal_punish_div"] = coalesce_series(
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
        case_code_value = row["vignette_condition_position_1"]
        if case_code_value == "CC":
            return row[column_cc]
        if case_code_value == "CH":
            return row[column_ch]
        if case_code_value == "DIV":
            return row[column_div]
        return np.nan

    raw_dataframe["first_vignette_distal_blame"] = raw_dataframe.apply(
        lambda row: choose_first_case_value(row, "distal_blame_cc", "distal_blame_ch", "distal_blame_div"),
        axis=1,
    )

    raw_dataframe["first_vignette_distal_wrong"] = raw_dataframe.apply(
        lambda row: choose_first_case_value(row, "distal_wrong_cc", "distal_wrong_ch", "distal_wrong_div"),
        axis=1,
    )

    raw_dataframe["first_vignette_distal_punish"] = raw_dataframe.apply(
        lambda row: choose_first_case_value(row, "distal_punish_cc", "distal_punish_ch", "distal_punish_div"),
        axis=1,
    )

    "Within-subject deltas (Clark: CH-CC and DIV-CC; Proximate: CC-CH) for blame, wrongness, punishment"
    raw_dataframe["distal_blame_ch_minus_cc"] = raw_dataframe["distal_blame_ch"] - raw_dataframe["distal_blame_cc"]
    raw_dataframe["distal_blame_div_minus_cc"] = raw_dataframe["distal_blame_div"] - raw_dataframe["distal_blame_cc"]
    raw_dataframe["proximate_blame_cc_minus_ch"] = raw_dataframe["proximate_blame_cc"] - raw_dataframe["proximate_blame_ch"]

    raw_dataframe["distal_wrong_ch_minus_cc"] = raw_dataframe["distal_wrong_ch"] - raw_dataframe["distal_wrong_cc"]
    raw_dataframe["distal_wrong_div_minus_cc"] = raw_dataframe["distal_wrong_div"] - raw_dataframe["distal_wrong_cc"]
    raw_dataframe["proximate_wrong_cc_minus_ch"] = raw_dataframe["proximate_wrong_cc"] - raw_dataframe["proximate_wrong_ch"]

    raw_dataframe["distal_punish_ch_minus_cc"] = raw_dataframe["distal_punish_ch"] - raw_dataframe["distal_punish_cc"]
    raw_dataframe["distal_punish_div_minus_cc"] = raw_dataframe["distal_punish_div"] - raw_dataframe["distal_punish_cc"]
    raw_dataframe["proximate_punish_cc_minus_ch"] = raw_dataframe["proximate_punish_cc"] - raw_dataframe["proximate_punish_ch"]

    "Responsibility Shielding Effect (per preregistration): min(CH, DIV) - CC (computed within-subjects per participant)"
    raw_dataframe["distal_blame_min_ch_div_minus_cc"] = np.minimum(raw_dataframe["distal_blame_ch"], raw_dataframe["distal_blame_div"]) - raw_dataframe["distal_blame_cc"]
    raw_dataframe["distal_wrong_min_ch_div_minus_cc"] = np.minimum(raw_dataframe["distal_wrong_ch"], raw_dataframe["distal_wrong_div"]) - raw_dataframe["distal_wrong_cc"]
    raw_dataframe["distal_punish_min_ch_div_minus_cc"] = np.minimum(raw_dataframe["distal_punish_ch"], raw_dataframe["distal_punish_div"]) - raw_dataframe["distal_punish_cc"]

    "Interpersonal blame disparity within a given vignette (diagnostic, not the definition of shielding)"
    raw_dataframe["bill_minus_distal_cc_blame"] = raw_dataframe["proximate_blame_cc"] - raw_dataframe["distal_blame_cc"]
    raw_dataframe["bill_minus_distal_ch_blame"] = raw_dataframe["proximate_blame_ch"] - raw_dataframe["distal_blame_ch"]

    raw_dataframe["bill_minus_distal_cc_wrong"] = raw_dataframe["proximate_wrong_cc"] - raw_dataframe["distal_wrong_cc"]
    raw_dataframe["bill_minus_distal_ch_wrong"] = raw_dataframe["proximate_wrong_ch"] - raw_dataframe["distal_wrong_ch"]

    raw_dataframe["bill_minus_distal_cc_punish"] = raw_dataframe["proximate_punish_cc"] - raw_dataframe["distal_punish_cc"]
    raw_dataframe["bill_minus_distal_ch_punish"] = raw_dataframe["proximate_punish_ch"] - raw_dataframe["distal_punish_ch"]

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
        ("distal_cc", "pucc", "tucc"),
        ("distal_ch", "puch", "tuch"),
        ("distal_div", "pudiv", "tudiv"),
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
        "vignette_condition_order",
        "vignette_condition_position_1",
        "vignette_condition_position_2",
        "vignette_condition_position_3",
        "first_vignette_distal_blame",
        "first_vignette_distal_wrong",
        "first_vignette_distal_punish",
        "distal_blame_ch_minus_cc",
        "distal_blame_div_minus_cc",
        "proximate_blame_cc_minus_ch",
        "distal_wrong_ch_minus_cc",
        "distal_wrong_div_minus_cc",
        "proximate_wrong_cc_minus_ch",
        "distal_punish_ch_minus_cc",
        "distal_punish_div_minus_cc",
        "proximate_punish_cc_minus_ch",
        "distal_blame_min_ch_div_minus_cc",
        "distal_wrong_min_ch_div_minus_cc",
        "distal_punish_min_ch_div_minus_cc",
        "twoafc_bill_vs_clark",
        "twoafc_ch_vs_cc",
        "twoafc_div_vs_cc",
        "distal_blame_cc",
        "distal_blame_ch",
        "distal_blame_div",
        "distal_wrong_cc",
        "distal_wrong_ch",
        "distal_wrong_div",
        "distal_punish_cc",
        "distal_punish_ch",
        "distal_punish_div",
        "proximate_blame_cc",
        "proximate_blame_ch",
        "proximate_wrong_cc",
        "proximate_wrong_ch",
        "proximate_punish_cc",
        "proximate_punish_ch",
        "comprehension_probability_same_bool",
        "comprehension_distal_necessary_bool",
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


