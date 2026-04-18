from __future__ import annotations
from scipy import stats
from pathlib import Path
from typing import Sequence, Any
from plotly.subplots import make_subplots
import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf, plotly.graph_objects as go, copy, re
from preprocessing import load_or_build_cleaned_dataframe, load_analysis_dataframe
from config import GeneralSettings, Filing, Visuals


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
    if subjects_normalized in {"both", "all", "pooled", "combined", "subplots"}:
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
        "Unrecognized dependent_variable. "
        "Expected one of: 'blame' (or 'blameworthiness'), 'wrong' (or 'wrongness'), 'punish' (or 'punishment', 'years', 'prison'). "
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
                "pooled"
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

        if token_normalized in {"pooled", "both", "all_conditions", "all conditions"}:
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

        if token_normalized in {"both", "pooled", "all_deltas", "all deltas"}:
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
            "column": f"distal_{dv_suffix}_ch_minus_cc",
            "label": "CH - CC",
            "long_label": "Shielding (CH - CC)",
        },
        "DIV_CC": {
            "column": f"distal_{dv_suffix}_div_minus_cc",
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
    figure_layout = general_settings["visuals"]["figure_layout"]
    default_marker_size = general_settings["visuals"]["marker_size"]
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

    first_vignette_value_column = f"first_vignette_distal_{dv_suffix}"
    if first_vignette_value_column not in analysis_dataframe.columns:
        raise KeyError(
            f"Expected column {first_vignette_value_column!r} in the cleaned dataframe, but it was not found."
        )

    between_long = pd.DataFrame(
        {
            "condition_code": analysis_dataframe["vignette_condition_position_1"].astype(str).str.upper(),
            "rating_value": analysis_dataframe[first_vignette_value_column],
        }
    )
    between_long["condition_label"] = between_long["condition_code"].map(condition_code_to_distal_label)
    between_long = between_long.dropna(subset=["condition_label", "rating_value"]).copy()

    within_value_columns = {
        distal_category_labels_in_order[0]: f"distal_{dv_suffix}_cc",
        distal_category_labels_in_order[1]: f"distal_{dv_suffix}_ch",
        distal_category_labels_in_order[2]: f"distal_{dv_suffix}_div",
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
            - Which dependent variable to use. Supported: 'blame', 'wrong', 'punish'.
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
    figure_layout = general_settings["visuals"]["figure_layout"]
    dv_suffix, dv_label, _ = _normalize_dependent_variable_input(dv)

    analysis_dataframe = _get_filtered_plotting_dataframe(
        general_settings=general_settings,
        story_condition=story_condition,
        cognitive_load=cognitive_load,
        only_included_participants=only_included_participants,
    )

    x_column_name = f"distal_{dv_suffix}_ch_minus_cc"
    y_column_name = f"distal_{dv_suffix}_div_minus_cc"

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
            - Which dependent variable to use. Supported: 'blame', 'wrong', 'punish'.
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
    figure_layout = general_settings["visuals"]["figure_layout"]
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
                - Which dependent variable to use. Supported: 'blame', 'wrong', 'punish'.
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
                ("CC Distal", "distal_blame_cc"),
                ("CH Distal", "distal_blame_ch"),
                ("DIV Distal", "distal_blame_div"),
            ]
            if include_proximate_agent:
                ordered_category_metadata += [
                    ("CC Proximate", "proximate_blame_cc"),
                    ("CH Proximate", "proximate_blame_ch"),
                ]
        elif dv_suffix == "wrong":
            ordered_category_metadata = [
                ("CC Distal", "distal_wrong_cc"),
                ("CH Distal", "distal_wrong_ch"),
                ("DIV Distal", "distal_wrong_div"),
            ]
            if include_proximate_agent:
                ordered_category_metadata += [
                    ("CC Proximate", "proximate_wrong_cc"),
                    ("CH Proximate", "proximate_wrong_ch"),
                ]
        else:
            ordered_category_metadata = [
                ("CC Distal", "distal_punish_cc"),
                ("CH Distal", "distal_punish_ch"),
                ("DIV Distal", "distal_punish_div"),
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
            - Which dependent variable to use. Supported: 'blame', 'wrong', 'punish'.
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
    figure_layout = general_settings["visuals"]["figure_layout"]
    default_marker_size = general_settings["visuals"]["marker_size"]
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
            - Which dependent variable to plot. Supported: 'blame', 'wrong', 'punish'.
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
    figure_layout = general_settings["visuals"]["figure_layout"]
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

        cc_value_column_name = f"distal_{dv_suffix}_cc"
        ch_value_column_name = f"distal_{dv_suffix}_ch"

        required_columns = [
            "response_id",
            "vignette_condition_position_1",
            "vignette_condition_position_2",
            "vignette_condition_position_3",
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
                "vignette_condition_position_1",
                "vignette_condition_position_2",
                "vignette_condition_position_3",
                cc_value_column_name,
                ch_value_column_name,
            ]
        ].copy()

        pairwise_order_dataframe["cc_position"] = np.select(
            condlist=[
                pairwise_order_dataframe["vignette_condition_position_1"] == "CC",
                pairwise_order_dataframe["vignette_condition_position_2"] == "CC",
                pairwise_order_dataframe["vignette_condition_position_3"] == "CC",
            ],
            choicelist=[1, 2, 3],
            default=np.nan,
        )

        pairwise_order_dataframe["ch_position"] = np.select(
            condlist=[
                pairwise_order_dataframe["vignette_condition_position_1"] == "CH",
                pairwise_order_dataframe["vignette_condition_position_2"] == "CH",
                pairwise_order_dataframe["vignette_condition_position_3"] == "CH",
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
        "CC": f"distal_{dv_suffix}_cc",
        "CH": f"distal_{dv_suffix}_ch",
        "DIV": f"distal_{dv_suffix}_div",
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
            "vignette_condition_position_1",
            "vignette_condition_position_2",
            "vignette_condition_position_3",
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
                "vignette_condition_position_1",
                "vignette_condition_position_2",
                "vignette_condition_position_3",
                value_column_name,
            ]
        ].copy()

        temp_dataframe["position_numeric"] = np.select(
            condlist=[
                temp_dataframe["vignette_condition_position_1"] == selected_condition_code,
                temp_dataframe["vignette_condition_position_2"] == selected_condition_code,
                temp_dataframe["vignette_condition_position_3"] == selected_condition_code,
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
    figure_layout = general_settings["visuals"]["figure_layout"]
    analysis_dataframe = _get_filtered_plotting_dataframe(
        general_settings=general_settings,
        story_condition=story_condition,
        cognitive_load=cognitive_load,
        only_included_participants=only_included_participants,
    )

    if all_ratings:
        ordered_rating_pairs = [
            ("CC Distal", "distal_blame_cc", "distal_wrong_cc"),
            ("CH Distal", "distal_blame_ch", "distal_wrong_ch"),
            ("DIV Distal", "distal_blame_div", "distal_wrong_div"),
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
                        ["distal_blame_cc", "distal_blame_ch", "distal_blame_div"]
                    ].mean(axis=1, skipna=True),
                    "wrong_value": analysis_dataframe[
                        ["distal_wrong_cc", "distal_wrong_ch", "distal_wrong_div"]
                    ].mean(axis=1, skipna=True),
                }
            ).dropna()

            figure_title = "Blame vs. Wrongness (Participant Means)"
        else:
            selected_condition_codes = _normalize_condition_subset_input(condition)

            condition_rows = []
            for selected_condition_code in selected_condition_codes:
                blame_column_name = f"distal_blame_{selected_condition_code.lower()}"
                wrong_column_name = f"distal_wrong_{selected_condition_code.lower()}"
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
                'blame', 'wrong', 'punish'
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
    figure_layout = general_settings["visuals"]["figure_layout"]
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
        delta_column_name = f"distal_{dv_suffix}_ch_minus_cc"
        left_prefix = "CH"
        right_prefix = "CC"
        comparison_label = "CH vs. CC"
    else:
        twoafc_column_name = "twoafc_div_vs_cc"
        delta_column_name = f"distal_{dv_suffix}_div_minus_cc"
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
            - Which dependent variable to use. Supported: 'blame', 'wrong', 'punish'.
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
    figure_layout = general_settings["visuals"]["figure_layout"]
    default_marker_size = general_settings["visuals"]["marker_size"]
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


def plot_response_distribution_histogram_by_condition(
    general_settings: GeneralSettings,
    dv: str | Any = "punish",
    include_proximate_agent: bool = True,
    story_condition: str | Any = None,
    cognitive_load: str | Any = None,
    only_included_participants: bool = True,
    export_html: bool = True,
    base_hue: int = 200,
) -> "object":
    """
    Plot a condition-selectable histogram with a count-scaled smooth KDE overlay.

    This function is designed to visualize the raw response distributions for one dependent variable
    across the main vignette-condition series:
        • CC Distal
        • CH Distal
        • DIV Distal
        • CC Proximate
        • CH Proximate

    A dropdown menu toggles between:
        • All conditions
        • each condition separately

    The x-axis and y-axis are fixed across dropdown selections so the plot remains visually comparable.

    Arguments:
        • general_settings: GeneralSettings
            - Master settings dictionary.
        • dv: str | Any
            - Which dependent variable to plot. Supported:
                "blame"
                "wrong"
                "punish"
        • include_proximate_agent: bool
            - If True, include CC Proximate and CH Proximate in the dropdown and plot.
            - If False, only the three distal series are shown.
        • story_condition: str | Any
            - If None, pool stories.
            - If "firework" or "trolley", filter to that story condition.
        • cognitive_load: str | Any
            - If None, pool load conditions.
            - If "high" or "low", filter to that load condition.
        • only_included_participants: bool
            - If True, restrict to participants who passed all comprehension checks.
        • export_html: bool
            - If True, export the Plotly figure to an .html file.
        • base_hue: int
            - Starting hue for hsla colors.

    Notes:
        • The KDE curve is rescaled into count units:
              density(x) × n × bin_width
          so that it sits naturally on the same y-axis as the histogram counts.
        • I fixed the y-axis by computing the tallest bin / scaled-KDE height across all condition
          selections ahead of time, so the axes do not jump when the dropdown changes.
        • For punishment, the histogram uses integer-year bins from 0 to 50.
        • For blame and wrongness, the histogram uses integer bins from 1 to 9.

    Returns:
        • plotly.graph_objects.Figure
    """
    figure_layout = general_settings["visuals"]["figure_layout"]
    "========================================="
    "Normalize inputs and filter the dataset."
    "========================================="
    dv_suffix, dv_label, _ = _normalize_dependent_variable_input(dv)
    story_condition_normalized = _normalize_story_condition_input(story_condition)
    cognitive_load_normalized = _normalize_load_condition_input(cognitive_load)

    analysis_dataframe = _get_filtered_plotting_dataframe(
        general_settings=general_settings,
        story_condition=story_condition,
        cognitive_load=cognitive_load,
        only_included_participants=only_included_participants,
    )

    "=========================================="
    "Resolve which rating columns to visualize."
    "=========================================="
    ordered_condition_metadata: list[tuple[str, str]] = [
        ("CC Distal", f"distal_{dv_suffix}_cc"),
        ("CH Distal", f"distal_{dv_suffix}_ch"),
        ("DIV Distal", f"distal_{dv_suffix}_div"),
    ]
    if include_proximate_agent:
        ordered_condition_metadata += [
            ("CC Proximate", f"proximate_{dv_suffix}_cc"),
            ("CH Proximate", f"proximate_{dv_suffix}_ch"),
        ]

    missing_required_columns = [
        column_name
        for _, column_name in ordered_condition_metadata
        if column_name not in analysis_dataframe.columns
    ]
    if missing_required_columns:
        raise KeyError(
            "Missing one or more expected rating columns required for the response-distribution histogram: "
            + ", ".join(repr(column_name) for column_name in missing_required_columns)
        )

    "==========================================="
    "Build a clean long dataframe for plotting."
    "==========================================="
    long_rows: list[pd.DataFrame] = []

    for condition_label, column_name in ordered_condition_metadata:
        temp_dataframe = pd.DataFrame(
            {
                "condition_label": condition_label,
                "rating_value": pd.to_numeric(analysis_dataframe[column_name], errors="coerce"),
            }
        ).dropna(subset=["rating_value"]).copy()

        long_rows.append(temp_dataframe)

    plotting_dataframe = pd.concat(long_rows, ignore_index=True)

    if plotting_dataframe.shape[0] == 0:
        raise ValueError("No valid ratings remain after filtering for the histogram plot.")

    "=========================================================="
    "Choose discrete histogram bins and fixed x-axis settings."
    "=========================================================="
    if dv_suffix in {"blame", "wrong"}:
        histogram_bin_edges = np.arange(0.5, 9.5 + 1.0, 1.0)
        histogram_bin_width = 1.0
        x_axis_range = [0.5, 9.5]
        x_axis_tick_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        x_axis_tick_text = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    else:
        histogram_bin_edges = np.arange(-0.5, 50.5 + 1.0, 1.0)
        histogram_bin_width = 1.0
        x_axis_range = [-0.5, 50.5]
        x_axis_tick_values = list(range(0, 51, 5))
        x_axis_tick_text = [str(tick_value) for tick_value in x_axis_tick_values]

    x_grid_for_kde = np.linspace(x_axis_range[0], x_axis_range[1], 800)

    "=================================="
    "Helper to compute a scaled KDE."
    "=================================="
    def compute_count_scaled_kde_curve(
        numeric_values: np.ndarray,
        x_grid: np.ndarray,
        histogram_bin_width_value: float,
    ) -> np.ndarray:
        """
        Compute a KDE curve rescaled into count units.

        Arguments:
            • numeric_values: np.ndarray
                - Clean numeric values for one condition.
            • x_grid: np.ndarray
                - Grid of x values where the KDE is evaluated.
            • histogram_bin_width_value: float
                - Width of one histogram bin.

        Returns:
            • np.ndarray
                - KDE values placed on the same count-like scale as the histogram.
        """
        numeric_values = numeric_values[~np.isnan(numeric_values)]

        if numeric_values.shape[0] < 2:
            return np.zeros_like(x_grid, dtype=float)

        if np.unique(numeric_values).shape[0] < 2:
            return np.zeros_like(x_grid, dtype=float)

        try:
            gaussian_kde_object = stats.gaussian_kde(numeric_values)
            density_values = gaussian_kde_object(x_grid)
            count_scaled_kde_values = density_values * numeric_values.shape[0] * histogram_bin_width_value
            return np.asarray(count_scaled_kde_values, dtype=float)
        except Exception:
            return np.zeros_like(x_grid, dtype=float)

    "=============================================================="
    "Precompute per-condition counts, KDEs, and fixed y-axis max."
    "=============================================================="
    condition_plotting_data: dict[str, dict[str, Any]] = {}
    global_max_y_value: float = 0.0

    for condition_index, (condition_label, _) in enumerate(ordered_condition_metadata):
        condition_values = pd.to_numeric(
            plotting_dataframe.loc[
                plotting_dataframe["condition_label"] == condition_label,
                "rating_value"
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        histogram_counts, _ = np.histogram(condition_values, bins=histogram_bin_edges)
        kde_curve_values = compute_count_scaled_kde_curve(
            numeric_values=condition_values,
            x_grid=x_grid_for_kde,
            histogram_bin_width_value=histogram_bin_width,
        )

        fill_color = _hsla_color(hue=base_hue + 30 * condition_index, alpha=0.40)
        line_color = _hsla_color(hue=base_hue + 30 * condition_index, alpha=1.00)

        condition_plotting_data[condition_label] = {
            "values": condition_values,
            "histogram_counts": histogram_counts,
            "kde_curve_values": kde_curve_values,
            "fill_color": fill_color,
            "line_color": line_color,
        }

        if histogram_counts.shape[0] > 0:
            global_max_y_value = max(global_max_y_value, float(np.max(histogram_counts)))
        if kde_curve_values.shape[0] > 0:
            global_max_y_value = max(global_max_y_value, float(np.max(kde_curve_values)))

    if global_max_y_value <= 0:
        global_max_y_value = 1.0

    y_axis_range = [0.0, global_max_y_value * 1.10]

    "=============================="
    "Construct the Plotly figure."
    "=============================="
    fig = go.Figure()
    trace_indices_by_condition_label: dict[str, list[int]] = {}

    for condition_label in [condition_label for condition_label, _ in ordered_condition_metadata]:
        condition_values = condition_plotting_data[condition_label]["values"]
        fill_color = condition_plotting_data[condition_label]["fill_color"]
        line_color = condition_plotting_data[condition_label]["line_color"]
        kde_curve_values = condition_plotting_data[condition_label]["kde_curve_values"]

        condition_trace_indices: list[int] = []

        fig.add_trace(
            go.Histogram(
                x=condition_values,
                name=condition_label,
                xbins=dict(
                    start=float(histogram_bin_edges[0]),
                    end=float(histogram_bin_edges[-1]),
                    size=float(histogram_bin_width),
                ),
                histfunc="count",
                opacity=0.55,
                marker=dict(
                    color=fill_color,
                    line=dict(color=line_color, width=1.8),
                ),
                hovertemplate=(
                    f"{condition_label}<br>"
                    "Response bin center: %{x}<br>"
                    "Count: %{y}<extra></extra>"
                ),
                showlegend=True,
                visible=True,
            )
        )
        condition_trace_indices.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=x_grid_for_kde,
                y=kde_curve_values,
                mode="lines",
                name=f"{condition_label} KDE",
                line=dict(color=line_color, width=5),
                hovertemplate=(
                    f"{condition_label} KDE<br>"
                    "Response value: %{x:.2f}<br>"
                    "Scaled density: %{y:.2f}<extra></extra>"
                ),
                showlegend=False,
                visible=True,
            )
        )
        condition_trace_indices.append(len(fig.data) - 1)

        trace_indices_by_condition_label[condition_label] = condition_trace_indices

    "============================================"
    "Build dropdown visibility masks and titles."
    "============================================"
    all_trace_count = len(fig.data)

    def build_visibility_mask_for_selection(
        selected_condition_label: str | None,
    ) -> list[bool]:
        """
        Build the visible-mask for one dropdown selection.

        Arguments:
            • selected_condition_label: str | None
                - If None, show all conditions.
                - Otherwise, show only that condition's histogram + KDE.

        Returns:
            • list[bool]
                - Visibility mask aligned to fig.data.
        """
        visible_mask = [False] * all_trace_count

        if selected_condition_label is None:
            for condition_trace_indices in trace_indices_by_condition_label.values():
                for trace_index in condition_trace_indices:
                    visible_mask[trace_index] = True
            return visible_mask

        for trace_index in trace_indices_by_condition_label[selected_condition_label]:
            visible_mask[trace_index] = True

        return visible_mask

    figure_title_base = f"{dv_label} Response Distribution by Condition"
    dropdown_buttons = [
        dict(
            label="All conditions",
            method="update",
            args=[
                {"visible": build_visibility_mask_for_selection(None)},
                {"title": f"{figure_title_base} - All conditions"},
            ],
        )
    ]

    for condition_label, _ in ordered_condition_metadata:
        dropdown_buttons.append(
            dict(
                label=condition_label,
                method="update",
                args=[
                    {"visible": build_visibility_mask_for_selection(condition_label)},
                    {"title": f"{figure_title_base} - {condition_label}"},
                ],
            )
        )

    "=========================="
    "Apply layout and styling."
    "=========================="
    fig.update_layout(**figure_layout, title=f"{figure_title_base} - All conditions")
    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_font_size=36,
        barmode="overlay",
        margin=dict(l=110, r=110, t=95, b=95),
        bargap=0.02,
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                x=1.02,
                xanchor="left",
                y=1.00,
                yanchor="top",
                bgcolor="hsla(0, 0%, 100%, 0.95)",
                bordercolor="hsla(0, 0%, 0%, 0.35)",
                borderwidth=1,
                font=dict(size=16, family="Calibri", color="black"),
            )
        ],
    )

    fig.update_layout(
        legend=dict(
            bgcolor="hsla(0, 0%, 100%, 0.2)",
            bordercolor="hsla(0, 0%, 80%, 0.6)",
            borderwidth=1, orientation="h", 
            font=dict(size=15, family="Calibri", color="black"),
            xanchor="center", x=0.5,
            yanchor="top", y=-0.14,              
        ),
        margin=dict(l=110, r=110, t=95, b=140),   # extra bottom room for the legend
    )

    if dv_suffix == "punish":
        x_axis_title = "Punishment Recommendation (years in prison)"
    elif dv_suffix == "blame":
        x_axis_title = "Blameworthiness Rating"
    else:
        x_axis_title = "Wrongness Rating"

    fig.update_xaxes(
        title_text=x_axis_title,
        range=x_axis_range,
        tickmode="array",
        tickvals=x_axis_tick_values,
        ticktext=x_axis_tick_text,
        zeroline=False,
        showline=True,
        mirror=True,
        tickwidth=0,
        ticklen=0,
        ticks="",
    )

    fig.update_yaxes(
        title_text="Count",
        range=y_axis_range,
        zeroline=False,
        showline=True,
        mirror=True,
        tickwidth=0,
        ticklen=0,
        ticks="",
    )

    fig.add_annotation(
        x=0.5,
        y=8.08,
        xref="paper",
        yref="paper",
        text="Solid bars = counts; smooth line = count-scaled KDE",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(size=18, family="Calibri", color="black"),
        bgcolor="hsla(0, 0%, 100%, 0.00)",
    )

    "============================"
    "Export the figure, if asked."
    "============================"
    if export_html:
        story_tag = "pooled" if story_condition_normalized is None else story_condition_normalized
        load_tag = "pooled" if cognitive_load_normalized is None else cognitive_load_normalized
        proximate_tag = "with_proximate" if include_proximate_agent else "distal_only"

        file_name_figure = f"figure_x_response_distribution_histogram_{dv_suffix}_{story_tag}_{load_tag}_{proximate_tag}"
        _export_plotly_figure_html(
            fig=fig,
            general_settings=general_settings,
            file_name=file_name_figure,
        )

    return fig


