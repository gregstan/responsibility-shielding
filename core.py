from __future__ import annotations
from pathlib import Path
from scipy import stats
from typing import Sequence, Any, Dict, List, Tuple
import pandas as pd, numpy as np, statsmodels.api as sm, \
    statsmodels.formula.api as smf, copy, os, re
from config import GeneralSettings, Filing, FileNames, FilePaths, MiscSettings, PunishSettings
from preprocessing import (
    load_or_build_cleaned_dataframe,
    load_analysis_dataframe,
    _save_analysis_dataframe_to_processed_folder,
)


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


def run_pooled_ols_planned_contrasts(dataframe: pd.DataFrame, dv_column_name: str, group_column_name: str = "vignette_condition_position_1", covariance_type: str | None = None) -> pd.DataFrame:
    """
    Fits the preregistered pooled OLS / one-way ANOVA style model and returns the two planned contrasts.

    Arguments:
        • dataframe: pd.DataFrame
            The dataframe containing the dependent variable and the three-level group factor.
        • dv_column_name: str
            The dependent variable column to analyze.
        • group_column_name: str = "vignette_condition_position_1"
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
        general_settings=general_settings,
        file_name_key="group_summaries",
        force_rebuild=force_rebuild,
    )
    if group_summaries_extraction["success"]:
        group_summaries_dataframe: pd.DataFrame = group_summaries_extraction["dataframe"]
        return group_summaries_dataframe
    if group_summaries_extraction["error"]:
        raise Exception(group_summaries_extraction["message"])

    "Load or rebuild preprocessed dataframe."
    cleaned_dataframe = load_or_build_cleaned_dataframe(
        general_settings=general_settings,
        force_rebuild=False,
    )

    punishment_analysis_mode = str(
        general_settings.get("punish", {}).get("analysis_mode", "raw_nonparametric")
    ).strip().lower()

    def summarize_numeric_values(values: pd.Series | np.ndarray) -> dict[str, float]:
        """
        Compute n, mean, median, and std for one numeric vector.
        """
        numeric_values = pd.to_numeric(values, errors="coerce")
        numeric_values = pd.Series(numeric_values).dropna()

        if numeric_values.shape[0] == 0:
            return {
                "n": 0,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
            }

        return {
            "n": int(numeric_values.shape[0]),
            "mean": float(numeric_values.mean()),
            "median": float(numeric_values.median()),
            "std": float(numeric_values.std(ddof=1)),
        }

    def transform_punishment_values_if_requested(values: pd.Series | np.ndarray, dv_name: str) -> pd.Series:
        """
        Transform punishment values for descriptive summaries when requested.

        Notes:
            • For blame and wrong, values are always returned unchanged.
            • For punish:
                - raw_nonparametric  -> raw values
                - raw_parametric     -> raw values
                - log1p_parametric   -> log(1 + x)
        """
        numeric_values = pd.to_numeric(values, errors="coerce")
        numeric_values = pd.Series(numeric_values).dropna()

        if dv_name != "punish":
            return numeric_values

        if punishment_analysis_mode == "log1p_parametric":
            if (numeric_values < 0).any():
                minimum_value = float(numeric_values.min())
                raise ValueError(
                    "compute_group_summaries encountered negative raw punishment values while "
                    f"punishment_analysis_mode='log1p_parametric'. Minimum offending value: {minimum_value:.4f}"
                )
            return np.log1p(numeric_values)

        return numeric_values

    def build_log1p_punishment_delta(
        subset_dataframe: pd.DataFrame,
        left_column_name: str,
        right_column_name: str,
    ) -> pd.Series:
        """
        Build paired log(1 + x) punishment deltas for descriptive summaries.
        """
        left_values = pd.to_numeric(subset_dataframe[left_column_name], errors="coerce")
        right_values = pd.to_numeric(subset_dataframe[right_column_name], errors="coerce")

        valid_mask = (~left_values.isna()) & (~right_values.isna())
        transformed_delta_values = np.log1p(left_values[valid_mask].to_numpy(dtype=float)) - np.log1p(
            right_values[valid_mask].to_numpy(dtype=float)
        )

        return pd.Series(transformed_delta_values)

    def build_log1p_punishment_min_delta(subset_dataframe: pd.DataFrame) -> pd.Series:
        """
        Build paired log(1 + x) MIN(CH, DIV) - CC punishment deltas for descriptive summaries.
        """
        ch_values = pd.to_numeric(subset_dataframe["distal_punish_ch"], errors="coerce")
        div_values = pd.to_numeric(subset_dataframe["distal_punish_div"], errors="coerce")
        cc_values = pd.to_numeric(subset_dataframe["distal_punish_cc"], errors="coerce")

        valid_mask = (~ch_values.isna()) & (~div_values.isna()) & (~cc_values.isna())

        transformed_delta_values = np.minimum(
            np.log1p(ch_values[valid_mask].to_numpy(dtype=float)),
            np.log1p(div_values[valid_mask].to_numpy(dtype=float)),
        ) - np.log1p(cc_values[valid_mask].to_numpy(dtype=float))

        return pd.Series(transformed_delta_values)

    dataframes = {}

    for inclusion_filter_value, analysis_dataframe in [
        ("included_only", cleaned_dataframe.loc[cleaned_dataframe["included"] == True].copy()),  # noqa: E712
        ("all_finishers", cleaned_dataframe.copy()),
    ]:
        summary_rows = []

        distal_condition_columns_by_dv = {
            "blame": ("distal_blame_cc", "distal_blame_ch", "distal_blame_div"),
            "wrong": ("distal_wrong_cc", "distal_wrong_ch", "distal_wrong_div"),
            "punish": ("distal_punish_cc", "distal_punish_ch", "distal_punish_div"),
        }

        proximate_condition_columns_by_dv = {
            "blame": ("proximate_blame_cc", "proximate_blame_ch"),
            "wrong": ("proximate_wrong_cc", "proximate_wrong_ch"),
            "punish": ("proximate_punish_cc", "proximate_punish_ch"),
        }

        first_vignette_columns_by_dv = {
            "blame": "first_vignette_distal_blame",
            "wrong": "first_vignette_distal_wrong",
            "punish": "first_vignette_distal_punish",
        }

        for story_condition_value in ["pooled", "firework", "trolley"]:
            for load_condition_value in ["pooled", "high", "low"]:
                subset_dataframe = analysis_dataframe.copy()

                if story_condition_value != "pooled":
                    subset_dataframe = subset_dataframe.loc[
                        subset_dataframe["story_condition"] == story_condition_value
                    ].copy()

                if load_condition_value != "pooled":
                    subset_dataframe = subset_dataframe.loc[
                        subset_dataframe["load_condition"] == load_condition_value
                    ].copy()

                "Within-subject all-vignettes summaries for the distal agent."
                for dv_name, (column_cc, column_ch, column_div) in distal_condition_columns_by_dv.items():
                    for condition_code, column_name in [
                        ("CC", column_cc),
                        ("CH", column_ch),
                        ("DIV", column_div),
                    ]:
                        transformed_values = transform_punishment_values_if_requested(
                            values=subset_dataframe[column_name],
                            dv_name=dv_name,
                        )
                        summary_rows.append(
                            {
                                "inclusion_filter": inclusion_filter_value,
                                "story_condition": story_condition_value,
                                "load_condition": load_condition_value,
                                "analysis_scope": "within_subjects_all_vignettes",
                                "agent_role": "distal",
                                "dv": dv_name,
                                "condition": condition_code,
                                **summarize_numeric_values(transformed_values),
                            }
                        )

                "Within-subject all-vignettes summaries for the proximate agent."
                for dv_name, (column_cc, column_ch) in proximate_condition_columns_by_dv.items():
                    for condition_code, column_name in [
                        ("CC", column_cc),
                        ("CH", column_ch),
                    ]:
                        transformed_values = transform_punishment_values_if_requested(
                            values=subset_dataframe[column_name],
                            dv_name=dv_name,
                        )
                        summary_rows.append(
                            {
                                "inclusion_filter": inclusion_filter_value,
                                "story_condition": story_condition_value,
                                "load_condition": load_condition_value,
                                "analysis_scope": "within_subjects_all_vignettes",
                                "agent_role": "proximate",
                                "dv": dv_name,
                                "condition": condition_code,
                                **summarize_numeric_values(transformed_values),
                            }
                        )

                "Between-subject first-vignette summaries for the distal agent only."
                for dv_name, first_column_name in first_vignette_columns_by_dv.items():
                    for condition_code in ["CC", "CH", "DIV"]:
                        subset_first_vignette = subset_dataframe.loc[
                            subset_dataframe["vignette_condition_position_1"] == condition_code
                        ].copy()

                        transformed_values = transform_punishment_values_if_requested(
                            values=subset_first_vignette[first_column_name],
                            dv_name=dv_name,
                        )
                        summary_rows.append(
                            {
                                "inclusion_filter": inclusion_filter_value,
                                "story_condition": story_condition_value,
                                "load_condition": load_condition_value,
                                "analysis_scope": "between_subjects_first_vignette",
                                "agent_role": "distal",
                                "dv": dv_name,
                                "condition": condition_code,
                                **summarize_numeric_values(transformed_values),
                            }
                        )

                "Within-subject delta summaries."
                delta_column_map = {
                    "blame": {
                        "CH - CC": "distal_blame_ch_minus_cc",
                        "DIV - CC": "distal_blame_div_minus_cc",
                        "MIN(CH, DIV) - CC": "distal_blame_min_ch_div_minus_cc",
                        "CC - CH": "proximate_blame_cc_minus_ch",
                    },
                    "wrong": {
                        "CH - CC": "distal_wrong_ch_minus_cc",
                        "DIV - CC": "distal_wrong_div_minus_cc",
                        "MIN(CH, DIV) - CC": "distal_wrong_min_ch_div_minus_cc",
                        "CC - CH": "proximate_wrong_cc_minus_ch",
                    },
                    "punish": {
                        "CH - CC": "distal_punish_ch_minus_cc",
                        "DIV - CC": "distal_punish_div_minus_cc",
                        "MIN(CH, DIV) - CC": "distal_punish_min_ch_div_minus_cc",
                        "CC - CH": "proximate_punish_cc_minus_ch",
                    },
                }

                for dv_name, contrast_map in delta_column_map.items():
                    for contrast_label, column_name in contrast_map.items():
                        if dv_name == "punish" and punishment_analysis_mode == "log1p_parametric":
                            if contrast_label == "CH - CC":
                                transformed_values = build_log1p_punishment_delta(
                                    subset_dataframe=subset_dataframe,
                                    left_column_name="distal_punish_ch",
                                    right_column_name="distal_punish_cc",
                                )
                            elif contrast_label == "DIV - CC":
                                transformed_values = build_log1p_punishment_delta(
                                    subset_dataframe=subset_dataframe,
                                    left_column_name="distal_punish_div",
                                    right_column_name="distal_punish_cc",
                                )
                            elif contrast_label == "MIN(CH, DIV) - CC":
                                transformed_values = build_log1p_punishment_min_delta(
                                    subset_dataframe=subset_dataframe,
                                )
                            elif contrast_label == "CC - CH":
                                transformed_values = build_log1p_punishment_delta(
                                    subset_dataframe=subset_dataframe,
                                    left_column_name="proximate_punish_cc",
                                    right_column_name="proximate_punish_ch",
                                )
                            else:
                                transformed_values = pd.Series(dtype=float)
                        else:
                            transformed_values = pd.to_numeric(
                                subset_dataframe[column_name],
                                errors="coerce",
                            )

                        summary_rows.append(
                            {
                                "inclusion_filter": inclusion_filter_value,
                                "story_condition": story_condition_value,
                                "load_condition": load_condition_value,
                                "analysis_scope": "within_subjects_deltas",
                                "agent_role": ("proximate" if contrast_label == "CC - CH" else "distal"),
                                "dv": dv_name,
                                "condition": contrast_label,
                                **summarize_numeric_values(transformed_values),
                            }
                        )

        dataframes[inclusion_filter_value] = pd.DataFrame(summary_rows)

    group_summaries_dataframe = pd.concat(
        list(dataframes.values()),
        axis=0,
        ignore_index=True,
    )

    _save_analysis_dataframe_to_processed_folder(
        dataframe_to_save=group_summaries_dataframe,
        general_settings=general_settings,
        file_name_key="group_summaries",
    )

    return group_summaries_dataframe


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

        add_counts_for_subset(analysis_dataframe, "pooled")
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
    Computes basic correlations among Clark blame/wrong/punishment.

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
        first_blame = pd.to_numeric(analysis_dataframe["first_vignette_distal_blame"], errors="coerce")
        first_wrong = pd.to_numeric(analysis_dataframe["first_vignette_distal_wrong"], errors="coerce")
        first_punish = pd.to_numeric(analysis_dataframe["first_vignette_distal_punish"], errors="coerce")

        for label_a, series_a in [("blame", first_blame), ("wrong", first_wrong), ("punish", first_punish)]:
            for label_b, series_b in [("blame", first_blame), ("wrong", first_wrong), ("punish", first_punish)]:
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
        analysis_dataframe["distal_blame_mean"] = analysis_dataframe[["distal_blame_cc", "distal_blame_ch", "distal_blame_div"]].mean(axis=1, skipna=True)
        analysis_dataframe["distal_wrong_mean"] = analysis_dataframe[["distal_wrong_cc", "distal_wrong_ch", "distal_wrong_div"]].mean(axis=1, skipna=True)
        analysis_dataframe["distal_punish_mean"] = analysis_dataframe[["distal_punish_cc", "distal_punish_ch", "distal_punish_div"]].mean(axis=1, skipna=True)

        mean_blame = pd.to_numeric(analysis_dataframe["distal_blame_mean"], errors="coerce")
        mean_wrong = pd.to_numeric(analysis_dataframe["distal_wrong_mean"], errors="coerce")
        mean_punish = pd.to_numeric(analysis_dataframe["distal_punish_mean"], errors="coerce")

        for label_a, series_a in [("blame_mean", mean_blame), ("wrong_mean", mean_wrong), ("punish_mean", mean_punish)]:
            for label_b, series_b in [("blame_mean", mean_blame), ("wrong_mean", mean_wrong), ("punish_mean", mean_punish)]:
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

        for outcome_column in ["distal_blame_ch_minus_cc", "distal_blame_div_minus_cc", "distal_blame_min_ch_div_minus_cc"]:
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

        for dv_prefix in ["distal_blame", "distal_wrong", "distal_punish"]:
            column_cc = f"{dv_prefix}_cc"
            column_ch = f"{dv_prefix}_ch"

            subset_cc_first = analysis_dataframe[analysis_dataframe["vignette_condition_position_1"] == "CC"].copy()
            subset_ch_first = analysis_dataframe[analysis_dataframe["vignette_condition_position_1"] == "CH"].copy()

            "CH rating: CH-first vs CC-first"
            test_ch_rating = run_welch_t_test_between_groups(
                pd.concat([subset_ch_first, subset_cc_first], axis=0),
                dv_column_name=column_ch,
                group_column_name="vignette_condition_position_1",
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
                group_column_name="vignette_condition_position_1",
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
            ("CH_vs_CC", "twoafc_ch_vs_cc", "distal_blame_ch_minus_cc", "CH", "CC"),
            ("DIV_vs_CC", "twoafc_div_vs_cc", "distal_blame_div_minus_cc", "DIV", "CC"),
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


def run_confirmatory_and_exploratory_tests(general_settings: GeneralSettings, confirmatory_pooled_ols_covariance_type: str | None = None, 
                                           cleaned_dataframe: pd.DataFrame | None = None, force_rebuild: bool | None = None) -> pd.DataFrame:
    """
    Runs the primary confirmatory tests plus a structured set of exploratory tests.

    Confirmatory (preregistered):
        • Between-subjects, included only:
            H1: first_vignette_distal_blame CH vs CC (two-sided)
            H2: first_vignette_distal_blame DIV vs CC (two-sided)
        • Holm correction across these two p-values.

    Returns:
        • pd.DataFrame
            - Long-format table of test results. One row per test. Columns:
                analysis_family, analysis_mode, test_type, transformation,
                location_statistic_reported, inclusion_filter, story_condition,
                load_condition, analysis_scope, dv, agent_role, contrast_type,
                group_a, group_b, n_a, n_b, mean_a, mean_b, median_a, median_b,
                mean_difference_a_minus_b, median_difference_a_minus_b,
                ci95_lower, ci95_upper, t_statistic, df,
                p_value_two_tailed, p_value_one_tailed,
                effect_size_name, effect_size, p_value_holm, notes.
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    "Load and return dataframe if one already exists and not directed to rebuild."
    tests_extraction = load_analysis_dataframe(
        general_settings=general_settings,
        file_name_key="tests",
        force_rebuild=force_rebuild,
    )
    if tests_extraction["success"]:
        tests_dataframe: pd.DataFrame = tests_extraction["dataframe"]
        return tests_dataframe
    if tests_extraction["error"]:
        raise Exception(tests_extraction["message"])

    "Load or rebuild preprocessed dataframe."
    if cleaned_dataframe is None:
        cleaned_dataframe = load_or_build_cleaned_dataframe(
            general_settings=general_settings,
            force_rebuild=False,
        )
    else:
        cleaned_dataframe = cleaned_dataframe.copy()

    "=============================="
    "Global settings and defaults."
    "=============================="
    confirmatory_between_subjects_method_normalized = str(
        general_settings["misc"]["confirmatory_between_subjects_method"]
    ).strip().lower()

    one_tailed_results_are_primary = bool(general_settings["misc"]["one_tailed"])

    punishment_settings: dict[str, Any] = dict(general_settings.get("punish", {}))
    punishment_analysis_mode = str(
        punishment_settings.get("analysis_mode", "raw_nonparametric")
    ).strip().lower()
    punishment_bootstrap_iterations = int(
        punishment_settings.get("bootstrap_iterations", 5000)
    )
    punishment_random_seed = int(
        punishment_settings.get("random_seed", 2026)
    )

    valid_punishment_analysis_modes = {
        "raw_nonparametric",
        "raw_parametric",
        "log1p_parametric",
    }
    if punishment_analysis_mode not in valid_punishment_analysis_modes:
        raise ValueError(
            "general_settings['punish']['analysis_mode'] must be one of "
            f"{sorted(valid_punishment_analysis_modes)}, not {punishment_analysis_mode!r}."
        )

    random_number_generator = np.random.default_rng(punishment_random_seed)

    "=================================="
    "Human-readable constants and maps."
    "=================================="
    participants_sort_map = {
        "included_only": 0,
        "all_finishers": 1,
    }
    story_condition_sort_map = {
        "pooled": 0,
        "firework": 1,
        "trolley": 2,
    }
    load_condition_sort_map = {
        "pooled": 0,
        "high": 1,
        "low": 2,
    }
    design_sort_map = {
        "between_subjects_first_vignette": 0,
        "within_subjects_all_vignettes": 1,
    }
    dv_sort_map = {
        "blame": 0,
        "wrong": 1,
        "punish": 2,
    }
    agent_role_sort_map = {
        "distal": 0,
        "proximate": 1,
    }
    contrast_type_sort_map = {
        "CH - CC": 0,
        "DIV - CC": 1,
        "CH - DIV": 2,
        "MIN(CH, DIV) - CC": 3,
        "CC - CH": 4,
    }
    dv_plain_language_map = {
        "blame": "blameworthiness",
        "wrong": "wrongness",
        "punish": "punishment",
    }

    "===================================="
    "Nested helpers kept local on purpose."
    "===================================="
    def coerce_numeric_array(series_or_array) -> np.ndarray:
        numeric_values = pd.to_numeric(series_or_array, errors="coerce")
        numeric_values = np.asarray(numeric_values, dtype=float)
        numeric_values = numeric_values[~np.isnan(numeric_values)]
        return numeric_values

    def compute_one_tailed_p_value_from_two_tailed_p_value(
        p_value_two_tailed: float,
        test_statistic: float,
    ) -> float:
        """
        Computes a one-tailed p-value in the positive direction.

        If the observed effect is in the predicted positive direction:
            p_one_tailed = p_two_tailed / 2

        If the observed effect is in the opposite direction:
            p_one_tailed = 1 - p_two_tailed / 2
        """
        if pd.isna(p_value_two_tailed) or pd.isna(test_statistic):
            return np.nan

        p_value_two_tailed = float(p_value_two_tailed)
        test_statistic = float(test_statistic)

        if test_statistic >= 0:
            return float(p_value_two_tailed / 2)

        return float(1 - p_value_two_tailed / 2)

    def filter_dataframe_by_story_condition_and_load_condition(
        analysis_dataframe: pd.DataFrame,
        story_condition_value: str,
        load_condition_value: str,
    ) -> pd.DataFrame:
        filtered_dataframe = analysis_dataframe.copy()

        if story_condition_value != "pooled":
            filtered_dataframe = filtered_dataframe.loc[
                filtered_dataframe["story_condition"] == story_condition_value
            ].copy()

        if load_condition_value != "pooled":
            filtered_dataframe = filtered_dataframe.loc[
                filtered_dataframe["load_condition"] == load_condition_value
            ].copy()

        return filtered_dataframe

    def resolve_analysis_mode_for_dependent_variable(dv_key: str) -> str:
        if dv_key == "punish":
            return punishment_analysis_mode
        return "raw_parametric"

    def resolve_location_statistic_reported(analysis_mode_value: str) -> str:
        if analysis_mode_value == "raw_nonparametric":
            return "median_difference"
        return "mean_difference"

    def bootstrap_percentile_ci_for_independent_statistic(
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        statistic_function,
    ) -> tuple[float, float]:
        bootstrap_statistics = []

        for _ in range(punishment_bootstrap_iterations):
            resampled_sample_a = random_number_generator.choice(
                sample_a, size=sample_a.shape[0], replace=True
            )
            resampled_sample_b = random_number_generator.choice(
                sample_b, size=sample_b.shape[0], replace=True
            )
            bootstrap_statistics.append(
                float(statistic_function(resampled_sample_a, resampled_sample_b))
            )

        ci95_lower, ci95_upper = np.percentile(bootstrap_statistics, [2.5, 97.5])
        return float(ci95_lower), float(ci95_upper)

    def bootstrap_percentile_ci_for_one_sample_statistic(
        sample_values: np.ndarray,
        statistic_function,
    ) -> tuple[float, float]:
        bootstrap_statistics = []

        for _ in range(punishment_bootstrap_iterations):
            resampled_sample_values = random_number_generator.choice(
                sample_values, size=sample_values.shape[0], replace=True
            )
            bootstrap_statistics.append(
                float(statistic_function(resampled_sample_values))
            )

        ci95_lower, ci95_upper = np.percentile(bootstrap_statistics, [2.5, 97.5])
        return float(ci95_lower), float(ci95_upper)

    def median_difference_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        return float(np.median(sample_a) - np.median(sample_b))

    def median_statistic(sample_values: np.ndarray) -> float:
        return float(np.median(sample_values))

    def mann_whitney_rank_biserial(u_statistic: float, n_a: int, n_b: int) -> float:
        if n_a <= 0 or n_b <= 0:
            return np.nan
        return float((2 * u_statistic) / (n_a * n_b) - 1)

    def matched_rank_biserial_from_delta_values(delta_values: np.ndarray) -> float:
        nonzero_delta_values = np.asarray(delta_values, dtype=float)
        nonzero_delta_values = nonzero_delta_values[~np.isnan(nonzero_delta_values)]
        nonzero_delta_values = nonzero_delta_values[nonzero_delta_values != 0]

        if nonzero_delta_values.shape[0] == 0:
            return np.nan

        absolute_ranks = stats.rankdata(np.abs(nonzero_delta_values), method="average")
        positive_rank_sum = float(np.sum(absolute_ranks[nonzero_delta_values > 0]))
        negative_rank_sum = float(np.sum(absolute_ranks[nonzero_delta_values < 0]))

        denominator = positive_rank_sum + negative_rank_sum
        if denominator == 0:
            return np.nan

        return float((positive_rank_sum - negative_rank_sum) / denominator)

    def transform_numeric_values_if_requested(
        numeric_values: np.ndarray,
        analysis_mode_value: str,
    ) -> tuple[np.ndarray, str]:
        numeric_values = np.asarray(numeric_values, dtype=float)
        numeric_values = numeric_values[~np.isnan(numeric_values)]

        if analysis_mode_value == "log1p_parametric":
            if np.any(numeric_values < 0):
                minimum_value = float(np.min(numeric_values))
                raise ValueError(
                    "log1p_parametric was asked to transform negative raw punishment values. "
                    f"Minimum offending value: {minimum_value:.4f}"
                )
            return np.log1p(numeric_values), "log1p"

        return numeric_values.copy(), "none"

    def build_plain_language_note(
        analysis_family: str,
        design: str,
        dv_key: str,
        agent_role: str,
        contrast_type: str,
        inclusion_filter: str,
        story_condition: str,
        load_condition: str,
        analysis_mode: str,
        transformation: str,
        location_statistic_reported: str,
        extra_detail: str = "",
    ) -> str:
        dv_plain_language = dv_plain_language_map[dv_key]
        design_plain_language = design.replace("_", " ")
        inclusion_filter_plain_language = inclusion_filter.replace("_", " ")
        story_plain_language = story_condition
        load_plain_language = load_condition

        if agent_role == "distal":
            agent_plain_language = "distal agent Clark"
        else:
            agent_plain_language = "proximate agent Bill or the proximate node"

        if transformation == "none":
            transformation_plain_language = "raw untransformed values"
        else:
            transformation_plain_language = "log(1 + x)-transformed values"

        if location_statistic_reported == "median_difference":
            location_plain_language = "median difference"
        else:
            location_plain_language = "mean difference"

        note_string = (
            f"{analysis_family.capitalize()} {design_plain_language} comparison on {dv_plain_language} "
            f"for the {agent_plain_language}. Contrast type: {contrast_type}. Participants: {inclusion_filter_plain_language}. "
            f"Story family: {story_plain_language}. Load condition: {load_plain_language}. "
            f"Analysis mode: {analysis_mode}. Transformation: {transformation_plain_language}. "
            f"The location statistic represented by the contrast columns is the {location_plain_language}. "
            f"p_value_one_tailed assumes that the predicted direction is positive for the contrast named in contrast_type."
        )

        if extra_detail != "":
            note_string += " " + extra_detail

        return note_string

    def make_standard_test_row(
        analysis_family: str,
        analysis_mode: str,
        test_type: str,
        transformation: str,
        location_statistic_reported: str,
        inclusion_filter: str,
        story_condition: str,
        load_condition: str,
        design: str,
        dv: str,
        agent_role: str,
        contrast_type: str,
        group_a: str,
        group_b: str,
        n_a: int,
        n_b: int,
        mean_a: float,
        mean_b: float,
        median_a: float,
        median_b: float,
        mean_difference_a_minus_b: float,
        median_difference_a_minus_b: float,
        ci95_lower: float,
        ci95_upper: float,
        t_statistic: float,
        df: float,
        p_value_two_tailed: float,
        p_value_one_tailed: float,
        effect_size_name: str,
        effect_size: float,
        p_value_holm: float,
        notes: str,
    ) -> dict[str, Any]:
        return {
            "analysis_family": analysis_family,
            "analysis_mode": analysis_mode,
            "test_type": test_type,
            "transformation": transformation,
            "location_statistic_reported": location_statistic_reported,
            "inclusion_filter": inclusion_filter,
            "story_condition": story_condition,
            "load_condition": load_condition,
            "analysis_scope": design,
            "dv": dv,
            "agent_role": agent_role,
            "contrast_type": contrast_type,
            "group_a": group_a,
            "group_b": group_b,
            "n_a": int(n_a),
            "n_b": int(n_b),
            "mean_a": float(mean_a) if not pd.isna(mean_a) else np.nan,
            "mean_b": float(mean_b) if not pd.isna(mean_b) else np.nan,
            "median_a": float(median_a) if not pd.isna(median_a) else np.nan,
            "median_b": float(median_b) if not pd.isna(median_b) else np.nan,
            "mean_difference_a_minus_b": float(mean_difference_a_minus_b) if not pd.isna(mean_difference_a_minus_b) else np.nan,
            "median_difference_a_minus_b": float(median_difference_a_minus_b) if not pd.isna(median_difference_a_minus_b) else np.nan,
            "ci95_lower": float(ci95_lower) if not pd.isna(ci95_lower) else np.nan,
            "ci95_upper": float(ci95_upper) if not pd.isna(ci95_upper) else np.nan,
            "t_statistic": float(t_statistic) if not pd.isna(t_statistic) else np.nan,
            "df": float(df) if not pd.isna(df) else np.nan,
            "p_value_two_tailed": float(p_value_two_tailed) if not pd.isna(p_value_two_tailed) else np.nan,
            "p_value_one_tailed": float(p_value_one_tailed) if not pd.isna(p_value_one_tailed) else np.nan,
            "effect_size_name": effect_size_name,
            "effect_size": float(effect_size) if not pd.isna(effect_size) else np.nan,
            "p_value_holm": float(p_value_holm) if not pd.isna(p_value_holm) else np.nan,
            "notes": notes,
        }

    def run_independent_samples_test(
        sample_a_raw: np.ndarray,
        sample_b_raw: np.ndarray,
        analysis_mode: str,
    ) -> dict[str, Any]:
        sample_a_raw = coerce_numeric_array(sample_a_raw)
        sample_b_raw = coerce_numeric_array(sample_b_raw)

        if sample_a_raw.shape[0] <= 1 or sample_b_raw.shape[0] <= 1:
            return {
                "analysis_mode": analysis_mode,
                "test_type": "insufficient_data",
                "transformation": "none",
                "location_statistic_reported": resolve_location_statistic_reported(analysis_mode),
                "n_a": int(sample_a_raw.shape[0]),
                "n_b": int(sample_b_raw.shape[0]),
                "mean_a": np.nan,
                "mean_b": np.nan,
                "median_a": np.nan,
                "median_b": np.nan,
                "mean_difference_a_minus_b": np.nan,
                "median_difference_a_minus_b": np.nan,
                "ci95_lower": np.nan,
                "ci95_upper": np.nan,
                "t_statistic": np.nan,
                "df": np.nan,
                "p_value_two_tailed": np.nan,
                "p_value_one_tailed": np.nan,
                "effect_size_name": "NA",
                "effect_size": np.nan,
            }

        if analysis_mode == "raw_nonparametric":
            mann_whitney_result_two_tailed = stats.mannwhitneyu(
                sample_a_raw,
                sample_b_raw,
                alternative="two-sided",
                method="auto",
            )
            mann_whitney_result_one_tailed = stats.mannwhitneyu(
                sample_a_raw,
                sample_b_raw,
                alternative="greater",
                method="auto",
            )

            ci95_lower, ci95_upper = bootstrap_percentile_ci_for_independent_statistic(
                sample_a=sample_a_raw,
                sample_b=sample_b_raw,
                statistic_function=median_difference_statistic,
            )

            return {
                "analysis_mode": analysis_mode,
                "test_type": "mann_whitney_u",
                "transformation": "none",
                "location_statistic_reported": "median_difference",
                "n_a": int(sample_a_raw.shape[0]),
                "n_b": int(sample_b_raw.shape[0]),
                "mean_a": float(np.mean(sample_a_raw)),
                "mean_b": float(np.mean(sample_b_raw)),
                "median_a": float(np.median(sample_a_raw)),
                "median_b": float(np.median(sample_b_raw)),
                "mean_difference_a_minus_b": float(np.mean(sample_a_raw) - np.mean(sample_b_raw)),
                "median_difference_a_minus_b": float(np.median(sample_a_raw) - np.median(sample_b_raw)),
                "ci95_lower": ci95_lower,
                "ci95_upper": ci95_upper,
                "t_statistic": float(mann_whitney_result_two_tailed.statistic),
                "df": np.nan,
                "p_value_two_tailed": float(mann_whitney_result_two_tailed.pvalue),
                "p_value_one_tailed": float(mann_whitney_result_one_tailed.pvalue),
                "effect_size_name": "rank_biserial",
                "effect_size": mann_whitney_rank_biserial(
                    u_statistic=float(mann_whitney_result_two_tailed.statistic),
                    n_a=int(sample_a_raw.shape[0]),
                    n_b=int(sample_b_raw.shape[0]),
                ),
            }

        sample_a_analysis_scale, transformation_label = transform_numeric_values_if_requested(
            sample_a_raw,
            analysis_mode,
        )
        sample_b_analysis_scale, _ = transform_numeric_values_if_requested(
            sample_b_raw,
            analysis_mode,
        )

        welch_result_two_tailed = stats.ttest_ind(
            sample_a_analysis_scale,
            sample_b_analysis_scale,
            equal_var=False,
            alternative="two-sided",
        )
        welch_result_one_tailed = stats.ttest_ind(
            sample_a_analysis_scale,
            sample_b_analysis_scale,
            equal_var=False,
            alternative="greater",
        )

        mean_difference_value, ci95_lower, ci95_upper, welch_df = compute_welch_mean_difference_ci(
            sample_a_analysis_scale,
            sample_b_analysis_scale,
        )

        return {
            "analysis_mode": analysis_mode,
            "test_type": "welch_t_test_independent",
            "transformation": transformation_label,
            "location_statistic_reported": "mean_difference",
            "n_a": int(sample_a_analysis_scale.shape[0]),
            "n_b": int(sample_b_analysis_scale.shape[0]),
            "mean_a": float(np.mean(sample_a_analysis_scale)),
            "mean_b": float(np.mean(sample_b_analysis_scale)),
            "median_a": float(np.median(sample_a_analysis_scale)),
            "median_b": float(np.median(sample_b_analysis_scale)),
            "mean_difference_a_minus_b": mean_difference_value,
            "median_difference_a_minus_b": float(np.median(sample_a_analysis_scale) - np.median(sample_b_analysis_scale)),
            "ci95_lower": ci95_lower,
            "ci95_upper": ci95_upper,
            "t_statistic": float(welch_result_two_tailed.statistic),
            "df": float(welch_df),
            "p_value_two_tailed": float(welch_result_two_tailed.pvalue),
            "p_value_one_tailed": float(welch_result_one_tailed.pvalue),
            "effect_size_name": "hedges_g",
            "effect_size": float(
                hedges_g_for_two_independent_samples(sample_a_analysis_scale, sample_b_analysis_scale)
            ),
        }

    def run_paired_samples_test(
        sample_a_raw: np.ndarray,
        sample_b_raw: np.ndarray,
        analysis_mode: str,
    ) -> dict[str, Any]:
        sample_a_raw = coerce_numeric_array(sample_a_raw)
        sample_b_raw = coerce_numeric_array(sample_b_raw)

        if sample_a_raw.shape[0] != sample_b_raw.shape[0]:
            minimum_n = min(sample_a_raw.shape[0], sample_b_raw.shape[0])
            sample_a_raw = sample_a_raw[:minimum_n]
            sample_b_raw = sample_b_raw[:minimum_n]

        if sample_a_raw.shape[0] <= 1:
            return {
                "analysis_mode": analysis_mode,
                "test_type": "insufficient_data",
                "transformation": "none",
                "location_statistic_reported": resolve_location_statistic_reported(analysis_mode),
                "n_a": int(sample_a_raw.shape[0]),
                "n_b": int(sample_a_raw.shape[0]),
                "mean_a": np.nan,
                "mean_b": np.nan,
                "median_a": np.nan,
                "median_b": np.nan,
                "mean_difference_a_minus_b": np.nan,
                "median_difference_a_minus_b": np.nan,
                "ci95_lower": np.nan,
                "ci95_upper": np.nan,
                "t_statistic": np.nan,
                "df": np.nan,
                "p_value_two_tailed": np.nan,
                "p_value_one_tailed": np.nan,
                "effect_size_name": "NA",
                "effect_size": np.nan,
            }

        if analysis_mode == "raw_nonparametric":
            delta_values_raw = sample_a_raw - sample_b_raw
            nonzero_delta_values = delta_values_raw[delta_values_raw != 0]

            if nonzero_delta_values.shape[0] == 0:
                wilcoxon_statistic = 0.0
                p_value_two_tailed = 1.0
                p_value_one_tailed = 1.0
            else:
                wilcoxon_result_two_tailed = stats.wilcoxon(
                    nonzero_delta_values,
                    alternative="two-sided",
                    zero_method="wilcox",
                    correction=False,
                    mode="auto",
                )
                wilcoxon_result_one_tailed = stats.wilcoxon(
                    nonzero_delta_values,
                    alternative="greater",
                    zero_method="wilcox",
                    correction=False,
                    mode="auto",
                )
                wilcoxon_statistic = float(wilcoxon_result_two_tailed.statistic)
                p_value_two_tailed = float(wilcoxon_result_two_tailed.pvalue)
                p_value_one_tailed = float(wilcoxon_result_one_tailed.pvalue)

            ci95_lower, ci95_upper = bootstrap_percentile_ci_for_one_sample_statistic(
                sample_values=delta_values_raw,
                statistic_function=median_statistic,
            )

            return {
                "analysis_mode": analysis_mode,
                "test_type": "wilcoxon_signed_rank",
                "transformation": "none",
                "location_statistic_reported": "median_difference",
                "n_a": int(sample_a_raw.shape[0]),
                "n_b": int(sample_b_raw.shape[0]),
                "mean_a": float(np.mean(sample_a_raw)),
                "mean_b": float(np.mean(sample_b_raw)),
                "median_a": float(np.median(sample_a_raw)),
                "median_b": float(np.median(sample_b_raw)),
                "mean_difference_a_minus_b": float(np.mean(delta_values_raw)),
                "median_difference_a_minus_b": float(np.median(delta_values_raw)),
                "ci95_lower": ci95_lower,
                "ci95_upper": ci95_upper,
                "t_statistic": wilcoxon_statistic,
                "df": np.nan,
                "p_value_two_tailed": p_value_two_tailed,
                "p_value_one_tailed": p_value_one_tailed,
                "effect_size_name": "matched_rank_biserial",
                "effect_size": matched_rank_biserial_from_delta_values(delta_values_raw),
            }

        sample_a_analysis_scale, transformation_label = transform_numeric_values_if_requested(
            sample_a_raw,
            analysis_mode,
        )
        sample_b_analysis_scale, _ = transform_numeric_values_if_requested(
            sample_b_raw,
            analysis_mode,
        )

        delta_values_analysis_scale = sample_a_analysis_scale - sample_b_analysis_scale

        paired_result_two_tailed = stats.ttest_1samp(
            delta_values_analysis_scale,
            popmean=0.0,
            alternative="two-sided",
        )
        paired_result_one_tailed = stats.ttest_1samp(
            delta_values_analysis_scale,
            popmean=0.0,
            alternative="greater",
        )

        mean_delta_value = float(np.mean(delta_values_analysis_scale))
        standard_deviation_delta = float(np.std(delta_values_analysis_scale, ddof=1))
        standard_error_delta = standard_deviation_delta / np.sqrt(delta_values_analysis_scale.shape[0])
        df_value = float(delta_values_analysis_scale.shape[0] - 1)

        t_critical = stats.t.ppf(0.975, df=df_value) if delta_values_analysis_scale.shape[0] > 1 else np.nan
        ci95_lower = float(mean_delta_value - t_critical * standard_error_delta) if delta_values_analysis_scale.shape[0] > 1 else np.nan
        ci95_upper = float(mean_delta_value + t_critical * standard_error_delta) if delta_values_analysis_scale.shape[0] > 1 else np.nan

        effect_size_value = np.nan
        if standard_deviation_delta > 0:
            effect_size_value = float(mean_delta_value / standard_deviation_delta)

        return {
            "analysis_mode": analysis_mode,
            "test_type": "paired_t_test",
            "transformation": transformation_label,
            "location_statistic_reported": "mean_difference",
            "n_a": int(sample_a_analysis_scale.shape[0]),
            "n_b": int(sample_b_analysis_scale.shape[0]),
            "mean_a": float(np.mean(sample_a_analysis_scale)),
            "mean_b": float(np.mean(sample_b_analysis_scale)),
            "median_a": float(np.median(sample_a_analysis_scale)),
            "median_b": float(np.median(sample_b_analysis_scale)),
            "mean_difference_a_minus_b": mean_delta_value,
            "median_difference_a_minus_b": float(np.median(delta_values_analysis_scale)),
            "ci95_lower": ci95_lower,
            "ci95_upper": ci95_upper,
            "t_statistic": float(paired_result_two_tailed.statistic),
            "df": df_value,
            "p_value_two_tailed": float(paired_result_two_tailed.pvalue),
            "p_value_one_tailed": float(paired_result_one_tailed.pvalue),
            "effect_size_name": "cohens_dz",
            "effect_size": effect_size_value,
        }

    def extract_independent_samples_for_between_subjects_contrast(
        subset_dataframe: pd.DataFrame,
        dv_key: str,
        story_condition: str,
        load_condition: str,
        contrast_type: str,
    ) -> tuple[np.ndarray, np.ndarray, str, str, str]:
        first_vignette_column_name = dv_specifications[dv_key]["first_vignette_column"]

        if contrast_type == "CH - CC":
            return (
                coerce_numeric_array(
                    subset_dataframe.loc[
                        subset_dataframe["vignette_condition_position_1"] == "CH",
                        first_vignette_column_name,
                    ]
                ),
                coerce_numeric_array(
                    subset_dataframe.loc[
                        subset_dataframe["vignette_condition_position_1"] == "CC",
                        first_vignette_column_name,
                    ]
                ),
                "CH",
                "CC",
                "",
            )

        if contrast_type == "DIV - CC":
            return (
                coerce_numeric_array(
                    subset_dataframe.loc[
                        subset_dataframe["vignette_condition_position_1"] == "DIV",
                        first_vignette_column_name,
                    ]
                ),
                coerce_numeric_array(
                    subset_dataframe.loc[
                        subset_dataframe["vignette_condition_position_1"] == "CC",
                        first_vignette_column_name,
                    ]
                ),
                "DIV",
                "CC",
                "",
            )

        if contrast_type == "CH - DIV":
            return (
                coerce_numeric_array(
                    subset_dataframe.loc[
                        subset_dataframe["vignette_condition_position_1"] == "CH",
                        first_vignette_column_name,
                    ]
                ),
                coerce_numeric_array(
                    subset_dataframe.loc[
                        subset_dataframe["vignette_condition_position_1"] == "DIV",
                        first_vignette_column_name,
                    ]
                ),
                "CH",
                "DIV",
                "",
            )

        if contrast_type == "MIN(CH, DIV) - CC":
            analysis_mode_for_dv = resolve_analysis_mode_for_dependent_variable(dv_key)

            sample_ch_raw = coerce_numeric_array(
                subset_dataframe.loc[
                    subset_dataframe["vignette_condition_position_1"] == "CH",
                    first_vignette_column_name,
                ]
            )
            sample_div_raw = coerce_numeric_array(
                subset_dataframe.loc[
                    subset_dataframe["vignette_condition_position_1"] == "DIV",
                        first_vignette_column_name,
                    ]
            )
            sample_cc_raw = coerce_numeric_array(
                subset_dataframe.loc[
                    subset_dataframe["vignette_condition_position_1"] == "CC",
                    first_vignette_column_name,
                ]
            )

            sample_ch_analysis_scale, transformation_label = transform_numeric_values_if_requested(
                sample_ch_raw,
                analysis_mode_for_dv,
            )
            sample_div_analysis_scale, _ = transform_numeric_values_if_requested(
                sample_div_raw,
                analysis_mode_for_dv,
            )

            if np.mean(sample_ch_analysis_scale) <= np.mean(sample_div_analysis_scale):
                extra_detail = (
                    f"For this subset, MIN(CH, DIV) is realized by CH because the "
                    f"{'transformed ' if transformation_label != 'none' else ''}mean of CH is lower than or equal to the mean of DIV."
                )
                return sample_ch_raw, sample_cc_raw, "CH", "CC", extra_detail

            extra_detail = (
                f"For this subset, MIN(CH, DIV) is realized by DIV because the "
                f"{'transformed ' if transformation_label != 'none' else ''}mean of DIV is lower than the mean of CH."
            )
            return sample_div_raw, sample_cc_raw, "DIV", "CC", extra_detail

        raise ValueError(f"Unknown between-subjects contrast_type: {contrast_type!r}")

    def extract_paired_samples_for_within_subjects_contrast(
        subset_dataframe: pd.DataFrame,
        dv_key: str,
        contrast_type: str,
    ) -> tuple[np.ndarray, np.ndarray, str, str]:
        dv_columns = dv_specifications[dv_key]

        if contrast_type == "CH - CC":
            group_a_values = pd.to_numeric(subset_dataframe[dv_columns["distal_columns"]["CH"]], errors="coerce")
            group_b_values = pd.to_numeric(subset_dataframe[dv_columns["distal_columns"]["CC"]], errors="coerce")
            valid_mask = (~group_a_values.isna()) & (~group_b_values.isna())
            return (
                np.asarray(group_a_values[valid_mask], dtype=float),
                np.asarray(group_b_values[valid_mask], dtype=float),
                "CH",
                "CC",
            )

        if contrast_type == "DIV - CC":
            group_a_values = pd.to_numeric(subset_dataframe[dv_columns["distal_columns"]["DIV"]], errors="coerce")
            group_b_values = pd.to_numeric(subset_dataframe[dv_columns["distal_columns"]["CC"]], errors="coerce")
            valid_mask = (~group_a_values.isna()) & (~group_b_values.isna())
            return (
                np.asarray(group_a_values[valid_mask], dtype=float),
                np.asarray(group_b_values[valid_mask], dtype=float),
                "DIV",
                "CC",
            )

        if contrast_type == "CH - DIV":
            group_a_values = pd.to_numeric(subset_dataframe[dv_columns["distal_columns"]["CH"]], errors="coerce")
            group_b_values = pd.to_numeric(subset_dataframe[dv_columns["distal_columns"]["DIV"]], errors="coerce")
            valid_mask = (~group_a_values.isna()) & (~group_b_values.isna())
            return (
                np.asarray(group_a_values[valid_mask], dtype=float),
                np.asarray(group_b_values[valid_mask], dtype=float),
                "CH",
                "DIV",
            )

        if contrast_type == "MIN(CH, DIV) - CC":
            group_ch_values = pd.to_numeric(subset_dataframe[dv_columns["distal_columns"]["CH"]], errors="coerce")
            group_div_values = pd.to_numeric(subset_dataframe[dv_columns["distal_columns"]["DIV"]], errors="coerce")
            group_cc_values = pd.to_numeric(subset_dataframe[dv_columns["distal_columns"]["CC"]], errors="coerce")

            valid_mask = (~group_ch_values.isna()) & (~group_div_values.isna()) & (~group_cc_values.isna())

            group_a_values = np.minimum(
                np.asarray(group_ch_values[valid_mask], dtype=float),
                np.asarray(group_div_values[valid_mask], dtype=float),
            )
            group_b_values = np.asarray(group_cc_values[valid_mask], dtype=float)

            return group_a_values, group_b_values, "MIN(CH,DIV)", "CC"

        if contrast_type == "CC - CH":
            group_a_values = pd.to_numeric(subset_dataframe[dv_columns["proximate_columns"]["CC"]], errors="coerce")
            group_b_values = pd.to_numeric(subset_dataframe[dv_columns["proximate_columns"]["CH"]], errors="coerce")
            valid_mask = (~group_a_values.isna()) & (~group_b_values.isna())
            return (
                np.asarray(group_a_values[valid_mask], dtype=float),
                np.asarray(group_b_values[valid_mask], dtype=float),
                "CC",
                "CH",
            )

        raise ValueError(f"Unknown within-subjects contrast_type: {contrast_type!r}")

    "========================================"
    "Core column specifications by DV family."
    "========================================"
    dv_specifications = {
        "blame": {
            "first_vignette_column": "first_vignette_distal_blame",
            "distal_columns": {
                "CC": "distal_blame_cc",
                "CH": "distal_blame_ch",
                "DIV": "distal_blame_div",
            },
            "proximate_columns": {
                "CC": "proximate_blame_cc",
                "CH": "proximate_blame_ch",
            },
        },
        "wrong": {
            "first_vignette_column": "first_vignette_distal_wrong",
            "distal_columns": {
                "CC": "distal_wrong_cc",
                "CH": "distal_wrong_ch",
                "DIV": "distal_wrong_div",
            },
            "proximate_columns": {
                "CC": "proximate_wrong_cc",
                "CH": "proximate_wrong_ch",
            },
        },
        "punish": {
            "first_vignette_column": "first_vignette_distal_punish",
            "distal_columns": {
                "CC": "distal_punish_cc",
                "CH": "distal_punish_ch",
                "DIV": "distal_punish_div",
            },
            "proximate_columns": {
                "CC": "proximate_punish_cc",
                "CH": "proximate_punish_ch",
            },
        },
    }

    between_subjects_contrast_types = [
        "CH - CC",
        "DIV - CC",
        "CH - DIV",
        "MIN(CH, DIV) - CC",
    ]
    within_subjects_distal_contrast_types = [
        "CH - CC",
        "DIV - CC",
        "CH - DIV",
        "MIN(CH, DIV) - CC",
    ]
    within_subjects_proximate_contrast_types = [
        "CC - CH",
    ]

    "============================================="
    "Build all rows using one standardized schema."
    "============================================="
    all_test_rows = []

    for inclusion_filter_label, analysis_dataframe in [
        ("included_only", cleaned_dataframe.loc[cleaned_dataframe["included"] == True].copy()),  # noqa: E712
        ("all_finishers", cleaned_dataframe.copy()),
    ]:
        for story_condition_value in ["pooled", "firework", "trolley"]:
            for load_condition_value in ["pooled", "high", "low"]:
                subset_dataframe = filter_dataframe_by_story_condition_and_load_condition(
                    analysis_dataframe=analysis_dataframe,
                    story_condition_value=story_condition_value,
                    load_condition_value=load_condition_value,
                )

                if subset_dataframe.shape[0] == 0:
                    continue

                "============================================"
                "Between-subject first-vignette distal rows."
                "============================================"
                for dv_key in ["blame", "wrong", "punish"]:
                    analysis_mode_for_dv = resolve_analysis_mode_for_dependent_variable(dv_key)

                    for contrast_type in between_subjects_contrast_types:
                        sample_a_raw, sample_b_raw, group_a_label, group_b_label, extra_detail = extract_independent_samples_for_between_subjects_contrast(
                            subset_dataframe=subset_dataframe,
                            dv_key=dv_key,
                            story_condition=story_condition_value,
                            load_condition=load_condition_value,
                            contrast_type=contrast_type,
                        )

                        is_confirmatory_row = (
                            dv_key == "blame"
                            and inclusion_filter_label == "included_only"
                            and story_condition_value == "pooled"
                            and load_condition_value == "pooled"
                            and contrast_type in {"CH - CC", "DIV - CC"}
                        )

                        if is_confirmatory_row and confirmatory_between_subjects_method_normalized in {"pooled_ols", "ols", "anova"}:
                            pooled_ols_rows_dataframe = run_pooled_ols_planned_contrasts(
                                dataframe=subset_dataframe,
                                dv_column_name=dv_specifications[dv_key]["first_vignette_column"],
                                group_column_name="vignette_condition_position_1",
                                covariance_type=confirmatory_pooled_ols_covariance_type,
                            ).copy()

                            pooled_ols_rows_dataframe["p_value_two_tailed"] = pooled_ols_rows_dataframe["p_value"].astype(float)
                            pooled_ols_rows_dataframe["p_value_one_tailed"] = pooled_ols_rows_dataframe.apply(
                                lambda row: compute_one_tailed_p_value_from_two_tailed_p_value(
                                    p_value_two_tailed=float(row["p_value"]),
                                    test_statistic=float(row["t_statistic"]),
                                ),
                                axis=1,
                            )

                            matching_ols_row = pooled_ols_rows_dataframe.loc[
                                (pooled_ols_rows_dataframe["group_a"] == group_a_label)
                                & (pooled_ols_rows_dataframe["group_b"] == group_b_label)
                            ].iloc[0]

                            standard_test_row = make_standard_test_row(
                                analysis_family="confirmatory",
                                analysis_mode="raw_parametric",
                                test_type="pooled_ols_planned_contrast",
                                transformation="none",
                                location_statistic_reported="mean_difference",
                                inclusion_filter=inclusion_filter_label,
                                story_condition=story_condition_value,
                                load_condition=load_condition_value,
                                design="between_subjects_first_vignette",
                                dv=dv_key,
                                agent_role="distal",
                                contrast_type=contrast_type,
                                group_a=group_a_label,
                                group_b=group_b_label,
                                n_a=int(matching_ols_row["n_a"]),
                                n_b=int(matching_ols_row["n_b"]),
                                mean_a=float(matching_ols_row["mean_a"]),
                                mean_b=float(matching_ols_row["mean_b"]),
                                median_a=float(np.median(sample_a_raw)) if sample_a_raw.shape[0] > 0 else np.nan,
                                median_b=float(np.median(sample_b_raw)) if sample_b_raw.shape[0] > 0 else np.nan,
                                mean_difference_a_minus_b=float(matching_ols_row["mean_difference_a_minus_b"]),
                                median_difference_a_minus_b=float(np.median(sample_a_raw) - np.median(sample_b_raw)) if sample_a_raw.shape[0] > 0 and sample_b_raw.shape[0] > 0 else np.nan,
                                ci95_lower=float(matching_ols_row["ci95_lower"]),
                                ci95_upper=float(matching_ols_row["ci95_upper"]),
                                t_statistic=float(matching_ols_row["t_statistic"]),
                                df=float(matching_ols_row["df"]),
                                p_value_two_tailed=float(matching_ols_row["p_value_two_tailed"]),
                                p_value_one_tailed=float(matching_ols_row["p_value_one_tailed"]),
                                effect_size_name=str(matching_ols_row["effect_size_name"]),
                                effect_size=float(matching_ols_row["effect_size"]),
                                p_value_holm=np.nan,
                                notes=build_plain_language_note(
                                    analysis_family="confirmatory",
                                    design="between_subjects_first_vignette",
                                    dv_key=dv_key,
                                    agent_role="distal",
                                    contrast_type=contrast_type,
                                    inclusion_filter=inclusion_filter_label,
                                    story_condition=story_condition_value,
                                    load_condition=load_condition_value,
                                    analysis_mode="raw_parametric",
                                    transformation="none",
                                    location_statistic_reported="mean_difference",
                                    extra_detail=(
                                        f"Model formula: {matching_ols_row['model_formula']}. "
                                        f"Covariance type: {matching_ols_row['model_covariance_type']}."
                                    ),
                                ),
                            )
                        else:
                            independent_samples_test_row = run_independent_samples_test(
                                sample_a_raw=sample_a_raw,
                                sample_b_raw=sample_b_raw,
                                analysis_mode=analysis_mode_for_dv,
                            )

                            analysis_family_value = (
                                "exploratory"
                                if not is_confirmatory_row
                                else "confirmatory"
                            )

                            standard_test_row = make_standard_test_row(
                                analysis_family=analysis_family_value,
                                analysis_mode=independent_samples_test_row["analysis_mode"],
                                test_type=independent_samples_test_row["test_type"],
                                transformation=independent_samples_test_row["transformation"],
                                location_statistic_reported=independent_samples_test_row["location_statistic_reported"],
                                inclusion_filter=inclusion_filter_label,
                                story_condition=story_condition_value,
                                load_condition=load_condition_value,
                                design="between_subjects_first_vignette",
                                dv=dv_key,
                                agent_role="distal",
                                contrast_type=contrast_type,
                                group_a=group_a_label,
                                group_b=group_b_label,
                                n_a=int(independent_samples_test_row["n_a"]),
                                n_b=int(independent_samples_test_row["n_b"]),
                                mean_a=float(independent_samples_test_row["mean_a"]),
                                mean_b=float(independent_samples_test_row["mean_b"]),
                                median_a=float(independent_samples_test_row["median_a"]),
                                median_b=float(independent_samples_test_row["median_b"]),
                                mean_difference_a_minus_b=float(independent_samples_test_row["mean_difference_a_minus_b"]),
                                median_difference_a_minus_b=float(independent_samples_test_row["median_difference_a_minus_b"]),
                                ci95_lower=float(independent_samples_test_row["ci95_lower"]) if not pd.isna(independent_samples_test_row["ci95_lower"]) else np.nan,
                                ci95_upper=float(independent_samples_test_row["ci95_upper"]) if not pd.isna(independent_samples_test_row["ci95_upper"]) else np.nan,
                                t_statistic=float(independent_samples_test_row["t_statistic"]) if not pd.isna(independent_samples_test_row["t_statistic"]) else np.nan,
                                df=float(independent_samples_test_row["df"]) if not pd.isna(independent_samples_test_row["df"]) else np.nan,
                                p_value_two_tailed=float(independent_samples_test_row["p_value_two_tailed"]),
                                p_value_one_tailed=float(independent_samples_test_row["p_value_one_tailed"]),
                                effect_size_name=str(independent_samples_test_row["effect_size_name"]),
                                effect_size=float(independent_samples_test_row["effect_size"]) if not pd.isna(independent_samples_test_row["effect_size"]) else np.nan,
                                p_value_holm=np.nan,
                                notes=build_plain_language_note(
                                    analysis_family=analysis_family_value,
                                    design="between_subjects_first_vignette",
                                    dv_key=dv_key,
                                    agent_role="distal",
                                    contrast_type=contrast_type,
                                    inclusion_filter=inclusion_filter_label,
                                    story_condition=story_condition_value,
                                    load_condition=load_condition_value,
                                    analysis_mode=independent_samples_test_row["analysis_mode"],
                                    transformation=independent_samples_test_row["transformation"],
                                    location_statistic_reported=independent_samples_test_row["location_statistic_reported"],
                                    extra_detail=extra_detail,
                                ),
                            )

                        all_test_rows.append(standard_test_row)

                "=========================================="
                "Within-subject distal and proximate rows."
                "=========================================="
                for dv_key in ["blame", "wrong", "punish"]:
                    analysis_mode_for_dv = resolve_analysis_mode_for_dependent_variable(dv_key)

                    for contrast_type in within_subjects_distal_contrast_types:
                        sample_a_raw, sample_b_raw, group_a_label, group_b_label = extract_paired_samples_for_within_subjects_contrast(
                            subset_dataframe=subset_dataframe,
                            dv_key=dv_key,
                            contrast_type=contrast_type,
                        )

                        paired_samples_test_row = run_paired_samples_test(
                            sample_a_raw=sample_a_raw,
                            sample_b_raw=sample_b_raw,
                            analysis_mode=analysis_mode_for_dv,
                        )

                        standard_test_row = make_standard_test_row(
                            analysis_family="exploratory",
                            analysis_mode=paired_samples_test_row["analysis_mode"],
                            test_type=paired_samples_test_row["test_type"],
                            transformation=paired_samples_test_row["transformation"],
                            location_statistic_reported=paired_samples_test_row["location_statistic_reported"],
                            inclusion_filter=inclusion_filter_label,
                            story_condition=story_condition_value,
                            load_condition=load_condition_value,
                            design="within_subjects_all_vignettes",
                            dv=dv_key,
                            agent_role="distal",
                            contrast_type=contrast_type,
                            group_a=group_a_label,
                            group_b=group_b_label,
                            n_a=int(paired_samples_test_row["n_a"]),
                            n_b=int(paired_samples_test_row["n_b"]),
                            mean_a=float(paired_samples_test_row["mean_a"]),
                            mean_b=float(paired_samples_test_row["mean_b"]),
                            median_a=float(paired_samples_test_row["median_a"]),
                            median_b=float(paired_samples_test_row["median_b"]),
                            mean_difference_a_minus_b=float(paired_samples_test_row["mean_difference_a_minus_b"]),
                            median_difference_a_minus_b=float(paired_samples_test_row["median_difference_a_minus_b"]),
                            ci95_lower=float(paired_samples_test_row["ci95_lower"]) if not pd.isna(paired_samples_test_row["ci95_lower"]) else np.nan,
                            ci95_upper=float(paired_samples_test_row["ci95_upper"]) if not pd.isna(paired_samples_test_row["ci95_upper"]) else np.nan,
                            t_statistic=float(paired_samples_test_row["t_statistic"]) if not pd.isna(paired_samples_test_row["t_statistic"]) else np.nan,
                            df=float(paired_samples_test_row["df"]) if not pd.isna(paired_samples_test_row["df"]) else np.nan,
                            p_value_two_tailed=float(paired_samples_test_row["p_value_two_tailed"]),
                            p_value_one_tailed=float(paired_samples_test_row["p_value_one_tailed"]),
                            effect_size_name=str(paired_samples_test_row["effect_size_name"]),
                            effect_size=float(paired_samples_test_row["effect_size"]) if not pd.isna(paired_samples_test_row["effect_size"]) else np.nan,
                            p_value_holm=np.nan,
                            notes=build_plain_language_note(
                                analysis_family="exploratory",
                                design="within_subjects_all_vignettes",
                                dv_key=dv_key,
                                agent_role="distal",
                                contrast_type=contrast_type,
                                inclusion_filter=inclusion_filter_label,
                                story_condition=story_condition_value,
                                load_condition=load_condition_value,
                                analysis_mode=paired_samples_test_row["analysis_mode"],
                                transformation=paired_samples_test_row["transformation"],
                                location_statistic_reported=paired_samples_test_row["location_statistic_reported"],
                            ),
                        )
                        all_test_rows.append(standard_test_row)

                    for contrast_type in within_subjects_proximate_contrast_types:
                        sample_a_raw, sample_b_raw, group_a_label, group_b_label = extract_paired_samples_for_within_subjects_contrast(
                            subset_dataframe=subset_dataframe,
                            dv_key=dv_key,
                            contrast_type=contrast_type,
                        )

                        paired_samples_test_row = run_paired_samples_test(
                            sample_a_raw=sample_a_raw,
                            sample_b_raw=sample_b_raw,
                            analysis_mode=analysis_mode_for_dv,
                        )

                        standard_test_row = make_standard_test_row(
                            analysis_family="exploratory",
                            analysis_mode=paired_samples_test_row["analysis_mode"],
                            test_type=paired_samples_test_row["test_type"],
                            transformation=paired_samples_test_row["transformation"],
                            location_statistic_reported=paired_samples_test_row["location_statistic_reported"],
                            inclusion_filter=inclusion_filter_label,
                            story_condition=story_condition_value,
                            load_condition=load_condition_value,
                            design="within_subjects_all_vignettes",
                            dv=dv_key,
                            agent_role="proximate",
                            contrast_type=contrast_type,
                            group_a=group_a_label,
                            group_b=group_b_label,
                            n_a=int(paired_samples_test_row["n_a"]),
                            n_b=int(paired_samples_test_row["n_b"]),
                            mean_a=float(paired_samples_test_row["mean_a"]),
                            mean_b=float(paired_samples_test_row["mean_b"]),
                            median_a=float(paired_samples_test_row["median_a"]),
                            median_b=float(paired_samples_test_row["median_b"]),
                            mean_difference_a_minus_b=float(paired_samples_test_row["mean_difference_a_minus_b"]),
                            median_difference_a_minus_b=float(paired_samples_test_row["median_difference_a_minus_b"]),
                            ci95_lower=float(paired_samples_test_row["ci95_lower"]) if not pd.isna(paired_samples_test_row["ci95_lower"]) else np.nan,
                            ci95_upper=float(paired_samples_test_row["ci95_upper"]) if not pd.isna(paired_samples_test_row["ci95_upper"]) else np.nan,
                            t_statistic=float(paired_samples_test_row["t_statistic"]) if not pd.isna(paired_samples_test_row["t_statistic"]) else np.nan,
                            df=float(paired_samples_test_row["df"]) if not pd.isna(paired_samples_test_row["df"]) else np.nan,
                            p_value_two_tailed=float(paired_samples_test_row["p_value_two_tailed"]),
                            p_value_one_tailed=float(paired_samples_test_row["p_value_one_tailed"]),
                            effect_size_name=str(paired_samples_test_row["effect_size_name"]),
                            effect_size=float(paired_samples_test_row["effect_size"]) if not pd.isna(paired_samples_test_row["effect_size"]) else np.nan,
                            p_value_holm=np.nan,
                            notes=build_plain_language_note(
                                analysis_family="exploratory",
                                design="within_subjects_all_vignettes",
                                dv_key=dv_key,
                                agent_role="proximate",
                                contrast_type=contrast_type,
                                inclusion_filter=inclusion_filter_label,
                                story_condition=story_condition_value,
                                load_condition=load_condition_value,
                                analysis_mode=paired_samples_test_row["analysis_mode"],
                                transformation=paired_samples_test_row["transformation"],
                                location_statistic_reported=paired_samples_test_row["location_statistic_reported"],
                            ),
                        )
                        all_test_rows.append(standard_test_row)

    "=============================================="
    "Convert rows to dataframe and apply correction."
    "=============================================="
    dataframe_tests = pd.DataFrame(all_test_rows)

    confirmatory_row_mask = (
        (dataframe_tests["analysis_family"] == "confirmatory")
        & (dataframe_tests["inclusion_filter"] == "included_only")
        & (dataframe_tests["story_condition"] == "pooled")
        & (dataframe_tests["load_condition"] == "pooled")
        & (dataframe_tests["analysis_scope"] == "between_subjects_first_vignette")
        & (dataframe_tests["dv"] == "blame")
        & (dataframe_tests["agent_role"] == "distal")
        & (dataframe_tests["contrast_type"].isin(["CH - CC", "DIV - CC"]))
    )

    confirmatory_p_values_to_correct = dataframe_tests.loc[
        confirmatory_row_mask,
        "p_value_two_tailed",
    ].tolist()

    if len(confirmatory_p_values_to_correct) == 2:
        holm_adjusted_p_values = holm_bonferroni_correct_p_values(confirmatory_p_values_to_correct)
        dataframe_tests.loc[confirmatory_row_mask, "p_value_holm"] = holm_adjusted_p_values

        holm_note_suffix = (
            "Holm correction in p_value_holm is always based on the preregistered two-tailed confirmatory p-values."
        )
        dataframe_tests.loc[confirmatory_row_mask, "notes"] = (
            dataframe_tests.loc[confirmatory_row_mask, "notes"].astype(str)
            + " "
            + holm_note_suffix
        )

    "======================="
    "Sort and reorder rows."
    "======================="
    dataframe_tests["inclusion_filter_sort_order"] = dataframe_tests["inclusion_filter"].map(participants_sort_map)
    dataframe_tests["story_condition_sort_order"] = dataframe_tests["story_condition"].map(story_condition_sort_map)
    dataframe_tests["load_condition_sort_order"] = dataframe_tests["load_condition"].map(load_condition_sort_map)
    dataframe_tests["design_sort_order"] = dataframe_tests["analysis_scope"].map(design_sort_map)
    dataframe_tests["dv_sort_order"] = dataframe_tests["dv"].map(dv_sort_map)
    dataframe_tests["agent_role_sort_order"] = dataframe_tests["agent_role"].map(agent_role_sort_map)
    dataframe_tests["contrast_type_sort_order"] = dataframe_tests["contrast_type"].map(
        lambda value: contrast_type_sort_map.get(value, 999)
    )
    dataframe_tests["analysis_family_sort_order"] = dataframe_tests["analysis_family"].map(
        {"confirmatory": 0, "exploratory": 1}
    )

    dataframe_tests = dataframe_tests.sort_values(
        by=[
            "inclusion_filter_sort_order",
            "story_condition_sort_order",
            "load_condition_sort_order",
            "design_sort_order",
            "dv_sort_order",
            "agent_role_sort_order",
            "contrast_type_sort_order",
            "analysis_family_sort_order",
        ],
        kind="stable",
    ).drop(
        columns=[
            "inclusion_filter_sort_order",
            "story_condition_sort_order",
            "load_condition_sort_order",
            "design_sort_order",
            "dv_sort_order",
            "agent_role_sort_order",
            "contrast_type_sort_order",
            "analysis_family_sort_order",
        ],
        errors="ignore",
    ).reset_index(drop=True)

    desired_column_order = [
        "analysis_family",
        "analysis_mode",
        "test_type",
        "transformation",
        "location_statistic_reported",
        "inclusion_filter",
        "story_condition",
        "load_condition",
        "analysis_scope",
        "dv",
        "agent_role",
        "contrast_type",
        "group_a",
        "group_b",
        "n_a",
        "n_b",
        "mean_a",
        "mean_b",
        "median_a",
        "median_b",
        "mean_difference_a_minus_b",
        "median_difference_a_minus_b",
        "ci95_lower",
        "ci95_upper",
        "t_statistic",
        "df",
        "p_value_two_tailed",
        "p_value_one_tailed",
        "effect_size_name",
        "effect_size",
        "p_value_holm",
        "notes",
    ]

    for column_name in desired_column_order:
        if column_name not in dataframe_tests.columns:
            dataframe_tests[column_name] = np.nan

    dataframe_tests = dataframe_tests[desired_column_order].copy()

    "Save and return"
    _save_analysis_dataframe_to_processed_folder(
        dataframe_to_save=dataframe_tests,
        general_settings=general_settings,
        file_name_key="tests",
    )

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
                "story_condition": "pooled",
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


def compute_first_vignette_distal_blame_integrated_models(
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
            - All rows carry analysis_scope = "between_subjects_first_vignette".
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
                "vignette_condition_position_1",
                "first_vignette_distal_blame",
            ]
        ].copy()

        if include_only_passers:
            analysis_dataframe = analysis_dataframe.loc[analysis_dataframe["included"] == True].copy()  # noqa: E712

        analysis_dataframe["story_condition"] = pd.Categorical(
            analysis_dataframe["story_condition"],
            categories=["firework", "trolley"],
            ordered=True,
        )
        analysis_dataframe["vignette_condition_position_1"] = pd.Categorical(
            analysis_dataframe["vignette_condition_position_1"],
            categories=["CC", "CH", "DIV"],
            ordered=True,
        )
        analysis_dataframe["first_vignette_distal_blame"] = pd.to_numeric(
            analysis_dataframe["first_vignette_distal_blame"],
            errors="coerce",
        )

        analysis_dataframe = analysis_dataframe.dropna(
            subset=["first_vignette_distal_blame", "story_condition", "vignette_condition_position_1"]
        ).copy()

        model_formula_string = (
            "first_vignette_distal_blame ~ "
            "C(vignette_condition_position_1, Treatment(reference='CC')) * "
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
            .groupby(["story_condition", "vignette_condition_position_1"], observed=True)["first_vignette_distal_blame"]
            .agg(["count", "mean", "std", "median"])
            .reset_index()
        )

        for _, descriptive_row in observed_descriptives.iterrows():
            story_condition_value: str = str(descriptive_row["story_condition"])
            vignette_condition_value: str = str(descriptive_row["vignette_condition_position_1"])

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
            "C(vignette_condition_position_1, Treatment(reference='CC'))[T.CH]": "CH - CC in the firework story family.",
            "C(vignette_condition_position_1, Treatment(reference='CC'))[T.DIV]": "DIV - CC in the firework story family.",
            "C(story_condition, Treatment(reference='firework'))[T.trolley]": "How the CC baseline changes in trolley relative to firework.",
            "C(vignette_condition_position_1, Treatment(reference='CC'))[T.CH]:C(story_condition, Treatment(reference='firework'))[T.trolley]": "How the CH - CC contrast changes in trolley relative to firework.",
            "C(vignette_condition_position_1, Treatment(reference='CC'))[T.DIV]:C(story_condition, Treatment(reference='firework'))[T.trolley]": "How the DIV - CC contrast changes in trolley relative to firework.",
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
                    "vignette_condition_position_1": [contrast_group_a_value, contrast_group_a_value],
                    "story_condition": ["firework", "trolley"],
                }
            )
            pooled_grid_group_b = pd.DataFrame(
                {
                    "vignette_condition_position_1": [contrast_group_b_value, contrast_group_b_value],
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
                    story_condition="pooled",
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
                        "vignette_condition_position_1": [contrast_group_a_value],
                        "story_condition": [story_condition_value],
                    }
                )
                story_grid_group_b = pd.DataFrame(
                    {
                        "vignette_condition_position_1": [contrast_group_b_value],
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
                    "story_condition": "pooled",
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


def compute_within_subject_distal_blame_integrated_models(
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
            - All rows carry analysis_scope = "within_subjects_repeated_measures".

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
                "vignette_condition_position_1",
                "vignette_condition_position_2",
                "vignette_condition_position_3",
                "distal_blame_cc",
                "distal_blame_ch",
                "distal_blame_div",
            ]
        ].copy()

        if include_only_passers:
            wide_analysis_dataframe = wide_analysis_dataframe.loc[
                wide_analysis_dataframe["included"] == True
            ].copy()  # noqa: E712

        long_rows: list[dict[str, Any]] = []

        for _, participant_row in wide_analysis_dataframe.iterrows():
            vignette_position_by_condition = {
                participant_row["vignette_condition_position_1"]: 1,
                participant_row["vignette_condition_position_2"]: 2,
                participant_row["vignette_condition_position_3"]: 3,
            }

            for vignette_condition_value in ["CC", "CH", "DIV"]:
                long_rows.append(
                    {
                        "response_id": participant_row["response_id"],
                        "story_condition": participant_row["story_condition"],
                        "vignette_condition": vignette_condition_value,
                        "vignette_position": vignette_position_by_condition[vignette_condition_value],
                        "distal_rating_value": participant_row[f"distal_blame_{vignette_condition_value.lower()}"],
                    }
                )

        analysis_dataframe_long = pd.DataFrame(long_rows)

        analysis_dataframe_long["distal_rating_value"] = pd.to_numeric(
            analysis_dataframe_long["distal_rating_value"],
            errors="coerce",
        )
        analysis_dataframe_long = analysis_dataframe_long.dropna(
            subset=["distal_rating_value"]
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
            "distal_rating_value ~ "
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
            "distal_rating_value ~ "
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
            .groupby(["vignette_condition"], observed=True)["distal_rating_value"]
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
                    "story_condition": "pooled",
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
                    story_condition="pooled",
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
                    "story_condition": "pooled",
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
                    "story_condition": "pooled",
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


def compute_integrated_distal_blame_results(
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

    dataframe_first_vignette_models = compute_first_vignette_distal_blame_integrated_models(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )
    dataframe_within_subject_models = compute_within_subject_distal_blame_integrated_models(
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
            - Same dataframe returned by compute_first_vignette_distal_blame_integrated_models.
    """
    return compute_first_vignette_distal_blame_integrated_models(
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
            - Same dataframe returned by compute_within_subject_distal_blame_integrated_models.
    """
    return compute_within_subject_distal_blame_integrated_models(
        general_settings=general_settings,
        cleaned_dataframe=cleaned_dataframe,
        force_rebuild=force_rebuild,
    )




def generate_processed_data_codebook(general_settings: GeneralSettings, force_rebuild: bool | None = None) -> pd.DataFrame:
    """
    Generate a column-by-column codebook for the main processed CSVs.

    This version is designed first and foremost to help a human reader understand the cleaned dataframe.
    It creates one row for every single column in:
        • responsibility_shielding_cleaned.csv
        • responsibility_shielding_tests.csv
        • responsibility_shielding_group_summaries.csv

    Rows are grouped by file and preserve the actual column order in each CSV.

    Columns in the codebook:
        • file_name
        • column_name
        • logical_field
        • meaning
        • allowed_values_or_pattern
        • example_values
        • notes

    Arguments:
        • general_settings: GeneralSettings
            - Master settings dictionary.
        • force_rebuild: bool | None
            - If None, inherit general_settings["misc"]["force_rebuild"].
            - If False and the codebook already exists, load and return it.

    Returns:
        • pd.DataFrame
            - Codebook dataframe saved to processed/responsibility_shielding_processed_codebook.csv
    """
    if force_rebuild is None:
        force_rebuild = general_settings["misc"]["force_rebuild"]

    file_name_codebook = dict(general_settings["filing"]["file_names"]).get(
        "codebook",
        "responsibility_shielding_processed_codebook.csv",
    )
    file_path_codebook = general_settings["filing"]["file_paths"]["processed"] / file_name_codebook

    "Load and return dataframe if one already exists and not directed to rebuild."
    if file_path_codebook.exists() and not force_rebuild:
        return pd.read_csv(file_path_codebook, encoding="utf-8-sig")

    "Load the three processed dataframes that the codebook should document."
    cleaned_dataframe = load_or_build_cleaned_dataframe(
        general_settings=general_settings,
        force_rebuild=False,
    )
    tests_dataframe = run_confirmatory_and_exploratory_tests(
        general_settings=general_settings,
        force_rebuild=False,
    )
    group_summaries_dataframe = compute_group_summaries(
        general_settings=general_settings,
        force_rebuild=False,
    )

    "========================"
    "Small formatting helpers."
    "========================"
    def format_example_values(column_series: pd.Series, max_examples: int = 5) -> str:
        """
        Build a human-readable example string from the first few distinct non-missing values.
        """
        nonmissing_values = column_series.dropna()

        if nonmissing_values.shape[0] == 0:
            return "No non-missing values"

        distinct_values_in_order = list(dict.fromkeys(nonmissing_values.astype(str).tolist()))
        distinct_values_in_order = distinct_values_in_order[:max_examples]

        return ", ".join(distinct_values_in_order)

    def infer_allowed_values_or_pattern_from_series(column_series: pd.Series) -> str:
        """
        Infer a simple allowed-values / type description from the series itself.
        """
        nonmissing_values = column_series.dropna()

        if nonmissing_values.shape[0] == 0:
            return "No observed non-missing values"

        unique_values = pd.unique(nonmissing_values)
        unique_count = len(unique_values)

        if pd.api.types.is_bool_dtype(nonmissing_values):
            return "True | False"

        if pd.api.types.is_numeric_dtype(nonmissing_values):
            "Threshold of 8: 9-point Likert scales are treated as continuous numeric, so any numeric field"
            "with 9 or more distinct values is summarized as 'numeric' rather than enumerated."
            if unique_count <= 8:
                displayed_values = [str(value) for value in unique_values]
                return " | ".join(displayed_values)
            return "numeric"

        if pd.api.types.is_datetime64_any_dtype(nonmissing_values):
            return "datetime"

        if unique_count <= 8:
            displayed_values = [str(value) for value in unique_values]
            return " | ".join(displayed_values)

        return "string / text"

    def humanize_dv_code(dv_code: str) -> str:
        dv_map = {
            "blame": "blameworthiness",
            "wrong": "wrongness",
            "punish": "punishment",
        }
        return dv_map.get(dv_code, dv_code)

    def humanize_condition_code(condition_code: str) -> str:
        condition_map = {
            "cc": "Choice-Choice (CC; shielded)",
            "ch": "Choice-Chance (CH; non-shielded)",
            "div": "Division (DIV)",
            "CC": "Choice-Choice (CC; shielded)",
            "CH": "Choice-Chance (CH; non-shielded)",
            "DIV": "Division (DIV)",
        }
        return condition_map.get(condition_code, condition_code)

    "===================================="
    "Per-file metadata inference helpers."
    "===================================="
    def infer_cleaned_column_metadata(column_name: str, column_series: pd.Series) -> dict[str, str]:
        """
        Infer codebook metadata for one cleaned-dataframe column.
        """
        cleaned_exact_map: dict[str, dict[str, str]] = {
            "response_id": {
                "logical_field": "participant identifier",
                "meaning": "Unique Qualtrics response identifier for one participant.",
                "allowed_values_or_pattern": "string identifier",
                "notes": "Useful for row-level inspection, merging, and traceability; not theoretically meaningful.",
            },
            "start_date": {
                "logical_field": "survey start time",
                "meaning": "Timestamp when the participant began the Qualtrics survey.",
                "allowed_values_or_pattern": "datetime",
                "notes": "Used for freeze-window filtering and duration calculations.",
            },
            "end_date": {
                "logical_field": "survey end time",
                "meaning": "Timestamp when the participant finished the Qualtrics survey.",
                "allowed_values_or_pattern": "datetime",
                "notes": "Used for freeze-window filtering and duration calculations.",
            },
            "story_condition": {
                "logical_field": "story family",
                "meaning": "Which story family the participant received.",
                "allowed_values_or_pattern": "firework | trolley",
                "notes": "Firework and trolley are structurally matched story families.",
            },
            "load_condition": {
                "logical_field": "cognitive load condition",
                "meaning": "Working-memory load manipulation assigned to the participant.",
                "allowed_values_or_pattern": "high | low",
                "notes": "High = 7-digit memory load; Low = 2-digit memory load.",
            },
            "cognitive_load_digits_response": {
                "logical_field": "digits recall response",
                "meaning": "The digits string the participant typed during the cognitive-load recall check.",
                "allowed_values_or_pattern": "string / numeric text",
                "notes": "Raw memory-check response rather than a scored variable.",
            },
            "cognitive_load_digits_correct_bool": {
                "logical_field": "digits recall accuracy",
                "meaning": "Whether the participant correctly recalled the assigned high- or low-load digits.",
                "allowed_values_or_pattern": "True | False",
                "notes": "Manipulation-check variable for the cognitive-load task.",
            },
            "vignette_condition_order": {
                "logical_field": "vignette order string",
                "meaning": "The randomized order in which the participant encountered the three main vignette conditions.",
                "allowed_values_or_pattern": "Three-condition order string, e.g. CC-CH-DIV",
                "notes": "Convenient summary of the full order assignment.",
            },
            "case_order": {
                "logical_field": "vignette order string",
                "meaning": "The randomized order in which the participant encountered the three main vignette conditions.",
                "allowed_values_or_pattern": "Three-condition order string, e.g. CC-CH-DIV",
                "notes": "Older name for the same logical field as vignette_condition_order.",
            },
            "vignette_condition_position_1": {
                "logical_field": "first vignette condition",
                "meaning": "Which main vignette condition appeared first for that participant.",
                "allowed_values_or_pattern": "CC | CH | DIV",
                "notes": "Used in the confirmatory between-subjects analysis.",
            },
            "vignette_condition_position_2": {
                "logical_field": "second vignette condition",
                "meaning": "Which main vignette condition appeared second for that participant.",
                "allowed_values_or_pattern": "CC | CH | DIV",
                "notes": "Used for order-sensitivity analyses.",
            },
            "vignette_condition_position_3": {
                "logical_field": "third vignette condition",
                "meaning": "Which main vignette condition appeared third for that participant.",
                "allowed_values_or_pattern": "CC | CH | DIV",
                "notes": "Used for order-sensitivity analyses.",
            },
            "case_code_position_1": {
                "logical_field": "first vignette condition",
                "meaning": "Which main vignette condition appeared first for that participant.",
                "allowed_values_or_pattern": "CC | CH | DIV",
                "notes": "Older name for the same logical field as vignette_condition_position_1.",
            },
            "case_code_position_2": {
                "logical_field": "second vignette condition",
                "meaning": "Which main vignette condition appeared second for that participant.",
                "allowed_values_or_pattern": "CC | CH | DIV",
                "notes": "Older name for the same logical field as vignette_condition_position_2.",
            },
            "case_code_position_3": {
                "logical_field": "third vignette condition",
                "meaning": "Which main vignette condition appeared third for that participant.",
                "allowed_values_or_pattern": "CC | CH | DIV",
                "notes": "Older name for the same logical field as vignette_condition_position_3.",
            },
            "twoafc_bill_vs_clark": {
                "logical_field": "2AFC interpersonal blame response",
                "meaning": "Collapsed forced-choice response comparing Bill versus Clark on blame.",
                "allowed_values_or_pattern": "Bill > Clark | Bill ≥ Clark | Bill ≤ Clark | Bill < Clark",
                "notes": "Combines the initial 2AFC choice with the follow-up roughly-equal probe.",
            },
            "twoafc_ch_vs_cc": {
                "logical_field": "2AFC CH versus CC response",
                "meaning": "Collapsed forced-choice response comparing Clark in CH versus Clark in CC.",
                "allowed_values_or_pattern": "CH > CC | CH ≥ CC | CH ≤ CC | CH < CC",
                "notes": "Combines the initial 2AFC choice with the follow-up roughly-equal probe.",
            },
            "twoafc_div_vs_cc": {
                "logical_field": "2AFC DIV versus CC response",
                "meaning": "Collapsed forced-choice response comparing Clark in DIV versus Clark in CC.",
                "allowed_values_or_pattern": "DIV > CC | DIV ≥ CC | DIV ≤ CC | DIV < CC",
                "notes": "Combines the initial 2AFC choice with the follow-up roughly-equal probe.",
            },
            "comprehension_probability_same_bool": {
                "logical_field": "comprehension check: matched probability",
                "meaning": "Whether the participant correctly understood that the upstream actor faced the same downstream harm probability across conditions.",
                "allowed_values_or_pattern": "True | False",
                "notes": "One of the preregistered comprehension checks.",
            },
            "comprehension_distal_necessary_bool": {
                "logical_field": "comprehension check: distal necessity",
                "meaning": "Whether the participant correctly understood that the distal actor’s action was necessary for harm.",
                "allowed_values_or_pattern": "True | False",
                "notes": "One of the preregistered comprehension checks.",
            },
            "comprehension_clark_necessary_bool": {
                "logical_field": "comprehension check: distal necessity",
                "meaning": "Whether the participant correctly understood that the distal actor’s action was necessary for harm.",
                "allowed_values_or_pattern": "True | False",
                "notes": "Older name for the same logical field as comprehension_distal_necessary_bool.",
            },
            "comprehension_bill_necessary_bool": {
                "logical_field": "comprehension check: proximate necessity",
                "meaning": "Whether the participant correctly understood that the proximate node’s action was necessary for harm.",
                "allowed_values_or_pattern": "True | False",
                "notes": "One of the preregistered comprehension checks.",
            },
            "comprehension_all_correct_bool": {
                "logical_field": "comprehension pass indicator",
                "meaning": "Whether the participant passed all preregistered comprehension checks.",
                "allowed_values_or_pattern": "True | False",
                "notes": "Feeds directly into the included variable.",
            },
            "crt_score": {
                "logical_field": "cognitive reflection score",
                "meaning": "Total number of correct answers on the 3-item Cognitive Reflection Test.",
                "allowed_values_or_pattern": "0 | 1 | 2 | 3",
                "notes": "Exploratory individual-differences variable.",
            },
            "individualism_horizontal": {
                "logical_field": "horizontal individualism score",
                "meaning": "Derived INDCOL horizontal individualism summary score.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Exploratory individual-differences variable.",
            },
            "individualism_vertical": {
                "logical_field": "vertical individualism score",
                "meaning": "Derived INDCOL vertical individualism summary score.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Exploratory individual-differences variable.",
            },
            "individualism_score": {
                "logical_field": "overall individualism score",
                "meaning": "Composite individualism score derived from the INDCOL measure.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Exploratory individual-differences variable.",
            },
            "included": {
                "logical_field": "analysis inclusion flag",
                "meaning": "Whether the participant was included under the preregistered comprehension-based inclusion rule.",
                "allowed_values_or_pattern": "True | False",
                "notes": "True means included in the main analyses; False means all-finisher robustness only.",
            },
            "age": {
                "logical_field": "age",
                "meaning": "Participant age.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Demographic variable.",
            },
            "gender": {
                "logical_field": "gender",
                "meaning": "Participant gender response.",
                "allowed_values_or_pattern": infer_allowed_values_or_pattern_from_series(column_series),
                "notes": "Demographic variable.",
            },
            "race_compact": {
                "logical_field": "race",
                "meaning": "Participant race response in compact form, including self-describe text when applicable.",
                "allowed_values_or_pattern": "categorical text",
                "notes": "Demographic variable.",
            },
            "political_compact": {
                "logical_field": "political orientation",
                "meaning": "Participant political orientation response in compact form, including self-describe text when applicable.",
                "allowed_values_or_pattern": "categorical text",
                "notes": "Demographic variable.",
            },
            "ip_address": {
                "logical_field": "IP address",
                "meaning": "Qualtrics IP address field, retained only if still present in a local non-public copy.",
                "allowed_values_or_pattern": "string",
                "notes": "Potentially identifying. Should normally be absent from the public repo.",
            },
            "location_latitude": {
                "logical_field": "latitude",
                "meaning": "Qualtrics latitude field, retained only if still present in a local non-public copy.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Potentially identifying. Should normally be absent from the public repo.",
            },
            "location_longitude": {
                "logical_field": "longitude",
                "meaning": "Qualtrics longitude field, retained only if still present in a local non-public copy.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Potentially identifying. Should normally be absent from the public repo.",
            },
            "duration_seconds": {
                "logical_field": "survey duration",
                "meaning": "Qualtrics completion duration in seconds.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Descriptive timing variable.",
            },
        }

        if column_name in cleaned_exact_map:
            return cleaned_exact_map[column_name]

        first_vignette_match = re.match(r"^first_vignette_(distal|clark)_(blame|wrong|punish)$", column_name)
        if first_vignette_match:
            agent_role_code, dv_code = first_vignette_match.groups()
            return {
                "logical_field": f"first-vignette {humanize_dv_code(dv_code)} rating",
                "meaning": f"The participant’s first-vignette rating of the distal agent on {humanize_dv_code(dv_code)}.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Used in the confirmatory between-subjects analyses.",
            }

        level_rating_match = re.match(r"^(distal|clark|proximate)_(blame|wrong|punish)_(cc|ch|div)$", column_name)
        if level_rating_match:
            agent_role_code, dv_code, condition_code = level_rating_match.groups()

            if agent_role_code in {"distal", "clark"}:
                agent_role_label = "distal agent"
            else:
                agent_role_label = "proximate node"

            return {
                "logical_field": f"{agent_role_label} {humanize_dv_code(dv_code)} rating",
                "meaning": f"Rating of the {agent_role_label} on {humanize_dv_code(dv_code)} in {humanize_condition_code(condition_code)}.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Raw rating column before contrast construction.",
            }

        delta_match = re.match(
            r"^(distal|clark|proximate)_(blame|wrong|punish)_(ch|div)_minus_cc$",
            column_name,
        )
        if delta_match:
            agent_role_code, dv_code, left_condition_code = delta_match.groups()

            if agent_role_code in {"distal", "clark"}:
                agent_role_label = "distal agent"
            else:
                agent_role_label = "proximate node"

            return {
                "logical_field": f"within-subject {humanize_dv_code(dv_code)} contrast",
                "meaning": f"Participant-level within-subject difference on {humanize_dv_code(dv_code)} for the {agent_role_label}: {left_condition_code.upper()} minus CC.",
                "allowed_values_or_pattern": "numeric difference score",
                "notes": "Positive values mean the left condition exceeds CC for that participant.",
            }

        cc_minus_ch_match = re.match(
            r"^(distal|clark|proximate)_(blame|wrong|punish)_cc_minus_ch$",
            column_name,
        )
        if cc_minus_ch_match:
            agent_role_code, dv_code = cc_minus_ch_match.groups()

            if agent_role_code in {"distal", "clark"}:
                agent_role_label = "distal agent"
            else:
                agent_role_label = "proximate node"

            return {
                "logical_field": f"within-subject {humanize_dv_code(dv_code)} contrast",
                "meaning": f"Participant-level within-subject difference on {humanize_dv_code(dv_code)} for the {agent_role_label}: CC minus CH.",
                "allowed_values_or_pattern": "numeric difference score",
                "notes": "Most often used for the proximate manipulation-check comparison.",
            }

        min_delta_match = re.match(
            r"^(distal|clark|proximate)_(blame|wrong|punish)_min_ch_div_minus_cc$",
            column_name,
        )
        if min_delta_match:
            agent_role_code, dv_code = min_delta_match.groups()

            if agent_role_code in {"distal", "clark"}:
                agent_role_label = "distal agent"
            else:
                agent_role_label = "proximate node"

            return {
                "logical_field": f"strict shielding contrast on {humanize_dv_code(dv_code)}",
                "meaning": f"Participant-level difference between CC and the smaller of CH and DIV on {humanize_dv_code(dv_code)} for the {agent_role_label}.",
                "allowed_values_or_pattern": "numeric difference score",
                "notes": "Implements the stricter MIN(CH, DIV) − CC style comparison.",
            }

        interpersonal_match = re.match(
            r"^bill_minus_(distal|clark)_(cc|ch)_(blame|wrong|punish)$",
            column_name,
        )
        if interpersonal_match:
            distal_code, condition_code, dv_code = interpersonal_match.groups()
            return {
                "logical_field": f"interpersonal disparity on {humanize_dv_code(dv_code)}",
                "meaning": f"Proximate minus distal difference on {humanize_dv_code(dv_code)} within {humanize_condition_code(condition_code)}.",
                "allowed_values_or_pattern": "numeric difference score",
                "notes": "Diagnostic interpersonal disparity, not the main counterfactual shielding definition.",
            }

        timing_match = re.match(
            r"^(.+)_timing_(first_click_seconds|last_click_seconds|last_minus_first_seconds)$",
            column_name,
        )
        if timing_match:
            timing_block, timing_measure = timing_match.groups()
            return {
                "logical_field": "page timing variable",
                "meaning": f"Unified Qualtrics timing measure for the {timing_block} block: {timing_measure}.",
                "allowed_values_or_pattern": "numeric seconds",
                "notes": "Derived from story-specific Qualtrics timing fields.",
            }

        return {
            "logical_field": "cleaned dataframe field",
            "meaning": "Column in the cleaned participant-level dataframe.",
            "allowed_values_or_pattern": infer_allowed_values_or_pattern_from_series(column_series),
            "notes": "No specialized rule matched this column name. Read alongside neighboring columns for context.",
        }

    def infer_tests_column_metadata(column_name: str, column_series: pd.Series) -> dict[str, str]:
        tests_exact_map: dict[str, dict[str, str]] = {
            "analysis_family": {
                "logical_field": "analysis family",
                "meaning": "Whether the row belongs to the preregistered confirmatory family or the exploratory family.",
                "allowed_values_or_pattern": "confirmatory | exploratory",
                "notes": "Used heavily by the manuscript tables.",
            },
            "analysis_mode": {
                "logical_field": "analysis mode",
                "meaning": "How the row was analyzed statistically.",
                "allowed_values_or_pattern": "raw_parametric | raw_nonparametric | log1p_parametric",
                "notes": "Especially important for punishment rows.",
            },
            "test_type": {
                "logical_field": "statistical test type",
                "meaning": "The specific test used to compute the row.",
                "allowed_values_or_pattern": infer_allowed_values_or_pattern_from_series(column_series),
                "notes": "Examples include pooled OLS planned contrasts, Welch tests, and nonparametric tests.",
            },
            "transformation": {
                "logical_field": "transformation flag",
                "meaning": "Whether the dependent variable was transformed before analysis.",
                "allowed_values_or_pattern": "none | log1p",
                "notes": "Most relevant for punishment rows under log1p_parametric.",
            },
            "location_statistic_reported": {
                "logical_field": "reported location statistic",
                "meaning": "Which location difference the estimate columns should be interpreted as.",
                "allowed_values_or_pattern": "mean_difference | median_difference",
                "notes": "Mean for parametric rows; median for nonparametric rows.",
            },
            "inclusion_filter": {
                "logical_field": "inclusion filter",
                "meaning": "Which participant set the row was computed on.",
                "allowed_values_or_pattern": "included_only | all_finishers",
                "notes": "Whether the row was computed on preregistered included participants only, or all participants who finished.",
            },
            "story_condition": {
                "logical_field": "story family",
                "meaning": "Whether the row pools stories or restricts to one story family.",
                "allowed_values_or_pattern": "pooled | firework | trolley | all",
                "notes": "Used in story-specific and pooled contrasts.",
            },
            "load_condition": {
                "logical_field": "load condition slice",
                "meaning": "Whether the row pools load conditions or restricts to one load condition.",
                "allowed_values_or_pattern": "pooled | high | low",
                "notes": "Used in the cognitive-load breakdowns.",
            },
            "analysis_scope": {
                "logical_field": "analysis design",
                "meaning": "Whether the row compares different participant groups (first-vignette only) or paired ratings from the same participants (all vignettes).",
                "allowed_values_or_pattern": "between_subjects_first_vignette | within_subjects_all_vignettes",
                "notes": "Matches the analysis_scope column in group_summaries and integrated_blame_models, using the same fine-grained value names.",
            },
            "dv": {
                "logical_field": "dependent variable",
                "meaning": "Which dependent variable the row analyzes.",
                "allowed_values_or_pattern": "blame | wrong | punish",
                "notes": "Short schema names used throughout the code.",
            },
            "agent_role": {
                "logical_field": "judged agent role",
                "meaning": "Whether the row pertains to the distal actor or the proximate node.",
                "allowed_values_or_pattern": "distal | proximate",
                "notes": "Between-subject rows should always be distal in the new schema.",
            },
            "contrast_type": {
                "logical_field": "contrast label",
                "meaning": "Which substantive comparison the row represents.",
                "allowed_values_or_pattern": "CH - CC | DIV - CC | CH - DIV | MIN(CH, DIV) - CC | CC - CH",
                "notes": "The positive direction of the contrast is the direction assumed by p_value_one_tailed.",
            },
            "contrast_or_condition": {
                "logical_field": "contrast label",
                "meaning": "Which substantive comparison the row represents.",
                "allowed_values_or_pattern": "CH - CC | DIV - CC | CH - DIV | MIN(CH, DIV) - CC | CC - CH",
                "notes": "Older name for the same logical field as contrast_type.",
            },
            "group_a": {
                "logical_field": "left side of contrast",
                "meaning": "The group or condition on the left side of the estimated contrast.",
                "allowed_values_or_pattern": "string label",
                "notes": "Used together with group_b to define the sign of the estimate.",
            },
            "group_b": {
                "logical_field": "right side of contrast",
                "meaning": "The group or condition on the right side of the estimated contrast.",
                "allowed_values_or_pattern": "string label",
                "notes": "Used together with group_a to define the sign of the estimate.",
            },
            "n_a": {
                "logical_field": "sample size for group A",
                "meaning": "Number of usable observations on the left side of the contrast.",
                "allowed_values_or_pattern": "numeric count",
                "notes": "For within-subject paired rows this should match n_b.",
            },
            "n_b": {
                "logical_field": "sample size for group B",
                "meaning": "Number of usable observations on the right side of the contrast.",
                "allowed_values_or_pattern": "numeric count",
                "notes": "For within-subject paired rows this should match n_a.",
            },
            "mean_a": {
                "logical_field": "group A mean",
                "meaning": "Mean of the left-side group or condition on the analysis scale.",
                "allowed_values_or_pattern": "numeric",
                "notes": "If transformation='log1p', this mean is on the transformed scale.",
            },
            "mean_b": {
                "logical_field": "group B mean",
                "meaning": "Mean of the right-side group or condition on the analysis scale.",
                "allowed_values_or_pattern": "numeric",
                "notes": "If transformation='log1p', this mean is on the transformed scale.",
            },
            "median_a": {
                "logical_field": "group A median",
                "meaning": "Median of the left-side group or condition on the analysis scale.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Especially relevant for nonparametric rows.",
            },
            "median_b": {
                "logical_field": "group B median",
                "meaning": "Median of the right-side group or condition on the analysis scale.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Especially relevant for nonparametric rows.",
            },
            "mean_difference_a_minus_b": {
                "logical_field": "mean contrast estimate",
                "meaning": "Left-side mean minus right-side mean on the analysis scale.",
                "allowed_values_or_pattern": "numeric contrast estimate",
                "notes": "Primary estimate for parametric rows.",
            },
            "median_difference_a_minus_b": {
                "logical_field": "median contrast estimate",
                "meaning": "Left-side median minus right-side median on the analysis scale.",
                "allowed_values_or_pattern": "numeric contrast estimate",
                "notes": "Primary estimate for nonparametric rows.",
            },
            "ci95_lower": {
                "logical_field": "lower confidence bound",
                "meaning": "Lower 95% confidence interval bound for the reported location statistic.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Interpret together with ci95_upper and location_statistic_reported.",
            },
            "ci95_upper": {
                "logical_field": "upper confidence bound",
                "meaning": "Upper 95% confidence interval bound for the reported location statistic.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Interpret together with ci95_lower and location_statistic_reported.",
            },
            "t_statistic": {
                "logical_field": "test statistic",
                "meaning": "Test statistic reported for the row.",
                "allowed_values_or_pattern": "numeric",
                "notes": "For some nonparametric rows this is not literally a t statistic; it is the main test statistic field retained for consistency.",
            },
            "df": {
                "logical_field": "degrees of freedom",
                "meaning": "Degrees of freedom for parametric rows; often missing for nonparametric rows.",
                "allowed_values_or_pattern": "numeric or missing",
                "notes": "Missing values are expected for some nonparametric tests.",
            },
            "p_value_two_tailed": {
                "logical_field": "two-tailed p-value",
                "meaning": "Two-sided p-value for the row.",
                "allowed_values_or_pattern": "numeric p-value",
                "notes": "Use this when the manuscript or table is set to two-tailed reporting.",
            },
            "p_value_one_tailed": {
                "logical_field": "one-tailed p-value",
                "meaning": "One-tailed p-value assuming the positive direction of the named contrast.",
                "allowed_values_or_pattern": "numeric p-value",
                "notes": "The positive direction is defined by contrast_type / contrast_or_condition.",
            },
            "p_value_holm": {
                "logical_field": "Holm-corrected p-value",
                "meaning": "Holm-adjusted p-value for the confirmatory between-subject blame contrasts when applicable.",
                "allowed_values_or_pattern": "numeric or missing",
                "notes": "Usually populated only for the preregistered confirmatory family.",
            },
            "effect_size_name": {
                "logical_field": "effect size label",
                "meaning": "Name of the reported effect size.",
                "allowed_values_or_pattern": infer_allowed_values_or_pattern_from_series(column_series),
                "notes": "Examples include Hedges g, Cohen’s dz, and rank-biserial correlations.",
            },
            "effect_size": {
                "logical_field": "effect size value",
                "meaning": "Numeric effect-size estimate.",
                "allowed_values_or_pattern": "numeric",
                "notes": "Interpret together with effect_size_name.",
            },
            "notes": {
                "logical_field": "plain-language row explanation",
                "meaning": "Human-readable explanation of what the row means.",
                "allowed_values_or_pattern": "free text",
                "notes": "This is the main plain-English interpretation column in tests.csv.",
            },
        }

        if column_name in tests_exact_map:
            return tests_exact_map[column_name]

        return {
            "logical_field": "tests dataframe field",
            "meaning": "Column in the standardized direct-tests dataframe.",
            "allowed_values_or_pattern": infer_allowed_values_or_pattern_from_series(column_series),
            "notes": "No specialized rule matched this tests.csv column name.",
        }

    def infer_group_summaries_column_metadata(column_name: str, column_series: pd.Series) -> dict[str, str]:
        group_summaries_exact_map: dict[str, dict[str, str]] = {
            "inclusion_filter": {
                "logical_field": "inclusion filter",
                "meaning": "Which participant set the summary row was computed on.",
                "allowed_values_or_pattern": "included_only | all_finishers",
                "notes": "Matches the preregistered inclusion split used throughout the analysis.",
            },
            "story_condition": {
                "logical_field": "story family",
                "meaning": "Whether the summary row pools stories or refers to one story family.",
                "allowed_values_or_pattern": "pooled | firework | trolley | all",
                "notes": "Used in story-specific and pooled contrasts.",
            },
            "load_condition": {
                "logical_field": "load condition slice",
                "meaning": "Whether the summary row pools load conditions or refers to one load condition.",
                "allowed_values_or_pattern": "pooled | high | low | all",
                "notes": "Used for descriptive breakdowns by cognitive load.",
            },
            "analysis_scope": {
                "logical_field": "summary design scope",
                "meaning": "What kind of descriptive aggregation the row represents.",
                "allowed_values_or_pattern": "between_subjects_first_vignette | within_subjects_all_vignettes | within_subjects_deltas",
                "notes": "Used across tests.csv, group_summaries.csv, and integrated_blame_models.csv with consistent fine-grained value names.",
            },
            "agent_role": {
                "logical_field": "judged agent role",
                "meaning": "Whether the row summarizes the distal agent or the proximate node.",
                "allowed_values_or_pattern": "distal | proximate | clark",
                "notes": "Some transitional files may still use clark instead of distal.",
            },
            "dv": {
                "logical_field": "dependent variable",
                "meaning": "Which dependent variable the summary row refers to.",
                "allowed_values_or_pattern": "blame | wrong | punish | wrongness | punishment",
                "notes": "Transitional files may still use longer DV names.",
            },
            "condition": {
                "logical_field": "condition or delta label",
                "meaning": "Raw vignette condition for level summaries or contrast label for delta summaries.",
                "allowed_values_or_pattern": "CC | CH | DIV | CH - CC | DIV - CC | MIN(CH, DIV) - CC | CC - CH",
                "notes": "Interpret together with analysis_scope.",
            },
            "n": {
                "logical_field": "sample size",
                "meaning": "Number of usable observations contributing to the descriptive row.",
                "allowed_values_or_pattern": "numeric count",
                "notes": "Counts can differ across rows because of missingness.",
            },
            "mean": {
                "logical_field": "mean",
                "meaning": "Mean of the summarized values.",
                "allowed_values_or_pattern": "numeric",
                "notes": "For punishment under log1p_parametric, this should be on the transformed scale.",
            },
            "median": {
                "logical_field": "median",
                "meaning": "Median of the summarized values.",
                "allowed_values_or_pattern": "numeric",
                "notes": "For punishment under log1p_parametric, this should be on the transformed scale.",
            },
            "std": {
                "logical_field": "standard deviation",
                "meaning": "Standard deviation of the summarized values.",
                "allowed_values_or_pattern": "numeric",
                "notes": "For punishment under log1p_parametric, this should be on the transformed scale.",
            },
        }

        if column_name in group_summaries_exact_map:
            return group_summaries_exact_map[column_name]

        return {
            "logical_field": "group summaries field",
            "meaning": "Column in the descriptive summaries dataframe.",
            "allowed_values_or_pattern": infer_allowed_values_or_pattern_from_series(column_series),
            "notes": "No specialized rule matched this group_summaries.csv column name.",
        }

    def infer_column_metadata(
        file_key: str,
        column_name: str,
        column_series: pd.Series,
    ) -> dict[str, str]:
        if file_key == "cleaned":
            return infer_cleaned_column_metadata(column_name, column_series)
        if file_key == "tests":
            return infer_tests_column_metadata(column_name, column_series)
        if file_key == "group_summaries":
            return infer_group_summaries_column_metadata(column_name, column_series)

        return {
            "logical_field": "processed dataframe field",
            "meaning": "Processed-data column.",
            "allowed_values_or_pattern": infer_allowed_values_or_pattern_from_series(column_series),
            "notes": "",
        }

    "============================================="
    "Build one row for every column in every file."
    "============================================="
    dataframe_rows = []

    file_iteration_specifications = [
        ("cleaned", dict(general_settings["filing"]["file_names"]).get("cleaned", "responsibility_shielding_cleaned.csv"), cleaned_dataframe),
        ("tests", dict(general_settings["filing"]["file_names"]).get("tests", "responsibility_shielding_tests.csv"), tests_dataframe),
        ("group_summaries", dict(general_settings["filing"]["file_names"]).get("group_summaries", "responsibility_shielding_group_summaries.csv"), group_summaries_dataframe),
    ]

    for file_key, file_name, source_dataframe in file_iteration_specifications:
        for column_position, column_name in enumerate(source_dataframe.columns, start=1):
            column_series = source_dataframe[column_name]
            metadata = infer_column_metadata(
                file_key=file_key,
                column_name=column_name,
                column_series=column_series,
            )

            dataframe_rows.append(
                {
                    "file_name": file_name,
                    "column_name": column_name,
                    "logical_field": metadata["logical_field"],
                    "meaning": metadata["meaning"],
                    "allowed_values_or_pattern": metadata["allowed_values_or_pattern"],
                    "example_values": format_example_values(column_series),
                    "notes": metadata["notes"],
                    "file_order": {"cleaned": 0, "tests": 1, "group_summaries": 2}[file_key],
                    "column_position": column_position,
                }
            )

    dataframe_codebook = pd.DataFrame(dataframe_rows).sort_values(
        by=["file_order", "column_position"],
        kind="stable",
    ).drop(columns=["file_order", "column_position"]).reset_index(drop=True)

    dataframe_codebook.to_csv(file_path_codebook, index=False, encoding="utf-8-sig")

    return dataframe_codebook


