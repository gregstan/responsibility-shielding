from __future__ import annotations
from pathlib import Path
from typing import TypedDict, Dict, Any


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
    codebook : Path

class TableNames(TypedDict):
    table_1_participant_counts: Path
    table_2_means_by_dv_and_condition: Path
    table_3_primary_distal_blame_contrasts: Path
    table_4_story_specific_distal_blame_contrasts: Path
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
    one_tailed: bool

class PunishSettings(TypedDict, total=False):
    analysis_mode: str
    bootstrap_iterations: int
    random_seed: int

class GeneralSettings(TypedDict):
    filing: Filing
    visuals: Visuals
    misc: MiscSettings
    punish: PunishSettings


