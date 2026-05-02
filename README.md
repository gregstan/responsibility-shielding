# Responsibility Shielding
### Greg Stanley

This repository contains the code, data products, manuscript tables, figures, study materials, and preregistration for my paper on **responsibility shielding**: the tendency to assign less blame to a causally distal enabler when a morally responsible proximate actor stands between the enabling act and the harmful outcome.

The repository is designed to be **easy to inspect immediately** and **easy to reproduce from scratch**. All default settings in the main analysis script are the settings used in the paper. The repo includes the final processed outputs, manuscript tables, and interactive Plotly figures, but all of them can also be rebuilt directly from the code.

---

## Start here

If you only open a few things, start with these:

- [`README.md`](README.md) — this file
- [`analysis.py`](analysis.py) — the main analysis script
- [`tables/`](tables/) — manuscript tables generated directly by the analysis pipeline
- [`visuals/`](visuals/) — interactive Plotly figures
- [`processed/`](processed/) — saved intermediate and analysis CSVs
- [`shielding_starter_pack.md`](shielding_starter_pack.md) — a compact conceptual overview of the project
- [`paper/`](paper/) — manuscript and supplementary materials (if included in the final repo)

---

## What this repository is for

This is not a general-purpose software toolkit. It is a **paper-linked analysis repository** designed so that another researcher can:

1. understand the logic of the experiment,
2. inspect the final outputs immediately,
3. rerun the full pipeline with one click,
4. compare the generated tables and figures directly to the manuscript,
5. and use the code as a transparent record of how the reported results were produced.

The code is intentionally organized for readability rather than minimalism.

---

## One-click reproduction

### Option A: VS Code (recommended)
1. Open this repository in **VS Code**
2. Create or activate a Python environment
3. Install the required packages from `requirements.txt`
4. Open [`analysis.py`](analysis.py)
5. Click **Run Python File**

The script’s `main()` function is designed so that the default settings reproduce the paper’s outputs.

### Option B: terminal
If you prefer, you can also run:

```bash
python analysis.py
```

---

## What the script does

At a high level, the analysis pipeline does the following:

1. loads or rebuilds the cleaned dataframe from the raw Qualtrics export,
2. filters responses to the preregistered / manuscript-relevant collection window,
3. removes unnecessary identifying Qualtrics columns from the cleaned dataframe,
4. runs the core confirmatory and exploratory analyses,
5. runs the integrated model analyses,
6. generates all manuscript and supplementary tables,
7. generates the Plotly figures in the order they appear in the paper,
8. saves all outputs to the repo before returning them.

By default, most analysis functions **reuse existing CSV outputs** if they already exist. To force regeneration, set `force_rebuild = True` either globally in `general_settings["misc"]` or locally in the function call.

---

## Repository map

### Main files
- [`analysis.py`](analysis.py) — main analysis script
- [`requirements.txt`](requirements.txt) — Python dependencies
- [`shielding_starter_pack.md`](shielding_starter_pack.md) — conceptual overview of the phenomenon and study
- [`preregistered_early_draft.txt`](preregistered_early_draft.txt) — Early draft of paper written in 2024.
  - Shows the extent to which I thought through this study prior to data collection. Part of OSF prefegistration.
- [`preregistration.txt`](preregistration.txt) — preregistration text
- [`qualtrics_file.qsf`](qualtrics_file.qsf) — Qualtrics study file 

### Main folders
- [`raw_data/`](raw_data/) — raw or de-identified raw data exports
- [`processed/`](processed/) — saved cleaned dataframes and analysis CSVs
- [`tables/`](tables/) — manuscript-facing tables
- [`visuals/`](visuals/) — interactive Plotly figures (`.html`)
- [`images/figures/`](images/figures/) — static figure images that appear in the paper
- [`images/stimuli/`](images/stimuli/) — illustrations that accompanied the vignettes

---

## Code Book
- [`processed/responsibility_shielding_processed_codebook.csv`](processed/responsibility_shielding_processed_codebook.csv)
  - Explains how to interpret the main preprocessed CSV files: cleaned, group_summaries, and tests
  - Maps all column names to example values and the meanings of those fields 

---

## How the repo maps onto the manuscript

The repository is organized so that the paper’s outputs can be matched directly to files in the repo.

### Main text tables
- **Table 1** → [`tables/Table_1_Participant_Counts.csv`](tables/Table_1_Participant_Counts.csv) → participant counts by condition
- **Table 2** → [`tables/Table_2_Means_by_DV_and_Condition.csv`](tables/Table_2_Means_by_DV_and_Condition.csv) → participant counts by condition
- **Table 3** → [`tables/table_3_Primary_Distal_Blame_Contrasts.csv`](tables/table_3_Primary_Distal_Blame_Contrasts.csv) → primary Clark-blame contrasts
- **Table 4** → [`tables/table_4_Story_Specific_Distal_Blame_Contrasts.csv`](tables/table_4_Story_Specific_Distal_Blame_Contrasts.csv) → story-specific Clark-blame contrasts
- **Table 5** → [`tables/table_5_Two_Alternative_Forced_Choice_Distribution.csv`](tables/table_5_Two_Alternative_Forced_Choice_Distribution.csv) → 2AFC response distribution

### Supplementary tables
- **Table 6** → [`tables/table_6_Within_Subject_Pairwise_Blame_Matrix.csv`](tables/table_6_Within_Subject_Pairwise_Blame_Matrix.csv) → within-subject pairwise blame matrix
- **Table 7** → [`tables/table_7_Cognitive_Load_Blame_Contrasts.csv`](tables/table_7_Cognitive_Load_Blame_Contrasts.csv) → cognitive-load contrasts
- **Table 8** → [`tables/table_8_Order_Effects_Summary.csv`](tables/table_8_Order_Effects_Summary.csv) → vignette-position / order effects
- **Table 9** → [`tables/table_9_Secondary_DV_Contrasts.csv`](tables/table_9_Secondary_DV_Contrasts.csv) → secondary dependent variables

### Main text figures
- **Figure 1** → [`images/figures/figure_1.png`](images/figures/figure_1.png) → Schematic of three main conditions *(static only — not generated by analysis.py)*
- **Figure 2** → [`images/figures/figure_2.png`](images/figures/figure_2.png) → Illustrations that accompany vignettes *(static only — not generated by analysis.py)*
- **Figure 3** → [`images/figures/figure_3.png`](images/figures/figure_3.png) → Blame ratings by vignette conditions
- **Figure 4** → [`images/figures/figure_4.png`](images/figures/figure_4.png) → Distribution of shielding effects
- **Figure 5** → [`images/figures/figure_5.png`](images/figures/figure_5.png) → Roadmap to future experiments *(static only — not generated by analysis.py)*

### Supplementary figures
- **Figure 6** → [`images/figures/figure_6.png`](images/figures/figure_6.png) → Cognitive load violin plot
- **Figure 7** → [`images/figures/figure_7.png`](images/figures/figure_7.png) → Order-effects line graph
- **Figure 8** → [`images/figures/figure_8.png`](images/figures/figure_8.png) → Blame versus wrongness

> Interactive HTML versions of Figures 3, 4, and 6–8 live in [`visuals/`](visuals/)
> Double-click any Plotly `.html` file in `visuals/` to open it in a browser as an interactive figure.
> To save a static image, use the download button in the Plotly toolbar in your browser tab.

### Experimental Stimlui - Vignette Illustrations
- **Trolley -  Choice-Choice** → [`images/stimuli/rshielding_trolley_choice-chance_dark_mode.png`](images/stimuli/rshielding_trolley_choice-chance_dark_mode.png) 
- **Trolley -  Choice-Chance** → [`images/stimuli/rshielding_trolley_choice-choice_dark_mode.png`](images/stimuli/rshielding_trolley_choice-choice_dark_mode.png) 
- **Trolley -  Division** →      [`images/stimuli/rshielding_trolley_division_dark_mode.png`](images/stimuli/rshielding_trolley_division_dark_mode.png) 
- **Firework - Choice-Choice** → [`images/stimuli/rshielding_fireworks_choice-chance_dark_mode.png`](images/stimuli/rshielding_fireworks_choice-chance_dark_mode.png) 
- **Firework - Choice-Chance** → [`images/stimuli/rshielding_fireworks_choice-choice_dark_mode.png`](images/stimuli/rshielding_fireworks_choice-choice_dark_mode.png) 
- **Firework - Division** →      [`images/stimuli/rshielding_fireworks_division_dark_mode.png`](images/stimuli/rshielding_fireworks_division_dark_mode.png) 

---

## Main script architecture

The major sections of [`analysis.py`](analysis.py) are demarcated in the file and appear in this order:

1. **Type aliases**
2. **Preprocessing**
3. **Core analysis helpers**
4. **Core analysis functions**
5. **Integrated models**
6. **Visualization helpers**
7. **Data visualization**
8. **Table generation**
9. **Common variables**
10. **`main()`**

The ordering of the figure-generating functions follows the order of the figures in the paper.

---

## Philosophy of `general_settings`

The repository uses a single bundled settings object, `general_settings`, to keep the analysis pipeline explicit and reproducible.

This object stores:
- `filing` — file names and file paths,
- `visuals` — plotting styles and Plotly layout defaults,
- `misc` — manuscript-relevant default analysis settings, rebuild behavior, and data-freeze timestamps,
- `punish` — settings for analyzing punishment data (bootstrap iterations, random seed, analysis mode).

The philosophy is simple:

- **Global defaults live in one place**
- **Function-specific options are passed only when they differ from the defaults**
- **The repo’s default settings are the paper’s settings**

This makes it easier to see exactly what assumptions and file locations the pipeline is using.

---

## Common patterns in the code

Most analysis functions follow the same pattern:

1. if `force_rebuild` is `None`, inherit the default from `general_settings["misc"]["force_rebuild"]`
2. try to load an existing output CSV from the `processed/` folder
3. if no saved CSV exists, load or build the cleaned dataframe
4. run the analysis
5. save the output CSV to `processed/`
6. return the dataframe

This means the pipeline is both:
- **reproducible from scratch**, and
- **fast for repeated inspection once outputs already exist**

Other common patterns:

- Many core analysis functions run for both:
  - `included_only`
  - `all_finishers`
- Many also iterate over:
  - story condition
  - cognitive load
  - first-vignette condition
- Plotly figure functions rely on the shared `figure_layout` settings
- Most visualization functions can be filtered by:
  - inclusion status
  - story condition
  - cognitive load
  - dependent variable
  - figure type (e.g., violin vs. boxplot)

---

## Core analyses vs. integrated models

One possible source of confusion is that the repo contains **two complementary analysis layers**.

### Core analysis functions
These produce the direct tests, summaries, triangulation results, correlations, and other outputs that are closest to the preregistered and descriptive analyses.

### Integrated models
These produce model-based summaries of the same design, including:
- first-vignette condition × story models
- within-subject repeated-measures models
- compact pooled and story-specific contrasts
- omnibus position/story interaction tests

### Table generation
The manuscript tables are generated by functions that draw from **both layers**. This is deliberate:

- direct tests are often the cleanest source for preregistered rows,
- integrated models are often the cleanest source for story- and position-adjusted rows.

The repo therefore keeps the raw outputs conceptually separate and integrates them only at the manuscript-table stage.

---

## Data freeze window

The analysis intentionally excludes any responses outside the final collection window used for the paper.

The preprocessing stage keeps only responses whose Qualtrics timestamp falls between:

- `freeze_timestamp_first = "2/19/2026 10:57:56 PM"`
- `freeze_timestamp_last  = "3/20/2026 10:00:09 AM"`

This protects reproducibility. If additional respondents take the study after data collection ended, those rows are not included in the manuscript analyses.

---

## Data privacy / de-identification

The cleaned dataframe intentionally drops unnecessary identifying Qualtrics columns before being saved publicly.

If you are opening this repo from GitHub:
- the **processed outputs** are the intended public-facing data products
- I removed the identifying columns from the raw Qualtrics data file 

If you plan to reuse the code for another project, I recommend reviewing any raw Qualtrics export for:
- IP addresses
- latitude/longitude
- external references
- or any other potentially identifying metadata

---

## Main output CSVs

The most important CSVs live in [`processed/`](processed/).

- `responsibility_shielding_cleaned.csv`  
  Cleaned participant-level dataframe used by downstream analyses

- `responsibility_shielding_tests.csv`  
  Direct confirmatory and exploratory tests

- `responsibility_shielding_group_summaries.csv`  
  Condition means, counts, standard deviations, and related summaries

- `responsibility_shielding_integrated_blame_models.csv`  
  Integrated-model outputs for Clark blame, including contrasts and omnibus tests

- `responsibility_shielding_2afc_counts_table.csv`  
  Forced-choice response counts and table-form output

- other processed CSVs  
  Correlations, regressions, triangulation outputs, and manuscript table inputs

Many of these files include a column that explains, in plain language, what the row means.

<details>
<summary><strong>Common recurring columns</strong></summary>

- `inclusion_filter`  
  Usually `included_only` or `all_finishers`

- `story_condition`  
  Usually `pooled`, `firework`, or `trolley`

- `load_condition`  
  Usually `pooled`, `high`, or `low`

- `design` 
  Indicates if the contrast is between-subjects (first vignettes) or within-subjects (all vignettes) 

- `agent_role`  
  Indicates if the row analyzes contrasts for the `distal` agent Clark or the `proximate` agent Bill

- `dv`
  Indicates if the dependent variable is `blame`, `wrong`, or `punish`

- `contrast_type`  
  Labels like `CH - CC`, `DIV - CC`, or `CH - DIV`

- `analysis_scope`  
  Design scope, used consistently across `tests.csv`, `group_summaries.csv`, and `integrated_blame_models.csv`.  
  Values: `between_subjects_first_vignette`, `within_subjects_all_vignettes`, `within_subjects_deltas`, `within_subjects_repeated_measures`.

- `notes`  
  Plain-language description of what the row represents (appears in output CSVs)

- `meaning`  
  Plain-language description of what a column represents (appears in the codebook)

</details>

---

## Most important functions

<details>
<summary><strong>Key functions in the main script</strong></summary>

### `load_or_build_cleaned_dataframe(...)`
The entry point for the cleaned participant-level dataframe. Loads an existing cleaned CSV if available; otherwise rebuilds it from raw data.

### `load_analysis_dataframe(...)`
Shared loader used by downstream analysis functions to retrieve already-saved CSV outputs.

### `_save_analysis_dataframe_to_processed_folder(...)`
Shared saver used across the pipeline to write CSV outputs to `processed/`.

### `run_confirmatory_and_exploratory_tests(...)`
The main direct-test engine. Produces the core test CSV used by many manuscript tables and supplementary results.

### Core analysis functions
These compute descriptive summaries, triangulation results, cognitive-load breakdowns, correlations, regressions, and other direct-test outputs.

### Functions contributing to `compute_integrated_distal_blame_results(...)`
These build the integrated first-vignette and within-subject blame model outputs used for story-specific and model-based summaries.

### `plot_ratings_by_vignette_condition(...)`
Generates the main condition-level distributions used in the paper’s rating plots.

### `plot_participant_level_shielding_heatmap(...)`
Generates the participant-level shielding map / heatmap.

### `generate_manuscript_and_supplementary_tables(...)`
Runs the table-building layer and saves manuscript-facing tables to `tables/`.

</details>

---

## Study materials and conceptual overview

This repo includes materials beyond the code itself:

- [`shielding_starter_pack.md`](shielding_starter_pack.md)  
  A compact overview of the concept, design, and logic of the study

- preregistration text file(s)  
  Included for transparency

- Qualtrics study file  
  Included for transparency and reuse (if present in the final repo)

Readers are also welcome to provide the starter pack, manuscript, and code to an AI assistant to help navigate the repo.

---

## More real-world manifestations of shielding

If you are interested in the broader significance of the phenomenon, I especially recommend the supplementary section on **more real-world manifestations of shielding**. It contains examples that help connect the formal experimental design to practical moral and social situations.

---

## Requirements

Install the required packages listed in [`requirements.txt`](requirements.txt).

If you want a clean environment, a typical workflow is:

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
pip install -r requirements.txt
```

---

## Defaults and reproducibility

All default settings in this repo are the settings used in the paper.

That includes:
- inclusion/exclusion behavior
- freeze timestamps
- default rebuild behavior
- default manuscript table settings
- default Plotly layout and figure styling

The repository is intended so that a fresh rerun of the pipeline reproduces the manuscript-facing outputs as closely as possible.

---

## Robot participant experiment

The `robot_experiment/` directory runs AI models through the same experimental protocol human participants received. Each robot instance reads the vignettes, provides blame/wrongness/punishment ratings, and completes the full Qualtrics page sequence (cognitive load, comprehension checks, 2AFC, CRT, INDCOL). Output goes to `robot_raw_data/` in the same CSV format as the human raw data, so `preprocessing.py` processes it without modification. After data collection, the runner automatically prints Tables 2/3/4/9 using the same analysis functions as `analysis.py`.

**Supported models:** Claude, GPT-4o, Gemini, Grok, DeepSeek, and Ollama (local/free). List multiple models in `ROBOT_EXPERIMENT_CONFIG["models"]` to run them all in one call.

**Quick start:**
```bash
pip install anthropic openai google-generativeai aiohttp python-dotenv
cp .env.example .env          # then paste API keys into .env
python robot_experiment/run_robot_participants.py   # beta_mode=True by default (3 participants)
```

See [AGENTS.md](AGENTS.md) §10 for full documentation of the design, protocol fidelity choices, config options, and instructions for adding new providers.

---

## Citation, license, and contact

If you use this repository, please cite the paper and/or repository once the public citation information is finalized.

- License: see [`LICENSE`](LICENSE)
- Citation metadata: see [`CITATION.cff`](CITATION.cff)

Author: **Greg Stanley**
