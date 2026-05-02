# AGENTS.md — Responsibility Shielding

This file orients an AI collaborator to the project, codebase, and collaboration style.
Read this before writing or suggesting any code changes.

---

## 1. What this project is

**Responsibility shielding** is a moral psychology phenomenon: people assign less blame to a causally
distal enabler when a morally responsible proximate actor stands between the enabling act and the
harmful outcome. The key signature is *counterfactual* — it is not merely that the proximate actor
is blamed more, but that the distal actor is blamed *less* when the intermediate link is a
responsibility-capable agent than when that same link is not.

The paper has been submitted (to *JDM* but has now been submitted to *Cognition*). This repo is the 
complete, reproducible record of that paper's analyses. A follow-up study using economic games is 
planned, and this repo may eventually connect to that work.

For a deeper orientation, read [`shielding_starter_pack.md`](shielding_starter_pack.md) first, then
the [`README.md`](README.md).

---

## 2. Domain vocabulary

Understanding these terms is necessary for reading the code and contributing correctly.

| Term | Meaning |
|------|---------|
| **CC** | Choice–Choice: Clark enables, Bill (a morally responsible agent) chooses to trigger |
| **CH** | Choice–Chance: Clark enables, but the proximate node lacks responsibility capacity |
| **DIV** | Division: simultaneous conjunctive structure — Clark and Bill each choose independently |
| **Clark** | The distal enabler — the agent whose blame is the primary dependent variable |
| **Bill** | The proximate actor — a morally responsible agent in CC, not in CH |
| **distal** | Refers to Clark's role and perspective throughout the code |
| **proximate** | Refers to Bill's role throughout the code |
| **responsibility capacity** | Whether an agent is a proper target of moral responsibility (requires agency + veridicality) |
| **shielding effect** | CH − CC or DIV − CC on Clark blame; the core contrast of interest |
| **shielding beyond division** | DIV − CC; rules out generic blame-sharing as an explanation |
| **included_only** | Participants who passed preregistered comprehension exclusion criteria |
| **all_finishers** | All participants who completed the study (used as robustness checks) |
| **first_vignette** | Between-subjects analysis using only each participant's first vignette |
| **within_subject** | Repeated-measures analysis using all three vignettes per participant |
| **pooled** | Aggregated across story families (firework + trolley) |
| **firework** | The firework story family (Clark arms a canister, Bill or a mechanism ignites it) |
| **trolley** | The trolley story family (Clark routes a trolley, Bill or a mechanism triggers harm) |
| **blame / wrong / punish** | The three dependent variable types (blame, wrongness, years in prison) |
| **2AFC** | Two-alternative forced-choice triangulation items |
| **CRT** | Cognitive Reflection Test — used to index cognitive load manipulation |
| **INDCOL** | Individualism–collectivism scale — an exploratory covariate |

---

## 3. Key files and folders

```
analysis.py       ← entry point: general_settings dict + main(); run with python analysis.py
config.py         ← TypedDicts (GeneralSettings, FileNames, FilePaths, etc.)
preprocessing.py  ← Qualtrics cleaning, parsing, exclusion, file I/O helpers
core.py           ← core analysis helpers/functions + integrated models
visualization.py  ← Plotly visualization helpers + all figure-generating functions
tables.py         ← manuscript table builders (Tables 1–9)

shielding_starter_pack.md            ← compact conceptual + design overview (read this first)
README.md                            ← full repo guide with table/figure map
requirements.txt                     ← Python dependencies
preregistration.txt                  ← preregistration text
preregistered_early_draft.txt        ← early manuscript draft (part of OSF preregistration)
qualtrics_file.qsf                   ← Qualtrics study file

raw_data/                            ← raw (de-identified) Qualtrics export
processed/                           ← cleaned participant dataframe + all analysis CSVs
tables/                              ← manuscript-facing tables (Table 1–9)
visuals/                             ← interactive Plotly figures (.html)
images/figures/                      ← static PNGs used in the paper
images/stimuli/                      ← illustrated causal diagrams shown to participants
```

The codebook lives at:
`processed/responsibility_shielding_processed_codebook.csv`

**Import flow (no circular dependencies):**
```
config.py      →  (nothing from this project)
preprocessing  →  config
core           →  config, preprocessing
visualization  →  config, preprocessing
tables         →  config, preprocessing, core
analysis       →  config, preprocessing, core, visualization, tables
```

---

## 4. How to run

```bash
python analysis.py
```

That reproduces all manuscript outputs. Default settings in `general_settings` are the paper's
settings. To force regeneration of all outputs from scratch, set `force_rebuild = True` in
`general_settings["misc"]`. By default, functions skip regeneration if a saved CSV already exists.

---

## 5. Architecture

The pipeline is split across six Python files. The original ten logical sections now map to files:

| Original section | File |
|---|---|
| Type aliases | `config.py` |
| Preprocessing + file I/O helpers | `preprocessing.py` |
| Core analysis helpers + functions + integrated models | `core.py` |
| Visualization helpers + data visualization | `visualization.py` |
| Table generation | `tables.py` |
| Common variables + `main()` | `analysis.py` |

### `general_settings`

One bundled dict controls everything: file paths, file names, plotting defaults, rebuild behavior,
freeze timestamps, and manuscript-relevant defaults. This is the single source of truth for pipeline
configuration.

### Two-layer analysis design

The repo intentionally keeps two complementary layers separate:

- **Core analysis functions** → direct tests, summaries, triangulation (closest to preregistered
  analyses); outputs go to `processed/tests.csv`, `group_summaries.csv`, etc.
- **Integrated models** → OLS and repeated-measures model-based contrasts; story-adjusted summaries

The **table generators** draw from both layers. Some manuscript rows come from `tests.csv`; others
come from the integrated-model CSVs. This is deliberate, not an inconsistency.

### Common function pattern

Most analysis functions follow this sequence:
1. Inherit `force_rebuild` from `general_settings["misc"]` if not explicitly passed
2. Try to load an existing CSV from `processed/`
3. If no CSV exists, load or build the cleaned dataframe
4. Run the analysis
5. Save the output CSV to `processed/`
6. Return the dataframe

---

## 6. Output → manuscript map

| Manuscript item | File |
|----------------|------|
| Table 1 | `tables/Table_1_Participant_Counts.csv` |
| Table 2 | `tables/Table_2_Means_by_DV_and_Condition.csv` |
| Table 3 | `tables/table_3_Primary_Distal_Blame_Contrasts.csv` |
| Table 4 | `tables/table_4_Story_Specific_Distal_Blame_Contrasts.csv` |
| Table 5 | `tables/table_5_Two_Alternative_Forced_Choice_Distribution.csv` |
| Table 6–9 | `tables/table_6_*` through `table_9_*` (supplementary) |
| Figures 3–8 | `visuals/*.html` (interactive); `images/figures/*.png` (static) |
| Core test CSVs | `processed/responsibility_shielding_tests.csv` |
| Cleaned data | `processed/responsibility_shielding_cleaned.csv` |

---

## 7. Coding style — read this carefully

This is the most important section for any AI collaborator. Greg's style is distinctive and should
be preserved exactly. Do not "clean up" or rewrite code to match generic Python conventions.

### Long, descriptive variable names

Variable and argument names are intentionally long and self-documenting. There are no abbreviations
beyond established domain shorthand (e.g., `cc`, `ch`, `div`, `dv`).

```python
"Good"
first_vignette_condition_means_by_story = ...
distal_agent_clark_blame_rating = ...
included_participants_only_dataframe = ...

"Bad — do not write these"
fv_means = ...
clark_rating = ...
incl_df = ...
```

### Docstrings with bullet-point argument lists

Docstrings use a specific format: a plain-English description, then an `Arguments:` block where
each argument is introduced with a bullet point `•`, and a `Returns:` block.

```python
def example_function(first_argument: pd.DataFrame, second_argument: bool) -> pd.DataFrame:
    """
    One-sentence description of what the function does or longer is fine too.

    Arguments:
        • first_argument:
            What this argument is and how it is used.
        • second_argument:
            What this argument controls.

    Returns:
        • Description of the return value.
    """
```

### Comments as plain strings, not `#` lines

Comments are written as standalone string literals on their own line, not with `#`. This is
intentional and should be maintained throughout the codebase.

```python
"Good — this is how comments are written in this repo"
snake_string = "_".join([chunk for chunk in snake_string.split("_") if chunk != ""]).lower()

# Bad — do not introduce hash comments
```

### Design philosophy

The overriding goal is **maximum readability with minimum working memory overhead**. A reader
should be able to understand what any line does without holding context from twenty lines earlier.
This means:

- Prefer clarity over brevity
- Name things for what they *are*, not what they *do*
- Let the variable name carry the meaning so comments are rarely needed
- Do not introduce abstractions or helpers beyond what the task requires

---

## 8. Data freeze window

The manuscript analyses are restricted to responses between two timestamps:

```python
freeze_timestamp_first = "2/19/2026 10:57:56 PM"
freeze_timestamp_last  = "3/20/2026 10:00:09 AM"
```

Any Qualtrics rows outside this window are excluded. Do not change these timestamps.

---

## 9. Future work context

A follow-up study is planned that validates and models responsibility shielding using **economic
games**. That study is in early planning and may eventually be connected to this repo. When working
on extensions or new analyses, keep this downstream use case in mind: the core constructs, naming
conventions, and file patterns established here will likely carry forward.

---

## 10. Robot participant experiment

The `robot_experiment/` directory contains a pipeline for running AI models as participants in the
responsibility shielding experiment. Each robot instance is exposed to the same vignettes, questions,
and protocol flow that human participants received in Qualtrics, then provides structured numerical
ratings. The output is written in a format that `preprocessing.py` can process with zero
modification.

After data collection, the runner automatically executes the same core analysis functions used for
human data (Tables 2, 3, 4, 9) and prints results to the terminal. All robot analysis outputs go to
`robot_raw_data/processed/` and `robot_raw_data/tables/` — the human `processed/` folder is never
touched. Raw data is appended to (never overwritten) unless `overwrite_raw_data=True` is set
explicitly in `ROBOT_EXPERIMENT_CONFIG`.

### Files

```
robot_experiment/
├── stimuli.py                  ← all vignette text, question text, and column mappings (from QSF)
├── model_clients.py            ← provider abstraction: Claude, OpenAI, Gemini, Grok, DeepSeek, Ollama
└── run_robot_participants.py   ← experiment runner; edit ROBOT_EXPERIMENT_CONFIG to configure a run

robot_raw_data/                 ← output CSVs (parallel to raw_data/)
robot_raw_data/processed/       ← robot analysis outputs (parallel to processed/)
robot_raw_data/tables/          ← robot tables (parallel to tables/)
.env                            ← API keys (gitignored; copy from .env.example)
.env.example                    ← template showing which keys are needed
```

### Supported models

Add any of these to the `"models"` list in `ROBOT_EXPERIMENT_CONFIG`. Provider is auto-detected
from the model name via `MODEL_TO_PROVIDER` in `model_clients.py`.

| Model name | Provider | Cost per 50 participants |
|---|---|---|
| `"claude-sonnet-4-6"` | Anthropic | ~$3–8 |
| `"claude-opus-4-7"` | Anthropic | ~$15–30 |
| `"gpt-4o"` | OpenAI | ~$5–12 |
| `"gpt-4o-mini"` | OpenAI | ~$0.50–1 |
| `"gemini-2.0-flash"` | Google | ~$0–1 |
| `"grok-3"` | xAI | ~$2–5 |
| `"deepseek-chat"` | DeepSeek | ~$0.50 |
| `"mistral"` | Ollama (local) | free |

### How to run

```bash
# beta test (3 participants per model, prints full transcripts)
python robot_experiment/run_robot_participants.py

# full run — edit ROBOT_EXPERIMENT_CONFIG first:
#   set beta_mode=False, n_participants_per_model=50
#   add more models to the "models" list as needed
python robot_experiment/run_robot_participants.py
```

### Key config options

| Option | Default | Meaning |
|---|---|---|
| `models` | `["claude-sonnet-4-6"]` | List of models to run in one call |
| `n_participants_per_model` | `10` | Participants per model (full run) |
| `beta_n_participants` | `3` | Participants per model (beta) |
| `overwrite_raw_data` | `False` | If False (default), appends to existing CSV |
| `run_analysis_after_collection` | `True` | Prints Tables 2/3/4/9 after data collection |
| `run_models_sequentially` | `True` | Finishes one model before starting the next |

### Design notes

- **Protocol fidelity**: the robot sees the exact participant-facing text from `qualtrics_file.qsf`
  in the same page sequence (welcome → cognitive load → vignettes → proximate ratings → comprehension
  → 2AFC → CRT → INDCOL). Each Qualtrics page = one conversation turn; the robot cannot revise
  earlier answers.
- **Blinding**: the system prompt assigns the robot a participant role without mentioning the
  experimental hypothesis. Each API call is stateless by default.
- **Variance**: use `temperature=1.0` (default). At temperature 0, all runs would give identical
  responses, making N > 1 meaningless.
- **Comprehension checks**: robot instances should pass all three. The `included` flag in the cleaned
  dataframe will be True for any instance that answers all three correctly.
- **Individual differences**: CRT, INDCOL, and cognitive load are included as-is (not persona
  manipulations). Run robots as themselves; persona injection is a separate extension.
- **Column compatibility**: the output CSV uses the same column names as the human raw data export,
  plus a `model_name` column (robot-only; NaN for human participants). The only other intentional
  difference is `indcol_hi_1_1` (correct spelling) vs. `incdol_hi_1_1` (typo in the original
  Qualtrics export).

### Adding a new provider

Subclass `ModelClient` in `model_clients.py`, implement `async chat(messages) -> str`, add the
new class to `get_client()`, and add the model-to-provider mapping to `MODEL_TO_PROVIDER`. Any
provider with an OpenAI-compatible API (Grok, DeepSeek, and most others) can reuse `OpenAIClient`
with a custom `base_url`.
