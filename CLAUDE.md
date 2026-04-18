# CLAUDE.md — Responsibility Shielding

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
analysis.py                          ← main analysis script; single entry point
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

---

## 4. How to run

```bash
python analysis.py
```

That reproduces all manuscript outputs. Default settings in `general_settings` are the paper's
settings. To force regeneration of all outputs from scratch, set `force_rebuild = True` in
`general_settings["misc"]`. By default, functions skip regeneration if a saved CSV already exists.

---

## 5. Architecture of `analysis.py`

The script has ten major sections in this order:

1. **Type aliases** — TypedDicts (`GeneralSettings`, `Filing`, `Visuals`, `MiscSettings`, etc.)
2. **Preprocessing** — raw Qualtrics cleaning, parsing, exclusion logic
3. **Core analysis helpers** — shared utilities (statistical tests, formatters, loaders/savers)
4. **Core analysis functions** — direct tests, descriptives, triangulation, correlations, regressions
5. **Integrated models** — OLS/repeated-measures models; first-vignette condition × story; within-subject
6. **Visualization helpers** — shared Plotly utilities
7. **Data visualization** — figure-generating functions in paper order
8. **Table generation** — manuscript table builders (draw from both analysis layers)
9. **Common variables / settings** — `general_settings` dict
10. **`main()`** — runs the pipeline in full

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
