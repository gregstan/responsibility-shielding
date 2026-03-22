# Responsibility Shielding Starter Pack

**Purpose of this starter pack:**  
This document is a compact orientation guide for a new reader, human or AI, who needs to get up to speed on the *Responsibility Shielding* project without reading the full manuscript first. It summarizes the core concept, design, results, analysis logic, and repo structure at a level that should make the paper and codebase much easier to navigate.

---

## 1) Core phenomenon in one paragraph

**Responsibility shielding** is the tendency to assign less blame to a causally distal enabler when a morally responsible proximate actor stands between that enabling act and the harmful outcome. The defining signature is **counterfactual**, not merely interpersonal: the key question is not whether the proximate actor is blamed more than the distal actor, but whether the distal actor is blamed **less** when the intermediate link is a responsibility-capable agent than when that same link is not a suitable target of moral responsibility.

---

## 2) What the phenomenon is and is not

### It is **counterfactual**, not merely interpersonal
A common byproduct of shielding is an interpersonal blame disparity, such as *Bill > Clark*. But that is not the phenomenon itself. The phenomenon is the **change in blame toward Clark** across matched causal structures.

### It is **sequential**, not merely multi-agent
Responsibility shielding is defined over a **sequential** causal chain: a distal agent enables, then a proximate agent triggers. This is different from simultaneous-choice blame sharing.

### It is **social**, not merely causal
The crucial manipulation is whether the proximate node is a morally responsible agent. This makes shielding a phenomenon of **moral responsibility attribution**, not just domain-general causal attribution.

---

## 3) Central theoretical distinction

The core conceptual contrast is between:

- **Intervening causation** as the broader legal / historical precursor
- **Responsibility shielding** as a narrower psychological label for the specific counterfactual blame pattern isolated here

The point is **not** to rename the whole intervening-causation tradition. The point is to mark a more specific psychological pattern within that broader tradition.

---

## 4) Responsibility capacity

Shielding depends on whether the intermediate node is treated as a proper target of moral responsibility. I call that property **responsibility capacity**.

In the paper, responsibility capacity is treated as depending at least on:
- **agency**: the ability to choose among meaningful alternatives and carry out that choice
- **veridicality**: sufficient understanding of what one is doing and what the likely consequences are

This is a descriptive folk-psychological construct, not a metaphysical theory of free will.

---

## 5) Experimental backbone: the three conditions

Each vignette instantiates one of three causal structures involving:
- **Clark** = distal enabler
- **Bill** (or a non-agentic process) = proximate link
- a harmful outcome that occurs only if the relevant enabling and triggering events both occur

### CC = Choice–Choice
Sequential two-agent structure. Clark enables. Bill, a mentally competent morally responsible agent, later chooses whether to trigger the harm.

### CH = Choice–Chance
Sequential structure in which the proximate node is **not** a suitable target of moral responsibility. In the firework story, this is Bill with severe acquired brain injury; in the trolley story, it is a computerized switching mechanism.

### DIV = Division
Simultaneous-choice two-agent conjunctive structure. Clark and Bill each choose without knowing the other’s choice. Harm occurs iff both act.

---

## 6) Why DIV matters

DIV is a control for the alternative explanation that lower blame toward Clark in CC is merely a consequence of **blame being divided across multiple responsible people**. DIV matches the presence of two responsible agents but removes the sequential shielding structure.

The strongest version of the phenomenon is therefore not just:
- **CH > CC**

but also:
- **DIV > CC**

That second contrast is what the paper calls **shielding beyond division of responsibility**.

---

## 7) Design logic

Several design choices are central:

- **Conjunctive harm structure across conditions:** harm requires the relevant enabling and triggering actions in CC, CH, and DIV.
- **Matched foreseeability and prevention opportunities:** Clark’s expected contribution to harm is held constant across versions.
- **Independent blame scales:** blame is not forced into a zero-sum allocation across agents.
- **Illustrated game-tree structures:** participants see causal diagrams in addition to text.
- **Strict comprehension gating:** confirmatory analyses use only participants who understood the key structural features.

---

## 8) Story families

The experiment uses two structurally isomorphic story families:

### Firework story family
Clark is a clerk who can arm a dangerous flare canister for Bill, who may ignite it over a stadium crowd.

### Trolley story family
Clark is an operator whose switch can route a trolley toward a second fork, where Bill or a mechanism determines whether a worker is harmed.

The trolley family helps reduce one alternative explanation present in the firework CH condition: that participants might blame Clark extra for exploiting a mentally impaired person rather than because shielding disappears when the proximate node lacks responsibility capacity.

---

## 9) Sample and exclusion logic

Final dataset in the current manuscript:
- **432** completed participants
- **282** included participants after preregistered comprehension-based exclusion

The main confirmatory analyses use **included participants**. Parallel analyses on **all completers** are reported as robustness checks.

---

## 10) Main dependent measures

### Primary
- **Clark blameworthiness** (9-point Likert scale)

### Secondary
- wrongness
- punishment (years in prison)
- 2AFC comparisons

### Manipulation check
- proximate blame in CC vs CH

---

## 11) Confirmatory vs complementary analyses

### Preregistered confirmatory analyses
The preregistration specifies two first-vignette contrasts on Clark blame, pooled across story family and cognitive load:
- **CH − CC**
- **DIV − CC**

These are the confirmatory family, with **Holm correction** across the two planned tests.

### Complementary integrated models
The paper also reports:
- a first-vignette **condition × story** model
- a within-subject repeated-measures model with **condition, story family, and vignette position**

These models are used to summarize the design more compactly, provide story-specific decompositions, and reduce reliance on a parade of isolated pairwise tests.

---

## 12) Final main results

### Confirmatory between-subjects first-vignette results
Among included participants:

- **CH − CC = 1.03**, 95% CI **[0.55, 1.51]**, Holm-corrected **p = 7.86e-05**
- **DIV − CC = 0.67**, 95% CI **[0.16, 1.18]**, Holm-corrected **p = 0.0108**

So both preregistered contrasts are positive and survive correction.

### Complementary within-subject repeated-measures results
Among included participants:

- **CH − CC = 0.99**, 95% CI **[0.78, 1.20]**, **p = 1.28e-20**
- **DIV − CC = 0.60**, 95% CI **[0.39, 0.80]**, **p = 1.07e-08**

The within-subject ordering is the same:
- **CH > DIV > CC**

---

## 13) What the results mean

The strongest conclusion is:

1. **Core shielding exists**: Clark is blamed less in CC than in CH.
2. **Shielding is not exhausted by generic blame division**: Clark is also blamed less in CC than in DIV.
3. **The pattern appears both between subjects and within persons.**

The paper therefore treats:
- **CH − CC** as the core shielding contrast
- **DIV − CC** as shielding beyond division of responsibility

---

## 14) Story-family pattern

The main pattern appears in both story families, though magnitudes vary somewhat. The firework CH effect is somewhat larger, but the effect is not confined to one vignette frame.

This is useful because it suggests the phenomenon is not merely an artifact of one surface story.

---

## 15) 2AFC triangulation

The forced-choice items converge with the rating data.

Among included participants:
- most lean **CH > CC**
- a smaller majority lean **DIV > CC**
- most lean **Bill > Clark**

These categorical judgments are treated as triangulation rather than as the primary inferential engine.

---

## 16) Cognitive load, order, and secondary DVs

### Cognitive load
High load produced directionally larger shielding effects, but not clearly so. The paper now treats this as a **weak mechanism probe** rather than strong evidence for a heuristic account.

### Vignette position
Blame drifts slightly upward for later-position vignettes, but the central **CH > DIV > CC** ordering remains. The omnibus condition × position interaction is weak.

### Wrongness and punishment
The same qualitative pattern extends beyond blame, though the clearest evidence remains the preregistered blame analyses.

---

## 17) Heuristic interpretation

A live interpretation in the paper is that shielding may partly reflect a **stop-search heuristic**:

1. observe harm
2. search backward for a responsible agent
3. assign blame to the first responsibility-capable proximate actor encountered
4. search less, or stop, leaving upstream enablers relatively shielded

The paper also discusses an adaptive account based on **ambiguity amplification**: in the real world, distal actors are often genuinely less informed about downstream specifics, so discounting them may often be ecologically sensible even if it overgeneralizes in the present experimental setting.

---

## 18) Broader significance

The paper argues that responsibility shielding matters because many real harms unfold through enabling chains. Examples discussed include:
- negligent entrustment
- incitement and doxing
- technology and product design
- addiction and drug supply chains
- parental liability
- AI delegation

The core normative concern is that shielding may systematically underweight the responsibility of people who make harms possible in the first place.

---

## 19) Code / repo architecture

The repo is organized around one main analysis script plus data/output folders.

Main script sections:
1. type aliases / typed dictionaries
2. preprocessing
3. core analysis helpers
4. core analysis functions
5. integrated models
6. visualization helpers
7. data visualization
8. table generation
9. common variables / settings
10. `main()`

The repo uses a bundled `general_settings` object to centralize:
- file paths and file names
- plotting settings
- rebuild behavior
- freeze timestamps
- manuscript-default settings

---

## 20) Common code patterns

Most analysis functions:
1. inherit `force_rebuild` from `general_settings` if not specified
2. try to load an existing CSV from `processed/`
3. if needed, rebuild from the cleaned dataframe
4. save their CSV before returning

Many functions run over:
- `included_only`
- `all_finishers`

Many also iterate over:
- story family
- first-vignette condition
- cognitive load

---

## 21) Integrated models vs core analysis functions

The repo intentionally separates:
- **core analysis outputs** (direct tests, summaries, triangulation)
- **integrated model outputs** (story-adjusted and repeated-measures model-based contrasts)

The manuscript table generator draws from **both** layers.

This means:
- some main-text table rows come from `tests.csv`
- some come from the integrated-model CSVs

That is deliberate, not a mistake.

---

## 22) Reproducibility / freeze window

The final analysis window is bounded by two timestamps:
- `freeze_timestamp_first = "2/19/2026 10:57:56 PM"`
- `freeze_timestamp_last  = "3/20/2026 10:00:09 AM"`

Any later Qualtrics rows are excluded from the manuscript analyses.

The cleaned dataframe also drops unnecessary identifying Qualtrics columns before being saved.

---

## 23) Main repo outputs

The repo is designed so a reader can either inspect outputs immediately or regenerate them.

Important folders:
- `processed/` → cleaned dataframe and analysis CSVs
- `tables/` → manuscript-facing tables
- `visuals/` → interactive Plotly figures
- `images/` → static pngs / thumbnails
- `paper/` → manuscript and supplement (if included)

---

## 24) Elevator pitch

Responsibility shielding is a moral attribution pattern in which upstream enablers receive a blame discount when a responsibility-capable proximate actor stands between their enabling choice and the harm. The paper tests this using preregistered first-vignette contrasts, complementary integrated models, and triangulation across ratings, 2AFC judgments, and auxiliary measures, while tightly controlling causal structure and expected harm.

---

## 25) What this starter pack is for

This file is meant to help a new reader—or a new AI chat—start from the current state of the project without needing to reconstruct the logic from scattered notes. It is not a substitute for the paper, but it should make the paper and codebase far easier to navigate.

