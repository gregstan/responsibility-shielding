# Responsibility Shielding Starter Pack (paste into a new chat)

**Purpose of this starter pack:**  
This is a long-form context dump so a new chat can immediately “pick up where we left off” on the *Responsibility Shielding* project—definitions, theory, experimental design, preregistration logic, measurement strategy, analysis plan, key decisions, and how to frame the manuscript.

---

## 0) Who Greg is and why this project matters (context + motivation)

**Greg Stanley** is a University of Michigan Psychology Ph.D. candidate working in moral psychology / cognitive science / behavioral game theory (and building an online multiplayer experiment platform). Greg has been on the academic job market and wants this project to be a major research “win” that’s publishable, career-relevant, and socially important.

**Why responsibility shielding matters to Greg:**  
The project is motivated by a practical moral concern: in modern life, many harms occur through *chains of enabling* (not just direct acts). If people systematically under-assign blame to upstream enablers—because a downstream agent “soaks up” responsibility—then institutions (law, public opinion, policy) may under-deter or under-sanction the actors who shape large-scale harms. Greg’s personal aim is to “deprive ultimate agents of excuses” by making the cognitive phenomenon visible and measurable.

---

## 1) The core phenomenon in one paragraph

**Responsibility shielding** is the hypothesis that people assign *less* blame to a causally upstream decision-maker when a *morally responsible* downstream agent sits between that upstream choice and the harmful outcome. The downstream agent functions like a “responsibility shield”: blame attribution tends to stop at the first salient responsible agent encountered when tracing backward from harm, rather than continuing upstream to fully credit the enabling role.  

**Key point:** the defining signature is **counterfactual**, not interpersonal. The effect is not primarily “Bill gets more blame than Clark,” but:  
> “Clark gets **less** blame when Bill is in the chain as a responsible agent than Clark gets when the intermediate step is non-agentic / non-responsible.”

---

## 2) What responsibility shielding is NOT

### Not the interpersonal blame gap
A gap like *Bill > Clark* is expected in many moral scenarios for lots of reasons (salience, personal force, last-mover advantage, knowledge/foreseeability increasing downstream). That interpersonal gap is **not** the definition of shielding.

### Not simply diffusion of responsibility / “division”
“Diffusion” is typically about multiple people sharing responsibility in some way. Your project distinguishes **sequential shielding** from **simultaneous division** by building a specific control condition (DIV).

### Not the bystander effect
Bystander effect is about helping/inaction behavior. Shielding here is about **third-party blame assignment** (moral evaluation of an upstream agent in a causal chain).

### Not merely domain-general causal attribution
People can assign “causal responsibility” to objects/events, but your claim is about **moral responsibility attribution** in *social* chains. A big conceptual theme is: shielding should depend on whether the intermediate node is perceived as a morally responsible *agent*.

---

## 3) The game-tree / node-language framework

Everything is formalized as a simple causal chain / game tree:

- **Ultimate node (upstream)**: Clark chooses whether to enable/arm something (or flip a first switch).  
- **Intermediate node (downstream/proximate)**: Bill (or a non-agentic mechanism) determines whether harm occurs.  
- Harm occurs only if the relevant enabling and triggering actions happen.

Your key structural manipulation is **what sits at the proximate node**:
- a morally responsible agent (Choice)
- a non-agentic / non-responsible process (Chance, in the “responsibility-eligibility” sense)
- or a simultaneous two-agent conjunctive structure (Division)

---

## 4) The three main conditions (the backbone of the project)

You standardized three primary causal structures:

### A) CC = Choice–Choice (sequential, “shield present”)
- Clark acts first (enables/arms/redirects).
- Bill is a mentally competent adult and acts second.
- Harm occurs if Bill triggers the harmful outcome after Clark enabled it.
- Clark has a strong incentive to enable (commission).
- Bill has a strong incentive to trigger (prize/bonus).
- Clark believes Bill will trigger with probability **p = 0.75** (from an extremely accurate risk algorithm).  
- If Bill triggers, harm is guaranteed (p(harm | Bill triggers) = 1.0).

**Intuition:** Bill is a clear morally responsible proximate agent. If shielding exists, Clark’s blame should be **lower** here than when the intermediate step is non-agentic/non-responsible.

### B) CH = Choice–Chance (sequential, “no moral shield”)
Same structure and incentives, but the proximate “link” is **not** treated as a morally responsible agent.

Operationalized differently in the two story families:
- Fireworks story: Bill has a severe acquired brain injury that prevents properly understanding/controlling actions in complex situations (so he is not a normal candidate for responsibility).
- Trolley story: the second fork is controlled by a non-agentic probabilistic mechanism (or equivalent “chance-like” intermediate).

Clark still believes the probability the harmful downstream event occurs conditional on Clark acting is **the same** (p ≈ 0.75), i.e., expected harm from Clark’s perspective is held constant across conditions.

**Prediction:** Clark gets **more** blame here than in CC, because there is no morally responsible proximate agent to “absorb” blame.

### C) DIV = Division (simultaneous, conjunctive causation; “division control”)
- Both Clark and Bill must act for harm to occur (two-key / two-person safety lock / simultaneous switches).
- They act **without knowing** the other’s choice at the time of choice (no communication).
- Harm occurs iff both activate.
- Clark still has the same incentive (commission) to activate.
- Bill still has the same incentive (prize/bonus) to activate.

**Why DIV exists:** to test whether differences between CC and CH are merely because CC has two agents involved (diffusion-like effects), versus a genuine sequential shielding phenomenon.

---

## 5) The “mud splatter” analogy you developed (useful rhetorical move)

- **Division (diffusion-ish):** two people stand side-by-side; mud splatter is “shared.”  
- **Shielding:** one person stands in front and catches the splatter; the other behind receives less.

This is a clean way to help readers not collapse shielding into “generic diffusion.”

---

## 6) What the “shielding signature” is (the counterfactual rule)

Let:
- **B_CC** = mean blame for Clark in CC (using the **first vignette** participants see; between-subjects)
- **B_CH** = mean blame for Clark in CH (first vignette; between-subjects)
- **B_DIV** = mean blame for Clark in DIV (first vignette; between-subjects)

**Confirmatory bets in the preregistration:**
- H1: **B_CH ≠ B_CC** (expected B_CH > B_CC)
- H2: **B_DIV ≠ B_CC** (expected B_DIV > B_CC)

**Responsibility Shielding Effect** was defined as:
> **min(B_CH, B_DIV) − B_CC**

Interpretation: CC is the “shield present” baseline. If either CH or DIV increases blame relative to CC, that indicates CC is “discounting” Clark.

---

## 7) Why the design leans hard on counterfactual comparisons

In real life, people rarely see counterfactual causal structures side-by-side. They see “proximate actor did harm,” and they blame them—often without noticing that the upstream enabler’s blame *might have been higher* if the proximate agent weren’t there. Your experiment makes that counterfactual legible.

This is one reason you think the phenomenon hasn’t been “cleanly detected” in prior work even if related ideas exist.

---

## 8) Normative anchor: “utilitarian sameness” across conditions

A major design principle was:  
**Clark’s expected harm (if Clark acts) is held constant across conditions**, so from an expected-utility perspective Clark should be similarly condemnable across conditions.

Therefore, **if blame differs across conditions**, it suggests a psychological heuristic or bias (or at least non-utilitarian sensitivity to structural features like agency, proximity, joint causation).

This is why your comprehension checks specifically probe:
- that Clark’s believed probability of downstream harm (conditional on acting) is the same across versions,
- and that each actor had an option that would have prevented harm.

---

## 9) Responsibility eligibility: what makes Bill a “shield” vs not

A key conceptual piece is that shielding should depend on whether the intermediate entity is perceived as a **proper target of moral responsibility**.

Folk prerequisites you emphasized:
- **Control / ability to do otherwise**
- **Knowledge / understanding of what one is doing / consequences**
- **Options in the choice set** (a real alternative that avoids harm)

You operationalize removal of responsibility eligibility via:
- severe brain injury (fireworks)
- non-agentic mechanism (trolley CH)

**Diagnostic expectation:** Bill should be blamed far less in the brain-injury / mechanism case than in the competent-agent case. That demonstrates the manipulation worked.

Related rhetorical example (to potentially mention in intro/discussion):
- The 1966 UT Austin tower shooter Charles Whitman was found on autopsy to have a brain tumor, often cited in debates about diminished responsibility / insanity defenses. You use this as a culturally familiar example of “intermediate actor’s responsibility eligibility being contested.”

---

## 10) Temporal vs causal proximity (explicit distinction)

You repeatedly distinguished:
- **Temporal proximity** (how close in time an action is to harm)
- **Causal proximity** (how close in the causal chain an action is to harm)

Responsibility shielding is hypothesized to be about **causal structure**, not merely temporal distance—though you acknowledge that temporal distance can be psychologically influential and is a potential future manipulation.

---

## 11) The “while-loop / stop-search” heuristic model (mechanism story)

Mechanism hypothesis: responsibility shielding results from a cognitive shortcut:

**Informal algorithm:**
1. When harm occurs, search backward along the causal chain for a responsible agent.
2. When you hit a clear responsible agent, assign substantial blame there.
3. Stop searching (or search less), leaving upstream agents with reduced blame.

This resembles the “modular myopia” style explanation used by Joshua Greene in other moral cognition contexts (e.g., means/side-effect patterns): cognition uses local cues and stops early rather than computing global expected utility across the whole structure.

**Mechanism prediction:** anything that reduces reflective capacity or increases reliance on heuristics should **increase shielding**.

---

## 12) Mechanism probes you included (secondary/exploratory)

### Cognitive load manipulation (high vs low)
- Digit-span style memory task.
- Prediction: high load → larger shielding effects (bigger CH−CC and/or DIV−CC; or bigger min(CH,DIV)−CC).

### Trait reflectivity (CRT)
- Prediction (exploratory): higher CRT → smaller shielding (more reflective, less heuristic).

### Culture / social orientation (INDCOL)
- Prediction (exploratory): individualism/collectivism dimensions may moderate shielding strength.

These are “cherries on top” in the narrative: the phenomenon is first established; then you explore mechanism hints.

---

## 13) Two story families (robustness / generality)

To avoid “one vignette fluke” accusations, you built **two structurally isomorphic story families**:

1) **Fireworks / pyrotechnic flare canister story**
- Stadium spectators, “high-intensity display” prize.
- Clark is the clerk who can arm/enable a dangerous flare canister.
- Bill is the buyer who ignites it over the stadium.
- Algorithm predicts probability Bill will ignite (p = 0.75).
- Harm outcome was calibrated to avoid ceiling (spectator is burned, needs treatment, fully recovers).

2) **Trolley story**
- Clark is the first switch operator.
- Bill is either the second operator (CC) or a mechanism (CH) or a simultaneous co-operator (DIV).
- Harm similarly toned down (workman knocked down, needs treatment, full recovery).

You also created **illustrations** for each condition to reduce confusion and to signal the vignettes differ.

---

## 14) Measurement strategy: triangulation across multiple DV types

You intentionally triangulated the phenomenon via:

### A) Likert blame ratings (primary)
- 9-point: 1 = not blameworthy at all; 9 = extremely blameworthy.
- Primary DV: **Clark blame**.

### B) Wrongness ratings (secondary/diagnostic)
- Same 1–9 scale for act wrongness.

### C) Punishment (years in prison) (exploratory/secondary)
- Included because it connects the phenomenon to justice-relevant outcomes.
- Recognized as multi-determined (retribution/deterrence/incapacitation), especially relevant for brain injury cases.

### D) 2AFC forced-choice items (secondary triangulation)
You added forced-choice items because they:
- reduce ceiling/floor ambiguity,
- force discrimination when Likert responses cluster,
- provide a separate format to corroborate directionality.

You used a two-stage 2AFC format:
1) forced direction (e.g., “CH story makes Clark more blameworthy” vs “CC story makes Clark more blameworthy”)
2) follow-up: “one is more blameworthy” vs “roughly equal”

This yields **four interpretable categories**, e.g.:
- **CH > CC**
- **CH ≥ CC**
- **CH ≤ CC**
- **CH < CC**

Likewise for DIV vs CC and Bill vs Clark.

---

## 15) Sacred measurement principle: blame is NOT zero-sum

A critical design choice:
- You do **not** force participants to allocate a fixed pie of blame across actors (which would artificially generate diffusion).
- Each actor is rated independently, so responsibility can in principle be “non-conserved” (both can be highly blameworthy).

This matters conceptually and rhetorically: shielding is not “because blame is finite,” but because cognition stops early.

---

## 16) Within-subject vs between-subject (and the “consistency bias” logic)

Participants see all three structures, but:

### Primary confirmatory inference uses the **first vignette encountered**
Reason: once participants see multiple versions, they may:
- try to be consistent with earlier judgments,
- infer the manipulation and “correct” themselves,
- anchor and adjust rather than evaluate freshly.

So:
- **Between-subjects (first vignette)** = confirmatory / cleanest.
- **Within-subject deltas** (CH−CC, DIV−CC within person) = secondary/exploratory corroboration.

You also explicitly anticipated:
- within-subject shielding might be weaker due to a “desire for consistency.”
- finding within-subject shielding anyway would be especially compelling (it survives side-by-side comparison).

---

## 17) Comprehension checks: what they are and why they matter

You designed comprehension items to be **minimal but diagnostic**, and used **strict exclusion** (must be perfect) for confirmatory analyses, because misunderstanding the structure makes blame comparisons uninterpretable.

You converged on true/false style checks that target exactly what must be understood:

- Clark’s probability estimate about downstream harm (conditional on Clark acting) was the same across versions.
- If Clark did not act, harm would not occur (Clark’s action is necessary).
- If Bill did not act, harm would not occur (Bill’s action is necessary).

These checks are aimed at preventing “utilitarian confounds” like:
- participants thinking Clark’s action was more dangerous in one version,
- participants thinking only one character could prevent harm,
- participants thinking harm was inevitable regardless.

---

## 18) Key Qualtrics / implementation decisions (for methods & reproducibility)

- **Forced responses** for key items.
- **Timers** to prevent ultra-fast skimming and accidental “Next.”
- **No back button** (to reduce revisiting and demand).
- **Randomization**:
  - story family assigned between-subjects
  - first vignette condition assigned between-subjects
  - order of remaining conditions counterbalanced
- Visual aids (images) to reduce “the survey is repeating” confusion.
- Participants recruited via UM subject pool / SONA; pilot data excluded from confirmatory.

---

## 19) Preregistration snapshot (what is confirmatory vs exploratory)

**Confirmatory:**
- Between-subjects first-vignette contrasts on Clark blame:
  - CH vs CC (two-tailed in preregistration, directional expectation CH > CC)
  - DIV vs CC (two-tailed, directional expectation DIV > CC)
- Correction for the two contrasts (Holm-Bonferroni).

**Exploratory / secondary:**
- CH vs DIV (direction not preregistered).
- Story family moderation (fireworks vs trolley).
- Cognitive load moderation.
- CRT / INDCOL moderation.
- Within-subject analyses (mixed models; within-subject deltas).
- 2AFC analyses.
- Punishment distributions, transformations, etc.

You also explicitly excluded:
- pilot participants before preregistration timestamp.

---

## 20) Why DIV is a “bundle” (and how you defend it)

You repeatedly wrestled with the fact that DIV differs from CC/CH in more than one way:
- simultaneity / ignorance of other’s choice
- conjunctive causation (both required)
- causal proximity location of the “joint” node

Your stance:
- In all conditions, harm requires both agents’ actions (joint necessity), so this is controlled.
- DIV is still the best practical control for “multiple agents involved” without adding many more conditions and exhausting participants.
- You acknowledge limitations transparently and propose follow-up factorial expansions (e.g., DIV-U variants) as future work.

---

## 21) Alternative game-tree you considered but did not adopt

You considered adding an extra chance node after Bill’s action (so Bill acts for sure, but harm occurs with probability p), creating a utilitarian symmetry where both agents’ expected contribution is equal across conditions.

You decided against it because:
- it complicates the vignette,
- increases cognitive load/confusion,
- risks undermining comprehension.

You keep it as a discussion/future-experiment idea.

---

## 22) Why it’s “amazing” the effect can show up even in a clear design

A theme you want to emphasize (without overselling):
- You made the causal structure unusually explicit with text + images + comprehension checks.
- That should, if anything, reduce “confused blame differences.”
- So if shielding still appears, it’s evidence that the phenomenon may be robust and not just due to ambiguity.

You also hypothesize: in the real world, shielding might be stronger because:
- upstream actors are often less salient,
- causal structure is ambiguous,
- proximate actors are vivid and center-stage,
- evidence deteriorates upstream,
- institutions reward stopping once someone “convictable” is found.

---

## 23) Real-world manifestation list (use in intro/discussion to motivate significance)

High-leverage examples that match the sequential enabling structure:

### A) Negligent entrustment
- lending a car to a drunk driver, giving a gun to someone irresponsible.
- upstream enables; downstream commits harm; upstream claims “they chose it.”

### B) Incitement / doxing / instigation
- upstream spreads target info or cues aggression; downstream performs attack/harassment.
- modern relevance: online scale, plausible deniability, “I didn’t do it; they did.”

### C) Drug supply chains / last-mile scapegoating
- street-level actor takes most blame; upstream suppliers/architects comparatively protected.

### D) Political instigation
- leaders’ rhetoric vs followers’ actions (e.g., riots).
- structural shielding: one-to-many influence + downstream agency.

### E) Technology & AI
- “users chose it” / “the model did it” as shields for designers/operators.
- The “Frankenstein” crescendo: once a creation is treated as an agent, it can become a moral shield.

These examples do double duty:
1) clarify what shielding *means* structurally  
2) show why the phenomenon matters socially

---

## 24) Suggested mathematical modeling idea to include (optional but cool)

You discussed including a simple formal model to make the concept crisp for future researchers:
- blame decays with causal distance (a decay function)
- plus a “stop-search probability” when a responsible agent is encountered
- plus a division component that reduces blame when responsibility is shared simultaneously

Even a toy model (e.g., exponential decay + stopping rule) could help formalize what counts as shielding vs division.

---

## 25) How you want to frame novelty relative to prior work (placeholder, to refine)

You discovered the idea independently, but there is adjacent literature on:
- intervening causation and mitigation of responsibility (e.g., Fincham & Shultz, 1981)
- how moral judgments can be manipulated via perceived intentions / causation (e.g., Phillips & Shaw, 2014)
- philosophical/legal discussions of proximate cause (e.g., Knobe / legal philosophy materials)

Your planned “novelty” claims (to be checked precisely against those papers):
- formalizing shielding as a **counterfactual upstream blame discount** (not just interpersonal blame allocation)
- clean game-tree style operationalization with matched expected harm
- explicit separation from diffusion via DIV condition
- triangulation across Likert + 2AFC + within/between approaches
- mechanistic probe via cognitive load
- robust replication across two isomorphic story families + images + strong comprehension constraints

---

## 26) Where the manuscript can go (high-level narrative arc)

A paper-ready story structure that fits your strongest conceptual commitments:

1) **Hook with real-world enabling cases** (entrustment / doxing / tech design)
2) Define responsibility shielding **counterfactually**
3) Formalize with game-tree language
4) Present the three conditions CC / CH / DIV and the logic of each
5) Explain why primary inference is between-subjects first vignette (consistency bias)
6) Show main results: CH vs CC; DIV vs CC; ordering; effect sizes
7) Triangulation: within-sub deltas; 2AFC patterns
8) Mechanism probes: cognitive load (and CRT/INDCOL as exploratory)
9) Discussion:
   - what shielding is, what it isn’t
   - why it matters in complex causal systems
   - limitations (DIV as bundle; student pool; effect sizes)
   - future work: Prolific replication; economic games; more factorial controls; salience/time pressure

---

## 27) Key implementation details (for future chats about code + analysis)

- You prefer **long descriptive variable names** to reduce working memory load while reading code.
- You like **descriptive docstrings** in a consistent format.
- You often use **bare strings** as “comments” instead of `#` comments.
- Data pipeline:
  - preprocess Qualtrics export → clean analysis dataframe
  - compute Included flag (Finished + 100% progress + comprehension pass)
  - compute case ordering columns (case_code_position_1/2/3)
  - compute primary between-sub dataset from first vignette only
  - compute within-sub deltas (CH−CC, DIV−CC, Bill CC−CH, etc.)
  - compress 2AFC to four-category symbols
  - produce summary tables + test output CSVs

---

## 28) One-sentence “elevator pitch” (for emails / job market)

“Responsibility shielding is a moral attribution bias where upstream enablers receive a blame discount when a downstream responsible agent stands between their action and the harm; I test it with preregistered game-tree vignettes that hold expected harm constant, separate shielding from division of responsibility, and triangulate the effect with ratings, forced-choice measures, and cognitive-load moderation.”

---

## 29) What to tell the next chat assistant you want help with

- Turning this into a crisp Introduction + literature review that accurately positions novelty vs prior work.
- Writing Results that are preregistration-faithful (confirmatory vs exploratory clearly labeled).
- Plot plan: figures that make the counterfactual signature obvious (CC baseline vs CH/DIV).
- Framing: why even moderate effect sizes matter when the structure is explicit and comprehension-checked.
- Future studies: Prolific replication, time pressure manipulation, economic games, more factorial “DIV-U” style controls.

---

**End of starter pack.**