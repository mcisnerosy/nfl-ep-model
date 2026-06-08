# Process Log: NFL Expected Points Model

This document explains what was done at each phase of the project, why each decision was made, and what problems came up along the way. Written for anyone reading the code who wants to understand the reasoning, not just the output.

---

## Phase 1: Data Collection and Next Score Derivation

### What the goal was

Every EP model needs a labeled dataset. Each play needs to know what the next scoring event was after it happened. That label is the outcome variable the model learns to predict. The goal of Phase 1 was to build that labeled dataset from raw play-by-play data without relying on any pre-built EP columns.

### Where the data came from

We used `nfl_data_py`, a Python library that pulls from the nflverse data repository on GitHub. No API key required. We pulled three seasons: 2021, 2022, and 2023. That gave us 149,021 rows, one row per play, with 396 columns covering everything from yard line to player tracking data.

The plan is to train the model on 2021 and 2022, then validate on 2023. Keeping 2023 completely held out until evaluation simulates what actually happens when you deploy a model. You train on what you have and test on what comes next.

### Filtering to scrimmage plays

The raw dataset includes every play type: kickoffs, punts, extra points, timeouts, and penalties. Most of those have no meaningful EP context. A kickoff does not have a down, a yards to go, or a field position the way a scrimmage play does.

We filtered to runs and passes from regular season and postseason games where `down` and `yardline_100` were not null. Null values in those columns were a signal that the play was not a scrimmage play. After filtering we had 106,386 plays, roughly 35,000 per season.

### How we built the next score variable

For each play we needed to know the next scoring event that happened later in the same game. The steps were:

**1. Calculate score change per play.**
For each play we computed how much the score changed from the offensive team's perspective:

```
score_change = (posteam_score_post - posteam_score) - (defteam_score_post - defteam_score)
```

A positive value means the offense scored. A negative value means the defense scored.

**2. Classify scoring events.**
We mapped score change values to six outcome labels. Touchdowns showed up as 6 and -6, not 7 and -7, because nflfastR records the PAT as a separate play. Field goals were 3 and -3. Safeties were 2 and -2. Plays with no score change got a null value.

**3. Look forward within each game.**
For each play we scanned forward through the rest of the game and found the first play where a scoring event occurred. That event became the `next_score` label for the original play. If no scoring event occurred before the end of the game, the play was labeled `no_score`.

**4. Join back to scrimmage plays.**
Once every play in the full dataset had a `next_score` assigned, we joined that column back to the 106,386 scrimmage plays.

### Problems we ran into

**Null detection bug.**
The `score_event` column ended up stored as a pandas StringArray, which represents missing values as the string `'nan'` rather than Python `None` or `float('nan')`. Our loop was checking `if events[j] is not None`, which always returned True because `'nan'` is not `None`. This caused every play to be assigned the very next play's event rather than the next actual scoring event. We fixed it by switching to `pd.isna()` which correctly identifies all forms of missing values.

**Large file rejected by GitHub.**
The saved CSV was 270MB, which exceeds GitHub's 100MB file size limit. We removed the file from git history using `git filter-branch` and added the `data/` directory to `.gitignore`. Raw data files do not belong in a repo. Anyone reproducing this work pulls the data themselves using the notebook.

### Final dataset

106,386 scrimmage plays across three seasons, each labeled with one of six next score outcomes:

| Outcome | Count |
|---|---|
| off_td | 53,418 |
| off_fg | 41,023 |
| no_score | 9,057 |
| def_td | 2,173 |
| def_safety | 497 |
| off_safety | 218 |

`off_td` and `off_fg` dominate because the offense has possession on every scrimmage play, making an offensive score the most likely next event.

---

## Phase 2: Feature Engineering

*Coming next session.*

---

## Phase 3: Model Training

*Coming after Phase 2.*

---

## Phase 4: Validation

*Coming after Phase 3.*

---

## Phase 5: Fourth Down Application

*Coming after Phase 4.*