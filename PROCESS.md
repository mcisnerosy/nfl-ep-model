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

## Phase 3: Model Training

Loaded the engineered dataset and split into train (2021-2022, 70,912 plays)
and val (2023, 35,474 plays) using a temporal split on the season column.
Refit the StandardScaler on train only before applying it to val.

Trained two models: multinomial logistic regression with lbfgs solver and
XGBoost with multi:softprob objective. Added early stopping with a 25-round
patience window to XGBoost after an initial run showed val loss bottoming out
around round 150 and creeping back up through round 299. Best checkpoint was
round 146.

Final log-loss on the 2023 val set: logistic regression 0.9235, XGBoost 0.8979.
XGBoost won by 0.026. The margin is smaller than expected, which suggests the
features are carrying most of the predictive signal and the nonlinear
interactions are limited. Calibration curves for off_td and off_fg track the
diagonal reasonably well through the middle probability range. no_score shows
the most calibration drift at higher predicted probabilities but is not a
primary driver of EP values. Saved both models, the scaler, and the label
encoder to models.

---

## Phase 4: Validation

## Phase 4: Validation

Loaded the 2023 val set and the artifacts saved in Phase 3. Pulled the
nflfastR ep column from the raw scrimmage CSV saved in Phase 1 rather than
re-pulling from the nflverse servers, which timed out during initial attempts.
Merged on game_id and play_id to get a clean one-to-one join at the play level.

Ran calibration curves broken out by field position zone for off_td and off_fg.
off_fg tracks the diagonal closely in opponent territory and the red zone, which
are the zones where field goal probability matters most for EP. off_td is
systematically overconfident across all zones -- when the model predicts 60%
touchdown probability it happens only about 55% of the time.

Built a synthetic grid of neutral situations (10 yards to go, tied game, second
quarter) to generate smooth EP curves by field position and down for our model.
Compared against nflfastR EP values averaged from actual 2023 val set plays.

Our EP curves are essentially flat across the field, ranging only about 0.2
points from own territory to the red zone. nflfastR shows a steep slope from
roughly 6.0 near the opponent end zone down to near zero or negative in own
territory. The numeric summary confirmed the gap: MAE of 4.047 in own
territory shrinking to 0.709 in the red zone.

Diagnosed the root cause: next_score is defined as the next scoring event of
the entire game, not the current drive. Without drive context features the
model anchors near the off_td base rate of 49% regardless of field position.
Documented in FINDINGS.md as a known limitation with a clear explanation of
what features would fix it.

---

## Phase 5: Fourth Down Application

Built a fourth down decision tool on top of the trained XGBoost model. The
goal was to translate model outputs into something a coach or analyst could
actually use: given a fourth down situation, which option returns the most
expected points.

Wrote a feature construction function that takes raw inputs (down, distance,
yard line, score differential, seconds remaining) and applies the same scaling
pipeline used in Phase 3. Wrote a separate EP calculation function that
multiplies predicted probabilities by point values for each outcome class.

Built decision logic for three options. Going for it uses an estimated
conversion probability weighted against EP after success and EP after a failed
attempt. Field goal uses a distance-based make percentage formula weighted
against EP after a made kick and EP after a miss. Punt assumes a 40 yard net
and computes opponent EP from the new field position, negated.

Ran the tool across 15 common situations and built a breakeven chart showing
EP(go) minus EP(punt) by yard line and distance. Exported results to
data/fourth_down_results.csv.

Identified during output review that the punt column was returning
approximately -4.8 in every situation regardless of field position. Traced
the cause back to the flat EP curve from Phase 4. The model assigns nearly
identical EP to all field positions, so possession change math after a punt
produces the same result whether you punt from your own 25 or the opponent's
40. Documented in FINDINGS.md. The go-for-it and field goal columns are
reliable. Punt comparisons are not.