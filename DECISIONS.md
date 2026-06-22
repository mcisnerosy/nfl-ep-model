# Decisions Log: NFL Expected Points Model

Every modeling decision made in this project, why it was made, and what was considered but rejected.

---

## Phase 1: Data

**Why 2021-2023 and not further back?**
Three seasons gives roughly 106,000 scrimmage plays, enough for a stable multinomial model. Going further back introduces era differences: rule changes, pace of play, and scoring environment have shifted enough that 2015 data looks different from 2023. Keeping the window recent keeps the model relevant to the current game.

**Why train on 2021-2022 and validate on 2023?**
Standard k-fold cross-validation is wrong for this data because plays within a game are not independent. A model validated on a random 20% sample of the same seasons it trained on will look better than it actually is. Temporal validation, training on earlier seasons and testing on a later one, simulates real deployment conditions.

**Why filter to run and pass plays only?**
EP applies to scrimmage situations where a team has a down, yards to go, and a field position. Kickoffs, punts, and extra points do not have that context. Including them would add noise without adding signal.

---

## Phase 1: Next Score Variable

**Why derive next score from the full dataset instead of scrimmage plays only?**
Scoring events happen on all play types, not just runs and passes. A field goal happens on a field goal play. If I ran the next score lookup on scrimmage plays only, a run on 3rd and 2 followed by a field goal would never see the field goal in the lookup and would find the wrong next score. Running the lookup on all 149,021 plays and joining back to scrimmage plays ensures no scoring event gets missed.

**Why six outcome categories instead of seven?**
The original plan called for seven outcomes using +7 and -7 for touchdowns. The nflfastR data records touchdowns as 6 points with the PAT tracked separately. The data never shows a +7 score change on a single play, so the categories were updated to match what actually exists in the data. The EP calculation still uses 6.96 as the touchdown point value in the weighted sum, which approximates the true value of a touchdown including the expected PAT.

**Why use pd.isna() instead of checking for None?**
Pandas StringArray stores missing values as `pd.NA`, not Python `None` or `float('nan')`. Checking `is not None` on a StringArray returns True for missing values because `pd.NA` is not `None`. This caused every play to be assigned the wrong next score. `pd.isna()` correctly handles all forms of missing values across pandas dtypes.

---

## Phase 2: Feature Engineering

**Why log1p on ydstogo?**
`ydstogo` is right-skewed and the strategic difference between 1 and 5 yards to go is much larger than the difference between 15 and 20. Log compression reflects that. The raw skewness of roughly 2.5 drops to 0.5-0.8 after log1p. This is consistent with how Burke (2009) and the nflfastR team handle it.

**Why one-hot encode down instead of treating it as ordinal?**
The jump from 2nd to 3rd down changes play-calling behavior fundamentally. Conversion pressure on 3rd down is categorically different from 2nd down, not just incrementally worse. Encoding down as 1/2/3/4 forces the model to assume a linear, equal-spacing relationship between downs. One-hot encoding drops that constraint. `down_1` is dropped as the reference category to avoid multicollinearity in logistic regression.

**Why include a squared yardline term?**
`off_td` probability rises sharply inside the 20 yard line. A linear field position term cannot capture that step change on its own. Adding `yardline_100_sq` gives the model a way to fit that nonlinearity without requiring a manual spline or zone flag. `is_red_zone` provides an additional explicit signal at the 20 yard line boundary.

**Why StandardScaler instead of MinMaxScaler?**
MinMaxScaler compresses everything to [0, 1] but is sensitive to outliers. A 50-point blowout pushes `score_differential` to the edge and distorts the scale for all other plays. StandardScaler is more robust to outliers and produces zero-mean unit-variance features, which stabilizes gradient descent for logistic regression and makes coefficients directly comparable. For XGBoost, scaling does not affect performance, but I kept the same feature set across both models for consistency.

**Why exclude qtr?**
An initial version of Phase 2 included `qtr_scaled`. In Phase 3, `qtr_scaled` dominated feature importance at 44x the gain of `yardline_100_scaled`. It was acting as a game script proxy, encoding information about score state and time pressure that suppressed the field position signal entirely. Removing it let the model weight field position and down correctly. `half_seconds_remaining` captures time context without the game script conflation.

**Why fit the scaler on the full dataset in Phase 2?**
Phase 2 is producing the engineered dataset for inspection. The scaler is refit exclusively on 2021-2022 training data in Phase 3 before being applied to the 2023 val set. Fitting on the full dataset in Phase 2 is fine for EDA but would be a data leak if carried into model training.

---

## Phase 3: Model Training

**Why a temporal train/val split instead of random k-fold?**
Plays within a game are correlated: same teams, same weather, same game script. Random k-fold would let the model see 2023 game context during training, which makes val performance look better than it is. Holding out the full 2023 season tests whether the model generalizes forward in time, which is the only direction it will ever predict in practice.

**Why early stopping on XGBoost?**
The first run without early stopping showed val log-loss bottoming out at round 150 and rising through round 299. Early stopping with a 25-round patience window saved the round 146 checkpoint instead of the overfit round 299 version. The difference is small but there is no reason to keep a worse model.

**Why is the XGBoost margin over logistic regression small?**
XGBoost improved log-loss by 0.026 over logistic regression. The features explain most of the signal: down, distance, field position, and game context are strong predictors and logistic regression can extract most of that linearly. XGBoost adds value at the margins, particularly on `no_score` and `off_fg`, likely by capturing red zone nonlinearity and field goal range interactions that a linear model misses.

---

## Phase 4: Validation

**Why are the EP curves flat and what would fix it?**
`next_score` is defined as the next scoring event of the entire game, not the current drive. A team punting from their own 1 yard line still gets an `off_td` label if their offense scores three drives later. At the game level, field position on the current play has limited predictive power for who scores next because so much can happen in between.

nflfastR closes this gap with drive context features, possession-adjusted variables, and game script indicators that I deliberately excluded to keep the feature set defensible. Without them the model anchors near the `off_td` base rate of 49% regardless of field position.

Two approaches would fix this in a future version. First, redefine `next_score` as the next scoring event of the current drive only. This gives field position direct predictive power but changes the meaning of EP from the standard football analytics definition. Second, add drive-level context features: yards per carry on the current drive, completion percentage, and time of possession. These are available in the nflverse play-by-play data and would give the model the ability to distinguish high-probability scoring situations from low-probability ones at the same field position.

---

## Phase 5: Fourth Down Application

**Why use a conversion rate approximation instead of fitting one from the data?**
I used a simple linear approximation: `max(0.10, 0.75 - 0.06 * (ydstogo - 1))`. This gives roughly 75% on 4th and 1, 51% on 4th and 5, and floors at 10% for long distances. A more rigorous approach would fit conversion rates directly from the play-by-play data by distance bucket. The approximation is sufficient for a portfolio demonstration and is labeled as such in the notebook.

**Why use an external formula for field goal make percentage?**
The model was not trained to predict field goal outcomes directly. Using model EP for the possession change after a missed field goal would have compounded the flat curve problem. An external distance-based make percentage formula isolates the FG column from that limitation and makes it the most interpretable output in the results table.

**Why assume a fixed 40 yard punt net?**
Real punt outcomes vary by punter, game situation, and return. A fixed assumption is standard in EP models at this level of complexity and is noted in the notebook. Improving this would require either a separate punt outcome model or position-specific averages from the data.
