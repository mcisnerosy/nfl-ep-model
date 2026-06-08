# Decisions Log: NFL Expected Points Model

Every modeling decision made in this project, why it was made, and what was considered but rejected. This is the document to read before asking "why did you do it this way."

---

## Data

**Why 2021-2023 and not further back?**
Three seasons gives roughly 106,000 scrimmage plays, enough for a stable multinomial logistic regression. Going further back introduces era differences: rule changes, pace of play, and scoring environment have shifted enough that 2015 data looks different from 2023. Keeping the window recent keeps the model relevant to the current game.

**Why train on 2021-2022 and validate on 2023?**
Standard k-fold cross-validation is wrong for this data because plays within a game are not independent. A model validated on a random 20% sample of the same seasons it trained on will look better than it actually is. Temporal validation, training on earlier seasons and testing on a later one, simulates real deployment conditions.

**Why filter to run and pass plays only?**
EP is a concept that applies to scrimmage situations where a team has a down, yards to go, and a field position. Kickoffs, punts, and extra points do not have that context. Including them would add noise without adding signal.

---

## Next Score Variable

**Why derive next score from the full dataset instead of scrimmage plays only?**
Scoring events happen on all play types, not just runs and passes. A field goal happens on a field goal play. If we ran the next score lookup on scrimmage plays only, a run on 3rd and 2 followed by a field goal would never see the field goal in the lookup and would find the wrong next score. Running the lookup on all 149,021 plays and then joining back to scrimmage plays ensures no scoring event gets missed regardless of what type of play it happened on.

**Why six outcome categories instead of seven?**
The plan originally called for seven outcomes using +7 and -7 for touchdowns. The nflfastR data records touchdowns as 6 points with the PAT tracked as a separate play. The data never shows a +7 score change on a single play, so the categories were updated to reflect what actually exists in the data. The EP calculation still uses 7 as the touchdown value in the weighted sum since that is the true point value of a touchdown including the PAT.

**Why use pd.isna() instead of checking for None?**
Pandas StringArray stores missing values as pd.NA, not Python None or float nan. Checking `is not None` on a StringArray returns True for missing values because pd.NA is not None. This caused every play to be assigned the wrong next score. pd.isna() correctly handles all forms of missing values across pandas dtypes.

---

## Model

*Decisions to be added during Phase 3.*

---

## Validation

*Decisions to be added during Phase 4.*