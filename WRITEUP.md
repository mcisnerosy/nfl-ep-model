# Building an NFL Expected Points Model From Scratch

---

## Why I Built This

Expected Points is the foundation most football analytics work sits on top of: EPA, win probability, fourth down decision tools. I wanted to build one from scratch rather than pulling nflfastR's pre-built `ep` column, because understanding how the labels get derived and where a model like this breaks matters more to me than having a clean number to point at.

This writeup covers the methodology, the results, and a limitation I found during validation that I think is more interesting than if everything had worked cleanly the first time.

---

## The Goal

Predict the probability of each possible next scoring outcome for any down, distance, field position, score, and time situation. Use those probabilities to compute Expected Points. Apply the model to fourth down decisions, where the choice between going for it, punting, and kicking a field goal can be evaluated directly in EP terms.

---

## Data

I pulled three seasons of play-by-play data (2021-2023) using `nfl_data_py`, which sources from the nflverse repository. That gave 149,021 plays. After filtering to scrimmage plays only, runs and passes with a valid down and field position, I had 106,386 plays to work with.

I trained on 2021-2022 and held out 2023 entirely for validation. This is a temporal split rather than random k-fold. Plays within a game are correlated: same teams, same weather, same game script. A random split lets the model see future game context during training, which makes validation performance look better than it would in actual deployment. Holding out a full season tests whether the model generalizes forward in time, which is the only direction it will ever be asked to predict.

---

## Deriving the Outcome Variable

This was the part I cared most about getting right, because nflfastR's `ep` column doesn't show you how the labels were built. I derived `next_score` directly from the raw data:

1. Computed the score change on every play from the offense's perspective.
2. Classified each score change into one of six outcomes: offensive touchdown, offensive field goal, offensive safety, defensive touchdown, defensive safety, or no score.
3. For every play, scanned forward through the rest of the game to find the next scoring event and assigned that as the label.
4. Ran this lookup on the full 149,021-play dataset, not just scrimmage plays, since scores can happen on any play type. Joined the result back to the scrimmage subset.

One bug surfaced here that's worth mentioning because it's a pandas-specific trap: a column ended up stored as a StringArray, which represents missing values as `pd.NA`. My initial null check used `is not None`, which returns `True` for `pd.NA` since it isn't Python's `None`. Every play was getting mislabeled with the very next play's event instead of the actual next score. Switching to `pd.isna()` fixed it. I mention this because it's the kind of bug that produces a model that trains and runs without errors but learns the wrong thing entirely, and it wouldn't show up until validation.

---

## Features

Nine features went into the final model:

- Down, one-hot encoded with 1st down as the reference category. Down is categorical, not ordinal: the jump from 2nd to 3rd down changes play-calling in a way that a linear 1-2-3-4 encoding can't represent.
- Yards to go, log-transformed before scaling. The difference between 1 and 5 yards to go matters more than the difference between 15 and 20, and the raw distribution is heavily right-skewed.
- Field position, both linear and squared terms, plus a binary red zone flag. Touchdown probability rises sharply inside the 20, and a single linear term can't capture that without help.
- Score differential and seconds remaining in the half, both standardized.

I excluded quarter as a feature after an early version showed it dominating feature importance at roughly 44 times the gain of the field position term. It was acting as a proxy for game script rather than contributing real signal, and it was suppressing the field position signal the model actually needed.

---

## Model

I trained two models and compared them on validation log-loss:

| Model | Log-Loss |
|---|---|
| Multinomial Logistic Regression | 0.9235 |
| XGBoost | 0.8979 |

XGBoost won, but by a smaller margin than I expected going in. That tells me the features I chose carry most of the linear signal already, since logistic regression got most of the way there on its own. XGBoost's advantage shows up mainly in `no_score` and `off_fg` predictions, where red zone nonlinearity and field goal range interactions matter more.

---

## What I Found in Validation

When I compared my model's EP curve to nflfastR's, mine was flat. Across the entire field, EP ranged about 0.2 points. nflfastR's curve has the slope you'd expect: roughly 6.0 near the opponent end zone, dropping to near zero or negative near the offense's own end zone.

The cause traces back to the outcome variable itself. I defined `next_score` as the next scoring event of the entire game, not the current drive. A team punting from their own 1 can still get labeled `off_td` if their defense gets a stop and their offense scores three drives later. At the game level, current field position has limited predictive power over who scores next, because so much football happens in between. nflfastR's model includes drive context and possession-adjusted features that close that gap. Mine deliberately didn't, to keep the feature set small and defensible, and that tradeoff is what produced the flat curve.

I could have caught this earlier by comparing against a reference EP curve before getting to the fourth down application. I didn't, and it surfaced downstream instead, in the punt comparisons in Phase 5. The punt EP came back at roughly -4.8 in every single situation regardless of field position, because the model can't tell the difference between an opponent starting at their own 20 versus their own 40. Tracing that back to the root cause in the outcome variable was the most useful part of the validation process.

---

## Fourth Down Application

Despite the flat curve issue, the go-for-it and field goal columns held up. Short yardage situations in opponent territory were go-for-it positive across the board, and field goals became the better option inside the opponent's 35 as kick distance shortened. Those conclusions match what you'd expect from any reasonable EP model and from what teams already do on short yardage in scoring range.

The punt column is the one part of the output I don't trust, and I say so directly in the project documentation rather than smoothing over it. A model that produces a clean number isn't the same as a model that's right, and I'd rather flag where mine isn't than present results that look more finished than they are.

---

## What I'd Do Differently

Two changes would fix the flat curve directly. Either redefine `next_score` at the drive level instead of the game level, which gives field position direct predictive power but changes what EP means relative to the standard definition used across the industry. Or keep the game-level definition and add drive context features, like current-drive yards per carry, completion percentage, and time of possession, which would let the model separate high-probability scoring drives from low-probability ones at the same field position.

If I rebuilt this, I'd compare against a reference EP curve right after Phase 3, before building anything downstream of it. That would have caught the flat curve a phase earlier and saved the rework in the fourth down application.

---

## What This Project Demonstrates

I can derive a labeled dataset from raw play-by-play data without relying on a pre-built target variable, defend every feature engineering decision with a reason rather than a default, diagnose a structural problem in a model's output back to its root cause in the data, and document a limitation honestly instead of hiding it behind a clean-looking result.

---

**Repo:** https://github.com/mcisnerosy/nfl-ep-model