# Findings: NFL Expected Points Model

---

## Phase 4: EP Curve Comparison vs. nflfastR

### What I built

A multinomial XGBoost classifier trained on 2021-2022 NFL play-by-play data to predict the probability of each of six next scoring outcomes: offensive touchdown, offensive field goal, no score, defensive touchdown, offensive safety, and defensive safety. Expected Points is the probability-weighted sum of point values across all six outcomes.

### What the comparison showed

Our EP curves are significantly flatter than nflfastR's across all four downs. The model produces EP values in roughly the 4.5-5.0 range regardless of field position, while nflfastR shows a clear slope from roughly 6.0 near the opponent end zone down to near zero or negative at the offense's own end zone. The numeric summary confirmed the gap: MAE of 4.047 in own territory shrinking to 0.709 in the red zone.

### Why this happened

The gap is not a bug, but ga direct consequence of how `next_score` is defined and what features are available.

`next_score` in this model is the next scoring event of the entire game, not the current drive. A team punting from their own 1 yard line still gets an `off_td` label if their defense gets a stop and their offense scores a touchdown three drives later. At the game level, field position on the current play has limited predictive power for who scores next because so much can happen in between.

nflfastR closes this gap by including features this model does not have: drive context, possession-adjusted variables, and game script indicators. Those features give their model the ability to distinguish good field position from bad field position even when predicting game-level next score. Without them, this model learns that `off_td` is the most common next scoring event at roughly 49% of plays and anchors most predictions near that base rate regardless of where the ball is.

### What would fix it

Two approaches would produce EP curves with the correct slope:

1. Redefine `next_score` as the next scoring event of the current drive only. This gives field position direct predictive power but changes the meaning of EP from the standard football analytics definition.

2. Add drive-level context features: yards per carry on the current drive, completion percentage, time of possession, and turnover history. These are available in the nflverse play-by-play data and would give the model the ability to distinguish high-probability scoring situations from low-probability ones at the same field position.

### What this model gets right

The calibration curves for `off_fg` track the diagonal closely in opponent territory and the red zone, which are the zones where field goal probability matters most. The model correctly learned that field goals are the dominant outcome in the 20-40 yard range and touchdowns dominate inside the 10. The class probabilities are reasonable at the play level. The structural limitation is in how EP aggregates those probabilities across field position, not in the individual class predictions themselves.

---

## Phase 5: Fourth Down Application

### What the results showed

The go-for-it and field goal columns produced directionally correct results. Short yardage in opponent territory is almost always EP-positive for going for it. The field goal column correctly reflects make probability rising as distance drops, with FG becoming the best option inside the opponent's 35.

### Known limitation: punt column

The punt EP column is not a reliable output.

The model assigns roughly 4.8-5.0 expected points to nearly every scrimmage situation regardless of field position. When opponent EP after a possession change is negated to get the kicking team's EP, the result is approximately -4.8 to -5.0 in every situation. A punt from your own 25 and a punt from the opponent's 40 look identical to the model.

The go-for-it column is more trustworthy because it depends primarily on conversion probability and the EP of a successful drive, not on field position differentiation. The FG column is the most reliable of the three because it uses an external make percentage formula rather than model-derived EP for the possession change outcome.

Punt comparisons should be disregarded until a drive-level next score definition is implemented.
