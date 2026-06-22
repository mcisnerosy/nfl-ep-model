# Reading Notes: Expected Points Model
**Sources:** Burke (Advanced Football Analytics, 2010) | nflfastR EP/WP Models (Open Source Football, 2021)

---

## What Is Expected Points

Expected Points (EP) is how many points the offense is projected to score before the next change of possession. It is calculated using three inputs: down, yards to go, and field position. Every unique combination of those three things has a historical average attached to it.

Expected Points Added (EPA) is the change in EP from one play to the next. If a play moves the offense from a situation worth 0.5 EP to one worth 1.2 EP, that play generated 0.7 EPA. EPA is what gets assigned to individual plays and players.

---

## Why 2nd and 3rd Down EP Is Harder to Calculate

First down has a massive sample. There are thousands of 1st and 10 snaps at every yard line going back decades, so averaging next score outcomes is stable.

Second and third down situations multiply the variable space. You have to account for yards to go on top of field position, which means far fewer examples per bucket. A specific situation like 2nd and 7 from the 43 has limited historical examples, so simple averaging breaks down.

---

## How the Model Calculates EP

The model does not predict a raw point total directly. It predicts the probability of each of six possible next scoring outcomes:

| Outcome | Point Value |
|---|---|
| Offensive touchdown | +6.96 |
| Defensive touchdown | -6.96 |
| Offensive field goal | +3 |
| Offensive safety | -2 |
| Defensive safety | -2 |
| No score | 0 |

EP is the weighted sum of those probabilities:

```
EP = (6.96 × P(off TD)) + (3 × P(off FG))
   + (-6.96 × P(def TD)) + (-2 × P(off safety)) + (-2 × P(def safety))
   + (0 × P(no score))
```

Note: touchdowns are valued at 6.96 rather than 7 because nflfastR records the PAT as a separate play. The 6.96 value approximates the expected points of a touchdown including the likely PAT outcome.

---

## Why nflfastR Switched to XGBoost

Burke's original approach fit smoothing curves across down-distance-field position situations. That works reasonably well but struggles with sparse, nonlinear interactions, particularly end-of-half situations where time, score, and timeouts all interact in ways a curve cannot capture cleanly.

nflfastR switched to XGBoost because tree-based methods handle those nonlinear interactions without requiring the modeler to specify the relationship in advance. It improved calibration on edge cases while maintaining accuracy across standard situations.

---

## What Calibration Means

Calibration measures whether a model's predicted probabilities match real-world frequencies. If the model assigns 30% probability to an offensive touchdown across a set of plays, then roughly 30% of those plays should actually result in an offensive touchdown.

A model can have decent accuracy but poor calibration. In football analytics that distinction matters because EP values feed directly into decisions. A miscalibrated model produces EP numbers that look precise but are systematically wrong in certain situations.

Calibration is checked by bucketing predicted probabilities and comparing them to actual outcome rates. A perfectly calibrated model plots as a straight diagonal line.

---

## What the EP Chart Shows

EP is highest on 1st down and lowest on 4th down at any given yard line. More downs remaining means more opportunities to move the ball and score.

EP increases as the offense approaches the opponent's end zone, but not linearly. The curve steepens sharply inside the 20 yard line.

All four down curves converge near the opponent's goal line. When the offense is at the 1 yard line, the separation between 1st and 4th down matters far less than the field position itself.

---

## Questions to Carry Into the Build

- How do I derive the next score variable from raw play-by-play data rather than pulling a pre-built column?
- At what point in a drive does a score get assigned back to a play? How do turnovers factor in?
- How should I handle end-of-half plays where the next score is structurally different from mid-drive plays?
- Where will my CFB model diverge from nflfastR's NFL model and why?

---

*Last updated: June 2026*
