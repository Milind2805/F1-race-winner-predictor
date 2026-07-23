# F1-race-winner-predictor
# 🏎️ F1 Race Winner Predictor

Predicts F1 race winners using Machine Learning (XGBoost, 97.72% accuracy)

## Features
- 7 seasons of data (2019-2025)
- Weather, championship standings, home circuit advantage
- Interactive Streamlit UI
- Race Simulation
- Championship Projection
- Shap Explainibilty
- Head to Head Driver Comparison


## 🏁 Prediction

The core prediction interface. Select a driver, team, and circuit, then set 
grid position, qualifying position, championship points entering the race, 
and race conditions (wet/dry, air temp, track temp, home race). The model 
returns a win/no-win classification along with a win probability, displayed 
as both a percentage and a gauge chart.

Built on an XGBoost classifier trained on 2019–2025 race data (qualifying 
position, grid position, championship standings, weather, and home-circuit 
advantage as features), with SMOTE applied to the training set to correct for 
the natural class imbalance between race winners and non-winners.

## 📊 Season Analysis

Visualizes a full season's points progression and race-win distribution. 
Select any season from 2019–2025 to see:

- **Top 5 drivers' cumulative points** across the season, race by race
- **Race wins by driver**, ranked in a horizontal bar chart

Useful for sanity-checking the training data itself — trends here (e.g. a 
driver dominating a season) should visibly match real-world results, since 
this is a direct reflection of the underlying dataset the model was trained on.

## 🏆 Constructor Stats

Same idea as Season Analysis, but aggregated at the team level. Select a 
season to see:

- **Constructor championship points**, summed across all races and drivers 
  for each team
- **Constructor race wins**, counting a win for the team regardless of which 
  of its drivers took it

## 🔍 Model Insights

Shows the model's built-in XGBoost feature importance (distinct from the SHAP 
values in the SHAP Explainer tab — this uses XGBoost's native importance 
scores rather than Shapley values). Ranks all ten input features by how much 
the model relies on them when making predictions, giving a quick, 
computationally cheaper alternative view of what drives its decisions 
alongside the more detailed SHAP breakdown elsewhere in the app.

## 🧠 SHAP Explainer

Uses SHAP (SHapley Additive exPlanations) to show *why* the model predicts 
what it predicts, rather than just outputting a probability.

**Global feature importance:** a bar chart ranking all ten input features by 
their average impact on the model's win predictions across the dataset. 
Rankings and the accompanying insight text are generated dynamically from the 
saved SHAP values, so they update automatically whenever the model is retrained 
rather than needing manual edits.

**Individual prediction explanation:** pick any driver, team, circuit, and 
race conditions to get a specific win/no-win prediction alongside a SHAP 
waterfall chart showing which features pushed that particular prediction 
toward or away from a win, and by how much.

**Key insight from the current model:** qualifying position is the single 
strongest predictor of race outcome, consistent with real-world F1 — pole 
position winners historically convert to race wins at a high rate.

## 🎮 Race Simulation

Simulates a full 20-driver grid for a single race and predicts a complete 
finishing-probability order, rather than evaluating one driver at a time.

**How it works:** choose between the 2026 grid (current season) or the 2023 
grid (the model's original training season, included since all its drivers 
are fully supported), set qualifying order and race conditions (circuit, 
weather, temperature), then run the simulation. The model scores every 
driver's win probability independently given their grid slot and the shared 
race conditions, and the results are ranked to produce a predicted podium 
and full grid breakdown.

**Rookie handling:** 2026 grid drivers with no historical race data in the 
training set (2019–2025) are flagged directly in the UI rather than silently 
predicted for — the model has no basis to estimate their performance, so it 
says so instead of guessing.

## 📈 Season Points Projection

Simulates the remainder of a season using Monte Carlo methods rather than 
producing a single deterministic guess.

**How it works:**
1. For each remaining race, the model estimates each driver's win probability 
   using their recent-form average (grid/quali position) as a stand-in for 
   future qualifying results.
2. A full race finishing order is sampled using a Plackett-Luce model — 
   higher win probability increases the *likelihood* of finishing higher, 
   but doesn't guarantee it, so underdogs can still podium in a given simulation.
3. This is repeated hundreds to thousands of times, and points are accumulated 
   across all simulated races to build a distribution of plausible final 
   standings rather than one number.

**Output:** projected mean points, a 10th–90th percentile range per driver, 
and championship win probability, all derived from the simulation distribution.

**Known limitations:**
- No DNF, safety car, or race-strategy modeling — every simulated race assumes 
  a clean finish for all drivers
- Weather/temperature inputs are held at neutral defaults during projection 
  rather than forecasted
- Drivers with no historical race data (e.g. new 2026 rookies) are assigned a 
  small flat win-probability floor rather than a learned estimate
- Simulation count affects precision for rare events — low sim counts can 
  round very unlikely outcomes down to 0% when the true probability is small 
  but nonzero

This tab is meant to demonstrate probabilistic scenario modeling on top of the 
core classifier, not to serve as a precise championship forecast.

## Live App
Click here to try it!
[F1 Race Winner Predictor](https://f1-race-winner-predictor-74j3eweg5ymmbmtk5quijg.streamlit.app/)
