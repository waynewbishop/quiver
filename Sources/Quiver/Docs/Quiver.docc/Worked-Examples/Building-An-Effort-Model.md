# Building an Effort Model

We measure exercise effort by looking at many signals at once, rather than collapsing effort into a single number.

## Overview

Measuring exercise effort is complex because our bodies produce multiple signals simultaneously. Heart rate, pace, cadence, grade, altitude, and vertical oscillation are all parts of one underlying physiological state. Most training tools simplify this by looking at a single signal, such as heart rate, and converting it to a score. This reduction often misses the real story of a workout.

A **True Effort Score** (TES) keeps all six dimensions of a run in play. We look at heart rate, pace, cadence, grade, altitude, and vertical oscillation together to form a complete picture of what the body is doing in each moment. A heart rate might be high because of heat or a steep climb; by looking at all signals at once, we can tell these cases apart.

## Reading the full signal

We treat heart rate as a reaction to work rather than a measure of it. We use other signals to predict what the heart rate should be in any given moment. The gap between our prediction and the real heart rate, a diagnostic tool we call a **residual**, measures how far the heart rate has drifted from what the workload explains.

In exercise science, this drift is called **cardiac decoupling**. It happens when heart rate and workload stop moving together. For instance, heart rate might climb while pace holds steady. A single-signal score cannot see this divergence, but our model reads heart rate in context and decides when to trust the signal and when to look at the other data instead.

### The models that do the work

We combine four types. [StandardScaler](<doc:Feature-Scaling>) normalizes signals to prevent dimension dominance; <doc:Ridge-Regression> predicts expected heart rate from pace, cadence, grade, vertical oscillation, and altitude; <doc:Residual-Model> surfaces the gap between expected and observed heart rate; and <doc:Nearest-Neighbors-Classification> classifies the moment to catch cases the residual misses. The classifier fits inside a `Pipeline` to ensure consistency. Each moment is treated as a vector, flowing through these components without reduction until the final score.

> Important: The sensor values throughout this article are illustrative, chosen to make the arithmetic checkable. A real model would train on a runner's own recorded history. The score classifies the character of a moment; it is not a validated training load, and no number here should be read as a prescription.

> Note: The [quiver-demo-watchos](https://github.com/waynewbishop/quiver-demo-watchos) is the reference implementation of this article: the Ridge baseline, the `ResidualModel` wrapper, the k-nearest-neighbors classifier, and the session accumulator assembled into a working watchOS app.

## Why one number falls short

On a steep descent, heart rate can hold at 160 while cardiovascular demand drops. A heart-rate-only score reads this as "hard," missing that the effort is muscular (quadriceps load) rather than cardiac. A single number overstates the cardiovascular demand and misses the mechanical load entirely. Both "hard" and "easy" are true, but for different systems; one number cannot hold both.

On steep climbs, runners may drop to a power-hike, causing pace and cadence to collapse. Pace-based scores misinterpret this as rest, failing to account for the work of lifting bodyweight against gravity. Here, heart rate provides an honest reflection of effort, showing that pace is misleading.

This problem does not require extreme terrain. At a steady, conversational pace on a flat loop on a warm afternoon, heart rate drifts ten or fifteen beats upward over the back half. The effort never changed; the cost of cooling did, as blood shunts to the skin and the heart beats faster to do the same job. The flat, unchanging pace is the signal that identifies the rising heart rate as drift rather than effort.

### The same number, different efforts

Heart rate carries different meanings depending on what the rest of the body is doing. A 160 bpm plunging downhill, grinding uphill, or drifting upward on a hot flat road represent three distinct physiological states. Heart rate alone cannot separate these; only pace, cadence, and grade together distinguish the effort. We keep these signals together to ask not "how high is the heart rate," but "how high is it given the rest of the body's activity?"

## Keeping every dimension

A wrist sensor produces far more than a heart rate. Each instant of a run is a six-signal snapshot: a point in six-dimensional space describing what the body is doing. Reducing that point to one number projects it onto a single axis and discards the other dimensions. Two efforts that coincide on the heart-rate axis can sit far apart in the full space; our model keeps all six dimensions to separate them.

Quiver treats each moment as a complete vector (a plain `[Double]` of every signal at once), and the same array flows through scaling, regression, and classification. Working in the full space allows us to hold a fast descent and a threshold effort apart instead of scoring them alike. While the model may not use every signal in every component, the starting point is always the full vector.

## Sorting what misleads the reading

We design the model based on a single organizing idea: categorize every misleading effect by which signal it distorts and in which direction, then hand each kind to the component best equipped to handle it.

- **Inflating and smooth:** heat, cardiac drift, altitude. Heart rate reads higher than the workload warrants, gradually. The regression residual owns these.
- **Masking and discontinuous:** a downhill, a sudden change of footing. Heart rate or pace under-reads the moment abruptly. The **classifier** owns these, because the moment resembles past moments of the same kind regardless of heart rate.
- **Session-level:** intervals, total duration, abrupt transitions. These are properties of the whole sequence, not of any one sample. An **accumulator** owns these at the end of the session.

Each component catches a type of distortion the others are blind to. This is why an honest effort model needs both a regression and a classifier.

## Predicting expected heart rate

The baseline is a **multiple linear regression**: it predicts expected heart rate as an intercept plus one weighted term per standardized signal. Each signal contributes linearly, using workload predictors: pace, cadence, grade, vertical oscillation, and altitude. We treat these as plain `[Double]` columns.

We fit it with ``Ridge`` rather than ``LinearRegression`` because the signals overlap (pace and grade, in particular, carry some of the same information), making a plain least-squares fit numerically unstable. Ridge's L2 penalty steadies the fit by gently shrinking the coefficients toward zero, trading a little bias for a large drop in variance. See <doc:Regularization-Primer> for why the penalty works this way, and <doc:Ridge-Regression> for the model itself.

```swift
import Quiver

// [pace, cadence, grade, verticalOscillation, altitude] → expected heart rate
let workload: [[Double]] = [
    [6.5, 165,  0.0, 8.6, 100],
    [6.2, 168,  1.0, 8.3, 250],
    [5.5, 172,  2.0, 8.0, 400],
    [5.0, 176,  1.0, 7.6, 250],
    [6.8, 162, -2.0, 9.0, 100],
    [4.8, 180,  0.0, 7.0, 550],
]
let heartRates = [132.0, 140, 150, 158, 128, 170]

let scaler = StandardScaler.fit(features: workload)
let baseline = try Ridge.fit(
    features: scaler.transform(workload), targets: heartRates, lambda: 1.0)
```

> Note: The intercept is never penalized; only the per-signal slopes shrink. The intercept carries the runner's resting baseline, which should not be pulled toward zero.

### Choosing the penalty strength

The `lambda` of `1.0` above is a starting point. If the penalty is too small, overlapping signals destabilize the fit; if too large, it flattens slopes until the baseline ignores the workload. We let the runner's own history pick the penalty via held-out validation: fit candidate penalties on most of the recorded history, score each one on a held-out slice, and keep the penalty with the lowest error on data it never trained on. See <doc:Train-Test-Split> for more on this evaluation.

## Scaling the signals

We must transform the features with a ``StandardScaler`` before training. The signals live on wildly different scales: cadence near `170`, grade roughly between `−3` and `+4`, and altitude in the hundreds of meters. Ridge compares coefficient magnitudes, so standardizing puts every signal in the same z-score units to apply `lambda` fairly. See <doc:Feature-Scaling> for the transform itself.

We fit the scaler once on the training data and store it. At prediction time, we transform each live sample with that same stored scaler, and never re-fit it. Re-fitting on new data would compute new means and standard deviations, silently shifting features out from under the trained coefficients.

```swift
import Quiver

// Reuse the stored scaler — fit once, transform forever.
let liveSample: [[Double]] = [[5.2, 174, 1.0, 7.7, 300]]
let expected = baseline.predict(scaler.transform(liveSample))[0]
// 152.9 — the heart rate this workload predicts
```

## Reading the residual

The residual is `observed − expected`, and ``ResidualModel`` is the Quiver type that handles this. We wrap the fitted baseline and read the gap on each live sample. The wrapper holds the model it is handed, ensuring the baseline keeps its fitted coefficients and the residual math stays consistent. See <doc:Residual-Model> for the wrapper on its own.

```swift
import Quiver

// Wrap the fitted baseline; read the gap on one scaled live sample.
let residualModel = ResidualModel(model: baseline)
let scaled = scaler.transform(liveSample)[0]
let residual = residualModel.residual(features: scaled, observed: 162.0)
// 9.1 — observed 162 against the 152.9 the workload predicts
```

That `9.1` beats per minute is heart rate not explained by the external workload: a signature of an inflating effect such as heat or drift. The sign carries meaning: a residual of `−7` would mean the heart is running cooler than the workload predicts, a structural blind spot in regression representing a high-speed, low-demand coasting phase.

The residual detects decoupling; it does not diagnose the cause. A large positive residual is consistent with heat, altitude, dehydration, or fatigue, but the model cannot separate them: it has heart rate and workload, not a thermometer.

## Classifying the moment

The residual is blind to masking effects, so we use a ``KNearestNeighbors`` classifier to label each moment as Easy, Steady, Tempo, or Hard based on its resemblance to past efforts. A downhill sample (high heart rate, fast pace, negative grade) lands near other downhill samples and is labeled Easy, regardless of the heart rate. The classifier doesn't need to reason about terrain; it just needs past examples.

We bundle the classifier with its own scaler in a ``Pipeline`` to ensure they are always applied together. See <doc:Nearest-Neighbors-Classification> for the classifier and <doc:Pipeline> for why the bundle matters.

```swift
import Quiver

// [heart rate, pace, cadence, grade, verticalOscillation] → effort label
let kinematics: [[Double]] = [
    [130, 6.5, 165,  0.0, 8.6],   // Easy — flat, slow
    [160, 6.5, 165, -3.0, 9.2],   // Easy — steep downhill, heart rate inflated
    [150, 5.5, 172,  1.0, 8.0],   // Steady — climbing
    [172, 4.6, 178,  0.0, 7.1],   // Hard — fast, flat
]
let labels = [0, 0, 1, 3]   // Easy, Easy, Steady, Hard

let classifier = Pipeline.fit(features: kinematics, labels: labels, k: 3)
```

The downhill row earns the classifier its place: it sits at the same heart rate as a hard effort but is labeled Easy because the other signals indicate low cardiovascular demand. The model correctly avoids overscoring the descent, even if it doesn't capture all the mechanical load.

### How the model learns a runner

A personal model is only as good as the history behind it. A new watch provides provisional numbers that earn trust as the watch learns from the runner's sessions. We view this in three phases:

1.  **Cold start:** With no personal data, the classifier trains on ground-truth examples (steep downhills at high heart rate are Easy, fast flat stretches are Hard), providing a usable score immediately. The baseline starts from population-level expectations.
2.  **Personalizing:** As the runner logs sessions and corrects effort labels, the baseline refits on their specific data, and the classifier retrains. The model drifts toward the runner's resting baseline and pace-to-effort mapping.
3.  **Established:** After enough history, residuals center near zero on unseen sessions, and labels match reported feelings. This trust is built gradually, just as a new watch requires a few weeks for long-term metrics to settle.

## Reading the baseline as math

We can inspect the baseline's reasoning because it is linear. The `asExpression` method on the coefficients prints the intercept and each signal's weight:

```swift
baseline.coefficients.asExpression(form: .inline)
// ⟨146.3006, -3.0469, 3.0932, 0.4378, -3.077, 2.4706⟩
```

These weights are in standardized units: each is the change in expected heart rate per one standard deviation of its signal, so they are directly comparable. A weight near zero can mean the signal carries little information or that the signal barely varied in the training data (e.g., constant altitude). See <doc:Model-Interpretation-Primer> for reading fitted coefficients.

Reading the comparison as a ranking matters here because signals often overlap. A weight is evidence about a signal's role rather than a final measure of its importance. While the penalty keeps the fit stable, it does not cleanly separate collinear signals: their individual weights should be read together. A high `conditionNumber` for the standardized feature matrix indicates how much caution to apply. See <doc:Model-Interpretation-Primer> for the full diagnosis.

## From a moment to a session

The per-sample models classify each instant. We fold in session-level effects at the end: a variance term for interval oscillation, a duration term for fatigue past forty-five minutes, and a transition term for neuromuscular costs of abrupt changes. The duration term is deliberately weak and capped, so a short, sharp session is not under-credited.

Including absolute altitude as a baseline signal helps the model learn that, for an acclimatized runner, a given workload costs slightly more beats per minute up high, so the expected heart rate shifts up and the residual stays honest. This calibrates for a runner's habitual environment, not for sea-level runners arriving at altitude.

## The score and what it measures

Everything contributes to one number: the **True Effort Score**, our measure of a session's cost. The accumulator turns labeled moments into this score, anchored so that **one hour held at threshold effort scores about 100**. Threshold, the steady-hard pace a runner could hold for roughly an hour, is a real, repeatable anchor rather than an abstract maximum.

This anchor makes the number meaningful. An hour entirely at an easy jog lands near `33`, an hour at hard effort lands near `133`, and a mixed session falls between. The score climbs with both intensity and time; it has no ceiling, so it correctly differentiates between long and short efforts at the same intensity.

### How a moment becomes load

The path from sensor sample to final score is a short pipeline. For each sample, the classifier returns an effort level (Easy, Steady, Tempo, or Hard) carrying a fixed weight (`0.25`, `0.50`, `0.75`, `1.00`). The moment's contribution to the load is that weight times the sample duration. We sum these contributions and express the total against the reference of one hour at threshold (Tempo, `0.75`):

`TES = 100 × (Σₜ L(lₜ)·Δtₜ) / (L(Tempo) · 3600)`

Here `L(lₜ)` maps a sample's effort label to its weight, `Δtₜ` is the sample's duration in seconds, and `L(Tempo) · 3600` is the anchor: one hour held at threshold. The factor of `100` sets the scale, so an hour at Tempo lands at `100`, an hour at Easy near `33`, and an hour at Hard near `133`.

Three session-level terms (**variance**, **duration**, and **transition**) are then folded onto that base. The base carries the session's intensity and time; these terms carry the structure those two numbers alone cannot see.

Here `TrueEffortScore` is the type we are building in our own app, not a Quiver type. It wraps the Quiver models shown above (a Ridge baseline and a k-nearest-neighbors classifier) behind a single `score(for:)` call:

```swift
// Fit from the runner's labeled history, then score a run against it.
let model = try TrueEffortScore.fit(history: pastRuns)
let result = model.score(for: todaysRun)

result.value     // ≈100 — one hour held at threshold effort
result.residual  // the latest moment's gap, read through ResidualModel
```

## The score and the residual are two different readouts

While both numbers rise as effort climbs, they measure different things. The **score** is a cumulative load that answers *how much has this session cost so far*. The **residual** is an instantaneous gap that answers *how far is the heart from its expected rate right now*.

```swift
// Illustrative app code: `tes` is our TrueEffortScore, and `result` exposes
// the readouts it computes from the Quiver models inside it.
let result = tes.score(for: liveRun)
let score    = Int(result.value.rounded())                       // cumulative effort load
let expected = Int(result.expectedHeartRate.rounded())           // the Ridge baseline's prediction
let residual = Int(segment.hr.rounded()) - expected              // observed − expected, this moment
```

We keep them apart intentionally. The residual is a diagnostic that rides alongside the score; it never adds to it. If we let a transient distortion inflate the session load, we would create exactly the conflation this multi-signal model was built to avoid.

## Knowing whether it works

We check our model on reserved data the fit never saw. For personal models, this means training on part of the runner's recorded history and testing on the rest.

Two readouts confirm trust: residuals on held-out runs should center near zero without systematic drift, and the classifier's labels on held-out moments should match the runner's reported feelings. The classifier's `k` is the knob: too small follows every local wobble and reads as noisy; too large oversmooths. We choose the `k` that holds up on reserved runs.

## Where to go from here

This model combines simple, interpretable components rather than a single complex algorithm: regression carries the baseline, classification carries the context-specific patterns, and the full signal vector flows through the pipeline without being reduced to a scalar that loses its context. Personalization is calibration, not a separate model: the same baseline starts from population-level anchors and centers its residuals around zero as it learns one runner's responses to workload and environment. That path, from general baseline to personal calibration, is a design pattern that carries to any model operating on individual data.

> Experiment: **The Quiver Notebook** is the right place to watch the residual and the score move independently. Fit a ``Ridge`` baseline on a handful of workload samples, wrap it in a ``ResidualModel``, and read the residual as we push one observed heart rate up while holding the workload fixed: the residual climbs while the workload-driven prediction stays put. Then sweep `lambda` and watch the baseline's coefficients shrink without the residual's job changing. Seeing the gap respond to heart rate while the prediction tracks workload is the clearest way to feel why the two readouts measure different things. See <doc:Quiver-Notebook>.
