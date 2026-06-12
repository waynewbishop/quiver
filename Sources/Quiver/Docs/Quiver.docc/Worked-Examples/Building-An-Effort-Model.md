# Building an Effort Model

Measuring activity from many signals at once, rather than collapsing effort to a single number.

## Overview

Tracking one's effort is multi-dimensional. Heart rate, pace, cadence, grade, and the rest are simultaneous readings of one underlying state, and the effort that state represents is encoded across all of them at once. When exercising, most training-load metrics reduce that state to a single signal — usually heart rate — and convert it to a score.

A **True Effort Score** (TES) keeps all six dimensions in play. The score reads effort from heart rate, pace, cadence, grade, altitude, and vertical oscillation together — the full signal the body produces in each moment — rather than from any one of them alone. Each is a raw reading from the watch, never a pre-computed effort number from another model. A heart rate inflated by heat, by a climb, or by a fast descent looks the same on its own, but sits in a different place once the other five signals are present, and the model can tell the cases apart.

### Reading the full signal

The way it reads that full signal is to treat heart rate as a downstream response to workload rather than a measure of it, and to predict the heart rate the other signals say a moment should have produced. The gap between that expectation and the observed heart rate — the **residual** — is a diagnostic, not the score itself. It measures how far heart rate has drifted from what the workload explains, which is the information a single-signal reading throws away. That gap has a name in exercise science: **cardiac decoupling**, when heart rate and the workload that drives it stop moving together. Heart rate drifts up while the work holds steady, or stays high while the work drops away. That divergence is exactly what a single-signal score cannot see, and it is what lets the model read the rest of the signals in context rather than trusting heart rate alone.

### The models that do the work

Three Quiver models do the work, each with a distinct job. A ``StandardScaler`` puts the signals on one common scale, so no single dimension dominates the others by its raw units. A ``Ridge`` regression is the baseline: it predicts the expected heart rate from the workload signals — pace, cadence, grade, vertical oscillation, and altitude — and the residual against the observed heart rate is the diagnostic that reads heart rate in context, the gap a single-signal reading misses. A ``KNearestNeighbors`` classifier then labels the moment by how it resembles past efforts, catching the abrupt cases the residual cannot — the descent where heart rate stays high while the work falls away. The label drives the score; the residual reads heart rate in context. The classifier rides inside a ``Pipeline`` with its own scaler, so the two are always applied together.

Throughout, the signals stay together. Each moment remains a single vector, flowing through scaling, regression, and classification without being flattened to a single number first — the full-signal principle the whole approach rests on. Nothing here is a Quiver type to import; the effort model is one we assemble, the same way <doc:Semantic-Search> assembles a search pipeline from smaller parts.

> Important: The sensor values throughout this article are illustrative, chosen to make the arithmetic checkable. A real model would train on a runner's own recorded history. The score classifies the character of a moment; it is not a validated training load, and no number here should be read as a prescription.

### Why one number falls short

On a steep descent, heart rate can hold at `160` while cardiovascular demand drops — the runner is braking against gravity, not driving the pace. A heart-rate-only score reads that `160` and calls the stretch hard, overstating the cardiovascular cost. The descent is not free, but the cost is muscular, not cardiac: each downhill stride lands with the quadriceps lengthening under load — eccentric loading, the largest single source of next-day soreness. The single number is wrong twice over: it overstates cardiovascular demand and misses the mechanical load entirely. Both "hard" and "easy" are true, of different systems, and one number cannot hold both.

On a climb too steep to run, the runner drops to a power-hike. Pace falls toward eighteen or twenty minutes per mile and cadence collapses, so a pace- or cadence-based score reads the moment as nearly a rest. That score is wrong in the other direction: lifting bodyweight up a steep grade is hard work, and here the heart rate reflects it honestly. This is the cleaner case — cardiovascular and muscular demand both rise and roughly agree, so heart rate tells close to the truth. Pace is the misleading signal, reading slow while the work is among the hardest of the session.

The problem does not need terrain. At a steady, conversational pace on a flat loop on a warm afternoon, heart rate drifts ten or fifteen beats upward over the back half. The effort never changed; the cost of cooling did, as blood shunts to the skin and the heart beats faster to do the same job. The flat, unchanging pace is the signal that identifies the rising heart rate as drift rather than effort.

### The same number, different efforts

The thread through all three is one idea: the same heart rate means different things depending on what the rest of the body is doing. A `160` plunging downhill, a `160` grinding uphill, and a `160` drifting upward on a hot flat road are three different efforts, and the information that separates them does not live in the heart rate. It lives in the pace, the cadence, and the grade alongside it. The training-load scores most runners carry collapse a whole workout down to that one signal, and one signal cannot tell those three moments apart. The effort model in this article keeps the signals together and reads heart rate in their context — asking not "how high is the heart rate" but "how high is it, given everything else the body is doing right now."

### Keeping every dimension

A wrist sensor produces far more than a heart rate. Each instant of a run is a six-signal snapshot — heart rate, pace, cadence, grade, altitude, and vertical oscillation — a point in six-dimensional space describing what the body is doing right then. Reducing that point to one number is a projection onto a single axis, and a projection discards the other dimensions. Two efforts that coincide on the heart-rate axis sit far apart in the full space: the downhill `160` and the uphill `160` are one number standing for two different points. Keeping all six dimensions separates them, and a model can label them apart.

Quiver treats each moment as the whole vector it is — a plain `[Double]` of every signal at once — and the same array flows through scaling, regression, and classification without being flattened to a scalar first. Working in the full space rather than one axis at a time is what lets the model hold a fast descent and a threshold effort apart instead of scoring them alike. The model below does not use every signal in every component — a coefficient earns its slot only when it carries information the others do not — but the starting point is always the full vector, never a single projection of it.

## Sorting what misleads the reading

The design follows a single organizing idea: sort every misleading effect by which signal it distorts and in which direction, then hand each kind to the component whose math can own it.

- **Inflating and smooth** — heat, cardiac drift, altitude. Heart rate reads higher than the workload warrants, gradually. The regression residual owns these.
- **Masking and discontinuous** — a downhill, a sudden change of footing. Heart rate or pace under-reads the moment abruptly. The **classifier** owns these, because the moment resembles past moments of the same kind regardless of heart rate.
- **Session-level** — intervals, total duration, abrupt transitions. These are properties of the whole sequence, not of any one sample. An **accumulator** owns these at the end of the session.

Each component catches a kind of distortion the others are blind to. That is why an honest effort model needs both a regression and a classifier, not one or the other.

## Predicting expected heart rate

The baseline is a **multiple linear regression**: it predicts expected heart rate as an intercept plus one weighted term per standardized signal. "Multiple" means many predictors, not many powers — each signal contributes linearly, with no squared or interaction terms. The signals are the external workload: pace, cadence, grade, vertical oscillation, and altitude. These arrive from the Apple Watch sensors during a run; from here on we treat them as plain `[Double]` columns.

We fit it with ``Ridge`` rather than ``LinearRegression`` because the signals overlap — pace and grade in particular carry some of the same information — and that overlap makes a plain least-squares fit numerically unstable. On a device, an unstable fit is not an abstraction: nearly redundant columns drive the solution toward a singular matrix, and the solve can fail outright. Even short of failure, the overlap makes the fit high-variance — the coefficients swing widely from one training run to the next, chasing noise the signals cannot actually separate. Ridge's L2 penalty steadies the fit by gently shrinking the coefficients toward zero, trading a little bias for a large drop in that variance. The result is stable and slightly biased, which is the trade we want. See <doc:Regularization-Primer> for why the penalty does this, and <doc:Ridge-Regression> for the model itself.

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

> Note: The intercept is never penalized — only the per-signal slopes shrink. The intercept carries the runner's resting baseline, which should not be pulled toward zero.

### Choosing the penalty strength

The `lambda` of `1.0` above is a starting point, not a fixed truth. A penalty too small lets the overlapping signals destabilize the fit; a penalty too large flattens the slopes until the baseline ignores the workload it is supposed to read. The right `lambda` sits between those failures, and the honest way to find it is to let the runner's own history pick it rather than to guess. The method is cross-validation: fit a few candidate penalties on most of the recorded history, score each one on a held-out slice, and keep the penalty whose error on data it never trained on is lowest. See <doc:Train-Test-Split> for the held-out evaluation this rests on.

## Scaling the signals

The fit above transforms the features with a ``StandardScaler`` before training, and that step is not optional. The signals live on wildly different scales: cadence near `170`, grade between roughly `−3` and `+4`, altitude in the hundreds of meters. A penalty that compares coefficient magnitudes — which is exactly what Ridge does — falls unevenly when the inputs are on different scales, over-shrinking the small-range signals and barely touching the large-range ones. Standardizing puts every signal in the same z-score units so one `lambda` applies fairly across all of them. See <doc:Feature-Scaling> for the transform itself.

The scaler is fit once, on the training data, and stored. At prediction time we transform each live sample with that same stored scaler — we never re-fit it. Re-fitting on new data would compute new means and standard deviations, silently shifting every feature out from under the trained coefficients.

```swift
import Quiver

// Reuse the stored scaler — fit once, transform forever.
let liveSample: [[Double]] = [[5.2, 174, 1.0, 7.7, 300]]
let expected = baseline.predict(scaler.transform(liveSample))[0]
// 152.9 — the heart rate this workload predicts
```

## Reading the residual

With an expected heart rate in hand, the corrected signal is plain subtraction. The watch reports a heart rate of `162`. The baseline, given the same pace, cadence, grade, oscillation, and altitude, expected `152.9`. The residual is the difference:

```swift
let observed = 162.0
let residual = observed - expected   // 162.0 − 152.9 = 9.1
```

That `9.1` beats per minute is heart rate the external workload does not explain — the signature of an inflating effect such as heat or drift. The sign carries meaning in both directions. A residual of `−7` would mean the heart is running cooler than the workload predicts — the structural blind spot a regression cannot see on its own, a high-speed, low-demand coasting phase where pace is high but cardiovascular demand is not.

Being exact about what the residual is and is not matters here. It detects that decoupling is happening; it does not diagnose the cause. A large positive residual is consistent with heat, altitude, dehydration, or accumulating fatigue, and the model cannot separate them — it has a heart rate and a workload, not a thermometer. Naming a single cause would claim more than the signal supports.

## Classifying the moment

The residual is blind to the masking effects, so a second model covers them. A ``KNearestNeighbors`` classifier labels each moment Easy, Steady, Tempo, or Hard by its resemblance to past moments across the full kinematic signature. A downhill sample — high heart rate, fast pace, negative grade — lands near other downhill samples and is labeled Easy, regardless of how high the heart rate reads. The classifier does not need to reason about terrain; it only needs past examples of it.

We bundle the classifier with its own scaler in a ``Pipeline`` so the two are always applied together and can never drift apart when the model is saved and reloaded. See <doc:Nearest-Neighbors-Classification> for the classifier and <doc:Pipeline> for why the bundle matters.

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

The downhill row is the one that earns the classifier its place: it sits at the same heart rate as a hard effort but is labeled Easy because every other signal says so. The label means low cardiovascular demand — not low total stress. The eccentric load of braking down a steep grade is real muscular cost, and these five signals do not capture it. The win is that the model avoids overscoring the descent, not that it understood everything about it.

### How the model learns a runner

A personal model is only as good as the history behind it, and on day one there is no history. Every wearable faces this same warmup: the first readings are provisional, and the numbers earn trust only after the watch has seen the runner across enough sessions. The honest way to present the model is in three phases, not as a finished product that works perfectly from the first run.

The first phase is the cold start. A runner who has just installed the app has confirmed no efforts, so the classifier trains on a small set of labeled examples grounded in physiology rather than in this runner's data — a steep downhill at a high heart rate is Easy, a fast flat stretch is Hard, and so on. These examples are ground truth the model can lean on immediately, which is why a usable score exists from the first run. The baseline, meanwhile, starts from population-level expectations until the runner's own recorded sessions replace them.

The second phase is personalizing. As the runner logs sessions and confirms or corrects how each felt, the baseline refits on their own recorded heart rate and the classifier retrains on their corrected labels. The model drifts away from the population anchors and toward this runner — their resting baseline, their pace-to-effort mapping, the heart-rate cost a climb carries for them specifically.

The third phase is established. After enough personal history, the baseline's residuals center near zero on the runner's unseen sessions and the labels match what they report feeling. That is the model earning its trust, and it is exactly the check the next section describes. The transition is gradual, not a switch — the same reason a new watch asks for a couple of weeks before its long-term metrics settle.

## Reading the baseline as math

An effort score is only trustworthy if its reasoning can be inspected. Because the baseline is linear, we can render it as the equation it is. The `asExpression` method on the coefficients prints the intercept and each signal's weight:

```swift
baseline.coefficients.asExpression(form: .inline)
// ⟨146.3006, -3.0469, 3.0932, 0.4378, -3.077, 2.4706⟩
```

These weights are in standardized units: each is the change in expected heart rate per one standard deviation of its signal, so they are directly comparable to one another but are not raw beats-per-percent-of-grade. A weight near zero carries a caution worth stating — it can mean the signal carries little information, or simply that the signal barely varied in the training data. A run held at constant altitude, for instance, gives altitude no variation to learn from, and its weight collapses toward zero regardless of any true relationship. See <doc:Model-Interpretation-Primer> for reading fitted coefficients in general.

Reading the comparison as a ranking matters more here than in most fits, because the reason we reached for `Ridge` is the same reason the weights need care. Pace and grade overlap, and two overlapping signals can split a single underlying influence between them, so a coefficient is evidence about a signal's role rather than a final measure of its importance. The penalty keeps the pair stable instead of letting them swing into the large, opposing values that collinearity produces in an unpenalized fit, but stable is not the same as cleanly separable. The `conditionNumber` of the standardized feature matrix says how far to trust the comparison. A low number means the weights stand on their own. A high one means two signals are speaking as one, so their individual weights should be read together rather than ranked against each other. This trust check is what protects the score, since a coefficient the data cannot justify produces a residual the data cannot justify. The <doc:Model-Interpretation-Primer> walks through the full diagnosis and the condition-number reading.

## From a moment to a session

The per-sample models classify each instant on its own. The session-level effects — the third row above — are folded in once, at the end. An accumulator gathers the labeled moments and applies the costs that only exist across the whole sequence. Three terms carry those costs: a variance term that credits the oscillation cost of intervals, a gentle duration term past forty-five minutes for the fatigue a long run accumulates, and a transition term for the neuromuscular cost of abrupt changes. The duration term is deliberately weak and capped, so a short, sharp session is not under-credited; for a ninety-minute run it works out to a factor of about `1.07`.

Including absolute altitude as a baseline signal lets the model learn that, for a runner acclimatized to elevation, a given workload costs a few more beats per minute up high — so the expected heart rate shifts up and the residual stays honest. This is calibration for one runner's habitual environment, not a general altitude correction; it says nothing about the acute response of a sea-level runner arriving at altitude, which is beyond what these signals support.

## The score and what it measures

Everything so far produces one number: the **True Effort Score**, a single measure of how much a session cost. The accumulator turns the labeled, time-stamped moments into that score, and the rest of this section is about what the number means and how to read the result.

The score is a **time-weighted effort load**, not a rating on a fixed scale. Each moment contributes its effort level times its duration: a second at Hard counts for more than a second at Easy, and a long stretch at one level counts for more than a brief one. Summing those contributions across the session and expressing the total against a fixed reference gives a number anchored so that **one hour held at threshold effort scores about 100**. Threshold is the steady-hard pace a runner could hold for roughly an hour, so the anchor is a real, repeatable session rather than an abstract maximum.

The anchor is what makes the number mean something. An hour entirely at an easy jog lands near `33`, the same hour entirely at hard effort lands near `133`, and a mixed session falls between. The score climbs with both intensity and time, so it has no ceiling — a three-hour effort outscores a one-hour effort at the same intensity, which is the point. A rating capped at 100 would call a marathon and a hard 10K the same once both maxed out the scale; a load that keeps climbing keeps them apart.

```swift
// The accumulator's running total, anchored so 1 hour at threshold ≈ 100.
let score = session.currentScore
// 100.0 — one hour held at threshold effort
```

Three session-level terms adjust that base figure, each covered above: the variance term lifts the score for the oscillation cost of intervals, the duration term adds a small graduated surcharge past forty-five minutes, and the transition term credits the neuromuscular cost of abrupt surges. The adjusted score is the base load times the variance and duration factors, plus the transition surcharge. The base load carries the session's intensity and time; the three factors carry the structure those two numbers alone cannot see.

The score is comparable across one runner's own sessions, and that is the comparison it is built for. Because the baseline is personal and the anchor is fixed, today's `120` and last week's `95` are the same measurement taken twice, so the higher number means the harder session for this runner. The score is not built to compare one runner against another: two runners with different baselines produce numbers on two different personal scales, and reading one against the other would compare the scales, not the efforts.

## Knowing whether it works

Building the model and interpreting its weights is only half the task. A model we cannot check is a model we cannot trust, and the check has to run on data the fit never saw. Because this is a personal model, the held-out data is the runner's own: we train the baseline on part of their recorded history and reserve the rest. The reserved runs are the test of whether the residual actually generalizes rather than memorizing the training set. See <doc:Train-Test-Split> for the split itself.

Two readouts tell us whether the baseline earned its trust. On the held-out runs, the residuals should center near zero with no systematic drift — a baseline that consistently over- or under-predicts heart rate on unseen runs is mis-fit, not measuring effort. And the classifier's labels on held-out moments should match what the runner reports feeling, moment to moment. The classifier carries its own version of the bias-and-variance trade we made for the baseline: its `k` is the knob. A small `k` follows every local wobble in the training points and reads as noisy; a large `k` oversmooths and blurs the boundaries between efforts. The right `k` is the one that holds up on the reserved runs, not the one that looks best on the training set.

### Where to go from here

The effort model is an assembly of parts, each documented on its own: the stabilized fit in <doc:Ridge-Regression>, the standardization it depends on in <doc:Feature-Scaling>, and the resemblance-based label in <doc:Nearest-Neighbors-Classification>. To see the same pipeline running on a wrist — pulling live signals from HealthKit and rendering the result on the watch face — see <doc:watchOS-Apps>, which hosts the demonstration this article is drawn from.

> Tip: These ideas run end to end in a small watchOS app. The baseline, the residual, and the effort classifier drive a live watch face, with a screen that shows observed heart rate against the rate the baseline expected — the residual made visible. The full source is on GitHub at [quiver-demo-watchos](https://github.com/waynewbishop/quiver-demo-watchos).

> Experiment: **The Quiver Notebook** is the right place to watch the residual do its work. Build the baseline, then take one steady stretch of a run and add a slow upward drift to its heart-rate series — a few beats climbing over several minutes, as heat would produce. Re-read the residual and watch it absorb the drift while the effort label holds steady. The number the baseline could not explain is exactly the injected drift. See <doc:Quiver-Notebook>.
