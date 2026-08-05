# Building an Effort Model

Scoring exercise effort from a live sensor stream by reading every signal at once, not just heart rate.

## Overview

Measuring exercise effort is hard because the body produces several signals at once. Heart rate, pace, cadence, grade, altitude, and vertical oscillation are all facets of one underlying physiological state. Most training tools reduce that state to a single signal, usually heart rate, and convert it to a score. That reduction often misses the real story of a workout.

A **True Effort Score** keeps every dimension of a run in play. It reads heart rate, pace, cadence, grade, altitude, and vertical oscillation together to picture what the body is doing in each moment. A heart rate might be high because of altitude or a steep climb. By reading all the signals at once, the model tells those cases apart.

``TrueEffortScore`` packages this whole model as one streaming value type. It composes a ``Ridge`` baseline, a ``ResidualModel``, and a ``KNearestNeighbors`` classifier internally. A caller feeds it raw sensor moments and reads back a live score, an effort band, and a residual, the lifecycle running from first tick to final breakdown. Unlike a fit-then-predict model, ``TrueEffortScore`` fits incrementally: each `finalize()` refolds history and refits the baseline, while `record` streams moments in between.

## Reading the full signal

The model treats heart rate as a reaction to work. It uses the other signals to predict what the heart rate should be in a given moment, then measures the gap between that prediction and the real reading. That gap is the **residual**, and it is how far the heart has drifted from the expected workload. The shaded band between the flat expected line and the rising observed line is the residual growing across a session:

![A flat expected heart-rate line at 150 bpm with the observed heart rate rising above it over 40 minutes, the widening gap between them shaded as the residual](diagram-residual-modeling)

The residual and the effort score are two distinct readouts: both climb as effort rises, but the score measures cumulative load and the residual measures an instantaneous gap. Carrying that distinction through makes every later mention of the residual read the same way.

This kind of divergence between heart rate and workload is closely related to what endurance coaches call **aerobic decoupling** — heart rate and the work being done stop tracking each other. The classic metric compares the first and second halves of a sustained effort; this model measures the divergence moment to moment as a residual. Heart rate might climb while pace holds steady. A single-signal score cannot see that divergence. The model reads heart rate in context and decides when to trust the signal and when to lean on the other data instead.

Internally the type wires three Quiver models together. A ``Ridge`` baseline predicts expected heart rate from pace, cadence, grade, vertical oscillation, and altitude. A ``ResidualModel`` over that baseline surfaces the gap between expected and observed heart rate. A ``KNearestNeighbors`` classifier labels each moment to catch the cases the residual misses. The caller never assembles these by hand — construction, fitting, and scaling all happen inside the streaming type.

> Important: The sensor values here are illustrative, chosen to make the arithmetic checkable. A real model trains on a runner's own recorded history.

## Why one number falls short

On a steep descent heart rate can stay low while the legs take a heavy pounding. The quadriceps work eccentrically to brake against gravity on every footstrike, a genuinely hard muscular load. The cardiovascular cost is light, so heart rate sits well below what the effort deserves. A heart-rate-only score reads this as easy and misses the mechanical load entirely — the load that shows up as sore quads a day or two later, long after the heart rate said easy. The effort is real for the legs and light for the heart at the same time:

![One instant on a steep descent shown as two bars, cardiovascular cost from heart rate reading low while muscular load from eccentric quad braking reads high](diagram-load-cost)

On steep climbs runners often drop to a power-hike, and pace and cadence collapse. Pace-based scores misread this as rest, failing to account for the work of lifting bodyweight against gravity. Here heart rate reflects the effort honestly, and pace is the misleading signal.

### The same number, different efforts

Heart rate carries different meanings depending on what the rest of the body is doing. A low heart rate plunging downhill, a high one grinding uphill, and a high one drifting upward on a hot flat road are three distinct physiological states. A single number flattens them — it reads the downhill as easy while the legs work, and it cannot separate the honest uphill cost from the false rise of heat. Only pace, cadence, and grade together separate these efforts. Other divergences — caffeine, dehydration, a fatigued nervous system — leave the same fingerprint on heart rate and are the caller's to interpret, not the model's.

## Keeping every dimension

A modern watch reports more than heart rate alone. Each instant of a run is a six-signal snapshot, a point in six-dimensional space describing what the body is doing. The model treats each moment as a complete set of signals, and the same moment flows through the baseline and the classifier.

The two internal models read different slices of that moment on purpose, and the split is unrepresentable by accident:

- The regression reads `[pace, cadence, grade, verticalOscillation, altitude]` and predicts heart rate. Heart rate is the target, never a feature.
- The classifier reads `[heartRate, pace, cadence, grade, verticalOscillation]` and labels the effort. Altitude is left out here — as an absolute position it describes where the runner is, not how hard they are moving. Its influence on heart rate is the baseline's job, not the classifier's.

Heart rate is only ever the regression target and never an effort feature. That single rule is what lets the model hold a fast descent and a threshold effort apart instead of scoring them alike.

## Sorting what misleads the reading

Every misleading effect has a characteristic signature, and that signature decides which internal model catches the effect.

- **Inflating and gradual** — heat, cardiac drift, or altitude. Heart rate reads higher than the workload warrants, and usually creeps rather than jumps. The regression residual catches these.
- **Masking and discontinuous** — a downhill, or a power-hike up a steep grade. Heart rate or pace misreads the moment abruptly. A descent keeps heart rate low while the legs work. A power-hike drops pace while the climb stays hard. The classifier catches these, because the effort resembles past moments of the same kind regardless of heart rate.
- **Session-level** — intervals, total duration, abrupt transitions. These are properties of the whole sequence, not of any one sample. The breakdown folds them in at the end, in ``TESResult``.

## Constructing the model

An app picks one of two constructors at startup, and every later call uses the same `tes` binding. A brand-new runner starts with no history and no personal baseline. The empty initializer seeds the classifier from a bundled anchor set and leaves the baseline uncalibrated. In this model, `nil` is information, not absence — each `nil` below marks a meaningful state rather than a missing value.

```swift
import Quiver

// Cold start — a brand-new runner, no history:
var tes = TrueEffortScore(lambda: 1.0, k: 3, historyLimit: 60)

tes.baseline       // nil — the cold-start tell
tes.sessionCount   // 0
```

The classifier is seeded from a fixed anchor set so a score exists from the very first tick. Those anchors are inspectable as `TrueEffortScore.anchorSamples` and `TrueEffortScore.anchorLabels`, a parallel set of classifier feature rows and their effort labels. The specific anchor rows are not the lesson — only that the classifier is never empty, so a score exists from the first tick.

> Important: The cold-start initializer traps if `k` exceeds the anchor count. Validate a user-supplied `k` at the call site before constructing the model, since a precondition failure is a programmer error, not a recoverable one. The anchor count is a public property, so the guard is one line:
>
> ```swift
> guard k <= TrueEffortScore.anchorSamples.count else { /* reject the k */ }
> ```

A returning runner seeds from recorded history instead. History is an array of ``Workout`` values — the settled state of past finalized runs, which the model itself produces, since a finalized ``TESResult`` folds into history automatically. A returning app reloads its persisted model rather than rebuilding this array by hand. This throwing initializer fits the baseline once from that history and can add the runner's own labeled samples to the classifier.

```swift
import Quiver

// — or, a returning runner, seeded from saved workouts (pastRuns is [Workout]):
var tes = try TrueEffortScore(history: pastRuns, lambda: 1.0, k: 3)
```

To seed the classifier with the runner's own labeled moments, pass matching `labeledSamples` and `labels` arrays. Each `labeledSamples` row is a classifier feature vector in the order `[heartRate, pace, cadence, grade, verticalOscillation]`, appended to the bundled anchors so the classifier never trains on fewer than the anchor set. The two arrays must be the same length — an unequal count traps rather than throws, so match them at the call site.

```swift
// Seed the classifier with the runner's own labeled moments alongside history:
var tes = try TrueEffortScore(
    history: pastRuns,
    labeledSamples: labeledRows,   // rows of [heartRate, pace, cadence, grade, verticalOscillation]
    labels: sampleLabels,          // one EffortClass per row; counts must match
    lambda: 1.0, k: 3)
```

The seeding initializer throws `TESError.insufficientLabeledSamples` when `k` is larger than the anchor set plus the supplied labeled samples, and `TESError.baselineDiverged` if that first fit fails to converge. Only this initializer and a decode-time refit throw. The live recording path never does. Because `record`, `pause`, `resume`, and `finalize` all mutate the model, the binding is a `var` — a `let` would refuse every one of them.

## Recording a live run

The lifecycle has three steps in a fixed order: construct once and hold it in a `var`, call `record` once per sensor tick for the life of the run, then call `finalize` a single time to close it. The whole arc reads as one block before we drill into the pieces:

```swift
var tes = TrueEffortScore(lambda: 1.0, k: 3, historyLimit: 60)

// One call per tick, each arriving from the sensor stream with its own timestamp:
for tick in sensorStream {
    tes.record(
        heartRate: tick.heartRate, pace: tick.pace, cadence: tick.cadence,
        grade: tick.grade, verticalOscillation: tick.verticalOscillation,
        altitude: tick.altitude, at: tick.time)
}

// Close the run once, at the end. nil when active time never reached 60 seconds:
guard let result = tes.finalize() else { return }
print(result.adjusted)
```

Everything below drills into one step of that arc. The caller feeds one moment per sensor tick, and each `record` call takes the raw signals and an absolute `Date`. The model derives the time delta between samples internally, so the caller never tracks elapsed time. All six signals are required on every call, since there is no missing-value handling inside the model. When a sensor drops a reading, carry the last value forward before calling `record`.

```swift
// Feed each tick as it arrives from the sensor stream.
tes.record(
    heartRate: 152, pace: 5.2, cadence: 174, grade: 1.0,
    verticalOscillation: 7.7, altitude: 300, at: sampleTime)
```

Two contracts protect the stream from noisy hardware. A non-positive delta is dropped, so a duplicate or out-of-order callback discards itself with no upstream de-duplication. The first sample of a run carries zero load but stamps the run's start, so the clock begins on real data.

The `hrTrust` parameter reports optical-sensor reliability and is clamped to `0...1`. A fully doubted reading is pulled toward the training mean, so classification falls back to the four kinematic signals, and the doubted stretch cannot swamp the session's residual.

```swift
// A stretch where the wrist sensor is unreliable: trust the kinematics, not the heart rate.
tes.record(
    heartRate: 210, pace: 5.0, cadence: 176, grade: 0.0,
    verticalOscillation: 7.4, altitude: 300, at: sampleTime, hrTrust: 0.0)
```

Pausing and resuming keep two clocks apart. A paused span counts toward wall-clock `elapsedTime` but not toward `timerTime`, and `timerTime` is the basis for the score. After a resume, the next sample re-seeds the clock across the gap and advances the wall clock only, never the timer and never the run's start.

```swift
tes.pause()              // stop the timer; wall clock keeps running
// ... rest interval ...
tes.resume()             // next sample bridges the gap on wall clock alone

tes.isPaused             // false
```

## Reading the live values

While the run streams, the model exposes live readouts a UI can poll each tick. The score climbs and never decreases.

```swift
tes.currentScore     // cumulative load so far, 0 on an empty run
tes.currentEffort    // the latest sample's EffortClass, nil before the first sample
tes.sampleCount      // moments recorded into the live buffer
tes.timerTime        // active seconds, excludes pauses — the score's basis
tes.elapsedTime      // wall-clock seconds, includes pauses
```

The residual is available live once a baseline exists. On a cold-start model it reads `nil`, which is itself the tell that the personal baseline has not yet calibrated.

```swift
tes.currentResidual  // observed − expected heart rate, nil until a baseline exists
```

The subtraction is the ``ResidualModel``'s, computed against the baseline's own scaled features — positive means the heart is running above what the workload predicts. That is a signature of an inflating effect such as heat or drift. A negative residual would mean the heart is running cooler than the workload predicts, the high-speed, low-demand coasting of a fast descent. The residual reports a gap and never names its cause.

The classifier also exposes the latest sample's signals as z-scores against its training distribution, which lets a UI flag whichever signal is the outlier.

```swift
tes.currentSignals   // EffortSignals: five z-scores in σ units, nil before the first sample
```

A large negative `grade` z-score paired with a near-zero `heartRate` z-score is the downhill signature — the legs working while the heart stays calm. ``EffortSignals`` carries the five values as `heartRate`, `pace`, `cadence`, `grade`, and `verticalOscillation`, in the classifier's feature order.

## Finalizing the run

Calling `finalize` closes the run and returns a ``TESResult``. It returns an optional, and the contract is strict: a run with under 60 seconds of active timer time is cleared and returns `nil`. That is neither an error nor a zero score. It is the model declining to fold a run too short to mean anything.

```swift
// Close the run. nil when the active timer never reached 60 seconds.
guard let result = tes.finalize() else {
    // Too short to keep — the live buffer is already cleared.
    return
}

print(result)
```

At finalize the run is folded into history, trimmed to `historyLimit`, and the baseline refits from the accumulated history. That refit runs through `try?`, so a divergent fit is swallowed and the previous baseline stays in place. The classifier never refits — it stays frozen after seeding, and only the baseline personalizes over time.

## Reading a result

``TESResult`` exposes no bare score accessor, because the headline number is one of several readouts a session earns. The headline is `adjusted`, and it is anchored so that one hour held at threshold effort scores about 100.

```swift
result.adjusted          // headline effort score, ≈100 for one hour at threshold
result.raw               // 100 × Σ(weight · Δt) / 2700, before session terms
result.meanResidual      // trust- and time-weighted mean gap in bpm
result.effortDistribution  // [EffortClass: Double] — share of time per band
result.timerTime         // active seconds
result.elapsedTime       // wall-clock seconds, includes pauses
```

Threshold, the steady-hard pace a runner could hold for roughly an hour, is the anchor rather than an abstract maximum. That makes the number meaningful. An hour entirely at an easy jog lands near `33`, an hour spent entirely in the vo2max band lands near `133` — a mix no real session sustains, shown only to bound the scale — and a mixed session falls between. The score climbs with both intensity and time and has no ceiling, so it separates a long effort from a short one at the same intensity.

The `meanResidual` is the session's average gap between observed and expected heart rate, in beats per minute, weighted by trust and duration. A positive value means the heart ran above prediction across the session, a signature of heat or drift. It reports the gap and never names the cause. Without a baseline it is `0`.

The `effortDistribution` is a share of time, not a share of samples. Each band's value is the fraction of active time spent there, and the shares sum to about `1.0`. Because it is time-indexed rather than sample-indexed, a run with pauses or dropped ticks will show a time-share that diverges from the raw row count. The same is true of the result's `loadCurve`, which traces cumulative load indexed by recorded moment.

Three session-level terms sit alongside the raw score and are kept separate rather than blended into one opaque number.

```swift
result.varianceMultiplier  // 1 + min(0.35, 0.25 · variance of steady-state ordinals)
result.durationFactor      // 1.0 up to 45 min, then grows with ln(minutes / 45)
result.transitionLoad      // Σ|Δclass| over jumps of two bands or more
```

The variance term catches a session that swings between effort levels, computed on steady-state moments only. The duration term adds fatigue cost past forty-five minutes. The transition term catches abrupt jumps across two or more bands, which is what makes a session of hill repeats cost more than the same minutes run at a steady middle pace. The headline `adjusted` combines the raw score with the variance and duration terms and adds a small share of the transition load.

Printing a result is always a labeled multi-line block, never a bare number — adjusted, raw, mean residual, the three session terms, the two clocks, and the baseline expression.

## Reading the baseline

The personal half of the model lives in ``TESBaseline``, reachable read-only as `tes.baseline`. It is `nil` until the first finalize folds a run in and fits it. When present, it exposes the fitted heart-rate model as inspectable math.

```swift
// nil on a cold-start model; present once a run has been finalized.
if let baseline = tes.baseline {
    baseline.labeledExpression  // "expected HR = 146.3 - 3.047·pace + ..."
    baseline.coefficients       // per-feature weights, intercept at index 0
    baseline.conditioning       // 1-norm condition of XᵀX, nil when not measurable
}
```

The coefficients are on standardized features, with the intercept at index 0, in the order pace, cadence, grade, vertical oscillation, altitude. Each slope is the change in expected heart rate per one standard deviation of its signal, so the weights are directly comparable. A weight near zero can mean the signal carries little information, or that it barely varied in the training data.

To make this concrete, a six-row illustrative fit that converged after 416 iterations produces an intercept and five slopes of `⟨146.3006, −3.0469, 3.0932, 0.4378, −3.0770, 2.4706⟩`. Predicting a live workload of `[5.2, 174, 1.0, 7.7, 300]`, which the baseline standardizes with its stored scaler before applying those weights, gives an expected heart rate of `152.9019`, and an observed `162.0` reading leaves a residual of `9.0981` bpm.

The `conditioning` value is the 1-norm condition number of the standardized `XᵀX`, not a singular-value ratio. A value in the tens is information about signal overlap, not a failure. It reads `nil` when the fit is not meaningfully conditioned. The baseline caches no fit-quality metric on purpose — quality is measured on held-out data, not read off the trained object.

Before any fit has run, `tes.baseline` is `nil` and reading `tes.baseline?.labeledExpression` yields `nil`. A ``TESResult`` reports the same uncalibrated state through its `baselineExpression`, which falls back to the literal `y = (uncalibrated)` when no baseline is present. That is the honest state of a first-ever finalize, and a UI should read it as "no personal calibration yet" rather than as a model error.

## How the model learns an athlete

A personal model is only as good as the history behind it. A new watch gives provisional numbers that earn trust as it learns from the athlete's sessions, and the model moves through three phases.

At cold start there is no personal data. The classifier scores from the anchor set immediately, and the baseline is uncalibrated, so `currentResidual` reads `nil`. As the athlete logs sessions, each finalize folds a run into history and refits the baseline, and the model drifts toward the athlete's own resting baseline and pace-to-effort mapping. This is also how a sea-level runner who starts training at altitude comes back into calibration. The first runs in thin air read as inflated residuals. As those runs enter the history the baseline refits, and the residuals settle. After enough history, residuals center near zero on unseen sessions and labels stabilize into consistent bands for recurring workout types:

![Session residuals falling across three phases, large and positive during cold-start sessions, shrinking through personalizing sessions, and settled near zero once the model is established](diagram-personalization-model)

Personalization is asymmetric, and knowing which half moves matters for reading the model. The baseline refits on every finalize and becomes the runner's own. The classifier does not — labeled moments enter only through the seeding initializer's `history` and `labeledSamples`, never through `record`, so a caller who wants a denser classifier supplies labeled samples up front. In the cold-start path the classifier stays on the anchor set for the life of the model, and only the baseline learns.

The classifier needs density to be reliable. A handful of anchor rows shows the shape of each effort but cannot classify a boundary query with confidence, because the effort levels sit close together in signal space, especially threshold against vo2max. A dense classifier accumulates thousands of labeled moments per effort, so each query lands deep inside a crowd of its own kind and a small `k` returns a stable label. The anchor rows name the kinds of effort to tell apart; the reliability comes from the volume of real data supplied at seeding.

## How a moment becomes load

The path from sensor sample to score is short. For each sample the classifier returns an ``EffortClass``, and each class carries a fixed illustrative weight — `easy` is `0.25`, `tempo` is `0.50`, `threshold` is `0.75`, and `vo2max` is `1.00`. A moment's contribution to the load is its weight times its duration in seconds. Summing those contributions and expressing the total against one hour held at threshold gives the raw score.

`TES = 100 × (Σₜ L(lₜ) · Δtₜ) / (0.75 · 3600)`

Here `L(lₜ)` maps a sample's effort label to its weight, `Δtₜ` is the sample's duration in seconds, and `0.75 · 3600 = 2700` is the anchor of one hour held at threshold. The factor of `100` sets the scale, so an hour at threshold lands at `100`, an hour at easy near `33`, and an hour spent entirely in the vo2max band near `133` — a scale bound, not a session any runner sustains. The variance, duration, and transition terms then fold onto that base to produce the headline `adjusted` value.

The tempo-and-threshold boundary is the softest one, and a misclassification there is the cheapest the model can make, because the two weights sit next to each other. The real distinction between them is the lactate threshold, not kinematics, so the model treats the boundary as weight-adjacent rather than sharp.

## The score and the residual are two readouts

Both numbers rise as effort climbs, but they measure different things. The score is a cumulative load that answers how much a session has cost so far. The residual is an instantaneous gap that answers how far the heart is from its expected rate right now. A UI reads both off the live model without any hand arithmetic.

```swift
let score    = Int(tes.currentScore.rounded())   // cumulative effort load
let residual = tes.currentResidual               // instantaneous gap, nil before a baseline
let effort   = tes.currentEffort                 // the latest sample's band
```

The score comes from the accumulated load; the residual comes from the baseline. On a cold-start model the score already reads a real number while the residual is still `nil`, which is the clearest picture of what personalization adds — the band and the load work from the first tick, and the residual arrives once the runner's own baseline is fitted.

## Knowing whether it works

The model breaks in ways that are about data and math, not about physiology. A baseline fitted on too few workouts cannot separate the terrain regimes, and its residuals will not settle. A query scaled differently from the training data reads as a false residual, which is why the scaler is fit once and reused inside the type rather than re-computed per sample. If the baseline is over-fitted, its residuals collapse toward zero and erase the very gap the residual is meant to expose.

The residual detects a misleading reading. It does not diagnose the cause. A positive `meanResidual` says the heart ran above prediction; it does not say whether the reason was heat, altitude, dehydration, or a hard day. Naming the cause is the caller's job with the context the app already holds, and the model deliberately stops at reporting the gap.

The model is a plain value type. It is `Codable`, so a run persists between sessions by encoding the model itself — there is no separate load or save. Only the settled state encodes, and the transient live-run buffer is excluded from both `Codable` and equality, so a model must be encoded between runs and never mid-run.

## Where to go from here

This model combines simple, interpretable pieces rather than one opaque algorithm. The ``Ridge`` baseline carries the expected-heart-rate math, the ``ResidualModel`` surfaces the gap, and the ``KNearestNeighbors`` classifier carries the context the residual cannot see. Personalization here is calibration rather than a separate model — the same baseline starts from population anchors and centers its residuals near zero as it learns one runner's responses to workload and environment. That path from a general baseline to a personal calibration is a pattern that carries to any model operating on individual data. See <doc:Ridge-Regression> for the baseline, <doc:Residual-Model> for the gap it wraps, <doc:Nearest-Neighbors-Classification> for the classifier, and <doc:Feature-Scaling> for the standardization that keeps them aligned.

> Experiment: **The Quiver Notebook** is the right place to watch the score and the residual move independently. Fit a ``Ridge`` baseline on a handful of samples, wrap it in a ``ResidualModel``, and raise one observed value while holding its inputs fixed. The residual climbs while the prediction stays put. Then refit across a few `lambda` values — try `0.1`, `1`, and `10` — and watch the coefficients shrink without the residual's job changing. Seeing the gap respond to the observation while the prediction tracks the inputs is the clearest way to feel why the two readouts measure different things. See <doc:Quiver-Notebook>.

## Topics

### Effort model
- ``TrueEffortScore``

### Inputs
- ``Workout``
- ``EffortSignals``

### Results
- ``TESResult``
- ``TESBaseline``
- ``EffortClass``

### Errors
- ``TESError``
