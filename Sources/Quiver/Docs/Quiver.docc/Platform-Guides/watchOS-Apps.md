# watchOS Guide

Planning, building, and operating an on-device model across the lifecycle of a watchOS app.

## Overview

A watchOS app reads telemetry and turns it into a small number of values the wearer acts on. Building that pipeline is different from wrist than from a phone or desktop. The data is small and personal, the timers are short, and the result is read in a glance, not studied in a report. This guide walks the lifecycle in the order an app lives it: deciding what the model supplies, fitting and persisting it on the device, designing the glance that presents it, and operating inside the platform's budgets. The math underneath each step lives in the primers; this guide is about the watch.

### Setup and lifecycle

A workout app reads its data through `HKWorkoutSession` paired with `HKLiveWorkoutBuilder`. The session prepares the sensors and keeps the app running while the workout is active; the builder, backed by an `HKLiveWorkoutDataSource`, collects samples automatically and reports each new batch as `HKStatistics`. The delegate callback is where Quiver's input is assembled — one reading per quantity type, appended to the buffer the model will later read:

```swift
import HealthKit

extension WorkoutManager: HKLiveWorkoutBuilderDelegate {
    func workoutBuilder(_ builder: HKLiveWorkoutBuilder,
                        didCollectDataOf collectedTypes: Set<HKSampleType>) {
        let beatsPerMinute = HKUnit.count().unitDivided(by: .minute())

        for type in collectedTypes {
            guard let quantityType = type as? HKQuantityType,
                  let statistics = builder.statistics(for: quantityType),
                  let latest = statistics.mostRecentQuantity() else { continue }

            // One Double per delivery — the array Quiver computes on.
            heartRates.append(latest.doubleValue(for: beatsPerMinute))
        }
    }
}
```

The builder reports aggregates rather than raw sample arrays: `mostRecentQuantity()` for the current reading, `averageQuantity()` and `sumQuantity()` for the session so far. A per-moment feature reads the most recent value; a session statistic can often be read straight off the builder without recomputing it.

Anything that should outlive a single workout — a fitted model, a calibration, a set of stored windows — encodes to JSON and writes to `Documents/`. The next session decodes the value at launch. Models are `Codable` and `Sendable`, allowing them to cross task boundaries and persist without ceremony. See <doc:Model-Persistence> for the shared persistence pattern.

## Planning the work

Every fitted value on the watch starts as a decision made before the first line of model code: what the number is for, whether the signal exists, and when the fitting happens. These are platform decisions as much as modeling ones, and they are cheaper to correct on paper than after the model is embedded in a complication.

### Deciding what the model supplies

The watch produces a number the rest of the system consumes, and the consumer decides the model's shape. A complication needs one scalar that fits in a corner of a watch face. An in-workout screen needs a per-moment value that updates as samples arrive. A weekly summary needs a statistic computed over finished sessions. Before choosing a model family, name the consumer and the shape of value it receives: a **label** (which activity is this), an **expected value** (heart rate at this power), or a **score** (how far today's effort sits from the wearer's own). Each shape is served by a different family of fitted model, and the sections below treat the family as a choice, never a default.

There is a second question behind the first: whether a model is needed at all. The watch already measures heart rate, altitude, distance, and pace, and a value the sensors already report is not something a model should supply a second time. What a model adds is the part no sensor can read — what this wearer's heart rate *should* be at this effort, how far today sits from their normal, which of their past situations this moment resembles. Designing the screen as measurement beside inference, rather than folding both into one number, keeps a single source for each value and leaves the wearer to draw the conclusion the model cannot. See <doc:Machine-Learning-Primer> for the framing of features, labels, and the trade-offs that decide which family fits.

### Verifying the signal exists

Before fitting anything, confirm the quantity the model must predict actually moves with something the watch can sense. The cheapest verification is descriptive: window the raw stream and read its shape. If a rolling mean over the buffered samples never drifts, or a rolling standard deviation never settles, the sensor is not carrying the information the model would need. No model family supplies a missing signal.

```swift
import Quiver

// Pace samples (m/s) from one steady stretch of a run.
let pace: [Double] = [3.5, 3.4, 3.6, 3.4, 3.5, 3.6, 3.3, 3.5]

pace.rollingMean(window: 3).last // 3.4667 — the trend a model would fit against
```

When the quantity is geometric rather than statistical — grade from gravity, gait symmetry from stride — the same verification runs on vector primitives: a normalized gravity reading dotted against the vertical reference produces a slope cosine that separates a flat road from a climb, or it does not. See <doc:Vector-Operations> for the operations and <doc:Linear-Algebra-Primer> for the geometry.

> Note: For noisy mid-stride readings, average over a short still window, or read Core Motion's `gravity` vector, before trusting the angle.

### Separating detection from attribution

The watch's job is detection: flagging that today departs from the wearer's baseline. Attribution — why it departed — is a different job, and it belongs to the wearer, the phone, or a later session, not to the wrist. Heat, altitude, caffeine, and genuine fitness change all move a heart-rate-to-power relationship in the same direction; a departure flag does not distinguish them, and a design that promises it will overclaim.

The empirical rule is the watch-sized version of this discipline. `empiricalRule()` compares the observed fraction of a buffered window inside one, two, and three standard deviations against the fractions a steady, Gaussian window is expected to show. A within-1σ fraction far below the expected value says the window is not steady — that gap is the flag, and nothing more.

```swift
import Quiver

// Ten pace samples (m/s) from one window of a steady run.
let paceBuffer = [3.5, 3.3, 3.4, 3.6, 3.2, 3.5, 3.4, 3.3, 3.6, 3.2]

guard let rule = paceBuffer.empiricalRule() else { return }
print(rule.within1Sigma) // 0.60 — observed 1σ fraction
print(rule.expected1Sigma) // 0.6827 — expected for a steady window
```

A significant gap flags terrain changes or surges rather than diagnosing their cause. See <doc:Statistics-Primer> for center and spread, and <doc:Inferential-Statistics-Primer> for the framework that turns flags into confident action.

> Note: A within-1σ fraction far below 68.27% means the buffered effort is not steady or normally distributed. That gap is a flag to investigate, not a diagnosis.

### Choosing the fitting cadence

The cadence is a battery decision as much as a math one. The natural fitting moments are three: the end of a session, when the data is complete and the app is already awake; a scheduled background refresh, when several sessions fold into one fit; or never on the watch at all, with the phone fitting and transmitting the value. Each single-pass fit in this guide is cheap enough to run at session end. Each iterative fit is expensive enough that its cadence should be measured in days, not samples.

One rule constrains all three moments: fit between sessions, predict within one. A model re-fit partway through a workout absorbs the very deviation it exists to report. Suppose the baseline is re-fit at minute seventy of a long run, on the samples from minutes zero through seventy. Those samples include the elevated heart rate the wearer needs flagged, so the refreshed model now expects it, the departure collapses toward zero, and the watch reports nothing unusual during the one stretch that was. The fit has explained away its own signal.

> Important: Fit between sessions, predict within one. A model re-fit mid-session absorbs the deviation it exists to report.

```swift
import Quiver

// At session end, after the last sample has arrived.
func sessionDidEnd(power: [Double], heartRate: [Double]) throws -> LinearRegression {
    try LinearRegression.fit(features: power, targets: heartRate)
}
```

Within a session the app only predicts; the fitted value is read, never rewritten.

Where that fit runs matters as much as when. A fit on the main thread stutters the screen the wearer is looking at, and on a watch the cooldown screen at session end is exactly where a wearer is looking. The fit belongs in a detached task, and because every fitted value Quiver produces is an immutable `Sendable` value, it crosses back to the main thread as a plain assignment — no locking, no copy, no bridging type between the model and the view. A prediction is different: scoring one window against an already-fitted value is cheap enough to run wherever it is needed. The expensive step moves off the main thread; the cheap one does not have to. See <doc:Concurrency-Primer> for the task-and-observation pattern, <doc:Optimization-Primer> for what an iterative fit spends per pass, and <doc:Model-Persistence> for the value that carries between sessions.

## Implementing on device

With the decisions made, implementation is a short pipeline: fit a value from the wearer's own data, persist it, score new samples against it, and refresh it only when new data carries ground truth.

### Fitting a personal baseline

Every model Quiver ships produces the same kind of artifact from the wearer's data: a small fitted value. A regression stores coefficients. A classifier stores per-class statistics, or the labeled points themselves. A clustering fit stores centroids. Different shapes, same idea — the wearer's history compressed into a value the watch can score new samples against and persist unchanged.

The lead example fits one family's shape: a personal heart-rate-versus-power regression for cycling. At the end of a ride, we fit a linear regression of heart rate against power on the workout's samples. The fitted slope is the wearer's heart-rate-per-watt sensitivity, a baseline that varies hugely between individuals.

```swift
import Quiver

// Seven aligned samples from one ride — power (watts) as the feature, HR (bpm) as the target.
let power: [Double] = [100, 125, 150, 175, 225, 250, 275]
let hr: [Double] = [100, 110, 120, 130, 150, 160, 170]

let model = try LinearRegression.fit(features: power, targets: hr)

// Score 200 watts against this wearer's fitted slope.
print(model.predict(200.0)) // 140.0 — expected HR at 200W for this wearer
```

The fitted `model` is a value with a `predict` method. A different wearer's data fits a different slope and a different intercept, and the predicted heart rate at 200 watts lands somewhere else; the model is *personal* in exactly that sense. This example shows the regression family's shape — coefficients plus a `predict` call — but the fitted-value pattern is identical across families: <doc:Naive-Bayes> stores per-class means and variances, <doc:Nearest-Neighbors-Classification> stores the labeled points, and <doc:KMeans-Clustering> stores centroids. With a single feature there is no overlap between predictors to destabilize the fit, so the closed-form least-squares line is exact and stable. The moment several correlated signals enter the same fit — power alongside pace and grade, which carry much of the same information — the plain fit turns unstable and the regression needs a regularized form instead. See <doc:Ridge-Regression>.

### Choosing between a single pass and an iterative fit

Fitting methods come in two shapes. A **closed-form** method answers in a single pass: the normal equation behind `LinearRegression` is one matrix solve that finds the best line without iteration. An **iterative** method — `GradientDescent` and anything built on it — improves a guess step by step until the updates stop mattering. On the watch, the single pass wins by default: its cost is known before it starts, it cannot fail to converge, and it finishes between samples. Iterative methods expose knobs — step size, iteration cap, tolerance — whose wrong settings return a half-converged value that looks correct.

> Important: `Ridge` reaches its coefficients by descent, not in a single pass. Reaching for it to steady a collinear fit also moves the fit from one shape to the other.

The split does not follow the model family, and this is worth checking before assuming. `Ridge` is a regression like `LinearRegression`, but it reaches its coefficients by descent under a learning rate and an iteration cap, so reaching for it to steady a collinear fit also moves the fit from one shape to the other. An iterative fit on the wrist is not disqualified; it is required to report how it stopped, and the app is required to read that report rather than trusting the returned value on sight. See <doc:Gradient-Descent> for the iterative machinery and <doc:Optimization-Primer> for the convergence trade-offs.

### Persisting the fitted model

A model fit at the end of one ride is only useful if the next ride can read the fitted value. Whatever the family, the fitted value is `Codable`, which means one `JSONEncoder` call turns it into bytes the watch writes to `Documents/` and decodes at the next launch. See <doc:Model-Persistence> for the encode-once, decode-on-launch shape every persisted model on the watch shares.

```swift
import Quiver
import Foundation

// Refit the same HR-vs-power model so this block stands alone.
let power: [Double] = [100, 125, 150, 175, 225, 250, 275]
let hr: [Double] = [100, 110, 120, 130, 150, 160, 170]
let model = try LinearRegression.fit(features: power, targets: hr)

// Encode the fitted value and write it to the Documents directory.
let documentsURL = FileManager.default
    .urls(for: .documentDirectory, in: .userDomainMask)[0]

let encoded = try JSONEncoder().encode(model)
try encoded.write(to: documentsURL.appendingPathComponent("hr-power.json"))
```

### Detecting that today departs from the baseline

Persisting the fitted value sets up the question the whole system exists to answer: *did today depart from the baseline?* The question is family-independent; only the answer's shape changes with the model.

For a regression, the answer is the **residual** — the gap between the value the fitted coefficients predict and the value the sensor reports. The next session decodes the stored model and compares expectation against observation:

```swift
import Quiver
import Foundation

let documentsURL = FileManager.default
    .urls(for: .documentDirectory, in: .userDomainMask)[0]
let data = try Data(contentsOf: documentsURL.appendingPathComponent("hr-power.json"))
let model = try JSONDecoder().decode(LinearRegression.self, from: data)

// Today's observed HR (bpm) at 200 watts, against the wearer's stored slope.
let observed: Double = 152
let expected = model.predict(200.0)
print(observed - expected) // 12.0 — today's HR sits 12 bpm above baseline
```

A residual this large flags that today's heart-rate-to-power relationship has shifted from baseline. Heat, altitude, cardiac drift, caffeine, or a genuine fitness change can all move it; the residual surfaces the divergence, and interpreting it is the wearer's call — detection and attribution stay separate.

For a classifier, the answer is the **class probability**. A window the model used to call *walk* at 0.95 confidence now returns 0.55; the departure reads as collapsing confidence rather than a wrong label, because the label alone never changes until it suddenly flips. Watching the probability trend gives the glance a smoother, more honest signal than the discrete prediction.

For a clustering fit, the answer is the **centroid distance**. A fit that groups the wearer's windows by effort has a center for each group; a new window that sits further from its nearest centroid than any window in the fit did is a departure by definition, whether or not it crosses into a neighboring cluster.

### Fitting from a calibration session

Some values cannot be fit from telemetry alone; they need labels. A **calibration session** is a short, wearer-driven flow: the app asks the wearer to perform each activity — or each intensity — for a fixed window while recording the label, and those labeled windows become the fit. The requirement is labels; any labeled method qualifies. `GaussianNaiveBayes` fits per-class statistics in a single pass, the watch-friendly shape. <doc:Nearest-Neighbors-Classification> and <doc:Logistic-Regression> serve the same calibration data behind their own fitted values, chosen by the shape of value the model must supply.

Features come from fixed-length sensor windows — rate, magnitude, symmetry — aggregated into one feature vector per window. The columns below are cadence in steps per minute, vertical oscillation in meters, and arm swing as a normalized magnitude.

```swift
import Quiver

// Four labeled windows from a calibration session: [cadence, verticalOscillation, armSwing].
let calibrationFeatures: [[Double]] = [
    [170, 0.06, 0.30], // run
    [165, 0.07, 0.32], // run
    [120, 0.03, 0.15], // walk
    [115, 0.03, 0.14] // walk
]
let calibrationLabels: [Int] = [0, 0, 1, 1] // 0 = run, 1 = walk

let activity = GaussianNaiveBayes.fit(features: calibrationFeatures, labels: calibrationLabels)

// Classify the next live window during the workout.
let live: [[Double]] = [[168, 0.06, 0.29]]
print(activity.predict(live)) // [0] — a run
print(activity.predictProbabilities(live)) // per-class confidence
```

The fit-on-the-wrist shape captures a different personal artifact for each discipline: a heart-rate-versus-power curve, a walk-vs-hike-vs-climb classifier, a stroke-type model. Each persists to `Documents/` as a `Codable` value the next session decodes unchanged.

### Re-fitting on what has ground truth

Sessions keep arriving after the fit, but only some of what arrives carries ground truth: a labeled calibration window, a GPS-graded distance, a wearer-confirmed activity. Re-fitting on what has ground truth means folding only the graded windows into the fit and leaving the rest as observation. A full retrain on every new sample sounds thorough and is neither — ungraded windows dilute the fit with the model's own guesses, and each full retrain on the wrist spends battery the glance needs more. Treat re-fitting as a merge of labeled data, scheduled at the fitting cadence chosen during planning, not as a reflex after every session.

```swift
import Quiver

// Stored calibration windows, plus whatever the latest session graded.
var features: [[Double]] = storedFeatures
var labels: [Int] = storedLabels

for window in session.windows {
    guard let label = window.confirmedLabel else { continue } // ungraded: observe only
    features.append(window.features)
    labels.append(label)
}

let refreshed = GaussianNaiveBayes.fit(features: features, labels: labels)
```

> Note: A classifier re-fit on its own predicted labels is a self-training loop. It drifts with nothing to correct it, because no sensor on the wrist grades a class.

The rule holds across families: what a sensor or a wearer can grade may re-fit, and what only the model itself produced may not. A regression whose target is an observed value — heart rate against power, where the heart rate is measured — re-fits honestly on every completed session, because the target was never a guess. A classifier has no such grader on the wrist, so it stays frozen between calibrations rather than learning from labels it wrote itself.

### Holding the unfitted state as an Optional

Before the first fit — first launch, or a fresh install after a restore — there is no model, and the code should say so. Holding the fitted value as an `Optional` makes the unfitted state a value the compiler tracks, not a file-existence check scattered through the app.

```swift
import Quiver
import Foundation

struct SessionState {
    var baseline: LinearRegression? // nil until the first fit lands
}

func loadBaseline(from documentsURL: URL) -> LinearRegression? {
    guard let data = try? Data(contentsOf: documentsURL.appendingPathComponent("hr-power.json"))
    else { return nil }
    return try? JSONDecoder().decode(LinearRegression.self, from: data)
}
```

> Tip: `nil` is the calibrating state. Holding the fitted value as an `Optional` gives the interface a case to render rather than a placeholder number to invent.

A prompt to calibrate is not the only thing that screen can show. A descriptive statistic needs no fit and no history beyond the current session, so the parts of the interface that depend on one can work from the first workout while the fitted parts wait. Percentiles over the wearer's own readings place a value in their own range without a model:

```swift
import Quiver

// This wearer's heart-rate readings so far — no fit, no stored model.
let readings: [Double] = [98, 112, 124, 131, 138, 145, 151, 160]

readings.percentile(80) // 148.6 — the top of this wearer's own range
```

The distinction worth designing around is what each number needs. A percentile needs only the readings in hand. An expected value needs a fitted baseline, and therefore needs sessions the wearer has not run yet. Splitting the screen along that line means the app says something true on day one and says more as history accumulates, rather than showing nothing until the first fit lands.

The glance reads the same `Optional`. A `nil` baseline renders the unfitted state as its own screen — a prompt to complete a calibration ride — not a placeholder number and not a crash.

## Designing for the glance

### The few seconds a wearer gives the screen

A person glances at a watch for seconds, not minutes. Every value the model produces competes for that window, and most of it will lose. The glance is the harshest consumer named in *Deciding what the model supplies*, and it earns its own section because it disciplines everything upstream: how many values ship, what shape they take, and what words accompany them.

### One number, one supporting value, one word

The display rule that survives the glance is a budget of three elements: one number, one supporting value, one word. **One number** is the headline — the value the wearer came to check. **One supporting value** is the context that makes the number interpretable — the baseline it departs from, the unit, the day's trend. **One word** is the interpretation — *steady*, *surge*, *above baseline* — so the wearer can absorb the meaning without parsing digits. Three elements is a budget, not a layout: a complication that fits all three has room for nothing else, and a screen that needs more than three is answering a question the wearer has not asked.

### Choosing the surface for the value

Each surface of the platform has its own update cadence and its own space. A complication gets one scalar and a handful of updates per hour. An in-workout view gets per-moment values at the delivery cadence of the sensor. A Smart Stack widget gets a summary computed at session end.

The mismatch between those cadences is the design constraint. A heart-rate stream arriving several times a minute against a complication refreshed a few times an hour means all but a handful of computed values are discarded unseen. That is not a rendering problem to solve with a faster refresh; it is a signal that the wrong value was routed to the surface. A complication earns a value that stays true between refreshes — a session score, a daily baseline, a state word — and the per-moment stream belongs where a person is already looking.

The model supplies a different shape of value to each surface, so the surface is chosen with the value: a per-moment stream on a complication floods a surface that updates hourly, and a session-level summary inside an active workout answers a question the wearer will not ask until the cooldown.

The workout screen has a second cadence the designer does not choose. When the wrist lowers, the display enters its Always On state, and an app with an active workout session may refresh at most once per second — against roughly thirty times per second while the wearer is looking. That ratio is the design constraint: a value worth computing thirty times a second is a value the wearer sees only when actively watching, and everything else on the screen has to remain true and readable at one update per second. The practical rule is to compute what the screen shows at the cadence the screen can show it, and to let the reduced state drop precision rather than drop meaning — fewer decimal places, not a stale number or a blank.

> Important: An app with an active workout session refreshes at most once per second in the Always On state. A per-moment value must stay readable and correct at that rate.

### Routing per-moment values and sequence properties

The values a model produces sort into two kinds, and they route to different surfaces. **Per-moment values** — heart rate now, pace now, power now — belong on the workout screen, updated as the delivery stream arrives. **Sequence properties** — whether the session departed from baseline, whether the trend is rising — are only knowable once a window closes, and they belong on summaries, complications, and widgets.

The test is whether the value is defined for a single sample. A prediction scored against one incoming window is; the spread of a session is not, because a standard deviation over two samples is a number the interface should never show. The same buffer answers both questions at different moments:

```swift
import Quiver

let baseline = try LinearRegression.fit(features: [100.0, 150, 200, 250],
                                        targets: [100.0, 120, 140, 160])
let sessionHeartRates: [Double] = [128, 131, 134, 130, 136]

// Per-moment: defined for the newest sample, safe to render live.
baseline.predict(200.0) // 140.0 — expected for this wearer at 200 watts

// Sequence property: defined only once the window has closed.
sessionHeartRates.standardDeviation() // 3.1937 — meaningless mid-session
```

Mixing the routing in either direction fails the same way: the glance receives a stream it cannot pace, or the workout screen receives a summary that is stale the moment it renders.

### Standing on its own

Every surface must be interpretable without opening the app. A complication has no room for a legend, a widget has no footnote, and neither can rely on color alone. The number-and-word pair must mean the same thing on every surface it appears on: if *surge* means one thing in the app and another on the watch face, the wearer learns to trust neither.

This extends to the phone. A watch app may run with its paired phone out of range, in another room, or switched off, so a design that fetches its model or its baseline from the phone has a screen with nothing to show at the moment the wearer looks. The fitted value lives on the wrist, persists there, and is read there. The phone may enrich the experience with a larger screen and a longer history; it may not be a dependency the glance waits on.

## Operating within the platform's constraints

### Sensor timing and sample rates

Each stream arrives at its own cadence. Heart rate lands roughly once per beat and irregularly. Accelerometer samples arrive tens of times a second. Location updates about once a second. A ten-second window holds hundreds of accelerometer samples, ten location fixes, and perhaps twenty heart-rate readings — three columns of a feature row that do not line up.

> Note: Carry a slow channel forward rather than interpolating it. A held value is honest about being stale; an interpolated one invents a reading the sensor never took.

Two rules follow. Window each stream independently, then align the windows rather than the samples: the feature row is one value per stream per window, not one row per delivery. And carry the slow channels forward rather than interpolating them. A held value is honest about being stale; an interpolated one invents a reading the sensor never took, and a model fit on invented readings will faithfully measure the invention.

> Note: A window labeled by heart rate alone can invert an interval and its recovery, because the reading arrives after the effort that caused it.

There is a second timing effect the rates do not show. The optical heart-rate sensor lags the wearer's actual effort — a hard interval can be over before the reading catches up, and the peak often lands during the recovery that follows. A window labeled by heart rate alone can therefore label the effort easy and the recovery hard, exactly inverted. Features that combine heart rate with an immediate signal such as cadence or pace do not have this failure.

### Derived signals amplify noise

Every quantity computed from a sensor rather than read from it — respiratory rate derived from heart-rate variability, grade derived from barometric change, cadence derived from acceleration — inherits the sensor's noise and adds its own. The empirical-rule gap from the planning section is the honest check on a derived signal: if the derived values do not hold a steady shape in the windows where they should, the derivation is amplifying noise, not extracting signal.

> Note: A signal derived from heart rate in the respiratory band (0.15–0.40 Hz) needs spectral resolution `Δf = sampleRate / paddedLength` fine enough to separate structure. A 120-second window is typically required to avoid blurring the signal.

> Tip: These primitives compose into `powerSpectralDensity` and `trapezoidalIntegral` to extract motion and effort metrics from a window.

See <doc:Physics-Primitives-Primer> for the physical interpretation of integrals, derivatives, and frequency content over short sensor windows.

### Handling the delivery stream

`HKLiveWorkoutBuilder` reports what it has collected since the last callback, not a rewindable history. The delegate appends each delivery to a fixed-capacity buffer of `[Double]` — by count for high-rate streams, by time for low-rate ones — and the model reads the buffer, never the delivery. Holding every sample since the session started is how a correct model exhausts a workout that should have run for hours.

Not every value the app maintains is a window, and the capacity rule applies to only one of them. A **feature window** answers a question about recent conditions — the last ten seconds of cadence, a rolling mean of pace — and a bounded buffer is correct, because older samples are not part of the answer. An **accumulator** answers a question about the whole session — total distance, energy burned, a time-weighted load — and it must see every sample that arrives. Capping an accumulator does not cost precision; it produces a wrong total, and one that looks plausible. Keep the two separate: a bounded buffer the model reads features from, and a running total the session adds to and never trims.

Three deliveries need handling before either is trustworthy. A paused session keeps the query alive, so samples arriving while the wearer is stopped at a crossing must be dropped rather than recorded, and an automatic pause routes to the same handler as a manual one. Duplicate deliveries are normal, so a sample whose timestamp does not advance past the last one is discarded. And a wrist lowered for several minutes backfills a burst on the way up, which is a batch to append at once rather than a signal that effort spiked.

```swift
import HealthKit

struct MetricBuffer {
    private(set) var values: [Double] = []
    private var lastRecorded: Date = .distantPast
    let capacity = 300

    // Append only what advances the session clock.
    mutating func record(_ statistics: HKStatistics, unit: HKUnit, sessionIsRunning: Bool) {
        guard sessionIsRunning,
              statistics.endDate > lastRecorded,
              let latest = statistics.mostRecentQuantity() else { return }

        values.append(latest.doubleValue(for: unit))
        if values.count > capacity { values.removeFirst(values.count - capacity) }
        lastRecorded = statistics.endDate
    }
}
```

### Working within the timer and power budget

Background work on the watch is measured in seconds, and the floating-point unit carries a precision profile tuned for battery life, not for numerical analysis. These budgets decide which fitting methods belong on the wrist.

The risk is not a fit that runs slowly. At the sizes a wrist produces — tens to low hundreds of windows, a handful of features — a single-pass fit finishes in a fraction of the budget. The risk is an iterative fit that stops because it ran out of iterations rather than because it converged, and returns a plausible-looking value with no error to catch. An iterative method therefore has to report which of the two happened, and the app has to read that report before trusting the result.

> Tip: Sensor sampling and the display dominate a workout's power draw. One fit at session end is not what drains the battery.

It is also worth knowing what the fit does not cost. Sensor sampling and the display dominate a workout's power draw by orders of magnitude; one fit at session end is not what drains the battery. Optimizing the arithmetic while streaming the accelerometer at full rate is effort spent on the wrong term.

> Important: Iterative-convergence methods over bulk data — `KMeans` with high `maxIterations`, `KNearestNeighbors` over thousands of training points, `Matrix.invert` on large dimensions — do not belong on the wrist. The background timer budget and the watch floating-point profile can return non-converged results that look correct. An iterative fit over the wearer's own history is a different matter, and it is allowed on the condition above: it reports how it stopped, and the app reads that report. Move iterative-bulk fitting to iOS or the server.

### Knowing what accuracy cannot be claimed

Small per-user data means wide uncertainty, and a departure flag computed from a handful of sessions carries a confidence interval wide enough to straddle the decision it feeds. There is a sharper problem underneath: a model fit on one wearer's history has no held-out set. Every session is training data, so there is no honest accuracy figure to compute and none to display. An app that shows a confidence percentage derived from the data the model was fit on is reporting how well the model memorized, not how well it predicts.

> Warning: A confidence percentage computed from the data a model was fit on reports how well the model memorized, not how well it predicts. On one wearer's history there is no held-out set, so there is no such figure to show.

What the watch can honestly report is the departure itself, stated as the computation that produced it. A number the model can defend — the observed value, the expected value, the gap between them — survives a wearer asking where it came from. A label the model cannot defend, naming a cause it has no sensor for, does not.

The same discipline applies to what the model is allowed to have learned. A baseline fit from easy sessions has never seen a hard one, and its expectation at race effort is an extrapolation beyond anything in the wearer's history. Reporting the number of sessions behind a value lets the interface say less when it knows less. See <doc:Numerical-Literacy> for reading uncertainty in computed values and <doc:Inferential-Statistics-Primer> for the confidence-interval framing.

## Where to go from here

The sections above each have a deeper layer of math underneath them, and that math is the next step for watchOS developers moving into numerical work. <doc:Statistics-Primer> builds the vocabulary of variance, distributions, and the empirical rule that the planning sections lean on. <doc:Linear-Algebra-Primer> extends vectors and dot products into the geometric operations that read sensor data as orientation and alignment. <doc:Physics-Primitives-Primer> covers the signal-processing surface — integrals, derivatives, and frequency content — that turns short sensor windows into physical quantities. <doc:Machine-Learning-Primer> closes the loop with features, labels, training, and the trade-offs that decide which fitted shape to reach for on the wrist.

> Experiment: **The Quiver Notebook** is the right place to feel how `empiricalRule()` reads against personal baselines. Load a vector of recent heart-rate samples, compare the observed within-1σ fraction to the theoretical 68.27%, and watch the gap shift as the workout type changes. The same `Codable` model that fits in the Notebook decodes unchanged on the watch. See <doc:Quiver-Notebook>.
