# watchOS Guide

Building a personal baseline, ranking against history, and fitting a classifier once for many predictions on Apple Watch.

## Overview

Quiver gives us the building blocks to compose an answer. A running app wants to tell a user whether the heart rate they just hit is high for them. A sleep app wants to flag a short night against their own typical. A focus app wants to know whether the current motion stretch is unusually still. Each situation has the same shape — a fresh reading on one side, the user's accumulated history on the other, and an answer composed from the arrays the app already holds.

### Building a personal baseline

The first thing many on-wrist features need is context. A `PersonalBaseline` captures that context as a few summary statistics over the user's history: the mean, the typical spread, and a few percentile breakpoints. Once computed, the baseline is held in observable state for the rest of the session and every new sample is interpreted against it. For a sleep app the history is a year of nightly hours. For a focus app the history is a week of motion samples. For the running example used below the history is a runner's accumulated heart-rate readings — but the type and the five statistics it carries are the same in every case.

```swift
import Quiver

struct PersonalBaseline: Codable {
    let mean: Double
    let standardDeviation: Double
    let standardError: Double
    let p25: Double
    let p50: Double
    let p75: Double

    static func from(history: [Double]) -> PersonalBaseline? {
        guard let mean = history.mean(),
              let standardDeviation = history.standardDeviation(),
              let standardError = history.standardError(),
              let p25 = history.percentile(25),
              let p50 = history.percentile(50),
              let p75 = history.percentile(75) else {
            return nil
        }
        return PersonalBaseline(
            mean: mean,
            standardDeviation: standardDeviation,
            standardError: standardError,
            p25: p25,
            p50: p50,
            p75: p75
        )
    }
}
```

Six Quiver methods turn an array of historical readings into a structured snapshot — `mean` for the center, `standardDeviation` for the typical spread, `standardError` for how precise the estimate of the mean is, and three calls to `percentile` for the breakpoints. The baseline is `Codable` and `Sendable` for free, so it crosses task boundaries during a workout without copies and persists to disk between sessions without ceremony.

The baseline is a snapshot, not a model. The user's fitness, sleep, or focus patterns drift over weeks of training or life changes, so the app should recompute the baseline on a cadence that matches the underlying signal — weekly for a trained runner whose resting heart rate moves with fitness, monthly for a sleep tracker, end-of-session for a focus app where every block updates the history. The shape of the recomputation is identical to the initial build; only the cadence changes.

### Ranking a sample against history

With a baseline in hand, every new sample becomes a question — where does *this* reading fall in the distribution of the user's history? Two Quiver methods handle the two common shapes of that question. The `percentileRank` method gives a 0-to-100 position; the baseline's `standardDeviation` lets us compute a z-score.

```swift
import Quiver

extension PersonalBaseline {
    // 0-100: where does this reading fall in the user's history?
    func percentileRank(of value: Double, in history: [Double]) -> Double {
        return history.percentileRank(of: value)
    }

    // How many standard deviations from the user's typical reading?
    func zScore(of value: Double) -> Double {
        return (value - mean) / standardDeviation
    }
}
```

The two views differ in what they emphasize. Percentile rank answers "is this rare?" — a reading at the 95th percentile is rare regardless of how skewed the distribution is. Z-score answers "is this extreme?" — a reading three standard deviations above the mean is extreme even when many readings fall in that direction. A dashboard often shows both, side by side.

A z-score can also be turned into a probability with `Distributions.normal.cdf`. For a reading 2.1 standard deviations above the mean, the area in the upper tail is the probability of seeing a value at least that extreme:

```swift
import Quiver

let z = baseline.zScore(of: 175.0)                                       // 2.1 above the mean
let cdf = Distributions.normal.cdf(x: z, mean: 0, standardDeviation: 1) ?? 0
let probability = 1 - cdf                                                // 0.018 — top 1.8%
```

The whole calculation runs on the watch with no network call and no permissions. A glance complication can show "this reading is in the top 2% of your history" the moment a sample arrives. For the full distribution surface — `Distributions.t`, `Distributions.chiSquared`, and inference helpers — see <doc:Working-With-Distributions>.

> Experiment: **The Quiver Notebook** is the right place to compare the distribution functions side by side. Try `Distributions.normal.cdf` and `Distributions.t.cdf` on the same array of recent readings — the normal turns a z-score into a probability, while the t-distribution answers the same question honestly for small samples where the normal assumption is too strong. Running both is the fastest way to feel when each one applies. See <doc:Quiver-Notebook>.

### Fitting a classifier once and predicting many times

Where `PersonalBaseline` summarized one signal against a user's history, the next pattern handles a different question: given several signals at once, which effort regime is the user in? The answer comes from a fit-once-predict-many classifier trained on a small labeled multi-signal dataset. The training data is collected during a short calibration session — the user runs at a known effort for a minute, taps the effort level on screen, and the app stores the multi-signal samples that arrived during that minute. A few minutes of calibration produces a few dozen labeled samples that cover every effort regime the app cares about.

During the calibration session, each sensor stream is appended to its own buffer as new samples arrive (`heartRate` from `HKAnchoredObjectQuery`, `cadence` from `HKQuantityType.runningStrideLength`, `pace` from `HKQuantityType.runningSpeed`), and the user's tap on the effort label is appended in parallel. At session end, the four aligned buffers go into a <doc:Panel> together — keeping row alignment across every column without manual bookkeeping:

```swift
import Quiver

// Calibration session result — each row across the Panel is one sample in time
let training = Panel([
    ("heartRate", [128, 132, 135, 148, 151, 154, 165, 168, 171, 178, 181, 184]),
    ("cadence",   [165, 168, 170, 175, 176, 178, 182, 184, 185, 188, 189, 190]),
    ("pace",      [9.5, 9.2, 9.0, 8.0, 7.8, 7.6, 6.8, 6.6, 6.4, 5.8, 5.6, 5.4]),
    ("effort",    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])  // 0=easy, 1=moderate, 2=tempo, 3=hard
])

// Extract the feature matrix and the label vector — no manual slicing
let features = training.toMatrix(columns: ["heartRate", "cadence", "pace"])
let labels = training.labels("effort")

// One call fits the scaler and the classifier together as a Pipeline
let pipeline = Pipeline.fit(features: features, labels: labels, k: 3)
```

The four buffers stay aligned because the app appends to all four on the same tick — each row across the Panel is one sample in time. The `Panel` type then takes over the alignment guarantee: `toMatrix(columns:)` returns the feature matrix in the order requested, and `labels(_:)` returns the effort column as `[Int]`. Calling `Pipeline.fit` takes it from there — it fits a `StandardScaler` on the raw features, applies it, trains the `KNearestNeighbors` model on the scaled data, and returns the two as one bundled value. KNN scales linearly with the training set, so a few dozen calibration samples on Apple Watch silicon is comfortable.

During the session, every new reading runs through the same pipeline:

```swift
// Every 1.5 seconds, as a new multi-signal sample arrives
func classify(_ sample: [Double]) -> Int {
    return pipeline.predict([sample])[0]
}
```

Calling `Pipeline.predict` applies the stored `StandardScaler` to the raw sample and then runs the model on the scaled result. That single call is what keeps every prediction in the same coordinate system the model was trained on. Because the pipeline is `Codable`, the fitted scaler and model travel together when written to `Documents/` at session end and decoded at the next launch — no risk of a mismatched pair from saving them separately. For the full Pipeline surface, see <doc:Pipeline>.

### Integrating with a live workout session

The pattern above runs against a simulated sample stream so the demo can be explored in isolation. In a shipping watchOS app, the same sample-rate cadence comes from `HKWorkoutSession` and the live `HKWorkoutBuilder` sample stream. The classifier, the scaler, and the predict-per-sample loop stay identical — what changes is the data source. The app starts an `HKWorkoutSession`, asks for the relevant `HKQuantityType` samples (heart rate, distance, active energy, running cadence), and feeds each new reading through `classify(_:)` as it arrives. For the authorization-and-runtime considerations of running long tasks on watchOS, see Apple's [HealthKit Workouts documentation](https://developer.apple.com/documentation/healthkit/workouts_and_activity_rings).

### Sensor realities on watchOS

Apple Watch sensors have characteristics that a developer used to offline training data needs to understand before building a real-time classifier. Machine learning is only as good as the data feeding it.

**Sample rate mismatch.** Different sensors report at different rates. Heart rate updates roughly once per second during a workout, motion data can arrive at up to 100 Hz, GPS-derived pace is smoothed and reported at around 1 Hz with variable latency. A naive feature vector that grabs the latest value from each sensor will pair fresh motion data with stale heart rate. Resample everything to a common cadence — 1.5 seconds is a practical floor for cardiovascular work — before building the multi-signal sample.

**Heart rate latency.** Apple Watch heart rate readings lag the true value by roughly 10 to 30 seconds during rapid changes — the start of a sprint, a sudden hill, the first 60 seconds of exercise. A classifier reading during those transitions will be classifying the *previous* effort level, not the current one. The fit-once-predict-many pattern handles this gracefully because the model itself does not adapt during the lag — it just labels what the sensors report.

**Session state.** A live workout session can be active, paused, or ended. Paused sessions still emit samples for some sensors, and those samples will contaminate the classifier's output if the ingestion code does not filter them. Check session state on every sample and skip the predict call when the session is not active.

> Tip: The training data is the load-bearing artifact. A multi-signal classifier built from 12 hand-labeled samples per regime works because the regimes are well-separated in feature space. Adding a tenth feature does not make the classifier better when the labels are still drawn from a 50-sample training set — invest in label quality and signal separation before adding more features.

> Experiment: [quiver-demo-watchos](https://github.com/waynewbishop/quiver-demo-watchos) is a two-tab Apple Watch app that streams a simulated workout and labels each sample two ways — by heart-rate zone alone, and by a four-signal KNN classifier.
