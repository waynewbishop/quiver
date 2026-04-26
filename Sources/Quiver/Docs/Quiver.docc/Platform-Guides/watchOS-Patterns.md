# watchOS Patterns

Classifying effort, building personal baselines, and sizing models for Quiver on Apple Watch.

## Overview

The <doc:watchOS-Guide> covers the foundations — rolling windows, sensor realities, and authorized sessions. This article builds on those foundations with applied patterns that solve specific problems watchOS developers encounter when shipping Quiver-based features.

### Discovering effort regimes from biometric streams

One of the clearest watchOS use cases for Quiver is discovering structure in a live biometric stream during an active workout. A runner on a rolling-hill loop moves through several distinct effort regimes — recovery on the flats, aerobic on the climbs, threshold on the hard sections, and easy returns on the descents. These regimes are visible in the combined pattern of heart rate, pace, grade, and cadence, even though no single signal captures them on its own.

#### Why multi-feature clustering matters

Clustering over raw heart rate alone will only rediscover duration-weighted mode heart rate, which is not the same thing as a physiological zone. Clustering over paired signals — heart rate with pace-and-grade, for example — produces behavioral clusters that correspond to how the body is actually working at each moment.

```swift
import Quiver

// Each row: [heart rate, grade-adjusted pace, cadence, grade]
let samples: [[Double]] = [
    [132, 6.5, 168, 0.0],  [135, 6.4, 169, 0.5],  [138, 6.3, 170, 1.0],   // recovery
    [155, 5.8, 175, 3.5],  [158, 5.7, 176, 3.8],  [160, 5.6, 177, 4.0],   // aerobic climb
    [172, 5.0, 182, 2.0],  [175, 4.9, 184, 2.5],  [178, 4.8, 185, 3.0],   // threshold push
    [145, 6.8, 172, -3.0], [142, 7.0, 170, -3.5], [140, 7.2, 168, -4.0]   // descent
]

// Scale features so no single signal dominates the distance calculation
let scaler = FeatureScaler.fit(features: samples)
let scaled = scaler.transform(samples)

// Cluster into four effort regimes
let model = KMeans.fit(data: scaled, k: 4, seed: 42)
```

#### Behavioral clusters, not physiological zones

The resulting clusters describe how the athlete is currently working — but they describe *behavior*, not calibrated physiology. A lactate threshold test is still the gold standard for identifying physiological zones. What multi-feature clustering provides is a view of which segments of the current workout share structural similarity, adapted to the athlete's own data rather than a generic heart rate table.

Apple Watch sensors capture mechanical and cardiovascular load — heart rate, cadence, pace, grade, motion — in real time. They do not measure hydration, sleep quality, muscular damage, or psychological stress. Effort regimes discovered from the sensor stream reflect what the watch can observe, which is a meaningful subset of what the body is doing, not the whole picture.

> Tip: Feature scaling is load-bearing for clustering. Heart rate in the 130–180 range, pace in the 4.5–7.5 range, and grade in the −4 to +4 range live on wildly different scales. Without `FeatureScaler`, the distance calculation would be dominated by whichever column has the largest raw numbers. See <doc:Feature-Scaling>.

### Building a personal baseline

On-device modeling on watchOS becomes compounding when a model persists useful statistics across sessions rather than starting fresh every time. A personal baseline is a set of summary statistics, cluster centroids, or scaling parameters that reflect the user's own data across the history of their activity. Each new session is interpreted relative to that baseline, and the baseline itself updates as new sessions arrive.

#### Persisting across sessions

`Pipeline` bundles a `StandardScaler` and a model into a single `Codable` value. At session end, encode the pipeline to disk. On next launch, decode it and predict immediately — the scaler and model always travel together, so there is no risk of a mismatched pair. For the full persistence pattern, see <doc:Model-Persistence> and <doc:Pipeline>.

#### What a baseline is not

A Quiver baseline is a lightweight, honest way to make the model user-specific — but it is not adaptation in the physiological sense. The user's fitness changes over weeks of training, and a persisted scaler tracks the *sensor range* of that change, not the underlying adaptation. It is also not equivalent to training-load models like CTL, ATL, or TSB from sports science, which use decades-validated exponentially weighted averages. For athletes who need training-load tracking, that is a separate problem with separate tools.

> Tip: When the baseline file does not yet exist — the user's first workout — fall back to fitting on the current session alone. After the session ends, persist the result. On subsequent sessions, the baseline is available from the start.

### When to train, when to predict

One of the engineering decisions in-session modeling on watchOS forces is when to refit the model and when to simply run predictions against a model that was fit earlier. The answer depends on how quickly the signal changes and how stable the patterns are within a session.

#### Stable signals, single fit

For signals that change slowly — a steady-state heart rate during a long run, or a stable navigation pattern during a drive — refitting every few seconds is wasteful. A model fit once at the start of the session and used for the entire session will behave well, and predictions are cheap enough to run on every new sample.

#### Rapidly-changing signals, periodic refit

For signals that change rapidly — interval workouts, turn-by-turn navigation, ambient context shifts — the model needs to be refit as the window refreshes. The middle path is periodic refitting: refit every N seconds or every N samples, and run predictions on every new arrival in between.

```swift
import Quiver

// A stable model fit once per session, reused for many predictions
let sessionModel = KNearestNeighbors.fit(features: window, labels: labels, k: 5)

// Predictions run on every new sample — fast, no refit needed
func classify(_ newSample: [Double]) -> Int {
    return sessionModel.predict([newSample])[0]
}

// Periodically refit as the window refreshes — every 30 seconds, for example
var lastFitTime: Date = .now
func maybeRefit(currentWindow: [[Double]], currentLabels: [Int]) -> KNearestNeighbors {
    guard Date.now.timeIntervalSince(lastFitTime) > 30 else {
        return sessionModel
    }
    lastFitTime = .now
    return KNearestNeighbors.fit(features: currentWindow, labels: currentLabels, k: 5)
}
```

K-Nearest Neighbors is particularly well suited to this pattern because its `fit()` is effectively free — it just stores the training data — while its `predict()` does the real work.

> Tip: For heart rate specifically, avoid refitting during the first minute after an effort change. The heart rate lag means the sensor is still catching up to the actual work level, and a model fit during that window will be learning from a stale signal. Wait for the signal to stabilize before refitting.

### Sizing a model for the wrist

Quiver's classifiers and regressions are small and fast enough that fitting them on sensor-window data is a millisecond-scale operation, not a seconds-scale one. The models themselves are Swift structs built from plain arrays and numbers, so their memory footprint scales with the training data they store, not with a compiled runtime.

#### Measured performance

The numbers below come from Quiver's stress test suite, built in release mode on current Apple Silicon. Apple Watch silicon is slower than macOS, so wall-clock times on device will be higher — but the order of magnitude holds, and the memory deltas transfer directly because Swift's `Array` and struct layout behave identically across platforms.

On a thousand samples with ten features — roughly seventeen minutes of 1 Hz sensor data — K-Means fit completes in about one millisecond with no measurable memory increase. K-Nearest Neighbors fit and one hundred predictions against that same training set takes three to four milliseconds and stays under a tenth of a megabyte. Gaussian Naive Bayes scales further: ten thousand samples with twenty features, plus a thousand predictions, finishes in about one millisecond at two-tenths of a megabyte. Linear Regression on five thousand samples with ten features takes about two milliseconds and uses roughly 2.5 MB at peak — the largest of the four, and still well inside the memory budget a foreground watchOS app has available.

#### What these numbers mean for watchOS

These input sizes match realistic watchOS workloads. A thousand samples is a comfortable rolling window for live classification during an active session. Ten features is enough for a multi-signal classifier built from heart rate, pace, grade, cadence, and a few derived values.

The operations worth thinking about for watchOS are the ones Quiver is built for: fitting a classical model on a rolling window of sensor data, running predictions against it as new samples arrive, and persisting the result across session boundaries. These all live in the millisecond-per-operation regime, which means a `.userInitiated` task can complete a fit without interrupting the user interface and without being interrupted by the watchOS runtime.

### See also

- <doc:watchOS-Guide> — Foundations: sensor streams, sensor realities, and authorized sessions
- <doc:Model-Persistence> — Saving and loading fitted models with `Codable`
- <doc:Pipeline> — Bundling a scaler and model into a single matched pair
- <doc:Feature-Scaling> — Min-max normalization for multi-signal classification
- <doc:KMeans-Clustering> — K-Means clustering for unsupervised learning
- <doc:Nearest-Neighbors-Classification> — K-Nearest Neighbors classifier
