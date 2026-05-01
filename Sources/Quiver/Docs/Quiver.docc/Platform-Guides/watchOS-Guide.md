# watchOS Guide

Live sensor processing and in-session summaries on Apple Watch with Quiver.

## Overview

Apps on Apple Watch work with data the moment it arrives. A workout view smooths a noisy heart-rate stream into a stable display. A sleep tracker groups motion samples into stages while the user is still asleep. A focus-mode app classifies ambient motion into walking, sitting, or driving. A weather complication folds an hourly forecast into a single number that fits on a watch face. Each new sample arrives within a few seconds of the last one, and the computation has to keep up.

Quiver fits this reality by running entirely in-process as pure Swift, with no bridging layer and no external dependencies. Rolling windows of `mean` and `std` smooth a live sensor stream without dropping samples. `KNearestNeighbors` classifies the current window against a small set of labeled patterns the app shipped with. `Pipeline` keeps the scaler and the classifier as one matched pair, so the same transformation applied during fitting applies during prediction. Because every Quiver model is `Sendable`, the fitted state crosses task boundaries during an `HKWorkoutSession` without locks or copies.

### Live sensor streams and rolling windows

Quiver on Apple Watch is almost always applied to a rolling window of recent samples, not a static training set. Sensors arrive continuously. The model has to track what is happening *now* without accumulating unbounded history, because memory is finite and stale data dilutes the signal.

The pattern is simple: keep a fixed-size buffer of the most recent feature vectors, re-fit the model when the buffer changes, and use the fitted model to classify or predict the next incoming sample. As old samples fall off the back of the buffer and new ones arrive at the front, the model adapts continuously to the current conditions.

```swift
import Quiver

// A rolling window of the most recent sensor readings
var window: [[Double]] = []
let windowSize = 60  // 60 samples at 1 Hz = one minute of data

func ingest(_ sample: [Double]) {
    window.append(sample)
    if window.count > windowSize {
        window.removeFirst()
    }
}

// Re-fit the model whenever the window has filled with fresh data
func currentModel() -> KMeans {
    let scaler = FeatureScaler.fit(features: window)
    let scaled = scaler.transform(window)
    return KMeans.fit(data: scaled, k: 3, seed: 42)
}
```

Because every Quiver model is a value type, the re-fit produces a fresh model on every call and the previous one is discarded. Nothing mutates, nothing leaks, and the old model remains valid for anyone who happens to be holding a reference to it from an earlier frame — a property that matters the moment a second concurrent task is reading predictions while the main task is refitting.

> Tip: The right window size depends on the signal and the sample rate. A 60-second window at 1 Hz gives the model enough data to find structure without overwhelming the memory budget. For rarer events (lap detection, mode transitions) the window may need to be larger. For fast-moving signals (motion bursts) it may need to be smaller.

### Sensor realities on watchOS

Apple Watch sensors have characteristics that a developer used to offline training data needs to understand before building a rolling-window classifier. Machine learning is only as good as the data feeding it, and the data coming off the wrist has a few practical quirks worth knowing.

**Sample rate mismatch.** Different sensors report at different rates. Heart rate updates roughly once per second during a workout, motion data can arrive at up to 100 Hz, GPS-derived pace is smoothed and reported at around 1 Hz with variable latency, and altimeter readings depend on the activity type. A naive feature vector built by grabbing the latest value from each sensor will pair fresh motion data with stale heart rate. For most classifiers, resampling everything to a common cadence — 1 Hz is a practical floor for cardiovascular work — produces cleaner training data than mixing native rates.

**Heart rate latency.** Apple Watch heart rate readings lag true heart rate by roughly 10 to 30 seconds during rapid changes — the start of a sprint, a sudden hill, the first 60 seconds of exercise. A classifier that re-fits during those transitions will be learning from a signal that is telling it about the *previous* effort level, not the current one. Ignoring the first minute after an effort change, or time-boxing the rolling window to exclude those moments, keeps the model honest.

**Workout session states.** An `HKWorkoutSession` can be active, paused, or ended. Paused sessions still emit samples for some sensors, and those samples will contaminate a rolling window if the ingestion code doesn't filter them. Check the session state on every sample and skip ingestion when the session is not active.

> Tip: The goal of this section is not to make machine learning on watchOS sound fragile. It's to surface the few realities that separate a demo that works once from a model that behaves correctly across every run, every walk, and every paused interval a user actually puts it through.

### Running training during an authorized session

Long-running tasks on watchOS are only reliable inside an authorized session. An active `HKWorkoutSession` is the most common example: while the workout is running, the system grants the app extended runtime and continuous sensor access. A `Task.detached` that fits a model during the workout will run to completion because the workout itself is the authorization.

The pattern combines three things: a workout session to authorize the runtime, a rolling window to capture live sensor data, and a Swift Concurrency task to run the fit off the main thread so the UI stays responsive.

```swift
import Quiver
import HealthKit

@Observable
@MainActor
final class WorkoutClassifier {
    var currentModel: KMeans?
    var window: [[Double]] = []

    // Called by the sensor delegate as new samples arrive
    func ingest(_ sample: [Double]) async {
        window.append(sample)
        if window.count > 60 {
            window.removeFirst()
        }

        // Only refit when the window is full and every few seconds
        guard window.count == 60 else { return }

        // Fit in a detached task so the view stays responsive
        let buffer = window
        let fitted = await Task.detached(priority: .userInitiated) {
            let scaler = FeatureScaler.fit(features: buffer)
            let scaled = scaler.transform(buffer)
            return KMeans.fit(data: scaled, k: 3, seed: 42)
        }.value

        // Assignment runs on the main actor — the view updates automatically
        currentModel = fitted
    }
}
```

The `buffer = window` copy before the detached task captures a snapshot of the current window as a value, so the task operates on its own copy without competing with concurrent ingestion. When the fit completes, the fresh `KMeans` model crosses back into the main actor as a `Sendable` value and the assignment updates any SwiftUI view bound to `currentModel`. For the full set of concurrency patterns, see <doc:Concurrency-Primer>.

> Tip: The `guard window.count == 60` check means the model is only refitted when there is enough fresh data to be meaningful. Refitting on a half-full window produces unstable clusters and wastes battery.

