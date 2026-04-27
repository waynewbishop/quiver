# Concurrency Primer

Training models off the main thread and passing fitted results back to the interface.

## Overview

Swift Concurrency is the language's built-in way to run work without blocking the main thread. Tasks describe units of work, `async` functions suspend and resume, and the compiler verifies that values crossing between tasks are safe to share. Quiver fits into this model naturally because its models are value types and defined as an immutable Swift `struct`. A fitted model can be trained inside a task, returned from it, and handed to a view without locks, wrappers, or `@unchecked` annotations.

This primer covers the core patterns — training a model inside a task, keeping long-running fits off the main thread, and updating a SwiftUI view when training completes.

> Tip: New to Quiver's machine learning workflow? Start with <doc:Machine-Learning-Primer> for core vocabulary like features, `labels`, `fit`, and `predict` before diving into the concurrency patterns below.

### What Sendable means for Quiver

A type that is safe to move between tasks has no shared mutable state to protect. The Swift compiler verifies this property through a protocol called `Sendable`. When the compiler sees a `Sendable` value crossing a task boundary, it knows the value can be handed off without coordination. Every public type in Quiver conforms to `Sendable`, which is why a fitted model can flow cleanly from a background task to the caller without any extra ceremony.

```swift
import Quiver

// A fitted model is a value. Pass it between tasks freely.
let model = try LinearRegression.fit(features: sqft, targets: prices)

Task {
    // The model crosses into the task as a value — the compiler verifies it's safe
    let prediction = model.predict([2000.0])
    print(prediction)
}
```

The reason this works comes back to how Quiver builds its models. Every property is a plain value — `[Double]` coefficients, `Int` dimensions, stored statistics — and every model is immutable once `fit` returns. There's nothing to mutate, nothing to share, and nothing that could change on one thread while another thread is reading it.

The same property applies to `predict`. A fitted model can be called from any number of concurrent tasks without serialization, because prediction is a pure read against immutable properties — the model computes an answer from its stored parameters and returns, without touching any shared state along the way.

> Tip: Because models are also `Codable` and `Equatable`, the same value semantics that make them safe to share also make them easy to save, load, and compare. See <doc:Machine-Learning-Primer> for the full picture of how Quiver's models are designed.

### Training inside a task

The most common pattern is training a model inside an `async` function. The function describes the steps — fit a scaler, transform the features, fit the model — and Swift Concurrency handles running them. When the function returns, the caller receives a fitted model as its result.

```swift
import Quiver

func trainClassifier(features: [[Double]], labels: [Int]) async -> KNearestNeighbors {
    // Fit a scaler and transform the training data
    let scaler = FeatureScaler.fit(features: features)
    let scaled = scaler.transform(features)

    // Fit the model on the scaled features
    let model = KNearestNeighbors.fit(features: scaled, labels: labels, k: 3)
    return model
}
```

This is the default pattern for training in response to a user action, a file load, or a network response. The caller marks the call site with `await`, and the work runs without blocking the caller's thread.

```swift
let model = await trainClassifier(features: data, labels: labels)
```

Some Quiver models can throw when their inputs don't support a solution. `LinearRegression.fit()` throws `MatrixError.singular` when the normal equation has no unique answer — which happens when the feature columns are linearly dependent, meaning one feature is an exact combination of the others. Combining a throwing fit with an `async` function is straightforward: mark the wrapper `async throws` and use `try await` at the call site.

```swift
import Quiver

func trainRegression(features: [[Double]], targets: [Double]) async throws -> LinearRegression {
    return try LinearRegression.fit(features: features, targets: targets)
}

// Caller side
let model = try await trainRegression(features: sqft, targets: prices)
```

> Tip: If training pairs a scaler with a model, keep them together so they stay matched at prediction time. For bundling them into a single persistable value, see <doc:Pipeline>.

### Long-running training

Some training workloads are the work itself. A `KMeans` fit over a large dataset, or a clustering run with a high iteration count, can take long enough that we want the work to run independently of the calling context. `Task.detached` starts a new top-level task that runs on its own — the right choice when training should complete regardless of what the surrounding code is doing.

```swift
import Quiver

func trainClusters(from data: [[Double]]) async -> KMeans {
    // Run training as a detached task so it proceeds independently
    return await Task.detached {
        KMeans.fit(data: data, k: 5, seed: 42)
    }.value
}
```

The closure runs as a standalone task. When it finishes, `.value` returns the fitted `KMeans` model, and Swift Concurrency hands that value back to the caller. Because `KMeans` is a `Sendable` value, nothing in the crossing needs special handling.

> Tip: Once a long-running fit is complete, save the model with `JSONEncoder` so the next session can skip training entirely. See <doc:Model-Persistence> for the full pattern.

### Updating SwiftUI when training completes

SwiftUI's `@Observable` macro gives a view model observable properties that trigger view updates when they change. Combined with `@MainActor`, it makes the flow from background training to visible result straightforward: a view model kicks off training inside a task, and the assignment back to the model property happens on the main thread automatically.

```swift
import Quiver
import SwiftUI

@Observable
@MainActor
final class WorkoutAnalysisViewModel {
    var model: KMeans?
    var isTraining = false

    func train(from sessions: [[Double]]) async {
        isTraining = true

        // Fit in a detached task so the view stays responsive
        let fitted = await Task.detached {
            KMeans.fit(data: sessions, k: 3, seed: 42)
        }.value

        // Assignment runs on the main actor — the view updates automatically
        model = fitted
        isTraining = false
    }
}
```

`@MainActor` is Swift's way of saying "everything in this class runs on the main thread." When `Task.detached` suspends the function, the actual training runs elsewhere. When the task finishes and execution returns to the `await` line, Swift puts the function back on the main thread before the next statement. The assignment to `model` happens there, and any SwiftUI view observing the view model updates without extra work.

> Tip: The same pattern works for `UIKit` view controllers marked `@MainActor`. The `await` marks the boundary, the training runs off the main thread, and the fitted model arrives back on the main thread as a `Sendable` value.

