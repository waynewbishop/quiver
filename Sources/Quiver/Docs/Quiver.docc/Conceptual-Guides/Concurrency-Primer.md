# Concurrency Primer

Training models off the main thread and passing fitted results back to the interface.

## Overview

Swift Concurrency is the language's built-in way to run work without blocking the main thread. Tasks describe units of work, `async` functions suspend and resume, and the compiler verifies that values crossing between tasks are safe to share. Quiver fits into this model naturally because its models are value types, each defined as an immutable Swift `struct`. A fitted model can be trained inside a task, returned from it, and handed to a view without locks, wrappers, or `@unchecked` annotations.

This primer covers the core patterns: training a model inside a task, keeping long-running fits off the main thread, and updating a SwiftUI view when training completes.

> Note: For core vocabulary like `features`, `labels`, `fit`, and `predict`, start with <doc:Machine-Learning-Primer> before working through the concurrency patterns below.

### What Sendable means for Quiver

A type that is safe to move between tasks has no shared mutable state to protect. The Swift compiler verifies this property through a protocol called `Sendable`. When the compiler sees a `Sendable` value crossing a task boundary, it knows the value can be handed off without coordination. Every public type in Quiver conforms to `Sendable`, which is why a fitted model can flow cleanly from a background task to the caller without any extra ceremony.

```swift
import Quiver

// A fitted model is a value. Pass it between tasks freely.
let model = try LinearRegression.fit(features: sqft, targets: prices)

Task {
    // The model crosses into the task as a value — the compiler verifies it's safe
    let prediction = model.predict(2000.0)
    print(prediction)
}
```

The reason this works comes back to how Quiver builds its models. Every property is a plain value (`[Double]` coefficients, `Int` dimensions, stored statistics), and every model is immutable once `fit` returns. There's nothing to mutate, nothing to share, and nothing that could change on one thread while another thread is reading it.

The same property applies to `predict`. A fitted model can be called from any number of concurrent tasks without serialization, because prediction is a pure read against immutable properties: the model computes an answer from its stored parameters and returns, without touching any shared state along the way.

> Note: Because models are also `Codable` and `Equatable`, the same value semantics that make them safe to share also make them easy to save, load, and compare. See <doc:Machine-Learning-Primer> for the full picture of how Quiver's models are designed.

### Training inside a task

The most common pattern is training a model inside an `async` function. The function describes the steps (fit a scaler, transform the features, fit the model), and Swift Concurrency handles running them. When the function returns, the caller receives a fitted model as its result.

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

Some Quiver models can throw when their inputs don't support a solution. `LinearRegression.fit()` throws ``MatrixError/singular`` when the normal equation has no unique answer, which happens when the feature columns are linearly dependent, meaning one feature is an exact combination of the others. Combining a throwing fit with an `async` function is straightforward: mark the wrapper `async throws` and use `try await` at the call site.

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

Some training workloads are the work itself. A ``KMeans`` fit over a large dataset, or a clustering run with a high iteration count, can take long enough that we want the work to run independently of the calling context. `Task.detached` starts a new top-level task that runs on its own: the right choice when training should complete regardless of what the surrounding code is doing.

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

### Dividing work across cores

The patterns above move work off the main thread so the interface stays responsive. A different goal is to make a single large computation finish faster by running parts of it on several cores at once. A fitted model already gives us the natural seam: predicting one batch of inputs never depends on predicting another, so we can hand each batch to `predict` on its own core. Because the model is a `Sendable` value, every task shares the same fitted model without coordination.

The same task group from earlier does the work. Each task calls `predict` on one batch and returns the result.

```swift
import Quiver

func predict(_ batches: [[[Double]]], with model: LinearRegression) async -> [[Double]] {
    await withTaskGroup(of: (Int, [Double]).self) { group in
        // Each task predicts one batch — an independent call to the same model
        for (index, batch) in batches.enumerated() {
            group.addTask { (index, model.predict(batch)) }
        }

        // Collect the finished predictions in their original order
        var results = [[Double]](repeating: [], count: batches.count)
        for await (index, predictions) in group {
            results[index] = predictions
        }
        return results
    }
}
```

Each `predict` runs the same Quiver call we would make sequentially; the task group only decides which core runs which batch. The same shape applies to any independent work, such as a pairwise comparison like `clusterCohesion` or a batch handed to `KMeans.predict`.

> Note: Running batches concurrently changes the order in which they finish, even though each is placed back in its original position. For most operations the recombined result is identical to the sequential one. Voting models such as ``KNearestNeighbors`` are the exception: when two classes tie within a neighborhood, the order in which votes are counted can settle the tie either way, so a concurrent split may differ from a single call on a few predictions. When exact reproducibility matters more than speed, predict in one call.

### What stays sequential

Not every operation divides this way, and recognizing which ones do not is as useful as knowing which ones do. Some computations are a chain in which each step consumes the result of the step before it. Matrix inversion and the determinant both work by Gaussian elimination, where every pivot transforms the matrix that the next pivot depends on — there is no way to run a later step before an earlier one finishes. Iterative model fitting has the same shape: each `KMeans` iteration places its centroids based on the assignment from the previous iteration, so the iterations cannot overlap.

These operations gain nothing from `concurrentPerform`, because there are no independent pieces to hand out. What they can still do is run off the main thread using the task patterns shown earlier, so a long inversion or a high-iteration fit proceeds without freezing the interface. The work itself stays sequential; only its relationship to the main thread changes.

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

### From off the main thread to into an app

The patterns here all rest on one property: a fitted Quiver model is an immutable, `Sendable` value, so it crosses task and actor boundaries without a copy ceremony or a lock. Splitting a batch, fitting off the main thread, and handing the result back to a view are three uses of that single guarantee. The <doc:Machine-Learning-Primer> covers the models these patterns wrap, and <doc:Pipeline> shows how scaling and fitting compose into one `Sendable` unit that moves across threads as cleanly as a single model does.

> Experiment: **The Quiver Notebook** is a quick place to watch these patterns run before wiring them into an app. Try launching two fits with `async let`, printing on entry and exit, and watching the output interleave from run to run. That ordering is the visible proof the work ran concurrently. The Notebook has no view to update, so the SwiftUI hand-off above belongs in an app. See <doc:Quiver-Notebook>.

