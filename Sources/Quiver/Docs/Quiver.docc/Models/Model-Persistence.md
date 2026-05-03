# Model Persistence

Save trained models to disk without retraining.

## Overview

Quiver's ML models conform to Swift's `Codable` protocol. This means a model trained once can be encoded to JSON, saved to disk, transmitted over a network, or stored in a database — then decoded back into an identical, ready-to-use model. No retraining, no reconfiguration, no data required.

This works because Quiver models are immutable value types whose stored properties are all basic Swift types — arrays of numbers, integers, and booleans. The Swift compiler auto-synthesizes the encoding and decoding logic, so there is no custom serialization code to maintain or debug.

### The pattern

Every model follows the same three-step pattern: `fit`, `encode`, `decode`. The decoded model is identical to the original — same properties, same predictions, same `Equatable` comparison:

```swift
import Quiver
import Foundation

// Train the model
let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
let targets = [2.1, 3.9, 6.1, 8.0, 9.8]
let model = try LinearRegression.fit(features: features, targets: targets)

// Convert the model to JSON bytes held in memory
let data = try JSONEncoder().encode(model)

// Convert the JSON bytes back into a model
let restored = try JSONDecoder().decode(LinearRegression.self, from: data)

print(model == restored)  // true
print(restored.predict([[6.0]]))
```

The same pattern works for every model type — `LinearRegression`, `GaussianNaiveBayes`, `KNearestNeighbors`, `KMeans`, and `FeatureScaler`.

### Saving and loading from disk

To persist a model between app launches, write the encoded data to a file and read it back on the next launch:

```swift
import Quiver
import Foundation

// Train the model
let model = KMeans.fit(data: sensorData, k: 3, seed: 42)

// Convert to JSON bytes, then write those bytes to a file
let encoded = try JSONEncoder().encode(model)
try encoded.write(to: modelURL)

// On next launch, read the file back into bytes
let saved = try Data(contentsOf: modelURL)

// Convert the bytes back into a trained model
let restored = try JSONDecoder().decode(KMeans.self, from: saved)
let labels = restored.predict(newReadings)
```

### When to persist the scaler

Distance-based models like `KNearestNeighbors` and `KMeans` measure how far apart data points are. When features have different scales — a credit score ranging 300-850 and an account balance ranging 0-250,000 — the larger feature dominates every distance calculation. Feature scaling normalizes all columns to the same range so each one contributes equally.

When scaling is used, the scaler and model become a matched pair. The model's learned distances and boundaries exist in the scaled coordinate space, so every future input must be scaled using the same min and max values from training. Losing the scaler means new inputs land in a different coordinate space, producing incorrect predictions with no error or warning.

> Tip: `LinearRegression` and `GaussianNaiveBayes` do not require scaling — regression coefficients compensate for different magnitudes mathematically, and Naive Bayes evaluates each feature independently. For these models, the scaler is optional and the model can be persisted on its own.

### Persisting a full pipeline

When scaling is part of the workflow, `Pipeline` bundles the scaler and model into a single `Codable` value. One encode, one decode — they always travel together:

```swift
import Quiver
import Foundation

// Train and bundle
let scaler = FeatureScaler.fit(features: trainingData)
let model = KNearestNeighbors.fit(
    features: scaler.transform(trainingData), labels: labels, k: 5
)
let pipeline = Pipeline(scaler: scaler, model: model)

// Save the entire pipeline as one JSON blob
let data = try JSONEncoder().encode(pipeline)
try data.write(to: pipelineURL)

// On next launch, decode and predict immediately
let saved = try Data(contentsOf: pipelineURL)
let restored = try JSONDecoder().decode(
    Pipeline<KNearestNeighbors>.self, from: saved
)
let predictions = restored.predict(newData)
```

`Pipeline` scales inputs automatically at prediction time, so the caller never touches the scaler directly. See <doc:Pipeline> for the full API.

### Verifying round-trip fidelity

Since all models conform to `Equatable`, verifying that encoding and decoding preserved the model exactly is a single expression:

```swift
let model = GaussianNaiveBayes.fit(features: features, labels: labels)
let json = try JSONEncoder().encode(model)
let decoded = try JSONDecoder().decode(GaussianNaiveBayes.self, from: json)

assert(model == decoded)
```

### Loading models in async contexts

The same value semantics that make Quiver models `Codable` also make them `Sendable`. A model decoded inside an `async` function can be returned to the caller, passed across task boundaries, or handed to a SwiftUI view without any additional ceremony — the decoded value crosses freely because there's nothing shared or mutable inside it.

```swift
import Quiver
import Foundation

func loadClassifier(from url: URL) async throws -> KNearestNeighbors {
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(KNearestNeighbors.self, from: data)
}

// Call from anywhere — main actor, background task, detached task
let classifier = try await loadClassifier(from: modelURL)
```

This is the natural pattern for apps that load a pre-trained model at startup: decode it off the main thread, hand the result to the view layer, and start predicting. The fitted model behaves the same whether it was just trained or just decoded — immutable, thread-safe, and ready to use.

> Tip: For the full set of concurrency patterns — training inside a task, long-running fits, and SwiftUI integration — see <doc:Concurrency-Primer>.

### Where to store the encoded bytes

Once a model is encoded, the resulting `Data` value can go anywhere Swift can write bytes. On iOS and macOS, write to the app's Application Support directory with `FileManager`, store in `UserDefaults` for small models, or persist as a `Data` property in SwiftData. On watchOS, save to the local documents directory for on-device models, or use `WatchConnectivity` to transfer encoded bytes from a paired iPhone. On server-side Swift with Vapor, write to a file path at deployment time and decode once at startup — the model stays in memory to serve concurrent requests.

### Related
- <doc:Pipeline>
- <doc:Machine-Learning-Primer>
- <doc:Linear-Regression>
- <doc:Naive-Bayes>
- <doc:Nearest-Neighbors-Classification>
- <doc:KMeans-Clustering>
- <doc:Feature-Scaling>
