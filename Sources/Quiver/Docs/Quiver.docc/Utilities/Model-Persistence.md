# Model Persistence

Save trained models to disk without retraining.

## Overview

Quiver's ML models conform to Swift's `Codable` protocol. This means a model trained once can be encoded to JSON, saved to disk, transmitted over a network, or stored in a database — then decoded back into an identical, ready-to-use model. No retraining, no reconfiguration, no data required.

This works because Quiver models are immutable value types whose stored properties are all basic Swift types — arrays of numbers, integers, and booleans. The Swift compiler auto-synthesizes the encoding and decoding logic, so there is no custom serialization code to maintain or debug.

### The pattern

Every model follows the same three-step pattern: `train`, `encode`, `decode`. The decoded model is identical to the original — same properties, same predictions, same `Equatable` comparison:

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

### Persisting a full pipeline

In a typical ML workflow, the scaler and the model must stay paired — applying a different scaler to test data than was used during training produces incorrect predictions. Both conform to `Codable`, so they can be saved together:

```swift
import Quiver
import Foundation

// Train the pipeline — scaler and model together
let scaler = FeatureScaler.fit(features: trainingData)
let scaledTrain = scaler.transform(trainingData)
let model = KNearestNeighbors.fit(
    features: scaledTrain, labels: labels, k: 5
)

// Convert each to JSON bytes and write to separate files
let scalerData = try JSONEncoder().encode(scaler)
let modelData = try JSONEncoder().encode(model)
try scalerData.write(to: scalerURL)
try modelData.write(to: modelURL)

// Read files back into bytes, then convert to trained objects
let loadedScaler = try JSONDecoder().decode(
    FeatureScaler.self, from: Data(contentsOf: scalerURL)
)
let loadedModel = try JSONDecoder().decode(
    KNearestNeighbors.self, from: Data(contentsOf: modelURL)
)
let prediction = loadedModel.predict(loadedScaler.transform(newData))
```

### Verifying round-trip fidelity

Since all models conform to `Equatable`, verifying that encoding and decoding preserved the model exactly is a single expression:

```swift
let model = GaussianNaiveBayes.fit(features: features, labels: labels)
let json = try JSONEncoder().encode(model)
let decoded = try JSONDecoder().decode(GaussianNaiveBayes.self, from: json)

assert(model == decoded)
```

### Where to store the encoded bytes

Once a model is encoded, the resulting `Data` value can go anywhere Swift can write bytes. On iOS and macOS, write to the app's Application Support directory with `FileManager`, store in `UserDefaults` for small models, or persist as a `Data` property in SwiftData. On watchOS, save to the local documents directory for on-device models, or use `WatchConnectivity` to transfer encoded bytes from a paired iPhone. On server-side Swift with Vapor, write to a file path at deployment time and decode once at startup — the model stays in memory to serve concurrent requests.

### Related
- <doc:Machine-Learning-Primer>
- <doc:Linear-Regression>
- <doc:Naive-Bayes>
- <doc:Nearest-Neighbors-Classification>
- <doc:KMeans-Clustering>
- <doc:Feature-Scaling>
