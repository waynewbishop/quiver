# Explore

@Metadata {
  @TitleHeading("Essentials")
}

Explore what becomes possible with Quiver across Apple platforms.

## Overview

This is a starting point for developers who want to understand what's possible with Quiver before diving into the API.

### iOS — the primary deployment target

iOS is where the majority of Swift developers ship code, and where Quiver's capabilities integrate most naturally with existing app architectures. A common pattern is preparing data for Swift Charts — computing the statistics, trends, and distributions that charts display.

```swift
import Quiver

// Median home prices by month in your neighborhood
let prices = [485_000.0, 492_000.0, 510_000.0, 525_000.0, 540_000.0, 568_000.0]

// Smooth the trend for a line chart
let trend = prices.rollingMean(window: 3)

// Month-over-month price changes for a bar chart
let changes = prices.percentChange(lag: 1)

// Summary statistics for a dashboard
let avg = prices.mean()       // average price
let spread = prices.std()     // price variability
let mid = prices.median()     // middle value
```

**What iOS enables:**

- Smooth noisy data with `rollingMean()` for trend lines in Swift Charts
- Calculate growth rates with `percentChange()` for bar and area charts
- Compute statistics with `mean()`, `std()`, and `median()` for dashboard summaries
- Build distributions with `histogram()` for frequency charts
- Train models on local data with `LinearRegression`, `KNearestNeighbors`, `GaussianNaiveBayes`, or `KMeans`

> Tip: Quiver computes the data. Swift Charts renders it. See <doc:Statistical-Operations> for the full set of statistical operations and <doc:Data-Visualization> for charting patterns.

### watchOS — live, on-device intelligence

watchOS is the platform where Quiver's zero-dependency design matters most. During an active workout session, the watch has continuous access to heart rate, cadence, pace, and motion data — and Quiver can process that stream in real time, updating a model incrementally as each new sample arrives. There is no network round-trip, no companion app required, and no model file to bundle or keep in sync. The intelligence lives entirely on the wrist.

Consider a user on a run. As heart rate samples stream in every few seconds, Quiver can maintain a rolling window of feature vectors and re-cluster them continuously, giving the app a live picture of which intensity zone the user is currently in — one that adapts to that user's personal physiology rather than fixed generic thresholds.

```swift
import Quiver

// Rolling window of biometric samples — [heartRate, cadence, elapsedMinutes]
let samples: [[Double]] = [
    [142.0, 82.0, 1.0], [148.0, 84.0, 2.0], [155.0, 86.0, 3.0],
    [160.0, 88.0, 4.0], [158.0, 85.0, 5.0], [145.0, 80.0, 6.0]
]

// Scale features and cluster into intensity zones
let scaler = FeatureScaler.fit(features: samples)
let scaled = scaler.transform(samples)
let model  = KMeans.fit(data: scaled, k: 3, seed: 42)

// Each label is a detected zone — adapted to this user's physiology
print(model.labels)
```

**What watchOS enables:**

- Classify workout intensity zones in real time using the user's own live biometric stream — no predefined thresholds
- Detect pace or heart rate anomalies during a run by measuring distance from the current rolling centroid
- Fit a linear regression trend on elapsed heart rate data mid-workout to anticipate fatigue before it peaks
- Detect class imbalance in accumulated session labels with `imbalanceRatio()` and oversample before re-fitting — no server needed
- Build a personal performance baseline that refines itself across every session, entirely on the watch

> Tip: Quiver's models are fast enough to re-fit on every incoming sensor sample. K-Means on a 20-sample window completes in milliseconds on Apple Watch hardware, making continuous in-session updates practical. Feature scaling is essential when combining heart rate (40–200 bpm) with elapsed time (minutes) — see <doc:Feature-Scaling>.

### Swift server and Vapor

Quiver runs on Linux and integrates naturally with Vapor, enabling a class of server-side Swift applications that handle the full numerical layer of an ML pipeline in Swift. Vapor handles routing and request lifecycle. Quiver handles everything that happens to the data — embedding lookup, similarity ranking, clustering, and duplicate detection — in the same Swift process.

```swift
import Vapor
import Quiver

// Word embeddings — loaded from any source (GloVe, Word2Vec, custom)
let embeddings: [String: [Double]] = loadEmbeddings()

// Convert a search query to a vector
let query = "comfortable running shoes"
let tokens  = query.tokenize()                  // ["comfortable", "running", "shoes"]
let vectors = tokens.embed(using: embeddings)   // look up each word's vector
let queryVec = vectors.meanVector()             // average into a single vector

// Rank stored documents by similarity to the query
guard let queryVec else { return }
let scores  = docVectors.cosineSimilarities(to: queryVec)
let results = scores.topIndices(k: 5, labels: docTitles)
```

**What Swift server and Vapor enable:**

- A semantic search endpoint that tokenizes queries, looks up embeddings, and ranks results
- A clustering microservice that accepts feature vectors and returns K-Means assignments on demand
- A duplicate detection pipeline that identifies near-identical content before it reaches persistent storage
- A vector search API built entirely in Swift

> Tip: The `tokenize()` → `embed(using:)` → `meanVector()` pipeline converts raw text into a single vector in three chained calls. See <doc:Semantic-Search> for the full walkthrough and <doc:Similarity-Operations> for batch comparison patterns.

### Swift Playgrounds — interactive learning

Xcode 26 introduces the `#Playground` macro — an interactive environment that works directly with SPM packages. This makes Quiver a great option for students and developers exploring numerical computing interactively.

Quiver can generate realistic test data, split it for training, and fit a model — all interactively. This example uses linear regression to predict weight from height, a classic first ML exercise:

```swift
import Playgrounds
import Quiver

#Playground {
    // Generate 100 people with random height and weight
    let height = [Double].randomNormal(100, mean: 170.0, std: 10.0)
    let weight = [Double].randomNormal(100, mean: 75.0, std: 12.0)

    // Split into training (80%) and testing (20%)
    let (trainH, testH) = height.trainTestSplit(testRatio: 0.2, seed: 42)
    let (trainW, testW) = weight.trainTestSplit(testRatio: 0.2, seed: 42)

    // Train: learn the relationship between height and weight
    let model = try LinearRegression.fit(features: trainH, targets: trainW)

    // Predict weight for the test heights
    let predictions = model.predict(testH)

    // How accurate? R² of 1.0 = perfect, 0.0 = no relationship
    let r2 = predictions.rSquared(actual: testW)
    print("R² = \(r2)")
}
```

**What Swift Playgrounds enables:**

- Generate test datasets with built-in random distributions
- Explore Quiver's full API interactively — statistics, linear algebra, similarity, and ML models including Linear Regression, K-Nearest Neighbors, K-Means Clustering, and Gaussian Naive Bayes
- Complete course assignments using `#Playground` with `import Quiver`
- Visualize computed results with Swift Charts in the same project
