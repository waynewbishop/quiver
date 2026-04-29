# iOS Patterns

Personalizing classifiers, surfacing anomalies, and ranking recommendations on the device.

## Overview

The <doc:iOS-Guide> covers the foundations — loading models at launch, how a screen's life on display affects the work, and decoding sensor and storage inputs into `[Double]`. This article builds on those foundations with applied patterns iOS developers reach for repeatedly: a personalized classifier fit from a single user's history, anomaly surfacing in a list view, similarity-based recommendation against a precomputed item index, and data aggregation that feeds a Swift Charts visualization.

### Personalization in a single-user app

iOS apps usually have one user across many sessions. Personalization on iOS therefore happens between sessions rather than during them — a model fit from a user's accumulated history and held in observable state across launches. The pattern is to load the user's data from `Documents/`, fit a `Pipeline` that pairs a scaler with a classifier, and save the fitted pipeline back to disk so the next launch is ready to predict immediately.

```swift
import SwiftUI
import Quiver

@Observable
@MainActor
final class PersonalBaseline {
    var pipeline: Pipeline<KNearestNeighbors>?

    private var fileURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("baseline.json")
    }

    // Fits a personalized classifier from the user's stored history
    func refit(features: [[Double]], labels: [Int]) throws {
        let fitted = Pipeline.fit(features: features, labels: labels, k: 3)
        let data = try JSONEncoder().encode(fitted)
        try data.write(to: fileURL)
        pipeline = fitted
    }

    // Restores the most recent pipeline from disk on app launch
    func load() {
        guard let data = try? Data(contentsOf: fileURL) else { return }
        pipeline = try? JSONDecoder().decode(Pipeline<KNearestNeighbors>.self, from: data)
    }
}
```

The `Pipeline` keeps the scaler and the classifier as one matched pair, so encoding and decoding move them together. Refitting does not happen during view updates — it is triggered by a settings action when the user wants to update their baseline, or scheduled at the end of a session. Predictions read from `pipeline` and run cheaply on every view update. For the saving side of this pattern, see <doc:Model-Persistence>; for why the pairing matters, see <doc:Pipeline>.

What this pattern is and what it is not: it is a per-user personalization fit, saved across launches, refit on demand. It is not a model that keeps learning from every new piece of data as it arrives. The user explicitly asks for a refit, or the app does it at the start or end of a session — never while SwiftUI is drawing a view.

### Surfacing anomalies in a list view

Many iOS apps display a list or chart of measurements where the interesting values are the unusual ones. A spending feed wants to highlight the outsized transaction; a health dashboard wants to flag the abnormally high reading; a logging app wants to mark the request that took ten times longer than the rest. Quiver's `outlierMask(threshold:)` produces a boolean array aligned with the values, and SwiftUI renders against that mask the same way it renders against any other state.

```swift
import SwiftUI
import Quiver

struct Reading: Identifiable {
    let id = UUID()
    let label: String
    let value: Double
}

struct ReadingsList: View {
    let readings: [Reading]

    // Flag any reading more than two standard deviations from the mean
    private var anomalyMask: [Bool] {
        readings.map(\.value).outlierMask(threshold: 2.0)
    }

    var body: some View {
        List(Array(zip(readings, anomalyMask)), id: \.0.id) { reading, isAnomaly in
            HStack {
                Text(reading.label)
                Spacer()
                Text(reading.value, format: .number.precision(.fractionLength(1)))
                if isAnomaly {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                }
            }
        }
    }
}
```

The mask is computed once per render against the current list of values. Because Swift's `Array` is copy-on-write, mapping out the values and computing the mask does not allocate the underlying storage twice. As the list grows, the mask grows with it; as the user filters or searches, the mask recomputes against the visible subset.

The shape here is a one-shot z-score check against the dataset currently in view, not a moving-window detector running on a live stream — that is the watchOS shape, where the window slides forward continuously. On iOS the dataset is whatever the screen is currently showing, and the mask is recomputed when that dataset changes.

### Recommendation and similarity routes

Recommendation on iOS rarely needs a server when the catalog is small enough to fit on the device. The pattern is to embed every catalog item as a feature vector once — at app launch from a bundled file, or when the catalog updates from a download — and rank against the precomputed vectors at view time. Quiver's `cosineSimilarities(to:)` and `topIndices(k:labels:)` do the ranking; SwiftUI renders the result.

```swift
import SwiftUI
import Quiver

struct Item: Identifiable {
    let id: Int
    let name: String
    let vector: [Double]
}

@Observable
@MainActor
final class Catalog {
    var items: [Item] = []

    // Returns the top-k items most similar to the given query vector
    func recommend(matching query: [Double], k: Int = 5) -> [(rank: Int, item: Item, score: Double)] {
        let vectors = items.map(\.vector)
        let similarities = vectors.cosineSimilarities(to: query)
        return similarities.topIndices(k: k, labels: items).map {
            (rank: $0.rank, item: $0.label, score: $0.score)
        }
    }
}

struct RecommendationsView: View {
    let catalog: Catalog
    let queryVector: [Double]

    var body: some View {
        List(catalog.recommend(matching: queryVector), id: \.item.id) { result in
            HStack {
                Text("\(result.rank). \(result.item.name)")
                Spacer()
                Text(result.score, format: .number.precision(.fractionLength(2)))
            }
        }
    }
}
```

The full embedding pipeline — tokenizing text, building vectors, reducing with `meanVector` — is covered in <doc:Semantic-Search>. What changes on iOS is where each piece runs: embedding happens when the catalog loads, the precomputed vectors live in observable state, and only the ranking call runs while a view is on screen. For the underlying operations, see <doc:Similarity-Operations>.

What this pattern is and what it is not: it is a top-k similarity ranking against a precomputed index. It is not a learned recommender that personalizes from interaction history — that pattern combines this index with the personalization shape from earlier in the article, where a per-user model produces the query vector instead of a fixed input.

### Aggregating for charts

iOS is where Quiver-aggregated data most often meets [Swift Charts](https://developer.apple.com/documentation/charts). The chart screen receives raw measurements, runs them through a `Panel` for column-aware summarization, and feeds the result to the chart view. The intermediate `Panel` keeps column identity, supports per-column statistics, and converts cleanly to whatever shape Swift Charts expects.

```swift
import SwiftUI
import Charts
import Quiver

struct DistributionChart: View {
    let temperatures: [Double]

    // Bucket the values into a histogram with Quiver
    private var bins: [(midpoint: Double, count: Int)] {
        temperatures.histogram(bins: 12)
    }

    var body: some View {
        Chart(bins, id: \.midpoint) { bin in
            BarMark(
                x: .value("Temperature", bin.midpoint),
                y: .value("Count", bin.count)
            )
        }
    }
}
```

The histogram is a direct array transform. For richer summaries — five-number summary, quartiles, standard deviation — the same pattern applies: the chart view reads precomputed Quiver results from observable state. For the Panel-to-Charts pipeline end to end, see <doc:Panel-and-Charts>; for the broader catalog of summary methods and chart shapes, see <doc:Data-Visualization> and <doc:Panel>.

### Sizing a model for the phone

A useful baseline for iOS sizing is the watchOS measurement set in <doc:watchOS-Patterns> under "Sizing a model for the wrist." If a model fits and predicts inside a few milliseconds on Apple Watch silicon, it does the same on iPhone silicon with room to spare. The interesting questions on iOS are not raw fit speed but when the work happens.

The first question is whether to load a saved model or to fit one at launch. Decoding a fitted `Pipeline` from JSON is faster than refitting it, and trades a CPU-heavy operation for a disk read. For models that do not need to adapt to the current user — a recommendation index, a content classifier — loading is preferred. For models that do need to be user-specific, the fit happens once at the end of a session or when the user explicitly opts in, never as a side effect of a view appearing.

The second question is doing work off the main thread under SwiftUI. A fitted Quiver model is `Sendable`, so a `.task` modifier on a view can call `predict(...)` on a background priority without blocking the render loop. The result crosses back to the main actor as a value and the SwiftUI binding updates the view automatically. For the full set of concurrency patterns, see <doc:Concurrency-Primer>.

> Tip: For models that ship in the bundle as JSON, the decode happens once the first time the model is used. Hold the fitted value in an `@Observable` store and let every view read from that store rather than decoding inside `onAppear` — decoding twice is a silent regression that only shows up under profiling.

