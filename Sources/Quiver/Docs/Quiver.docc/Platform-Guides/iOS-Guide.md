# iOS Guide

Presenting statistics, summaries, and derived insights inside iOS apps with Quiver.

## Overview

Most iOS apps already have the data they need to feel intelligent. A fitness app has a year of workouts. A finance app has a month of transactions. A reading app knows what the user has read. The hard part is turning that history into something the screen can show — a trend, a typical range, a flagged outlier, a personalized recommendation.

Quiver is the layer between the data the app already holds and the insights it wants to show. It runs inside the app, no network call, no service. With Quiver an app can summarize what is in a list, highlight the unusual entries, plot a trend that fits the data, group history by category for a chart, and recommend the next item a user is likely to want — all from arrays the app already has.

### Loading models when the app launches

The shape we recommend for an iOS app is straightforward: train the model elsewhere, load it when the app launches, and let SwiftUI views read predictions from it. Training might happen in a notebook, a command-line tool, a scheduled job, or a previous app session — anywhere except the main thread of a view that the user is waiting on. The result is encoded to JSON and shipped in the app bundle or downloaded into `Documents/`. At launch, the app decodes it once and reads from the decoded value for the rest of the session.

```swift
import Foundation
import Quiver

// Decodes a fitted model from JSON shipped in the app bundle
func loadRecommender() -> KNearestNeighbors? {
    guard let url = Bundle.main.url(forResource: "recommender", withExtension: "json"),
          let data = try? Data(contentsOf: url) else { return nil }
    return try? JSONDecoder().decode(KNearestNeighbors.self, from: data)
}
```

The model is loaded once and held wherever the app keeps app-launch state — an `@Observable` store, a SwiftUI environment object, or a singleton. SwiftUI re-renders cheaply against the same fitted value, so reading predictions on every view update is fine. For the saving side of this pattern, including writing a fitted model out at the end of a session, see <doc:Model-Persistence>.

### Practical realities of sensors and storage on iOS

iOS has more sources of data than watchOS and more places to put that data than Vapor. A few practical realities are worth knowing before building a screen on top of a Quiver model.

**When a screen appears and disappears.** Models load once when the app launches, or the first time the screen that needs them appears, and then live in observable state for the rest of the session. SwiftUI re-renders cheaply against the same fitted value, so reading predictions on every view update is fine. Refitting on every view update is not — it is a common cause of stutter that goes away the moment the fit moves into a `.task` modifier or a settings action the user triggers.

**Storage across launches.** Pre-fitted models that ship with the app live in the bundle as read-only JSON, decoded with `JSONDecoder` at launch. User-derived state — a personal baseline, a fitted personalization model, a precomputed collection of item vectors — lives in `Documents/`, written through `JSONEncoder` at the end of a session and read at the start of the next one. The two locations correspond to a single distinction. Models that are the same for every user live in the bundle. Models specific to this user live in `Documents/`.

**How much can fit in memory.** The iPhone has substantially more memory than the Watch and noticeably less than a server. A fitted Quiver model with ten thousand training samples and a dozen features is comfortable. A precomputed collection of a few thousand item vectors at moderate dimensionality fits without strain. Half a million samples held in memory is a different problem and calls for downsampling, paging, or moving the work to a background process.

> Tip: A view that refits the model in `onAppear` will refit every time the user navigates back to it. Move the fit to a `.task` that captures a stable identifier, or to a settings action the user explicitly triggers, so the cost is paid once per session rather than once per appearance.

### Working with sensor inputs and user data

iOS apps have many places to read from but Quiver does not interact with any of these sources directly. The pattern is to decode whatever the source produces into `[Double]` once, near the boundary, and let everything downstream operate on the same plain arrays Quiver computes on.

```swift
import Quiver

// Once measurements are in a [Double], a Panel turns them into a typed summary
func summary(of measurements: [Double]) -> PanelSummary? {
    let panel = Panel([("measurements", measurements)])
    return panel.summary()
}
```

`Panel.summary()` returns a `PanelSummary` with count, mean, standard deviation, quartiles, min, max, and IQR — every field a dashboard typically needs in one typed value. The result is `Codable` and `Sendable`, so it crosses task boundaries and persists to disk without ceremony.

> Experiment: [quiver-demo-ios](https://github.com/waynewbishop/quiver-demo-ios) is a personal-finance dashboard that pairs three Quiver aggregations with three Swift Charts views — a donut from `groupedData(.percentage)`, a weekly bar from `downsample(.sum)`, and a scatter with outliers flagged by `outlierMask`. Opening the project and changing the spending data in `FinanceModel.swift` shows the same chart shapes adapting to whatever the app already holds.
