# iOS Guide

Presenting statistics, summaries, and derived insights inside iOS apps with Quiver.

## Overview

Many iOS apps spend most of their screen time presenting derived data. A travel app shows a year of flights as totals per airport and a trend across months. A health dashboard summarizes the last thirty days of readings into a mean, a standard deviation, and a list of the unusual days. A spending feed flags the transactions that fall outside the user's typical range. A reading app shows the running pace of a habit. The data already exists — in the app's storage, in `HKHealthStore`, in a CSV the user imported — and the work is in the layer between that data and the screen.

Quiver fits this reality by running inside the app's own Swift process and computing those derived numbers from `[Double]` arrays the app already holds. Aggregations like `mean()`, `std()`, `quartiles()`, and `outlierMask()` produce the values a dashboard surfaces. A `Panel` of named columns groups personal history by month, category, or label. A fitted `LinearRegression` becomes a trend line in a Swift Chart. A fitted `KNearestNeighbors` becomes a personalized recommendation index. Every Quiver model is `Codable` and `Sendable`, so app-launch loading, persistence to `Documents/`, and SwiftUI's data flow fall out naturally.

### Loading models when the app launches

The shape we recommend for an iOS app is straightforward: train the model elsewhere, load it when the app launches, and let SwiftUI views read predictions from it. Training might happen in a notebook, a command-line tool, a scheduled job, or a previous app session — anywhere except the main thread of a view that the user is waiting on. The result is encoded to JSON and shipped in the app bundle or downloaded into `Documents/`. At launch, the app decodes it into an `@Observable` holder, and any SwiftUI view reads from that holder for the rest of the session.

```swift
import SwiftUI
import Quiver

@Observable
@MainActor
final class RecommenderStore {
    var model: KNearestNeighbors?

    // Loads the fitted model from the app bundle on first reference
    func load() {
        guard let url = Bundle.main.url(forResource: "recommender", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return }
        model = try? JSONDecoder().decode(KNearestNeighbors.self, from: data)
    }
}

struct RecommenderView: View {
    @State private var store = RecommenderStore()

    var body: some View {
        List { /* render predictions from store.model */ }
            .task { store.load() }
    }
}
```

The model is loaded once, held in `@Observable` state, and read by every view that observes the store. The decoding cost is paid at launch rather than on every navigation. For the saving side of this pattern, including writing a fitted model out at the end of a session, see <doc:Model-Persistence>.

### Practical realities of sensors and storage on iOS

iOS has more sources of data than watchOS and more places to put that data than Vapor. A few practical realities are worth knowing before building a screen on top of a Quiver model.

**When a screen appears and disappears.** Models load once when the app launches, or the first time the screen that needs them appears, and then live in observable state for the rest of the session. SwiftUI re-renders cheaply against the same fitted value, so reading predictions on every view update is fine. Refitting on every view update is not — it is a common cause of stutter that goes away the moment the fit moves into a `.task` modifier or a settings action the user triggers.

**Storage across launches.** Pre-fitted models that ship with the app live in the bundle as read-only JSON, decoded with `JSONDecoder` at launch. User-derived state — a personal baseline, a fitted personalization model, a precomputed collection of item vectors — lives in `Documents/`, written through `JSONEncoder` at the end of a session and read at the start of the next one. The two locations correspond to a single distinction. Models that are the same for every user live in the bundle. Models specific to this user live in `Documents/`.

**How much can fit in memory.** The iPhone has substantially more memory than the Watch and noticeably less than a server. A fitted Quiver model with ten thousand training samples and a dozen features is comfortable. A precomputed collection of a few thousand item vectors at moderate dimensionality fits without strain. Half a million samples held in memory is a different problem and calls for downsampling, paging, or moving the work to a background process.

> Tip: A view that refits the model in `onAppear` will refit every time the user navigates back to it. Move the fit to a `.task` that captures a stable identifier, or to a settings action the user explicitly triggers, so the cost is paid once per session rather than once per appearance.

### Working with sensor inputs and user data

iOS apps have many places to read from — `HKHealthStore` for health data, `CMMotionManager` for motion, `UserDefaults` and `FileManager` for everything the app has stored itself, the keyboard and form fields for user input. Quiver does not interact with any of these sources directly. The pattern is to decode whatever the source produces into `[Double]` once, near the boundary, and let everything downstream operate on the same plain arrays Quiver computes on.

```swift
import Foundation
import Quiver

@MainActor
final class MeasurementAnalysis {
    // Decodes a CSV column of numeric measurements into a Quiver vector
    func loadMeasurements(from url: URL, column: Int) throws -> [Double] {
        let text = try String(contentsOf: url, encoding: .utf8)
        return text
            .split(separator: "\n")
            .dropFirst()  // skip the header row
            .compactMap { row in
                let fields = row.split(separator: ",")
                guard column < fields.count else { return nil }
                return Double(fields[column])
            }
    }

    // Summarizes the values with Quiver
    func summary(from url: URL) throws -> (mean: Double, std: Double, anomalies: Int)? {
        let values = try loadMeasurements(from: url, column: 1)
        guard let mean = values.mean(), let std = values.std() else { return nil }
        let mask = values.outlierMask(threshold: 2.0)
        return (mean, std, mask.filter { $0 }.count)
    }
}
```

The CSV becomes a `[Double]` at the boundary. From there, every Quiver method — `mean()`, `std()`, `standardized()`, `quartiles()`, `outlierMask(threshold:)`, or a fitted model's `predict(...)` — works the same way it does anywhere else. The same shape applies to `HKHealthStore` samples mapped into a `[Double]` at the query callback, `CMMotionManager` accelerometer values pulled out of a `CMAccelerometerData`, or a list of numeric form entries collected from a SwiftUI form. Decode once, then treat the array as a Quiver value for the rest of the time the screen is on display.

> Tip: iOS users notice when an app asks for permissions it does not need. Reading from `HKHealthStore` or `CMMotionManager` requires explicit authorization, and the permission prompt happens the first time the app reads. Bundle the read with a clear context — a settings action the user opted into, or the first appearance of a feature that requires it — rather than at app launch.

## See also

- <doc:iOS-Patterns> — Personalization, anomaly surfacing, recommendation, and chart aggregation patterns
- <doc:Concurrency-Primer> — Swift Concurrency patterns for fitting off the main thread
- <doc:Model-Persistence> — Saving and loading fitted models with `Codable`
- <doc:Pipeline> — Bundling a scaler and model into a single matched pair
- <doc:Panel> — Named columns over `[Double]` for multi-feature work
- <doc:watchOS-Guide> — The watchOS counterpart, focused on live sensor streams
- <doc:Vapor-Guide> — The Swift on Server counterpart, focused on routes that share a fitted model
