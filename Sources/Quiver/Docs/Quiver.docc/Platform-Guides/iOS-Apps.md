# iOS Guide

Summarize history, rank similar items, and learn from a user's own behavior on iPhone.

## Overview

A consumer iOS app generates a stream of small events as the user moves through it: a trip taken, a session completed, a photo saved, a purchase recorded. These surfaces are rarely the raw events. Often they are derived numbers we compute from user actions or transactions.

**Statistics** turns a list of past events into a summary the iPhone can render. Linear algebra ranks items by relation in feature space. Machine learning fits a model of the user's own behavior so we can score a new event against the user's own pattern. With Quiver, every one of those actions runs on-device, against the user's own generated data.

### Setup and lifecycle

An iOS app that computes on user history needs two storage surfaces. **Bundle storage** holds artifacts the developer ships with the binary: a fitted model trained offline, a precomputed table, or a reference vector, encoded as JSON and read on first launch. **Documents storage** holds artifacts derived from the running user, including their accumulated history, their personal baseline, and a model fit from their own data, encoded at session end and decoded at the next launch.

The convention is worth holding from the first line of code. Anything that came from the developer lives in the bundle. Anything that came from the user lives in Documents. Mixing the two is the source of bugs that survive uninstall and lose user history on update. The pillar sections below build up the values that flow into each surface: descriptive statistics, similarity rankings, and fitted models. See <doc:Model-Persistence> for the encoding-and-decoding mechanics once the values exist.

## Statistics on iOS

An iOS app accumulates user events as it runs: trips taken, workouts logged, purchases recorded, sessions completed. A list of eighty-four trip delays is not a screen. The job of descriptive statistics is to turn that list into the four or five numbers a screen renders: a typical value, a spread, a cut-off the user is asking about. Quiver computes these from the raw `[Double]` the app already holds, returns a typed snapshot ready for SwiftUI, and runs entirely on-device.

### From raw history to typed summary

The **median** is the middle value when the data is sorted; it answers "what is typical" without letting one extreme value pull the answer around. **Quartiles** are the three cut points that divide the sorted data into four equal-sized groups, and the **third quartile** is the value below which three-quarters of the history sits. Together, those two ideas turn raw history into the building blocks of a year-in-review screen, a personal-best banner, or a "three out of four trips arrive within" line. See <doc:Statistics-Primer> for the full vocabulary of central tendency and spread.

The lead example is an on-time-performance screen. Each entry is a trip delay in minutes — negative means the trip arrived early, positive means late. We report the share of on-time arrivals alongside the median delay and a typical-ceiling line drawn from the third quartile.

```swift
import Quiver

// Past trip history — minutes late (negative = early).
let delays = [-5.0, -2.0, 3.0, 7.0, 9.0, 14.0, 18.0, 42.0]

guard let summary = delays.summary() else { return }

print(summary.median) // 8.0 — half the trips arrived within eight minutes of schedule
print(summary.q3)     // 15.0 — three out of four trips arrived within fifteen minutes

// Share of trips within ±15 minutes of schedule.
let onTime = delays.filter { abs($0) <= 15 }.count
print(Double(onTime) / Double(delays.count)) // 0.75 — three-of-four on-time rate
```

The `summary()` method returns a typed snapshot of the entire history: central tendency, spread, and the quartiles that bracket the middle 50%. The reason to take a typed snapshot rather than call eight statistics in eight places is consistency. The view, the cache, and the persistence layer all read the same value, so the median the user sees on screen is the same number the app wrote to disk last night. A typed value also crosses task boundaries as one argument and persists with one `Codable` encode, which matters in a view hierarchy that computes the summary once and reads it many times during a render cycle.

### Counting events by category

A second pattern shows up wherever the app needs to count occurrences rather than summarize a continuous quantity: most-played track, most-tagged photo location, most-frequent transaction merchant. A **frequency table** maps each distinct value to the number of times it appears, and a small typed value reports the mode and the count. See <doc:Frequency-Tables> for the operations Quiver exposes for categorical history.

The same statistical surface generalizes across any screen whose product is a history of repeated events. A sleep app reads bedtime durations the same way the trip app reads delays. A workout app reads pace samples the same way. The concept is unchanged — turn a list of past events into a typed summary — even when the events themselves are not.

### Computing the summary at the right moment

A SwiftUI view that reads `summary()` inside `body` recomputes on every render, which walks the whole history each time the parent state changes. We move the call to `.task` or compute it once when the underlying data changes, then hold the result in observable state.

## Linear algebra on iOS

Sorting by a column ranks items by one number: newest first, biggest first, or closest first. Ranking by closeness asks a different question. Given the item the user is reading now, which other items resemble it across many dimensions at once.

### Items as feature vectors

The answer requires treating each item as a **feature vector**, a short list of numbers, one per attribute, that places the item as a point in space. Distance in that space stands in for similarity. The most common measure of that distance is **cosine similarity**, the cosine of the angle between two feature vectors, ignoring how long either one is. Two items pointing in the same direction score close to 1; two unrelated items score close to 0. This is the math underneath every "items like this one" feature in a consumer app. See <doc:Linear-Algebra-Primer> for the geometric foundation and <doc:Vector-Operations> for the operations Quiver exposes on `[Double]`.

The lead example is a similar-items recommendation. We summarize each item in the catalog as a small feature vector, then rank the catalog against the current item by cosine similarity.

```swift
import Quiver

// Each item summarized as [length_km, avg_grade_percent, surface_code].
let catalog: [[Double]] = [
    [10.0,  2.0, 0.0],   // long, low-grade, road
    [12.0,  2.5, 0.0],   // long, low-grade, road
    [ 8.0,  3.0, 0.0],   // long, low-grade, road
    [ 3.0, 12.0, 1.0],   // short, steep, trail
    [ 2.5, 11.0, 1.0]    // short, steep, trail
]
let current: [Double] = [11.0, 2.0, 0.0]   // long, low-grade, road

// Score every catalog item against the query and return the closest matches.
let scores = catalog.cosineSimilarities(to: current)
// scores: [0.9998, 0.9997, 0.9840, 0.4108, 0.3910]

let top = scores.topIndices(k: 3)
// top: [(rank: 1, index: 0, score: 0.9998),
//       (rank: 2, index: 1, score: 0.9997),
//       (rank: 3, index: 2, score: 0.9840)]
```

The `cosineSimilarities` call returns one score per catalog item: a value close to 1 means the item points in the same direction as the query across every dimension, and `topIndices` reads off the closest matches in one pass.

### Fair comparison across scales

The cosine works cleanly only when every feature lives on the same scale. A length measured in kilometers and a grade measured in percent live on different scales, and the larger numbers dominate the cosine. `FeatureScaler` is the bridge between raw features and a fair comparison: `fit` learns the column means and standard deviations from the catalog, and `transform` applies them to any vector before the ranking step.

```swift
import Quiver

// Same catalog and query as the previous block.
let catalog: [[Double]] = [
    [10.0, 2.0, 0.0], [12.0, 2.5, 0.0], [8.0, 3.0, 0.0],
    [3.0, 12.0, 1.0], [2.5, 11.0, 1.0]
]
let current: [Double] = [11.0, 2.0, 0.0]

// Scale features so any single dimension does not dominate.
let scaler = FeatureScaler.fit(features: catalog)
let scaledCatalog = scaler.transform(catalog)
let scaledQuery = scaler.transform([current])[0]

// Same ranking call — now over scaled vectors.
let fairScores = scaledCatalog.cosineSimilarities(to: scaledQuery)
print(fairScores.topIndices(k: 3)) // ranks 1–3 are indices 0, 1, 2 — same ranking, now defensible
```

After scaling, the cosine reflects similarity along every dimension equally, and the same pipeline drives every "items like this one" feature the app exposes.

### Direction tests for gameplay and spatial features

The same primitives cover a second common case: short fixed-dimension vector math for gameplay and spatial features. A unit-direction `dot` product answers whether one object is in front of another, and the related projection operations described in <doc:Vector-Projections> turn alignment into a usable scalar.

```swift
import Quiver

// Player facing forward (+z); enemy position relative to the player.
let playerFacing: [Double] = [0.0, 0.0, 1.0]
let toEnemy: [Double] = [3.0, 0.0, 4.0]

// Positive dot with the normalized direction = in front of the player.
print(playerFacing.dot(toEnemy.normalized) > 0) // true — enemy is in front
```

The dot product of `playerFacing` and the normalized direction to the enemy reads as a sign test: positive means in front, negative means behind, zero means directly to the side.

A separate question — whether two motions are pointing the same way — calls for `cosineOfAngle` between two unit vectors. The result is a number between -1 and 1 that describes alignment regardless of speed.

```swift
import Quiver

// Same player and enemy as the previous block.
let playerFacing: [Double] = [0.0, 0.0, 1.0]
let toEnemy: [Double] = [3.0, 0.0, 4.0]

// Cosine close to 1.0 means two motion units are pointing the same way.
print(playerFacing.cosineOfAngle(with: toEnemy.normalized)) // 0.8 — pointing mostly the same way
```

A value close to `1.0` means the two directions are aligned; a value near `0` means they are perpendicular; a value near `-1` means they oppose each other.

## Machine learning on iOS

On iOS, machine learning comes up when the app needs to model the user's own behavior. We categorize the user's entries, predict what they will care about, or score a new event against their pattern, all without sending data off the device. See <doc:Machine-Learning-Primer> for the framing of features, labels, training, and evaluation that the rest of this section assumes.

### Fitting a model of the user

Two methods cover most of the surface: <doc:KMeans-Clustering> for unsupervised grouping, and <doc:Linear-Regression> for personal prediction. Each fits from a small history and returns a value the app can hold in observable state, persist to Documents, and read again on the next launch.

The lead example is a personal-prediction regression. We fit `LinearRegression` to a history of feature vectors and the outcome that followed each one. A new feature vector is scored against the user's own pattern, and a large residual is itself the signal — it means the new event broke the user's own pattern.

```swift
import Quiver

// A history of past events: each row is one event, the outcome that followed it is aligned by index.
let history: [[Double]] = [
    [1.0, 2.0, 1.0],
    [2.0, 3.0, 2.0],
    [3.0, 1.0, 4.0],
    [4.0, 5.0, 2.0],
    [5.0, 2.0, 3.0],
    [6.0, 4.0, 1.0]
]
let outcomes: [Double] = [9.0, 15.0, 13.0, 25.0, 19.0, 25.0]

let model = try LinearRegression.fit(features: history, targets: outcomes)

// Score an upcoming event against the user's own history.
let upcoming: [Double] = [3.0, 4.0, 2.0]
print(model.predict([upcoming])[0]) // 20.0 — the model recovered the underlying weights exactly
```

### Persisting the fitted value

The fitted model is a value, not a service. It encodes to JSON, persists to Documents, and decodes unchanged on the next launch. The same `Codable` shape covers <doc:KMeans-Clustering> for unsupervised grouping, <doc:Linear-Regression> for prediction, and <doc:Nearest-Neighbors-Classification> for similarity-based labeling. See <doc:Model-Persistence> for the encode-once, decode-on-launch pattern every model on iOS shares.

### Composing scaling and fitting end-to-end

Most personal models need more than one step: a `FeatureScaler` learns the column statistics, then the fitted estimator runs on the scaled features. <doc:Pipeline> ties the two together into one fit-and-predict workflow that round-trips through `Codable` as a single value, which keeps the scaling rule and the model coefficients from ever drifting out of sync across app launches.

> Tip: Quiver ships four fitted models on iOS: `LinearRegression` for personal prediction, `KNearestNeighbors` for similarity-based classification, `KMeans` for unsupervised grouping, and `GaussianNaiveBayes` for probabilistic classification. Each composes with `FeatureScaler` for consistent feature scaling and `Pipeline` for end-to-end fit-and-predict workflows.

## Where to go from here

The three sections above each have a deeper layer of math underneath them, and that math is the next step for iOS developers moving into numerical work. <doc:Statistics-Primer> builds the vocabulary of variance, distributions, and inference. <doc:Linear-Algebra-Primer> extends vectors and dot products into matrices, transformations, and projection. <doc:Machine-Learning-Primer> ties both together — features, labels, training, evaluation, and the trade-offs that decide which model to reach for. Once an app has numbers worth showing, <doc:Data-Visualization> covers the aggregation primitives that feed Swift Charts directly.

> Experiment: **The Quiver Notebook** is the right place to see how these surfaces compose on an iOS-sized dataset. Fit `KMeans` on a few dozen session vectors, then `LinearRegression` on a personal history of outcomes, and watch the same `Codable` model drop from Notebook into an iOS bundle unchanged. See <doc:Quiver-Notebook>.
