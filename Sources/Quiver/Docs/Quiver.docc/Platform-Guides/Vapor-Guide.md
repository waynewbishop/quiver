# Vapor Guide

Serving fitted models and computing aggregates across requests with Quiver and Vapor.

## Overview

A Swift server typically does two kinds of numerical work. The first is per-request: a recommendation endpoint takes a user's recent activity and returns a ranked list, a scoring endpoint takes a feature vector and returns a probability, a personalization endpoint returns the next item to show. The second is across requests: an analytics route aggregates a metric across the user base, a daily job recomputes a fitted model from accumulated data, an experiment endpoint compares two cohorts. Both kinds happen on the same routes, in the same process, hit by many concurrent requests.

Quiver fits this reality by running inside the Vapor process as pure Swift, with no separate prediction service and no language bridge. A model fit at boot time lives as a `Sendable` value the app holds once and reads from every request handler — no locks, no copies, no per-request load cost. A `Panel` of named columns aggregates a metric across user records the same way it would in an iOS app. Evaluation metrics — `accuracy`, `precision`, `recall`, `R²` — close the loop on whether the model is still serving well. The same `Codable` types and value semantics the iOS app uses on the client work on the server.

### Where Quiver fits

Quiver is built as extensions on the standard library `Array`, so a `[Double]` decoded from a JSON request body is the same `[Double]` the model computes on. A fitted model is a plain Swift struct — `Sendable`, so the compiler verifies it is safe to share across the handlers that [Vapor](https://vapor.codes) runs in parallel, and `Codable`, so we can train a model on a developer machine, encode it to JSON, ship it alongside the server, and load it once when the server starts.

That foundation supports three kinds of service a Vapor application is well-suited to host.

#### Prediction endpoints

A prediction endpoint serves a model that was trained somewhere else. Historical data feeds a regression or classifier in a notebook or a scheduled job, the result is saved as JSON, and the server reads that JSON at startup so it can answer prediction requests using fresh feature vectors.

#### Vector databases

A vector database holds a collection of documents that have been tokenized and turned into vectors once when the server starts. Search requests turn the incoming query into a vector the same way and rank it against those precomputed document vectors. The collection rarely changes; reads happen constantly.

#### Statistical services

A statistical service runs the kind of work that does not need a trained model at all — confidence intervals, hypothesis tests, rolling summaries. A simple route runs a regression or hypothesis test on the sample of numbers the request carries, or a long-lived helper holds a moving window of values that other routes write to and read summaries from.

The patterns in this guide and in <doc:Vapor-Patterns> show how to pair each shape with Vapor's request handling, Swift Concurrency, and the values the server loads at startup.

### Routes that share a single fitted model

The foundational pattern on Vapor is many requests reading the same fitted model in parallel without stepping on each other. The model is trained once — when the server starts, or earlier on a different machine — and held in a value the route closure can capture. Because the model never changes after fitting and is `Sendable`, no locks are needed. Every request handler computes a prediction against the same parameters and returns.

```swift
import Vapor
import Quiver

func routes(_ app: Application) throws {
    // Fit once at boot. The model is immutable from this point forward.
    let sqft   = [1200.0, 1500.0, 1800.0, 2100.0]
    let prices = [250_000.0, 320_000.0, 380_000.0, 440_000.0]
    let model = try LinearRegression.fit(features: sqft, targets: prices)

    // Every request reads the same fitted model — no locks, no copies of state
    app.post("estimate") { request -> [Double] in
        let input = try request.content.decode(EstimateRequest.self)
        return model.predict(input.values)
    }
}

struct EstimateRequest: Content { let values: [Double] }
```

The closure captures `model` as a value. There is one fitted instance, every request handler reads it in parallel, and prediction is a pure read of values that never change — which is what makes Quiver models safe to share without converting them into a transport format.

> Tip: This is the same pattern the running-shoes demo uses. A `ProductStore` is built once when the server starts, captured by the route closures, and read from every request. The work that happens per request is decoding, computing similarities, and encoding — none of it touches the fitted state.

### Practical realities on Vapor

A few practical points are worth knowing before designing a Vapor route around Quiver. Most are about where state lives and what crosses which boundary.

**What persists across requests, and what does not.** Anything that should outlive a single request — a fitted model, a precomputed collection of document vectors, a dictionary of word embeddings — lives in values created when the server starts and captured by the route closures. Anything specific to one request — a decoded body, a query parameter, the result of one similarity computation — lives in local variables inside the closure. Mixing the two is where bugs come from: a per-request mutation that accidentally touches a server-wide value breaks the unchanging guarantee that lets parallel requests share it safely.

**JSON in, `[Double]` out.** Vapor decodes `Codable` request bodies; Quiver computes on `[Double]` and `[[Double]]`. They are the same arrays. There is no conversion step between them.

```swift
struct ScoreRequest: Content { let features: [Double] }

app.post("score") { request -> [Double] in
    let input = try request.content.decode(ScoreRequest.self)
    return model.predict([input.features])
}
```

The `[Double]` that came off the wire is handed directly to `predict(...)`. No copy, no conversion, no second representation.

**Pay loading costs once, not on every request.** Word embedding dictionaries, fitted models, and precomputed document vectors should be loaded when the server starts, not inside a route handler. Loading them inside a handler — opening a JSON file or rebuilding a vector cache — would put that cost on every single request and slow the server down badly when traffic increases. The pattern in the next section is how to avoid this.

> Tip: `Array` in Swift is copy-on-write, so passing a `[Double]` to a Quiver method does not silently allocate. But constructing a fresh `[[Double]]` for every request out of stored data does, and that cost adds up. Build the data once when the server starts, read it many times per request.

### Loading models when the server starts

The shape we recommend for a Vapor server is straightforward: train the model elsewhere, load it when the server starts, and only run predictions inside request handlers. Training might happen in a command-line tool, a notebook, or a scheduled job — anywhere except inside a handler that a user is waiting on. The result is encoded to JSON and shipped alongside the server. When the server starts, it decodes the JSON into a constant, and the route closures read from that constant for the life of the process.

```swift
import Vapor
import Quiver

func routes(_ app: Application) throws {
    // Load the fitted model at startup from a JSON file shipped with the server
    let modelURL = URL(fileURLWithPath: "Resources/lead-scorer.json")
    let modelData = try Data(contentsOf: modelURL)
    let model = try JSONDecoder().decode(LinearRegression.self, from: modelData)

    // The route reads the loaded model — training does not happen inside a request
    app.post("score") { request -> [Double] in
        let input = try request.content.decode(ScoreRequest.self)
        return model.predict([input.features])
    }
}

struct ScoreRequest: Content { let features: [Double] }
```

The `Codable` conformance on every Quiver model is what makes this work. Training happens once, somewhere away from the server, and the result is just a JSON file. The server has no notion of training — it only loads and predicts. For the saving side of this pattern, including encoding the model after fitting, see <doc:Model-Persistence>. For the case where a scaler and a model need to travel together as one matched pair, see <doc:Pipeline>.
