# Vapor Patterns

Serving vector databases, inference endpoints, and statistical engines with Quiver on Swift on Server.

## Overview

The <doc:Vapor-Guide> covers the foundations — routes that share one fitted model, the JSON-to-`[Double]` boundary, and loading fitted state when the server starts. This article builds on those foundations with applied patterns Vapor and Quiver developers reach for repeatedly: a semantic search index behind an HTTP route, a prediction endpoint that loads its model once and uses it for every request, a stats engine for hypothesis tests and rolling summaries, and the question of how a fitted model behaves when many requests arrive at the same time.

### Vector database — semantic search behind a route

The clearest pattern for Quiver on Vapor is a semantic search index. A collection of documents — products, articles, notes, anything textual — is tokenized and turned into vectors once. The resulting document vectors are kept in memory for the life of the server, and every search request decodes a query, turns it into a vector, ranks it against the precomputed collection, and returns the top matches. The collection rarely changes; reads happen constantly; the work all stays inside the Swift process.

The full embedding pipeline is covered in <doc:Semantic-Search>. What changes when the pipeline lives behind HTTP is where each piece runs: tokenization happens twice (once when documents are loaded, once per query), document vectors are precomputed and held in memory, and only the query side runs inside the request handler.

```swift
import Vapor
import Quiver

func routes(_ app: Application) throws {
    let store = ProductStore()
    seed(store)  // Tokenize, embed, and store vectors once at server start

    app.post("products") { request throws -> HTTPStatus in
        let input = try request.content.decode(AddRequest.self)
        store.add(input.description)  // Updates the catalog separately, not inside a search request
        return .created
    }

    // Search runs against the precomputed collection — no fitting, no re-embedding documents
    app.get("search") { request -> [SearchResult] in
        guard let query = request.query[String.self, at: "q"] else {
            throw Abort(.badRequest, reason: "Missing query parameter ?q=")
        }
        return store.search(query: query).map {
            SearchResult(rank: $0.rank, description: $0.label, similarity: $0.score)
        }
    }
}

struct AddRequest: Content { let description: String }
struct SearchResult: Content { let rank: Int; let description: String; let similarity: Double }
```

This is the shape the [running-shoes demo](https://github.com/waynewbishop/quiver-demo-vapor) ships. Inside `ProductStore.search(...)`, the query is tokenized, embedded, and reduced to a single vector with `meanVector`, then ranked against precomputed document vectors with `cosineSimilarities(to:)` and `topIndices(k:labels:)`.

What this pattern is and what it is not: it is a search index over precomputed embeddings, optimized for fast reads. It is not a retraining loop. The catalog changes on demand through `POST /products`, never inside the search request handler. For the underlying operations, see <doc:Semantic-Search> and <doc:Similarity-Operations>.

### Prediction endpoint — load once, predict on every request

The most common machine-learning shape on Vapor is a prediction endpoint backed by a model that was trained somewhere else. Training runs in a notebook, a command-line tool, or a scheduled job. The result is saved as JSON through `Codable`. The server loads it when it starts and exposes a route that decodes a feature vector and returns a prediction. Training never happens inside a request handler.

A lead-scoring regression is a clean example. Historical features and outcomes feed a `LinearRegression` fit somewhere away from the server; the resulting model is shipped to the server as a JSON file.

```swift
import Vapor
import Quiver

func routes(_ app: Application) throws {
    // Load the fitted model when the server starts — request handlers never refit
    let modelURL = URL(fileURLWithPath: "Resources/lead-scorer.json")
    let data = try Data(contentsOf: modelURL)
    let model = try JSONDecoder().decode(LinearRegression.self, from: data)

    app.post("score") { request -> ScoreResponse in
        let input = try request.content.decode(ScoreRequest.self)
        let prediction = model.predict([input.features])[0]
        return ScoreResponse(score: prediction)
    }
}

struct ScoreRequest: Content { let features: [Double] }
struct ScoreResponse: Content { let score: Double }
```

This pattern serves a fitted model, but does not retrain inside a request handler — for that, the same logic that <doc:watchOS-Patterns> walks through under "When to train, when to predict" applies on the server, just with cron jobs or scheduled workers instead of workout session boundaries.

> Tip: When training pairs a scaler with the model, save them together so they stay matched at prediction time. Decoding two separate JSON files invites a mismatched pair the first time someone retrains one and forgets the other. See <doc:Pipeline>.

### Stats engine — self-contained tests and rolling-summary services

Some Vapor and Quiver workloads are not classification or regression at all. They are hypothesis tests, confidence intervals, or rolling summaries: confirm whether two samples differ, summarize a stream of incoming observations, flag anomalies as they arrive. Quiver fits both shapes, but they call for different ways of holding state.

**Self-contained test endpoint.** A request body carries a sample of observations, the route runs a hypothesis test or a regression on that sample, and the response is the result. No state carries over from one request to the next. Each request is complete on its own, which means the route stays safe even when many requests arrive in parallel.

```swift
import Vapor
import Quiver

struct RegressionRequest: Content {
    let features: [[Double]]
    let targets: [Double]
}

struct RegressionResponse: Content {
    let coefficients: [Double]
    let hasIntercept: Bool
}

app.post("experiments/regression") { request -> RegressionResponse in
    let input = try request.content.decode(RegressionRequest.self)
    let model = try LinearRegression.fit(features: input.features, targets: input.targets)
    return RegressionResponse(coefficients: model.coefficients, hasIntercept: model.hasIntercept)
}
```

Each request fits a fresh model on the data it brought, returns a structured result, and exits. Nothing accumulates between requests.

**Rolling-summary actor.** When the server needs to take in a stream of observations and expose a running summary — anomaly detection over incoming metrics, z-scores against a moving window — an `actor` is the right shape. The actor owns the moving buffer; routes write into it and read summaries from it. Swift Concurrency makes sure only one task touches the actor's state at a time, so the buffer stays consistent without manual locks.

```swift
actor MetricsAccumulator {
    private var window: [Double] = []
    private let capacity = 1_000

    func ingest(_ value: Double) {
        window.append(value)
        if window.count > capacity { window.removeFirst() }
    }

    func summary() -> (mean: Double, std: Double) {
        return (window.mean() ?? 0, window.std() ?? 0)
    }
}
```

What these patterns are and what they are not: these are prediction and analytical operations. Quiver supplies the math; storing the accumulated observations long-term is still a database problem.

### Sizing a model for the server

Server hardware is faster than a Watch and has more memory available, so the timing numbers from <doc:watchOS-Patterns>'s "Sizing a model for the wrist" are an upper bound for the same operations on Vapor. A K-Means fit that runs in about a millisecond on Apple Silicon will run faster on a server CPU. The interesting sizing question on Vapor is not how fast a single fit runs — it is what happens when many requests arrive at once.

Can many request handlers all call `predict(...)` against the same fitted model, in parallel, without any coordination between them? Yes. The model never changes after `fit` returns. `predict(...)` is a pure read — coefficients in, prediction out, no shared values touched along the way. Because the model is `Sendable`, the compiler verifies the value is safe to share across tasks. Because Swift's `Array` is copy-on-write, the captured `[Double]` parameters do not duplicate storage on read.

The result is that fitting is the costly operation (kept off the request handler, run when the server starts or on a schedule), and prediction is the only operation that runs per request. With Quiver's classical models running in milliseconds even on Watch silicon, prediction on a server fits comfortably inside the time budget of an HTTP handler with room to spare.

> Tip: `Array` in Swift is copy-on-write, so passing a `[Double]` to `predict(...)` shares storage. But mutating a captured array inside a closure forces a copy. In a busy request handler, that cost is small per request but adds up when traffic stays high — prefer reading from server-wide arrays rather than mutating them.

