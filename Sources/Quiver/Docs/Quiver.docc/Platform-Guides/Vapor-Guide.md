# Vapor Guide

Loading state when the server starts and sharing it safely across concurrent requests.

## Overview

A Swift server doing numerical work wants two things at once — answering each request quickly and sharing one fitted model or precomputed index across every concurrent handler. The hard part is doing both without a separate prediction service that adds latency to every request. Quiver builds the model once when the server starts and shares it across every handler as plain Swift, with no copies and no per-request load cost. This guide shows the pattern with a semantic-search server as the example.

### Loading state at boot

The shape we recommend for a Vapor server is straightforward: build any reference data the server needs once when the application starts, capture it in a value the route closures can read, and never reconstruct it inside a request handler. Boot-time state might be a fitted model decoded from a JSON file, a dictionary of word embeddings loaded from a bundled resource, or a precomputed catalog of document vectors built from the seed data the application ships with.

```swift
import Vapor
import Quiver

func configure(_ app: Application) throws {
    // Built once when the server starts — never rebuilt inside a request
    let store = ProductStore()
    seededStore(store)

    try routes(app, store: store)
}
```

The seeded `ProductStore` holds the catalog of product descriptions, the embeddings dictionary, and the precomputed vector for every product. Building it is the expensive operation. Reading from it is fast. Doing the build inside a request handler — opening a JSON file, tokenizing every document, computing every vector — would put that cost on every single request and slow the server down badly when traffic increases.

### Sharing fitted state across concurrent handlers

The foundational pattern on Vapor is many requests reading the same boot-time state in parallel without stepping on each other. Quiver's value types make this natural: a fitted model, a precomputed vector index, a dictionary of embeddings — none of these change after construction, and Swift's `Sendable` system lets the compiler verify that sharing them across concurrent tasks is safe.

```swift
import Vapor
import Quiver

// The store is captured by the route closure and read from every request handler
final class ProductStore: @unchecked Sendable {
    private(set) var shoes: [Shoe] = []

    func add(_ description: String) {
        let tokens = description.tokenize()
        let vectors = tokens.embed(using: embeddings)
        guard let vector = vectors.meanVector() else { return }
        shoes.append(Shoe(description: description, vector: vector))
    }
}

func routes(_ app: Application, store: ProductStore) throws {
    // Every search request reads the same store — no locks, no copies of catalog state
    app.get("search") { request -> [SearchResult] in
        guard let query = request.query[String.self, at: "q"] else {
            throw Abort(.badRequest, reason: "Missing query parameter ?q=")
        }
        return store.search(query: query)
    }
}
```

The closure captures `store` as a reference. There is one catalog, every search handler reads it in parallel, and the search itself is a pure read of the precomputed vectors — tokenize the query, embed it, rank against the stored document vectors. No state crosses request boundaries.

The `@unchecked Sendable` annotation on `ProductStore` is the Swift-server idiom for "this reference type is safe to share, trust me." It is appropriate here because Vapor's async context serializes the mutations (`add`, `remove`) that the catalog needs to support, while reads (`search`) happen in parallel against unchanging vector data. For state that changes more often or in less predictable patterns, an `actor` is the better tool — the compiler enforces serialization without the trust.

### Reading Swift arrays from the request

Vapor decodes `Codable` request bodies; Quiver computes on `[Double]` and `[[Double]]`. They are the same arrays. There is no conversion step between them.

```swift
import Vapor
import Quiver

struct ScoreRequest: Content { let features: [Double] }
struct ScoreResponse: Content { let score: Double }

app.post("score") { request -> ScoreResponse in
    // Decoded directly into a Quiver-shaped value
    let input = try request.content.decode(ScoreRequest.self)
    let prediction = model.predict([input.features])[0]
    return ScoreResponse(score: prediction)
}
```

The `[Double]` that came off the wire is handed directly to `predict(...)`. No copy, no conversion, no second representation. The reverse is also true: a Quiver method's return value is a `[Double]` or a primitive that drops straight into a `Codable` response struct. Vapor encodes it; the client decodes it; the math the server did is the same math the client receives.

> Tip: `Array` in Swift is copy-on-write, so passing a `[Double]` to a Quiver method does not silently allocate. But constructing a fresh `[[Double]]` for every request out of stored data does, and that cost adds up. Build the data once when the server starts, read it many times per request.

### Practical realities on Vapor

A few practical points shape every Vapor route built around Quiver. Most are about where state lives and what crosses which boundary.

**What persists across requests, and what does not.** Anything that should outlive a single request — a fitted model, a precomputed collection of document vectors, a dictionary of word embeddings — lives in values created when the server starts and captured by the route closures. Anything specific to one request — a decoded body, a query parameter, the result of one similarity computation — lives in local variables inside the closure. Mixing the two is where bugs come from: a per-request mutation that accidentally touches a server-wide value breaks the unchanging guarantee that lets parallel requests share it safely.

**Pay loading costs once, not on every request.** Word embedding dictionaries, fitted models, and precomputed document vectors should be loaded when the server starts. Loading them inside a route handler would put that cost on every single request and degrade badly under load.

**Codable everywhere.** Every Quiver model is `Codable`, so a model trained on a developer machine can be encoded to JSON, shipped alongside the server, and decoded once at startup. Training never happens inside a request handler. For the saving side of this pattern, including encoding a model after fitting, see <doc:Model-Persistence>. For pairing a scaler with a model so they stay matched at prediction time, see <doc:Pipeline>.

> Experiment: [quiver-demo-vapor](https://github.com/waynewbishop/quiver-demo-vapor) is a small REST server that searches a catalog of running shoes by meaning rather than by keyword. A query for "cushioned long run" returns shoes whose descriptions never used those exact words but match the intent. Running the server and trying different queries shows Quiver turning free text into a ranked answer entirely in the Swift process, with no separate search service.
