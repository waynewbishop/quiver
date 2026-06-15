# Vapor Guide

Summarize traffic, search by meaning, and serve fitted models from a Swift server on Linux.

## Overview

A Swift server needs to handle requests quickly while sharing expensive resources—like fitted models or embedding catalogs—across concurrent handlers. Quiver loads these resources at boot time and shares them as immutable Swift values, avoiding the cost of reconstruction per request.

### Lifecycle and resource sharing

To maximize performance, reference data (the model, [embedding dictionary](<doc:Embedding-Sources>), or vector catalog) should be built once at startup. Because Quiver’s model types are `Sendable` and immutable, we can safely share them across request handlers as plain Swift values with no locking or per-request overhead.

```swift
import Vapor
import Quiver

func configure(_ app: Application) throws {
    // Build expensive resources at boot—never inside a request.
    let store = ProductStore()
    seededStore(store)

    try routes(app, store: store)
}
```

Building the `ProductStore` at boot is a one-time cost; reading from it is fast. If we built these resources inside a request handler, we would impose that cost on every user, degrading server throughput under load.

> Tip: The search pipeline surveyed in this guide is assembled end-to-end in two worked examples: <doc:Semantic-Search> for the tokenize-embed-rank flow, and <doc:Embedding-Sources> for swapping the vector source behind one contract.

## Statistics on Vapor

A server is a stream of events. Requests arrive, handlers run, and latencies accumulate. The question the dashboard, the alert, and the SLO report all ask is the same: what is the 95th percentile of the last ten thousand request latencies, and is it inside our budget? Quiver answers that with one call against a `[Double]`, and the same primitives that compute the percentile compute the median, the IQR, and the typed snapshot the report renders from.

### Why percentiles beat averages

A **percentile** is the value below which a given fraction of the data sits. The 95th percentile of latency is the slowest second a typical user will see, ignoring the worst 5% of requests. Servers report percentiles instead of averages because one slow request can drag the mean upward without telling the operator how many users actually felt the slowness. See <doc:Statistics-Primer> for the vocabulary of central tendency, spread, and the quartile family the percentile call generalizes.

The lead example is a per-endpoint latency summary served as JSON from an admin route. The server holds the last N latencies in a rolling buffer; one call turns that buffer into a typed value the handler can return directly.

```swift
import Vapor
import Quiver

struct EndpointStats: Content {
    let count: Int
    let mean: Double
    let median: Double
    let p95: Double
    let p99: Double
}

app.get("admin", "stats", ":endpoint") { request -> EndpointStats in
    guard let endpoint = request.parameters.get("endpoint"),
          let latencies = store.recentLatencies(for: endpoint),
          let summary = latencies.summary(),
          let p95 = latencies.percentile(95),
          let p99 = latencies.percentile(99) else {
        throw Abort(.notFound)
    }

    return EndpointStats(
        count: summary.count,
        mean: summary.mean,
        median: summary.median,
        p95: p95,
        p99: p99
    )
}
```

### Typed snapshots cross task boundaries

The `summary()` method returns a typed snapshot of the buffer rather than a tuple of free-floating numbers. The reason matters more than the shape. A typed snapshot is one value: it crosses an `async` boundary as a single argument, persists to disk as one `Codable` write, and travels to a downstream handler as one parameter. We read fields by name, so the code that consumes the snapshot does not have to remember tuple positions or recompute statistics it already needs. The same shape describes one endpoint's latencies and a thousand endpoints' latencies; the consuming code does not change when the buffer changes size.

### Experiment significance from the same primitives

A second pattern covers experiment significance. When a server runs an experiment, each variant accumulates a stream of conversion outcomes, and the product decision rides on whether the difference between variants is real or a coincidence of the sample. A **confidence interval** is the range within which the true rate is expected to sit at a chosen confidence level. "Variant B converts 12% better, 95% CI [4%, 20%]" says the observed lift is unlikely to be zero. The handler computes the point estimate from `mean()`, the spread of that estimate from `standardError()`, and the critical multiplier from `Distributions.t.quantile`. No external service participates. See <doc:Inferential-Statistics-Primer> for the framing of standard error, t-distribution quantiles, and p-values: the foundation the experiment-significance route requires.

> Note: A handler that recomputes `summary()` on every request walks the entire rolling buffer every time. Hold the latest snapshot in the shared store and refresh it on a timer or at the end of every recording window, not in the request path.

## Linear algebra on Vapor

Keyword search looks for the words a user typed. Meaning search looks for what the user was after. The bridge between the two is a small stack of ideas that share one geometric picture: words become vectors, queries become vectors, and the catalog answers a question about distance. The next five subsections walk that stack from the words on the wire to the in-memory similarity search that serves the response.

### Tokenization and embedding

**Tokenization** is splitting a query string into a sequence of words, or *tokens*, that can each be looked up in an embedding dictionary. "running shoes for trail" becomes `["running", "shoes", "for", "trail"]`, and from that point on the search system works on tokens, not characters. The next step turns each token into a vector.

An **embedding** is a learned mapping from discrete tokens (words, items, users) to dense vectors in a space where related meanings end up geometrically close. "running" and "jogging" land near each other; "running" and "stapler" do not. The vectors are *dense*: every coordinate carries information, and the dimensions are tens or hundreds, not the millions of a one-hot encoding. A **one-hot encoding** is the alternative: a vector of length equal to the vocabulary, all zeros except for a single `1` at the word's index. One-hot is sparse, has no notion of similarity between words, and grows with the vocabulary. An embedding is the dense, semantic upgrade. See <doc:Linear-Algebra-Primer> for the geometric foundation and <doc:Vector-Operations> for the operations Quiver exposes on `[Double]`.

### The embedding dictionary

The lookup itself runs against an **embedding dictionary**, a precomputed `[String: [Double]]` the server loads once at boot. Every token the system can recognize has a row in this table; tokens the dictionary does not contain are dropped or routed to an `<unknown>` vector. In production the dictionary's contents typically come from a downloaded model file such as GloVe, word2vec, or the per-token outputs of a sentence-transformer. The server treats those files as data, not code.

Two practical numbers shape how the dictionary lives in memory. The first is **dimensionality**, the length of each vector. Common choices are 50, 100, 200, or 300 dimensions for word-level embeddings, and 384, 768, or 1,024 for sentence-transformer outputs. A 300-dimensional vector of `Double` is 2,400 bytes; a 100,000-word dictionary at that dimensionality is roughly 230 megabytes in memory. The second number is **vocabulary coverage**: how much of the language the server has to recognize. A consumer search box needs a broad vocabulary; a domain-specific catalog (medical, legal, technical) often does better with a smaller, specialized dictionary trained on the relevant corpus. The right dictionary is the smallest one that covers the queries the server will actually see.

The dictionary shown here is the simplest source of vectors, and Quiver names that source as a contract: the `Embedder` protocol is text in, `[Double]` out. A server that outgrows a word-vector table swaps in an on-device sentence model behind the same contract, and every line that tokenizes, ranks, and reports stays as written. See <doc:Embedding-Sources> for the swappable-source pattern and the levels from a hand-built table to a custom model.

```swift
import Vapor
import Quiver

app.get("search") { request -> [SearchResult] in
    guard let query = request.query[String.self, at: "q"] else {
        throw Abort(.badRequest, reason: "Missing query parameter ?q=")
    }

    // Tokenize the query, then look each token up in the embeddings dictionary.
    let tokens = query.tokenize()
    let vectors = tokens.embed(using: embeddings)

    // ... mean-pooling and ranking continue below
}
```

`tokenize` splits the query into words and `embed(using:)` looks each word up in the shared embeddings dictionary, returning one vector per token.

### Mean-pooling a query into one vector

The catalog stores one vector per document, so the query has to collapse to one vector too. The standard reduction is **mean-pooling**: average the per-token vectors element-by-element to produce a single vector that represents the whole query. Quiver exposes the reduction as `meanVector()`.

```swift
    // Reduce the per-token vectors to one query vector.
    guard let queryVector = vectors.meanVector() else {
        return []
    }
```

The reason mean-pooling works is geometric. Each token vector points toward the meaning of that word in the embedding space, and the average of those vectors lands near the centroid of the meaning: the point that minimizes the total distance to the contributing tokens. For short queries of three to ten words, the centroid is close enough to the query's overall meaning that the cosine comparison downstream gives sensible rankings. The technique is cheap, has no learned parameters, and is the right default for query-to-document retrieval at this scale.

Two limitations are worth naming so the model on the server side stays calibrated. Mean-pooling treats every token with equal weight, which means a query like "running shoes" gives "running" and "shoes" the same pull on the centroid even though one carries most of the semantic load. Stop-word filtering before pooling reduces the dilution. The second limitation is order: averaging discards the sequence the words arrived in, so "dog bites man" and "man bites dog" pool to the same vector. For retrieval over short noun-phrase queries, neither limitation hurts the results enough to justify a more expensive pooler. Longer or more structured inputs benefit from learned poolers, but those belong to the model that produced the embedding, not to the retrieval step on the server.

### Ranking against the catalog

The query is now one vector in the same space as every document vector. Ranking is asking which catalog vector is closest. **Cosine similarity** is the standard measure of "close" in this setting, the cosine of the angle between two vectors, ignoring how long either one is. Two items pointing in the same direction score close to 1; two unrelated items score close to 0. Quiver's `cosineSimilarities` computes one score per document in a single call; `topIndices(k:labels:)` reads off the closest few paired with their original labels. See <doc:Similarity-Operations> for the family of operations and <doc:Semantic-Search> for the end-to-end query-to-results pattern this endpoint follows.

```swift
    // Rank the query vector against precomputed document vectors.
    let productVectors = store.products.map(\.vector)
    let similarities = productVectors.cosineSimilarities(to: queryVector)
    let top = similarities.topIndices(k: 5, labels: store.products)

    return top.map { result in
        SearchResult(
            rank: result.rank,
            description: result.label.description,
            similarity: result.score
        )
    }
}

struct SearchResult: Content {
    let rank: Int
    let description: String
    let similarity: Double
}
```

`cosineSimilarities` returns one score per document, and `topIndices(k:labels:)` reads off the five closest matches in one pass. The query side and the catalog side ran through the same tokenize-embed-mean steps, which keeps both sides of the comparison in the same vector space. If we embed documents one way and queries another, the cosine numbers mean nothing.

> Tip: Embedding the catalog is the expensive step. The server pays that cost exactly once per document, at the moment the document enters the system, never at search time. The same pipeline drives both sides of the catalog, so the create-time path and the query-time path stay in sync without a second representation.

### In-memory similarity search

The pattern above is **in-memory similarity search**: store precomputed embeddings in a process-local collection, then serve nearest-neighbor queries against them. The Vapor process running this code holds the catalog, embeds incoming queries, and returns the top-k by cosine similarity. The pieces are an embedding step at write time, a stored collection of vectors, a query path that embeds the input, and a similarity computation that returns the top-k.

The most common application of this shape today is **retrieval-augmented generation**, or RAG: the retrieval half of the pipeline finds the top-k most relevant documents for a user's question, and a language model receives those documents as context for its answer. Quiver's role in a RAG system is the retrieval half. Quiver does not ship a language model, does not call out to one, and does not pretend to. The Vapor route shown above is the retrieval surface; the generation step belongs to whatever model the application chooses to call afterward.

Two practical observations follow from that framing. First, the retrieval layer does not have to be a separate service. For catalogs the Vapor process can hold in memory (thousands to low millions of vectors), the catalog is the in-process value the routes already read from. Second, the catalog does not have to be permanent. Catalogs that fit in memory are fast to read but do not survive a restart. When the catalog needs to outlast process lifetime or grow beyond memory, the embedding and ranking math stays in the Vapor process and the storage moves to the application's existing data layer.

### Exact versus approximate nearest neighbors

The `cosineSimilarities` call is **exact**: it computes the similarity between the query vector and every catalog vector, then returns the top-k. The cost is linear in the catalog size, and the result is the true top-k by definition. For catalogs a single Vapor process holds in memory, thousands to low millions of vectors, exact search is the right default. The compute budget is small, the answer is correct, and the implementation is one line.

**Approximate nearest-neighbor** methods trade exactness for speed at larger scale. Algorithms such as HNSW and IVF build an index ahead of time that lets a query skip most of the catalog and still return results that are *almost* the true top-k. The trade is real: the index takes memory, takes time to build, and returns answers that are correct most of the time but not always. Approximate methods become relevant when the catalog outgrows memory or when the exact pass would miss the latency budget. Below that threshold, exact search is faster to ship and easier to reason about. See <doc:Semantic-Search> for the broader retrieval framing and the conditions under which an approximate index becomes the right call.

The threshold is set by the latency budget, not by a hard catalog size. An exact pass over 100,000 three-hundred-dimensional vectors is a few million floating-point operations — single-digit milliseconds on a modern server core. The same pass over ten million vectors is two orders of magnitude slower and starts to crowd a typical 50-millisecond budget for an interactive search response. Two signals say the server has crossed the threshold: the exact pass shows up in the p95 latency, and the catalog stops fitting in the process's memory headroom. Until both are true, the right answer is the exact pass and a clear path to swap implementations later. The query handler, the cosine math, and the response shape do not change when the storage and ranking layer moves underneath them.

> Note: Swapping the implementation later is the design point worth defending early. The route signature, the request and response types, and the call into Quiver stay constant; only the layer that holds the vectors and answers the nearest-neighbor question changes. A move to an external vector-database service or an on-disk index does not require rewriting the handler.

## Machine learning on Vapor

Training and inference are two different jobs. **Training** is the slow, one-shot process of fitting a model to historical data: sweeping many examples, adjusting coefficients, sometimes iterating to convergence. **Inference** is the fast, repeated process of asking the trained model what it predicts for one new input. A server only does the second one. We build the fitted model ahead of time, serialize it to JSON, and read it back at startup; from that point on, every request is one `predict()` call against the same in-memory value. See <doc:Machine-Learning-Primer> for the framing of features, labels, training, and evaluation the rest of this section assumes.

### The shape of a fitted value

A fitted model is what the server reads at boot, so the shape of that fitted value is worth seeing first. On the developer machine, <doc:Linear-Regression> takes a feature matrix and a target vector and returns a `LinearRegression` value with a `predict` method. When the feature columns carry overlapping information (the multi-signal case where a plain least-squares fit turns unstable), <doc:Ridge-Regression> swaps in behind the identical request-handling shape, fit offline and decoded at boot the same way.

```swift
import Quiver

// Fit on the developer machine against a small training set.
let features: [[Double]] = [
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 5.0],
    [4.0, 7.0]
]
let targets: [Double] = [3.1, 5.0, 8.2, 11.1]

let fitted = try LinearRegression.fit(features: features, targets: targets)
print(fitted.predict([[5.0, 9.0]])[0]) // ≈ 14.1 — the fitted model recovers x₁ + x₂ on this small training set
```

A model fit this way, on a developer machine, encodes to JSON for the server to load at startup. That JSON is what every Vapor instance reads at boot. See <doc:Model-Persistence> for the encode-once, decode-on-boot shape every model on the server shares.

### Fit at boot and predict per request

The lead example is a fit-at-boot scoring endpoint. The fitted model lives in JSON, which we decode once at `Application.boot()`. Every `/score` request decodes a JSON body, calls `predict()`, and returns the result.

```swift
import Vapor
import Quiver

// Boot-time setup.
struct ModelStore: Sendable {
    let model: LinearRegression

    init() throws {
        guard let url = Bundle.module.url(forResource: "lead-scorer", withExtension: "json") else {
            throw Abort(.internalServerError, reason: "lead-scorer.json missing from bundle")
        }
        let data = try Data(contentsOf: url)
        self.model = try JSONDecoder().decode(LinearRegression.self, from: data)
    }
}

// Per-request scoring.
struct ScoreRequest: Content { let features: [Double] }
struct ScoreResponse: Content { let score: Double }

app.post("score") { request -> ScoreResponse in
    let input = try request.content.decode(ScoreRequest.self)
    let prediction = store.model.predict([input.features])[0]
    return ScoreResponse(score: prediction)
}
```

### Immutable values and many readers

One fitted model serves many concurrent requests. The request body decodes directly into `[Double]`, `predict()` consumes that same array, and the result encodes back to JSON. No intermediate representation sits between the wire format and the math; Vapor decodes into the type Quiver computes on. This is the practical reason the model is **immutable** after fitting: an immutable value can be read by many tasks at once without locks, because no task can change what another task is reading.

The same shape covers any scoring service the server needs to provide. A classifier returns a category, a regressor returns a number, and a clustering model returns a cohort label. Each is fit offline, encoded once, and decoded once. Multiple models can share the same store and live behind the same routes; the model changes, the request-handling shape does not. See <doc:Pipeline> for the end-to-end fit-and-predict workflow that round-trips a scaler and an estimator through `Codable` as one value, which the server can load at boot and never reassemble.

> Note: Quiver's fitted models are `Sendable` and immutable after `fit`. Many concurrent requests can read the same model in parallel with no locks, no copies, and no race conditions. The boot-time-load-immutable-share pattern is the entire story.

> Tip: A server has the compute to run every fitted model Quiver ships: `LinearRegression` and `Ridge` for prediction, `LogisticRegression` for binary scoring, `KNearestNeighbors` and `GaussianNaiveBayes` for classification, `KMeans` for grouping, and `GradientDescent` for the iterative fits underneath them. Each fits offline, encodes once, decodes at boot, and serves many concurrent readers through the same store.

## Where to go from here

The three sections above each have a deeper layer of math underneath them, and that math is the next step for server developers moving into numerical work. <doc:Statistics-Primer> builds the vocabulary of variance, distributions, and percentiles the latency-summary route leans on. <doc:Inferential-Statistics-Primer> covers standard error, t-distribution quantiles, and the confidence-interval framework the experiment-significance route depends on. <doc:Linear-Algebra-Primer> extends vectors and cosine similarity into matrices and projections, <doc:Semantic-Search> walks the end-to-end embedding-and-ranking pipeline the search route follows, and <doc:Embedding-Sources> shows how to swap the vector source behind one contract. <doc:Machine-Learning-Primer> closes the loop with features, labels, training, and the trade-offs that decide which model to fit offline and load at boot.

> Experiment: **The Quiver Notebook** is the right place to feel how a fitted model crosses the deploy boundary. Fit a `LinearRegression` in the Notebook, encode it to JSON, then decode the same file in a Vapor route and call `predict()` on a request body. The model on the server is the same model that ran in the Notebook; the boundary is one `JSONDecoder` call. See <doc:Quiver-Notebook>.
