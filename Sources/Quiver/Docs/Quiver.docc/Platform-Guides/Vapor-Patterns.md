# Vapor Patterns

Building semantic-search and CRUD endpoints over a precomputed vector catalog with Quiver and Vapor.

## Overview

A Vapor server that hosts a vector catalog has the same shape as almost any small database service. Clients ask for the list of items, add new ones, remove the ones they no longer want, and search for the items closest to whatever query they have in mind. What makes the Quiver version different is what happens at search time: instead of matching on exact words, the server ranks by semantic similarity — a query for "cushioned long run" finds shoes whose descriptions never used those exact words but match the meaning.

This article walks through the four endpoints that make up that service — list, create, delete, and semantic search — and the small amount of vector math that turns text into a ranked answer. The <doc:Vapor-Guide> covers the foundations these patterns build on.

### Semantic-search endpoint

The clearest pattern for Quiver on Vapor is a semantic-search index. A collection of items — products, articles, notes, anything textual — is tokenized and turned into vectors when the server starts. The resulting document vectors stay in memory for the life of the server, and every search request decodes a query, turns it into a vector the same way, ranks it against the precomputed collection, and returns the top matches.

The pipeline is five Quiver methods chained together:

```swift
import Vapor
import Quiver

app.get("search") { request -> [SearchResult] in
    guard let query = request.query[String.self, at: "q"] else {
        throw Abort(.badRequest, reason: "Missing query parameter ?q=")
    }

    // Tokenize the query → embed each token → reduce to one vector
    let tokens = query.tokenize()
    let vectors = tokens.embed(using: embeddings)
    guard let queryVector = vectors.meanVector() else {
        return []
    }

    // Rank the query vector against precomputed document vectors
    let productVectors = store.shoes.map(\.vector)
    let similarities = productVectors.cosineSimilarities(to: queryVector)
    let top = similarities.topIndices(k: 5, labels: store.shoes)

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

The full embedding pipeline is covered in <doc:Semantic-Search>. What changes when it lives behind HTTP is where each step runs: tokenization happens twice (once when documents are loaded at boot, once per query), document vectors are precomputed and held in memory, and only the query side runs inside the request handler. The handler's work is decode → tokenize → embed → reduce → rank — every step is a pure read against precomputed data.

What this pattern is and what it is not: it is a search index over precomputed embeddings, optimized for fast reads. It is not a retraining loop. The catalog changes through a separate endpoint, never inside the search handler. For the underlying operations, see <doc:Similarity-Operations>.

### Adding entries with vectors at creation time

New entries enter the catalog through a creation endpoint that takes a description, embeds it on receipt, and stores both the text and the vector. The embedding step is the same five Quiver methods used by search, just running once at create time instead of once per query.

```swift
import Vapor
import Quiver

struct AddRequest: Content { let description: String }

app.post("products") { request throws -> HTTPStatus in
    let input = try request.content.decode(AddRequest.self)

    // Embed the new description with the same pipeline search uses
    let tokens = input.description.tokenize()
    let vectors = tokens.embed(using: embeddings)
    guard let vector = vectors.meanVector() else {
        throw Abort(.unprocessableEntity, reason: "Description produced no embeddable tokens.")
    }

    store.add(Shoe(description: input.description, vector: vector))
    return .created
}
```

The pattern keeps the catalog and its vector representation in sync. Every entry that enters the store enters with its vector already computed, so search never has to embed a stored document at query time. The cost of embedding is paid once at creation, not once per future search.

### Managing catalog membership through removal

Removal closes the CRUD loop without touching the vector math. The catalog is keyed by description (or by an `id` field in production-shaped applications); the handler looks up the matching entry and removes it.

```swift
import Vapor

app.delete("products", ":description") { request -> HTTPStatus in
    guard let description = request.parameters.get("description") else {
        throw Abort(.badRequest, reason: "Missing path parameter.")
    }
    guard store.remove(description: description) else {
        throw Abort(.notFound, reason: "No matching entry.")
    }
    return .ok
}
```

The `store.remove(description:)` call mutates the shared catalog. As discussed in <doc:Vapor-Guide>'s "Sharing fitted state" section, this is the kind of mutation that requires the `@unchecked Sendable` annotation on the store type — Vapor's async context serializes the writes, and the read-heavy search path stays unaffected.

For a fourth, read-only endpoint — listing the current catalog — the handler simply returns `store.shoes.map(\.description)` without touching any vectors. The same store backs all four endpoints; the same `Sendable` value-sharing pattern keeps them safe under concurrent load.

### When the in-memory catalog stops scaling

This is a vector catalog with CRUD endpoints, optimized for fast reads and occasional writes. It is not a database. There is no persistence across server restarts, no transaction guarantees, no replication. For applications where the catalog needs to survive a restart or grow beyond what fits in memory, the same Quiver pipeline runs unchanged in front of a real database — the embedding and ranking math stay in the Vapor process; the storage moves to PostgreSQL, SQLite, or whatever the application already uses.

> Experiment: The patterns in this article are from [quiver-demo-vapor](https://github.com/waynewbishop/quiver-demo-vapor), a semantic-search server that scores fifteen shoes against a six-dimension hand-built embedding dictionary. Running the four endpoints — list, create, delete, search — in sequence against the same in-memory store, then comparing two queries that differ by one word, shows how `cosineSimilarities` and `topIndices` collapse free text into a ranked answer.
