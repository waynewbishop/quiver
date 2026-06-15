# Embedding Sources

Connect Quiver's search surface to other embedding sources through a single contract.

## Overview

The `Embedder` protocol bridges text and ranked search. Just as <doc:Data-Visualization> prepares data for charting, `Embedder` separates vector creation from vector ranking. We own the ranking and reporting logic; you own the vector source.

We need a way to convert text into vectors. Quiver assembles this conversion manually (tokenize, look up, average), but the source is left open. A small word-vector table works for learning; a production app reaches for an on-device sentence model. The `Embedder` protocol defines a single operation—text in, vector out—allowing ranking methods to work against any source without modification.

The `Embedder` protocol names a single operation, text in and vector out, and lets every downstream method work against that operation rather than against any particular source. Conform once, and the ranking surface treats a hand-built dictionary and a Core AI model exactly alike.

> Note: This article builds on <doc:Semantic-Search>, which introduces tokenization, embedding lookup, and cosine similarity. The examples here are self-contained, but the vocabulary of vectors and similarity carries over.

## The embedder contract

An `Embedder` is any type that can turn a string into a fixed-dimension vector:

```swift
import Quiver

protocol Embedder: Sendable {
    func embed(_ text: String) -> [Double]?
}
```

The method returns `nil` when the text carries nothing to embed: an empty string, or one with no recognized tokens. That optional return matches the one `meanVector` already produces, so the two compose without special handling. The `Sendable` requirement lets an embedder cross task and actor boundaries, which matters the moment a model runs its inference off the main thread.

> Note: Quiver defines the contract but ships no embedder of its own. Where the vectors come from (a word-vector table, an on-device sentence model, or a custom model on Apple silicon) is ours to decide, and everything downstream stays the same.

## Conforming to an embedding source

Conformance is a single method. A word-vector embedder is the simplest case: it averages the vectors of the words it recognizes, reusing the `embed(using:)` lookup from the search pipeline:

```swift
import Quiver

struct TableEmbedder: Embedder {
    let table: [String: [Double]]

    func embed(_ text: String) -> [Double]? {
        // Split the text into the words the table is keyed on.
        let words = text.tokenize()
        // Look each word up, producing one vector per recognized word.
        let vectors = words.embed(using: table)
        // Collapse those many word vectors into a single document vector, since the table knows only words.
        return vectors.meanVector()
    }
}
```

Averaging token vectors is deliberately coarse. Because it sums and divides, it ignores word order entirely: "a long slow rise" and "a slow long rise" collapse to the same vector. Averaging is the right starting point for learning the pipeline and a serviceable baseline; a richer source that encodes order and context conforms through the very same method when we need it.

## Embedding a collection

Quiver adds `embedded(using:)` to `Array` where the element is `String`, so we call it on an array of documents, passing the embedder in. The method runs each string through that embedder and returns one `(text, vector)` pair per string it can embed. Pairing the text with its own vector is what keeps labels aligned and organized.

```swift
import Quiver

let embeddings: [String: [Double]] = [
    "rise":  [0.95, 0.10, 0.10],
    "slow":  [0.80, 0.20, 0.15],
    "long":  [0.75, 0.15, 0.20],
    "proof": [0.55, 0.45, 0.30],
    "yeast": [0.50, 0.55, 0.35],
    "knead": [0.15, 0.85, 0.20],
    "dough": [0.20, 0.90, 0.25],
    "well":  [0.30, 0.60, 0.40]
]

let docs = ["a long slow rise", "knead the dough well", "proof the yeast"]
let embedder = TableEmbedder(table: embeddings)

let embedded = docs.embedded(using: embedder)
// 3 pairs: each document with its averaged vector
```

Each pair holds the original text and the vector built from it. Documents the embedder cannot process drop out, and the rest keep their text attached. Keeping the text beside its vector is a deliberate choice, not a convenience: the vector is the searchable form, but it cannot be read back into the words that produced it. Pairing the two means what ranks is also what we read. There is no separate, readable copy to recover once a vector wins.

## Ranking against a query

Ranking is another array method, this time on the array of pairs that `embedded(using:)` returned. We call `mostSimilar(to:k:)` directly on `embedded` and pass the query vector in. The call ranks the pairs against that query in one pass: it scores every vector by cosine similarity, sorts, and returns the top matches as `(rank, text, score)`. The text is drawn from the same pair that produced the score:

```swift
// Using embedded and embedder from the previous example

if let query = embedder.embed("a slow long rise") {
    let hits = embedded.mostSimilar(to: query, k: 3)

    for hit in hits {
        print("#\(hit.rank) \(hit.text)  \(String(format: "%.3f", hit.score))")
    }
}
// #1 a long slow rise  1.000
// #2 proof the yeast  0.821
// #3 knead the dough well  0.460
```

The query "a slow long rise" scores a perfect 1.000 against "a long slow rise" — the same words in a different order. That perfect score is the order-blindness of averaging: the two phrases share every token, so their averaged vectors are identical. A source that encoded word order would tell them apart.

> Tip: For large collections, pre-compute and store the embedded pairs rather than rebuilding them for each query. Only the query vector needs to be built at search time.

## The embedder levels

A single contract spans the full range of embedding sources, from a hand-typed table to a custom model. Each level is a value of `some Embedder`, so the same `embedded(using:)` and `mostSimilar(to:k:)` calls serve every one:

| Level | Source | Character |
|---|---|---|
| 0 | Hand-built `[String: [Double]]` | Zero setup; every number inspectable |
| 1 | [GloVe](https://github.com/stanfordnlp/GloVe) word vectors with averaging | Teaching baseline; order-blind |
| 2 | [NLEmbedding](https://developer.apple.com/documentation/naturallanguage/nlembedding) sentence vectors | Production quality; returns a vector natively |
| 3 | Custom Core AI model on Apple silicon | Highest fidelity; vectors converted to `[Double]` |

> Important: The contract is the boundary. Code that ranks, stores, or reports results depends on `Embedder`, never on the source behind it. Swapping level 1 for level 3 changes the one line that constructs the embedder and nothing else.

## Where to go from here

The embedder is the swappable front of the search pipeline; the rest of that pipeline lives in <doc:Semantic-Search>, which shows how tokenization, embedding lookup, and cosine similarity fit together. For the vector operations underneath (dot products, magnitudes, and cosine similarity itself), see <doc:Vector-Operations> and <doc:Similarity-Operations>. The averaging step that builds a single document vector is one application of the descriptive statistics in <doc:Statistics-Primer>.

> Experiment: **The Quiver Notebook** is the right place to feel the swap that the contract enables. Load `Dataset.glove50d`, wrap it in a `TableEmbedder`, and rank a three-document corpus against a query with `embedded(using:)` then `mostSimilar(to:k:)`. Then change only the embedder (swap in a different table, or a richer source) and re-run the exact same ranking lines. The results shift while the pipeline stays still, and that gap between source and pipeline is the whole point of the contract. See <doc:Quiver-Notebook>.
