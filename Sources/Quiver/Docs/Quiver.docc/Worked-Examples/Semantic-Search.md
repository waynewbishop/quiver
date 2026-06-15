# Semantic Search

Implementing semantic search through tokenization, embeddings, and cosine similarity.

## Overview

Semantic search finds information by comparing **meaning** rather than matching keywords. A traditional keyword search for "running shoes" would miss a document titled "athletic footwear" because the words don't match, even though the concepts are nearly identical. Semantic search solves this by representing words and phrases as numeric [vectors](<doc:Linear-Algebra-Primer>), where similar meanings occupy nearby positions in multidimensional space. Comparing those vectors with cosine similarity reveals relationships that keyword matching cannot.

> Note: This article builds on concepts from <doc:Similarity-Operations> and <doc:Vector-Operations>. Familiarity with cosine similarity and the dot product will help, but the examples are self-contained.

## Words as vectors

Semantic search begins with the idea that words can be represented as arrays of numbers (vectors) where each dimension captures some aspect of the word's meaning. Words that share similar contexts in language end up with vectors that point in similar directions. The example below sketches a small vocabulary in which words related to athletics cluster together while unrelated words sit far apart:

> Important: The vector values throughout this article are hypothetical, chosen to illustrate the math. In practice, these values could come from a trained language model, a domain-specific dataset (e.g. home prices versus property features), or any source that maps items to numeric vectors.

```swift
import Quiver

// Hypothetical word vectors (4 dimensions for illustration)
let running  = [0.8, 0.7, 0.9, 0.2]
let jogging  = [0.8, 0.7, 0.8, 0.2]
let computer = [0.1, 0.3, 0.2, 0.9]

// Measure directional similarity between word vectors
running.cosineOfAngle(with: jogging)   // ~0.99 (near-synonyms)
running.cosineOfAngle(with: computer)  // ~0.47 (unrelated concepts)
```

The similarity score tells us "running" and "jogging" are nearly interchangeable, while "running" and "computer" share little in common. This is what makes semantic search possible: comparing vectors is effectively comparing meaning.

> Note: Cosine similarity measures **direction**, not magnitude: a word vector scaled to any length still produces the same similarity score. For details on why this matters, see the normalization discussion in <doc:Similarity-Operations>.

## Vector arithmetic captures relationships

Word vectors do more than encode similarity — they also encode relationships that show up under arithmetic. Subtracting one vector from another isolates the dimensions that differ between them; adding a third vector reintroduces a different attribute. This is what makes the classic `king − man + woman ≈ queen` analogy hold.

```swift
import Quiver

// Hypothetical word vectors encoding semantic properties
let king   = [0.9, 0.2, 0.8, 0.7]   // Male royalty
let man    = [0.8, 0.1, 0.2, 0.6]   // Male concept
let woman  = [0.2, 0.8, 0.2, 0.6]   // Female concept
let queen  = [0.3, 0.9, 0.8, 0.7]   // Female royalty

// Vector arithmetic: king - man + woman ≈ queen
let result = king.subtract(man).add(woman)
// result = [0.3, 0.9, 0.8, 0.7]

// Confirm the result vector aligns with the expected word
result.cosineOfAngle(with: queen)  // ~1.0
result.cosineOfAngle(with: king)   // ~0.79
```

The subtraction isolates the "royalty" component by removing the "male" direction. Adding the woman vector reintroduces a gender direction, landing at "female royalty." Word vectors encode semantic properties as geometric directions, and similarity operations measure those directions directly.

> Note: The element-wise arithmetic shown here relies on Quiver's vector methods. See <doc:Vector-Operations> for the full set of vector operations available on arrays.

## Tokenizing text

The first step in any text pipeline converts raw text into individual words. Quiver's `tokenize` method lowercases, splits on whitespace, and removes punctuation, producing clean tokens that match embedding dictionary keys:

```swift
import Quiver

let query = "Comfortable Running Shoes"
let tokens = query.tokenize()
// ["comfortable", "running", "shoes"]

let review = "Great shoes! Lightweight, comfortable, and fast."
let reviewTokens = review.tokenize()
// ["great", "shoes", "lightweight", "comfortable", "and", "fast"]
```

Lowercasing ensures that "Running" and "running" map to the same vector. Punctuation removal ensures that "shoes!" and "shoes," both match the dictionary key "shoes." Without this, punctuated words would silently miss their embeddings. Contractions like "don't" preserve their interior apostrophe.

To keep punctuation attached to tokens (for example, when token boundaries carry meaning), pass `removingPunctuation: false`:

```swift
let raw = "Hello, world!".tokenize(removingPunctuation: false)
// ["hello,", "world!"]
```

## Looking up embeddings

The `embed(using:)` method converts an array of tokens into an array of vectors. The method looks up each token in a dictionary keyed by string and returns only the vectors for tokens it finds. Unknown tokens are filtered out automatically:

> Tip: **The Quiver Notebook** ships 25,000 of the most-frequent English words from Stanford's GloVe corpus, each as a 50-dimensional vector. See <doc:Notebook-Datasets>.

```swift
import Quiver

// Word vectors — the source is up to the developer
let embeddings: [String: [Double]] = [
    "comfortable": [0.7, 0.8, 0.3, 0.1],
    "running":     [0.8, 0.7, 0.9, 0.2],
    "shoes":       [0.6, 0.9, 0.4, 0.1],
    "lightweight": [0.5, 0.6, 0.3, 0.2],
    "trail":       [0.4, 0.3, 0.8, 0.7],
    "sneakers":    [0.6, 0.8, 0.5, 0.2],
    "outdoor":     [0.3, 0.2, 0.7, 0.8],
    "training":    [0.7, 0.6, 0.8, 0.3]
]

let queryTokens = "Comfortable Running Shoes".tokenize()
let queryVectors = queryTokens.embed(using: embeddings)
// [[0.7, 0.8, 0.3, 0.1],   comfortable
//  [0.8, 0.7, 0.9, 0.2],   running
//  [0.6, 0.9, 0.4, 0.1]]   shoes
```

Words like "for" that don't appear in the dictionary are silently skipped. Common words like "the," "for," and "is" often carry little semantic weight, so filtering them out can improve results. Because `tokenize` removes punctuation by default, input like "Comfortable, Running Shoes!" produces clean tokens that match dictionary keys directly, with no manual cleanup required.

> Note: The `embed(using:)` method accepts any `[String: [Double]]` dictionary. How that dictionary is built (whether from a pre-trained model, a custom training pipeline, or a server-side API) is entirely up to the developer.

## Searching the embedding dictionary

Once an embedding dictionary is in hand, `nearest(to:k:)` ranks every word against a query vector and returns the top matches. The method composes cosine similarity, sorting, and ranking into a single call: the same operation that would otherwise take three or four chained calls.

```swift
import Quiver

let king  = [0.9, 0.2, 0.8, 0.7]
let queen = [0.3, 0.9, 0.8, 0.7]
let man   = [0.8, 0.1, 0.2, 0.6]
let woman = [0.2, 0.8, 0.2, 0.6]

let embeddings: [String: [Double]] = [
    "king":  king,
    "queen": queen,
    "man":   man,
    "woman": woman
]

// king - man + woman should land closest to queen
let target = king.subtract(man).add(woman)

let results = embeddings.nearest(to: target, k: 2)
// [(rank: 1, word: "queen", score: 1.0),
//  (rank: 2, word: "king",  score: 0.79)]
```

Each result carries a 1-based rank, the matching word, and the cosine similarity score. Entries whose vector dimension does not match the query are skipped silently, which makes the method forgiving when the dictionary mixes embeddings from different sources.

> Tip: For ranked search across an array of document vectors rather than a string-keyed dictionary, use `cosineSimilarities(to:)` followed by `topIndices(k:labels:)`. The dictionary form is a convenience for the embedding-table case where the keys are already the labels. When the vectors come from an ``Embedder`` rather than a literal table, rank the `(text, vector)` pairs it produces with `mostSimilar(to:k:)` instead. See <doc:Embedding-Sources>.

## Building document vectors

The `meanVector` method collapses an array of token vectors into a single vector by averaging element-by-element. Each document is now represented by one vector that blends the meaning of every recognized token:

```swift
// Using queryVectors from the previous example

guard let documentVector = queryVectors.meanVector() else {
    return  // No recognized words found
}
// [0.7, 0.8, 0.533, 0.133]
//  ↑ element-wise average of comfortable, running, and shoes vectors
```

The averaged vector blends the athletic meaning of "running" with the product meaning of "shoes" and the quality meaning of "comfortable." Documents with similar averages will score highest.

> Note: The `meanVector` method returns an optional: it returns `nil` if the array is empty or if vectors have inconsistent dimensions. See <doc:Statistics-Primer> for the broader vocabulary of descriptive statistics the same `[Double]` columns support.

## Ranking results

The `cosineSimilarities(to:)` method scores every document vector against a query vector in one call. Pairing it with `topIndices(k:labels:)` returns the highest-scoring documents in ranked order:

```swift
// Using queryVector and docVectors built from the previous steps

// Rank all documents by similarity to the query
let scores = docVectors.cosineSimilarities(to: queryVector)
let results = scores.topIndices(k: 2, labels: labels)

for result in results {
    print("#\(result.rank) \(result.label): \(String(format: "%.1f%%", result.score * 100)) match")
}
```

The full pipeline chains five Quiver methods in order: `tokenize`, `embed(using:)`, `meanVector`, `cosineSimilarities(to:)`, and `topIndices(k:labels:)`. For a reusable, swappable source of those vectors, see <doc:Embedding-Sources>.

> Tip: For large collections, pre-compute and store document vectors rather than recalculating them for each query. Only the query vector needs to be built at search time.

> Experiment: **The Quiver Notebook** is the right place to watch search-by-meaning surface in real time. Load `Dataset.glove50d`, build a tiny three-document corpus, and rank the docs against a query phrase. Then edit one document (add a synonym, remove a word, replace a noun with a related one) and re-run. The ranking shifts in a way keyword matching would never produce, and that shift is the whole point of embedding-based search. See <doc:Quiver-Notebook>.

