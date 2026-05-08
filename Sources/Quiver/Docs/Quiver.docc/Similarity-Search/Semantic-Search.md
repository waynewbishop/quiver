# Semantic Search

Implementing semantic search through tokenization, embeddings, and cosine similarity.

## Overview

Semantic search finds information by comparing **meaning** rather than matching keywords. A traditional keyword search for "running shoes" would miss a document titled "athletic footwear" because the words don't match — even though the concepts are nearly identical. Semantic search solves this by representing words and phrases as numeric [vectors](<doc:Linear-Algebra-Primer>), where similar meanings occupy nearby positions in multidimensional space. Comparing those vectors with cosine similarity reveals relationships that keyword matching cannot.

> Note: This article builds on concepts from <doc:Similarity-Operations> and <doc:Vector-Operations>. Familiarity with cosine similarity and the dot product will help, but the examples are self-contained.

## Words as vectors

Semantic search begins with the idea that words can be represented as arrays of numbers — vectors — where each dimension captures some aspect of the word's meaning. Words that share similar contexts in language end up with vectors that point in similar directions. The example below sketches a small vocabulary in which words related to athletics cluster together while unrelated words sit far apart:

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

The similarity score tells us "running" and "jogging" are nearly interchangeable, while "running" and "computer" share little in common. This is what makes semantic search possible — comparing vectors is effectively comparing meaning.

> Note: Cosine similarity measures **direction**, not magnitude — a word vector scaled to any length still produces the same similarity score. For details on why this matters, see the normalization discussion in <doc:Similarity-Operations>.

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

The `tokenize` method converts raw text into individual tokens — the input shape that the rest of the pipeline expects. It lowercases the source string, splits on whitespace, and strips punctuation, producing clean tokens that match embedding dictionary keys:

```swift
import Quiver

let query = "Comfortable Running Shoes"
let tokens = query.tokenize()
// ["comfortable", "running", "shoes"]

let review = "Great shoes! Lightweight, comfortable, and fast."
let reviewTokens = review.tokenize()
// ["great", "shoes", "lightweight", "comfortable", "and", "fast"]
```

Lowercasing ensures that "Running" and "running" map to the same vector. Punctuation stripping ensures that "shoes!" and "shoes," both match the dictionary key "shoes" — without this, punctuated words would silently miss their embeddings. Contractions like "don't" preserve their interior apostrophe.

To keep punctuation attached to tokens (for example, when token boundaries carry meaning), pass `strippingPunctuation: false`:

```swift
let raw = "Hello, world!".tokenize(strippingPunctuation: false)
// ["hello,", "world!"]
```

## Looking up embeddings

The `embed(using:)` method converts an array of tokens into an array of vectors. It looks up each token in a dictionary keyed by string and returns only the vectors for tokens it finds — unknown tokens are filtered out automatically:

> Tip: **The Quiver Notebook** ships 5,000 of the most-frequent English words from Stanford's GloVe corpus, each as a 50-dimensional vector. See <doc:Notebook-Datasets>.

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

Tokens that do not appear in the dictionary are silently skipped. Common words like "the," "for," and "is" often carry little semantic weight, so filtering them out can improve results. Because `tokenize` strips punctuation by default, input like "Comfortable, Running Shoes!" produces clean tokens that match dictionary keys directly — no manual cleanup required.

> Note: The `embed(using:)` method accepts any `[String: [Double]]` dictionary. How that dictionary is built — whether from a pre-trained model, a custom training pipeline, or a server-side API — is entirely up to the developer.

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

> Note: The `meanVector` method returns an optional — it returns `nil` if the array is empty or if vectors have inconsistent dimensions. See <doc:Statistical-Operations> for additional aggregation operations on vector collections.

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

The full pipeline chains five Quiver methods in order: `tokenize`, `embed(using:)`, `meanVector`, `cosineSimilarities(to:)`, and `topIndices(k:labels:)`.

> Tip: For large collections, pre-compute and store document vectors rather than recalculating them for each query. Only the query vector needs to be built at search time.

