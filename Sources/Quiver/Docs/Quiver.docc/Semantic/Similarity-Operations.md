# Similarity Operations

Compute similarity between vectors using cosine similarity and distance metrics.

## Overview

Similarity operations measure how **related** two vectors are. These operations are fundamental for machine learning applications including recommendation systems, word prediction, clustering, <doc:Semantic-Search>, and nearest neighbor classification.

### Dot product

The dot product computes the sum of element-wise products between two vectors. It's the fundamental operation underlying cosine similarity and many machine learning algorithms.

```swift
let v1 = [2.0, 3.0, 4.0]
let v2 = [1.0, 2.0, 3.0]

let dotProduct = v1.dot(v2)
// 20.0 = (2×1) + (3×2) + (4×3)
```

**Mathematical definition:**
```
dot(v, w) = v₁w₁ + v₂w₂ + ... + vₙwₙ
```

### Relationship to other operations

The dot product is the foundation for other similarity metrics:

```swift
let v1 = [3.0, 4.0]
let v2 = [5.0, 12.0]

// Raw dot product
let dot = v1.dot(v2)  // 63.0

// Cosine similarity: normalized dot product
let cosine = v1.cosineOfAngle(with: v2)
// dot / (||v1|| × ||v2||) = 63.0 / (5.0 × 13.0) = 0.969
```

### Magnitude vs distance

Both magnitude and [Euclidean distance](<doc:Linear-Algebra-Primer>) use the Pythagorean theorem, but measure different things. `Magnitude` provides an answer to "how far am I from home (origin)" while Euclidean distance solves "how far is the coffee shop from the library".

```swift
// Magnitude: distance from origin 
let v = [3.0, 4.0]
let mag = v.magnitude  // 5.0 = sqrt(3² + 4²)

// Euclidean distance: distance between any two vectors
let v1 = [1.0, 2.0]
let v2 = [4.0, 6.0]
let dist = v1.distance(to: v2)  // 5.0 = sqrt((4-1)² + (6-2)²)

// Magnitude is a special case - measurement from origin
let equivalentDist = [0.0, 0.0].distance(to: v)  // 5.0 (same as magnitude)
```

This distinction matters for cosine similarity, which normalizes by dividing by the product of magnitudes (`||v1|| × ||v2||`).

### Normalization and dot product

Normalization transforms the dot product into a pure directional similarity measure. Without it, the dot product mixes alignment with magnitude, making comparisons unreliable. Two vectors pointing identically but with different lengths produce vastly different dot products.

```swift
let v1 = [3.0, 4.0]
let v2 = [6.0, 8.0]  // Same direction, 2× longer

v1.dot(v2)  // 50.0

// Scale both by 10×
let v3 = [30.0, 40.0]
let v4 = [60.0, 80.0]

v3.dot(v4)  // 5000.0 (100× larger, same direction!)

// Cosine similarity normalizes to measure pure direction
v1.cosineOfAngle(with: v2)  // 1.0
v3.cosineOfAngle(with: v4)  // 1.0 (consistent)
```

While `cosineOfAngle` is determined based on normalized vectors, we can also calculate a unit vector using the `normalized` property.

```swift
let v1 = [3.0, 4.0]

// Create a unit vector (same direction, length of 1)
let unitVector = v1.normalized  // [0.6, 0.8]
```

> Tip: For normalized vectors (magnitude = 1), dot product equals cosine similarity.

## Cosine similarity

Cosine similarity measures the angle between vectors, ranging from -1 (opposite) to 1 (identical). It focuses on direction rather than magnitude.

```swift
let v1 = [0.8, 0.6, 0.0]
let v2 = [0.4, 0.3, 0.0]

let similarity = v1.cosineOfAngle(with: v2)
// 1.0 - identical direction despite different magnitudes
```

**Mathematical definition:**
```
cosine_similarity(v, w) = (v · w) / (||v|| × ||w||)
```

> Important: `cosineOfAngle(with:)` returns `0.0` if either vector has zero magnitude. Check for zero vectors before interpreting results.

### When to use cosine similarity

Cosine similarity is the right choice when direction matters more than magnitude. In text analysis, it ensures that a long document and a short document on the same topic score as similar. In recommendation systems, it compares user preference vectors regardless of how many items each user has rated. In classification, `KNearestNeighbors` supports cosine distance (`1 − cosine similarity`) as a metric for finding nearest neighbors in high-dimensional spaces like text embeddings — see <doc:Nearest-Neighbors-Classification>. In clustering, `KMeans` uses Euclidean distance by default, but cosine similarity can pre-filter or validate cluster coherence.

### From similarity to angle

Cosine similarity and angle measurement are the same algorithm at different stages. The `cosineOfAngle(with:)` method returns the raw cosine value — the number between -1 and 1 used for similarity comparisons. The `angle(with:)` method applies `acos()` to that value to produce the actual angle in radians, and `angleInDegrees(with:)` converts to degrees:

```swift
let v1 = [3.0, 4.0]
let v2 = [5.0, 12.0]

v1.cosineOfAngle(with: v2)    // 0.969 (similarity score)
v1.angle(with: v2)            // 0.248 radians
v1.angleInDegrees(with: v2)   // 14.23 degrees
```

In machine learning and information retrieval, the raw cosine value is typically all that's needed — "document A is 0.92 similar to document B." In physics and graphics, the actual angle matters — "rotate 45 degrees" or "the force acts at 30 degrees." See <doc:Vector-Operations> for more on angle calculations and <doc:Vector-Projections> for decomposing vectors into parallel and perpendicular components.

### Range interpretation

- `1.0`: Identical direction (very similar)
- `0.5-0.8`: Related
- `0.0`: Orthogonal (unrelated)
- `-1.0`: Opposite direction

## Batch operations

Compare one vector against many efficiently:

```swift
let query = [0.8, 0.7, 0.9]

let database = [
    [0.8, 0.6, 0.9],  // Vector 1
    [0.2, 0.3, 0.1],  // Vector 2
    [0.7, 0.7, 0.8]   // Vector 3
]

// Compute all similarities at once
let similarities = database.cosineSimilarities(to: query)
// [0.99, 0.88, 0.99]
```

**Result preservation:** `similarities[i]` is the similarity between `database[i]` and `query`.

## Common use cases

### Recommendation systems

Compare user preference vectors against item vectors to suggest relevant content. Higher similarity scores indicate better matches for personalized recommendations.

```swift
// User's preference vector
let userProfile = [0.8, 0.3, 0.9, 0.2]

// Item vectors
let items = [[0.7, 0.4, 0.8, 0.3], [0.2, 0.9, 0.1, 0.7]]

// Find similar items
let scores = items.cosineSimilarities(to: userProfile)
// [0.99, 0.45] - first item matches user preferences
```

### Duplicate detection

Identify near-duplicate documents or data by computing pairwise similarities and filtering pairs above a threshold. Quiver provides a built-in method to efficiently find redundant content in large datasets.

```swift
let documents = [
    [0.8, 0.7, 0.9],
    [0.81, 0.69, 0.91],  // Near-duplicate of document 0
    [0.1, 0.2, 0.1]
]

// Find near-duplicates with default threshold (0.95)
let duplicates = documents.findDuplicates()

// Or specify custom threshold
let nearDuplicates = documents.findDuplicates(threshold: 0.90)

// Results are sorted by similarity (highest first)
duplicates.forEach { result in
    print("Documents \(result.index1) and \(result.index2) are \(Int(result.similarity * 100))% similar")
}
```

### Clustering validation

Measure cluster cohesion by calculating average pairwise similarity within a group. Higher values indicate tighter, more homogeneous clusters.

```swift
let cluster = [
    [0.8, 0.7, 0.9],
    [0.7, 0.8, 0.8],
    [0.9, 0.6, 0.9]
]

// Calculate cluster cohesion
let cohesion = cluster.clusterCohesion()  // 0.0 to 1.0

// Compare different clusters
let technicalDocs = [[0.8, 0.3, 0.9], [0.7, 0.4, 0.8]]
let sportsDocs = [[0.2, 0.9, 0.1], [0.3, 0.8, 0.2]]

let techCohesion = technicalDocs.clusterCohesion()
let sportsCohesion = sportsDocs.clusterCohesion()

// Higher cohesion = better clustering quality
print("Technical cluster quality: \(Int(techCohesion * 100))%")
print("Sports cluster quality: \(Int(sportsCohesion * 100))%")
```

### Semantic search

Find relevant content by comparing the meaning of a query against a collection of documents. Unlike keyword search, semantic search surfaces results based on conceptual similarity — a query for "running shoes" can match documents about "athletic footwear" if their embeddings are close.

```swift
import Quiver

// Query and document embeddings (pre-computed from a language model)
let query = [0.8, 0.7, 0.2]

let documents = [
    [0.7, 0.6, 0.3],  // "Athletic Footwear Guide"
    [0.1, 0.1, 0.9],  // "Best Cooking Recipes"
    [0.8, 0.8, 0.2]   // "Running Shoe Reviews"
]

// Rank all documents by similarity to the query
let scores = documents.cosineSimilarities(to: query)
let results = scores.topIndices(k: 2, labels: ["Athletic Footwear", "Cooking Recipes", "Running Shoes"])

// results: [(rank: 1, label: "Running Shoes", score: 1.0),
//           (rank: 2, label: "Athletic Footwear", score: 0.99)]
```

The `topIndices(k:labels:)` method pairs each score with its original label and a 1-based rank, making it straightforward to map similarity results back to content.

> Tip: For a complete pipeline that starts from raw text — including tokenization, embedding lookup, and document vector averaging — see <doc:Semantic-Search>.

## See also

- <doc:Semantic-Search> - Full text-to-results pipeline using tokenization, embeddings, and similarity
- <doc:Vector-Operations> - Vector operations
- <doc:Matrix-Operations> - Matrix operations

## Topics

### Basic operations
- ``Swift/Array/dot(_:)``

### Similarity metrics
- ``Swift/Array/cosineOfAngle(with:)``
- ``Swift/Array/distance(to:)``

### Batch operations
- ``Swift/Array/cosineSimilarities(to:)->[Double]``

### Similarity analysis
- ``Swift/Array/findDuplicates(threshold:)``
- ``Swift/Array/clusterCohesion()``
