# Basics

@Metadata {
  @TitleHeading("Essentials")
}

A starting point for Swift developers new to numerical computing.

## Overview

If we can write Swift, we already have most of what we need to work with data. The intimidating part of numerical computing is rarely the code — it's the vocabulary that surrounds it. Variance, regression, eigenvalue, normalization, gradient. These words sound like a different field of study, but every one of them describes something we can compute, see, and verify in a few lines of Swift.

This page is a starting point organized around three pillars that show up in any data work: **statistics** (describing what is in the data), **linear algebra** (treating numbers as positions in space), and **machine learning** (using the first two to make predictions). Each section teaches the smallest useful idea, then points to a deeper guide when we want to go further.

### Statistics — describing what we have

Most numerical work starts with a set of numbers and a need to describe them. The answer is a small handful of summaries — the average, the typical spread around that average, the unusual values worth flagging. Quiver computes all of these as methods on `[Double]`.

```swift
import Quiver

let responseTimes = [120.0, 145.0, 160.0, 175.0, 180.0, 195.0, 210.0, 320.0]

responseTimes.mean()       // 188.125 — the average response time
responseTimes.std()        // 56.4   — typical distance from the mean
responseTimes.median()     // 177.5  — the middle value
responseTimes.outlierMask(threshold: 2.0)  // [false, false, ..., true] — flags 320.0
```

Mean tells us the center. Standard deviation tells us how spread out the values are. Median is a more robust center when there are outliers. The outlier mask flags values that sit far from the rest. A dashboard, a health summary, a feed that highlights the unusual entry — all of them are built from these four ideas.

For the full vocabulary — variance, quartiles, percentiles, z-scores, hypothesis testing — see <doc:Statistics-Primer>. For the complete API, see <doc:Statistical-Operations>.

### Linear algebra — numbers as positions

A list of numbers can also be thought of as a single point in space. `[3.0, 4.0]` is a point in two dimensions. `[3.0, 4.0, 5.0]` is a point in three. A user's preferences across ten categories is a point in ten dimensions. We cannot picture ten dimensions, but the math that compares two points works exactly the same way it does in two.

```swift
import Quiver

let userA = [4.0, 5.0, 1.0, 2.0]   // ratings for four products
let userB = [5.0, 4.0, 1.0, 3.0]   // another user

userA.dot(userB)                  // 47.0 — sum of element-wise products
userA.cosineOfAngle(with: userB)  // 0.97 — directional alignment, 1.0 = identical
```

The dot product combines two vectors into a single number. Cosine similarity normalizes that number to a value between -1 and 1 that measures *direction*, not size. This single operation is the engine behind recommendation systems, semantic search, and similarity ranking. Two products with similar ratings have a cosine close to 1; two products with opposite ratings have a cosine close to -1.

For the full picture — vectors, matrices, transformations, distance, projection — see <doc:Linear-Algebra-Primer>. For the operations themselves, see <doc:Vector-Operations>.

### Machine learning — predicting from examples

Machine learning sounds like a separate discipline, but at the level we usually need it, it is one of three patterns applied to data we have already prepared. Given a set of examples, predict a number (regression). Given a set of labeled examples, predict a category (classification). Given a set of unlabeled examples, find the natural groupings (clustering).

```swift
import Quiver

// Regression — predict weight from height
let heights = [165.0, 170.0, 175.0, 180.0, 185.0]
let weights = [60.0, 68.0, 75.0, 82.0, 90.0]

let model = try LinearRegression.fit(features: heights, targets: weights)
model.predict([172.0])   // [70.56] — predicted weight for a 172cm person
```

Every Quiver model follows the same shape: `fit` takes the training data, `predict` takes new inputs and returns answers. `LinearRegression` predicts a number, `KNearestNeighbors` predicts a category, `KMeans` finds groupings. The choice between them depends on what we are trying to answer, not on a library to learn — they all read from the same `[Double]` arrays the rest of Quiver uses.

For the full conceptual frame — features, labels, training, evaluation, overfitting — see <doc:Machine-Learning-Primer>. For the individual models, see <doc:Linear-Regression>, <doc:Nearest-Neighbors-Classification>, and <doc:KMeans-Clustering>.

### Where to go from here

The three pillars above sit on top of a few practical concerns that come up the moment we work with real data. Floating-point arithmetic does not always behave the way decimal math does — `0.1 + 0.2` is not exactly `0.3`, and <doc:Numerical-Literacy> covers when to worry and when not to. Turning Quiver's output into a Swift Chart is the next step for most apps, and <doc:Data-Visualization> shows the patterns. For working through any of this interactively while we learn, <doc:Usage> walks through the `#Playground` macro that lets us see results inline as we type.
