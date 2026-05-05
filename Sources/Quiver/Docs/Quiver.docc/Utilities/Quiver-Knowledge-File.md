# Quiver Knowledge File

Pairing the Quiver reference with Claude as a study companion for the math and code.

## Overview

The Quiver knowledge file is special Markdown document that captures the Quiver's API surface, design principles, and validation status in one place. Uploaded into a conversation with Claude, it becomes a study companion — a place to ask questions about the math behind a method, the meaning of a result, or the reasoning behind a Quiver design choice. The book, the cookbook, and documentation is where Claude is taught.

### What Claude is

Claude is an AI assistant from [Anthropic](https://www.anthropic.com). It answers questions in plain language. When given a reference document — like the Quiver knowledge file — it grounds its answers in that document, which keeps responses accurate to the framework as it actually exists rather than to a general impression of how Swift numerical libraries usually work.

### How knowledge files work

Claude accepts uploaded files as context for a conversation. A markdown file dropped into a chat window is read, indexed, and used as the source of truth for any question that follows. Two patterns are common:

- **One-off conversation** — drag the knowledge file into a fresh chat, ask a question, get an answer. The context lives for the duration of that conversation.
- **Claude Project** — for longer study sessions or a full course, create a Project, attach the knowledge file once, and every conversation inside that Project reads the same reference automatically. This is the recommended pattern for anyone working through the cookbook or a primer over multiple sittings.

In both cases, the assistant's answers cite the parts of the file it draws from, so a learner can cross-check against the framework's actual API rather than trusting a paraphrase.

### Obtaining the file

The knowledge file lives in the main Quiver repository. Clone the repo and the file is in the `Docs/` folder:

```bash
git clone https://github.com/waynewbishop/quiver.git
```

The file is at `Docs/quiver-knowledge.md`. A new copy ships with each Quiver release, so re-cloning or pulling the latest version keeps the reference aligned with the API the other framework utilities.

### A statistics question

A learner working through the <doc:Statistics-Primer> hand-computes the standard deviation of a small array using a textbook formula and gets a different number than Quiver returns. They open Claude, attach the knowledge file, and ask:

> I read in a stats textbook that standard deviation divides by N − 1, but `data.std()` in Quiver divides by N. Why?

The companion explains that `std()` defaults to **population** standard deviation (`ddof: 0`), where N is the divisor, while the textbook formula is **sample** standard deviation, where the divisor is N − 1. Both are correct — they answer different questions. Population standard deviation treats the array as the complete data; sample standard deviation treats it as a draw from a larger population. Quiver exposes both:

```swift
let data = [4.0, 7.0, 2.0, 9.0, 3.0]
let pop = data.std() ?? 0           // 2.61 (population, ddof: 0)
let sample = data.std(ddof: 1) ?? 0 // 2.92 (sample)
```

The companion closes by pointing back to <doc:Statistics-Primer> for the conceptual treatment of when each form is appropriate.

### A linear algebra question

A learner reading the <doc:Linear-Algebra-Primer> computes a dot product, gets a single number back, and is not sure what that number is supposed to mean. They ask:

> I computed `[1, 2, 3].dot([4, 5, 6])` and got 32.0. What does that number actually mean?

The companion explains the dot product as a measure of how aligned two vectors are: `a · b = |a| · |b| · cos(θ)`. The result is large when the vectors point in similar directions and small when they are perpendicular. Working through the same arrays:

```swift
let a = [1.0, 2.0, 3.0]
let b = [4.0, 5.0, 6.0]
let dotProduct = a.dot(b)                   // 32.0
let cosineSim = a.cosineOfAngle(with: b)    // 0.974
```

A cosine of 0.974 means these vectors point in nearly the same direction. The dot product is high because the vectors are aligned, not just because the components are large. The companion points to <doc:Linear-Algebra-Primer> for the geometric foundations and to <doc:Similarity-Operations> for the recommendation-engine context where this calculation matters most.

### A machine learning question

A learner running a clustering example notices that `KMeans.fit` produces sensible groupings without ever being given target labels. They ask:

> I trained a K-Means model on some points and it produced clusters without me ever providing labels. How does it know what the groupings should be?

The companion explains that K-Means does not know what the groupings should be — it discovers structure by minimizing the distance between points and their assigned cluster center. The algorithm has three steps that repeat until assignments stop changing: place k centroids randomly, assign each point to its nearest centroid, then move each centroid to the mean of its assigned points. The contrast with <doc:Linear-Regression>, where every training example needs a target value, is what makes K-Means **unsupervised** — there are no answer keys, only structure the algorithm finds in the data itself:

```swift
let points: [[Double]] = [[1, 1], [1, 2], [2, 1], [8, 8], [8, 9], [9, 8]]
let model = KMeans.fit(data: points, k: 2, seed: 42)
print(model)
// KMeans: 2 clusters, 6 points, converged in 2 iterations
```

The companion points the learner to <doc:KMeans-Clustering> for the full algorithm and to <doc:Machine-Learning-Primer> for the broader supervised-versus-unsupervised distinction.

### Where the companion fits

The knowledge file works alongside the primary learning surfaces, each playing to its strength: the <doc:Quiver-Notebook> for runnable code, the <doc:Quiver-Cookbook> for worked recipes, and the documentation for method signatures. The companion answers questions across all three — a learner stuck on a primer paragraph, unsure why a recipe produced a particular number, or wanting a worked example phrased a different way can bring those questions to the companion and get a grounded explanation that points back to the primary resource where the topic lives.

### Privacy and what gets sent

The knowledge file and any questions a learner types are sent to Claude's servers for processing. This is different from the <doc:Quiver-Notebook>, which runs entirely on the local machine and sends nothing across the network. For classroom labs, exam settings, or air-gapped environments, the Notebook is the appropriate surface; the knowledge file pattern requires an internet connection and acceptance of the [Anthropic terms of service](https://www.anthropic.com/legal/consumer-terms).
