# ``Quiver``

@Metadata {
    @PageKind(article)
    @SupportedLanguage(swift)
    @Available(iOS, introduced: "15.0")
    @Available(macOS, introduced: "12.0")
    @Available(tvOS, introduced: "15.0")
    @Available(watchOS, introduced: "8.0")
    @Available(visionOS, introduced: "1.0")
}

@Options(scope: local)

A Swift package for statistics, linear algebra, and machine learning.

## Overview

[Quiver](https://github.com/waynewbishop/quiver) is a pure-Swift package for statistics, linear algebra, and machine learning. The package extends the standard `Array` type, so its operations read as ordinary Swift and compose with the language a developer already knows. The mathematical surface (vectors, matrices, statistics, and models) works the same on iOS, macOS, watchOS, and the server.

### Data science in Swift

As Swift expands beyond app development into server-side computing, machine learning, and data analysis, it needs mathematical tools to match. Quiver is the numerical foundation for those workflows, covering the operations that fields like machine learning, semantic search, computer vision, signal processing, and scientific computing depend on.

### Why Quiver

Quiver provides developers the tools to work with data directly. This includes analyzing data with statistics, transforming it with linear algebra, preparing it for modeling, training machine learning models, or building retrieval pipelines. As a pure Swift library with zero external dependencies, Quiver runs on every Apple platform (iOS, macOS, watchOS, tvOS, and visionOS) as well as server-side Swift with frameworks like Vapor, Linux environments, and containerized deployments. As a lightweight framework, Quiver is ideal for [teaching environments](<doc:Quiver-Notebook-For-Classrooms>), on-device processing, and any context where minimal dependencies and platform portability matter.

## Topics

### Essentials
- <doc:Installation>
- <doc:Basics>
- <doc:How-It-Works>

### Conceptual Guides

- <doc:Numerical-Literacy>
- <doc:Rendering-Math-Primer>
- <doc:Linear-Algebra-Primer>
- <doc:Determinants-Primer>
- <doc:Eigenvalues-Primer>
- <doc:Machine-Learning-Primer>
- <doc:Statistics-Primer>
- <doc:Probability-Primer>
- <doc:Central-Limit-Theorem>
- <doc:Inferential-Statistics-Primer>
- <doc:Calculus-Primer>
- <doc:Optimization-Primer>
- <doc:Regularization-Primer>
- <doc:Model-Interpretation-Primer>
- <doc:Concurrency-Primer>

### Platform Guides
- <doc:iOS-Apps>
- <doc:watchOS-Apps>
- <doc:Vapor-Server>

### Vectors
- <doc:Vector-Operations>
- <doc:Vector-Projections>

### Matrices
- <doc:Shape-And-Size>
- <doc:Matrix-Operations>
- <doc:Broadcasting-Operations>
- <doc:Array-Generation>
- <doc:Random-Number-Generation>

### Transformations
- <doc:Matrix-Transformations>
- <doc:Composing-Transformations>

### Signal Processing
- <doc:Fourier-Transform>
- <doc:Physics-Primitives-Primer>

### Statistics
- <doc:Frequency-Tables>
- <doc:Boolean-Masking>
- <doc:Working-With-Distributions>
- <doc:Identifying-A-Distribution>
- <doc:Correlation>
- <doc:Random-Sampling>

### Data Preparation
- <doc:Working-With-Panels>
- <doc:Data-Visualization>
- <doc:Train-Test-Split>
- <doc:Feature-Scaling>
- <doc:Working-With-Pipelines>

### Similarity and Search
- <doc:Text-Tokenization>
- <doc:Similarity-Operations>

### Models
- <doc:Linear-Regression>
- <doc:Regression-Summary>
- <doc:Gradient-Descent>
- <doc:Ridge-Regression>
- <doc:Residual-Model>
- <doc:Polynomials>
- <doc:Naive-Bayes>
- <doc:Logistic-Regression>
- <doc:Nearest-Neighbors-Classification>
- <doc:KMeans-Clustering>
- <doc:Principal-Component-Analysis>
- <doc:Activation-Functions>
- <doc:Evaluation-Metrics>
- <doc:Model-Persistence>

### Worked Examples
- <doc:Semantic-Search>
- <doc:Embedding-Sources>
- <doc:Retrieving-Context-For-Generation>
- <doc:Panel-Workflows>

### Utilities
- <doc:Quiver-Notebook>
- <doc:Quiver-Notebook-For-Classrooms>
- <doc:Notebook-Datasets>
- <doc:Xcode-Playground>
- <doc:Quiver-Cookbook>
- <doc:Quiver-Knowledge-File>
- <doc:Working-With-Fractions>
