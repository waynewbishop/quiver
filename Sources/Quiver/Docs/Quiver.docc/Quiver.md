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

[Quiver](https://github.com/waynewbishop/quiver) expands the Swift ecosystem with a pure, Swift-first approach to statistics, linear algebra, and machine learning. By building directly on Swift's powerful type system and syntax, Quiver creates an intuitive bridge between traditional array operations and advanced mathematical concepts. Built as an extension on the standard `Array` type, the framework embraces Swift's emphasis on readability and expressiveness, offering mathematical operations that feel natural to iOS and macOS developers.

### Data science in Swift

As Swift continues to expand beyond app development into domains like server-side computing, machine learning, and data analysis, the need for robust mathematical tools becomes increasingly important. Quiver serves as a foundation for data science workflows in Swift, enabling operations that are fundamental to fields like computer vision, game development, machine learning, and scientific computing.

### Why Quiver

Quiver provides developers the tools to work with data directly. This includes analyzing data with statistics, transforming it with linear algebra, preparing it for modeling, or training machine learning models. As a pure Swift library with zero external dependencies, Quiver runs on every Apple platform — iOS, macOS, watchOS, tvOS, and visionOS — as well as server-side Swift with frameworks like Vapor, Linux environments, and containerized deployments. As a lightweight framework, Quiver is ideal for [teaching environments](<doc:Quiver-Notebook-For-Classrooms>), on-device processing, and any context where minimal dependencies and platform portability matter.

### Learn by example

This framework is companion to [Swift Algorithms & Data Structures](https://waynewbishop.github.io/swift-algorithms/), a comprehensive guide that teaches algorithmic thinking through hands-on Swift examples.

## Topics

### Essentials
- <doc:Installation>
- <doc:Basics>
- <doc:How-It-Works>

### Conceptual Guides

- <doc:Numerical-Literacy>
- <doc:Rendering-Math-Primer>
- <doc:Statistics-Primer>
- <doc:Probability-Primer>
- <doc:Central-Limit-Theorem>
- <doc:Inferential-Statistics-Primer>
- <doc:Linear-Algebra-Primer>
- <doc:Determinants-Primer>
- <doc:Calculus-Primer>
- <doc:Physics-Primitives-Primer>
- <doc:Machine-Learning-Primer>
- <doc:Concurrency-Primer>

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

### Statistics
- <doc:Frequency-Tables>
- <doc:Boolean-Masking>
- <doc:Working-With-Distributions>
- <doc:Identifying-A-Distribution>

### Data Preparation
- <doc:Panel>
- <doc:Panel-Workflows>
- <doc:Data-Visualization>
- <doc:Correlation>
- <doc:Random-Sampling>
- <doc:Train-Test-Split>
- <doc:Feature-Scaling>
- <doc:Pipeline>

### Similarity and Search
- <doc:Text-Tokenization>
- <doc:Similarity-Operations>
- <doc:Semantic-Search>

### Models
- <doc:Linear-Regression>
- <doc:Regression-Summary>
- <doc:Gradient-Descent>
- <doc:Polynomials>
- <doc:Naive-Bayes>
- <doc:Nearest-Neighbors-Classification>
- <doc:KMeans-Clustering>
- <doc:Activation-Functions>
- <doc:Evaluation-Metrics>
- <doc:Model-Persistence>

### Platform Guides
- <doc:iOS-Apps>
- <doc:watchOS-Apps>
- <doc:Vapor-Server>

### Rendering
- <doc:Fraction>

### Utilities
- <doc:Quiver-Notebook>
- <doc:Quiver-Notebook-For-Classrooms>
- <doc:Notebook-Datasets>
- <doc:Xcode-Playground>
- <doc:Quiver-Cookbook>
- <doc:Quiver-Knowledge-File>

### Supporting Types
- ``LogDeterminant``
- ``Classification``
- ``Classifier``
- ``AggregationMethod``
- ``MatrixError``
- ``Distributions``
- ``Polynomial``
