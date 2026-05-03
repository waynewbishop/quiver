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

A Swift package for vector mathematics, numerical computing, and machine learning.

## Overview

[Quiver](https://github.com/waynewbishop/quiver) expands the Swift ecosystem with a pure, Swift-first approach to vector mathematics and numerical computing. By building directly on Swift's powerful type system and syntax, Quiver creates an intuitive bridge between traditional array operations and advanced mathematical concepts. Built as an extension on the standard `Array` type, the framework embraces Swift's emphasis on readability and expressiveness, offering mathematical operations that feel natural to iOS and macOS developers.

### Data science in Swift

As Swift continues to expand beyond app development into domains like server-side computing, machine learning, and data analysis, the need for robust mathematical tools becomes increasingly important. Quiver serves as a foundation for data science workflows in Swift, enabling operations that are fundamental to fields like computer vision, game development, machine learning, and scientific computing.

### Why Quiver

Quiver provides developers the tools to work with data directly. This includes analyzing data with statistics, transforming it with linear algebra, preparing it for modeling, or training machine learning models. As a pure Swift library with zero external dependencies, Quiver runs on every Apple platform — iOS, macOS, watchOS, tvOS, and visionOS — as well as server-side Swift with frameworks like Vapor, Linux environments, and containerized deployments. As a lightweight framework, Quiver is ideal for teaching environments, on-device processing, and any context where minimal dependencies and platform portability matter.
 
### Learn by example

This framework is companion to [Swift Algorithms & Data Structures](https://waynewbishop.github.io/swift-algorithms/), a comprehensive guide that teaches algorithmic thinking through hands-on Swift examples. 

## Topics

### Essentials
- <doc:Installation>
- <doc:Basics>
- <doc:Usage>
- <doc:How-It-Works>

### Conceptual Guides
- <doc:Linear-Algebra-Primer>
- <doc:Determinants-Primer>
- <doc:Statistics-Primer>
- <doc:Inferential-Statistics-Primer>
- <doc:Machine-Learning-Primer>
- <doc:Numerical-Literacy>
- <doc:Concurrency-Primer>

### iOS
- <doc:iOS-Guide>
- <doc:iOS-Patterns>

### watchOS
- <doc:watchOS-Guide>
- <doc:watchOS-Patterns>

### Vapor
- <doc:Vapor-Guide>
- <doc:Vapor-Patterns>

### Vectors
- <doc:Vector-Operations>
- <doc:Vector-Projections>

### Statistics
- <doc:Statistical-Operations>
- <doc:Frequency-Tables>
- <doc:Working-With-Distributions>
- <doc:Boolean-Masking>

### Algebra
- <doc:Fraction>
- <doc:Polynomials>

### Matrices
- <doc:Matrix-Operations>
- <doc:Shape-And-Size>
- <doc:Reshape-And-Flatten>
- <doc:Broadcasting-Operations>
- <doc:Array-Generation>
- <doc:Random-Number-Generation>

### Transformations
- <doc:Matrix-Transformations>
- <doc:Common-Transformations>
- <doc:Composing-Transformations>

### Similarity and Search
- <doc:Text-Tokenization>
- <doc:Similarity-Operations>
- <doc:Semantic-Search>

### Signal Processing
- <doc:Fourier-Transform>

### Data Preparation
- <doc:Panel>
- <doc:Panel-and-Charts>
- <doc:Data-Visualization>
- <doc:Train-Test-Split>
- <doc:Feature-Scaling>
- <doc:Pipeline>

### Models
- <doc:Naive-Bayes>
- <doc:Linear-Regression>
- <doc:Nearest-Neighbors-Classification>
- <doc:KMeans-Clustering>
- <doc:Evaluation-Metrics>
- <doc:Activation-Functions>

### Utilities
- <doc:Model-Persistence>
- <doc:Quiver-Notebook>
- <doc:Notebook-Datasets>
- <doc:Quiver-Notebook-For-Classrooms>

### Supporting Types
- ``LogDeterminant``
- ``Classification``
- ``Classifier``
- ``AggregationMethod``
- ``MatrixError``
- ``Distributions``
- ``Polynomial``
