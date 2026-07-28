// Copyright 2026 Wayne W Bishop. All rights reserved.
//
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under
// the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
// ANY KIND, either express or implied. See the License for the specific language governing
// permissions and limitations under the License.

import Foundation

// MARK: - Principal Component Analysis

/// A dimensionality-reduction transformer that projects features onto the
/// directions of largest variance.
///
/// `PCA` learns an orthonormal set of principal components from training data
/// by eigendecomposing the sample covariance matrix, then projects data onto
/// the leading components. The result keeps the directions that carry the
/// most variance and discards the ones that carry the least — compressing
/// correlated features into a smaller set of uncorrelated ones.
///
/// Standardize features first when they are measured in different units, the
/// same way `Ridge` recommends. Without standardization, the feature with the
/// largest numeric scale dominates the variance and therefore the components:
/// ```swift
/// import Quiver
///
/// let features: [[Double]] = [
///     [619, 15000, 0.08], [502, 78000, 0.04],
///     [850, 11000, 0.12], [720, 98000, 0.18]
/// ]
///
/// // Standardize first, then project onto two components
/// let scaler = StandardScaler.fit(features: features)
/// let scaled = scaler.transform(features)
/// let pca = PCA.fit(features: scaled, componentCount: 2)
///
/// let projected = pca.transform(scaled)
/// pca.explainedVarianceRatios  // how much variance each component keeps
/// ```
///
/// This is a value type. Once created via `fit(features:componentCount:)`,
/// the model is immutable — there is no unfitted state, transforming before
/// fitting is unrepresentable, and a fitted model can be safely shared across
/// concurrent code.
///
/// The same data always produces the same components — bit for bit, on every
/// input size — because a single deterministic algorithm handles every shape.
public struct PCA: Codable, CustomStringConvertible, Equatable, Sendable {

    public var description: String {
        let componentNoun = componentCount == 1 ? "component" : "components"
        let featureNoun = featureCount == 1 ? "feature" : "features"
        let percent = explainedVarianceRatios.sum() * 100.0
        return "PCA: \(componentCount) \(componentNoun), \(featureCount) \(featureNoun) "
            + "(\(String(format: "%.1f", percent))% variance explained)"
    }

    /// The principal components, one per row — a (componentCount × featureCount)
    /// matrix. With three input features and two components requested,
    /// `components` has two rows of three values, and `components[0]` is the
    /// direction of largest variance. Each row is a unit vector, sign-fixed so
    /// its largest-magnitude element is positive.
    public let components: [[Double]]

    /// Per-feature mean values learned from the training data.
    public let means: [Double]

    /// The variance captured by each retained component, largest first.
    public let explainedVariances: [Double]

    /// The fraction of total variance captured by each retained component.
    /// The values sum to 1.0 only when every component is retained
    /// (`componentCount == featureCount`); with fewer components the sum is
    /// the fraction of variance the projection keeps.
    public let explainedVarianceRatios: [Double]

    /// Number of principal components the model retains.
    public let componentCount: Int

    /// Number of feature columns the model was fitted on.
    public let featureCount: Int

    private enum CodingKeys: String, CodingKey {
        case components
        case means
        case explainedVariances
        case explainedVarianceRatios
        case componentCount
        case featureCount
    }

    /// Creates a fitted model from already-validated values.
    ///
    /// Internal by design: `fit(features:componentCount:)` is the only public
    /// construction path, which makes an unfitted model unrepresentable.
    ///
    /// - Parameters:
    ///   - components: The principal components, one per row.
    ///   - means: Per-feature training means.
    ///   - explainedVariances: Variance captured by each component.
    ///   - explainedVarianceRatios: Fraction of total variance per component.
    ///   - componentCount: Number of retained components.
    ///   - featureCount: Number of feature columns.
    init(
        components: [[Double]],
        means: [Double],
        explainedVariances: [Double],
        explainedVarianceRatios: [Double],
        componentCount: Int,
        featureCount: Int
    ) {
        self.components = components
        self.means = means
        self.explainedVariances = explainedVariances
        self.explainedVarianceRatios = explainedVarianceRatios
        self.componentCount = componentCount
        self.featureCount = featureCount
    }

    /// Decodes a fitted model, validating its invariants.
    ///
    /// The synthesized decoder would populate the stored properties directly,
    /// so hand-edited or corrupted JSON could produce a model whose
    /// `components` disagree with `componentCount` — plausible-looking output
    /// from inconsistent state. This decoder rejects any payload whose shapes
    /// disagree or whose values are not finite.
    ///
    /// - Parameter decoder: The decoder to read from.
    /// - Throws: `DecodingError.dataCorrupted` naming the violated invariant.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let components = try container.decode([[Double]].self, forKey: .components)
        let means = try container.decode([Double].self, forKey: .means)
        let explainedVariances = try container.decode([Double].self, forKey: .explainedVariances)
        let explainedVarianceRatios = try container.decode([Double].self, forKey: .explainedVarianceRatios)
        let componentCount = try container.decode(Int.self, forKey: .componentCount)
        let featureCount = try container.decode(Int.self, forKey: .featureCount)

        guard componentCount >= 1, featureCount >= 1, componentCount <= featureCount else {
            throw DecodingError.dataCorruptedError(forKey: .componentCount, in: container,
                debugDescription: "componentCount must be between 1 and featureCount (\(featureCount)), got \(componentCount)")
        }
        guard components.count == componentCount else {
            throw DecodingError.dataCorruptedError(forKey: .components, in: container,
                debugDescription: "components has \(components.count) rows but componentCount is \(componentCount)")
        }
        guard components.allSatisfy({ $0.count == featureCount }) else {
            throw DecodingError.dataCorruptedError(forKey: .components, in: container,
                debugDescription: "every components row must have \(featureCount) entries")
        }
        guard means.count == featureCount else {
            throw DecodingError.dataCorruptedError(forKey: .means, in: container,
                debugDescription: "means has \(means.count) entries but featureCount is \(featureCount)")
        }
        guard explainedVariances.count == componentCount else {
            throw DecodingError.dataCorruptedError(forKey: .explainedVariances, in: container,
                debugDescription: "explainedVariances has \(explainedVariances.count) entries but componentCount is \(componentCount)")
        }
        guard explainedVarianceRatios.count == componentCount else {
            throw DecodingError.dataCorruptedError(forKey: .explainedVarianceRatios, in: container,
                debugDescription: "explainedVarianceRatios has \(explainedVarianceRatios.count) entries but componentCount is \(componentCount)")
        }
        let allValues = components.flatMap { $0 } + means + explainedVariances + explainedVarianceRatios
        guard allValues.allSatisfy({ $0.isFinite }) else {
            throw DecodingError.dataCorruptedError(forKey: .components, in: container,
                debugDescription: "all stored values must be finite")
        }

        self.init(
            components: components,
            means: means,
            explainedVariances: explainedVariances,
            explainedVarianceRatios: explainedVarianceRatios,
            componentCount: componentCount,
            featureCount: featureCount
        )
    }

    /// Learns principal components from a feature matrix.
    ///
    /// Centers the data, forms the sample covariance matrix (ddof = 1), and
    /// eigendecomposes it. The eigenvectors with the largest eigenvalues
    /// become the components; each eigenvalue is the variance its component
    /// captures. Eigenvalues that are negative by rounding noise — the
    /// covariance matrix is positive semi-definite, so a tiny negative value
    /// is definitionally noise — are clamped to exactly zero before ratios
    /// are computed.
    ///
    /// - Parameters:
    ///   - features: 2D array where each row is a sample and each column is a feature.
    ///   - componentCount: Number of components to retain, from 1 up to the
    ///     number of feature columns.
    /// - Returns: A fitted `PCA` ready to transform data.
    /// - Complexity: O(*n*·*p*² + *p*³) where *n* is the number of samples and
    ///   *p* is the number of features.
    public static func fit(features: [[Double]], componentCount: Int) -> PCA {
        precondition(!features.isEmpty, "Features array must not be empty")

        let featureCount = features[0].count
        precondition(featureCount > 0, "Features must have at least one column")
        precondition(features.allSatisfy { $0.count == featureCount },
            "All rows must have the same number of features")
        precondition(features.count > 1,
            "PCA requires at least two samples, got \(features.count)")
        precondition(componentCount > 0, "componentCount must be positive")
        precondition(componentCount <= featureCount,
            "componentCount (\(componentCount)) cannot exceed the number of features (\(featureCount))")
        precondition(features.allSatisfy { row in row.allSatisfy { $0.isFinite } },
            "Features must contain only finite values")

        guard let means = features.meanVector(),
              let covariance = features.covarianceMatrix(ddof: 1) else {
            preconditionFailure("Covariance construction failed after shape validation")
        }

        // The covariance matrix is symmetric by construction (mirrored upper
        // triangle), so the internal kernel is called directly — the public
        // symmetric check cannot fail here and no error channel is crossed.
        let eigen = _Eigen.decompose(covariance)

        // Clamp rounding-noise negatives in (-1e-12 * largest, 0) to exactly
        // zero. Anything more negative is left in place to surface: it
        // indicates a real defect rather than noise.
        var eigenvalues = eigen.eigenvalues
        if let largest = eigenvalues.first, largest > 0.0 {
            let window = -1e-12 * largest
            for i in 0..<eigenvalues.count where eigenvalues[i] < 0.0 && eigenvalues[i] > window {
                eigenvalues[i] = 0.0
            }
        }

        let totalVariance = eigenvalues.sum()
        let retainedVariances = Array(eigenvalues[0..<componentCount])
        let ratios: [Double]
        if totalVariance > 0.0 {
            ratios = retainedVariances.map { $0 / totalVariance }
        } else {
            // Zero total variance (all features constant) — no direction
            // carries variance, so every ratio is zero rather than NaN.
            ratios = [Double](repeating: 0.0, count: componentCount)
        }

        return PCA(
            components: Array(eigen.eigenvectors[0..<componentCount]),
            means: means,
            explainedVariances: retainedVariances,
            explainedVarianceRatios: ratios,
            componentCount: componentCount,
            featureCount: featureCount
        )
    }

    /// Projects a feature matrix onto the principal components.
    ///
    /// Each row is centered using the training means, then projected onto the
    /// component directions. The result has one row per input sample and one
    /// column per retained component — a (sampleCount × componentCount)
    /// matrix, samples staying as rows.
    ///
    /// - Parameter features: 2D array to project, with the same number of
    ///   columns as the training data.
    /// - Returns: The projected matrix, one row per sample.
    /// - Complexity: O(*n*·*p*·*k*) where *n* is the number of samples, *p*
    ///   the number of features, and *k* the number of components.
    public func transform(_ features: [[Double]]) -> [[Double]] {
        guard !features.isEmpty else { return [] }
        for row in features {
            precondition(row.count == featureCount,
                "Row has \(row.count) features, model expects \(featureCount)")
        }

        let negatedMeans = means.map { -$0 }
        let centered = features.broadcast(addingToEachRow: negatedMeans)
        return centered.multiplyMatrix(components.transposed())
    }

    /// Maps projected data back into the original feature space.
    ///
    /// Reverses the projection and re-adds the training means. The
    /// round-trip `inverseTransform(transform(features))` reproduces the
    /// input exactly only when `componentCount == featureCount`; with fewer
    /// components the reconstruction is the closest approximation the
    /// retained subspace can express, and the difference is the variance the
    /// discarded components carried. That reconstruction error is the point:
    /// it measures what the compression gave up.
    ///
    /// - Parameter projected: 2D array of projected rows, with one column per
    ///   retained component.
    /// - Returns: The reconstructed matrix in the original feature space.
    /// - Complexity: O(*n*·*p*·*k*) where *n* is the number of rows, *p* the
    ///   number of features, and *k* the number of components.
    public func inverseTransform(_ projected: [[Double]]) -> [[Double]] {
        guard !projected.isEmpty else { return [] }
        for row in projected {
            precondition(row.count == componentCount,
                "Row has \(row.count) values, model expects \(componentCount) components")
        }

        let reconstructed = projected.multiplyMatrix(components)
        return reconstructed.broadcast(addingToEachRow: means)
    }
}
