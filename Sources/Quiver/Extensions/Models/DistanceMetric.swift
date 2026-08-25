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

// MARK: - Distance Metric

/// The distance metric used to compare two vectors.
///
/// Euclidean distance measures straight-line distance between points and works
/// well when features have similar scales. Cosine distance measures the angle
/// between vectors and works well for text embeddings and high-dimensional data
/// where magnitude is less meaningful than direction. Manhattan distance sums
/// the absolute differences along each axis and is more robust to outliers.
///
/// ``KNearestNeighbors`` accepts a metric when fitting a classifier, and the
/// vector method `distance(to:metric:)` applies the same metrics directly to
/// any two arrays.
public enum DistanceMetric: Codable, Equatable, Sendable {

    /// Euclidean distance: √Σ(aᵢ − bᵢ)².
    ///
    /// Sensitive to feature scale — consider using ``FeatureScaler`` before
    /// fitting when features have different units or magnitudes.
    case euclidean

    /// Cosine distance: 1 − cosine similarity.
    ///
    /// Scale-invariant — vectors pointing in the same direction have distance
    /// 0 regardless of their magnitude. Preferred for text embeddings, TF-IDF
    /// vectors, and other high-dimensional sparse data.
    case cosine

    /// Manhattan distance: Σ|aᵢ − bᵢ|.
    ///
    /// Sums the absolute differences along each axis — the number of grid steps
    /// between two points. More robust to outliers than Euclidean, since a large
    /// per-feature difference contributes linearly rather than squared. Like
    /// Euclidean, it is scale-sensitive; consider using ``FeatureScaler`` before
    /// fitting when features have different units or magnitudes.
    case manhattan

    /// Chebyshev distance: max|aᵢ − bᵢ|.
    ///
    /// The largest difference along any single axis, also called the L∞ or
    /// chessboard distance (the number of king moves between two squares). Useful
    /// when a match on every feature matters less than the single worst-fitting
    /// feature. It is the limiting case of ``minkowski(p:)`` as `p` grows without
    /// bound, but is kept as its own case because that limit cannot be evaluated
    /// with a finite exponent.
    case chebyshev

    /// Squared Euclidean distance: Σ(aᵢ − bᵢ)².
    ///
    /// Euclidean distance without the final square root. It ranks points in the
    /// same order as ``euclidean``, so nearest-neighbor and clustering results are
    /// identical, while skipping one square root per comparison. Prefer it when the
    /// ordering is all that matters and speed is at a premium.
    ///
    /// > Note: This is not a true distance metric — it does not satisfy the
    /// > triangle inequality, and its values are on a squared scale, so they should
    /// > not be read as straight-line distances or compared across differently
    /// > scaled feature sets.
    case squaredEuclidean

    /// Minkowski distance: (Σ|aᵢ − bᵢ|ᵖ)^(1/p).
    ///
    /// The general Lᵖ family that unifies the others: `p = 1` is ``manhattan``,
    /// `p = 2` is ``euclidean``, and the limit as `p → ∞` is ``chebyshev``. Tuning
    /// `p` interpolates between summing every difference (small `p`) and being
    /// dominated by the largest one (large `p`).
    ///
    /// - Parameter p: The order of the norm. Must be greater than 0.
    case minkowski(p: Double)
}

// MARK: - Codable

extension DistanceMetric {

    // The enum gained an associated value (`minkowski(p:)`), so it can no longer
    // use the compiler-synthesized Codable. This encodes a string discriminator
    // plus the Minkowski order, keeping archives stable across the value cases.

    private enum CodingKeys: String, CodingKey {
        case kind
        case p
    }

    private enum Kind: String, Codable {
        case euclidean, cosine, manhattan, chebyshev, squaredEuclidean, minkowski
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(Kind.self, forKey: .kind)
        switch kind {
        case .euclidean: self = .euclidean
        case .cosine: self = .cosine
        case .manhattan: self = .manhattan
        case .chebyshev: self = .chebyshev
        case .squaredEuclidean: self = .squaredEuclidean
        case .minkowski:
            let p = try container.decode(Double.self, forKey: .p)
            self = .minkowski(p: p)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .euclidean: try container.encode(Kind.euclidean, forKey: .kind)
        case .cosine: try container.encode(Kind.cosine, forKey: .kind)
        case .manhattan: try container.encode(Kind.manhattan, forKey: .kind)
        case .chebyshev: try container.encode(Kind.chebyshev, forKey: .kind)
        case .squaredEuclidean: try container.encode(Kind.squaredEuclidean, forKey: .kind)
        case .minkowski(let p):
            try container.encode(Kind.minkowski, forKey: .kind)
            try container.encode(p, forKey: .p)
        }
    }
}
