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

/// The distance metric used to compare samples in K-Nearest Neighbors.
///
/// Euclidean distance measures straight-line distance between points and works
/// well when features have similar scales. Cosine distance measures the angle
/// between vectors and works well for text embeddings and high-dimensional data
/// where magnitude is less meaningful than direction.
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
}
