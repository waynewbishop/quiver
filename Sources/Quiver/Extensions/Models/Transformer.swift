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

// MARK: - Transformer

/// A fitted preprocessing stage that maps a feature matrix to a new one.
///
/// A transformer is a value that has already learned its parameters — a
/// scaler's per-column means and standard deviations, a projection's
/// component directions — and applies them to any matrix with the same
/// column layout as its training data. The same fitted transformer serves
/// both training features and later query points, which is what keeps a
/// preprocessing step consistent between fit time and prediction time.
///
/// `Pipeline` is the caller: its predict path runs each stage's
/// `transform(_:)` in order before the model sees the data. Conforming
/// types are `Sendable` values, so a fitted stage crosses task boundaries
/// with the same guarantees as the models it feeds.
///
/// Example:
/// ```swift
/// import Quiver
///
/// let features: [[Double]] = [[1, 2], [3, 4], [5, 8], [6, 9]]
///
/// let scaler = StandardScaler.fit(features: features)
/// let reducer = PCA.fit(features: scaler.transform(features), componentCount: 1)
///
/// // Both fitted values are transformers — the same call shape applies each stage.
/// let stages: [any Transformer] = [scaler, reducer]
/// ```
public protocol Transformer: Sendable {

    /// Transforms a feature matrix using the parameters learned at fit time.
    ///
    /// - Parameter features: 2D array where each row is a sample and each
    ///   column is a feature, in the same column order as the training data.
    /// - Returns: The transformed matrix, one output row per input row.
    func transform(_ features: [[Double]]) -> [[Double]]
}

// MARK: - Conformances

// StandardScaler and PCA already expose transform(_:) with the required
// shape; conformance is declaration-only.

extension StandardScaler: Transformer {}

extension PCA: Transformer {}
