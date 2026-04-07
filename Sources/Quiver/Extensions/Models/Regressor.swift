// Copyright 2026 Wayne W Bishop. All rights reserved.
// Licensed under the Apache License, Version 2.0.

import Foundation

// MARK: - Regressor Protocol

/// A protocol for supervised regression models that predict continuous values.
///
/// Types conforming to `Regressor` provide a ``predict(_:)`` method that returns
/// predicted `[Double]` values. This is the regression counterpart to ``Classifier``,
/// which predicts discrete integer labels.
///
/// ``LinearRegression`` conforms to this protocol. Future regression models
/// (e.g., LogisticRegression for probability output) will conform as well.
///
/// Example:
/// ```swift
/// import Quiver
///
/// let model = try LinearRegression.fit(
///     features: [[1400], [1600], [1800], [2000]],
///     targets: [245000, 312000, 378000, 440000]
/// )
///
/// // predict() returns continuous values — not class labels
/// let predicted = model.predict([[1500], [1900]])
/// // [278000, 410000]
/// ```
public protocol Regressor {
    /// Predicts continuous target values for the given feature vectors.
    /// - Parameter features: 2D array where each row is a sample.
    /// - Returns: An array of predicted values, one per sample.
    func predict(_ features: [[Double]]) -> [Double]
}
