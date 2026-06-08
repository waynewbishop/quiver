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

extension Regressor {

    /// Predicts a single target value from one single-feature sample.
    ///
    /// A scalar convenience for single-feature models: pass one feature value
    /// and get one prediction back, with no array to wrap on the way in or
    /// unwrap on the way out.
    ///
    /// ```swift
    /// import Quiver
    ///
    /// let sqft   = [1000.0, 1500.0, 2000.0, 2500.0, 3000.0]
    /// let prices = [150000.0, 200000.0, 260000.0, 310000.0, 370000.0]
    ///
    /// let model = try LinearRegression.fit(features: sqft, targets: prices)
    /// let price = model.predict(3500.0)
    /// ```
    ///
    /// - Parameter value: A single feature value for a model trained on one feature.
    /// - Returns: The predicted target value.
    public func predict(_ value: Double) -> Double {
        return predict([[value]])[0]
    }
}
