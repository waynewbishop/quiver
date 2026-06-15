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

import XCTest
@testable import Quiver

final class ResidualModelTests: XCTestCase {

    // A perfectly linear dataset: a fitted model explains it exactly, so the
    // residuals are all (near) zero. This is the defining invariant.
    func testResidualsOfPerfectFitAreZero() throws {
        // y = 2x exactly.
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [2.0, 4.0, 6.0, 8.0]

        let model = try LinearRegression.fit(features: features, targets: targets)
        let residualModel = ResidualModel(model: model)
        let residuals = residualModel.residuals(features: features, targets: targets)

        for r in residuals {
            XCTAssertEqual(r, 0.0, accuracy: 1e-9,
                "A perfect linear fit should leave residuals at zero")
        }
    }

    // residual == observed − predicted, sample by sample.
    func testResidualEqualsObservedMinusPredicted() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [2.1, 3.9, 6.2, 7.8, 10.1]   // noisy y ≈ 2x

        let model = try LinearRegression.fit(features: features, targets: targets)
        let residualModel = ResidualModel(model: model)

        let predicted = residualModel.expected(features)
        let residuals = residualModel.residuals(features: features, targets: targets)

        for i in 0..<targets.count {
            XCTAssertEqual(residuals[i], targets[i] - predicted[i], accuracy: 1e-12)
        }
    }

    // A positive residual means the observed value ran above the model's
    // expectation — the de-confounding case the wrapper exists to surface.
    func testPositiveResidualWhereObservedExceedsExpected() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0]]
        let targets = [2.0, 4.0, 6.0]   // y = 2x

        let model = try LinearRegression.fit(features: features, targets: targets)
        let residualModel = ResidualModel(model: model)

        // An observed value well above the line at x = 2 (expected ≈ 4).
        let inflated = residualModel.residual(features: [2.0], observed: 9.0)
        XCTAssertGreaterThan(inflated, 0.0,
            "An observed value above the expectation yields a positive residual")
        XCTAssertEqual(inflated, 5.0, accuracy: 1e-9)   // 9 − 4
    }

    // The scalar convenience matches the array path for the same sample.
    func testScalarResidualMatchesArrayResidual() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [2.0, 4.1, 5.9, 8.0]

        let model = try LinearRegression.fit(features: features, targets: targets)
        let residualModel = ResidualModel(model: model)

        let arrayResiduals = residualModel.residuals(
            features: [[3.0]], targets: [5.9])
        let scalarResidual = residualModel.residual(features: [3.0], observed: 5.9)

        XCTAssertEqual(scalarResidual, arrayResiduals[0], accuracy: 1e-12)
    }

    // `expected(_:)` is a faithful pass-through to the wrapped model's predict.
    func testExpectedMatchesWrappedModelPredict() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0]]
        let targets = [3.0, 5.0, 7.0]   // y = 2x + 1

        let model = try LinearRegression.fit(features: features, targets: targets)
        let residualModel = ResidualModel(model: model)

        XCTAssertEqual(residualModel.expected(features), model.predict(features))
    }

    // Multi-feature regressor wraps the same way.
    func testWrapsMultiFeatureRegressor() throws {
        let features: [[Double]] = [
            [1.0, 10.0], [2.0, 18.0], [3.0, 33.0], [4.0, 39.0]
        ]
        let targets = [11.0, 22.0, 33.0, 44.0]

        let model = try LinearRegression.fit(features: features, targets: targets)
        let residualModel = ResidualModel(model: model)
        let residuals = residualModel.residuals(features: features, targets: targets)

        XCTAssertEqual(residuals.count, targets.count)
    }

    // Edge case: a single sample.
    func testSingleSample() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0]]
        let targets = [2.0, 4.0, 6.0]

        let model = try LinearRegression.fit(features: features, targets: targets)
        let residualModel = ResidualModel(model: model)

        let residuals = residualModel.residuals(features: [[2.0]], targets: [5.0])
        XCTAssertEqual(residuals.count, 1)
        XCTAssertEqual(residuals[0], 1.0, accuracy: 1e-9)   // 5 − 4
    }

    // Equatable: two wrappers of the same fitted model are equal.
    func testEquatable() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0]]
        let targets = [2.0, 4.0, 6.0]
        let model = try LinearRegression.fit(features: features, targets: targets)

        let a = ResidualModel(model: model)
        let b = ResidualModel(model: model)
        XCTAssertEqual(a, b)

        let otherModel = try LinearRegression.fit(
            features: features, targets: [3.0, 6.0, 9.0])
        let c = ResidualModel(model: otherModel)
        XCTAssertNotEqual(a, c)
    }

    // CustomStringConvertible: description names the wrapper and the wrapped
    // model type. The fitted values are read through `coefficients`, not here.
    func testDescription() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0]]
        let targets = [2.0, 4.0, 6.0]
        let model = try LinearRegression.fit(features: features, targets: targets)
        let residualModel = ResidualModel(model: model)

        XCTAssertTrue(residualModel.description.contains("ResidualModel"))
        XCTAssertTrue(residualModel.description.contains("LinearRegression"))
    }

    // Coefficient access: when the wrapped model reports coefficients, the
    // wrapper forwards them, so reading them off the ResidualModel matches
    // reading them off the model directly — consistent with the other models.
    func testCoefficientsForwarding() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [3.0, 5.0, 7.0, 9.0]   // y = 2x + 1
        let model = try LinearRegression.fit(features: features, targets: targets)
        let residualModel = ResidualModel(model: model)

        XCTAssertEqual(residualModel.coefficients, model.coefficients)
        XCTAssertEqual(residualModel.coefficients.count, 2)   // intercept + 1 slope
    }

    // Coefficient forwarding works for the Ridge regressor too — proving the
    // forwarding is generic over any coefficient-bearing model, not hardcoded.
    func testCoefficientsForwardingWithRidge() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [2.0, 4.0, 6.0, 8.0]
        let model = try Ridge.fit(features: features, targets: targets, lambda: 0.1)
        let residualModel = ResidualModel(model: model)

        XCTAssertEqual(residualModel.coefficients, model.coefficients)
    }

    // Codable round-trip: encode and decode preserves the wrapped model.
    func testCodableRoundTrip() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0]]
        let targets = [2.0, 4.0, 6.0]
        let model = try LinearRegression.fit(features: features, targets: targets)
        let original = ResidualModel(model: model)

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(
            ResidualModel<LinearRegression>.self, from: data)

        XCTAssertEqual(original, decoded)
    }

    // Intercept-orthogonality: an OLS fit that carries an intercept leaves
    // residuals that sum to zero (Σrᵢ = 0). This is a corollary of the normal
    // equations — the intercept's own first-order condition — not the full
    // Frisch–Waugh–Lovell theorem. It holds exactly for LinearRegression and is
    // only approximate-at-convergence for the iteratively fit regressors, so it
    // is asserted on LinearRegression alone.
    func testResidualsSumToZeroWithIntercept() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [2.2, 3.8, 6.1, 7.9, 10.3]   // noisy line

        let model = try LinearRegression.fit(features: features, targets: targets)
        let residualModel = ResidualModel(model: model)
        let residuals = residualModel.residuals(features: features, targets: targets)

        let sum = residuals.reduce(0, +)
        XCTAssertEqual(sum, 0.0, accuracy: 1e-6,
            "An OLS fit with an intercept leaves residuals that sum to zero")
    }

    // Reference cross-check: the residuals match a hand-computed
    // observed − predicted built independently of the wrapper's own loop, so
    // the assertion is on the arithmetic, not on the method calling itself.
    // (Stands in for the NumPy reference check; the formal cross-validation
    // against `y - model.predict(X)` remains a merge condition.)
    func testResidualsMatchIndependentReference() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [2.2, 3.8, 6.1, 7.9, 10.3]

        let model = try LinearRegression.fit(features: features, targets: targets)
        let residualModel = ResidualModel(model: model)
        let residuals = residualModel.residuals(features: features, targets: targets)

        // Independent reference: predict once, subtract by hand via zip.
        let predicted = model.predict(features)
        let reference = zip(targets, predicted).map { $0 - $1 }

        XCTAssertEqual(residuals.count, reference.count)
        for i in 0..<reference.count {
            XCTAssertEqual(residuals[i], reference[i], accuracy: 1e-12)
        }
    }
}
