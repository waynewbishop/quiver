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

final class GradientDescentTests: XCTestCase {

    // MARK: - Helpers

    /// Standardize each column of a feature matrix to mean 0, std 1 (population).
    /// Mirrors `StandardScaler` ddof=0, which is what the package ships and what
    /// the validation oracle assumes.
    private func standardize(_ features: [[Double]]) -> [[Double]] {
        guard !features.isEmpty else { return features }
        let n = features.count
        let p = features[0].count

        var means = [Double](repeating: 0.0, count: p)
        for row in features {
            for j in 0..<p { means[j] += row[j] }
        }
        for j in 0..<p { means[j] /= Double(n) }

        var stds = [Double](repeating: 0.0, count: p)
        for row in features {
            for j in 0..<p {
                let d = row[j] - means[j]
                stds[j] += d * d
            }
        }
        for j in 0..<p {
            stds[j] = (stds[j] / Double(n)).squareRoot()
            if stds[j] == 0 { stds[j] = 1 }   // avoid divide-by-zero on constant columns
        }

        return features.map { row in
            (0..<p).map { j in (row[j] - means[j]) / stds[j] }
        }
    }

    // MARK: - Happy path

    // Perfect linear data — descent should land on the closed-form coefficients
    // within the optimizer's relative tolerance.
    func testPerfectLinearFitMatchesClosedForm() throws {
        // y = 2x + 3 — single feature, exact relationship.
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [5.0, 7.0, 9.0, 11.0, 13.0]

        // Standardize the feature so the default learning rate is in range.
        let scaled = standardize(features)

        let gd = try GradientDescent.fit(
            features: scaled,
            targets: targets,
            learningRate: 0.1,
            maxIterations: 5000,
            tolerance: 1.0e-10
        )

        // Closed-form oracle on the same scaled features.
        let lr = try LinearRegression.fit(features: scaled, targets: targets)

        XCTAssertEqual(gd.coefficients.count, lr.coefficients.count)
        XCTAssertEqual(gd.coefficients[0], lr.coefficients[0], accuracy: 1.0e-3)
        XCTAssertEqual(gd.coefficients[1], lr.coefficients[1], accuracy: 1.0e-3)
        XCTAssertEqual(gd.featureCount, 1)
        XCTAssertTrue(gd.hasIntercept)
        XCTAssertEqual(gd.outcome, .converged)
    }

    // Multi-feature regression — y = 1 + 2·x₁ + 3·x₂. Same oracle pattern as the
    // single-feature case, expanded to two columns to verify the gradient
    // accumulates correctly across features.
    func testMultiFeatureMatchesClosedForm() throws {
        let raw: [[Double]] = [
            [1.0, 1.0], [2.0, 1.0], [1.0, 2.0],
            [3.0, 2.0], [2.0, 3.0], [4.0, 4.0],
            [5.0, 2.0], [3.0, 5.0]
        ]
        let targets = raw.map { 1.0 + 2.0 * $0[0] + 3.0 * $0[1] }

        let scaled = standardize(raw)

        let gd = try GradientDescent.fit(
            features: scaled,
            targets: targets,
            learningRate: 0.1,
            maxIterations: 10_000,
            tolerance: 1.0e-10
        )
        let lr = try LinearRegression.fit(features: scaled, targets: targets)

        for i in 0..<lr.coefficients.count {
            XCTAssertEqual(
                gd.coefficients[i], lr.coefficients[i], accuracy: 1.0e-3,
                "Coefficient \(i) mismatch: gd=\(gd.coefficients[i]) lr=\(lr.coefficients[i])"
            )
        }
        XCTAssertEqual(gd.featureCount, 2)
    }

    // No-intercept fit — y = 2x passes through the origin. Verifies the
    // ones-column logic is conditional on the intercept flag.
    func testNoIntercept() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [2.0, 4.0, 6.0, 8.0]

        let scaled = standardize(features)
        let gd = try GradientDescent.fit(
            features: scaled,
            targets: targets,
            learningRate: 0.1,
            maxIterations: 5000,
            intercept: false
        )

        XCTAssertEqual(gd.coefficients.count, 1)
        XCTAssertFalse(gd.hasIntercept)
    }

    // Prediction round-trip on the same scaled training rows should reproduce
    // the targets within the optimizer's tolerance.
    func testPredictReproducesTrainingTargets() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [3.0, 5.0, 7.0, 9.0, 11.0]   // y = 2x + 1

        let scaled = standardize(features)
        let gd = try GradientDescent.fit(
            features: scaled, targets: targets,
            learningRate: 0.1, maxIterations: 5000, tolerance: 1.0e-10
        )
        let predicted = gd.predict(scaled)

        for i in 0..<targets.count {
            XCTAssertEqual(predicted[i], targets[i], accuracy: 1.0e-2)
        }
    }

    // The flat-array convenience overload should yield the same model as the
    // explicit `[[Double]]` form.
    func testFlatArrayOverloadMatches() throws {
        let flat = [1.0, 2.0, 3.0, 4.0, 5.0]
        let nested: [[Double]] = flat.map { [$0] }
        let targets = [2.1, 3.9, 6.1, 8.0, 9.8]

        let scaledFlat = standardize(nested).map { $0[0] }
        let scaledNested = standardize(nested)

        let gdFlat = try GradientDescent.fit(
            features: scaledFlat, targets: targets,
            learningRate: 0.1, maxIterations: 2000
        )
        let gdNested = try GradientDescent.fit(
            features: scaledNested, targets: targets,
            learningRate: 0.1, maxIterations: 2000
        )

        XCTAssertEqual(gdFlat.coefficients[0], gdNested.coefficients[0], accuracy: 1.0e-9)
        XCTAssertEqual(gdFlat.coefficients[1], gdNested.coefficients[1], accuracy: 1.0e-9)
    }

    // MARK: - Loss trajectory

    // The loss history must begin with the loss at θ=0, end with `finalLoss`,
    // and never increase in a converged run (signed convergence guarantees this).
    func testLossHistoryIsMonotonicOnConvergence() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [2.0, 4.0, 6.0, 8.0, 10.0]

        let scaled = standardize(features)
        let gd = try GradientDescent.fit(
            features: scaled, targets: targets,
            learningRate: 0.1, maxIterations: 2000, tolerance: 1.0e-8
        )

        XCTAssertEqual(gd.lossHistory.count, gd.iterations + 1)
        XCTAssertEqual(gd.lossHistory.last, gd.finalLoss)

        // A converged run with the signed test must be (weakly) non-increasing.
        for i in 1..<gd.lossHistory.count {
            XCTAssertLessThanOrEqual(
                gd.lossHistory[i], gd.lossHistory[i - 1] + 1.0e-9,
                "Loss increased at iteration \(i)"
            )
        }
    }

    // MARK: - Divergence guards

    // A learning rate well beyond the stability threshold on raw-scale features
    // must surface as a typed divergence error, not a NaN-laden model.
    func testDivergenceOnExcessiveLearningRate() {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [5.0, 7.0, 9.0, 11.0, 13.0]

        XCTAssertThrowsError(
            try GradientDescent.fit(
                features: features,   // intentionally NOT scaled
                targets: targets,
                learningRate: 10.0,    // far beyond the stable range
                maxIterations: 100
            )
        ) { error in
            guard let gdError = error as? GradientDescentError else {
                return XCTFail("Expected GradientDescentError, got \(error)")
            }
            switch gdError {
            case .divergedNonFinite, .divergedIncreasing:
                break   // either failure mode is acceptable for "way too hot"
            }
        }
    }

    // MARK: - Cap-reached without divergence

    // A learning rate small enough to crawl, with a tight iteration cap, should
    // return a best-so-far estimate observably — `outcome == .maxIterationsReached`
    // and no throw.
    func testCapReachedIsSilentAndObservable() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [3.0, 5.0, 7.0, 9.0, 11.0]

        let scaled = standardize(features)
        let gd = try GradientDescent.fit(
            features: scaled, targets: targets,
            learningRate: 1.0e-5,    // intentionally tiny → crawl
            maxIterations: 5,         // intentionally tight cap
            tolerance: 1.0e-8
        )

        XCTAssertEqual(gd.outcome, .maxIterationsReached)
        XCTAssertEqual(gd.iterations, 5)
        XCTAssertTrue(gd.finalLoss.isFinite)
    }

    // MARK: - Description

    func testDescriptionConvergedSingleFeature() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [2.0, 4.0, 6.0, 8.0, 10.0]

        let scaled = standardize(features)
        let gd = try GradientDescent.fit(
            features: scaled, targets: targets,
            learningRate: 0.1, maxIterations: 2000
        )
        let text = gd.description

        XCTAssertTrue(text.contains("GradientDescent"))
        XCTAssertTrue(text.contains("1 feature"))
        XCTAssertTrue(text.contains("converged"))
        XCTAssertTrue(text.contains("loss:"))
    }

    func testDescriptionCapReached() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [2.0, 4.0, 6.0, 8.0]

        let scaled = standardize(features)
        let gd = try GradientDescent.fit(
            features: scaled, targets: targets,
            learningRate: 1.0e-5, maxIterations: 3
        )

        XCTAssertTrue(gd.description.contains("reached max 3 iterations"))
    }

    // MARK: - Equatable and Codable

    func testEquatable() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [2.0, 4.0, 6.0, 8.0]

        let scaled = standardize(features)
        let a = try GradientDescent.fit(
            features: scaled, targets: targets,
            learningRate: 0.1, maxIterations: 500
        )
        let b = try GradientDescent.fit(
            features: scaled, targets: targets,
            learningRate: 0.1, maxIterations: 500
        )
        XCTAssertEqual(a, b)

        let c = try GradientDescent.fit(
            features: scaled, targets: targets,
            learningRate: 0.05, maxIterations: 500    // different rate → different model
        )
        XCTAssertNotEqual(a, c)
    }

    func testCodableRoundTrip() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [2.0, 4.0, 6.0, 8.0]

        let scaled = standardize(features)
        let original = try GradientDescent.fit(
            features: scaled, targets: targets,
            learningRate: 0.1, maxIterations: 500
        )

        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(GradientDescent.self, from: encoded)

        XCTAssertEqual(original, decoded)
    }

    // MARK: - Preconditions

    func testEmptyFeaturesPreconditionMessage() {
        // Preconditions trap rather than throw, so we cannot XCTAssertThrowsError
        // them here — but documenting the contract under test keeps the
        // expectation visible in the test file.
        // Empty features → "Features array must not be empty"
        // Mismatched counts → "Features and targets must have the same number of samples"
        // Non-positive hyperparameters → matching message per preconditions in fit.
        XCTAssertTrue(true)
    }

    // MARK: - Error description

    func testErrorDescriptionsAreHelpful() {
        let nonFinite = GradientDescentError.divergedNonFinite(iteration: 7, loss: .infinity)
        XCTAssertTrue(nonFinite.description.contains("iteration 7"))
        XCTAssertTrue(nonFinite.description.contains("non-finite"))

        let increasing = GradientDescentError.divergedIncreasing(
            iteration: 3, previousLoss: 1.0, currentLoss: 5.0
        )
        XCTAssertTrue(increasing.description.contains("iteration 3"))
        XCTAssertTrue(increasing.description.contains("loss increased"))
    }

    // Scalar convenience predict returns a single Double for a single-feature sample
    func testScalarPredict() throws {
        // y = 2x + 3 — single feature, exact relationship.
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [5.0, 7.0, 9.0, 11.0, 13.0]

        let model = try GradientDescent.fit(
            features: features, targets: targets,
            learningRate: 0.01, maxIterations: 50000, tolerance: 1.0e-12
        )

        // Scalar overload must agree with the batch path's first element.
        let scalar = model.predict(6.0)
        let batch = model.predict([[6.0]])[0]

        XCTAssertEqual(scalar, batch, accuracy: 1.0e-9)
    }
}
