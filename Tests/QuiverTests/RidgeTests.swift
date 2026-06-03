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

final class RidgeTests: XCTestCase {

    // MARK: - Helpers

    /// Standardize each column to mean 0, std 1 (population, ddof=0) — matches
    /// `StandardScaler` and the GradientDescent test oracle.
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
            if stds[j] == 0 { stds[j] = 1 }
        }

        return features.map { row in
            (0..<p).map { j in (row[j] - means[j]) / stds[j] }
        }
    }

    /// L2 norm of the feature weights (everything past the intercept).
    private func weightNorm(_ model: Ridge) -> Double {
        let weights = model.hasIntercept ? Array(model.coefficients.dropFirst()) : model.coefficients
        return weights.reduce(0.0) { $0 + $1 * $1 }.squareRoot()
    }

    // Shared fixture: a two-feature regression with a bit of noise.
    private let features: [[Double]] = [
        [1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0],
        [5.0, 6.0], [6.0, 5.0], [7.0, 8.0], [8.0, 7.0]
    ]
    private let targets = [3.1, 2.9, 7.2, 6.8, 11.1, 10.9, 15.0, 14.8]

    // MARK: - λ = 0 reproduces ordinary least squares

    // The load-bearing correctness test: with lambda = 0 the penalty is inert,
    // so ridge must converge to the same coefficients as a plain GradientDescent
    // run with identical hyperparameters. If this drifts, the penalty has leaked
    // into the lambda = 0 path.
    func testZeroLambdaMatchesGradientDescent() throws {
        let scaled = standardize(features)

        let ridge = try Ridge.fit(
            features: scaled, targets: targets, lambda: 0.0,
            learningRate: 0.05, maxIterations: 20000, tolerance: 1.0e-12
        )
        let gd = try GradientDescent.fit(
            features: scaled, targets: targets,
            learningRate: 0.05, maxIterations: 20000, tolerance: 1.0e-12
        )

        XCTAssertEqual(ridge.coefficients.count, gd.coefficients.count)
        for (r, g) in zip(ridge.coefficients, gd.coefficients) {
            XCTAssertEqual(r, g, accuracy: 1.0e-9,
                "lambda = 0 must reproduce the unregularized fit exactly")
        }
    }

    // The same identity against the closed-form normal equation, the ultimate
    // oracle for an unregularized linear fit.
    func testZeroLambdaMatchesClosedForm() throws {
        let scaled = standardize(features)

        let ridge = try Ridge.fit(
            features: scaled, targets: targets, lambda: 0.0,
            learningRate: 0.05, maxIterations: 50000, tolerance: 1.0e-13
        )
        let ols = try LinearRegression.fit(features: scaled, targets: targets)

        for (r, o) in zip(ridge.coefficients, ols.coefficients) {
            XCTAssertEqual(r, o, accuracy: 1.0e-4,
                "lambda = 0 ridge should track the closed-form OLS solution")
        }
    }

    // MARK: - Rising λ shrinks the weights

    // The defining behavior of ridge: as the penalty grows, the weight norm
    // shrinks monotonically toward zero. Checked across a sweep.
    func testIncreasingLambdaShrinksWeightNorm() throws {
        let scaled = standardize(features)
        let lambdas = [0.0, 0.1, 1.0, 10.0, 100.0]

        // The L2 term raises the loss-surface curvature, so the safe step size
        // shrinks as λ grows. A conservative rate keeps every fit in the sweep
        // within the optimizer's divergence guard.
        var previousNorm = Double.infinity
        for lambda in lambdas {
            let model = try Ridge.fit(
                features: scaled, targets: targets, lambda: lambda,
                learningRate: 0.001, maxIterations: 200000, tolerance: 1.0e-11
            )
            let norm = weightNorm(model)
            XCTAssertLessThan(norm, previousNorm + 1.0e-6,
                "weight norm must not grow as λ increases (λ=\(lambda))")
            previousNorm = norm
        }
    }

    // A large penalty should drive the weights close to zero — the limiting
    // behavior of ridge.
    func testLargeLambdaDrivesWeightsTowardZero() throws {
        let scaled = standardize(features)
        // λ this large makes the penalty term 2λθ dominate the gradient, so the
        // stable step size is on the order of 1/(2λ). Use a correspondingly tiny
        // learning rate; the weights collapse toward zero in a few iterations.
        let model = try Ridge.fit(
            features: scaled, targets: targets, lambda: 1.0e4,
            learningRate: 1.0e-5, maxIterations: 50000, tolerance: 1.0e-14
        )
        XCTAssertLessThan(weightNorm(model), 0.05,
            "a very large λ should shrink the feature weights to near zero")
    }

    // MARK: - Intercept is never penalized

    // Under a crushing penalty the feature weights collapse toward zero, but the
    // intercept is excluded from the penalty and must stay free. The direct test
    // of "not penalized": compare the intercept to a *penalized* fit of the same
    // shape — the intercept must not be driven toward zero the way the weights
    // are. (A single global learning rate can't drive the tiny-curvature weights
    // and the unpenalized intercept to convergence simultaneously, so we test the
    // property — intercept stays large — not an exact converged value.)
    func testInterceptIsNotPenalized() throws {
        let scaled = standardize(features)
        let model = try Ridge.fit(
            features: scaled, targets: targets, lambda: 1.0e4,
            learningRate: 1.0e-5, maxIterations: 50000, tolerance: 1.0e-14
        )

        // Weights are crushed to ~0; the intercept is not.
        XCTAssertLessThan(weightNorm(model), 0.05,
            "feature weights should collapse under a large penalty")
        XCTAssertGreaterThan(abs(model.coefficients[0]), 1.0,
            "the unpenalized intercept must stay well away from zero")
        // And it is climbing toward the target mean, not toward zero.
        let targetMean = targets.reduce(0.0, +) / Double(targets.count)
        XCTAssertGreaterThan(model.coefficients[0], targetMean / 2.0,
            "the intercept tracks the target mean, not the penalty")
    }

    // With intercept disabled, every coefficient is a penalized weight.
    func testNoInterceptPenalizesAllCoefficients() throws {
        let scaled = standardize(features)
        let model = try Ridge.fit(
            features: scaled, targets: targets, lambda: 1.0e4,
            learningRate: 1.0e-5, maxIterations: 50000, tolerance: 1.0e-14,
            intercept: false
        )
        XCTAssertFalse(model.hasIntercept)
        // No intercept to absorb the mean — all weights penalized to near zero.
        XCTAssertLessThan(weightNorm(model), 0.05)
    }

    // MARK: - Happy path

    func testFitProducesUsablePredictions() throws {
        let scaled = standardize(features)
        let model = try Ridge.fit(features: scaled, targets: targets, lambda: 0.1)

        XCTAssertEqual(model.featureCount, 2)
        XCTAssertTrue(model.hasIntercept)
        XCTAssertEqual(model.coefficients.count, 3)   // intercept + 2 weights

        let preds = model.predict(scaled)
        XCTAssertEqual(preds.count, targets.count)
        // Predictions should track targets reasonably at a light penalty.
        for (p, t) in zip(preds, targets) {
            XCTAssertEqual(p, t, accuracy: 2.0)
        }
    }

    func testSingleFeatureOverload() throws {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y = [2.1, 3.9, 6.1, 8.0, 9.8]
        let scaled = standardize(x.map { [$0] }).map { $0[0] }

        let model = try Ridge.fit(features: scaled, targets: y, lambda: 0.01)
        XCTAssertEqual(model.featureCount, 1)
        let preds = model.predict([scaled[0], scaled[1]])
        XCTAssertEqual(preds.count, 2)
    }

    // MARK: - Equatable

    func testEquatable() throws {
        let scaled = standardize(features)
        let a = try Ridge.fit(features: scaled, targets: targets, lambda: 0.1)
        let b = try Ridge.fit(features: scaled, targets: targets, lambda: 0.1)
        let c = try Ridge.fit(features: scaled, targets: targets, lambda: 1.0)

        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }

    // MARK: - CustomStringConvertible

    func testDescriptionContainsLambdaAndShape() throws {
        let scaled = standardize(features)
        let model = try Ridge.fit(features: scaled, targets: targets, lambda: 0.1)
        let text = model.description

        XCTAssertTrue(text.contains("Ridge:"))
        XCTAssertTrue(text.contains("2 features"))
        XCTAssertTrue(text.contains("λ=0.1"))
    }

    // MARK: - Codable / Pipeline interop

    // Ridge must round-trip through Codable so it composes with Pipeline, which
    // is generic over Codable & Equatable & Sendable models.
    func testCodableRoundTrip() throws {
        let scaled = standardize(features)
        let model = try Ridge.fit(features: scaled, targets: targets, lambda: 0.1)

        let data = try JSONEncoder().encode(model)
        let restored = try JSONDecoder().decode(Ridge.self, from: data)

        XCTAssertEqual(model, restored)
    }

    // MARK: - Error handling

    func testDivergenceThrowsOnUnscaledFeaturesWithLargeRate() {
        // Raw-scale features + an aggressive learning rate diverges, just as it
        // does for plain GradientDescent — the shared optimizer's contract.
        XCTAssertThrowsError(
            try Ridge.fit(
                features: features, targets: targets, lambda: 0.1,
                learningRate: 1.0, maxIterations: 1000
            )
        ) { error in
            XCTAssertTrue(error is GradientDescentError)
        }
    }

    // MARK: - Full pipeline

    func testStandardScalerThenRidge() throws {
        let scaler = StandardScaler.fit(features: features)
        let scaled = scaler.transform(features)

        let model = try Ridge.fit(features: scaled, targets: targets, lambda: 0.5)
        let newRaw: [[Double]] = [[2.5, 2.5], [6.5, 6.5]]
        let newScaled = scaler.transform(newRaw)
        let preds = model.predict(newScaled)

        XCTAssertEqual(preds.count, 2)
        // Monotone targets → second prediction should exceed the first.
        XCTAssertGreaterThan(preds[1], preds[0])
    }
}
