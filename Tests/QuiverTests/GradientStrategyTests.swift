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

/// Tests the unified gradient-descent engine — the `_GradientStrategy` seam that
/// lets one shared `_GradientDescent.descend` loop serve both the squared-error
/// objective (`LinearRegression`, `Ridge`, `GradientDescent`) and the
/// cross-entropy objective (`LogisticRegression`). These reach into the internal
/// engine via `@testable` to pin the refactor's contract directly, beyond the
/// public model tests that already exercise it end-to-end.
final class GradientStrategyTests: XCTestCase {

    // MARK: - Helpers

    /// Standardize each column to mean 0, std 1 (population, ddof=0).
    private func standardize(_ features: [[Double]]) -> [[Double]] {
        let n = features.count, p = features[0].count
        var means = [Double](repeating: 0, count: p)
        for row in features { for j in 0..<p { means[j] += row[j] } }
        for j in 0..<p { means[j] /= Double(n) }
        var stds = [Double](repeating: 0, count: p)
        for row in features { for j in 0..<p { let d = row[j] - means[j]; stds[j] += d * d } }
        for j in 0..<p { stds[j] = (stds[j] / Double(n)).squareRoot(); if stds[j] == 0 { stds[j] = 1 } }
        return features.map { row in (0..<p).map { (row[$0] - means[$0]) / stds[$0] } }
    }

    /// Prepend a ones column for the intercept, matching the model `fit` contract.
    private func withIntercept(_ features: [[Double]]) -> [[Double]] {
        features.map { [1.0] + $0 }
    }

    // MARK: - Strategy correctness

    // The squared-error strategy's gradient and loss match hand-computed values
    // at a known point. Single feature, θ = [0, 0]: residual = −y, loss = mean(y²),
    // gradient = (2/n)·Xᵀ(−y).
    func testMSEStrategyGradientAndLossAtZero() {
        let X = [[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]]   // intercept + one feature
        let y = [3.0, 5.0, 7.0]
        let strategy = _MeanSquaredErrorStrategy(lambda: 0, penalizeFromIndex: 0)
        let theta = [0.0, 0.0]

        // loss at θ=0 is mean(y²) = (9+25+49)/3 = 27.666...
        XCTAssertEqual(strategy.loss(features: X, targets: y, parameters: theta),
                       (9.0 + 25.0 + 49.0) / 3.0, accuracy: 1e-12)

        // gradient = (2/3)·Xᵀ(Xθ − y) = (2/3)·Xᵀ(−y)
        //   component 0: (2/3)·(−3 −5 −7) = (2/3)(−15) = −10
        //   component 1: (2/3)·(−6 −20 −42) = (2/3)(−68) = −45.333...
        let g = strategy.gradient(features: X, targets: y, parameters: theta)
        XCTAssertEqual(g[0], -10.0, accuracy: 1e-12)
        XCTAssertEqual(g[1], (2.0 / 3.0) * -68.0, accuracy: 1e-12)
    }

    // The L2 penalty adds 2λθⱼ to the gradient from penalizeFromIndex onward and
    // λΣθⱼ² to the loss, leaving a leading intercept untouched.
    func testMSEStrategyPenaltySparesIntercept() {
        let X = [[1.0, 2.0], [1.0, 4.0]]
        let y = [1.0, 2.0]
        let theta = [5.0, 3.0]   // nonzero so the penalty is visible
        let plain = _MeanSquaredErrorStrategy(lambda: 0, penalizeFromIndex: 1)
        let ridge = _MeanSquaredErrorStrategy(lambda: 0.1, penalizeFromIndex: 1)

        let gPlain = plain.gradient(features: X, targets: y, parameters: theta)
        let gRidge = ridge.gradient(features: X, targets: y, parameters: theta)

        // Intercept (index 0) is spared — identical with and without penalty.
        XCTAssertEqual(gRidge[0], gPlain[0], accuracy: 1e-12)
        // Feature (index 1) gains exactly 2·λ·θ₁ = 2·0.1·3 = 0.6.
        XCTAssertEqual(gRidge[1] - gPlain[1], 0.6, accuracy: 1e-12)
        // Loss gains λ·θ₁² = 0.1·9 = 0.9 (intercept not in the norm).
        let lossDelta = ridge.loss(features: X, targets: y, parameters: theta)
                      - plain.loss(features: X, targets: y, parameters: theta)
        XCTAssertEqual(lossDelta, 0.9, accuracy: 1e-12)
    }

    // The cross-entropy strategy's loss at θ = 0 is exactly log 2 (every σ = 0.5),
    // and its gradient there is (1/n)Xᵀ(0.5 − y).
    func testCrossEntropyStrategyAtZero() {
        let X = [[1.0, 2.0], [1.0, 4.0], [1.0, 6.0], [1.0, 8.0]]
        let y = [0.0, 0.0, 1.0, 1.0]
        let strategy = _CrossEntropyStrategy()
        let theta = [0.0, 0.0]

        XCTAssertEqual(strategy.loss(features: X, targets: y, parameters: theta),
                       log(2.0), accuracy: 1e-12)

        // residual = 0.5 − y = [0.5, 0.5, −0.5, −0.5]
        //   component 0: (1/4)(0.5+0.5−0.5−0.5) = 0
        //   component 1: (1/4)(1 + 2 − 3 − 4) = (1/4)(−4) = −1
        let g = strategy.gradient(features: X, targets: y, parameters: theta)
        XCTAssertEqual(g[0], 0.0, accuracy: 1e-12)
        XCTAssertEqual(g[1], -1.0, accuracy: 1e-12)
    }

    // MARK: - Shared loop, both objectives

    // The shared loop with the MSE strategy converges to the closed-form OLS
    // coefficients — the same oracle the squared-error engine has always used.
    func testSharedLoopMSEConvergesToClosedForm() throws {
        let raw = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [2.0, 4.0, 6.0, 8.0, 10.0]   // exact y = 2x
        let X = withIntercept(standardize(raw))

        let result = try _GradientDescent.descend(
            features: X, targets: targets,
            learningRate: 0.1, maxIterations: 10_000, tolerance: 1e-12,
            strategy: _MeanSquaredErrorStrategy(lambda: 0, penalizeFromIndex: 0)
        )
        XCTAssertEqual(result.outcome, .converged)
        // Standardized intercept is the target mean (6.0); near-perfect fit.
        XCTAssertEqual(result.parameters[0], 6.0, accuracy: 1e-3)
        XCTAssertLessThan(result.lossHistory.last!, 1e-6)
    }

    // The shared loop with the cross-entropy strategy converges to the validated
    // maximum-likelihood estimate on non-separable data.
    func testSharedLoopCrossEntropyConvergesToMLE() throws {
        let raw = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [2.5], [4.5]]
        let targets = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        let X = withIntercept(standardize(raw))

        let result = try _GradientDescent.descend(
            features: X, targets: targets,
            learningRate: 0.5, maxIterations: 500_000, tolerance: 1e-12,
            strategy: _CrossEntropyStrategy()
        )
        XCTAssertEqual(result.outcome, .converged)
        XCTAssertEqual(result.parameters[0], 0.0, accuracy: 1e-4)       // intercept
        XCTAssertEqual(result.parameters[1], 0.897222, accuracy: 1e-4)  // slope
        XCTAssertEqual(result.lossHistory.first!, log(2.0), accuracy: 1e-12)
        XCTAssertEqual(result.lossHistory.last!, 0.608465, accuracy: 1e-5)
    }

    // MARK: - Refactor equivalence

    // The MSE delegating overload (descend without a strategy) must produce
    // results identical to calling the shared loop with _MeanSquaredErrorStrategy
    // directly — the proof that the old call path is now a faithful thin wrapper.
    func testMSEDelegationMatchesExplicitStrategy() throws {
        let raw = [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]]
        let targets = [3.1, 2.9, 7.2, 6.8, 11.5]
        let X = withIntercept(standardize(raw))

        // Path A: the historic signature (lambda defaulted), which now delegates.
        let viaDelegate = try _GradientDescent.descend(
            features: X, targets: targets,
            learningRate: 0.05, maxIterations: 5_000, tolerance: 1e-9
        )
        // Path B: the shared loop with the strategy spelled out explicitly.
        let viaStrategy = try _GradientDescent.descend(
            features: X, targets: targets,
            learningRate: 0.05, maxIterations: 5_000, tolerance: 1e-9,
            strategy: _MeanSquaredErrorStrategy(lambda: 0, penalizeFromIndex: 0)
        )

        XCTAssertEqual(viaDelegate.parameters, viaStrategy.parameters)
        XCTAssertEqual(viaDelegate.lossHistory, viaStrategy.lossHistory)
        XCTAssertEqual(viaDelegate.iterations, viaStrategy.iterations)
        XCTAssertEqual(viaDelegate.outcome, viaStrategy.outcome)
    }

    // The same Ridge penalty routed through the delegating overload and the
    // explicit strategy must also agree, confirming the penalized path delegates
    // faithfully too.
    func testRidgeDelegationMatchesExplicitStrategy() throws {
        let raw = [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]]
        let targets = [3.1, 2.9, 7.2, 6.8, 11.5]
        let X = withIntercept(standardize(raw))

        let viaDelegate = try _GradientDescent.descend(
            features: X, targets: targets,
            learningRate: 0.05, maxIterations: 5_000, tolerance: 1e-9,
            lambda: 0.3, penalizeFromIndex: 1
        )
        let viaStrategy = try _GradientDescent.descend(
            features: X, targets: targets,
            learningRate: 0.05, maxIterations: 5_000, tolerance: 1e-9,
            strategy: _MeanSquaredErrorStrategy(lambda: 0.3, penalizeFromIndex: 1)
        )

        XCTAssertEqual(viaDelegate.parameters, viaStrategy.parameters)
        XCTAssertEqual(viaDelegate.lossHistory, viaStrategy.lossHistory)
    }

    // MARK: - Shared divergence contract

    // Both objectives throw through the same DivergenceCause channel. An
    // over-large rate on raw-scale features overshoots and throws.
    func testBothObjectivesShareDivergenceChannel() {
        let X = [[1.0, 100.0], [1.0, 200.0], [1.0, 300.0], [1.0, 400.0]]
        let mseTargets = [1.0, 2.0, 3.0, 4.0]
        let ceTargets = [0.0, 0.0, 1.0, 1.0]

        XCTAssertThrowsError(try _GradientDescent.descend(
            features: X, targets: mseTargets,
            learningRate: 10.0, maxIterations: 1_000, tolerance: 1e-6,
            strategy: _MeanSquaredErrorStrategy(lambda: 0, penalizeFromIndex: 0)
        )) { XCTAssertTrue($0 is _GradientDescent.DivergenceCause) }

        XCTAssertThrowsError(try _GradientDescent.descend(
            features: X, targets: ceTargets,
            learningRate: 100.0, maxIterations: 1_000, tolerance: 1e-6,
            strategy: _CrossEntropyStrategy()
        )) { XCTAssertTrue($0 is _GradientDescent.DivergenceCause) }
    }
}
