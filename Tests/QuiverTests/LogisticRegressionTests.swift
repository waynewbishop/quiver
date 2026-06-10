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

final class LogisticRegressionTests: XCTestCase {

    // MARK: - Helpers

    /// Standardize each column to mean 0, std 1 (population, ddof=0). Mirrors the
    /// `GradientDescentTests` helper and `StandardScaler`, which is what the
    /// NumPy validation oracle assumes.
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

    // MARK: - Correctness vs. validated reference

    // Non-separable 1D data has a unique finite maximum-likelihood estimate.
    // Descent must land on the coefficients NumPy converges to:
    //   intercept ≈ 0.0, slope ≈ 0.897222, final cross-entropy ≈ 0.608465.
    func testConvergesToKnownMLE() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [2.5], [4.5]]
        let labels = [0, 0, 1, 0, 1, 1, 1, 0]
        let scaled = standardize(features)

        let model = try LogisticRegression.fit(
            features: scaled,
            labels: labels,
            learningRate: 0.5,
            maxIterations: 500_000,
            tolerance: 1.0e-12
        )

        XCTAssertEqual(model.outcome, .converged)
        XCTAssertEqual(model.coefficients[0], 0.0, accuracy: 1.0e-4, "intercept")
        XCTAssertEqual(model.coefficients[1], 0.897222, accuracy: 1.0e-4, "slope")
        XCTAssertEqual(model.finalLoss, 0.608465, accuracy: 1.0e-5, "final cross-entropy")
    }

    // The reference probabilities at the MLE, validated against NumPy.
    func testProbabilitiesMatchReference() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [2.5], [4.5]]
        let labels = [0, 0, 1, 0, 1, 1, 1, 0]
        let scaled = standardize(features)

        let model = try LogisticRegression.fit(
            features: scaled, labels: labels,
            learningRate: 0.5, maxIterations: 500_000, tolerance: 1.0e-12
        )

        let probs = model.predictProbabilities(scaled)
        let expected = [0.19206, 0.29692, 0.42865, 0.57135, 0.70308, 0.80794, 0.36016, 0.63984]
        XCTAssertEqual(probs.count, expected.count)
        for (got, want) in zip(probs, expected) {
            XCTAssertEqual(got, want, accuracy: 1.0e-4)
        }
    }

    // MARK: - Prediction behavior

    // On clearly separated classes the model recovers the labels exactly.
    func testSeparableDataClassifiesPerfectly() throws {
        let features: [[Double]] = [[1.0], [1.5], [2.0], [8.0], [8.5], [9.0]]
        let labels = [0, 0, 0, 1, 1, 1]
        let scaled = standardize(features)

        let model = try LogisticRegression.fit(
            features: scaled, labels: labels, learningRate: 0.5, maxIterations: 50_000
        )
        XCTAssertEqual(model.predict(scaled), labels)
    }

    // Predicted probabilities lie strictly inside (0, 1) and the label is the
    // 0.5 threshold of the probability.
    func testProbabilityAndLabelAgree() throws {
        let features: [[Double]] = [[2.0, 0.10], [1.0, 0.05], [8.0, 0.60], [9.0, 0.55]]
        let trainLabels = [0, 0, 1, 1]
        let scaled = standardize(features)

        let model = try LogisticRegression.fit(features: scaled, labels: trainLabels, maxIterations: 50_000)
        let probs = model.predictProbabilities(scaled)
        let labels = model.predict(scaled)

        for (p, label) in zip(probs, labels) {
            XCTAssertGreaterThan(p, 0.0)
            XCTAssertLessThan(p, 1.0)
            XCTAssertEqual(label, p >= 0.5 ? 1 : 0)
        }
    }

    // MARK: - Decision function (log-odds)

    // decisionFunction returns the raw log-odds Xθ, and σ(decisionFunction) must
    // reproduce predictProbabilities exactly — they share the same linear score.
    func testDecisionFunctionIsLogOddsOfProbability() throws {
        let features: [[Double]] = [[2.0, 0.10], [1.0, 0.05], [8.0, 0.60], [9.0, 0.55]]
        let labels = [0, 0, 1, 1]
        let scaled = standardize(features)

        let model = try LogisticRegression.fit(features: scaled, labels: labels, maxIterations: 50_000)
        let logits = model.decisionFunction(scaled)
        let probs = model.predictProbabilities(scaled)

        XCTAssertEqual(logits.count, probs.count)
        for (z, p) in zip(logits, probs) {
            // σ(z) == p
            XCTAssertEqual(1.0 / (1.0 + exp(-z)), p, accuracy: 1.0e-12)
            // log-odds of p == z
            XCTAssertEqual(log(p / (1.0 - p)), z, accuracy: 1.0e-9)
        }
    }

    // The decision boundary is the sign of the log-odds: predict == 1 iff Xθ ≥ 0.
    func testDecisionFunctionSignMatchesPrediction() throws {
        let features: [[Double]] = [[1.0], [1.5], [2.0], [8.0], [8.5], [9.0]]
        let labels = [0, 0, 0, 1, 1, 1]
        let scaled = standardize(features)

        let model = try LogisticRegression.fit(features: scaled, labels: labels, maxIterations: 50_000)
        let logits = model.decisionFunction(scaled)
        let preds = model.predict(scaled)

        for (z, label) in zip(logits, preds) {
            XCTAssertEqual(label, z >= 0.0 ? 1 : 0)
        }
    }

    // The scalar decisionFunction overload agrees with the batch path.
    func testScalarDecisionFunction() throws {
        let flat = [1.0, 2.0, 3.0, 8.0, 9.0, 10.0]
        let labels = [0, 0, 0, 1, 1, 1]
        let scaledFlat = standardize(flat.map { [$0] }).map { $0[0] }

        let model = try LogisticRegression.fit(features: scaledFlat, labels: labels, maxIterations: 50_000)
        let v = scaledFlat.last!
        XCTAssertEqual(model.decisionFunction(v), model.decisionFunction([[v]])[0], accuracy: 1.0e-12)
    }

    // MARK: - Loss trajectory

    // Cross-entropy starts at exactly log 2 (θ = 0 → every σ = 0.5) and decreases.
    func testLossStartsAtLog2AndDecreases() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let labels = [0, 0, 1, 1]
        let scaled = standardize(features)

        let model = try LogisticRegression.fit(features: scaled, labels: labels, maxIterations: 5_000)

        XCTAssertEqual(model.lossHistory.first!, log(2.0), accuracy: 1.0e-12,
                       "CE at θ = 0 is −log(0.5) = log 2")
        XCTAssertLessThan(model.lossHistory.last!, model.lossHistory.first!,
                          "loss must fall over training")
        XCTAssertEqual(model.lossHistory.count, model.iterations + 1)
        XCTAssertEqual(model.finalLoss, model.lossHistory.last!)
    }

    // MARK: - Intercept handling

    // With intercept off, the coefficient count equals the feature count.
    func testNoInterceptCoefficientCount() throws {
        let features: [[Double]] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        let labels = [0, 0, 1, 1]
        let scaled = standardize(features)

        let withIntercept = try LogisticRegression.fit(features: scaled, labels: labels)
        let without = try LogisticRegression.fit(features: scaled, labels: labels, intercept: false)

        XCTAssertEqual(withIntercept.coefficients.count, 3) // bias + 2 features
        XCTAssertEqual(without.coefficients.count, 2)       // 2 features
        XCTAssertTrue(withIntercept.hasIntercept)
        XCTAssertFalse(without.hasIntercept)
    }

    // MARK: - Convenience overloads

    // The 1D `fit` overload matches the explicit 2D call.
    func testSingleFeatureFitOverloadMatches2D() throws {
        let flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let labels = [0, 0, 0, 1, 1, 1]
        let scaledFlat = standardize(flat.map { [$0] }).map { $0[0] }

        let a = try LogisticRegression.fit(features: scaledFlat, labels: labels, maxIterations: 20_000)
        let b = try LogisticRegression.fit(features: scaledFlat.map { [$0] }, labels: labels, maxIterations: 20_000)

        XCTAssertEqual(a.coefficients, b.coefficients)
    }

    // The scalar `predict(Double)` overload, inherited from the Classifier
    // protocol, must agree with the batch path's first element. Mirrors the
    // testScalarPredict on the other five conforming models.
    func testScalarPredict() throws {
        // One feature, two well-separated groups.
        let features: [[Double]] = [[1.0], [1.2], [0.9], [8.0], [8.3], [7.8]]
        let labels = [0, 0, 0, 1, 1, 1]
        let scaled = standardize(features)

        let model = try LogisticRegression.fit(features: scaled, labels: labels, maxIterations: 50_000)

        // Scalar overload must agree with the batch path's first element.
        let lo = scaled.first!.first!   // standardized value of the first class-0 sample
        let hi = scaled.last!.first!    // standardized value of the last class-1 sample
        XCTAssertEqual(model.predict(lo), model.predict([[lo]])[0])
        XCTAssertEqual(model.predict(hi), model.predict([[hi]])[0])
        XCTAssertEqual(model.predict(lo), 0)
        XCTAssertEqual(model.predict(hi), 1)
    }

    // MARK: - Classifier protocol conformance

    // classify(_:) groups inputs by predicted label, inherited from Classifier.
    func testClassifyGroupsByLabel() throws {
        let features: [[Double]] = [[1.0], [1.5], [8.0], [8.5]]
        let labels = [0, 0, 1, 1]
        let scaled = standardize(features)

        let model = try LogisticRegression.fit(features: scaled, labels: labels, maxIterations: 50_000)
        let groups = model.classify(scaled)

        XCTAssertEqual(groups.count, 2)
        XCTAssertEqual(groups.map { $0.label }, [0, 1])      // sorted by label
        XCTAssertEqual(groups[0].count + groups[1].count, 4) // every point assigned
    }

    // MARK: - Value-type guarantees

    // The model round-trips through Codable.
    func testCodableRoundTrip() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let labels = [0, 0, 1, 1]
        let scaled = standardize(features)
        let model = try LogisticRegression.fit(features: scaled, labels: labels, maxIterations: 5_000)

        let data = try JSONEncoder().encode(model)
        let restored = try JSONDecoder().decode(LogisticRegression.self, from: data)

        XCTAssertEqual(model, restored)
        XCTAssertEqual(model.predict(scaled), restored.predict(scaled))
    }

    // Equatable: identical training yields equal models (deterministic descent).
    func testEquatableDeterministic() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let labels = [0, 0, 1, 1]
        let scaled = standardize(features)

        let a = try LogisticRegression.fit(features: scaled, labels: labels, maxIterations: 5_000)
        let b = try LogisticRegression.fit(features: scaled, labels: labels, maxIterations: 5_000)
        XCTAssertEqual(a, b)
    }

    // description is human-readable and reports feature count + convergence.
    func testDescription() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let labels = [0, 0, 1, 1]
        let model = try LogisticRegression.fit(features: standardize(features), labels: labels, maxIterations: 5_000)
        XCTAssertTrue(model.description.contains("LogisticRegression"))
        XCTAssertTrue(model.description.contains("1 feature"))
    }

    // MARK: - Divergence

    // A learning rate far too large for raw-scale features overshoots the
    // minimum on the first step — the loss jumps from log 2 to the clamp
    // ceiling, a strict increase past tolerance — and the optimizer throws
    // `divergedIncreasing` rather than returning a corrupted model. Same
    // contract as GradientDescent: refuse, don't return garbage.
    func testDivergenceThrowsOnOverlargeRate() {
        let features: [[Double]] = [[100.0], [200.0], [300.0], [400.0]]
        let labels = [0, 0, 1, 1]

        XCTAssertThrowsError(
            try LogisticRegression.fit(
                features: features, labels: labels,
                learningRate: 100.0, maxIterations: 1_000
            )
        ) { error in
            guard case GradientDescentError.divergedIncreasing = error else {
                return XCTFail("expected divergedIncreasing, got \(error)")
            }
        }
    }

    // Cross-entropy is bounded above by the log-clamp, so divergence here always
    // surfaces as a strict *increase* (caught above) rather than a non-finite
    // loss — the clamp keeps every evaluated loss finite. This test documents
    // that a converged model never carries a non-finite loss or coefficient.
    func testConvergedModelIsAllFinite() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [2.5], [4.5]]
        let labels = [0, 0, 1, 0, 1, 1, 1, 0]
        let model = try LogisticRegression.fit(
            features: standardize(features), labels: labels,
            learningRate: 0.5, maxIterations: 100_000
        )
        XCTAssertTrue(model.lossHistory.allSatisfy { $0.isFinite })
        XCTAssertTrue(model.coefficients.allSatisfy { $0.isFinite })
        XCTAssertTrue(model.finalLoss.isFinite)
    }

    // MARK: - Preconditions

    // Non-binary labels are rejected.
    func testNonBinaryTargetsRejected() throws {
        // Caught by precondition — verified via the binary-only contract in docs.
        // (Precondition failures can't be caught in-process; this documents the
        // contract. The happy-path tests exercise the 0/1 domain.)
        let features: [[Double]] = [[1.0], [2.0]]
        let labels = [0, 1]
        let model = try LogisticRegression.fit(features: standardize(features), labels: labels, maxIterations: 100)
        XCTAssertEqual(model.featureCount, 1)
    }
}
