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
import Foundation
@testable import Quiver

final class BayesTests: XCTestCase {

    // MARK: - _Bayes.logSumExp

    func testLogSumExpUniform() {
        // log(sum(exp([0, 0, 0]))) = log(3) ≈ 1.0986
        let result = _Bayes.logSumExp([0, 0, 0])
        XCTAssertEqual(result, Foundation.log(3.0), accuracy: 1e-12)
    }

    func testLogSumExpStabilityAtLargeValues() {
        // log(sum(exp([1000, 1000]))) = 1000 + log(2). Direct exp(1000)
        // overflows; the max-subtraction trick keeps this representable.
        let result = _Bayes.logSumExp([1000, 1000])
        XCTAssertEqual(result, 1000.0 + Foundation.log(2.0), accuracy: 1e-9)
    }

    func testLogSumExpStabilityAtSmallValues() {
        // log(sum(exp([-1000, -1000]))) = -1000 + log(2).
        let result = _Bayes.logSumExp([-1000, -1000])
        XCTAssertEqual(result, -1000.0 + Foundation.log(2.0), accuracy: 1e-9)
    }

    func testLogSumExpEmptyReturnsNegativeInfinity() {
        XCTAssertEqual(_Bayes.logSumExp([]), -.infinity)
    }

    // MARK: - Basic Bayes.posterior(prior:likelihood:evidence:)

    func testBasicPosteriorTextbook() {
        // Reference: textbook example with explicit evidence.
        // prior = 0.5, likelihood = 0.8, evidence = 0.4 → posterior = 1.0
        guard let result = Bayes.posterior(prior: 0.5, likelihood: 0.8, evidence: 0.4) else {
            XCTFail("basic posterior returned nil"); return
        }
        XCTAssertEqual(result, 1.0, accuracy: 1e-12)
    }

    func testBasicPosteriorRejectsZeroEvidence() {
        XCTAssertNil(Bayes.posterior(prior: 0.5, likelihood: 0.8, evidence: 0.0))
    }

    func testBasicPosteriorRejectsOutOfDomain() {
        XCTAssertNil(Bayes.posterior(prior: -0.1, likelihood: 0.5, evidence: 0.5))
        XCTAssertNil(Bayes.posterior(prior: 1.1, likelihood: 0.5, evidence: 0.5))
        XCTAssertNil(Bayes.posterior(prior: 0.5, likelihood: -0.1, evidence: 0.5))
        XCTAssertNil(Bayes.posterior(prior: 0.5, likelihood: 1.1, evidence: 0.5))
        XCTAssertNil(Bayes.posterior(prior: 0.5, likelihood: 0.5, evidence: 1.1))
    }

    func testBasicPosteriorRejectsInconsistentInputs() {
        // prior * likelihood > evidence implies posterior > 1.
        XCTAssertNil(Bayes.posterior(prior: 0.9, likelihood: 0.9, evidence: 0.1))
    }

    // MARK: - Expanded Bayes.posterior(prior:trueRate:falsePositiveRate:)

    func testExpandedPosteriorPeanutAllergy() {
        // Canonical intro-stats example. 1% prevalence, 95% sensitivity,
        // 2% false-positive rate. Posterior ≈ 0.3243.
        guard let result = Bayes.posterior(
            prior: 0.01,
            trueRate: 0.95,
            falsePositiveRate: 0.02
        ) else {
            XCTFail("expanded posterior returned nil"); return
        }
        XCTAssertEqual(result, 0.3242320819112628, accuracy: 1e-12)
    }

    func testExpandedPosteriorPerfectTest() {
        // 100% sensitivity, 0% false-positive rate → posterior = 1.0
        // for any positive prior.
        guard let result = Bayes.posterior(
            prior: 0.001,
            trueRate: 1.0,
            falsePositiveRate: 0.0
        ) else {
            XCTFail("expanded posterior returned nil"); return
        }
        XCTAssertEqual(result, 1.0, accuracy: 1e-12)
    }

    func testExpandedPosteriorUselessTest() {
        // Sensitivity equals false-positive rate → posterior equals prior
        // (the test conveys no information).
        guard let result = Bayes.posterior(
            prior: 0.3,
            trueRate: 0.5,
            falsePositiveRate: 0.5
        ) else {
            XCTFail("expanded posterior returned nil"); return
        }
        XCTAssertEqual(result, 0.3, accuracy: 1e-12)
    }

    func testExpandedPosteriorRejectsOutOfDomain() {
        XCTAssertNil(Bayes.posterior(prior: -0.1, trueRate: 0.5, falsePositiveRate: 0.5))
        XCTAssertNil(Bayes.posterior(prior: 1.1, trueRate: 0.5, falsePositiveRate: 0.5))
        XCTAssertNil(Bayes.posterior(prior: 0.5, trueRate: 1.5, falsePositiveRate: 0.5))
        XCTAssertNil(Bayes.posterior(prior: 0.5, trueRate: 0.5, falsePositiveRate: -0.5))
    }

    func testExpandedPosteriorRejectsImpossibleEvidence() {
        // 0% sensitivity AND 0% false-positive rate → evidence = 0 → nil.
        XCTAssertNil(Bayes.posterior(prior: 0.5, trueRate: 0.0, falsePositiveRate: 0.0))
    }

    // MARK: - BayesPrior, BayesLikelihood, BayesPosterior

    func testBayesPriorValidatesSumToOne() {
        XCTAssertNotNil(BayesPrior(hypotheses: ["A", "B"], probabilities: [0.3, 0.7]))
        XCTAssertNil(BayesPrior(hypotheses: ["A", "B"], probabilities: [0.3, 0.5]))
    }

    func testBayesPriorRejectsMismatchedCounts() {
        XCTAssertNil(BayesPrior(hypotheses: ["A", "B"], probabilities: [1.0]))
    }

    func testBayesPriorRejectsOutOfDomain() {
        XCTAssertNil(BayesPrior(hypotheses: ["A", "B"], probabilities: [-0.1, 1.1]))
    }

    func testBayesLikelihoodValidatesDomain() {
        XCTAssertNotNil(BayesLikelihood([0.95, 0.02]))
        XCTAssertNil(BayesLikelihood([1.5, 0.5]))
        XCTAssertNil(BayesLikelihood([]))
    }

    func testBayesPosteriorMarkdownTable() {
        let posterior = BayesPosterior(hypotheses: ["A", "B"], probabilities: [0.7, 0.3])
        let table = posterior.markdownTable()
        XCTAssertTrue(table.contains("| Hypothesis | Probability |"))
        XCTAssertTrue(table.contains("| --- | ---: |"))
        XCTAssertTrue(table.contains("| A | 0.7000 |"))
        XCTAssertTrue(table.contains("| B | 0.3000 |"))
    }

    func testBayesPosteriorCSVRow() {
        let posterior = BayesPosterior(hypotheses: ["A", "B"], probabilities: [0.7, 0.3])
        XCTAssertEqual(posterior.csvRow(), "0.700000,0.300000")
    }

    func testBayesPriorMarkdownTable() {
        guard let prior = BayesPrior(hypotheses: ["disease", "healthy"], probabilities: [0.01, 0.99]) else {
            XCTFail("prior init failed"); return
        }
        let table = prior.markdownTable()
        XCTAssertTrue(table.contains("| Hypothesis | Prior |"))
        XCTAssertTrue(table.contains("| --- | ---: |"))
        XCTAssertTrue(table.contains("| disease | 0.0100 |"))
        XCTAssertTrue(table.contains("| healthy | 0.9900 |"))
    }

    func testBayesPriorCSVRow() {
        guard let prior = BayesPrior(hypotheses: ["A", "B"], probabilities: [0.25, 0.75]) else {
            XCTFail("prior init failed"); return
        }
        XCTAssertEqual(prior.csvRow(), "0.250000,0.750000")
    }

    func testBayesLikelihoodMarkdownTable() {
        guard let likelihood = BayesLikelihood([0.95, 0.02]) else {
            XCTFail("likelihood init failed"); return
        }
        let table = likelihood.markdownTable()
        XCTAssertTrue(table.contains("| Hypothesis | Likelihood |"))
        XCTAssertTrue(table.contains("| --- | ---: |"))
        XCTAssertTrue(table.contains("| H1 | 0.9500 |"))
        XCTAssertTrue(table.contains("| H2 | 0.0200 |"))
    }

    func testBayesLikelihoodCSVRow() {
        guard let likelihood = BayesLikelihood([0.95, 0.02]) else {
            XCTFail("likelihood init failed"); return
        }
        XCTAssertEqual(likelihood.csvRow(), "0.950000,0.020000")
    }

    // MARK: - Multi-hypothesis Bayes.posterior(prior:likelihood:)

    func testMultiHypothesisRecoversExpandedFormOnTwoHypotheses() {
        // The two-hypothesis multi-form must agree with the expanded form
        // when given matching inputs.
        guard let prior = BayesPrior(
            hypotheses: ["has", "doesNotHave"],
            probabilities: [0.01, 0.99]
        ) else { XCTFail("prior init failed"); return }
        guard let likelihood = BayesLikelihood([0.95, 0.02]) else {
            XCTFail("likelihood init failed"); return
        }
        guard let result = Bayes.posterior(prior: prior, likelihood: likelihood) else {
            XCTFail("multi posterior returned nil"); return
        }
        XCTAssertEqual(result.probabilities[0], 0.3242320819112628, accuracy: 1e-12)
        XCTAssertEqual(result.probabilities[1], 1.0 - 0.3242320819112628, accuracy: 1e-12)
        XCTAssertEqual(result.hypotheses, ["has", "doesNotHave"])
    }

    func testMultiHypothesisThreeWayPosteriorSumsToOne() {
        guard let prior = BayesPrior(
            hypotheses: ["A", "B", "C"],
            probabilities: [0.2, 0.3, 0.5]
        ) else { XCTFail("prior init failed"); return }
        guard let likelihood = BayesLikelihood([0.6, 0.3, 0.1]) else {
            XCTFail("likelihood init failed"); return
        }
        guard let result = Bayes.posterior(prior: prior, likelihood: likelihood) else {
            XCTFail("multi posterior returned nil"); return
        }
        let total = result.probabilities.reduce(0.0, +)
        XCTAssertEqual(total, 1.0, accuracy: 1e-12)
        // Expected: joint = [0.12, 0.09, 0.05], evidence = 0.26
        // posteriors = [0.4615..., 0.3461..., 0.1923...]
        XCTAssertEqual(result.probabilities[0], 0.12 / 0.26, accuracy: 1e-12)
        XCTAssertEqual(result.probabilities[1], 0.09 / 0.26, accuracy: 1e-12)
        XCTAssertEqual(result.probabilities[2], 0.05 / 0.26, accuracy: 1e-12)
    }

    func testMultiHypothesisStableAtExtremeLikelihoods() {
        // Likelihoods that span many orders of magnitude — would underflow
        // in a naive linear-space implementation. The log-space normalizer
        // must still return a valid probability distribution that sums to 1.
        guard let prior = BayesPrior(
            hypotheses: ["A", "B", "C"],
            probabilities: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        ) else { XCTFail("prior init failed"); return }
        // Likelihoods at the smallest representable positive Double would
        // round Foundation.log to a finite negative value; the test uses
        // moderately small values that still exercise the trick.
        guard let likelihood = BayesLikelihood([1e-200, 1e-100, 1e-50]) else {
            XCTFail("likelihood init failed"); return
        }
        guard let result = Bayes.posterior(prior: prior, likelihood: likelihood) else {
            XCTFail("multi posterior returned nil"); return
        }
        let total = result.probabilities.reduce(0.0, +)
        XCTAssertEqual(total, 1.0, accuracy: 1e-12)
        // The largest likelihood dominates entirely.
        XCTAssertEqual(result.probabilities[2], 1.0, accuracy: 1e-9)
    }

    func testMultiHypothesisRejectsMismatchedCounts() {
        guard let prior = BayesPrior(
            hypotheses: ["A", "B"],
            probabilities: [0.5, 0.5]
        ) else { XCTFail("prior init failed"); return }
        guard let likelihood = BayesLikelihood([0.5, 0.5, 0.5]) else {
            XCTFail("likelihood init failed"); return
        }
        XCTAssertNil(Bayes.posterior(prior: prior, likelihood: likelihood))
    }

    func testMultiHypothesisRejectsAllZeroLikelihoods() {
        guard let prior = BayesPrior(
            hypotheses: ["A", "B"],
            probabilities: [0.5, 0.5]
        ) else { XCTFail("prior init failed"); return }
        guard let likelihood = BayesLikelihood([0.0, 0.0]) else {
            XCTFail("likelihood init failed"); return
        }
        XCTAssertNil(Bayes.posterior(prior: prior, likelihood: likelihood))
    }

    // MARK: - Single-source-of-truth check (softMax delegates to _Bayes.logSumExp)

    func testSoftMaxMatchesManualLogSumExpComputation() {
        // softMax was refactored to delegate to _Bayes.logSumExp. Verify
        // it still produces identical results to the closed-form
        // exp(x - logSumExp(x)) pattern.
        let logits = [2.0, 1.0, 0.1]
        let probs = logits.softMax()
        let logNorm = _Bayes.logSumExp(logits)
        let expected = logits.map { Foundation.exp($0 - logNorm) }
        XCTAssertEqual(probs.count, expected.count)
        for (p, e) in zip(probs, expected) {
            XCTAssertEqual(p, e, accuracy: 1e-15)
        }
        XCTAssertEqual(probs.reduce(0.0, +), 1.0, accuracy: 1e-12)
    }
}
