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

/// Bayes' theorem exposed as a stateless namespace.
///
/// `Bayes` groups the three forms of Bayes' theorem taught in introductory
/// statistics. The two-hypothesis forms are stateless functions on this
/// namespace; the multi-hypothesis form takes structured `BayesPrior` and
/// `BayesLikelihood` inputs and returns a labeled `BayesPosterior`.
///
/// All forms return optionals. Out-of-domain input — probabilities outside
/// `[0, 1]`, priors that do not sum to one, zero evidence — returns `nil`
/// rather than trapping or propagating `NaN`. This matches the convention
/// used across the `Distributions` namespace.
public enum Bayes: Sendable {

    /// Returns the posterior probability of a hypothesis given evidence.
    ///
    /// Computes P(H | E) using the basic form of Bayes' theorem:
    ///
    ///     P(H | E) = P(E | H) · P(H) / P(E)
    ///
    /// Use this form when the marginal probability of the evidence is known
    /// directly. When only the conditional probabilities are known —
    /// sensitivity P(E | H) and false-positive rate P(E | ¬H) — use the
    /// expanded form ``posterior(prior:trueRate:falsePositiveRate:)``
    /// instead.
    ///
    /// - Parameters:
    ///   - prior: The prior probability of the hypothesis, P(H), in `[0, 1]`.
    ///   - likelihood: The conditional probability of the evidence given
    ///     the hypothesis, P(E | H), in `[0, 1]`.
    ///   - evidence: The marginal probability of the evidence, P(E),
    ///     in `(0, 1]`. Zero would imply division by zero.
    /// - Returns: The posterior probability P(H | E) in `[0, 1]`, or `nil`
    ///   if any input is out of domain or the resulting posterior is not
    ///   finite.
    public static func posterior(
        prior: Double,
        likelihood: Double,
        evidence: Double
    ) -> Double? {
        guard (0.0...1.0).contains(prior),
              (0.0...1.0).contains(likelihood),
              (0.0...1.0).contains(evidence),
              evidence > 0.0 else {
            return nil
        }
        let result = (likelihood * prior) / evidence
        guard result.isFinite, (0.0...1.0).contains(result) else {
            return nil
        }
        return result
    }

    /// Returns the posterior probability of a hypothesis given a positive test.
    ///
    /// Computes P(H | positive) using the expanded form of Bayes' theorem,
    /// which derives the marginal probability of a positive test from the
    /// test's sensitivity and false-positive rate via the law of total
    /// probability:
    ///
    ///     P(positive) = P(positive | H) · P(H) + P(positive | ¬H) · P(¬H)
    ///     P(H | positive) = P(positive | H) · P(H) / P(positive)
    ///
    /// This is the form used to evaluate diagnostic tests, quality-control
    /// tests, and binary classifiers. It is the canonical example used in
    /// every introductory statistics course — given a low base rate and a
    /// test with high sensitivity but a nonzero false-positive rate, the
    /// posterior probability of actually having the condition after a
    /// positive test is often counterintuitively low.
    ///
    /// ```swift
    /// // 1% prevalence, 95% sensitivity, 2% false-positive rate.
    /// let posterior = Bayes.posterior(
    ///     prior: 0.01,
    ///     trueRate: 0.95,
    ///     falsePositiveRate: 0.02
    /// )
    /// // posterior ≈ 0.324 — a positive test on a rare condition still
    /// // implies only a ~32% chance of having it.
    /// ```
    ///
    /// - Parameters:
    ///   - prior: The prior probability of the hypothesis, P(H), in `[0, 1]`.
    ///   - trueRate: The test sensitivity, P(positive | H), in `[0, 1]`.
    ///   - falsePositiveRate: The false-positive rate, P(positive | ¬H),
    ///     in `[0, 1]`.
    /// - Returns: The posterior probability P(H | positive) in `[0, 1]`,
    ///   or `nil` if any input is out of domain or the implied marginal
    ///   P(positive) is zero.
    public static func posterior(
        prior: Double,
        trueRate: Double,
        falsePositiveRate: Double
    ) -> Double? {
        guard (0.0...1.0).contains(prior),
              (0.0...1.0).contains(trueRate),
              (0.0...1.0).contains(falsePositiveRate) else {
            return nil
        }
        let notPrior = 1.0 - prior
        let truePart = trueRate * prior
        let falsePart = falsePositiveRate * notPrior
        let evidence = truePart + falsePart
        guard evidence > 0.0 else { return nil }
        let result = truePart / evidence
        guard result.isFinite, (0.0...1.0).contains(result) else {
            return nil
        }
        return result
    }

    /// Returns the posterior distribution over a set of competing hypotheses.
    ///
    /// Computes P(H_i | E) for every hypothesis `H_i` given a structured
    /// `BayesPrior` and `BayesLikelihood`. This is the multi-hypothesis
    /// generalization of Bayes' theorem — the form used internally by
    /// classifiers like ``GaussianNaiveBayes`` to combine prior probabilities
    /// with per-class likelihoods.
    ///
    /// The public input is linear-space (priors and likelihoods are passed
    /// as ordinary probabilities). Internally, the computation runs in
    /// log-space and normalizes via ``Swift/Array/softMax()-(Self)``, which
    /// applies the same max-subtraction trick that `_Bayes.logSumExp` does.
    /// `softMax` is the single source of truth for log-space normalization
    /// across Quiver — `GaussianNaiveBayes.predictProbabilities` calls it
    /// for the same reason.
    ///
    /// - Parameters:
    ///   - prior: A labeled prior distribution. Probabilities must sum to one.
    ///   - likelihood: The likelihood P(E | H_i) for each hypothesis,
    ///     ordered to match the prior's hypotheses.
    /// - Returns: A `BayesPosterior` carrying the same hypothesis labels
    ///   with their updated probabilities, or `nil` if the hypothesis
    ///   counts do not match or every joint probability is zero.
    public static func posterior(
        prior: BayesPrior,
        likelihood: BayesLikelihood
    ) -> BayesPosterior? {
        guard prior.probabilities.count == likelihood.perHypothesis.count else {
            return nil
        }
        let count = prior.probabilities.count
        guard count > 0 else { return nil }

        var logJoint = [Double]()
        logJoint.reserveCapacity(count)
        for i in 0..<count {
            let p = prior.probabilities[i]
            let l = likelihood.perHypothesis[i]
            if p == 0.0 || l == 0.0 {
                logJoint.append(-.infinity)
            } else {
                logJoint.append(Foundation.log(p) + Foundation.log(l))
            }
        }

        // Every joint probability collapsed to zero — the evidence is
        // impossible under every hypothesis, so the posterior is undefined.
        guard logJoint.contains(where: { $0 > -.infinity }) else {
            return nil
        }

        let posteriors = logJoint.softMax()

        return BayesPosterior(
            hypotheses: prior.hypotheses,
            probabilities: posteriors
        )
    }
}

// MARK: - BayesPrior

/// A labeled prior distribution over a set of mutually exclusive hypotheses.
///
/// `BayesPrior` pairs hypothesis names with their prior probabilities. The
/// failable initializer validates that the hypothesis labels and probabilities
/// match in count, that every probability is in `[0, 1]`, and that the
/// probabilities sum to one within a small tolerance.
public struct BayesPrior: Equatable, Codable, Sendable, CustomStringConvertible {

    /// The hypothesis labels, in the same order as ``probabilities``.
    public let hypotheses: [String]

    /// The prior probability of each hypothesis. Sums to `1.0 ± 1e-9`.
    public let probabilities: [Double]

    /// Creates a labeled prior, validating that the inputs are well formed.
    ///
    /// Returns `nil` when the label and probability counts disagree, any
    /// probability is outside `[0, 1]`, or the probabilities do not sum
    /// to one within `1e-9`.
    public init?(hypotheses: [String], probabilities: [Double]) {
        guard hypotheses.count == probabilities.count,
              !hypotheses.isEmpty else {
            return nil
        }
        for p in probabilities {
            guard p.isFinite, (0.0...1.0).contains(p) else { return nil }
        }
        let total = probabilities.reduce(0.0, +)
        guard Foundation.fabs(total - 1.0) < 1e-9 else { return nil }
        self.hypotheses = hypotheses
        self.probabilities = probabilities
    }

    /// Returns a Markdown table of the prior distribution.
    ///
    /// Two columns — hypothesis and prior probability — with probabilities
    /// formatted to four decimal places.
    public func markdownTable() -> String {
        var lines = ["| Hypothesis | Prior |", "| --- | ---: |"]
        for (label, probability) in zip(hypotheses, probabilities) {
            lines.append("| \(label) | \(String(format: "%.4f", probability)) |")
        }
        return lines.joined(separator: "\n")
    }

    /// Returns a single CSV row of prior probabilities, in hypothesis order.
    public func csvRow() -> String {
        return probabilities
            .map { String(format: "%.6f", $0) }
            .joined(separator: ",")
    }

    public var description: String {
        let pairs = zip(hypotheses, probabilities)
            .map { "\($0): \(String(format: "%.4f", $1))" }
            .joined(separator: ", ")
        return "BayesPrior(\(pairs))"
    }
}

// MARK: - BayesLikelihood

/// The likelihood P(E | H_i) for each hypothesis under consideration.
///
/// The order of ``perHypothesis`` must match the order of the associated
/// ``BayesPrior``'s hypotheses. Likelihoods are not required to sum to one —
/// each value is an independent conditional probability.
public struct BayesLikelihood: Equatable, Codable, Sendable, CustomStringConvertible {

    /// The likelihood P(E | H_i) for each hypothesis, in matching order.
    public let perHypothesis: [Double]

    /// Creates a likelihood vector, validating that every entry is in `[0, 1]`.
    public init?(_ values: [Double]) {
        guard !values.isEmpty else { return nil }
        for v in values {
            guard v.isFinite, (0.0...1.0).contains(v) else { return nil }
        }
        self.perHypothesis = values
    }

    /// Returns a Markdown table of the per-hypothesis likelihoods.
    ///
    /// Two columns — hypothesis index and likelihood — with values formatted
    /// to four decimal places. Indices are 1-based to match how the values
    /// read alongside a printed `BayesPrior`.
    public func markdownTable() -> String {
        var lines = ["| Hypothesis | Likelihood |", "| --- | ---: |"]
        for (index, value) in perHypothesis.enumerated() {
            lines.append("| H\(index + 1) | \(String(format: "%.4f", value)) |")
        }
        return lines.joined(separator: "\n")
    }

    /// Returns a single CSV row of likelihoods, in hypothesis order.
    public func csvRow() -> String {
        return perHypothesis
            .map { String(format: "%.6f", $0) }
            .joined(separator: ",")
    }

    public var description: String {
        let formatted = perHypothesis
            .map { String(format: "%.4f", $0) }
            .joined(separator: ", ")
        return "BayesLikelihood(\(formatted))"
    }
}

// MARK: - BayesPosterior

/// A labeled posterior distribution returned from multi-hypothesis Bayes.
///
/// Carries the same hypothesis labels as the input ``BayesPrior`` along with
/// the updated probabilities computed from the prior and likelihood. The
/// probabilities sum to one.
public struct BayesPosterior: Equatable, Codable, Sendable, CustomStringConvertible {

    /// The hypothesis labels, in the same order as ``probabilities``.
    public let hypotheses: [String]

    /// The posterior probability of each hypothesis. Sums to `1.0`.
    public let probabilities: [Double]

    /// Returns a Markdown table summarizing the posterior distribution.
    ///
    /// The output has two columns — hypothesis and probability — with the
    /// probabilities formatted to four decimal places.
    public func markdownTable() -> String {
        var lines = ["| Hypothesis | Probability |", "| --- | ---: |"]
        for (label, probability) in zip(hypotheses, probabilities) {
            lines.append("| \(label) | \(String(format: "%.4f", probability)) |")
        }
        return lines.joined(separator: "\n")
    }

    /// Returns a single CSV row of probabilities, in hypothesis order.
    public func csvRow() -> String {
        return probabilities
            .map { String(format: "%.6f", $0) }
            .joined(separator: ",")
    }

    public var description: String {
        let pairs = zip(hypotheses, probabilities)
            .map { "\($0): \(String(format: "%.4f", $1))" }
            .joined(separator: ", ")
        return "BayesPosterior(\(pairs))"
    }
}
