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

// MARK: - Gradient Descent Error

/// Errors thrown by ``GradientDescent`` when the optimizer cannot return a
/// trustworthy result.
///
/// Divergence is the gradient-descent analog of a singular `X'X` in the closed
/// form: the underlying numerics have failed in a way that makes every
/// downstream prediction meaningless. Returning a model full of `NaN` or `±∞`
/// parameters would let that corruption propagate silently into prediction
/// pipelines and downstream models (LogisticRegression, SVM) that share this
/// optimizer. Throwing forces the caller to acknowledge the failure.
///
/// Cap-reached-while-improving is *not* a divergence — that case returns a
/// valid best-so-far estimate with ``GradientDescent/outcome`` set to
/// ``GradientDescent/Outcome/maxIterationsReached``.
public enum GradientDescentError: Error, Equatable, CustomStringConvertible, Sendable {

    /// The iterate produced a non-finite loss (`NaN` or `±∞`) at the given
    /// iteration. Typically caused by a learning rate that is too large for
    /// the feature scale — the parameters overshoot, the predictions blow up,
    /// and the squared residuals overflow.
    case divergedNonFinite(iteration: Int, loss: Double)

    /// The loss strictly increased between two consecutive iterations by more
    /// than one tolerance unit. The descent step overshot the minimum without
    /// yet producing non-finite values. Recovering would require a smaller
    /// learning rate or scaled features.
    case divergedIncreasing(iteration: Int, previousLoss: Double, currentLoss: Double)

    public var description: String {
        switch self {
        case .divergedNonFinite(let iteration, let loss):
            return "GradientDescent diverged at iteration \(iteration): non-finite loss (\(loss)). Reduce the learning rate or standardize features."
        case .divergedIncreasing(let iteration, let previousLoss, let currentLoss):
            let prev = String(format: "%.6g", previousLoss)
            let curr = String(format: "%.6g", currentLoss)
            return "GradientDescent diverged at iteration \(iteration): loss increased from \(prev) to \(curr). Reduce the learning rate or standardize features."
        }
    }
}
