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

/// What the seeding init and a decode-time refit can surface, as one catchable type, so a watch
/// app never traps. The empty init does not throw, and the live loop does not throw, so `try`
/// appears only when seeding from existing runs or decoding. `.baselineDiverged` nests the
/// underlying `GradientDescentError` whole, preserving its typed payload and description.
public enum TESError: Error, Equatable, CustomStringConvertible, Sendable {
    /// The seeding init was asked for more neighbors than it has labeled examples. The empty init
    /// preconditions `k <= anchorSamples.count` instead.
    case insufficientLabeledSamples(k: Int, available: Int)

    /// The Ridge baseline diverged while fitting, preserved whole.
    case baselineDiverged(GradientDescentError)

    public var description: String {
        switch self {
        case let .insufficientLabeledSamples(k, available):
            return "Insufficient labeled samples: k = \(k) but only \(available) labeled "
                + "example(s) are available. Lower k or supply more labeled samples."
        case let .baselineDiverged(error):
            return "Baseline fit diverged: \(error.description)"
        }
    }
}
