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

// MARK: - Vote Weighting

/// The weighting strategy for neighbor votes in K-Nearest Neighbors.
///
/// Uniform weighting gives each neighbor one vote. Distance weighting gives
/// closer neighbors more influence, which can improve accuracy when the
/// decision boundary is near the query point.
public enum VoteWeight: Codable, Equatable, Sendable {

    /// Each neighbor gets one vote regardless of distance.
    case uniform

    /// Closer neighbors get more influence: weight = 1 / distance.
    ///
    /// When a neighbor has distance 0 (exact match), the model predicts
    /// that neighbor's label immediately without voting.
    case distance
}
