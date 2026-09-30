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

/// Where a run takes place. The watch cannot sense treadmill incline: its barometric altimeter
/// measures a change in altitude, and a runner on an inclined belt never rises. An indoor run
/// is therefore scored at a fixed level-ground grade and kept out of the personal baseline, so a
/// grade the model cannot trust never shapes the runner's outdoor expectations.
///
/// - `outdoor`: the default. Grade is recorded as supplied and the run refits the baseline.
/// - `indoor`: a treadmill or other indoor run. Grade is recorded as
///   `TrueEffortScore.indoorGrade` and the run is scored but not folded into history.
public enum WorkoutLocation: String, Codable, Equatable, Sendable {
    case outdoor
    case indoor
}
