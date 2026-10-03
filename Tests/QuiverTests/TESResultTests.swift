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

final class TESResultTests: XCTestCase {

    // MARK: - Helpers

    /// Records one two-minute run whose workload and heart rate vary, returning each sample's trust.
    @discardableResult
    private func recordRun(_ tes: inout TrueEffortScore, start: Date, lift: Double) -> [Double] {
        var trusts: [Double] = []
        for i in 0..<120 {
            let pace = 5.0 + 0.01 * Double(i)
            let grade = Double(i % 7) - 3.0
            let trust = i % 4 == 0 ? 0.5 : 1.0
            tes.record(heartRate: 120 + 6 * pace + 1.5 * grade + lift,
                       pace: pace, cadence: 165 + Double(i % 9), grade: grade,
                       verticalOscillation: 7.5 + 0.1 * Double(i % 5),
                       altitude: 100 + Double(i % 11),
                       at: start.addingTimeInterval(Double(i)), hrTrust: trust)
            trusts.append(trust)
        }
        return trusts
    }

    // MARK: - Expected heart rate

    // A first run has no baseline, so there is no expected heart rate
    func testExpectedHeartRateIsNilOnFirstRun() throws {
        var tes = TrueEffortScore()
        recordRun(&tes, start: Date(timeIntervalSince1970: 1_000_000), lift: 0)
        let result = try XCTUnwrap(tes.finalize())

        XCTAssertNil(result.expectedHeartRate)
    }

    // Expected heart rate plus the mean residual is the run's weighted average heart rate
    func testExpectedHeartRatePlusResidualIsAverageHeartRate() throws {
        var tes = TrueEffortScore()
        recordRun(&tes, start: Date(timeIntervalSince1970: 1_000_000), lift: 0)
        XCTAssertNotNil(tes.finalize())

        let trusts = recordRun(&tes, start: Date(timeIntervalSince1970: 1_086_400), lift: 4)
        let result = try XCTUnwrap(tes.finalize())
        let expected = try XCTUnwrap(result.expectedHeartRate)

        let run = try XCTUnwrap(tes.history.last)
        var numerator = 0.0, denominator = 0.0
        for (i, heartRate) in run.heartRates.enumerated() {
            let weight = trusts[i] * run.deltaTimes[i]
            numerator += weight * heartRate
            denominator += weight
        }

        XCTAssertEqual(expected + result.meanResidual, numerator / denominator, accuracy: 1e-9)
    }
}
