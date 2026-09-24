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

final class TrueEffortScoreTests: XCTestCase {

    // MARK: - Helpers (fixtures live here, not in the API)

    /// Records `count` threshold-effort samples one second apart into the model.
    private func recordThreshold(
        _ tes: inout TrueEffortScore,
        count: Int,
        start: Date = Date(timeIntervalSince1970: 1_000_000),
        heartRate: Double = 170,
        hrTrust: Double = 1.0
    ) {
        for i in 0..<count {
            tes.record(heartRate: heartRate, pace: 4.8, cadence: 180, grade: 0.0,
                       verticalOscillation: 7.0, altitude: 100,
                       at: start.addingTimeInterval(Double(i)), hrTrust: hrTrust)
        }
    }

    // MARK: - Anchors

    func testAnchorSamplesAndLabelsAreParallel() {
        XCTAssertEqual(TrueEffortScore.anchorSamples.count, TrueEffortScore.anchorLabels.count)
        XCTAssertFalse(TrueEffortScore.anchorSamples.isEmpty)
    }

    func testEffortClassWeightsAndOrdinals() {
        XCTAssertEqual(EffortClass.easy.weight, 0.25)
        XCTAssertEqual(EffortClass.tempo.weight, 0.50)
        XCTAssertEqual(EffortClass.threshold.weight, 0.75)
        XCTAssertEqual(EffortClass.hard.weight, 1.00)
        XCTAssertEqual(EffortClass.allCases.map(\.ordinal), [0, 1, 2, 3])
    }

    // The top band stores and displays as "hard", the name the white paper argues for
    func testHardBandRawValueAndLabel() {
        XCTAssertEqual(EffortClass.hard.rawValue, "hard")
        XCTAssertEqual(EffortClass.hard.label, "Hard")
        XCTAssertEqual(EffortClass(rawValue: "hard"), .hard)
    }

    // MARK: - Construction

    func testColdStartHasNoBaselineButScoresFromFirstTick() {
        var tes = TrueEffortScore()
        XCTAssertNil(tes.baseline)
        XCTAssertEqual(tes.sessionCount, 0)
        recordThreshold(&tes, count: 120)
        XCTAssertGreaterThan(tes.currentScore, 0)
    }

    func testSeedingInsufficientLabeledSamplesThrows() {
        let k = TrueEffortScore.anchorSamples.count + 5
        XCTAssertThrowsError(try TrueEffortScore(history: [], k: k)) { error in
            guard case TESError.insufficientLabeledSamples = error else {
                return XCTFail("expected insufficientLabeledSamples, got \(error)")
            }
        }
    }

    // MARK: - Live scoring

    func testFirstSampleHasZeroDeltaAndZeroLoad() {
        var tes = TrueEffortScore()
        tes.record(heartRate: 170, pace: 4.8, cadence: 180, grade: 0.0,
                   verticalOscillation: 7.0, altitude: 100, at: Date(timeIntervalSince1970: 0))
        XCTAssertEqual(tes.sampleCount, 1)
        XCTAssertEqual(tes.timerTime, 0)
        XCTAssertEqual(tes.currentScore, 0)
    }

    func testDuplicateAndOutOfOrderSamplesAreDiscarded() {
        var tes = TrueEffortScore()
        let t0 = Date(timeIntervalSince1970: 100)
        func rec(_ t: Date) {
            tes.record(heartRate: 170, pace: 4.8, cadence: 180, grade: 0.0,
                       verticalOscillation: 7.0, altitude: 100, at: t)
        }
        rec(t0)
        rec(t0)                        // duplicate
        rec(t0.addingTimeInterval(-5)) // out of order
        XCTAssertEqual(tes.sampleCount, 1)
    }

    func testCurrentScoreClimbsWithTime() {
        var tes = TrueEffortScore()
        let start = Date(timeIntervalSince1970: 0)
        recordThreshold(&tes, count: 30, start: start)
        let early = tes.currentScore
        recordThreshold(&tes, count: 60, start: start.addingTimeInterval(30))
        XCTAssertGreaterThan(tes.currentScore, early)
    }

    // MARK: - Pause / resume clock

    func testPausedSpanCountsElapsedNotTimer() {
        var tes = TrueEffortScore()
        let start = Date(timeIntervalSince1970: 1000)
        for i in 0...10 {
            tes.record(heartRate: 170, pace: 4.8, cadence: 180, grade: 0.0,
                       verticalOscillation: 7.0, altitude: 100, at: start.addingTimeInterval(Double(i)))
        }
        XCTAssertEqual(tes.timerTime, 10, accuracy: 0.001)
        XCTAssertEqual(tes.elapsedTime, 10, accuracy: 0.001)

        tes.pause()
        tes.resume()
        let resumeAt = start.addingTimeInterval(40)   // 30s paused gap
        for i in 0...5 {
            tes.record(heartRate: 170, pace: 4.8, cadence: 180, grade: 0.0,
                       verticalOscillation: 7.0, altitude: 100, at: resumeAt.addingTimeInterval(Double(i)))
        }
        XCTAssertEqual(tes.timerTime, 15, accuracy: 0.001, "active time excludes the paused span")
        XCTAssertEqual(tes.elapsedTime, 45, accuracy: 0.001, "wall clock includes the paused span")
    }

    func testResumeDoesNotRewriteRunStartDate() {
        var tes = TrueEffortScore()
        let start = Date(timeIntervalSince1970: 5000)
        recordThreshold(&tes, count: 71, start: start)
        tes.pause(); tes.resume()
        recordThreshold(&tes, count: 71, start: start.addingTimeInterval(500))
        _ = tes.finalize()
        XCTAssertEqual(tes.history.first?.startDate, start,
                       "resume must not corrupt the run's start date")
    }

    // MARK: - Finalize

    func testFinalizeBelowFloorReturnsNilAndClears() {
        var tes = TrueEffortScore()
        recordThreshold(&tes, count: 30)   // 29s active, below the 60s floor
        XCTAssertNil(tes.finalize())
        XCTAssertEqual(tes.sampleCount, 0)
        XCTAssertEqual(tes.sessionCount, 0)
    }

    func testFinalizeAboveFloorReturnsResultAndFolds() {
        var tes = TrueEffortScore()
        recordThreshold(&tes, count: 120)   // 119s active
        let result = tes.finalize()
        XCTAssertNotNil(result)
        XCTAssertEqual(tes.sessionCount, 1)
        XCTAssertEqual(tes.sampleCount, 0)
        XCTAssertNotNil(tes.baseline)
        if let r = result {
            XCTAssertGreaterThan(r.raw, 0)
            XCTAssertEqual(r.timerTime, 119, accuracy: 0.001)
        }
    }

    // MARK: - Persistence (synthesized Codable)

    func testCodableRoundTripPreservesHistoryAndBaseline() throws {
        var tes = TrueEffortScore()
        recordThreshold(&tes, count: 120)
        _ = tes.finalize()
        let data = try JSONEncoder().encode(tes)
        let restored = try JSONDecoder().decode(TrueEffortScore.self, from: data)
        XCTAssertEqual(restored, tes)
        XCTAssertEqual(restored.sessionCount, 1)
        XCTAssertEqual(restored.baseline?.coefficients, tes.baseline?.coefficients)
    }

    func testDecodeStartsWithCleanLiveBuffer() throws {
        var tes = TrueEffortScore()
        recordThreshold(&tes, count: 120)
        _ = tes.finalize()
        recordThreshold(&tes, count: 1, start: Date(timeIntervalSince1970: 9_000_000))
        let data = try JSONEncoder().encode(tes)
        let restored = try JSONDecoder().decode(TrueEffortScore.self, from: data)
        XCTAssertEqual(restored.sampleCount, 0)
    }

    // MARK: - Baseline inspection (anti-black-box)

    func testRefitProducesLabeledExpression() {
        var tes = TrueEffortScore()
        let start = Date(timeIntervalSince1970: 0)
        for i in 0..<120 {
            let hard = i % 2 == 0
            tes.record(heartRate: hard ? 170 : 132, pace: hard ? 4.8 : 6.5,
                       cadence: hard ? 180 : 165, grade: 0.0,
                       verticalOscillation: hard ? 7.0 : 8.5, altitude: 100,
                       at: start.addingTimeInterval(Double(i)))
        }
        _ = tes.finalize()
        XCTAssertNotNil(tes.baseline)
        let labeled = tes.baseline?.labeledExpression ?? ""
        XCTAssertTrue(labeled.contains("pace"))
        XCTAssertTrue(labeled.contains("expected HR ="))
    }

    // MARK: - hrTrust

    func testHRTrustZeroDoesNotIntensifyClassification() {
        var tes = TrueEffortScore()
        // A misleading sample: HR screams hard, kinematics are easy.
        tes.record(heartRate: 185, pace: 7.0, cadence: 160, grade: 0.0,
                   verticalOscillation: 9.0, altitude: 100, at: Date(timeIntervalSince1970: 0),
                   hrTrust: 1.0)
        let trusted = tes.currentEffort
        tes.discardRun()
        tes.record(heartRate: 185, pace: 7.0, cadence: 160, grade: 0.0,
                   verticalOscillation: 9.0, altitude: 100, at: Date(timeIntervalSince1970: 0),
                   hrTrust: 0.0)
        let doubted = tes.currentEffort
        XCTAssertNotNil(trusted)
        XCTAssertNotNil(doubted)
        if let t = trusted, let d = doubted {
            XCTAssertLessThanOrEqual(d.ordinal, t.ordinal)
        }
    }

    // A fully doubted power-hike keeps its top-band label: heart rate is held at the anchor
    // mean in beats per minute, so the climb's pace, cadence, and grade still decide it
    func testHRTrustZeroPowerHikeStaysInTopBand() {
        var tes = TrueEffortScore()
        tes.record(heartRate: 166, pace: 9.6, cadence: 144, grade: 9.5,
                   verticalOscillation: 6.4, altitude: 100, at: Date(timeIntervalSince1970: 0),
                   hrTrust: 0.0)
        XCTAssertEqual(tes.currentEffort, .hard)
    }

    // The blend target is the anchor heart-rate mean in beats per minute, not the mean of the
    // standardized training rows (about zero)
    func testHRTrustBlendTargetIsRawAnchorMean() {
        let tes = TrueEffortScore()
        let anchorMean = TrueEffortScore.anchorSamples.map { $0[0] }.mean() ?? 0
        XCTAssertEqual(tes.classifier.scaler.means[0], anchorMean, accuracy: 1e-9)
        XCTAssertEqual(anchorMean, 150.571, accuracy: 0.001)
    }

    // MARK: - Signal z-scores (the "why" surface)

    func testCurrentSignalsNilBeforeFirstSample() {
        let tes = TrueEffortScore()
        XCTAssertNil(tes.currentSignals)
    }

    func testDownhillShowsAsGradeOutlierWithNormalHeartRate() {
        var tes = TrueEffortScore()
        // Steep downhill: fast pace, calm HR, sharply negative grade.
        tes.record(heartRate: 132, pace: 5.0, cadence: 158, grade: -6.0,
                   verticalOscillation: 11.0, altitude: 100, at: Date(timeIntervalSince1970: 0))
        let z = tes.currentSignals
        XCTAssertNotNil(z)
        if let z {
            // Grade is a strong negative outlier; heart rate is not elevated.
            XCTAssertLessThan(z.grade, -1.0, "steep downhill should read as a negative grade outlier")
            XCTAssertLessThan(abs(z.heartRate), abs(z.grade),
                              "the calm heart should be far less of an outlier than the grade")
            XCTAssertEqual(z.asArray.count, 5)
        }
    }

    // MARK: - Discard / history limit

    func testDiscardRunClearsWithoutFolding() {
        var tes = TrueEffortScore()
        recordThreshold(&tes, count: 120)
        tes.discardRun()
        XCTAssertEqual(tes.sampleCount, 0)
        XCTAssertEqual(tes.sessionCount, 0)
    }

    func testHistoryLimitTrimsOldestRuns() {
        var tes = TrueEffortScore(historyLimit: 2)
        let base = Date(timeIntervalSince1970: 0)
        for run in 0..<4 {
            recordThreshold(&tes, count: 120, start: base.addingTimeInterval(Double(run) * 10_000))
            _ = tes.finalize()
        }
        XCTAssertEqual(tes.sessionCount, 2)
    }
}
