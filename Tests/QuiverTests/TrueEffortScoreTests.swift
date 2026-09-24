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

    // MARK: - Transition debounce

    /// A 40-minute Tempo run with every fortieth sample misread as Hard, the white paper's
    /// section 6.4 flicker case, sampled at `hz`.
    private func flickeringTempo(hz: Double) -> (ordinals: [Double], durations: [TimeInterval]) {
        let count = Int(40 * 60 * hz)
        let ordinals = (0..<count).map { $0 % 40 == 39 ? 3.0 : 1.0 }
        let durations = (0..<count).map { $0 == 0 ? 0.0 : 1.0 / hz }
        return (ordinals, durations)
    }

    /// Consecutive bands held for the given seconds each, sampled at 1 Hz.
    private func heldBands(
        _ holds: [(band: Double, seconds: Int)]
    ) -> (ordinals: [Double], durations: [TimeInterval]) {
        var ordinals: [Double] = []
        for hold in holds {
            ordinals += Array(repeating: hold.band, count: hold.seconds)
        }
        let durations = ordinals.indices.map { $0 == 0 ? 0.0 : 1.0 }
        return (ordinals, durations)
    }

    // The flicker the white paper counts as 238 (1 Hz) and 478 (2 Hz) undebounced adds nothing
    func testDebounceRemovesFlickerAtAnySampleRate() {
        for (hz, undebounced) in [(1.0, 238.0), (2.0, 478.0)] {
            let run = flickeringTempo(hz: hz)
            let raw = zip(run.ordinals, run.ordinals.dropFirst())
                .map { abs($1 - $0) }.filter { $0 >= 2 }.reduce(0, +)
            XCTAssertEqual(raw, undebounced)
            XCTAssertEqual(
                TrueEffortScore.transitionLoad(ordinals: run.ordinals, durations: run.durations), 0)
        }
    }

    // A real surge, Easy to Hard and back, still counts both jumps
    func testDebounceKeepsARealSurge() {
        let run = heldBands([(0, 60), (3, 60), (0, 60)])
        XCTAssertEqual(
            TrueEffortScore.transitionLoad(ordinals: run.ordinals, durations: run.durations), 6)
    }

    // A band held for exactly ten seconds is real; nine seconds is a flicker
    func testDebounceBoundaryIsTenSeconds() {
        let held = heldBands([(1, 60), (3, 10), (1, 60)])
        let flicker = heldBands([(1, 60), (3, 9), (1, 60)])
        XCTAssertEqual(
            TrueEffortScore.transitionLoad(ordinals: held.ordinals, durations: held.durations), 4)
        XCTAssertEqual(
            TrueEffortScore.transitionLoad(
                ordinals: flicker.ordinals, durations: flicker.durations), 0)
    }

    // A short band at the very start takes the first band held long enough
    func testDebounceLeadingShortBandTakesFirstHeldBand() {
        let run = heldBands([(3, 5), (0, 60)])
        XCTAssertEqual(
            TrueEffortScore.debouncedBands(ordinals: run.ordinals, durations: run.durations), [0])
    }

    // MARK: - Classifying against the personal baseline

    // Steady Tempo kinematics from the D5 analysis: raw heart rate crosses into Threshold at
    // about 162 bpm when nothing caps it.
    private let tempoKinematics = (pace: 5.6, cadence: 171.0, grade: 0.5, verticalOscillation: 8.0)

    /// A model with one finished run of Easy, Tempo, and Threshold stretches, so the baseline
    /// expects about 131, 150, and 170 bpm at those three workloads.
    private func establishedModel() -> TrueEffortScore {
        var tes = TrueEffortScore()
        let stretches: [(hr: Double, pace: Double, cadence: Double, grade: Double, vo: Double)] = [
            (131, 6.6, 164, 0.0, 8.7), (150, 5.6, 171, 0.5, 8.0), (170, 4.8, 180, 0.0, 7.0),
        ]
        var second = 0.0
        for _ in 0..<2 {
            for stretch in stretches {
                for i in 0..<120 {
                    tes.record(heartRate: stretch.hr, pace: stretch.pace, cadence: stretch.cadence,
                               grade: stretch.grade, verticalOscillation: stretch.vo,
                               altitude: 100 + Double(i % 3),
                               at: Date(timeIntervalSince1970: second))
                    second += 1
                }
            }
        }
        XCTAssertNotNil(tes.finalize())
        return tes
    }

    /// Records one Tempo-workload moment at the given heart rate and trust, then reads the band.
    private func tempoEffort(_ tes: inout TrueEffortScore, heartRate: Double,
                             hrTrust: Double = 1.0) -> EffortClass? {
        tes.discardRun()
        tes.record(heartRate: heartRate, pace: tempoKinematics.pace,
                   cadence: tempoKinematics.cadence, grade: tempoKinematics.grade,
                   verticalOscillation: tempoKinematics.verticalOscillation,
                   altitude: 101, at: Date(timeIntervalSince1970: 0), hrTrust: hrTrust)
        return tes.currentEffort
    }

    // Drift at a steady workload lifts the band at cold start, which is the documented
    // first-run limit
    func testColdStartDriftLiftsTempoToThreshold() {
        var tes = TrueEffortScore()
        XCTAssertEqual(tempoEffort(&tes, heartRate: 150), .tempo)
        XCTAssertEqual(tempoEffort(&tes, heartRate: 166), .threshold)
    }

    // Once a baseline exists, heart rate above expected cannot lift the band; the excess
    // stays visible in the residual
    func testBaselineStopsDriftLiftingTheBand() {
        var tes = establishedModel()
        XCTAssertEqual(tempoEffort(&tes, heartRate: 166), .tempo)
        XCTAssertEqual(tempoEffort(&tes, heartRate: 180), .tempo)
        XCTAssertGreaterThan(tes.currentResidual ?? 0, 20)
    }

    // A reading below expected passes through unchanged, so a genuinely easier moment
    // still reads easier
    func testBaselineLetsALowerReadingThrough() {
        var tes = establishedModel()
        XCTAssertEqual(tempoEffort(&tes, heartRate: 126), .easy)
    }

    // The masking cases still classify Hard with a baseline, at full and zero trust
    func testBaselineKeepsDescentAndPowerHikeHard() {
        var tes = establishedModel()
        for trust in [1.0, 0.0] {
            tes.discardRun()
            tes.record(heartRate: 131, pace: 5.1, cadence: 157, grade: -5.8,
                       verticalOscillation: 10.9, altitude: 101,
                       at: Date(timeIntervalSince1970: 0), hrTrust: trust)
            XCTAssertEqual(tes.currentEffort, .hard, "descent at trust \(trust)")
            tes.discardRun()
            tes.record(heartRate: 166, pace: 9.6, cadence: 144, grade: 9.5,
                       verticalOscillation: 6.4, altitude: 101,
                       at: Date(timeIntervalSince1970: 0), hrTrust: trust)
            XCTAssertEqual(tes.currentEffort, .hard, "power-hike at trust \(trust)")
        }
    }

    // An optical spike on an easy jog reads Easy at every trust level once a baseline exists
    func testBaselineReadsASensorSpikeAsEasy() {
        var tes = establishedModel()
        for trust in [0.0, 0.5, 1.0] {
            tes.discardRun()
            tes.record(heartRate: 184, pace: 6.6, cadence: 164, grade: 0.0,
                       verticalOscillation: 8.7, altitude: 101,
                       at: Date(timeIntervalSince1970: 0), hrTrust: trust)
            XCTAssertEqual(tes.currentEffort, .easy, "spike at trust \(trust)")
        }
    }

    func testDefaultLambdaIsLight() {
        XCTAssertEqual(TrueEffortScore().lambda, 0.01)
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
