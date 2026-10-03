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
        XCTAssertEqual(EffortClass.recovery.weight, 0.10)
        XCTAssertEqual(EffortClass.easy.weight, 0.25)
        XCTAssertEqual(EffortClass.tempo.weight, 0.50)
        XCTAssertEqual(EffortClass.threshold.weight, 0.75)
        XCTAssertEqual(EffortClass.hard.weight, 1.00)
        XCTAssertEqual(EffortClass.allCases.map(\.ordinal), [-1, 0, 1, 2, 3])
    }

    // Recovery sits below Easy, so the classifier's labels for running keep their ordinals
    func testRecoveryBandRawValueLabelAndClamping() {
        XCTAssertEqual(EffortClass.recovery.rawValue, "recovery")
        XCTAssertEqual(EffortClass.recovery.label, "Recovery")
        XCTAssertEqual(EffortClass(clampingOrdinal: -1), .recovery)
        XCTAssertEqual(EffortClass(clampingOrdinal: -5), .recovery)
        XCTAssertEqual(EffortClass(clampingOrdinal: 0), .easy)
        XCTAssertFalse(TrueEffortScore.anchorLabels.contains(.recovery))
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

    func testRefitProducesNamedEquation() {
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
        let equation = tes.baseline?.equation() ?? ""
        XCTAssertTrue(equation.hasPrefix("expected HR = "))
        for name in ["pace", "cadence", "verticalOscillation"] {
            XCTAssertTrue(equation.contains("·" + name), "missing \(name) in \(equation)")
        }
        // Grade and altitude never vary in this history, so their weights are zero and drop out,
        // the same rule every linear model's equation() follows.
        XCTAssertFalse(equation.contains("grade"))
        XCTAssertFalse(equation.contains("altitude"))
    }

    /// A first run has no baseline to score with, so its result carries nil; the next run's result
    /// carries the baseline that scored it, captured before finalize() refits.
    func testResultCarriesTheBaselineThatScoredTheRun() {
        var tes = TrueEffortScore()
        var start = Date(timeIntervalSince1970: 0)

        for run in 0..<2 {
            for i in 0..<120 {
                let hard = i % 2 == 0
                // The second run reads a few beats higher, so its refit moves the baseline.
                tes.record(heartRate: (hard ? 170 : 132) + Double(run * 4), pace: hard ? 4.8 : 6.5,
                           cadence: hard ? 180 : 165, grade: 0.0,
                           verticalOscillation: hard ? 7.0 : 8.5, altitude: 100,
                           at: start.addingTimeInterval(Double(i)))
            }
            let before = tes.baseline
            guard let result = tes.finalize() else {
                XCTFail("run \(run) should be long enough to keep")
                return
            }
            XCTAssertEqual(result.baseline, before)
            if run == 0 {
                XCTAssertNil(result.baseline)
                XCTAssertTrue(result.description.contains("uncalibrated"))
            } else {
                XCTAssertNotNil(result.baseline)
                XCTAssertNotEqual(result.baseline, tes.baseline)
            }
            start = start.addingTimeInterval(86_400)
        }
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

    // MARK: - White paper fixtures (section 6)

    // Values computed through the committed code and cross-checked against an independent
    // port (Planning/tes/2026-09-24-team-review/memo-2e-fixtures.md). Band timelines score
    // through sessionScore, so these pin the scoring math independent of the classifier.

    /// A band timeline sampled at 1 Hz: a zero-duration first sample, then one sample per second,
    /// so the durations sum to exactly the held time.
    private func bandTimeline(
        _ holds: [(band: Double, seconds: Int)]
    ) -> (ordinals: [Double], durations: [TimeInterval]) {
        var ordinals = [holds.first?.band ?? 0]
        for hold in holds {
            ordinals += Array(repeating: hold.band, count: hold.seconds)
        }
        let durations = ordinals.indices.map { $0 == 0 ? 0.0 : 1.0 }
        return (ordinals, durations)
    }

    /// Scores a band timeline through the same math finalize() uses.
    private func score(_ holds: [(band: Double, seconds: Int)]) -> (
        raw: Double, varianceMultiplier: Double, durationFactor: Double,
        transitionLoad: Double, adjusted: Double
    ) {
        let timeline = bandTimeline(holds)
        return TrueEffortScore.sessionScore(
            ordinals: timeline.ordinals, durations: timeline.durations)
    }

    // Section 6.1: one steady hour in each band, and thirty minutes at threshold
    func testFixtureCalibrationTable() {
        let expected: [(band: Double, raw: Double, adjusted: Double)] = [
            (0, 33.3333, 34.2923), (1, 66.6667, 68.5845),
            (2, 100.0000, 102.8768), (3, 133.3333, 137.1691),
        ]
        for row in expected {
            let hour = score([(row.band, 3600)])
            XCTAssertEqual(hour.raw, row.raw, accuracy: 1e-4)
            XCTAssertEqual(hour.durationFactor, 1.0288, accuracy: 1e-4)
            XCTAssertEqual(hour.varianceMultiplier, 1.0, accuracy: 1e-12)
            XCTAssertEqual(hour.adjusted, row.adjusted, accuracy: 1e-4)
        }
        let halfHour = score([(2, 1800)])
        XCTAssertEqual(halfHour.raw, 50.0, accuracy: 1e-9)
        XCTAssertEqual(halfHour.adjusted, 50.0, accuracy: 1e-9)
    }

    // Section 6.2: 47-minute hill repeats against a 47-minute steady Tempo run
    func testFixtureIntervalsAgainstSteadyRun() {
        var holds: [(band: Double, seconds: Int)] = [(0, 600)]
        for _ in 0..<8 {
            holds += [(3, 120), (0, 120)]
        }
        holds.append((0, 300))
        let repeats = score(holds)
        XCTAssertEqual(repeats.raw, 52.7778, accuracy: 1e-4)
        XCTAssertEqual(repeats.varianceMultiplier, 1.35, accuracy: 1e-12)
        XCTAssertEqual(repeats.durationFactor, 1.0043, accuracy: 1e-4)
        XCTAssertEqual(repeats.transitionLoad, 48)
        XCTAssertEqual(repeats.adjusted, 76.3598, accuracy: 1e-4)

        let steady = score([(1, 2820)])
        XCTAssertEqual(steady.raw, 52.2222, accuracy: 1e-4)
        XCTAssertEqual(steady.varianceMultiplier, 1.0, accuracy: 1e-12)
        XCTAssertEqual(steady.transitionLoad, 0)
        XCTAssertEqual(steady.adjusted, 52.4493, accuracy: 1e-4)
    }

    // Section 6.4: the flickering Tempo run scores the same at 1 Hz and 2 Hz once debounced;
    // the paper's "Tempo raw 44.4" is the same run without the flicker
    func testFixtureFlickerRunScores() {
        let oneHertz = flickeringTempo(hz: 1)
        let flicker = TrueEffortScore.sessionScore(
            ordinals: oneHertz.ordinals, durations: oneHertz.durations)
        XCTAssertEqual(flicker.raw, 45.5370, accuracy: 1e-4)
        XCTAssertEqual(flicker.transitionLoad, 0)
        XCTAssertEqual(flicker.adjusted, 46.6578, accuracy: 1e-4)

        let twoHertz = flickeringTempo(hz: 2)
        let flickerTwo = TrueEffortScore.sessionScore(
            ordinals: twoHertz.ordinals, durations: twoHertz.durations)
        XCTAssertEqual(flickerTwo.adjusted, 46.6568, accuracy: 1e-4)

        let clean = TrueEffortScore.sessionScore(
            ordinals: oneHertz.ordinals.map { _ in 1.0 }, durations: oneHertz.durations)
        XCTAssertEqual(clean.raw, 44.4259, accuracy: 1e-4)
    }

    // Longer sessions and the transition bound under the 10-second debounce
    func testFixtureLongSessionsAndTransitionBound() {
        XCTAssertEqual(score([(0, 3 * 3600)]).adjusted, 113.8629, accuracy: 1e-4)
        XCTAssertEqual(score([(1, 90 * 60)]).adjusted, 106.9315, accuracy: 1e-4)

        var alternating: [(band: Double, seconds: Int)] = []
        for i in 0..<360 {
            alternating.append((i % 2 == 0 ? 0 : 3, 10))
        }
        let bound = score(alternating)
        XCTAssertEqual(bound.transitionLoad, 1077)
        XCTAssertEqual(bound.adjusted, 223.4364, accuracy: 1e-4)

        var nineSecond: [(band: Double, seconds: Int)] = []
        for i in 0..<400 {
            nineSecond.append((i % 2 == 0 ? 0 : 3, 9))
        }
        let flickers = score(nineSecond)
        XCTAssertEqual(flickers.transitionLoad, 0)
        XCTAssertEqual(flickers.adjusted, 115.7364, accuracy: 1e-4)
    }

    // End to end: an hour recorded through record() and finalize() matches the band-timeline
    // math, so the two layers cannot drift apart
    func testFixtureRecordedThresholdHourMatchesTimeline() throws {
        var tes = TrueEffortScore()
        recordThreshold(&tes, count: 3601)
        XCTAssertEqual(tes.currentEffort, .threshold)
        let result = try XCTUnwrap(tes.finalize())
        XCTAssertEqual(result.raw, 100.0, accuracy: 1e-9)
        XCTAssertEqual(result.adjusted, 102.8768, accuracy: 1e-4)
        XCTAssertEqual(result.adjusted, score([(2, 3600)]).adjusted, accuracy: 1e-12)
    }

    /// The three nearest anchors to a moment, by Euclidean distance in standardized units.
    private func nearestAnchors(
        _ row: [Double], in tes: TrueEffortScore
    ) -> [(label: EffortClass, distance: Double)] {
        let z = tes.classifier.scaler.transform([row])[0]
        let anchors = tes.classifier.model.trainingFeatures
        var ranked: [(label: EffortClass, distance: Double)] = []
        for (anchor, label) in zip(anchors, TrueEffortScore.anchorLabels) {
            let squared = zip(z, anchor).map { ($0 - $1) * ($0 - $1) }.reduce(0, +)
            ranked.append((label, squared.squareRoot()))
        }
        return Array(ranked.sorted { $0.distance < $1.distance }.prefix(3))
    }

    // Section 6.3: the steep descent and the power-hike at cold start
    func testFixtureMomentsHeartRateHides() throws {
        var tes = TrueEffortScore()
        let descent = [131.0, 5.1, 157, -5.8, 10.9]
        tes.record(heartRate: descent[0], pace: descent[1], cadence: descent[2], grade: descent[3],
                   verticalOscillation: descent[4], altitude: 100,
                   at: Date(timeIntervalSince1970: 0))
        let descentZ = try XCTUnwrap(tes.currentSignals).asArray
        for (value, target) in zip(descentZ, [-1.0328, -0.7023, -0.9579, -1.8381, 2.2947]) {
            XCTAssertEqual(value, target, accuracy: 1e-4)
        }
        XCTAssertEqual(tes.currentEffort, .hard)
        let descentNeighbors = nearestAnchors(descent, in: tes)
        XCTAssertEqual(descentNeighbors.map(\.label), [.hard, .hard, .easy])
        for (neighbor, target) in zip(descentNeighbors, [0.1609, 0.1718, 2.4049]) {
            XCTAssertEqual(neighbor.distance, target, accuracy: 1e-4)
        }

        tes.discardRun()
        let hike = [166.0, 9.6, 144, 9.5, 6.4]
        tes.record(heartRate: hike[0], pace: hike[1], cadence: hike[2], grade: hike[3],
                   verticalOscillation: hike[4], altitude: 100,
                   at: Date(timeIntervalSince1970: 0))
        let hikeZ = try XCTUnwrap(tes.currentSignals).asArray
        for (value, target) in zip(hikeZ, [0.8142, 2.5350, -2.1519, 2.2688, -1.3683]) {
            XCTAssertEqual(value, target, accuracy: 1e-4)
        }
        XCTAssertEqual(tes.currentEffort, .hard)
        let hikeNeighbors = nearestAnchors(hike, in: tes)
        XCTAssertEqual(hikeNeighbors.map(\.label), [.hard, .hard, .hard])
        for (neighbor, target) in zip(hikeNeighbors, [0.2026, 0.3004, 4.1148]) {
            XCTAssertEqual(neighbor.distance, target, accuracy: 1e-4)
        }
    }

    // Section 6.4: an optical spike on an easy jog at cold start reads Tempo, Tempo, then Easy
    // as trust falls from 1 to 0.5 to 0
    func testFixtureOpticalArtifactAtColdStart() {
        var tes = TrueEffortScore()
        let expected: [(trust: Double, band: EffortClass)] = [
            (1.0, .tempo), (0.5, .tempo), (0.0, .easy),
        ]
        for step in expected {
            tes.discardRun()
            tes.record(heartRate: 184, pace: 6.6, cadence: 164, grade: 0.0,
                       verticalOscillation: 8.7, altitude: 100,
                       at: Date(timeIntervalSince1970: 0), hrTrust: step.trust)
            XCTAssertEqual(tes.currentEffort, step.band, "trust \(step.trust)")
        }
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

    // MARK: - Workout location

    func testLocationDefaultsToOutdoor() {
        XCTAssertEqual(TrueEffortScore().location, .outdoor)
    }

    func testIndoorRunScoresWithoutFoldingIntoHistory() {
        var tes = establishedModel()
        let baselineBefore = tes.baseline
        tes.location = .indoor
        recordThreshold(&tes, count: 120, start: Date(timeIntervalSince1970: 5_000_000))
        let result = tes.finalize()
        XCTAssertNotNil(result)
        if let r = result {
            XCTAssertGreaterThan(r.raw, 0)
        }
        XCTAssertEqual(tes.sessionCount, 1, "an indoor run must not join history")
        XCTAssertEqual(tes.baseline, baselineBefore, "an indoor run must not refit the baseline")
    }

    func testIndoorRunRecordsTheFixedGrade() {
        let time = Date(timeIntervalSince1970: 0)
        var indoor = TrueEffortScore()
        indoor.location = .indoor
        indoor.record(heartRate: 150, pace: 5.6, cadence: 171, grade: 6.0,
                      verticalOscillation: 8.0, altitude: 100, at: time)
        var outdoor = TrueEffortScore()
        outdoor.record(heartRate: 150, pace: 5.6, cadence: 171, grade: TrueEffortScore.indoorGrade,
                       verticalOscillation: 8.0, altitude: 100, at: time)
        XCTAssertNotNil(indoor.currentSignals)
        XCTAssertEqual(indoor.currentSignals, outdoor.currentSignals,
                       "the supplied 6% grade should be replaced by the indoor grade")
    }

    func testOutdoorRunRecordsTheSuppliedGrade() {
        let time = Date(timeIntervalSince1970: 0)
        var steep = TrueEffortScore()
        steep.record(heartRate: 150, pace: 5.6, cadence: 171, grade: 6.0,
                     verticalOscillation: 8.0, altitude: 100, at: time)
        var level = TrueEffortScore()
        level.record(heartRate: 150, pace: 5.6, cadence: 171, grade: TrueEffortScore.indoorGrade,
                     verticalOscillation: 8.0, altitude: 100, at: time)
        XCTAssertNotEqual(steep.currentSignals?.grade, level.currentSignals?.grade)
    }

    func testEndingARunResetsLocationToOutdoor() {
        var finalized = TrueEffortScore()
        finalized.location = .indoor
        recordThreshold(&finalized, count: 120)
        XCTAssertNotNil(finalized.finalize())
        XCTAssertEqual(finalized.location, .outdoor)

        var tooShort = TrueEffortScore()
        tooShort.location = .indoor
        recordThreshold(&tooShort, count: 30)
        XCTAssertNil(tooShort.finalize())
        XCTAssertEqual(tooShort.location, .outdoor)

        var discarded = TrueEffortScore()
        discarded.location = .indoor
        recordThreshold(&discarded, count: 120)
        discarded.discardRun()
        XCTAssertEqual(discarded.location, .outdoor)
    }

    func testRunAfterIndoorRunFoldsIntoHistory() {
        var tes = TrueEffortScore()
        tes.location = .indoor
        recordThreshold(&tes, count: 120)
        _ = tes.finalize()
        XCTAssertEqual(tes.sessionCount, 0)
        recordThreshold(&tes, count: 120, start: Date(timeIntervalSince1970: 2_000_000))
        XCTAssertNotNil(tes.finalize())
        XCTAssertEqual(tes.sessionCount, 1)
        XCTAssertNotNil(tes.baseline)
    }

    func testDecodedModelStartsOutdoor() throws {
        var tes = TrueEffortScore()
        recordThreshold(&tes, count: 120)
        _ = tes.finalize()
        tes.location = .indoor
        let data = try JSONEncoder().encode(tes)
        let restored = try JSONDecoder().decode(TrueEffortScore.self, from: data)
        XCTAssertEqual(restored.location, .outdoor)
        XCTAssertEqual(restored, tes, "location is live-run state and does not affect equality")
    }

    // MARK: - Walking gate

    /// A moment with the given signals, for testing the walking gate directly.
    private func moment(heartRate: Double, pace: Double, cadence: Double, grade: Double,
                        hrTrust: Double = 1.0) -> Workout.Moment {
        Workout.Moment(heartRate: heartRate, pace: pace, cadence: cadence, grade: grade,
                       verticalOscillation: 4.5, altitude: 100, hrTrust: hrTrust, deltaTime: 5)
    }

    /// Records constant-signal blocks every 5 seconds, closing on the last block's signals.
    private func recordBlocks(
        _ tes: inout TrueEffortScore,
        _ blocks: [(seconds: Int, heartRate: Double, pace: Double, cadence: Double,
                    grade: Double, verticalOscillation: Double)]
    ) {
        var t = 0.0
        for block in blocks {
            for _ in 0..<(block.seconds / 5) {
                tes.record(heartRate: block.heartRate, pace: block.pace, cadence: block.cadence,
                           grade: block.grade, verticalOscillation: block.verticalOscillation,
                           altitude: 100, at: Date(timeIntervalSince1970: 1_000_000 + t))
                t += 5
            }
        }
        if let last = blocks.last {
            tes.record(heartRate: last.heartRate, pace: last.pace, cadence: last.cadence,
                       grade: last.grade, verticalOscillation: last.verticalOscillation,
                       altitude: 100, at: Date(timeIntervalSince1970: 1_000_000 + t))
        }
    }

    // A flat walk read Hard against the running anchors; the gate reads it Recovery
    func testFlatWalkReadsRecoveryAndScoresBelowAnEasyRun() throws {
        var walk = TrueEffortScore()
        recordBlocks(&walk, [(1800, 100, 12.8, 106, 0, 4.5)])
        XCTAssertEqual(walk.currentEffort, .recovery)
        let walked = try XCTUnwrap(walk.finalize())
        XCTAssertEqual(walked.effortDistribution[.recovery] ?? 0, 1.0, accuracy: 1e-9)
        XCTAssertEqual(walked.adjusted, 6.6667, accuracy: 1e-4)

        var run = TrueEffortScore()
        recordBlocks(&run, [(1800, 131, 6.6, 164, 0, 8.7)])
        let ran = try XCTUnwrap(run.finalize())
        XCTAssertEqual(ran.adjusted, 16.6667, accuracy: 1e-4)
    }

    // Every anchor row is a running effort and bypasses the gate, power-hikes included
    func testAnchorRowsBypassTheWalkingGate() {
        for row in TrueEffortScore.anchorSamples {
            let m = moment(heartRate: row[0], pace: row[1], cadence: row[2], grade: row[3])
            XCTAssertNil(TrueEffortScore.walkingEffort(for: m), "anchor \(row)")
        }
        XCTAssertNil(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 125, pace: 7.2, cadence: 154, grade: 0)))
        XCTAssertNil(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 140, pace: 6.0, cadence: 120, grade: 0)),
            "a fast low-cadence moment is running")
    }

    // A walk-run labels each segment on its own: the walks read Recovery, not Hard
    func testWalkRunReadsRecoveryBetweenRuns() throws {
        var tes = TrueEffortScore()
        var blocks: [(seconds: Int, heartRate: Double, pace: Double, cadence: Double,
                      grade: Double, verticalOscillation: Double)] = []
        for _ in 0..<17 {
            blocks += [(40, 125, 7.2, 154, 0, 8.5), (160, 112, 12.8, 104, 0, 4.5)]
        }
        recordBlocks(&tes, blocks)
        let result = try XCTUnwrap(tes.finalize())
        XCTAssertEqual(result.effortDistribution[.recovery] ?? 0, 0.8015, accuracy: 1e-4)
        XCTAssertEqual(result.effortDistribution[.easy] ?? 0, 0.1985, accuracy: 1e-4)
        XCTAssertNil(result.effortDistribution[.hard])
        XCTAssertEqual(result.adjusted, 17.3720, accuracy: 1e-4)
    }

    // A climb takes the higher of its vertical-speed band and its heart-rate band
    func testWalkingClimbTakesTheHigherBand() {
        // 15% at 14.3 min/km is about 629 m/h: Threshold whatever the heart rate says below 165
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 130, pace: 14.3, cadence: 98, grade: 15)), .threshold)
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 175, pace: 14.3, cadence: 98, grade: 15)), .hard)
        // 8% at 15 min/km is 320 m/h: Tempo
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 120, pace: 15, cadence: 100, grade: 8)), .tempo)
        // 5% at 15 min/km is 200 m/h: Easy, unless heart rate lifts it
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 120, pace: 15, cadence: 100, grade: 5)), .easy)
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 155, pace: 15, cadence: 100, grade: 5)), .threshold)
        // 25% at 12 min/km is 1250 m/h: Hard on vertical speed alone
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 110, pace: 12, cadence: 95, grade: 25)), .hard)
    }

    // A doubted or dropped heart rate leaves a climb on vertical speed, never below Easy
    func testWalkingClimbWithoutTrustedHeartRateUsesVerticalSpeed() {
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 180, pace: 15, cadence: 100, grade: 8, hrTrust: 0)), .tempo)
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 0, pace: 15, cadence: 100, grade: 5)), .easy)
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 0, pace: 14.3, cadence: 0, grade: 15)), .threshold,
            "a cadence dropout on a climb still gates and still reads vertical speed")
    }

    // A walking descent is Easy, or Tempo when fast, and never Hard; gentle grades are Recovery
    func testWalkingDescentNeverReadsHard() {
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 98, pace: 17, cadence: 120, grade: -12)), .easy)
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 98, pace: 13.5, cadence: 122, grade: -15)), .tempo)
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 98, pace: 9, cadence: 125, grade: -25)), .tempo)
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 102, pace: 12.9, cadence: 119, grade: -8)), .recovery)
        XCTAssertEqual(TrueEffortScore.walkingEffort(
            for: moment(heartRate: 0, pace: 13, cadence: 0, grade: 0)), .recovery,
            "a cadence dropout while walking still gates")
    }

    // The gate reads raw heart rate, so the baseline cap cannot pull a hard climb down
    func testWalkingClimbIgnoresTheBaselineCap() {
        var tes = establishedModel()
        tes.discardRun()
        tes.record(heartRate: 175, pace: 15, cadence: 95, grade: 8, verticalOscillation: 4.5,
                   altitude: 101, at: Date(timeIntervalSince1970: 0))
        XCTAssertEqual(tes.currentEffort, .hard)
    }

    // Walked recoveries sit further from the reps than jogged ones, so the session terms grow
    func testFixtureWalkedRecoveriesRaiseTransitionLoad() {
        var walked: [(band: Double, seconds: Int)] = [(0, 600)]
        var jogged: [(band: Double, seconds: Int)] = [(0, 600)]
        for _ in 0..<8 {
            walked += [(3, 120), (-1, 120)]
            jogged += [(3, 120), (0, 120)]
        }
        walked.append((0, 300))
        jogged.append((0, 300))
        let w = score(walked), j = score(jogged)
        XCTAssertEqual(j.adjusted, 76.3598, accuracy: 1e-4)
        XCTAssertEqual(w.transitionLoad, 63)
        XCTAssertEqual(w.raw, 47.4444, accuracy: 1e-4)
        XCTAssertEqual(w.adjusted, 70.6285, accuracy: 1e-4)
        XCTAssertLessThan(w.raw, j.raw)
    }

    // The paper's downhill example runs at 157 spm and 5.1 min/km, so the gate leaves it alone
    func testFixtureDownhillExampleIsUnchangedByTheGate() throws {
        var tes = TrueEffortScore()
        var blocks: [(seconds: Int, heartRate: Double, pace: Double, cadence: Double,
                      grade: Double, verticalOscillation: Double)] = [(300, 128, 6.6, 164, 0, 8.7)]
        for _ in 0..<4 {
            blocks += [(480, 130, 5.1, 157, -6, 10.9), (120, 128, 6.4, 162, -2, 9.0)]
        }
        blocks.append((600, 128, 6.6, 164, 0, 8.7))
        recordBlocks(&tes, blocks)
        let result = try XCTUnwrap(tes.finalize())
        XCTAssertEqual(result.adjusted, 117.92, accuracy: 0.01)
    }
}
