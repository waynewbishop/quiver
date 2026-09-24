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

/// A runner's effort model, composed from three Quiver models: a `Ridge` baseline that predicts
/// expected heart rate from workload, a `ResidualModel` over it that reports the gap, and a
/// `Pipeline<KNearestNeighbors>` that classifies each moment's effort band. It holds a session's
/// history, scores a run live tick by tick, and re-fits the baseline as history grows.
///
/// The design classifies every misleading effect by which signal lies and in which direction:
/// inflating effects (heat, drift, altitude) push heart rate up and surface as the baseline
/// residual; masking effects (downhill, power hiking) fool heart rate and are caught by the
/// classifier; session-level effects (intervals, duration) are applied at `finalize()`.
///
/// Construct empty for a cold start or decode a saved model to resume. Persistence is synthesized
/// `Codable`, so a caller uses `JSONEncoder` and `JSONDecoder` as with any Quiver model. Heart
/// rate is only ever the regression target, never an effort feature, so regressing effort on
/// heart rate is unrepresentable.
public struct TrueEffortScore: Codable, Equatable, CustomStringConvertible, Sendable {

    // MARK: Trained models (the composed Quiver half)

    /// The effort classifier, seeded from the population anchors and any personal labeled samples.
    /// Frozen after seeding, since re-fitting it on its own predictions would be self-training.
    public private(set) var classifier: Pipeline<KNearestNeighbors>

    /// The personal expected-heart-rate baseline, derived from `history`. nil at cold start; read
    /// its `expression`, `coefficients`, and `conditioning` once present.
    public private(set) var baseline: TESBaseline?

    // MARK: Owned history

    /// The runner's accumulated workouts — the data the baseline is fit on, and the thing to
    /// persist. Grows as `finalize()` folds each run in; bounded by `historyLimit`.
    public private(set) var history: [Workout]

    /// How many recent workouts to retain, oldest dropping off when exceeded. nil keeps all.
    public let historyLimit: Int?

    /// Sessions logged — the cold-start to established axis. The caller decides what "established"
    /// means; the model only reports the count.
    public var sessionCount: Int { history.count }

    // MARK: Configuration

    public let lambda: Double   // Ridge L2 penalty
    public let k: Int           // KNN neighbors

    // MARK: Live run state (transient; not encoded)

    private var liveMoments: [Workout.Moment] = []
    private var lastSampleTime: Date?
    private var runStartDate: Date?
    private var paused = false
    private var pausedAt: Date?
    private var accumulatedElapsed: TimeInterval = 0   // wall clock, includes pauses
    private var accumulatedTimer: TimeInterval = 0     // active time, excludes pauses

    /// One hour held at threshold defines a score of 100: 0.75 × 3600 = 2700.
    static let scoreAnchor = EffortClass.threshold.weight * 3600.0

    /// The active-time floor, in seconds, below which a run is too short to score or keep.
    static let minimumTimerSeconds: TimeInterval = 60

    /// The shortest hold, in seconds, that counts as a real band when measuring transitions.
    /// Anything briefer is a classifier flicker at a boundary, not a change of effort.
    static let minimumBandSeconds: TimeInterval = 10

    // MARK: Codable (transient live-run state is excluded)

    private enum CodingKeys: String, CodingKey {
        case classifier, baseline, history, historyLimit, lambda, k
    }

    public static func == (lhs: TrueEffortScore, rhs: TrueEffortScore) -> Bool {
        lhs.classifier == rhs.classifier && lhs.baseline == rhs.baseline
            && lhs.history == rhs.history && lhs.historyLimit == rhs.historyLimit
            && lhs.lambda == rhs.lambda && lhs.k == rhs.k
    }

    // MARK: Construction

    /// A brand-new runner: empty history, no personal baseline yet. The classifier is anchor-
    /// seeded from the population set, so a score exists from the first tick; `baseline == nil` is
    /// the cold-start tell.
    ///
    /// - Precondition: `k <= anchorSamples.count`.
    public init(lambda: Double = 1.0, k: Int = 3, historyLimit: Int? = 60) {
        precondition(k <= Self.anchorSamples.count,
                     "k (\(k)) must not exceed the anchor sample count (\(Self.anchorSamples.count)).")
        self.lambda = lambda
        self.k = k
        self.historyLimit = historyLimit
        self.history = []
        self.baseline = nil
        self.classifier = Self.seedClassifier(personalRows: [], personalLabels: [], k: k)
    }

    /// Seeds from existing workouts and optional personal labeled samples, then fits the baseline
    /// once. Each labeled sample is a classifier feature row
    /// `[heartRate, pace, cadence, grade, verticalOscillation]`, appended to the bundled anchors,
    /// so the classifier never trains on less than the anchor set.
    ///
    /// - Precondition: `labeledSamples.count == labels.count`.
    /// - Throws: `TESError.insufficientLabeledSamples` when k exceeds the labeled example count;
    ///   `TESError.baselineDiverged` if the seed fit diverges.
    public init(
        history: [Workout],
        labeledSamples: [[Double]] = [],
        labels: [EffortClass] = [],
        lambda: Double = 1.0,
        k: Int = 3,
        historyLimit: Int? = 60
    ) throws {
        precondition(labeledSamples.count == labels.count,
                     "labeledSamples.count must equal labels.count.")
        let available = Self.anchorSamples.count + labeledSamples.count
        guard k <= available else {
            throw TESError.insufficientLabeledSamples(k: k, available: available)
        }
        self.lambda = lambda
        self.k = k
        self.historyLimit = historyLimit
        self.classifier = Self.seedClassifier(
            personalRows: labeledSamples, personalLabels: labels, k: k)
        self.history = Self.trimmed(history, to: historyLimit)
        self.baseline = try Self.fitBaseline(history: self.history, lambda: lambda)
    }

    // MARK: Recording a live run

    /// Records one moment from its raw signals, stamped with an absolute time. The delta is derived
    /// internally, so the caller never tracks it. A non-positive delta is dropped, which self-
    /// discards a duplicate or out-of-order callback. The first sample of a run carries zero load
    /// but stamps the start; a post-resume gap advances wall clock only, never timer time or the
    /// run's start. `hrTrust` is the optical-sensor reliability, clamped to `0...1`; a fully
    /// doubted reading classifies on the kinematic signals alone. Grade is signed, negative for
    /// downhill.
    public mutating func record(
        heartRate: Double,
        pace: Double,
        cadence: Double,
        grade: Double,
        verticalOscillation: Double,
        altitude: Double,
        at time: Date,
        hrTrust: Double = 1.0
    ) {
        guard !paused else { return }

        let delta: TimeInterval
        if let last = lastSampleTime {
            delta = time.timeIntervalSince(last)
            guard delta > 0 else { return }
        } else {
            delta = 0
            if runStartDate == nil {
                runStartDate = time
            } else if let resumedFrom = pausedAt {
                accumulatedElapsed += time.timeIntervalSince(resumedFrom)
            }
        }

        liveMoments.append(Workout.Moment(
            heartRate: heartRate, pace: pace, cadence: cadence, grade: grade,
            verticalOscillation: verticalOscillation, altitude: altitude,
            hrTrust: Swift.min(1.0, Swift.max(0.0, hrTrust)), deltaTime: delta))
        lastSampleTime = time
        pausedAt = nil
        if delta > 0 {
            accumulatedElapsed += delta
            accumulatedTimer += delta
        }
    }

    /// Pauses the run. The paused span counts toward wall-clock `elapsedTime` but not `timerTime`.
    public mutating func pause() {
        guard !paused else { return }
        paused = true
        pausedAt = lastSampleTime
    }

    /// Resumes the run. The next recorded sample re-seeds the clock across the paused gap.
    public mutating func resume() {
        paused = false
        lastSampleTime = nil
    }

    /// Discards the in-progress run without folding it into history.
    public mutating func discardRun() { clearLiveRun() }

    // MARK: Live readouts

    /// The live True Effort Score, climbing through the run against the fixed 2700 anchor.
    public var currentScore: Double { rawScore(moments: liveMoments) }

    /// The classifier's effort band for the latest recorded sample, nil before the first sample.
    public var currentEffort: EffortClass? {
        liveMoments.last.map { effortClass(for: $0) }
    }

    /// Observed minus expected heart rate for the latest sample, nil before a baseline exists.
    public var currentResidual: Double? {
        guard let moment = liveMoments.last, let baseline else { return nil }
        let scaled = baseline.scaler.transform([moment.regressionFeatures])[0]
        return baseline.residualModel.residual(features: scaled, observed: moment.heartRate)
    }

    /// The latest sample's signals as z-scores against the classifier's training distribution, so
    /// a UI can spot which signal is an outlier — a large negative `grade` with a near-zero
    /// `heartRate` is the downhill signature. nil before the first sample.
    public var currentSignals: EffortSignals? {
        guard let moment = liveMoments.last else { return nil }
        let z = classifier.scaler.transform([moment.classifierFeatures])[0]
        return EffortSignals(
            heartRate: z[0], pace: z[1], cadence: z[2], grade: z[3], verticalOscillation: z[4])
    }

    public var sampleCount: Int { liveMoments.count }
    public var elapsedTime: TimeInterval { accumulatedElapsed }
    public var timerTime: TimeInterval { accumulatedTimer }
    public var isPaused: Bool { paused }

    // MARK: Finalization

    /// Closes the run: returns the breakdown, folds the run into history, and re-fits the baseline
    /// from the accumulated history (the classifier stays frozen). Returns nil when active
    /// `timerTime` is below the floor, in which case the run is cleared and not kept.
    public mutating func finalize() -> TESResult? {
        guard timerTime >= Self.minimumTimerSeconds else {
            clearLiveRun()
            return nil
        }

        let moments = liveMoments
        let start = runStartDate ?? lastSampleTime ?? Date(timeIntervalSince1970: 0)
        let result = buildResult(moments: moments)

        history = Self.trimmed(history + [Workout(moments: moments, startDate: start)],
                               to: historyLimit)
        if let refit = try? Self.fitBaseline(history: history, lambda: lambda) {
            baseline = refit
        }

        clearLiveRun()
        return result
    }

    // MARK: CustomStringConvertible

    public var description: String {
        let state = baseline?.conditioning.map { String(format: "conditioning=%.1f", $0) } ?? "cold start"
        return "TrueEffortScore: \(sessionCount) session(s), "
            + "currentScore=\(Int(currentScore.rounded())), \(state)"
    }
}

// MARK: - Internal machinery

extension TrueEffortScore {

    /// Seeds the classifier from the bundled anchor rows plus any personal labeled rows.
    static func seedClassifier(
        personalRows: [[Double]],
        personalLabels: [EffortClass],
        k: Int
    ) -> Pipeline<KNearestNeighbors> {
        Pipeline<KNearestNeighbors>.fit(
            features: anchorSamples + personalRows,
            labels: (anchorLabels + personalLabels).map(\.ordinal),
            k: k)
    }

    /// Fits the personal baseline over a history, standardizing then fitting atomically. Returns
    /// nil for empty history. Throws `TESError.baselineDiverged` if the descent diverges.
    static func fitBaseline(history: [Workout], lambda: Double) throws -> TESBaseline? {
        let moments = history.flatMap(\.moments)
        guard !moments.isEmpty else { return nil }

        let features = moments.map(\.regressionFeatures)
        let heartRates = moments.map(\.heartRate)
        let scaler = StandardScaler.fit(features: features)
        let scaled = scaler.transform(features)

        let ridge: Ridge
        do {
            ridge = try Ridge.fit(features: scaled, targets: heartRates, lambda: lambda)
        } catch let error as GradientDescentError {
            throw TESError.baselineDiverged(error)
        }

        let condition = scaled.transposed().multiplyMatrix(scaled).conditionNumber
        return TESBaseline(
            lambda: lambda,
            scaler: scaler,
            expectedHeartRate: ridge,
            residualModel: ResidualModel(model: ridge),
            conditionNumberValue: condition.isFinite ? condition : nil)
    }

    /// Keeps the most recent workouts when a limit is set.
    static func trimmed(_ history: [Workout], to limit: Int?) -> [Workout] {
        guard let limit, history.count > limit else { return history }
        return Array(history.suffix(limit))
    }

    /// Classifies one moment, discounting a doubted heart rate toward the training mean. A fully
    /// doubted reading holds heart rate at that mean, the center of the heart-rate axis, so the
    /// kinematic signals decide the label.
    func effortClass(for moment: Workout.Moment) -> EffortClass {
        var row = moment.classifierFeatures
        row[0] = moment.hrTrust * row[0] + (1 - moment.hrTrust) * trainingHeartRateMean
        return EffortClass(clampingOrdinal: classifier.predict([row])[0])
    }

    /// The training heart-rate mean in beats per minute, read from the pipeline's scaler. The
    /// model's stored training rows are already standardized, so averaging them gives about
    /// zero, not a heart rate; the blend above needs the raw mean.
    private var trainingHeartRateMean: Double {
        classifier.scaler.means[0]
    }

    /// The fixed-anchor score for classified moments: 100 × Σ(weight·Δt) / 2700.
    func rawScore(moments: [Workout.Moment]) -> Double {
        guard !moments.isEmpty else { return 0 }
        let load = moments.reduce(0.0) { $0 + effortClass(for: $1).weight * $1.deltaTime }
        return (load / Self.scoreAnchor) * 100.0
    }

    /// Resets all transient live-run state.
    mutating func clearLiveRun() {
        liveMoments = []
        lastSampleTime = nil
        runStartDate = nil
        paused = false
        pausedAt = nil
        accumulatedElapsed = 0
        accumulatedTimer = 0
    }

    /// Assembles the finalized breakdown.
    private func buildResult(moments: [Workout.Moment]) -> TESResult {
        let classes = moments.map { effortClass(for: $0) }

        var cumulative = 0.0
        var loadCurve: [Double] = []
        loadCurve.reserveCapacity(moments.count)
        for (cls, moment) in zip(classes, moments) {
            cumulative += cls.weight * moment.deltaTime
            loadCurve.append(cumulative)
        }
        let raw = (cumulative / Self.scoreAnchor) * 100.0

        let totalTime = moments.reduce(0.0) { $0 + $1.deltaTime }
        var distribution: [EffortClass: Double] = [:]
        if totalTime > 0 {
            for (cls, moment) in zip(classes, moments) {
                distribution[cls, default: 0] += moment.deltaTime / totalTime
            }
        }

        let ordinals = classes.map { Double($0.ordinal) }
        let variance = varianceMultiplier(ordinals: ordinals, moments: moments)
        let duration = durationFactor(totalTimerSeconds: totalTime)
        let transition = Self.transitionLoad(
            ordinals: ordinals, durations: moments.map(\.deltaTime))

        return TESResult(
            adjusted: raw * variance * duration + transition * 0.1,
            raw: raw,
            meanResidual: weightedMeanResidual(moments: moments),
            effortDistribution: distribution,
            varianceMultiplier: variance,
            durationFactor: duration,
            transitionLoad: transition,
            loadCurve: loadCurve,
            timerTime: accumulatedTimer,
            elapsedTime: accumulatedElapsed,
            baselineExpression: baseline?.expression ?? "y = (uncalibrated)")
    }

    /// Σ(hrTrust·Δt·residual) / Σ(hrTrust·Δt). Zero without a fitted baseline.
    private func weightedMeanResidual(moments: [Workout.Moment]) -> Double {
        guard let baseline else { return 0 }
        var numerator = 0.0, denominator = 0.0
        for moment in moments {
            let scaled = baseline.scaler.transform([moment.regressionFeatures])[0]
            let residual = baseline.residualModel.residual(
                features: scaled, observed: moment.heartRate)
            let weight = moment.hrTrust * moment.deltaTime
            numerator += weight * residual
            denominator += weight
        }
        return denominator > 0 ? numerator / denominator : 0
    }

    /// 1 + min(0.35, 0.25·v), excluding the first five minutes of elapsed time from the variance.
    private func varianceMultiplier(ordinals: [Double], moments: [Workout.Moment]) -> Double {
        var elapsed = 0.0
        var steadyState: [Double] = []
        for (ordinal, moment) in zip(ordinals, moments) {
            elapsed += moment.deltaTime
            if elapsed >= 300 { steadyState.append(ordinal) }
        }
        return 1.0 + Swift.min(0.35, (steadyState.variance() ?? 0) * 0.25)
    }

    /// 1 + 0.1·ln(minutes/45), gated at 45 minutes of active time.
    private func durationFactor(totalTimerSeconds: TimeInterval) -> Double {
        let minutes = totalTimerSeconds / 60.0
        guard minutes > 45 else { return 1.0 }
        return 1.0 + 0.1 * Foundation.log(minutes / 45.0)
    }

    /// Σ|Δclass| for jumps of two or more bands between held bands, after debouncing. Counting
    /// raw consecutive moments instead would score every boundary flicker as a surge, and the
    /// inflation would grow with the sample rate.
    static func transitionLoad(ordinals: [Double], durations: [TimeInterval]) -> Double {
        let bands = debouncedBands(ordinals: ordinals, durations: durations)
        return zip(bands, bands.dropFirst()).map { abs($1 - $0) }.filter { $0 >= 2 }.reduce(0, +)
    }

    /// Collapses per-moment bands into the sequence of bands actually held. A band held for less
    /// than `minimumBandSeconds` (summed moment durations, so the rule is the same at any sample
    /// rate) merges into the band before it. A short band at the very start of a run has nothing
    /// before it, so it takes the first band that is held long enough; if none is, the bands
    /// stand as recorded.
    static func debouncedBands(ordinals: [Double], durations: [TimeInterval]) -> [Double] {
        var runs: [(band: Double, seconds: TimeInterval)] = []
        for (band, seconds) in zip(ordinals, durations) {
            if let last = runs.last, last.band == band {
                runs[runs.count - 1].seconds += seconds
            } else {
                runs.append((band, seconds))
            }
        }

        let firstHeldBand = runs.first { $0.seconds >= minimumBandSeconds }?.band
        var held: [Double] = []
        for run in runs {
            let band: Double
            if run.seconds >= minimumBandSeconds {
                band = run.band
            } else if let previous = held.last {
                band = previous
            } else {
                band = firstHeldBand ?? run.band
            }
            if held.last != band { held.append(band) }
        }
        return held
    }
}
