# True Effort Score

A transparent, multi-signal model of running load on Apple Watch.

## Overview

`TrueEffortScore` (TES) measures athletic effort from six signals from Apple Watch. These include heart rate, pace, cadence, grade, vertical oscillation and altitude. A **baseline** model learns the heart rate expected for a runner's workload and reports the gap. A **classifier** model also labels each moment based on how the runner is moving. As a result, efforts that require significant biomechanical effort, such as running downhill, also count as hard work.

The whole lifecycle is three calls:

```swift
import Quiver

// 1. Start a runner.
var tes = TrueEffortScore()

// 2. Record once per sensor sample.
tes.record(heartRate: 152, pace: 5.4, cadence: 174, grade: 1.2,
           verticalOscillation: 7.9, altitude: 100, at: sampleDate)

// 3. Close the run.
if let result = tes.finalize() {
    print(result.adjusted)
}
```

## Where effort scores come from

In the 1970s, researchers proposed the fitness-fatigue model: each workout is an impulse that produces fitness, which builds slowly and lasts, and fatigue, which builds quickly and fades. To measure the size of each impulse they defined the training impulse, or TRIMP. By 1990 the number came from heart rate: the minutes of a workout, weighted by cardiovascular intensity. Summated heart-rate zones followed in 1993, and session rating of perceived exertion, a self-reported exertion multiplied by minutes, in 2001.

Most fitness watches still calculate training load from one or a combination of these ideas, and each one shares TRIMP's shape: intensity multiplied by time. TES keeps that shape. What changes is how intensity is decided.

## What a single number misses

Successful training requires balancing effort and recovery, and today's runners train muscles, tendons and ligaments as well as the heart. Several common efforts fall outside what a heart-rate score can see:

- **Running downhill.** On a steep descent the quadriceps contract eccentrically to brake every foot strike. Because gravity supplies much of the work, the effort may not register as significant cardiovascular activity.
- **Interval training.** Surges demand significant cardiovascular output, but averaging them together with the rest periods masks the true intensity.
- **Technical terrain.** Rocks, roots and loose footing demand concentration and agility on every stride. Pace slows and heart rate may stay moderate, so neither reflects the added work.
- **Power-hike.** On a steep climb a runner may drop to a hike. Pace and cadence collapse while cardiovascular effort increases.
- **Environmental factors.** In heat or late in a long run, heart rate climbs while the external workload holds steady. This is cardiac drift. At altitude, heart rate sits higher from the first step for the same work.
- **Returning from injury.** Measuring progress isn't about increasing cardiovascular effort but about carefully measuring biomechanical load, usually through a walk-run program.

Running power, the other established approach, reduces a run's biomechanics to watts. On a steep descent estimated power drops even as the quadriceps absorb heavy eccentric load, and on a hot day power never sees the heart. The comparison cuts both ways: a foot pod records better raw biomechanics than any wrist sensor, and for steady-state running a power-based load has years of field use behind it. On a descent, though, a heart-rate trace misses the same load that power does:

![One instant on a steep descent shown as two bars, cardiovascular cost from heart rate reading low while muscular load from eccentric quad braking reads high](diagram-load-cost)

## Keeping every dimension

TES captures each moment of a run as a vector of six signals and keeps them together rather than reducing them to a single number at the start. Each number represents one aspect of the moment, and together they position that moment in space. The distance between two vectors then shows their similarity: moments that look alike sit close together, moments that differ sit far apart.

A strong effort on flat ground and a rapid descent can share nearly the same pace, while their heart rate and vertical oscillation pull in opposite directions. As vectors, the two moments remain distinct:

![Two moments of a run as six-signal vectors, a threshold effort on flat ground and a fast descent, scaled to a common range](diagram-vector-modeling)

> Note: For more on vectors and distance, see <doc:Linear-Algebra-Primer>.

## How the model is built

TES asks two questions at each moment. First, what should the heart be doing for this workload? A personal baseline, learned from the runner's own history, predicts the heart rate the workload should produce. Second, what kind of effort does this moment look like? A classifier compares the movement pattern against known efforts and labels it. The score adds up these labeled moments, weighted by intensity and duration, plus a few costs that only exist across a whole run:

![The two branches of TES: the baseline predicts expected heart rate and reports the residual, while the classifier labels each moment and feeds the score](diagram-tes-architecture)

Internally the type wires three Quiver models together. A ``Ridge`` regression is the baseline, a ``ResidualModel`` over that baseline measures the gap, and a ``KNearestNeighbors`` classifier inside a ``Pipeline`` labels each moment. The two branches read different slices of the same moment on purpose:

| Model | Reads | Produces |
|---|---|---|
| Baseline | pace, cadence, grade, vertical oscillation, altitude | expected heart rate |
| Classifier | heart rate, pace, cadence, grade, vertical oscillation | an effort label |

Heart rate is the baseline's target, so it is never one of its inputs. Altitude is left out of the classifier because, as an absolute position, it describes where the runner is rather than how hard they are moving; its effect on heart rate belongs to the baseline.

## The baseline

The baseline learns the heart rate a runner's workload should produce from the last four weeks of that runner's history. It is a ridge regression, a linear regression with a light penalty that keeps the learning stable when signals overlap. The baseline doesn't require age, heart-rate zones or other demographic inputs, and it can be read directly:

```swift
if let baseline = tes.baseline {
    print(baseline.equation())
    // expected HR = 142.69 - 4.62·pace + 2.87·cadence + 4.85·grade
    //               - 3.95·verticalOscillation + 0.54·altitude

    let weights = baseline.coefficients    // intercept at index 0, then the five signals
    let overlap = baseline.conditioning    // 1-norm condition of XᵀX, nil when not measurable
}
```

This baseline was learned from a synthetic 12-run history. The first number, 142.7, is this runner's average heart rate: the heart rate expected when every signal sits at its usual value. Each coefficient then adds or subtracts beats per minute as its signal moves one standard deviation, a typical swing for this runner, above its average. Because every signal is put on the same scale, their sizes compare directly. For this runner, grade counts about nine times more than altitude. Pace is negative because it is measured in minutes per kilometer, so a larger number means slower running and a lower heart rate.

> Important: The coefficients describe standardized signals. Substituting a raw workload, such as an altitude of 300 meters, into the printed equation gives a meaningless result. The model applies its own scaler before predicting.

A weight near zero can mean a signal carries little information, or that it barely varied in the runner's history. The `conditioning` value often reads in the tens, because pace, cadence, grade and vertical oscillation move together. That is information about overlapping signals, not a failure; the ridge penalty keeps the fit stable. The baseline stores no fit-quality score on purpose, because quality is measured on held-out runs, not read off the trained model.

## The residual

Comparing expected heart rate against measured heart rate reveals the effect of everything the workload doesn't explain. A moment one standard deviation steeper than usual, with every other signal at its average, is expected to read 142.69 + 4.85 ≈ 147.5 bpm. If the watch reads 155 bpm at that moment, the gap of +7.5 bpm is the **residual**:

![A flat expected heart-rate line with the observed heart rate drifting above it over a run, the widening gap between them shaded as the residual](diagram-residual-modeling)

TES reports the residual live and as an average across the run:

```swift
let liveGap = tes.currentResidual        // observed − expected bpm; nil until a baseline exists
let sessionGap = result.meanResidual     // trust- and time-weighted; 0 without a baseline
let expected = result.expectedHeartRate  // mean expected bpm; nil on a first run
```

The residual is reported beside the score rather than added to it, because the model cannot tell heat from caffeine or fatigue. A high score driven by real work and a high score driven by heat must remain distinguishable. Because `meanResidual` reads 0 on a first run, an app checks `expectedHeartRate` for `nil` to tell a first run from an established one. On an established run the mean can also sit near 0 when gaps above and below prediction cancel, such as a hot climb followed by a shaded descent, so a near-zero mean does not by itself show that the heart tracked the workload all run. A positive session residual means the heart ran above prediction, the signature of heat or drift; a negative one means it ran cooler, as on a fast descent. Interpreting the cause is the app's job.

Heart rate never enters the score directly; it only helps decide which category a moment belongs to. Once a baseline exists, the residual also protects the effort label through a heart-rate cap. The classifier reads heart rate no higher than the baseline expects for the moment's workload, so heat and drift cannot lift a moment into a harder category; the excess stays visible in the residual instead. The cap is one-sided by design: a reading below expected passes through unchanged, so heart rate suppressed by cold or fatigue can still lower a label. A steady Tempo workload (5.6 min/km, 171 steps per minute, +0.5% grade) that reads Threshold at 162 bpm on a first run stays Tempo at 166 and even 180 bpm once a baseline exists.

## Effort categories

The classifier is a nearest-neighbor model that compares each moment with a set of known efforts and gives it the label of the closest ones. Each ``EffortClass`` carries a weight that sets how much a second of it counts:

| Category | Meaning | Weight |
|---|---|---|
| Recovery | Walking, assigned only by the walking gate | 0.10 |
| Easy | Recovery running and aerobic base | 0.25 |
| Tempo | Sustained sub-threshold, roughly marathon to half-marathon pace | 0.50 |
| Threshold | At lactate threshold | 0.75 |
| Hard | Above threshold, or a heavy biomechanical load heart rate doesn't show | 1.00 |

A steep-descent moment, with heart rate at 131 bpm, pace at 5.1 min/km and grade at −5.8%, reads Easy by heart rate alone. Its grade and vertical oscillation match the descent efforts among the known efforts, so TES labels it Hard. The top category is intentionally called Hard rather than a cardiovascular name such as VO₂ max. A steep eccentric descent belongs there because the legs are working at the top of their range while the heart is not.

The known efforts are all running, so walking never reaches the classifier. A walking gate labels it first. A moment with cadence under 130 steps per minute and pace slower than 8 min/km is walking, and its label depends on the terrain:

| Walking moment | Label |
|---|---|
| Climb, +3% or steeper | The higher of vertical speed in meters per hour (Tempo at 300, Threshold at 500, Hard at 700) and heart rate (Threshold at 150 bpm, Hard at 165) |
| Descent, −10% or steeper | Easy, or Tempo at 500 meters per hour |
| Anything else, flat included | Recovery |

The heart-rate thresholds are skipped when `hrTrust`, the app's confidence in each heart-rate reading described in *Trusting the heart-rate reading*, is under 0.5. A walking descent tops out at Tempo because braking at walking speed loads the legs far less than a running descent; the same slope taken at a running cadence, 130 steps per minute or more, can read Hard. Only the gate assigns Recovery; the classifier never predicts it.

The live label and the signals behind it are available on every sample:

```swift
let label = tes.currentEffort      // nil before the first sample
let signals = tes.currentSignals   // five z-scores against the known efforts, nil before the first sample
```

``EffortSignals`` reports the latest moment as `heartRate`, `pace`, `cadence`, `grade` and `verticalOscillation` z-scores. A large negative `grade` paired with a near-zero `heartRate` is the downhill signature: the legs working while the heart stays calm.

## From moments to a score

Time in each category adds up to a score, and harder categories count more. A moment's contribution is its weight multiplied by its duration in seconds, and the total is expressed against one hour held at threshold:

`raw = 100 × Σ(weight × Δt) / (0.75 × 3600)`

One hour at threshold reads 100, following the convention of power-based training stress scores. By convention, Threshold is a real, repeatable effort a runner can hold for about an hour, rather than an abstract maximum. Threshold efforts are fast running on flat to gently sloping ground. The score rises with both intensity and time and has no ceiling.

Values for easy running are low by design. An hour at Easy counts a third as much as an hour at Threshold. TES exists to price the work other scores miss, such as descents and surges, so its scale climbs most steeply at the hard end.

## Session costs

Some costs exist only across a whole run, and ``TESResult`` reports each one separately so an app can see what moved the score away from the raw total:

```swift
let swings = result.varianceMultiplier   // 1 + min(0.35, 0.25 × variance of the per-sample category level, Recovery −1 to Hard 3)
let fatigue = result.durationFactor      // 1.0 up to 45 minutes, then 1 + 0.1 × ln(minutes / 45)
let surges = result.transitionLoad       // total size of jumps of two categories or more
let headline = result.adjusted           // raw × swings × fatigue + 0.1 × surges
```

Intervals cost more than steady running of the same average effort, so a session that swings between categories raises the variance term; the first five minutes are left out so a warm-up doesn't count as a swing. Abrupt jumps of two or more categories, such as Easy straight to Hard, raise the transition term. Only categories held for at least ten seconds count, so a single misread sample at a boundary never registers as a surge.

Long runs get a small credit for time on feet past 45 minutes. With that credit, the same hour at threshold reads 102.9, the headline adjusted score. The credit grows slowly on purpose and flattens as runs get longer. For runners covering 40 or more miles a week, running coach Jack Daniels recommends a long run of no more than the lesser of 150 minutes or 25 percent of weekly mileage. At two hours the credit is about 10 percent; at 150 minutes, 12 percent.

## What a score means

The adjusted score sits on an absolute scale divided into five ranges. Each boundary represents slightly more than one continuous hour in a category: with the time-on-feet credit, a steady hour yields 34 at Easy, 69 at Tempo, 103 at Threshold and 137 at Hard. As a rule of thumb, every 35 points represents one hour spent at the next highest intensity.

| Range | Score | Equivalent continuous effort |
|---|---|---|
| Light | < 35 | Up to one hour at Easy |
| Moderate | 35–70 | Up to one hour at Tempo |
| Substantial | 70–105 | Up to one hour at Threshold |
| Heavy | 105–140 | Up to one hour at Hard |
| Extreme | ≥ 140 | More than one hour at Hard |

TES returns the number, and the app names the range:

```swift
// Returns the named range for an adjusted score.
func rangeName(for score: Double) -> String {
    switch score {
    case ..<35: return "Light"
    case ..<70: return "Moderate"
    case ..<105: return "Substantial"
    case ..<140: return "Heavy"
    default: return "Extreme"
    }
}
```

Because surges and abrupt changes compound the score, mixed-intensity sessions often reach higher ranges in less time than steady efforts. The ranges quantify the cost of a workout; they are not recovery suggestions for the next day. As a calibration point, a recreational runner logging 45 to 70 minute easy runs will generally score between 25 and 40, while a mountain runner sustaining high effort on climbs and descents will routinely land in the Heavy range.

## Scoring examples

Four constructed runs show how these design choices behave. Each notes whether it was scored as a *first run*, with no personal history, or by an *established runner* with a learned baseline. They illustrate intended model behavior rather than physiological accuracy. The downhill, hot-day and injury runs sample every five seconds, and the hill repeats every second.

**A downhill trail run (first run): 117.9, Heavy.** A 55-minute run with four extended −6% descents. Heart rate stays calm, but the movement data labels 58 percent of the run Hard. A heart-rate score would log an easy day; TES prices the eccentric load.

**Hill repeats against steady Tempo (first run): 76.4 against 52.4.** Both 47-minute sessions build nearly the same raw load, 52.8 against 52.2. The repeats, eight two-minute Hard efforts each followed by two minutes Easy, earn a variance multiplier of 1.350 and a transition load of 48. The session terms account for the whole difference.

**A hot day against a cool day (established runner): 62.4 against 62.3.** Both 60-minute runs hold 45 minutes at the same steady pace between a 10-minute warm-up and a 5-minute cool-down, both easy at 131 bpm. Cardiac drift on the hot day takes heart rate from 150 to 166 bpm across those 45 minutes. Because the work is the same, the extra heart rate can't raise the category, and the score barely changes. The extra beats appear in the residual instead: +3.1 bpm on the hot day against −2.8 bpm on the cool day.

**Returning from injury (established runner): 10.9 to 15.6, Light.** A runner recovering from a knee sprain follows a walk-run program. As running time grows from 5 to 28 minutes over six weeks, the score rises gradually from 10.9 to 15.6, with the walking read as Recovery. Adding an 8-minute descent to the same 28-minute run lifts the score to 39.6, Moderate, because grade and vertical oscillation label the downhill Hard while heart rate stays calm. Here TES acts as a load advisor: steady scores show a safe progression, and the spike flags stress heart rate alone would miss.

## Starting a run

An app picks one of two starting points. A brand-new runner starts empty. The classifier is seeded from a bundled set of 21 known efforts, inspectable as `TrueEffortScore.anchorSamples` and `TrueEffortScore.anchorLabels`, so a score exists from the first sample, while the baseline waits for a finished run:

```swift
import Quiver

var tes = TrueEffortScore()   // first run: no personal history yet

tes.baseline       // nil, the first-run tell
tes.sessionCount   // 0
```

The defaults are `lambda` 0.01 for the ridge penalty and `k` 3 neighbors. The initializer stops with a precondition failure if `k` exceeds the number of known efforts, so an app that lets a user choose `k` checks it first against the fixed count of bundled known efforts:

```swift
let neighborCount = 5
guard neighborCount <= TrueEffortScore.anchorSamples.count else { return }
var tunedModel = TrueEffortScore(k: neighborCount)
```

A returning runner can also be seeded from saved workouts. The throwing initializer fits the baseline once from that history and can add the runner's own labeled moments to the classifier, each a row of `[heartRate, pace, cadence, grade, verticalOscillation]`:

```swift
var tes = try TrueEffortScore(
    history: savedWorkouts,          // [Workout] from earlier finalized runs
    labeledSamples: labeledRows,     // optional personal rows for the classifier
    labels: rowLabels)               // one EffortClass per row; counts must match
```

Its limit depends on the rows the app supplies at runtime, so it throws instead: ``TESError/insufficientLabeledSamples(k:available:)`` when `k` exceeds the known efforts plus the supplied rows, and ``TESError/baselineDiverged(_:)`` if the first fit fails. Unequal `labeledSamples` and `labels` counts are a programmer error and stop with a precondition failure. Nothing after construction throws. Supplying personal rows also refits the classifier's standardization over the known efforts and those rows together, so the z-scores in `currentSignals` shift slightly.

> Important: Personal labels must come from ground truth: the runner, a coach, or a calibration session with a known effort. Feeding the model's own `currentEffort` readings back in as labels would train the classifier on its own guesses, a self-training loop with nothing to correct it.

## Recording a live run

The lifecycle runs in a fixed order: construct once and hold the model in a `var`, call `record` once per sensor sample, then call `finalize` once to close the run.

```swift
tes.record(
    heartRate: 152, pace: 5.4, cadence: 174, grade: 1.2,
    verticalOscillation: 7.9, altitude: 100,
    at: sampleDate)
// bpm, min/km, steps/min, percent (negative is downhill), cm, meters
```

Each call takes the raw signals and an absolute `Date`, and the model works out the time between samples itself. All six signals are required on every call; there is no missing-value handling, so when a sensor drops a reading, the app carries the last value forward. Two rules protect the stream from noisy hardware. A sample whose time isn't later than the previous one is dropped, so a duplicate or out-of-order callback discards itself. The first sample of a run carries no load but stamps the run's start.

Grade is a signed percentage, rise over run, negative downhill. No watch sensor reports it directly, so the app supplies it. It is one of the baseline's strongest signals and the basis for recognizing descents, so any smoothing applied before the value reaches `record` changes the scores TES produces.

Any steady sample interval works. Every moment is weighted by its duration, so the hill-repeat session in the scoring examples scores 76.4 whether it is sampled every 1, 2, 5 or 10 seconds. The interval should stay steady within a run, because the variance term counts samples rather than seconds: in an irregular stream, stretches sampled more often carry more weight in the variance.

Pace has no meaningful value when the runner stands still: zero speed is an infinite number of minutes per kilometer. An app pauses the model while the runner is stopped, at a crossing or a water stop, rather than recording that moment. Pausing keeps two clocks apart. A paused span counts toward wall-clock `elapsedTime` but not toward `timerTime`, the clock the score uses:

```swift
tes.pause()      // stop the timer; the wall clock keeps running
tes.resume()     // the next sample bridges the gap on the wall clock only
tes.isPaused     // false
```

A run the runner cancels is cleared without being kept:

```swift
tes.discardRun()   // clears the live run; history and baseline are untouched
```

While the run streams, the model exposes live readouts available after every sample. The score only climbs:

```swift
let score = tes.currentScore     // raw score so far, before session costs
let active = tes.timerTime       // active seconds, pauses excluded
let wall = tes.elapsedTime       // wall-clock seconds, pauses included
let samples = tes.sampleCount    // moments recorded this run
```

## Trusting the heart-rate reading

Optical heart rate is not always reliable. A loose band, cold skin, or cadence lock, where the sensor tracks foot strikes instead of heartbeats, can all distort it. The optional `hrTrust` parameter lets the app mark how far it trusts each reading, from 0 to 1, and it defaults to 1.

A doubted reading moves toward the runner's own expected heart rate once a baseline exists, or toward the average heart rate of the known efforts on a first run. The movement signals then decide the label, and the doubted stretch counts for less in the session residual. Apple Watch reports no per-sample confidence, so the app supplies the rule. One illustrative rule lowers trust when heart rate sits on cadence:

```swift
// Lowers trust when heart rate tracks cadence, a sign of cadence lock.
let trust = abs(heartRate - cadence) < 5 ? 0.2 : 1.0

tes.record(
    heartRate: heartRate, pace: pace, cadence: cadence, grade: grade,
    verticalOscillation: verticalOscillation, altitude: altitude,
    at: sampleDate, hrTrust: trust)
```

The 5 bpm and 0.2 values are illustrative. At hard efforts, heart rate and cadence can legitimately cross for a moment, so a production rule should require the match to persist, for about ten seconds, before lowering trust. The app computes trust; the model only applies it.

## Running indoors

The watch cannot sense treadmill incline: its barometric altimeter measures a change in altitude, and a runner on an inclined belt never rises. Marking the run indoors before the first sample records every grade as a fixed 0.5%, a common treadmill setting, scoring the run as level ground:

```swift
tes.location = .indoor   // set before the first sample
```

An indoor run is scored but kept out of history, so a grade the model cannot trust never shapes the runner's outdoor baseline. Ending the run resets the location to `.outdoor`.

## Finalizing the run

Calling `finalize()` closes the run and returns a ``TESResult``. A run with less than 60 seconds of active time is cleared and returns `nil`. That is neither an error nor a zero score; it is the model declining to keep a run too short to mean anything.

```swift
guard let result = tes.finalize() else { return }   // nil under 60 active seconds

print(result)
```

Printing a result always shows the breakdown, never a bare number. For the hot-day run in the scoring examples:

```
TESResult:
  adjusted:        62.4
  raw:             58.4
  meanResidual:    3.1 bpm
  expected HR:     148.0 bpm
  variance ×:      1.038
  duration ×:      1.029
  transitionLoad:  0.00
  timer/elapsed:   3600s / 3600s
  baseline:        expected HR = 142.69 - 4.62·pace + 2.87·cadence + 4.85·grade - 3.95·verticalOscillation + 0.54·altitude
```

``TESResult`` has no bare score accessor, because the headline is one of several readouts a run earns. Beyond `adjusted` and `raw`, two readouts suit charts:

```swift
let shares = result.effortDistribution   // [EffortClass: Double], share of active time per category
let easy = result.time(in: .easy)        // seconds spent in Easy, 0 if the run never entered it
let trace = result.loadCurve             // cumulative weighted load at each sample
let scoredBy = result.baseline           // the baseline that scored this run, nil on a first run
```

The shares are a fraction of active time, not of samples, and sum to about 1. The `baseline` on a result is captured before the refit, so it can differ from `tes.baseline`, which has already learned from this run.

## How the model learns a runner

At finalize an outdoor run joins history and the baseline refits. The baseline learns from the runs of the last 28 days. When fewer than eight runs fall inside that window, the eight most recent are kept instead, and no more than 60 are ever kept. All three are initializer parameters, `historyWindow`, `minimumHistoryRuns` and `historyLimit`. A window rather than the whole history lets the baseline follow a runner who is improving. The baseline refits between runs, never during one, so a run is always scored against expectations it has not yet changed; a baseline refit mid-run would absorb the very drift its residual exists to report. On a Mac with an M4 Pro, `finalize()` with a refit over 57 hour-long runs took 32 milliseconds. A watch is slower, so an app measures the refit on its target device.

The model moves through three phases. On a first run the classifier scores from the known efforts. Early on, the baseline has only a few runs to learn from, so residuals run large. As runs accumulate, the baseline drifts toward the runner's own heart-rate response, and residuals shrink. Once established, residuals center near zero on ordinary runs, so a large one means something:

![Session residuals falling across three phases, large during first runs, shrinking while personalizing, and settled near zero once established](diagram-personalization-model)

A sea-level runner who starts training at altitude follows the same path. Altitude is one of the baseline's signals, so the baseline adapts over several runs once altitude varies in the history. Until then, the heart-rate cap keeps the excess out of the effort label and in the residual.

## Saving the model

``TrueEffortScore`` is a plain value type built from numbers, and it conforms to `Codable`, `Equatable` and `Sendable`. Encoding the model saves the runner's history and fitted baseline together, and decoding restores the baseline exactly as stored, with no refit and no separate weights file:

```swift
let encoder = JSONEncoder()
let saved = try encoder.encode(tes)

let decoder = JSONDecoder()
let restored = try decoder.decode(TrueEffortScore.self, from: saved)
restored == tes   // true
```

Only settled state is encoded; the live run is excluded from both encoding and equality, so the model is saved between runs, never during one. A run in progress is therefore lost if the app ends unexpectedly. Because `record` takes absolute dates, replaying the same calls after a relaunch scores the run exactly as it would have scored live.

The saved model grows with its history, because each workout keeps every sample. A full history of 57 hour-long runs sampled every five seconds encodes to about 6.1 MB as JSON and about 1.6 MB as a binary property list, the smaller of the two. The binary figure assumes readings at sensor precision, such as whole beats per minute, because a binary property list stores each repeated value once; unrounded values grow it to about 5 MB:

```swift
let encoder = PropertyListEncoder()
encoder.outputFormat = .binary
let compact = try encoder.encode(tes)
```

`Sendable` lets the model cross actor boundaries without locks. Scoring is deterministic, with ties broken by the nearest neighbor, so the same run always scores the same.

## Where to go from here

TES combines simple, readable pieces rather than one opaque algorithm. See <doc:Ridge-Regression> for the baseline, <doc:Residual-Model> for the gap it measures, <doc:Nearest-Neighbors-Classification> for the classifier, <doc:Feature-Scaling> for the standardization that keeps the two aligned, and <doc:Model-Interpretation-Primer> for reading the baseline's coefficients honestly.

> Experiment: **The Quiver Notebook** is the right place to watch a label change while heart rate stands still. Start an empty ``TrueEffortScore``, record two minutes of flat running at 131 bpm, 5.1 min/km, 172 steps per minute and 8 cm of vertical oscillation, then two more minutes at the same heart rate, pace and cadence on a −6% grade with 10 cm of vertical oscillation. Print `currentEffort` and `currentSignals` after each stretch. The heart rate never moves, yet the label climbs from Tempo to Hard, and the `grade` and `verticalOscillation` z-scores show why. See <doc:Quiver-Notebook>.

## Topics

### Effort model
- ``TrueEffortScore``
- ``WorkoutLocation``

### Inputs
- ``Workout``
- ``EffortSignals``

### Results
- ``TESResult``
- ``TESBaseline``
- ``EffortClass``

### Errors
- ``TESError``
