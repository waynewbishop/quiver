# Physics Primitives Primer

Compute work, energy, and frequency from signals using vector and time-series tools in Quiver.

## Overview

Apple Watch and iPhone contain numerous sensors. The accelerometer reports how the device is being shaken or tilted, the altimeter reports how high it is and the heart-rate monitor reports beats per minute. These sensors produce arrays of numbers and can be analyzed and interpreted by Quiver.

> Note: This article assumes no physics background. However, it assumes familiarity with vectors, dot products, and the discrete Fourier transform. See <doc:Linear-Algebra-Primer> and <doc:Fourier-Transform> for an introduction and background.

### Vectors are physical quantities

A `[Double]` of three numbers is a point, a feature row, or a **physical vector**. An accelerometer sample is a three-number reading of force per unit mass, measured in metres per second squared along each of the watch's axes. Its magnitude tells us how hard the watch is being pulled, and the direction tells us which way:

```swift
import Quiver

// A still watch lying face-up: roughly 1g pointing down through the screen.
let sample: [Double] = [0.2, -0.1, 9.7]

sample.magnitude                      // ≈ 9.70  — close to g (9.81 m/s²)
sample.angle(with: [0, 0, 1])         // ≈ 0.024 rad — almost perfectly upright
```

The same `magnitude` we use to compute the length of an embedding vector in <doc:Semantic-Search> tells us the **strength** of an acceleration. The same `angle(with:)` we use to measure similarity between two documents tells us how tilted the watch is. The operations are identical. The units make them physics.

### The dot product is work

If a force vector pushes an object through a displacement vector, the **work done** is the dot product of the two. The formula in every physics textbook is `W = F · d`. The Swift call is one line:

```swift
import Quiver

let force: [Double]        = [120.0, 0.0, 0.0]    // 120 N forward
let displacement: [Double] = [10.0, 0.0, 2.0]     // 10 m forward, 2 m up

let work = force.dot(displacement)                // 1200 J
```

The runner pushed 120 N forward through a 10 m forward stride, so they did 1,200 J of work. The 2 m vertical component contributes nothing because the horizontal force has no vertical component.

The next example below gives us **power**, the rate at which work is being done. If `force` is the propulsive force vector and `velocity` is the velocity vector, then `P = F · v`:

```swift
let velocity: [Double] = [3.5, 0.0, 0.0]          // 3.5 m/s forward
let power = force.dot(velocity)                   // 420 W
```

### Integration transforms acceleration and velocity

A watch records samples. Each sample is a **rate**: how fast the heart is beating currently, how many watts the rider is producing currently. Often, what we actually want instead is the **total**. For example, how much energy was spent across the whole interval, not the instantaneous wattage. The mathematical name for that conversion is **integration**.

The idea is straightforward. Plot the rate on the vertical axis and time on the horizontal axis. The area under that curve is the total. If a cyclist holds 250 watts for 60 seconds, the rectangle under the line is 250 × 60 = 15,000 joules. The challenge is that real signals do not hold steady; the wattage wobbles up and down sample by sample, and the area under a wobbly curve is not a clean rectangle.

> Tip: Newton himself wrote about the **trapezoid rule** in the *Principia*. The idea is to treat each adjacent pair of samples as the two parallel sides of a trapezoid: average the two heights, multiply by the time between them, and that gives the area of that little slice. Add up every slice across the signal, and the total is a close approximation of the true area under the curve. The smaller the spacing between samples, the closer the approximation gets to the real answer.

Quiver exposes the trapezoid rule as `trapezoidalIntegral(dt:)`, where `dt` is the time between samples in seconds. The method takes a series of rate samples and returns the accumulated total. In the Notebook, substitute a real recording — see <doc:Notebook-Datasets>.

```swift
import Quiver

// Cycling power, sampled once per second over a 60-second interval.
let powerWatts: [Double] = [/* 60 power samples in W */]

// Total work done in joules: the area under the power-vs-time curve.
let totalWorkJ = powerWatts.trapezoidalIntegral(dt: 1.0) ?? 0
let totalWorkKJ = totalWorkJ / 1000.0             // kilojoules, the unit cyclists use
```

For a running total — work done so far at each time step, plotted as a curve — use `cumulativeTrapezoidal(dt:)`. The function returns an array the same length as the input, so we can chart it directly:

```swift
let workCurve = powerWatts.cumulativeTrapezoidal(dt: 1.0)
// workCurve[10] is total joules done by second 10
```

The same call structure converts an acceleration series to a velocity series, or a velocity series to a displacement series. But there is a trap here that the next paragraph names, and that every honest physics primer has to name.

> Note: Integrating noisy accelerometer data to recover velocity, and integrating velocity to recover position, is known in the sensor-fusion literature as the **double-integration problem**. A sensor bias as small as 0.01 m/s² accumulates to 18 m/s of false velocity after 30 minutes. Apple Watch and Garmin do not compute position from the accelerometer for this reason. The math is correct; the input is not clean enough. If we need accurate velocity from an accelerometer, we filter the signal first, or we use a different sensor. The primitive is honest about what it does — it integrates — and the calling code has to be honest about whether the input deserves to be integrated.

### Energy is squared velocity, height is potential

**Kinetic energy** is one-half mass times velocity squared. **Gravitational potential energy** is mass times g times height. Both follow from element-wise methods Quiver already provides.

We start with two constants. A 70-kilogram runner and the standard gravitational acceleration at sea level, 9.81 metres per second squared:

```swift
import Quiver

let mass = 70.0          // kg
let g = 9.81             // m/s²
```

Kinetic energy is computed in two steps. First, we square every velocity sample, which produces a new array of squared values. Then we multiply each squared value by one-half the mass, which converts the array into joules.

```swift
let velocity: [Double] = [3.0, 3.2, 3.4, 3.5, 3.6, 3.5, 3.4, 3.3, 3.2, 3.0]   // m/s samples

let velocitySquared = velocity.square()
let kineticEnergy = velocitySquared.broadcast(multiplyingBy: 0.5 * mass)   // joules per sample
```

The first line, `velocity.square()`, returns a new array where each element is the original value multiplied by itself. The second line, `broadcast(multiplyingBy:)`, multiplies every element of the array by the same scalar — a useful operation when one number applies uniformly to a whole series.

Potential energy is even simpler. There is no squaring step. We just multiply each altitude sample by the runner's mass and by g:

```swift
let altitude: [Double] = [120.0, 125.0, 132.0, 140.0, 148.0, 155.0, 161.0, 168.0, 174.0, 180.0]   // metres above sea level

let potentialEnergy = altitude.broadcast(multiplyingBy: mass * g)          // joules per sample
```

Each entry in `potentialEnergy` is the gravitational energy the runner has at that altitude, measured in joules.

The teaching moment is not the formulas. It is that squaring a velocity series produces a series with different *physical meaning* — energy instead of speed — even though the mathematical operation is the one we learned on page one. The primer's job here is to give the reader the habit of asking "what units are these numbers in now?" after every operation.

### Frequency is a physical quantity

A signal repeating 2.8 times per second has a frequency of 2.8 Hz. For a runner, that is cadence: 2.8 steps per second is 168 steps per minute. The discrete Fourier transform finds repeating rhythms in a signal, and the `dominantFrequency` property of the `PowerSpectrum` value returned by `powerSpectralDensity(sampleRate:windowed:)` reports the strongest one.

The next step is asking how much energy sits inside a given band of frequencies — for example, the cadence band between 2 Hz and 3.5 Hz for a typical runner. That question is answered by the **power spectral density**, which Quiver computes with `powerSpectralDensity(sampleRate:windowed:)`. The method returns a `PowerSpectrum` value that pairs each frequency with the energy at that frequency, and provides a method to ask for the energy inside any band. In the Notebook, substitute a real recording — see <doc:Notebook-Datasets>.

```swift
import Quiver

// 60 seconds of accelerometer magnitude sampled at 50 Hz.
let accelMagnitude: [Double] = [/* 3000 samples */]

guard let psd = accelMagnitude.powerSpectralDensity(sampleRate: 50, windowed: true) else { return }

let topHz = psd.dominantFrequency            // ≈ 2.83 Hz → 170 strides per minute
let cadenceEnergy = psd.bandEnergy(in: 2.0...3.5)
```

`PowerSpectrum` carries the frequencies, the densities, and the sample rate together as one value. The density is energy per hertz, so the band energy is the area under the spectrum across the requested range. For a runner, a high cadence-band energy means steady, rhythmic strides. For someone walking, the same energy sits in a different band. For a watch sitting on a desk, almost all the energy is at zero hertz. A classifier like K-Nearest Neighbors can tell those three states apart from band-energy features alone.

### What we can now compute from a workout

The six lines below were not single-call operations before this primer. Now each is one method on an array of doubles.

```swift
import Quiver

let totalWorkJ    = powerWatts.trapezoidalIntegral(dt: 1.0) ?? 0
let workCurve     = powerWatts.cumulativeTrapezoidal(dt: 1.0)
guard let spectrum = accelMagnitude.powerSpectralDensity(sampleRate: 50, windowed: true) else { return }
let cadenceHz     = spectrum.dominantFrequency
let cadenceEnergy = spectrum.bandEnergy(in: 2.0...3.5)
let kinetic       = velocity.square().broadcast(multiplyingBy: 0.5 * mass)
let potential     = altitude.broadcast(multiplyingBy: mass * 9.81)
```

The recipes that come next in the cookbook build on these primitives directly: total work as a fatigue feature for activity classification, band-energy as an input to a K-Nearest Neighbors run/walk/idle classifier, and gravitational work done climbing as a chartable summary of a hike.

> Experiment: Build a synthetic accelerometer trace in the **Quiver Notebook** — a sine wave at 2.8 Hz mixed with a little random noise, sampled 3,000 times to represent 60 seconds at 50 Hz. Compute its power spectral density and check that the dominant peak lands at 2.8 Hz. Now change the frequency to 1.8 Hz, regenerate the trace, and compute the band energy across 2 Hz to 3.5 Hz versus 1 Hz to 2 Hz. Which band wins in each case? The two scalars — high-cadence energy and low-cadence energy — are the first two features of a classifier that does not yet exist. See <doc:Quiver-Notebook>.

## Building physical intuition

The next places this primer connects are <doc:Fourier-Transform>, which goes deeper on frequency content and windowing, and <doc:Quiver-Notebook>, where these primitives can be explored against synthetic signals before they are wired into a watchOS or iOS app. From there, the natural step is feature engineering for the existing classifiers — using band energies and integrated work as inputs to <doc:Nearest-Neighbors-Classification> and <doc:Naive-Bayes>, or as numeric predictors for <doc:Linear-Regression> and as features for <doc:KMeans-Clustering>. Physics, in this library, is a vocabulary for what the math is already doing. The functions are the same. The units are the point.
