# Physics Primitives Primer

Compute work, energy, and frequency from signals using vector and time-series tools in Quiver.

## Overview

Apple Watch and iPhone sensors produce arrays of numbers that we can analyze and interpret using Quiver. The accelerometer reports motion force, the altimeter measures elevation, and the heart-rate monitor tracks beats per minute.

> Note: This article assumes familiarity with vectors, dot products, and the discrete Fourier transform. See <doc:Linear-Algebra-Primer> and <doc:Fourier-Transform> for introductions to these topics.

### Vectors as physical quantities

A `[Double]` of three numbers is more than a simple row of features; it is a **physical vector**. An accelerometer sample reads the force per unit mass in metres per second squared along each of the device's axes. Its magnitude tells us the strength of the pull, while its direction indicates the orientation:

```swift
import Quiver

// A still watch lying face-up: roughly 1g pointing down through the screen.
let sample: [Double] = [0.2, -0.1, 9.7]

sample.magnitude                      // ≈ 9.70  — close to g (9.81 m/s²)
sample.angle(with: [0, 0, 1])         // ≈ 0.024 rad — almost perfectly upright
```

The same `magnitude` we use for embedding vectors tells us the **strength** of an acceleration. The `angle(with:)` we use for document similarity indicates how tilted the device is. The operations are identical; only the physical units change their meaning.

### The dot product is work

If a force vector pushes an object through a displacement vector, the **work done** is the dot product of the two. While the physics textbook formula is `W = F · d`, the Swift implementation is one line:

```swift
import Quiver

let force: [Double]        = [120.0, 0.0, 0.0]    // 120 N forward
let displacement: [Double] = [10.0, 0.0, 2.0]     // 10 m forward, 2 m up

let work = force.dot(displacement)                // 1200 J
```

A runner pushing 120 N forward through a 10 m stride does 1,200 J of work. The 2 m vertical component contributes nothing, as the horizontal force lacks a vertical component.

**Power** — the rate at which work is done — follows the same principle. If `force` is the propulsive force vector and `velocity` is the velocity vector, then `P = F · v`:

```swift
let velocity: [Double] = [3.5, 0.0, 0.0]          // 3.5 m/s forward
let power = force.dot(velocity)                   // 420 W
```

### Integration converts rates to totals

A watch records instantaneous rates — heart rate, watts, or speed. To calculate **totals** — total energy spent, distance covered — we use **integration**.

Plot the rate on the vertical axis and time on the horizontal axis; the area under the curve is the total. If a cyclist holds 250 watts for 60 seconds, the area (a 250 × 60 rectangle) is 15,000 joules. Because real signals wobble, we treat adjacent pairs of samples as the sides of a trapezoid, average their heights, and multiply by the time interval.

We compute this using `trapezoidalIntegral(dt:)`, where `dt` is the time interval in seconds.

```swift
import Quiver

// Cycling power, sampled once per second over a 60-second interval.
let powerWatts: [Double] = [/* 60 power samples in W */]

// Total work in joules: the area under the power-vs-time curve.
let totalWorkJ = powerWatts.trapezoidalIntegral(dt: 1.0) ?? 0
let totalWorkKJ = totalWorkJ / 1000.0             // kilojoules
```

For a running total (work done so far at each time step), use `cumulativeTrapezoidal(dt:)` to get an array of accumulated values, ready for charting:

```swift
let workCurve = powerWatts.cumulativeTrapezoidal(dt: 1.0)
// workCurve[10] is total joules done by second 10
```

> Note: Integrating noisy sensor data, known as the **double-integration problem**, is famously sensitive to bias. A tiny sensor bias can accumulate into massive errors in velocity or position over minutes. Quiver's primitives are mathematically correct, but calling code must be honest about whether the input signal is clean enough for integration. If we need accurate velocity from an accelerometer, we should filter the signal first or use sensor fusion.

### Energy is squared velocity, height is potential

**Kinetic energy** is one-half mass times velocity squared; **gravitational potential energy** is mass times gravity times height. Both rely on element-wise methods Quiver already provides.

We start with two constants: a 70-kilogram runner and standard gravity, 9.81 metres per second squared:

```swift
import Quiver

let mass = 70.0          // kg
let g = 9.81             // m/s²
```

To compute kinetic energy, first square the velocity samples, then multiply each by half the runner's mass.

```swift
let velocity: [Double] = [3.0, 3.2, 3.4, 3.5, 3.6, 3.5, 3.4, 3.3, 3.2, 3.0]   // m/s samples

let velocitySquared = velocity.square()
let kineticEnergy = velocitySquared.broadcast(multiplyingBy: 0.5 * mass)   // joules per sample
```

`velocity.square()` returns a new array where each element is the original value squared. `broadcast(multiplyingBy:)` uniformly multiplies every element by a scalar — a useful pattern whenever one number applies to an entire series.

Potential energy is even simpler: multiply each altitude sample by mass and gravity:

```swift
let altitude: [Double] = [120.0, 125.0, 132.0, 140.0, 148.0, 155.0, 161.0, 168.0, 174.0, 180.0]   // metres

let potentialEnergy = altitude.broadcast(multiplyingBy: mass * g)          // joules per sample
```

The key takeaway is that squaring velocity changes the physical meaning from speed to energy. We must always ask, "what units are these numbers in now?" after every operation.

### Frequency is a physical quantity

A signal repeating 2.8 times per second has a frequency of 2.8 Hz. For a runner, this is cadence: 2.8 steps per second is 168 steps per minute. The discrete Fourier transform identifies these rhythms, and the `dominantFrequency` property of the `PowerSpectrum` returned by `powerSpectralDensity(sampleRate:windowed:)` reports the strongest one.

To find the energy within a specific range — such as the cadence band between 2 Hz and 3.5 Hz — we use the **power spectral density**. This method returns a `PowerSpectrum` value pairing each frequency with its energy density (energy per hertz). The energy inside any band is simply the area under the spectrum across that range.

```swift
import Quiver

// 60 seconds of accelerometer magnitude sampled at 50 Hz.
let accelMagnitude: [Double] = [/* 3000 samples */]

guard let psd = accelMagnitude.powerSpectralDensity(sampleRate: 50, windowed: true) else { return }

let topHz = psd.dominantFrequency            // ≈ 2.83 Hz → 170 strides per minute
let cadenceEnergy = psd.bandEnergy(in: 2.0...3.5)
```

`PowerSpectrum` bundles the frequencies, densities, and sample rate as one value. A high cadence-band energy indicates steady, rhythmic strides; for a watch on a desk, almost all energy concentrates at zero hertz. Classifiers like K-Nearest Neighbors use these band-energy features to distinguish between walking, running, and idling.

### What we can now compute from a workout

The following recipes were once complex routines; now, each is a single method on a `[Double]`.

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

These primitives are building blocks for higher-level analyses: total work serves as a fatigue feature for activity classification, band energy powers cadence classification, and gravitational work done climbing summarizes hike intensity.

> Experiment: Build a synthetic accelerometer trace in the **Quiver Notebook** — a sine wave at 2.8 Hz mixed with random noise, sampled 3,000 times (60 seconds at 50 Hz). Compute its power spectral density and verify the dominant peak at 2.8 Hz. Change the frequency to 1.8 Hz, regenerate the trace, and compare the band energy across 2 Hz to 3.5 Hz versus 1 Hz to 2 Hz. These scalars are the foundation for a run/walk/idle classifier. See <doc:Quiver-Notebook>.

## Building physical intuition

This primer connects to the <doc:Fourier-Transform> for deeper insights on frequency and windowing, and the <doc:Quiver-Notebook> to explore these primitives against synthetic signals before wiring them into an app. From there, the natural step is feature engineering for our classifiers — using band energies and integrated work as inputs for <doc:Nearest-Neighbors-Classification>, <doc:Naive-Bayes>, <doc:Linear-Regression>, or <doc:KMeans-Clustering>. Physics, in this library, is simply a vocabulary for what the math is already doing: the functions are the same; the physical units are the point.
