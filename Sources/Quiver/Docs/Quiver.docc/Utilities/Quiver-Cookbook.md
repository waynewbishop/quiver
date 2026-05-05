# The Quiver Cookbook

Learning Quiver through domain problems in science, engineering, and math.

## Overview

[The Quiver Cookbook](https://github.com/waynewbishop/quiver-cookbook) is a collection of single-file recipes that simulate various technical scenarios from different domains. Every recipe uses the `Playground` macro from Xcode 26+, which evaluates the file in the Canvas.

> Important: The `Playground` macro is not the same as a `.playground` file. Traditional `Playground` files run in a separate sandbox and cannot import Swift packages. The playground macro compiles as part of the project, with full access to SPM dependencies including Quiver.

### A wind tunnel example

This receipe trains a `LinearRegression` model on six wind tunnel measurements of a NACA 2412 airfoil and predicts the lift `coefficient` at an angle the tunnel did not test:

```swift
import Playgrounds
import Quiver

#Playground("Wind Tunnel Lift Predictor") {

    // Simulated wind tunnel readings for a NACA 2412 airfoil
    // Angle of attack (degrees) → measured lift coefficient
    let angle =  [0.0,  2.0,  4.0,  6.0,  8.0, 10.0]
    let liftCL = [0.25, 0.47, 0.69, 0.90, 1.10, 1.30]

    // Train a regression model on the wind tunnel data
    let model = try LinearRegression.fit(features: angle, targets: liftCL)
    print(model)

    // Predict lift at an angle the tunnel hasn't tested
    let predicted = model.predict([7.0])
    print("Angle: 7° → CL: \(String(format: "%.2f", predicted[0]))")

    // How well does the linear model fit?
    let r2 = model.predict(angle).rSquared(actual: liftCL)
    print("R²: \(String(format: "%.4f", r2))")
}
```

The Canvas shows the fitted slope at roughly 0.11 per degree — the published value for a typical airfoil — along with the predicted lift at 7° and an R² close to 1.0 confirming the linear fit. The same `fit()` that predicts house prices also predicts whether a wing generates enough lift to fly.

> Note: NACA stands for the National Advisory Committee for Aeronautics — a U.S. federal agency founded in 1915 that ran wind tunnel research on airfoils for decades. It was dissolved in 1958 when its assets, staff, and facilities became the core of the newly created NASA.

### Can Swift drive a car?

A second recipe takes a different domain — sensor-driven decision making — and the same recipe shape. Given speed, distance to an obstacle, and lane offset, a [K-Nearest Neighbors model](<doc:Nearest-Neighbors-Classification>) votes on whether to accelerate, maintain, brake, or steer:

```swift
import Playgrounds
import Quiver

#Playground("Can Swift Drive a Car?") {

    // Simulated sensor readings from a car's instruments
    // Each row: [speed in mph, distance to obstacle in meters, lane offset in degrees]
    let telemetry: [[Double]] = [
        [60, 200, 0], [65, 180, 1], [55, 150, 0],  // open road — safe to accelerate
        [50, 120, 0], [55, 140, 1], [60, 160, 0],  // steady traffic — maintain speed
        [50,  50, 0], [45,  30, 2], [55,  40, 1],  // obstacle approaching — brake
        [40,  15, 0], [35,  20, 0], [45,  25, 0]   // too close to stop — steer around it
    ]

    // Labels: what a human driver did in each scenario
    // 0 = accelerate, 1 = maintain, 2 = brake, 3 = steer
    let actions = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

    // Scale features so speed, distance, and lane offset contribute equally
    let scaler = FeatureScaler.fit(features: telemetry)
    let scaled = scaler.transform(telemetry)

    // Train on the driving scenarios — KNN classifies each new reading
    // by finding the 3 most similar situations and voting
    let model = KNearestNeighbors.fit(features: scaled, labels: actions, k: 3, weight: .distance)
    print(model)

    // Simulate driving toward an obstacle at 55 mph
    let labels = ["Accelerate", "Maintain", "Brake", "Steer"]

    for distance in stride(from: 200.0, to: 5.0, by: -25.0) {
        let reading = scaler.transform([[55.0, distance, 0.0]])
        let decision = model.predict(reading)[0]
        print("Distance: \(Int(distance))m → \(labels[decision])")
    }
}
```

The Canvas prints a sequence of decisions that shift from `Accelerate` to `Maintain` to `Brake` to `Steer` as the simulated obstacle gets closer — no hardcoded thresholds, no if-else cascade, just the nearest training examples voting. Clearly, this is not a self-driving car but a playful simulation. Real autonomous vehicles use neural networks processing camera feeds, lidar, and radar. But the underlying question is the same — given what the sensors see right now, what should the car do next?

### The recipes, organized by domain

**Engineering and physical systems.** Recipes that turn a sensor reading or physical measurement into a decision or prediction. [Naive Bayes](<doc:Naive-Bayes>) classifies wing panel rivets against ±0.030″ tolerances. [Linear regression](<doc:Linear-Regression>) on NACA airfoil data recovers the published slope of about 0.11 per degree. [K-Means](<doc:KMeans-Clustering>) clusters a delivery driver's stops into geographic zones, and [KNN](<doc:Nearest-Neighbors-Classification>) classifies decisions from simulated driving sensor data.

**Athletics and motion.** Recipes that work on the kind of signal a watch or phone records every second. KNN combines heart rate, cadence, pace, and grade into a single effort label. The [Fourier transform](<doc:Fourier-Transform>) pulls breaths per minute from heart-rate variability and a runner's cadence from an accelerometer stream. A finite-difference derivative converts an elevation track into a grade signal — sensor in, useful number out.

**Math made tangible.** Recipes that put textbook mathematics to work on problems you can hold. A 2×2 [rotation matrix](<doc:Matrix-Transformations>) spins a vector around the origin. A [matrix inverse](<doc:Matrix-Operations>) backs out two unknown product prices from two equations. [Polynomial regression](<doc:Polynomials>) recovers the coefficients of `2x² + 3x + 1`, and analytic derivatives produce position, velocity, and acceleration in three calls. [Bootstrap resampling](<doc:Inferential-Statistics-Primer>) produces a 95% confidence interval with no closed-form assumption.

> Tip: The remaining recipes cover vectors and similarity, descriptive statistics, model evaluation, data preparation, pipelines, and probability — the foundations the domain examples are built on.

### From a recipe to your own data

Working with recipes is a starting point. Once the wind tunnel example runs and the Canvas confirms the fit, the natural next step is a different dataset — a different airfoil, a different domain, a CSV from a project. The recipe's structure stays the same; the values change.

That iteration loop is what the [Quiver Notebook](https://github.com/waynewbishop/quiver-notebook) is built for. The Notebook is a browser-based Swift editor with Quiver pre-imported and a library of bundled datasets ready to load. Take a recipe from the cookbook, paste it into the Notebook, swap the input array for a real dataset, and re-run. See <doc:Quiver-Notebook> for setup and <doc:Notebook-Datasets> for the bundled data.

### Getting the cookbook

Clone the repository and open the project in Xcode 26+

```bash
git clone https://github.com/waynewbishop/quiver-cookbook.git
```

Quiver is included as a package dependency and resolves automatically — no manual setup. Browse `Sources/Recipes/`, pick a file, and run it. Each recipe uses the `#Playground` macro, so results appear inline in the Canvas as we read.
