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

final class BernoulliNaiveBayesTests: XCTestCase {

    // Shared fixture. Class 0 (3 samples): feature 0 present 2×, feature 1 present 1×.
    // Class 1 (2 samples): feature 0 never present, feature 1 present 2×.
    // With alpha = 1: class 0 → [3/5, 2/5] = [0.6, 0.4]; class 1 → [1/4, 3/4] = [0.25, 0.75].
    private let features: [[Double]] = [
        [1.0, 0.0], [1.0, 1.0], [0.0, 0.0],
        [0.0, 1.0], [0.0, 1.0]
    ]
    private let labels = [0, 0, 0, 1, 1]

    // MARK: - Fit

    // Laplace-smoothed feature probabilities and priors match hand computation
    func testFitClassStats() {
        let model = BernoulliNaiveBayes.fit(features: features, labels: labels)

        XCTAssertEqual(model.classes.count, 2)
        XCTAssertEqual(model.featureCount, 2)
        XCTAssertEqual(model.alpha, 1.0)

        let class0 = model.classes[0]
        XCTAssertEqual(class0.label, 0)
        XCTAssertEqual(class0.prior, 0.6, accuracy: 1e-12)
        XCTAssertEqual(class0.count, 3)
        XCTAssertEqual(class0.featureProbabilities[0], 0.6, accuracy: 1e-12)
        XCTAssertEqual(class0.featureProbabilities[1], 0.4, accuracy: 1e-12)

        let class1 = model.classes[1]
        XCTAssertEqual(class1.label, 1)
        XCTAssertEqual(class1.prior, 0.4, accuracy: 1e-12)
        XCTAssertEqual(class1.count, 2)
        XCTAssertEqual(class1.featureProbabilities[0], 0.25, accuracy: 1e-12)
        XCTAssertEqual(class1.featureProbabilities[1], 0.75, accuracy: 1e-12)
    }

    // A smaller alpha trusts the observed counts more closely
    func testCustomAlpha() {
        let model = BernoulliNaiveBayes.fit(features: features, labels: labels, alpha: 0.5)

        // Class 1, feature 0: (0 + 0.5) / (2 + 1.0) = 1/6
        XCTAssertEqual(model.alpha, 0.5)
        XCTAssertEqual(model.classes[1].featureProbabilities[0], 1.0 / 6.0, accuracy: 1e-12)
        // Class 0, feature 0: (2 + 0.5) / (3 + 1.0) = 0.625
        XCTAssertEqual(model.classes[0].featureProbabilities[0], 0.625, accuracy: 1e-12)
    }

    // Classes are stored in ascending label order regardless of training order
    func testNonContiguousLabelsSortAscending() {
        let model = BernoulliNaiveBayes.fit(
            features: [[1.0], [0.0], [1.0]],
            labels: [7, 2, 7]
        )
        XCTAssertEqual(model.classes.map { $0.label }, [2, 7])
        XCTAssertEqual(model.predict([[0.0], [1.0]]), [2, 7])
    }

    // A single-class model predicts that class for every sample
    func testSingleClass() {
        let model = BernoulliNaiveBayes.fit(features: [[1.0, 0.0], [0.0, 1.0]], labels: [3, 3])
        XCTAssertEqual(model.classes.count, 1)
        XCTAssertEqual(model.classes[0].prior, 1.0)
        XCTAssertEqual(model.predict([[1.0, 1.0], [0.0, 0.0]]), [3, 3])
        XCTAssertEqual(model.predictProbabilities([[1.0, 1.0]])[0], [1.0])
    }

    // MARK: - Predict

    // Present and absent features both shape the score
    func testPredict() {
        let model = BernoulliNaiveBayes.fit(features: features, labels: labels)
        XCTAssertEqual(model.predict([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]), [0, 1, 0])
    }

    // Unnormalized log joint probabilities match hand computation
    func testPredictLogProbabilities() {
        let model = BernoulliNaiveBayes.fit(features: features, labels: labels)
        let logProbabilities = model.predictLogProbabilities([[1.0, 1.0]])

        // Class 0: 0.6 · 0.6 · 0.4 = 0.144; class 1: 0.4 · 0.25 · 0.75 = 0.075
        XCTAssertEqual(logProbabilities[0][0], Foundation.log(0.144), accuracy: 1e-12)
        XCTAssertEqual(logProbabilities[0][1], Foundation.log(0.075), accuracy: 1e-12)
    }

    // A feature never seen in a class lowers that class's probability without zeroing it —
    // the case where the Gaussian variant's 1e-9 variance floor becomes near-certain
    func testUnseenFeatureIsNotOverconfident() {
        let sample: [[Double]] = [[1.0, 1.0]]

        let bernoulli = BernoulliNaiveBayes.fit(features: features, labels: labels)
        let bernoulliProbabilities = bernoulli.predictProbabilities(sample)[0]
        // 0.144 / (0.144 + 0.075) and 0.075 / (0.144 + 0.075)
        XCTAssertEqual(bernoulliProbabilities[0], 0.144 / 0.219, accuracy: 1e-12)
        XCTAssertEqual(bernoulliProbabilities[1], 0.075 / 0.219, accuracy: 1e-12)

        // Class 1 never saw feature 0, so its Gaussian variance is floored at 1e-9
        let gaussian = GaussianNaiveBayes.fit(features: features, labels: labels)
        let gaussianProbabilities = gaussian.predictProbabilities(sample)[0]
        XCTAssertLessThan(gaussianProbabilities[1], 1e-6)
    }

    // Three classes work without one-vs-rest, as in masculine / feminine / neuter
    func testThreeClasses() {
        // Features: [ends in "-ung", ends in "-chen", ends in "-er"]
        // Labels: 0 = masculine, 1 = feminine, 2 = neuter
        let endings: [[Double]] = [
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]
        ]
        let genders = [0, 0, 0, 1, 1, 1, 2, 2, 2]

        let model = BernoulliNaiveBayes.fit(features: endings, labels: genders)

        XCTAssertEqual(model.classes.count, 3)
        XCTAssertEqual(model.predict(endings), genders)
        XCTAssertEqual(model.predict([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), [1, 2, 0])
    }

    // Each row of predicted probabilities sums to 1.0
    func testPredictProbabilitiesSumToOne() {
        let model = BernoulliNaiveBayes.fit(features: features, labels: labels)
        let probabilities = model.predictProbabilities([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])

        for row in probabilities {
            XCTAssertEqual(row.reduce(0.0, +), 1.0, accuracy: 1e-12)
        }
    }

    // argmax of predictProbabilities matches predict()
    func testPredictProbabilitiesMatchesPredict() {
        let model = BernoulliNaiveBayes.fit(features: features, labels: labels)
        let samples: [[Double]] = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
        let probabilities = model.predictProbabilities(samples)
        let predicted = model.predict(samples)

        for (row, predictedLabel) in zip(probabilities, predicted) {
            guard let argmaxIndex = row.indices.max(by: { row[$0] < row[$1] }) else {
                XCTFail("empty probability row"); return
            }
            XCTAssertEqual(model.classes[argmaxIndex].label, predictedLabel)
        }
    }

    // Empty query returns empty predictions
    func testEmptyQuery() {
        let model = BernoulliNaiveBayes.fit(features: features, labels: labels)
        XCTAssertEqual(model.predict([]), [])
        XCTAssertEqual(model.predictProbabilities([]), [])
    }

    // Scalar convenience predict agrees with the batch path for a single-feature model
    func testScalarPredict() {
        let model = BernoulliNaiveBayes.fit(
            features: [[1.0], [1.0], [0.0], [0.0]],
            labels: [1, 1, 0, 0]
        )
        XCTAssertEqual(model.predict(1.0), model.predict([[1.0]])[0])
        XCTAssertEqual(model.predict(1.0), 1)
        XCTAssertEqual(model.predict(0.0), 0)
    }

    // MARK: - Classify

    // classify() groups inputs by predicted label
    func testClassifyGroupsByLabel() {
        let model = BernoulliNaiveBayes.fit(features: features, labels: labels)
        let results = model.classify([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].label, 0)
        XCTAssertEqual(results[0].count, 2)
        XCTAssertEqual(results[1].label, 1)
        XCTAssertEqual(results[1].count, 1)
    }

    // MARK: - Pipeline

    // Full pipeline: trainTestSplit → fit → predict → confusionMatrix
    func testFullPipeline() {
        var samples: [[Double]] = []
        var classLabels: [Int] = []

        // Deterministic binary data: feature 0 marks class 0, feature 1 marks class 1,
        // feature 2 is noise shared by both classes
        for i in 0..<60 {
            samples.append([1.0, 0.0, i % 3 == 0 ? 1.0 : 0.0])
            classLabels.append(0)
        }
        for i in 0..<40 {
            samples.append([0.0, 1.0, i % 3 == 0 ? 1.0 : 0.0])
            classLabels.append(1)
        }

        let (trainX, testX) = samples.trainTestSplit(testRatio: 0.25, seed: 42)
        let (trainY, testY) = classLabels.trainTestSplit(testRatio: 0.25, seed: 42)

        let model = BernoulliNaiveBayes.fit(features: trainX, labels: trainY)
        let predictions = model.predict(testX)

        let cm = predictions.confusionMatrix(actual: testY)
        XCTAssertEqual(cm.accuracy, 1.0)
    }

    // MARK: - Conformances

    // Same training data produces equal models; a different alpha does not
    func testEquatable() {
        let model1 = BernoulliNaiveBayes.fit(features: features, labels: labels)
        let model2 = BernoulliNaiveBayes.fit(features: features, labels: labels)
        let model3 = BernoulliNaiveBayes.fit(features: features, labels: labels, alpha: 0.5)
        XCTAssertEqual(model1, model2)
        XCTAssertNotEqual(model1, model3)
    }

    // description reports classes, features, and alpha in plain language
    func testDescription() {
        let model = BernoulliNaiveBayes.fit(features: features, labels: labels)
        XCTAssertEqual(model.description, "BernoulliNaiveBayes: 2 classes, 2 features, alpha: 1")
        XCTAssertEqual(
            model.classes[0].description,
            "Class 0: prior 60.0%, feature probabilities [0.60, 0.40], 3 samples"
        )

        let singleFeature = BernoulliNaiveBayes.fit(features: [[1.0]], labels: [0], alpha: 0.5)
        XCTAssertEqual(singleFeature.description, "BernoulliNaiveBayes: 1 class, 1 feature, alpha: 0.5")
    }

    // MARK: - Shared Naive Bayes Core

    // Grouping keeps ascending label order and original sample order within a class
    func testSampleIndicesGroupedByLabel() {
        let groups = _NaiveBayes.sampleIndicesGroupedByLabel([4, 1, 4, 1, 9])
        XCTAssertEqual(groups.map { $0.label }, [1, 4, 9])
        XCTAssertEqual(groups.map { $0.indices }, [[1, 3], [0, 2], [4]])
    }

    // Ties resolve to the earliest label; all negative infinity falls back to the first label
    func testMostProbableLabelTiesAndInfinity() {
        XCTAssertEqual(
            _NaiveBayes.mostProbableLabel(logJointProbabilities: [-1.0, -1.0, -2.0], labels: [5, 6, 7]),
            5
        )
        XCTAssertEqual(
            _NaiveBayes.mostProbableLabel(logJointProbabilities: [-3.0, -0.5, -2.0], labels: [5, 6, 7]),
            6
        )
        XCTAssertEqual(
            _NaiveBayes.mostProbableLabel(
                logJointProbabilities: [-Double.infinity, -Double.infinity],
                labels: [2, 8]
            ),
            2
        )
    }
}
