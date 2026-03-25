// Copyright 2025 Wayne W Bishop. All rights reserved.
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
import Foundation
@testable import Quiver

final class CodableTests: XCTestCase {

    // MARK: - Linear Regression

    // Round-trip encode/decode preserves all properties
    func testLinearRegressionCodable() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [2.1, 3.9, 6.1, 8.0, 9.8]
        let model = try LinearRegression.fit(features: features, targets: targets)

        let data = try JSONEncoder().encode(model)
        let decoded = try JSONDecoder().decode(LinearRegression.self, from: data)

        XCTAssertEqual(model, decoded)
    }

    // Decoded model produces identical predictions
    func testLinearRegressionPredictionsAfterDecode() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [2.1, 3.9, 6.1, 8.0, 9.8]
        let model = try LinearRegression.fit(features: features, targets: targets)

        let data = try JSONEncoder().encode(model)
        let decoded = try JSONDecoder().decode(LinearRegression.self, from: data)

        let testInput: [[Double]] = [[6.0], [7.0], [8.0]]
        XCTAssertEqual(model.predict(testInput), decoded.predict(testInput))
    }

    // Multi-feature regression round-trips correctly
    func testLinearRegressionMultiFeatureCodable() throws {
        let features: [[Double]] = [
            [1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]
        ]
        let targets = [5.0, 4.0, 11.0, 10.0, 17.0]
        let model = try LinearRegression.fit(features: features, targets: targets)

        let data = try JSONEncoder().encode(model)
        let decoded = try JSONDecoder().decode(LinearRegression.self, from: data)

        XCTAssertEqual(model, decoded)
        XCTAssertEqual(model.featureCount, decoded.featureCount)
        XCTAssertEqual(model.hasIntercept, decoded.hasIntercept)
    }

    // MARK: - Gaussian Naive Bayes

    // Round-trip encode/decode preserves model and class stats
    func testGaussianNaiveBayesCodable() {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]
        ]
        let labels = [0, 0, 1, 1]
        let model = GaussianNaiveBayes.fit(features: features, labels: labels)

        let data = try! JSONEncoder().encode(model)
        let decoded = try! JSONDecoder().decode(GaussianNaiveBayes.self, from: data)

        XCTAssertEqual(model, decoded)
    }

    // Decoded Naive Bayes produces identical predictions
    func testGaussianNaiveBayesPredictionsAfterDecode() {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]
        ]
        let labels = [0, 0, 1, 1]
        let model = GaussianNaiveBayes.fit(features: features, labels: labels)

        let data = try! JSONEncoder().encode(model)
        let decoded = try! JSONDecoder().decode(GaussianNaiveBayes.self, from: data)

        let testInput: [[Double]] = [[2.0, 2.5], [5.5, 7.0]]
        XCTAssertEqual(model.predict(testInput), decoded.predict(testInput))
    }

    // ClassStats round-trips independently
    func testClassStatsCodable() {
        let stats = GaussianNaiveBayes.ClassStats(
            label: 0, prior: 0.5, means: [1.25, 1.9], variances: [0.0625, 0.01], count: 2
        )

        let data = try! JSONEncoder().encode(stats)
        let decoded = try! JSONDecoder().decode(GaussianNaiveBayes.ClassStats.self, from: data)

        XCTAssertEqual(stats, decoded)
    }

    // MARK: - K-Means

    // Round-trip encode/decode preserves all KMeans properties
    func testKMeansCodable() {
        let trainingData: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
            [8.0, 8.0], [8.5, 7.5], [9.0, 8.5]
        ]
        let model = KMeans.fit(data: trainingData, k: 2, seed: 42)

        let data = try! JSONEncoder().encode(model)
        let decoded = try! JSONDecoder().decode(KMeans.self, from: data)

        XCTAssertEqual(model, decoded)
    }

    // Decoded KMeans produces identical cluster assignments
    func testKMeansPredictionsAfterDecode() {
        let trainingData: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
            [8.0, 8.0], [8.5, 7.5], [9.0, 8.5]
        ]
        let model = KMeans.fit(data: trainingData, k: 2, seed: 42)

        let data = try! JSONEncoder().encode(model)
        let decoded = try! JSONDecoder().decode(KMeans.self, from: data)

        let newPoints: [[Double]] = [[2.0, 2.0], [7.0, 7.0]]
        XCTAssertEqual(model.predict(newPoints), decoded.predict(newPoints))
    }

    // Cluster struct round-trips correctly
    func testClusterCodable() {
        let cluster = Cluster(
            centroid: [1.23, 1.97],
            points: [[1.0, 2.0], [1.5, 1.8], [1.2, 2.1]]
        )

        let data = try! JSONEncoder().encode(cluster)
        let decoded = try! JSONDecoder().decode(Cluster.self, from: data)

        XCTAssertEqual(cluster, decoded)
    }

    // MARK: - K-Nearest Neighbors

    // Round-trip encode/decode preserves all KNN properties
    func testKNearestNeighborsCodable() {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]
        ]
        let labels = [0, 0, 1, 1]
        let model = KNearestNeighbors.fit(features: features, labels: labels, k: 3)

        let data = try! JSONEncoder().encode(model)
        let decoded = try! JSONDecoder().decode(KNearestNeighbors.self, from: data)

        XCTAssertEqual(model, decoded)
    }

    // Decoded KNN produces identical predictions
    func testKNearestNeighborsPredictionsAfterDecode() {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]
        ]
        let labels = [0, 0, 1, 1]
        let model = KNearestNeighbors.fit(features: features, labels: labels, k: 3)

        let data = try! JSONEncoder().encode(model)
        let decoded = try! JSONDecoder().decode(KNearestNeighbors.self, from: data)

        let testInput: [[Double]] = [[2.0, 2.5], [5.5, 7.0]]
        XCTAssertEqual(model.predict(testInput), decoded.predict(testInput))
    }

    // KNN with cosine metric and distance weighting round-trips
    func testKNearestNeighborsCodableWithOptions() {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]
        ]
        let labels = [0, 0, 1, 1]
        let model = KNearestNeighbors.fit(
            features: features, labels: labels, k: 3,
            metric: .cosine, weight: .distance
        )

        let data = try! JSONEncoder().encode(model)
        let decoded = try! JSONDecoder().decode(KNearestNeighbors.self, from: data)

        XCTAssertEqual(model, decoded)
        XCTAssertEqual(decoded.metric, .cosine)
        XCTAssertEqual(decoded.weight, .distance)
    }

    // DistanceMetric enum round-trips
    func testDistanceMetricCodable() {
        for metric in [DistanceMetric.euclidean, DistanceMetric.cosine] {
            let data = try! JSONEncoder().encode(metric)
            let decoded = try! JSONDecoder().decode(DistanceMetric.self, from: data)
            XCTAssertEqual(metric, decoded)
        }
    }

    // VoteWeight enum round-trips
    func testVoteWeightCodable() {
        for weight in [VoteWeight.uniform, VoteWeight.distance] {
            let data = try! JSONEncoder().encode(weight)
            let decoded = try! JSONDecoder().decode(VoteWeight.self, from: data)
            XCTAssertEqual(weight, decoded)
        }
    }

    // MARK: - Feature Scaler

    // Round-trip encode/decode preserves scaler statistics
    func testFeatureScalerCodable() {
        let features: [[Double]] = [
            [619, 15000, 0.08], [502, 78000, 0.04],
            [850, 11000, 0.12], [720, 98000, 0.18]
        ]
        let scaler = FeatureScaler.fit(features: features)

        let data = try! JSONEncoder().encode(scaler)
        let decoded = try! JSONDecoder().decode(FeatureScaler.self, from: data)

        XCTAssertEqual(scaler, decoded)
    }

    // Decoded scaler produces identical transformations
    func testFeatureScalerTransformAfterDecode() {
        let features: [[Double]] = [
            [619, 15000, 0.08], [502, 78000, 0.04],
            [850, 11000, 0.12], [720, 98000, 0.18]
        ]
        let scaler = FeatureScaler.fit(features: features)

        let data = try! JSONEncoder().encode(scaler)
        let decoded = try! JSONDecoder().decode(FeatureScaler.self, from: data)

        let testInput: [[Double]] = [[680, 62000, 0.28]]
        XCTAssertEqual(scaler.transform(testInput), decoded.transform(testInput))
    }

    // MARK: - Classification

    // Classification result struct round-trips
    func testClassificationCodable() {
        let classification = Classification(
            label: 1,
            points: [[5.0, 8.0], [6.0, 9.0]]
        )

        let data = try! JSONEncoder().encode(classification)
        let decoded = try! JSONDecoder().decode(Classification.self, from: data)

        XCTAssertEqual(classification, decoded)
    }
}
