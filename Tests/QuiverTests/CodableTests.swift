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

    // Round-trip preserves equality, predictions, and multi-feature properties
    func testLinearRegressionCodable() throws {
        // Single-feature model — equality and predictions survive encode/decode
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [2.1, 3.9, 6.1, 8.0, 9.8]
        let model = try LinearRegression.fit(features: features, targets: targets)

        let data = try JSONEncoder().encode(model)
        let decoded = try JSONDecoder().decode(LinearRegression.self, from: data)
        XCTAssertEqual(model, decoded)

        let testInput: [[Double]] = [[6.0], [7.0], [8.0]]
        XCTAssertEqual(model.predict(testInput), decoded.predict(testInput))

        // Multi-feature — featureCount and hasIntercept preserved
        let multiFeatures: [[Double]] = [
            [1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]
        ]
        let multiTargets = [5.0, 4.0, 11.0, 10.0, 17.0]
        let multiModel = try LinearRegression.fit(features: multiFeatures, targets: multiTargets)
        let multiData = try JSONEncoder().encode(multiModel)
        let multiDecoded = try JSONDecoder().decode(LinearRegression.self, from: multiData)
        XCTAssertEqual(multiModel, multiDecoded)
        XCTAssertEqual(multiModel.featureCount, multiDecoded.featureCount)
        XCTAssertEqual(multiModel.hasIntercept, multiDecoded.hasIntercept)
    }

    // MARK: - Gaussian Naive Bayes

    // Round-trip preserves equality, predictions, and ClassStats
    func testGaussianNaiveBayesCodable() throws {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]
        ]
        let labels = [0, 0, 1, 1]
        let model = GaussianNaiveBayes.fit(features: features, labels: labels)

        let data = try JSONEncoder().encode(model)
        let decoded = try JSONDecoder().decode(GaussianNaiveBayes.self, from: data)
        XCTAssertEqual(model, decoded)

        // Predictions match after decode
        let testInput: [[Double]] = [[2.0, 2.5], [5.5, 7.0]]
        XCTAssertEqual(model.predict(testInput), decoded.predict(testInput))

        // ClassStats round-trips independently
        let stats = GaussianNaiveBayes.ClassStats(
            label: 0, prior: 0.5, means: [1.25, 1.9], variances: [0.0625, 0.01], count: 2
        )
        let statsData = try JSONEncoder().encode(stats)
        let statsDecoded = try JSONDecoder().decode(GaussianNaiveBayes.ClassStats.self, from: statsData)
        XCTAssertEqual(stats, statsDecoded)
    }

    // MARK: - K-Means

    // Round-trip preserves equality, cluster assignments, and Cluster struct
    func testKMeansCodable() throws {
        let trainingData: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
            [8.0, 8.0], [8.5, 7.5], [9.0, 8.5]
        ]
        let model = KMeans.fit(data: trainingData, k: 2, seed: 42)

        let data = try JSONEncoder().encode(model)
        let decoded = try JSONDecoder().decode(KMeans.self, from: data)
        XCTAssertEqual(model, decoded)

        // Predictions match after decode
        let newPoints: [[Double]] = [[2.0, 2.0], [7.0, 7.0]]
        XCTAssertEqual(model.predict(newPoints), decoded.predict(newPoints))

        // Cluster struct round-trips independently
        let cluster = Cluster(
            centroid: [1.23, 1.97],
            points: [[1.0, 2.0], [1.5, 1.8], [1.2, 2.1]]
        )
        let clusterData = try JSONEncoder().encode(cluster)
        let clusterDecoded = try JSONDecoder().decode(Cluster.self, from: clusterData)
        XCTAssertEqual(cluster, clusterDecoded)
    }

    // MARK: - K-Nearest Neighbors

    // Round-trip preserves equality, predictions, and configuration options
    func testKNearestNeighborsCodable() throws {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]
        ]
        let labels = [0, 0, 1, 1]
        let model = KNearestNeighbors.fit(features: features, labels: labels, k: 3)

        let data = try JSONEncoder().encode(model)
        let decoded = try JSONDecoder().decode(KNearestNeighbors.self, from: data)
        XCTAssertEqual(model, decoded)

        // Predictions match after decode
        let testInput: [[Double]] = [[2.0, 2.5], [5.5, 7.0]]
        XCTAssertEqual(model.predict(testInput), decoded.predict(testInput))

        // Cosine metric and distance weighting round-trip
        let configured = KNearestNeighbors.fit(
            features: features, labels: labels, k: 3,
            metric: .cosine, weight: .distance
        )
        let configuredData = try JSONEncoder().encode(configured)
        let configuredDecoded = try JSONDecoder().decode(KNearestNeighbors.self, from: configuredData)
        XCTAssertEqual(configured, configuredDecoded)
        XCTAssertEqual(configuredDecoded.metric, .cosine)
        XCTAssertEqual(configuredDecoded.weight, .distance)
    }

    // DistanceMetric and VoteWeight enums round-trip
    func testKNNEnumsCodable() throws {
        for metric in [DistanceMetric.euclidean, DistanceMetric.cosine] {
            let data = try JSONEncoder().encode(metric)
            let decoded = try JSONDecoder().decode(DistanceMetric.self, from: data)
            XCTAssertEqual(metric, decoded)
        }

        for weight in [VoteWeight.uniform, VoteWeight.distance] {
            let data = try JSONEncoder().encode(weight)
            let decoded = try JSONDecoder().decode(VoteWeight.self, from: data)
            XCTAssertEqual(weight, decoded)
        }
    }

    // MARK: - Feature Scaler

    // Round-trip preserves equality and transformation output
    func testFeatureScalerCodable() throws {
        let features: [[Double]] = [
            [619, 15000, 0.08], [502, 78000, 0.04],
            [850, 11000, 0.12], [720, 98000, 0.18]
        ]
        let scaler = FeatureScaler.fit(features: features)

        let data = try JSONEncoder().encode(scaler)
        let decoded = try JSONDecoder().decode(FeatureScaler.self, from: data)
        XCTAssertEqual(scaler, decoded)

        // Transformation output matches after decode
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
