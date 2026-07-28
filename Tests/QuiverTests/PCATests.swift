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

final class PCATests: XCTestCase {

    // The iris dataset (150 samples, 4 features), matching the validation script
    private let iris: [[Double]] = [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
        [5.4, 3.9, 1.7, 0.4],
        [4.6, 3.4, 1.4, 0.3],
        [5.0, 3.4, 1.5, 0.2],
        [4.4, 2.9, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1],
        [5.4, 3.7, 1.5, 0.2],
        [4.8, 3.4, 1.6, 0.2],
        [4.8, 3.0, 1.4, 0.1],
        [4.3, 3.0, 1.1, 0.1],
        [5.8, 4.0, 1.2, 0.2],
        [5.7, 4.4, 1.5, 0.4],
        [5.4, 3.9, 1.3, 0.4],
        [5.1, 3.5, 1.4, 0.3],
        [5.7, 3.8, 1.7, 0.3],
        [5.1, 3.8, 1.5, 0.3],
        [5.4, 3.4, 1.7, 0.2],
        [5.1, 3.7, 1.5, 0.4],
        [4.6, 3.6, 1.0, 0.2],
        [5.1, 3.3, 1.7, 0.5],
        [4.8, 3.4, 1.9, 0.2],
        [5.0, 3.0, 1.6, 0.2],
        [5.0, 3.4, 1.6, 0.4],
        [5.2, 3.5, 1.5, 0.2],
        [5.2, 3.4, 1.4, 0.2],
        [4.7, 3.2, 1.6, 0.2],
        [4.8, 3.1, 1.6, 0.2],
        [5.4, 3.4, 1.5, 0.4],
        [5.2, 4.1, 1.5, 0.1],
        [5.5, 4.2, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.2],
        [5.0, 3.2, 1.2, 0.2],
        [5.5, 3.5, 1.3, 0.2],
        [4.9, 3.6, 1.4, 0.1],
        [4.4, 3.0, 1.3, 0.2],
        [5.1, 3.4, 1.5, 0.2],
        [5.0, 3.5, 1.3, 0.3],
        [4.5, 2.3, 1.3, 0.3],
        [4.4, 3.2, 1.3, 0.2],
        [5.0, 3.5, 1.6, 0.6],
        [5.1, 3.8, 1.9, 0.4],
        [4.8, 3.0, 1.4, 0.3],
        [5.1, 3.8, 1.6, 0.2],
        [4.6, 3.2, 1.4, 0.2],
        [5.3, 3.7, 1.5, 0.2],
        [5.0, 3.3, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3],
        [6.5, 2.8, 4.6, 1.5],
        [5.7, 2.8, 4.5, 1.3],
        [6.3, 3.3, 4.7, 1.6],
        [4.9, 2.4, 3.3, 1.0],
        [6.6, 2.9, 4.6, 1.3],
        [5.2, 2.7, 3.9, 1.4],
        [5.0, 2.0, 3.5, 1.0],
        [5.9, 3.0, 4.2, 1.5],
        [6.0, 2.2, 4.0, 1.0],
        [6.1, 2.9, 4.7, 1.4],
        [5.6, 2.9, 3.6, 1.3],
        [6.7, 3.1, 4.4, 1.4],
        [5.6, 3.0, 4.5, 1.5],
        [5.8, 2.7, 4.1, 1.0],
        [6.2, 2.2, 4.5, 1.5],
        [5.6, 2.5, 3.9, 1.1],
        [5.9, 3.2, 4.8, 1.8],
        [6.1, 2.8, 4.0, 1.3],
        [6.3, 2.5, 4.9, 1.5],
        [6.1, 2.8, 4.7, 1.2],
        [6.4, 2.9, 4.3, 1.3],
        [6.6, 3.0, 4.4, 1.4],
        [6.8, 2.8, 4.8, 1.4],
        [6.7, 3.0, 5.0, 1.7],
        [6.0, 2.9, 4.5, 1.5],
        [5.7, 2.6, 3.5, 1.0],
        [5.5, 2.4, 3.8, 1.1],
        [5.5, 2.4, 3.7, 1.0],
        [5.8, 2.7, 3.9, 1.2],
        [6.0, 2.7, 5.1, 1.6],
        [5.4, 3.0, 4.5, 1.5],
        [6.0, 3.4, 4.5, 1.6],
        [6.7, 3.1, 4.7, 1.5],
        [6.3, 2.3, 4.4, 1.3],
        [5.6, 3.0, 4.1, 1.3],
        [5.5, 2.5, 4.0, 1.3],
        [5.5, 2.6, 4.4, 1.2],
        [6.1, 3.0, 4.6, 1.4],
        [5.8, 2.6, 4.0, 1.2],
        [5.0, 2.3, 3.3, 1.0],
        [5.6, 2.7, 4.2, 1.3],
        [5.7, 3.0, 4.2, 1.2],
        [5.7, 2.9, 4.2, 1.3],
        [6.2, 2.9, 4.3, 1.3],
        [5.1, 2.5, 3.0, 1.1],
        [5.7, 2.8, 4.1, 1.3],
        [6.3, 3.3, 6.0, 2.5],
        [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1],
        [6.3, 2.9, 5.6, 1.8],
        [6.5, 3.0, 5.8, 2.2],
        [7.6, 3.0, 6.6, 2.1],
        [4.9, 2.5, 4.5, 1.7],
        [7.3, 2.9, 6.3, 1.8],
        [6.7, 2.5, 5.8, 1.8],
        [7.2, 3.6, 6.1, 2.5],
        [6.5, 3.2, 5.1, 2.0],
        [6.4, 2.7, 5.3, 1.9],
        [6.8, 3.0, 5.5, 2.1],
        [5.7, 2.5, 5.0, 2.0],
        [5.8, 2.8, 5.1, 2.4],
        [6.4, 3.2, 5.3, 2.3],
        [6.5, 3.0, 5.5, 1.8],
        [7.7, 3.8, 6.7, 2.2],
        [7.7, 2.6, 6.9, 2.3],
        [6.0, 2.2, 5.0, 1.5],
        [6.9, 3.2, 5.7, 2.3],
        [5.6, 2.8, 4.9, 2.0],
        [7.7, 2.8, 6.7, 2.0],
        [6.3, 2.7, 4.9, 1.8],
        [6.7, 3.3, 5.7, 2.1],
        [7.2, 3.2, 6.0, 1.8],
        [6.2, 2.8, 4.8, 1.8],
        [6.1, 3.0, 4.9, 1.8],
        [6.4, 2.8, 5.6, 2.1],
        [7.2, 3.0, 5.8, 1.6],
        [7.4, 2.8, 6.1, 1.9],
        [7.9, 3.8, 6.4, 2.0],
        [6.4, 2.8, 5.6, 2.2],
        [6.3, 2.8, 5.1, 1.5],
        [6.1, 2.6, 5.6, 1.4],
        [7.7, 3.0, 6.1, 2.3],
        [6.3, 3.4, 5.6, 2.4],
        [6.4, 3.1, 5.5, 1.8],
        [6.0, 3.0, 4.8, 1.8],
        [6.9, 3.1, 5.4, 2.1],
        [6.7, 3.1, 5.6, 2.4],
        [6.9, 3.1, 5.1, 2.3],
        [5.8, 2.7, 5.1, 1.9],
        [6.8, 3.2, 5.9, 2.3],
        [6.7, 3.3, 5.7, 2.5],
        [6.7, 3.0, 5.2, 2.3],
        [6.3, 2.5, 5.0, 1.9],
        [6.5, 3.0, 5.2, 2.0],
        [6.2, 3.4, 5.4, 2.3],
        [5.9, 3.0, 5.1, 1.8],
    ]

    // Frozen wine-style fixture (60 samples, 5 features), matching the validation script
    private let wineStyle: [[Double]] = [
        [11.143027, 649.660515, 9.805627, 3.763063, 2.295651],
        [21.931293, 799.174606, 23.812558, 7.383548, 3.676939],
        [27.122844, 546.376243, 5.183708, 7.702405, 1.660557],
        [-5.152869, 913.544939, 31.082118, 2.039151, 3.920215],
        [12.255163, 715.987917, 19.40869, 5.959755, 2.525864],
        [-6.371454, 937.912885, 31.208846, 1.23161, 4.225197],
        [25.854996, 596.028933, 12.506932, 8.99435, 1.757582],
        [21.508455, 834.081799, 27.659314, 7.883777, 3.863073],
        [10.471131, 780.666365, 22.428247, 5.122103, 3.166622],
        [19.256749, 762.303865, 19.394012, 6.118946, 3.412561],
        [8.626912, 563.798325, 10.473132, 5.725231, 0.922903],
        [22.669689, 836.68516, 25.86392, 7.353588, 4.079237],
        [11.168348, 672.957003, 11.789258, 3.932694, 2.475745],
        [5.903332, 852.058167, 28.664205, 4.808728, 3.563089],
        [15.532236, 584.295978, 11.446025, 6.902544, 1.354569],
        [18.889474, 743.104847, 22.091308, 7.570372, 2.909003],
        [20.666271, 842.2627, 24.741325, 6.396408, 4.187557],
        [14.005819, 832.494294, 26.986371, 6.223934, 3.662767],
        [27.435375, 545.851343, 6.41634, 8.247618, 1.564561],
        [9.918496, 687.408444, 16.282101, 5.043017, 2.289104],
        [15.837628, 807.571697, 23.660188, 5.939145, 3.623047],
        [12.373256, 680.009146, 15.619239, 5.450191, 2.310292],
        [27.446163, 632.214238, 12.498672, 8.373993, 2.330725],
        [15.797231, 806.661879, 23.306328, 5.81917, 3.636779],
        [17.883552, 735.97459, 16.592082, 5.454327, 3.212185],
        [16.295206, 866.671272, 28.792094, 6.483285, 4.084091],
        [-9.907895, 834.425004, 25.509195, 1.010581, 3.072358],
        [-23.785985, 780.103921, 17.737958, -3.246391, 2.477388],
        [5.029958, 814.166348, 27.230705, 5.058909, 3.101989],
        [-9.898108, 678.53111, 17.482692, 1.918075, 1.455455],
        [28.424215, 755.839569, 20.629408, 8.519619, 3.502874],
        [17.334589, 928.42684, 32.737881, 6.61763, 4.695782],
        [-22.621308, 858.703943, 26.146467, -1.809852, 2.981767],
        [5.348931, 841.779106, 28.675602, 4.968721, 3.396407],
        [-2.131791, 730.405223, 18.648999, 2.534479, 2.352187],
        [-2.273465, 692.821112, 13.212091, 1.385448, 2.236401],
        [17.612955, 633.137924, 12.121941, 6.315254, 2.071486],
        [-7.613342, 549.263772, 6.703493, 1.530311, 0.512676],
        [24.546092, 768.948497, 23.180012, 8.415477, 3.370384],
        [23.38539, 600.501273, 7.906621, 6.64482, 2.112113],
        [7.421755, 887.046649, 27.30969, 3.689714, 4.223731],
        [19.85072, 813.883827, 25.979064, 7.43727, 3.653455],
        [26.108912, 723.110389, 17.349738, 7.654278, 3.218907],
        [17.721078, 784.675081, 23.667214, 6.888194, 3.349173],
        [18.304705, 978.379986, 39.131723, 7.97662, 4.939918],
        [17.271242, 886.65347, 32.893277, 7.730674, 4.077101],
        [-3.766994, 439.228349, -0.571927, 2.29306, -0.382426],
        [27.465942, 810.968998, 22.70558, 7.723364, 4.105903],
        [-5.085756, 769.98401, 20.70641, 1.743998, 2.670316],
        [30.225552, 666.62516, 12.360102, 7.977542, 2.925119],
        [-0.23315, 613.923764, 11.457322, 3.119173, 1.297236],
        [16.932816, 778.98465, 22.171431, 6.309359, 3.361534],
        [10.883411, 697.234727, 18.640247, 5.880326, 2.273442],
        [20.602069, 740.9725, 19.534255, 6.975555, 3.134182],
        [24.31135, 877.243511, 28.23769, 7.542958, 4.527785],
        [16.508081, 707.13251, 15.494407, 5.505584, 2.839864],
        [14.74884, 789.14986, 23.126514, 5.995523, 3.365457],
        [-0.930069, 909.089942, 31.305808, 3.050492, 3.967905],
        [29.976238, 908.640687, 31.237459, 8.979317, 4.917803],
        [2.364557, 728.956916, 19.052162, 3.590399, 2.437523],
    ]

    // MARK: - End-to-end reference comparison

    // Cross-validated against NumPy (means/components 1e-8, variances 1e-9 relative, ratios 1e-12)
    func testIrisMatchesReference() {
        let pca = PCA.fit(features: iris, componentCount: 2)

        XCTAssertEqual(pca.componentCount, 2)
        XCTAssertEqual(pca.featureCount, 4)

        let expectedMeans = [5.843333333333335, 3.057333333333334, 3.7580000000000027, 1.199333333333334]
        for f in 0..<4 {
            XCTAssertEqual(pca.means[f], expectedMeans[f], accuracy: 1e-8)
        }

        let expectedComponents: [[Double]] = [
            [0.3613865917853659, -0.08452251406457255, 0.8566706059498347, 0.3582891971515517],
            [0.6565887712868534, 0.7301614347850159, -0.17337266279585964, -0.0754810199174582]
        ]
        for k in 0..<2 {
            for f in 0..<4 {
                XCTAssertEqual(pca.components[k][f], expectedComponents[k][f], accuracy: 1e-8,
                    "Component \(k) entry \(f) mismatch")
            }
        }

        let expectedVariances = [4.228241706034831, 0.24267074792862006]
        for k in 0..<2 {
            XCTAssertEqual(pca.explainedVariances[k] / expectedVariances[k], 1.0, accuracy: 1e-9)
        }

        let expectedRatios = [0.9246187232017291, 0.05306648311706544]
        for k in 0..<2 {
            XCTAssertEqual(pca.explainedVarianceRatios[k], expectedRatios[k], accuracy: 1e-12)
        }
    }

    // Cross-validated against NumPy (transformed data, 1e-8 absolute)
    func testIrisTransformMatchesReference() {
        let pca = PCA.fit(features: iris, componentCount: 2)
        let projected = pca.transform(iris)

        XCTAssertEqual(projected.count, 150)
        XCTAssertEqual(projected[0].count, 2)

        let expectedFirstThree: [[Double]] = [
            [-2.6841256259695374, 0.3193972465850887],
            [-2.714141687294324, -0.17700122506478877],
            [-2.888990569059295, -0.14494942608557082]
        ]
        for row in 0..<3 {
            for k in 0..<2 {
                XCTAssertEqual(projected[row][k], expectedFirstThree[row][k], accuracy: 1e-8,
                    "Projected row \(row) component \(k) mismatch")
            }
        }
    }

    // Cross-validated against NumPy (inverse-transformed data, 1e-8 absolute)
    func testIrisInverseTransformMatchesReference() {
        let pca = PCA.fit(features: iris, componentCount: 2)
        let projected = pca.transform(iris)
        let reconstructed = pca.inverseTransform(Array(projected[0..<2]))

        let expectedFirstTwo: [[Double]] = [
            [5.083038967128149, 3.5174139311383756, 1.4032137224250771, 0.21353168781973186],
            [4.746261902457899, 3.1574999438188995, 1.4635617698194792, 0.2402459202316226]
        ]
        for row in 0..<2 {
            for f in 0..<4 {
                XCTAssertEqual(reconstructed[row][f], expectedFirstTwo[row][f], accuracy: 1e-8,
                    "Reconstructed row \(row) feature \(f) mismatch")
            }
        }
    }

    // Release gate: ratios sum to 1.0 when all components are retained,
    // and explained variances come back sorted descending
    func testFullRankRatiosSumToOne() {
        let pca = PCA.fit(features: iris, componentCount: 4)

        XCTAssertEqual(pca.explainedVarianceRatios.sum(), 1.0, accuracy: 1e-12)

        for k in 1..<4 {
            XCTAssertLessThanOrEqual(pca.explainedVariances[k], pca.explainedVariances[k - 1],
                "Explained variances must be sorted descending")
        }

        // Cross-validated against NumPy (ratios, 1e-12 absolute)
        let expectedRatios = [0.9246187232017291, 0.05306648311706544, 0.01710260980793043, 0.0052121838732750152]
        for k in 0..<4 {
            XCTAssertEqual(pca.explainedVarianceRatios[k], expectedRatios[k], accuracy: 1e-12)
        }
    }

    // Cross-validated against NumPy (standardize-first pipeline, 1e-8 absolute)
    func testStandardizedPipelineMatchesReference() {
        let scaler = StandardScaler.fit(features: wineStyle)
        let scaled = scaler.transform(wineStyle)
        let pca = PCA.fit(features: scaled, componentCount: 2)

        let expectedComponents: [[Double]] = [
            [0.15207414474460557, 0.5546508117763637, 0.5542600305266289, 0.17610534582726933, 0.5753422088007243],
            [0.6830063602184505, -0.22612997392314635, -0.18381822922839777, 0.6696916766414496, 0.00956366222540844]
        ]
        for k in 0..<2 {
            for f in 0..<5 {
                XCTAssertEqual(pca.components[k][f], expectedComponents[k][f], accuracy: 1e-8,
                    "Component \(k) entry \(f) mismatch")
            }
        }

        let expectedVariances = [2.9700878180303243, 2.0100392258143858]
        for k in 0..<2 {
            XCTAssertEqual(pca.explainedVariances[k] / expectedVariances[k], 1.0, accuracy: 1e-9)
        }

        let expectedRatios = [0.5841172708792972, 0.3953077144101626]
        for k in 0..<2 {
            XCTAssertEqual(pca.explainedVarianceRatios[k], expectedRatios[k], accuracy: 1e-12)
        }

        let projected = pca.transform(scaled)
        let expectedFirstThree: [[Double]] = [
            [-1.7287301890391304, -0.03222766451652703],
            [1.0751776521190115, 0.87131543231522],
            [-2.4320392899715215, 2.142655921633114]
        ]
        for row in 0..<3 {
            for k in 0..<2 {
                XCTAssertEqual(projected[row][k], expectedFirstThree[row][k], accuracy: 1e-8,
                    "Projected row \(row) component \(k) mismatch")
            }
        }
    }

    // MARK: - Numerical edge cases

    // Near-collinear features fit cleanly: no NaN, ratios still sum to one,
    // and the redundant direction carries almost no variance
    func testNearCollinearColumn() {
        var features = [[Double]]()
        features.reserveCapacity(200)
        for i in 0..<200 {
            let x = Double(i)
            let column1 = 0.05 * x
            let column2 = Foundation.sin(x)
            let column3 = Double(i % 7)
            let column4 = column1 + 1e-6 * Foundation.sin(1.7 * x)
            features.append([column1, column2, column3, column4])
        }

        let pca = PCA.fit(features: features, componentCount: 4)

        XCTAssertFalse(pca.explainedVariances.contains { $0.isNaN })
        XCTAssertFalse(pca.components.contains { row in row.contains { $0.isNaN } })
        XCTAssertEqual(pca.explainedVarianceRatios.sum(), 1.0, accuracy: 1e-12)

        // The collinear direction is noise-scale variance
        XCTAssertLessThan(pca.explainedVarianceRatios[3], 1e-10)
        XCTAssertGreaterThanOrEqual(pca.explainedVariances[3], 0.0)
    }

    // A constant column contributes exactly zero variance, never NaN
    func testConstantColumn() {
        let features: [[Double]] = [
            [5.0, 10.0],
            [5.0, 20.0],
            [5.0, 30.0],
            [5.0, 40.0]
        ]

        let pca = PCA.fit(features: features, componentCount: 2)

        XCTAssertFalse(pca.explainedVariances.contains { $0.isNaN })
        XCTAssertEqual(pca.explainedVariances[1], 0.0)
        XCTAssertEqual(pca.explainedVarianceRatios[0], 1.0, accuracy: 1e-12)
        XCTAssertEqual(pca.explainedVarianceRatios[1], 0.0, accuracy: 1e-12)
    }

    // All-constant data produces zero ratios rather than NaN
    func testAllConstantData() {
        let features: [[Double]] = [
            [3.0, 7.0],
            [3.0, 7.0],
            [3.0, 7.0]
        ]

        let pca = PCA.fit(features: features, componentCount: 2)

        XCTAssertEqual(pca.explainedVariances, [0.0, 0.0])
        XCTAssertEqual(pca.explainedVarianceRatios, [0.0, 0.0])
    }

    // Full-rank round-trip reproduces the input; reduced rank does not
    func testInverseTransformRoundTrip() {
        let features: [[Double]] = [
            [2.5, 24.0, 0.31],
            [0.5, 11.0, 0.20],
            [2.2, 22.0, 0.28],
            [1.9, 20.0, 0.25],
            [3.1, 30.0, 0.36]
        ]

        // Full rank: exact round-trip
        let fullRank = PCA.fit(features: features, componentCount: 3)
        let roundTrip = fullRank.inverseTransform(fullRank.transform(features))
        for row in 0..<5 {
            for f in 0..<3 {
                XCTAssertEqual(roundTrip[row][f], features[row][f], accuracy: 1e-9)
            }
        }

        // Reduced rank: reconstruction is an approximation, not the input
        let reduced = PCA.fit(features: features, componentCount: 1)
        let approximation = reduced.inverseTransform(reduced.transform(features))
        XCTAssertEqual(approximation.count, 5)
        XCTAssertEqual(approximation[0].count, 3)
    }

    // Empty input to transform and inverseTransform returns empty output
    func testEmptyTransformInput() {
        let pca = PCA.fit(features: [[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], componentCount: 2)
        XCTAssertTrue(pca.transform([]).isEmpty)
        XCTAssertTrue(pca.inverseTransform([]).isEmpty)
    }

    // Identical input produces a bit-identical model across reruns
    func testBitwiseDeterminism() {
        let first = PCA.fit(features: iris, componentCount: 3)
        let second = PCA.fit(features: iris, componentCount: 3)
        XCTAssertEqual(first, second)
    }

    // MARK: - Description

    // The printed form names the shape and total variance retained
    func testDescription() {
        let pca = PCA.fit(features: iris, componentCount: 2)
        XCTAssertEqual(pca.description, "PCA: 2 components, 4 features (97.8% variance explained)")

        let single = PCA.fit(features: [[1.0], [2.0], [3.0]], componentCount: 1)
        XCTAssertEqual(single.description, "PCA: 1 component, 1 feature (100.0% variance explained)")
    }

    // MARK: - Validating decoder

    // Hostile JSON whose shapes disagree is rejected, never decoded into
    // an inconsistent model
    func testHostileDecodeRejected() throws {
        let pca = PCA.fit(features: iris, componentCount: 2)
        let encoded = try JSONEncoder().encode(pca)
        var payload = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: encoded) as? [String: Any])

        // componentCount claims 7 while components holds 2 rows
        var corrupted = payload
        corrupted["componentCount"] = 7
        try assertDecodeFails(corrupted, "componentCount inflated")

        // Ragged component row
        corrupted = payload
        corrupted["components"] = [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3]]
        try assertDecodeFails(corrupted, "ragged component row")

        // means length disagrees with featureCount
        corrupted = payload
        corrupted["means"] = [1.0, 2.0]
        try assertDecodeFails(corrupted, "short means")

        // Ratio array length disagrees with componentCount
        corrupted = payload
        corrupted["explainedVarianceRatios"] = [0.9]
        try assertDecodeFails(corrupted, "short ratios")

        // componentCount exceeding featureCount is inconsistent even when
        // the arrays are internally sized to match
        corrupted = payload
        corrupted["componentCount"] = 0
        corrupted["components"] = [[Double]]()
        corrupted["explainedVariances"] = [Double]()
        corrupted["explainedVarianceRatios"] = [Double]()
        try assertDecodeFails(corrupted, "zero componentCount")

        // The untouched payload still decodes
        payload["featureCount"] = 4
        let data = try JSONSerialization.data(withJSONObject: payload)
        XCTAssertNoThrow(try JSONDecoder().decode(PCA.self, from: data))
    }

    // Decoding the corrupted payload throws DecodingError
    private func assertDecodeFails(_ payload: [String: Any], _ label: String) throws {
        let data = try JSONSerialization.data(withJSONObject: payload)
        XCTAssertThrowsError(try JSONDecoder().decode(PCA.self, from: data), label) { error in
            XCTAssertTrue(error is DecodingError, "\(label): expected DecodingError, got \(error)")
        }
    }

    // MARK: - Equatable

    // Same training data produces equal models
    func testPCAEquatable() {
        let first = PCA.fit(features: iris, componentCount: 2)
        let second = PCA.fit(features: iris, componentCount: 2)
        XCTAssertEqual(first, second)

        let different = PCA.fit(features: iris, componentCount: 3)
        XCTAssertNotEqual(first, different)
    }

    // MARK: - Codable

    // Round-trip preserves equality and transformation output
    func testPCACodable() throws {
        let pca = PCA.fit(features: iris, componentCount: 2)

        let data = try JSONEncoder().encode(pca)
        let decoded = try JSONDecoder().decode(PCA.self, from: data)
        XCTAssertEqual(pca, decoded)

        let testInput: [[Double]] = [[5.1, 3.5, 1.4, 0.2]]
        XCTAssertEqual(pca.transform(testInput), decoded.transform(testInput))
    }
}
