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

/// Tests for `Distributions.f` and `Distributions.chiSquared.quantile`.
///
/// Reference values were computed independently with SciPy (`f.pdf`, `f.cdf`, `f.ppf`,
/// `chi2.ppf`) and agree to better than 1e-9.
final class FDistributionTests: XCTestCase {

    // MARK: - F pdf

    func testPDFMatchesReference() throws {
        XCTAssertEqual(try XCTUnwrap(Distributions.f.pdf(x: 1.0, df1: 4, df2: 12)), 0.4671936035156267, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(Distributions.f.pdf(x: 0.5, df1: 2, df2: 2)), 0.4444444444444444, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(Distributions.f.pdf(x: 2.0, df1: 6, df2: 9)), 0.16573677685389684, accuracy: 1e-12)
    }

    func testPDFClosedFormAtTwoAndTwo() throws {
        // With df1 = df2 = 2 the density is 1 / (1 + x)².
        for x in [0.1, 0.5, 1.0, 3.0, 10.0] {
            let expected = 1.0 / ((1.0 + x) * (1.0 + x))
            XCTAssertEqual(try XCTUnwrap(Distributions.f.pdf(x: x, df1: 2, df2: 2)), expected, accuracy: 1e-13)
        }
    }

    func testPDFAtAndBelowZero() {
        XCTAssertEqual(Distributions.f.pdf(x: -1.0, df1: 4, df2: 6), 0.0)
        XCTAssertEqual(Distributions.f.pdf(x: 0.0, df1: 2, df2: 6), 1.0)
        XCTAssertEqual(Distributions.f.pdf(x: 0.0, df1: 5, df2: 6), 0.0)
        XCTAssertNil(Distributions.f.pdf(x: 0.0, df1: 1, df2: 6), "unbounded at zero when df1 < 2")
    }

    func testPDFIntegratesToOne() throws {
        // Trapezoid rule over a wide range captures essentially all of the mass for df2 = 12.
        var area = 0.0
        let step = 0.001
        var x = step
        var previous = try XCTUnwrap(Distributions.f.pdf(x: step, df1: 4, df2: 12))
        while x < 200.0 {
            x += step
            let current = try XCTUnwrap(Distributions.f.pdf(x: x, df1: 4, df2: 12))
            area += 0.5 * (previous + current) * step
            previous = current
        }
        XCTAssertEqual(area, 1.0, accuracy: 1e-3)
    }

    // MARK: - F cdf

    func testCDFMatchesReference() throws {
        XCTAssertEqual(try XCTUnwrap(Distributions.f.cdf(x: 2.5, df1: 4, df2: 12)), 0.9018381888268955, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(Distributions.f.cdf(x: 1.0, df1: 5, df2: 5)), 0.5, accuracy: 1e-12)
    }

    func testCDFBoundsAndMonotonicity() throws {
        XCTAssertEqual(Distributions.f.cdf(x: 0.0, df1: 3, df2: 7), 0.0)
        XCTAssertEqual(Distributions.f.cdf(x: -2.0, df1: 3, df2: 7), 0.0)
        XCTAssertEqual(Distributions.f.cdf(x: .infinity, df1: 3, df2: 7), 1.0)

        var previous = 0.0
        for x in stride(from: 0.1, through: 20.0, by: 0.1) {
            let value = try XCTUnwrap(Distributions.f.cdf(x: x, df1: 3, df2: 7))
            XCTAssertGreaterThanOrEqual(value, previous)
            XCTAssertLessThanOrEqual(value, 1.0)
            previous = value
        }
    }

    func testCDFReciprocalSymmetry() throws {
        // If F has (d1, d2) degrees of freedom then 1/F has (d2, d1), so P(F <= x) = 1 - P(F' <= 1/x).
        for x in [0.3, 0.8, 1.7, 4.0] {
            let direct = try XCTUnwrap(Distributions.f.cdf(x: x, df1: 3, df2: 9))
            let flipped = try XCTUnwrap(Distributions.f.cdf(x: 1.0 / x, df1: 9, df2: 3))
            XCTAssertEqual(direct, 1.0 - flipped, accuracy: 1e-12)
        }
    }

    func testCDFIsTheIntegralOfThePDF() throws {
        // Midpoint integration of the density from 0 to 2.5 reproduces the CDF for df1 = 4.
        var area = 0.0
        let steps = 5000
        let width = 2.5 / Double(steps)
        for index in 0..<steps {
            let midpoint = (Double(index) + 0.5) * width
            area += try XCTUnwrap(Distributions.f.pdf(x: midpoint, df1: 4, df2: 12)) * width
        }
        let cdf = try XCTUnwrap(Distributions.f.cdf(x: 2.5, df1: 4, df2: 12))
        XCTAssertEqual(area, cdf, accuracy: 1e-6)
    }

    // MARK: - F quantile

    func testQuantileMatchesReference() throws {
        XCTAssertEqual(try XCTUnwrap(Distributions.f.quantile(p: 0.95, df1: 3, df2: 20)), 3.0983912121407813, accuracy: 1e-8)
        XCTAssertEqual(try XCTUnwrap(Distributions.f.quantile(p: 0.99, df1: 2, df2: 10)), 7.559432157547899, accuracy: 1e-8)
        XCTAssertEqual(try XCTUnwrap(Distributions.f.quantile(p: 0.95, df1: 2, df2: 18)), 3.554557145661787, accuracy: 1e-8)
        XCTAssertEqual(try XCTUnwrap(Distributions.f.quantile(p: 0.5, df1: 1, df2: 1)), 1.0, accuracy: 1e-8)
    }

    func testQuantileInvertsCDF() throws {
        for p in [0.01, 0.1, 0.5, 0.9, 0.975, 0.999] {
            let x = try XCTUnwrap(Distributions.f.quantile(p: p, df1: 5, df2: 11))
            let back = try XCTUnwrap(Distributions.f.cdf(x: x, df1: 5, df2: 11))
            XCTAssertEqual(back, p, accuracy: 1e-9)
        }
    }

    // MARK: - F domain

    func testReturnsNilOutsideTheDomain() {
        XCTAssertNil(Distributions.f.pdf(x: 1.0, df1: 0, df2: 5))
        XCTAssertNil(Distributions.f.pdf(x: 1.0, df1: 5, df2: -1))
        XCTAssertNil(Distributions.f.cdf(x: 1.0, df1: 0, df2: 5))
        XCTAssertNil(Distributions.f.cdf(x: 1.0, df1: 5, df2: 0))
        XCTAssertNil(Distributions.f.quantile(p: 0.5, df1: -1, df2: 5))
        XCTAssertNil(Distributions.f.quantile(p: 0.5, df1: 5, df2: 0))
        XCTAssertNil(Distributions.f.quantile(p: 0.0, df1: 3, df2: 5))
        XCTAssertNil(Distributions.f.quantile(p: 1.0, df1: 3, df2: 5))
        XCTAssertNil(Distributions.f.quantile(p: 1.2, df1: 3, df2: 5))
    }

    // MARK: - Chi-squared quantile

    func testChiSquaredQuantileMatchesReference() throws {
        XCTAssertEqual(try XCTUnwrap(Distributions.chiSquared.quantile(p: 0.95, df: 5)), 11.070497693516351, accuracy: 1e-8)
        XCTAssertEqual(try XCTUnwrap(Distributions.chiSquared.quantile(p: 0.95, df: 1)), 3.8414588206941205, accuracy: 1e-8)
        XCTAssertEqual(try XCTUnwrap(Distributions.chiSquared.quantile(p: 0.01, df: 10)), 2.5582121601872063, accuracy: 1e-8)
    }

    func testChiSquaredQuantileClosedFormAtTwoDegrees() throws {
        // At df = 2 the CDF is 1 - exp(-x / 2), so the quantile is -2 ln(1 - p).
        for p in [0.1, 0.5, 0.9, 0.99] {
            let expected = -2.0 * Foundation.log(1.0 - p)
            XCTAssertEqual(try XCTUnwrap(Distributions.chiSquared.quantile(p: p, df: 2)), expected, accuracy: 1e-9)
        }
        XCTAssertEqual(try XCTUnwrap(Distributions.chiSquared.quantile(p: 0.5, df: 2)), 1.386294361119891, accuracy: 1e-9)
    }

    func testChiSquaredQuantileInvertsCDF() throws {
        for df in [1.0, 3.0, 10.0, 40.0, 120.0] {
            for p in [0.01, 0.5, 0.95, 0.999] {
                let x = try XCTUnwrap(Distributions.chiSquared.quantile(p: p, df: df))
                let back = try XCTUnwrap(Distributions.chiSquared.cdf(x: x, df: df))
                XCTAssertEqual(back, p, accuracy: 1e-9)
            }
        }
    }

    func testChiSquaredQuantileReturnsNilOutsideTheDomain() {
        XCTAssertNil(Distributions.chiSquared.quantile(p: 0.5, df: 0))
        XCTAssertNil(Distributions.chiSquared.quantile(p: 0.5, df: -3))
        XCTAssertNil(Distributions.chiSquared.quantile(p: 0.0, df: 5))
        XCTAssertNil(Distributions.chiSquared.quantile(p: 1.0, df: 5))
        XCTAssertNil(Distributions.chiSquared.quantile(p: -0.1, df: 5))
    }
}
