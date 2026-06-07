// Copyright 2025 Wayne W Bishop. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import XCTest
@testable import Quiver

/// Tests `asExpression()` across `[Double]`, `[[Double]]`, `[Fraction]`,
/// `[[Fraction]]`, plus the shared `UnicodeMath.formatCell` formatter that
/// every type routes through.
final class ExpressionRenderingTests: XCTestCase {

    // MARK: - UnicodeMath.formatCell

    // Integer-valued doubles render without a decimal point.
    func testFormatCellIntegers() {
        XCTAssertEqual(UnicodeMath.formatCell(3.0), "3")
        XCTAssertEqual(UnicodeMath.formatCell(-7.0), "-7")
        XCTAssertEqual(UnicodeMath.formatCell(0.0), "0")
    }

    // Negative zero is normalized to a leading-sign-free "0".
    func testFormatCellNegativeZero() {
        XCTAssertEqual(UnicodeMath.formatCell(-0.0), "0")
    }

    // NaN and ±∞ render with Unicode-appropriate strings.
    func testFormatCellNonFinite() {
        XCTAssertEqual(UnicodeMath.formatCell(.nan), "NaN")
        XCTAssertEqual(UnicodeMath.formatCell(.infinity), "∞")
        XCTAssertEqual(UnicodeMath.formatCell(-.infinity), "-∞")
    }

    // Sub-1e-3 magnitudes switch to %g so distinct small values stay distinct.
    func testFormatCellSmallMagnitudes() {
        let a = UnicodeMath.formatCell(3e-5)
        let b = UnicodeMath.formatCell(7e-5)
        XCTAssertNotEqual(a, b, "small magnitudes must not collapse to the same cell")
        XCTAssertNotEqual(a, "0", "1e-5 must not render as zero")
    }

    // Values close to but not equal to an integer keep their decimal places
    // rather than rounding into a misleading integer string.
    func testFormatCellNearIntegerNoLie() {
        let s = UnicodeMath.formatCell(0.99999)
        XCTAssertTrue(s.contains("."), "0.99999 must not render as a bare integer; got \(s)")
    }

    // MARK: - [Double].asExpression()

    // The default column form wraps the right number of cells in the right
    // bracket characters.
    func testVectorColumnForm() {
        let v: [Double] = [3, 4, 5]
        let s = v.asExpression()
        XCTAssertEqual(s, "⎡ 3 ⎤\n⎢ 4 ⎥\n⎣ 5 ⎦")
    }

    // The inline form is the NumPy-friendly angle-bracket variant.
    func testVectorInlineForm() {
        let v: [Double] = [3, 4, 5]
        XCTAssertEqual(v.asExpression(form: .inline), "⟨3, 4, 5⟩")
    }

    // Single-element vectors render as the scalar in either form.
    func testVectorSingleElement() {
        XCTAssertEqual([5.0].asExpression(), "5")
        XCTAssertEqual([5.0].asExpression(form: .inline), "5")
    }

    // Empty vectors render as ⟨⟩ in either form.
    func testVectorEmpty() {
        let empty: [Double] = []
        XCTAssertEqual(empty.asExpression(), "⟨⟩")
        XCTAssertEqual(empty.asExpression(form: .inline), "⟨⟩")
    }

    // Column-form negatives align under the positive cells' digits.
    func testVectorColumnAlignment() {
        let v: [Double] = [1, -10, 5]
        let s = v.asExpression()
        // The widest cell is "-10" (3 chars). Every row's content slot is
        // 3 chars wide, right-aligned.
        XCTAssertEqual(s, "⎡   1 ⎤\n⎢ -10 ⎥\n⎣   5 ⎦")
    }

    // MARK: - [[Double]].asExpression()

    // A 2×2 matrix renders with per-column right-alignment so the negative
    // sign in column 0 doesn't push column 1 off-line.
    func testMatrix2x2() {
        let A: [[Double]] = [[3, 1], [-2, 5]]
        let s = A.asExpression()
        XCTAssertEqual(s, "⎡  3  1 ⎤\n⎣ -2  5 ⎦")
    }

    // A single-row matrix collapses to inline row form.
    func testMatrixSingleRow() {
        let A: [[Double]] = [[1, 2, 3]]
        XCTAssertEqual(A.asExpression(), "[ 1  2  3 ]")
    }

    // An empty matrix renders as ⟨⟩.
    func testMatrixEmpty() {
        let empty: [[Double]] = []
        XCTAssertEqual(empty.asExpression(), "⟨⟩")
    }

    // MARK: - [Fraction].asExpression() and [[Fraction]].asExpression()

    // The closure-free composition `asFractions().asExpression()` produces
    // a column of rendered fractions for a vector.
    func testVectorFractionsComposition() {
        let v = [0.6, 0.75, 0.5]
        let s = v.asFractions().asExpression()
        XCTAssertEqual(s, "⎡ 3/5 ⎤\n⎢ 3/4 ⎥\n⎣ 1/2 ⎦")
    }

    // The same composition for a matrix produces a bracketed grid of
    // fractions.
    func testMatrixFractionsComposition() {
        let A: [[Double]] = [[0.5, 0.25], [0.75, 0.125]]
        let s = A.asFractions().asExpression()
        XCTAssertEqual(s, "⎡ 1/2  1/4 ⎤\n⎣ 3/4  1/8 ⎦")
    }
}
