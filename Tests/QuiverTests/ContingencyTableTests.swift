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
import Foundation
@testable import Quiver

final class ContingencyTableTests: XCTestCase {

    /// Penn ling052 faculty example. 1 woman + 37 men in Math; 17 women +
    /// 20 men in English. Used across most tests below.
    private func facultyTable() -> ContingencyTable {
        return ContingencyTable(
            rowLabels: ["Female", "Male"],
            columnLabels: ["Math", "English"],
            counts: [[1, 17], [37, 20]]
        )!
    }

    // MARK: - Validation

    func testInitRejectsMismatchedRowLabelCount() {
        let table = ContingencyTable(
            rowLabels: ["A"],
            columnLabels: ["X", "Y"],
            counts: [[1, 2], [3, 4]]
        )
        XCTAssertNil(table)
    }

    func testInitRejectsMismatchedColumnLabelCount() {
        let table = ContingencyTable(
            rowLabels: ["A", "B"],
            columnLabels: ["X"],
            counts: [[1, 2], [3, 4]]
        )
        XCTAssertNil(table)
    }

    func testInitRejectsRaggedRows() {
        let table = ContingencyTable(
            rowLabels: ["A", "B"],
            columnLabels: ["X", "Y"],
            counts: [[1, 2], [3]]
        )
        XCTAssertNil(table)
    }

    func testInitRejectsNegativeCounts() {
        let table = ContingencyTable(
            rowLabels: ["A", "B"],
            columnLabels: ["X", "Y"],
            counts: [[1, -1], [3, 4]]
        )
        XCTAssertNil(table)
    }

    func testInitRejectsEmptyTable() {
        XCTAssertNil(ContingencyTable(rowLabels: [], columnLabels: ["X"], counts: []))
        XCTAssertNil(ContingencyTable(rowLabels: ["A"], columnLabels: [], counts: [[]]))
    }

    // MARK: - Marginals

    func testRowTotalsMatchPennExample() {
        let table = facultyTable()
        XCTAssertEqual(table.rowTotals, [18, 57])
    }

    func testColumnTotalsMatchPennExample() {
        let table = facultyTable()
        XCTAssertEqual(table.columnTotals, [38, 37])
    }

    func testGrandTotalMatchesPennExample() {
        XCTAssertEqual(facultyTable().grandTotal, 75)
    }

    // MARK: - Joint distribution

    func testJointDistributionMatchesPennTable2() {
        // Penn Table 2: P(female, math)=.013, P(female, english)=.227,
        // P(male, math)=.493, P(male, english)=.267
        let joint = facultyTable().jointDistribution()
        XCTAssertEqual(joint[0][0], 1.0 / 75.0, accuracy: 1e-12)
        XCTAssertEqual(joint[0][1], 17.0 / 75.0, accuracy: 1e-12)
        XCTAssertEqual(joint[1][0], 37.0 / 75.0, accuracy: 1e-12)
        XCTAssertEqual(joint[1][1], 20.0 / 75.0, accuracy: 1e-12)
    }

    func testJointDistributionSumsToOne() {
        let total = facultyTable().jointDistribution().flatMap { $0 }.reduce(0.0, +)
        XCTAssertEqual(total, 1.0, accuracy: 1e-12)
    }

    func testJointDistributionWithAllZeroCountsReturnsZeros() {
        let table = ContingencyTable(
            rowLabels: ["A", "B"],
            columnLabels: ["X", "Y"],
            counts: [[0, 0], [0, 0]]
        )!
        let joint = table.jointDistribution()
        for row in joint {
            for cell in row {
                XCTAssertEqual(cell, 0.0)
            }
        }
    }

    // MARK: - Conditional distributions

    func testConditionalByColumnMatchesPennMaleGivenMath() {
        // Penn's worked example: P(male | math) = 37/38 ≈ 0.974
        let conditional = facultyTable().conditionalByColumn()
        XCTAssertEqual(conditional[1][0], 37.0 / 38.0, accuracy: 1e-12)
        XCTAssertEqual(conditional[0][0], 1.0 / 38.0, accuracy: 1e-12)
    }

    func testConditionalByColumnSumsToOnePerColumn() {
        let conditional = facultyTable().conditionalByColumn()
        for c in 0..<2 {
            let total = conditional.map { $0[c] }.reduce(0.0, +)
            XCTAssertEqual(total, 1.0, accuracy: 1e-12)
        }
    }

    func testConditionalByRowComputesMathGivenMale() {
        // P(math | male) = 37/57
        let conditional = facultyTable().conditionalByRow()
        XCTAssertEqual(conditional[1][0], 37.0 / 57.0, accuracy: 1e-12)
        XCTAssertEqual(conditional[1][1], 20.0 / 57.0, accuracy: 1e-12)
    }

    func testConditionalByRowSumsToOnePerRow() {
        let conditional = facultyTable().conditionalByRow()
        for row in conditional {
            XCTAssertEqual(row.reduce(0.0, +), 1.0, accuracy: 1e-12)
        }
    }

    // MARK: - Printed summaries

    func testMarkdownTableShowsCountsAndMarginals() {
        let table = facultyTable().markdownTable()
        XCTAssertTrue(table.contains("| | Math | English | Total |"))
        XCTAssertTrue(table.contains("| --- | ---: | ---: | ---: |"))
        XCTAssertTrue(table.contains("| Female | 1 | 17 | 18 |"))
        XCTAssertTrue(table.contains("| Male | 37 | 20 | 57 |"))
        XCTAssertTrue(table.contains("| Total | 38 | 37 | 75 |"))
    }

    func testProbabilityTableMatchesPennTable2() {
        let table = facultyTable().probabilityTable()
        XCTAssertTrue(table.contains("| | Math | English | Total |"))
        XCTAssertTrue(table.contains("| --- | ---: | ---: | ---: |"))
        // 1/75 = 0.013333..., rounds to 0.0133
        XCTAssertTrue(table.contains("| Female | 0.0133 | 0.2267 | 0.2400 |"))
        // 37/75 = 0.493333..., rounds to 0.4933
        XCTAssertTrue(table.contains("| Male | 0.4933 | 0.2667 | 0.7600 |"))
        XCTAssertTrue(table.contains("| Total | 0.5067 | 0.4933 | 1.0000 |"))
    }

    func testCSVRowEmitsRowMajorCounts() {
        XCTAssertEqual(facultyTable().csvRow(), "1,17,37,20")
    }

    func testDescriptionIncludesRowAndColumnLabels() {
        let description = facultyTable().description
        XCTAssertTrue(description.contains("Math"))
        XCTAssertTrue(description.contains("English"))
        XCTAssertTrue(description.contains("Female"))
        XCTAssertTrue(description.contains("Male"))
        XCTAssertTrue(description.contains("Total"))
        XCTAssertTrue(description.contains("75"))
    }

    // MARK: - Codable / Equatable

    func testCodableRoundTrip() throws {
        let original = facultyTable()
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ContingencyTable.self, from: data)
        XCTAssertEqual(original, decoded)
    }

    func testEquatable() {
        let a = facultyTable()
        let b = facultyTable()
        XCTAssertEqual(a, b)

        let different = ContingencyTable(
            rowLabels: ["Female", "Male"],
            columnLabels: ["Math", "English"],
            counts: [[2, 17], [37, 20]]
        )!
        XCTAssertNotEqual(a, different)
    }
}
