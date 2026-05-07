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
@testable import Quiver

final class PanelTests: XCTestCase {

    // Creating a panel and accessing columns returns the correct values
    func testInitAndColumnAccess() {
        let panel = Panel([
            ("age", [25.0, 30.0, 35.0]),
            ("score", [88.0, 92.0, 75.0])
        ])

        XCTAssertEqual(panel.rowCount, 3)
        XCTAssertEqual(panel.columnNames, ["age", "score"])
        XCTAssertEqual(panel["age"], [25.0, 30.0, 35.0])
        XCTAssertEqual(panel["score"], [88.0, 92.0, 75.0])
    }

    // shape returns the same (rows, columns) tuple format as matrix .shape
    func testShape() {
        let panel = Panel([
            ("a", [1.0, 2.0, 3.0]),
            ("b", [4.0, 5.0, 6.0]),
            ("c", [7.0, 8.0, 9.0])
        ])
        XCTAssertEqual(panel.shape.rows, 3)
        XCTAssertEqual(panel.shape.columns, 3)
    }

    // Extracting a matrix respects column selection and ordering
    func testToMatrix() {
        let panel = Panel([
            ("a", [1.0, 4.0]),
            ("b", [2.0, 5.0]),
            ("c", [3.0, 6.0])
        ])

        // Default order matches insertion order
        let all = panel.toMatrix()
        XCTAssertEqual(all, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        // Custom column selection and reordering
        let reversed = panel.toMatrix(columns: ["c", "a"])
        XCTAssertEqual(reversed, [[3.0, 1.0], [6.0, 4.0]])
    }

    // Train-test split preserves all rows and produces correct partition sizes
    func testTrainTestSplit() {
        let panel = Panel([
            ("x", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            ("y", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        ])

        let (train, test) = panel.trainTestSplit(testRatio: 0.2, seed: 42)

        XCTAssertEqual(train.rowCount + test.rowCount, 10)
        XCTAssertEqual(test.rowCount, 2)
        XCTAssertEqual(train.rowCount, 8)
        XCTAssertEqual(train.columnNames, ["x", "y"])
        XCTAssertEqual(test.columnNames, ["x", "y"])
    }

    // Boolean mask filtering returns only matching rows across all columns
    func testFilteredWithBooleanMask() {
        let panel = Panel([
            ("value", [10.0, 20.0, 30.0, 40.0]),
            ("label", [0.0, 1.0, 0.0, 1.0])
        ])

        let mask = panel["value"].isGreaterThan(15.0)
        let filtered = panel.filtered(where: mask)

        XCTAssertEqual(filtered.rowCount, 3)
        XCTAssertEqual(filtered["value"], [20.0, 30.0, 40.0])
        XCTAssertEqual(filtered["label"], [1.0, 0.0, 1.0])
    }

    // head() displays the first rows in a Pandas-style tabular format
    func testHeadDefault() {
        let panel = Panel([
            ("age", [25.0, 30.0, 35.0]),
            ("score", [88.0, 92.0, 85.0])
        ])

        let output = panel.head()
        let lines = output.split(separator: "\n")

        // Header plus 3 data rows
        XCTAssertEqual(lines.count, 4)

        // Header contains column names
        XCTAssertTrue(output.contains("age"))
        XCTAssertTrue(output.contains("score"))

        // Row indices appear
        XCTAssertTrue(lines[1].hasPrefix("0"))
        XCTAssertTrue(lines[2].hasPrefix("1"))
        XCTAssertTrue(lines[3].hasPrefix("2"))

        // Values appear
        XCTAssertTrue(output.contains("25.0"))
        XCTAssertTrue(output.contains("92.0"))
    }

    // head(n) limits to the specified number of rows
    func testHeadWithLimit() {
        let panel = Panel([
            ("x", [1.0, 2.0, 3.0, 4.0, 5.0])
        ])

        let output = panel.head(n: 2)
        let lines = output.split(separator: "\n")

        // Header plus 2 data rows
        XCTAssertEqual(lines.count, 3)
        XCTAssertTrue(output.contains("1.0"))
        XCTAssertTrue(output.contains("2.0"))
        XCTAssertFalse(output.contains("3.0"))
    }

    // head() right-aligns values within each column
    func testHeadAlignment() {
        let panel = Panel([
            ("value", [1.0, 1000.0])
        ])

        let output = panel.head()
        let lines = output.split(separator: "\n")

        // The "1.0" row should be padded so it aligns with "1000.0"
        let row1 = String(lines[1])
        let row2 = String(lines[2])

        // Both value strings should end at the same column position
        XCTAssertEqual(row1.count, row2.count)
    }

    // Same columns and data produce equal Panels
    func testPanelEquatable() {
        let panel1 = Panel([("age", [25.0, 30.0]), ("score", [88.0, 92.0])])
        let panel2 = Panel([("age", [25.0, 30.0]), ("score", [88.0, 92.0])])
        XCTAssertEqual(panel1, panel2)

        let different = Panel([("age", [25.0, 35.0]), ("score", [88.0, 92.0])])
        XCTAssertNotEqual(panel1, different)
    }

    // Describe produces per-column statistics with column names
    func testDescribeOutput() {
        let panel = Panel([
            ("x", [1.0, 2.0, 3.0, 4.0, 5.0])
        ])

        let output = panel.summary().description

        XCTAssertTrue(output.contains("x"), "Output should contain the column name")
        XCTAssertTrue(output.contains("mean"), "Output should contain a mean header")
        XCTAssertTrue(output.contains("3.0"), "Output should contain the mean value")
        XCTAssertTrue(output.contains("1.0"), "Output should contain the min value")
        XCTAssertTrue(output.contains("5.0"), "Output should contain the max value")
    }

    // tail returns the last n rows in tabular form
    func testTailReturnsLastRows() {
        let panel = Panel([
            ("day", [1.0, 2.0, 3.0, 4.0, 5.0]),
            ("revenue", [120.0, 135.0, 142.0, 128.0, 145.0])
        ])

        let output = panel.tail(n: 2)
        XCTAssertTrue(output.contains("4.0"), "Tail should contain row index 4's value")
        XCTAssertTrue(output.contains("145.0"), "Tail should contain the last revenue value")
        XCTAssertFalse(output.contains("120.0"), "Tail of 2 should not contain the first row")
    }

    // tail with n larger than rowCount returns all rows
    func testTailRequestingMoreRowsThanAvailable() {
        let panel = Panel([
            ("x", [1.0, 2.0])
        ])

        let output = panel.tail(n: 10)
        XCTAssertTrue(output.contains("1.0"))
        XCTAssertTrue(output.contains("2.0"))
    }

    // unique returns sorted distinct values
    func testUniqueReturnsSortedDistinct() {
        let panel = Panel([
            ("species", [0.0, 1.0, 0.0, 2.0, 1.0, 0.0])
        ])

        XCTAssertEqual(panel.unique(column: "species"), [0.0, 1.0, 2.0])
    }

    // unique returns nil for missing columns
    func testUniqueMissingColumnReturnsNil() {
        let panel = Panel([("species", [0.0, 1.0])])
        XCTAssertNil(panel.unique(column: "nonexistent"))
    }

    // valueCounts returns counts sorted by count descending
    func testValueCountsSortedByCount() {
        let panel = Panel([
            ("species", [0.0, 1.0, 0.0, 2.0, 1.0, 0.0])
        ])

        guard let counts = panel.valueCounts(column: "species") else {
            XCTFail("Expected non-nil result for existing column")
            return
        }

        XCTAssertEqual(counts.count, 3)
        XCTAssertEqual(counts[0].value, 0.0)
        XCTAssertEqual(counts[0].count, 3)
        XCTAssertEqual(counts[1].value, 1.0)
        XCTAssertEqual(counts[1].count, 2)
        XCTAssertEqual(counts[2].value, 2.0)
        XCTAssertEqual(counts[2].count, 1)
    }

    // valueCounts breaks ties by value ascending
    func testValueCountsBreaksTiesAscending() {
        let panel = Panel([
            ("x", [3.0, 1.0, 2.0])  // each appears once
        ])

        guard let counts = panel.valueCounts(column: "x") else {
            XCTFail("Expected non-nil result")
            return
        }

        XCTAssertEqual(counts.map { $0.value }, [1.0, 2.0, 3.0])
    }

    // valueCounts returns nil for missing columns
    func testValueCountsMissingColumnReturnsNil() {
        let panel = Panel([("x", [1.0])])
        XCTAssertNil(panel.valueCounts(column: "nonexistent"))
    }

    // sortedBy reorders all columns ascending by the specified column
    func testSortedByAscending() {
        let panel = Panel([
            ("id", [3.0, 1.0, 2.0]),
            ("score", [88.0, 95.0, 72.0])
        ])

        let sorted = panel.sortedBy(column: "score", ascending: true)

        XCTAssertEqual(sorted["score"], [72.0, 88.0, 95.0])
        // The id column moves with the score column.
        XCTAssertEqual(sorted["id"], [2.0, 3.0, 1.0])
    }

    // sortedBy descending reverses the ordering
    func testSortedByDescending() {
        let panel = Panel([
            ("id", [3.0, 1.0, 2.0]),
            ("score", [88.0, 95.0, 72.0])
        ])

        let sorted = panel.sortedBy(column: "score", ascending: false)

        XCTAssertEqual(sorted["score"], [95.0, 88.0, 72.0])
        XCTAssertEqual(sorted["id"], [1.0, 3.0, 2.0])
    }

    // sortedBy places NaN values at the end in both directions
    func testSortedByPlacesNaNAtEnd() {
        let panel = Panel([
            ("id", [1.0, 2.0, 3.0, 4.0]),
            ("score", [88.0, .nan, 72.0, 95.0])
        ])

        let asc = panel.sortedBy(column: "score", ascending: true)
        let descScores = panel.sortedBy(column: "score", ascending: false)["score"]
        let ascScores = asc["score"]

        XCTAssertEqual(ascScores[0], 72.0)
        XCTAssertEqual(ascScores[1], 88.0)
        XCTAssertEqual(ascScores[2], 95.0)
        XCTAssertTrue(ascScores[3].isNaN, "NaN should sort to the end ascending")

        XCTAssertEqual(descScores[0], 95.0)
        XCTAssertEqual(descScores[1], 88.0)
        XCTAssertEqual(descScores[2], 72.0)
        XCTAssertTrue(descScores[3].isNaN, "NaN should sort to the end descending too")
    }

    // standardized produces a column with mean ~0 and std ~1
    func testStandardizedProducesZeroMeanUnitStd() {
        let panel = Panel([
            ("age", [25.0, 30.0, 35.0, 40.0, 45.0]),
            ("score", [88.0, 92.0, 85.0, 91.0, 87.0])
        ])

        let zPanel = panel.standardized(column: "age")
        let zAge = zPanel["age"]

        XCTAssertEqual(zAge.mean() ?? .nan, 0.0, accuracy: 1e-10)
        XCTAssertEqual(zAge.std() ?? .nan, 1.0, accuracy: 1e-10)
    }

    // standardized leaves other columns unchanged
    func testStandardizedDoesNotMutateOtherColumns() {
        let panel = Panel([
            ("age", [25.0, 30.0, 35.0]),
            ("score", [88.0, 92.0, 85.0])
        ])

        let zPanel = panel.standardized(column: "age")
        XCTAssertEqual(zPanel["score"], [88.0, 92.0, 85.0])
    }

    // standardized on a constant column returns zeros for that column
    func testStandardizedConstantColumnReturnsZeros() {
        let panel = Panel([
            ("constant", [5.0, 5.0, 5.0])
        ])

        let zPanel = panel.standardized(column: "constant")
        XCTAssertEqual(zPanel["constant"], [0.0, 0.0, 0.0])
    }

    // correlationMatrix returns the labels in column order
    func testCorrelationMatrixReturnsOrderedLabels() {
        let panel = Panel([
            ("a", [1.0, 2.0, 3.0, 4.0, 5.0]),
            ("b", [2.0, 4.0, 6.0, 8.0, 10.0]),
            ("c", [5.0, 4.0, 3.0, 2.0, 1.0])
        ])

        let result = panel.correlationMatrix()
        XCTAssertEqual(result.columns, ["a", "b", "c"])
    }

    // correlationMatrix returns 1.0 on the diagonal
    func testCorrelationMatrixDiagonalIsOne() {
        let panel = Panel([
            ("a", [1.0, 2.0, 3.0, 4.0]),
            ("b", [4.0, 3.0, 2.0, 1.0])
        ])

        let result = panel.correlationMatrix()
        XCTAssertEqual(result.matrix[0][0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(result.matrix[1][1], 1.0, accuracy: 1e-10)
    }

    // correlationMatrix recovers known correlations
    func testCorrelationMatrixRecoversPerfectPositive() {
        let panel = Panel([
            ("x", [1.0, 2.0, 3.0, 4.0, 5.0]),
            ("y", [2.0, 4.0, 6.0, 8.0, 10.0])  // y = 2x
        ])

        let result = panel.correlationMatrix()
        XCTAssertEqual(result.matrix[0][1], 1.0, accuracy: 1e-10)
        XCTAssertEqual(result.matrix[1][0], 1.0, accuracy: 1e-10)
    }

    // correlationMatrix recovers perfect negative correlation
    func testCorrelationMatrixRecoversPerfectNegative() {
        let panel = Panel([
            ("x", [1.0, 2.0, 3.0, 4.0, 5.0]),
            ("y", [5.0, 4.0, 3.0, 2.0, 1.0])
        ])

        let result = panel.correlationMatrix()
        XCTAssertEqual(result.matrix[0][1], -1.0, accuracy: 1e-10)
    }

    // correlationMatrix returns NaN for off-diagonal entries involving a
    // constant column (zero variance has no defined correlation; matches
    // pandas df.corr() and np.corrcoef behavior). Returning 0.0 would
    // conflate "no linear relationship" with "undefined correlation" —
    // two different statements. Diagonals remain 1.0 by Quiver convention.
    func testCorrelationMatrixConstantColumnReturnsNaN() {
        let panel = Panel([
            ("varying",  [1.0, 2.0, 3.0, 4.0, 5.0]),
            ("constant", [3.0, 3.0, 3.0, 3.0, 3.0])
        ])

        let result = panel.correlationMatrix()

        // Off-diagonal entries involving the constant column are NaN —
        // the corrected behavior, was 0.0 before the May 6 validation pass
        XCTAssertTrue(result.matrix[0][1].isNaN, "Constant column should produce NaN correlation")
        XCTAssertTrue(result.matrix[1][0].isNaN, "Constant column should produce NaN correlation")

        // Diagonals stay at 1.0 by Quiver convention — the matrix
        // implementation short-circuits self-correlation rather than
        // computing it through pearsonCorrelation.
        XCTAssertEqual(result.matrix[0][0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(result.matrix[1][1], 1.0, accuracy: 1e-10)
    }

    // toPanel with the default name produces a "values" column with the array's data
    func testToPanelDefaultName() {
        let scores = [68.0, 72.0, 75.0, 77.0, 80.0, 82.0, 85.0, 88.0]
        let panel = scores.toPanel()

        XCTAssertEqual(panel.columnNames, ["values"])
        XCTAssertEqual(panel["values"], scores)
        XCTAssertEqual(panel.rowCount, 8)
    }

    // toPanel with a custom name honors the supplied label
    func testToPanelCustomName() {
        let scores = [68.0, 72.0, 75.0]
        let panel = scores.toPanel("scores")

        XCTAssertEqual(panel.columnNames, ["scores"])
        XCTAssertEqual(panel["scores"], scores)
    }

    // toPanel chains naturally into Panel-level descriptive operations
    func testToPanelChainsIntoSummary() {
        let scores = [68.0, 72.0, 75.0, 77.0, 80.0, 82.0, 85.0, 88.0]
        let summary = scores.toPanel("scores").summary()

        XCTAssertEqual(summary.columnNames, ["scores"])
        XCTAssertEqual(summary.columns["scores"]?.count, 8)
    }
}
