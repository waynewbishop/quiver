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

        let output = panel.describe()

        XCTAssertTrue(output.contains("x"), "Output should contain the column name")
        XCTAssertTrue(output.contains("mean"), "Output should contain a mean header")
        XCTAssertTrue(output.contains("3.0"), "Output should contain the mean value")
        XCTAssertTrue(output.contains("1.0"), "Output should contain the min value")
        XCTAssertTrue(output.contains("5.0"), "Output should contain the max value")
    }
}
