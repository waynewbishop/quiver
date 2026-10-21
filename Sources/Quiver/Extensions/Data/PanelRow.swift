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

import Foundation

// MARK: - Panel.Row

extension Panel {

    /// One row of a panel, read by column name.
    ///
    /// A `Panel` is organized by column, which is the right shape for statistics
    /// and model input. Some work happens one record at a time instead: feeding
    /// sensor samples to a scoring model, finding the moment a signal crosses a
    /// threshold, or filling a gap from the previous reading. Iterating a panel
    /// yields one `Row` per record, in row order:
    ///
    /// ```swift
    /// import Quiver
    ///
    /// let run = Panel([
    ///     ("heartRate", [142.0, 151.0, 158.0]),
    ///     ("pace", [5.6, 5.2, 4.9])
    /// ])
    ///
    /// for moment in run {
    ///     print(moment["heartRate"], moment["pace"])
    /// }
    /// // 142.0 5.6
    /// // 151.0 5.2
    /// // 158.0 4.9
    /// ```
    ///
    /// The subscript returns a `Double`, not an optional, matching the panel's own
    /// column subscript: reading a column the panel doesn't have stops with a
    /// message naming the column. A row is a read-only snapshot. To change data,
    /// build a new panel from its columns.
    public struct Row: CustomStringConvertible, Equatable, Sendable {

        /// Column names in the panel's declaration order.
        public let columnNames: [String]

        /// Values in the same order as `columnNames`.
        public let values: [Double]

        /// Position of each column in `values`, shared by every row of one iteration.
        private let positions: [String: Int]

        /// Creates a row from parallel names and values, with a prebuilt position lookup.
        internal init(columnNames: [String], values: [Double], positions: [String: Int]) {
            self.columnNames = columnNames
            self.values = values
            self.positions = positions
        }

        /// Returns the value in the named column.
        ///
        /// - Parameter column: The column name to look up.
        /// - Returns: The row's value in that column.
        public subscript(column: String) -> Double {
            guard let position = positions[column] else {
                preconditionFailure("Column '\(column)' does not exist in row")
            }
            return values[position]
        }

        /// Returns the row as name and value pairs in column order, such as
        /// `Row: heartRate 142.0, pace 5.6`.
        public var description: String {
            var pairs: [String] = []
            for (name, value) in zip(columnNames, values) {
                pairs.append("\(name) \(value)")
            }
            return "Row: " + pairs.joined(separator: ", ")
        }
    }
}

// MARK: - Sequence

extension Panel: Sequence {

    /// Steps through a panel one row at a time, from the first row to the last.
    public struct Iterator: IteratorProtocol, Sendable {

        private let columnNames: [String]
        private let columns: [[Double]]
        private let positions: [String: Int]
        private let rowCount: Int
        private var index = 0

        /// Creates an iterator over columns that were each read once from the panel.
        fileprivate init(columnNames: [String], columns: [[Double]], rowCount: Int) {
            var positions: [String: Int] = [:]
            for (position, name) in columnNames.enumerated() {
                positions[name] = position
            }
            self.columnNames = columnNames
            self.columns = columns
            self.positions = positions
            self.rowCount = rowCount
        }

        /// Returns the next row, or `nil` after the last one.
        public mutating func next() -> Row? {
            guard index < rowCount else { return nil }
            var values: [Double] = []
            values.reserveCapacity(columns.count)
            for column in columns {
                values.append(column[index])
            }
            index += 1
            return Row(columnNames: columnNames, values: values, positions: positions)
        }
    }

    /// Returns an iterator over the panel's rows in order.
    ///
    /// A panel with zero rows, such as the result of a filter that matched
    /// nothing, yields no rows.
    public func makeIterator() -> Iterator {
        var columns: [[Double]] = []
        columns.reserveCapacity(columnNames.count)
        for name in columnNames {
            columns.append(self[name])
        }
        return Iterator(columnNames: columnNames, columns: columns, rowCount: rowCount)
    }

    /// The number of rows, so collecting a panel's rows allocates once.
    public var underestimatedCount: Int {
        rowCount
    }
}
