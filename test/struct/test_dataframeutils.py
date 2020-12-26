# Copyright (C) 2020 Raven Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Tests for DataFrame utility functions.
"""

import unittest

from raven.struct.dataframe import (DataFrame,
                                    DefaultDataFrame,
                                    NullableDataFrame,
                                    DataFrameException,
                                    ByteColumn,
                                    ShortColumn,
                                    IntColumn,
                                    LongColumn,
                                    FloatColumn,
                                    DoubleColumn,
                                    StringColumn,
                                    CharColumn,
                                    BooleanColumn,
                                    BinaryColumn,
                                    NullableByteColumn,
                                    NullableShortColumn,
                                    NullableIntColumn,
                                    NullableLongColumn,
                                    NullableFloatColumn,
                                    NullableDoubleColumn,
                                    NullableStringColumn,
                                    NullableCharColumn,
                                    NullableBooleanColumn,
                                    NullableBinaryColumn)

# pylint: disable=too-many-lines
# pylint: disable=missing-function-docstring
# pylint: disable=consider-using-enumerate, invalid-name

class TestDataFrameUtils(unittest.TestCase):
    """Tests for DataFrame utility functions."""

    def setUp(self):
        self.column_names = [
            "byteCol",    # 0
            "shortCol",   # 1
            "intCol",     # 2
            "longCol",    # 3
            "stringCol",  # 4
            "charCol",    # 5
            "floatCol",   # 6
            "doubleCol",  # 7
            "booleanCol", # 8
            "binaryCol"   # 9
        ]

        self.df = DefaultDataFrame(
            ByteColumn(values=[1, 2, 3]),
            ShortColumn(values=[1, 2, 3]),
            IntColumn(values=[1, 2, 3]),
            LongColumn(values=[1, 2, 3]),
            StringColumn(values=["1", "2", "3"]),
            CharColumn(values=["a", "b", "c"]),
            FloatColumn(values=[1.0, 2.0, 3.0]),
            DoubleColumn(values=[1.0, 2.0, 3.0]),
            BooleanColumn(values=[True, False, True]),
            BinaryColumn(values=[
                bytearray.fromhex("0102030405"),
                bytearray.fromhex("0504030201"),
                bytearray.fromhex("0504010203")
            ]))

        self.nulldf = NullableDataFrame(
            NullableByteColumn(values=[1, None, 3]),
            NullableShortColumn(values=[1, None, 3]),
            NullableIntColumn(values=[1, None, 3]),
            NullableLongColumn(values=[1, None, 3]),
            NullableStringColumn(values=["1", None, "3"]),
            NullableCharColumn(values=["a", None, "c"]),
            NullableFloatColumn(values=[1.0, None, 3.0]),
            NullableDoubleColumn(values=[1.0, None, 3.0]),
            NullableBooleanColumn(values=[True, None, False]),
            NullableBinaryColumn(values=[
                bytearray.fromhex("0102030405"),
                None,
                bytearray.fromhex("0504010203")
            ]))

        self.df.set_column_names(self.column_names)
        self.nulldf.set_column_names(self.column_names)


    def test_exact_copy_for_default(self):
        MSG = "Value missmatch. Is not exact copy of original"
        copy = DataFrame.copy(self.df)
        self.assertTrue(
            isinstance(copy, DefaultDataFrame),
            "DataFrame should be of type DefaultDataFrame")

        self.assertTrue(copy.rows() == 3, "DataFrame should have 3 rows")
        self.assertTrue(copy.columns() == 10, "DataFrame should have 10 columns")
        self.assertTrue(self.column_names == copy.get_column_names(), "Column names should match")

        for i in range(1, copy.rows(), 1):
            self.assertTrue(copy.get_byte(0, i-1) == i, MSG)

        for i in range(1, copy.rows(), 1):
            self.assertTrue(copy.get_short(1, i-1) == i, MSG)

        for i in range(1, copy.rows(), 1):
            self.assertTrue(copy.get_int(2, i-1) == i, MSG)

        for i in range(1, copy.rows(), 1):
            self.assertTrue(copy.get_long(3, i-1) == i, MSG)

        for i in range(1, copy.rows(), 1):
            self.assertTrue(copy.get_string(4, i-1) == str(i), MSG)

        self.assertTrue(copy.get_char(5, 0) == "a", MSG)
        self.assertTrue(copy.get_char(5, 1) == "b", MSG)
        self.assertTrue(copy.get_char(5, 2) == "c", MSG)

        for i in range(1, copy.rows(), 1):
            self.assertTrue(copy.get_float(6, i-1) == float(i), MSG)

        for i in range(1, copy.rows(), 1):
            self.assertTrue(copy.get_double(7, i-1) == float(i), MSG)

        self.assertTrue(copy.get_boolean(8, 0), MSG)
        self.assertFalse(copy.get_boolean(8, 1), MSG)
        self.assertTrue(copy.get_boolean(8, 2), MSG)


        b0 = self.df.get_binary(9, 0)
        b1 = self.df.get_binary(9, 1)
        b2 = self.df.get_binary(9, 2)
        self.assertTrue(b0 == copy.get_binary(9, 0), MSG)
        self.assertTrue(b1 == copy.get_binary(9, 1), MSG)
        self.assertTrue(b2 == copy.get_binary(9, 2), MSG)
        self.assertTrue(b0 is not copy.get_binary(9, 0), "Copy should have different reference")
        self.assertTrue(b1 is not copy.get_binary(9, 1), "Copy should have different reference")
        self.assertTrue(b2 is not copy.get_binary(9, 2), "Copy should have different reference")

    def test_exact_copy_for_nullable(self):
        MSG = "Value missmatch. Is not exact copy of original"
        copy = DataFrame.copy(self.nulldf)
        self.assertTrue(
            isinstance(copy, NullableDataFrame),
            "DataFrame should be of type NullableDataFrame")

        self.assertTrue(copy.rows() == 3, "DataFrame should have 3 rows")
        self.assertTrue(copy.columns() == 10, "DataFrame should have 10 columns")
        self.assertTrue(
            self.column_names == copy.get_column_names(),
            "Column names should match")

        for i in range(1, copy.rows(), 1):
            if i == 2:
                self.assertTrue(copy.get_byte(0, i-1) is None, MSG)
            else:
                self.assertTrue(copy.get_byte(0, i-1) == i, MSG)

        for i in range(1, copy.rows(), 1):
            if i == 2:
                self.assertTrue(copy.get_short(1, i-1) is None, MSG)
            else:
                self.assertTrue(copy.get_short(1, i-1) == i, MSG)

        for i in range(1, copy.rows(), 1):
            if i == 2:
                self.assertTrue(copy.get_int(2, i-1) is None, MSG)
            else:
                self.assertTrue(MSG, copy.get_int(2, i-1) == i)

        for i in range(1, copy.rows(), 1):
            if i == 2:
                self.assertTrue(copy.get_long(3, i-1) is None, MSG)
            else:
                self.assertTrue(copy.get_long(3, i-1) == i, MSG)

        for i in range(1, copy.rows(), 1):
            if i == 2:
                self.assertTrue(copy.get_string(4, i-1) is None, MSG)
            else:
                self.assertTrue(copy.get_string(4, i-1) == str(i), MSG)

        self.assertTrue(copy.get_char(5, 0) == "a", MSG)
        self.assertTrue(copy.get_char(5, 1) is None, MSG)
        self.assertTrue(copy.get_char(5, 2) == "c", MSG)
        for i in range(1, copy.rows(), 1):
            if i == 2:
                self.assertTrue(copy.get_float(6, i-1) is None, MSG)
            else:
                self.assertTrue(copy.get_float(6, i-1) == float(i), MSG)

        for i in range(1, copy.rows(), 1):
            if i == 2:
                self.assertTrue(copy.get_double(7, i-1) is None, MSG)
            else:
                self.assertTrue(copy.get_double(7, i-1) == float(i), MSG)

        self.assertTrue(copy.get_boolean(8, 0), MSG)
        self.assertTrue(copy.get_boolean(8, 1) is None, MSG)
        self.assertFalse(copy.get_boolean(8, 2), MSG)

        b0 = self.nulldf.get_binary(9, 0)
        b2 = self.nulldf.get_binary(9, 2)
        self.assertTrue(b0 == copy.get_binary(9, 0), MSG)
        self.assertTrue(copy.get_binary(9, 1) is None, MSG)
        self.assertTrue(b2 == copy.get_binary(9, 2), MSG)

        self.assertTrue(b0 is not copy.get_binary(9, 0), "Copy should have different reference")
        self.assertTrue(b2 is not copy.get_binary(9, 2), "Copy should have different reference")

    def test_like_default(self):
        df2 = DataFrame.like(self.df)
        self.assertTrue(
            isinstance(df2, DefaultDataFrame),
            "DataFrame should be a DefaultDataFrame")

        self.assertTrue(df2.columns() == self.df.columns(), "DataFrame should have 10 columns")
        self.assertTrue(df2.is_empty(), "DataFrame should be empty")
        self.assertTrue(
            self.df.get_column_names() == df2.get_column_names(),
            "Columns names do not match")

        for i in range(df2.columns()):
            self.assertTrue(
                self.df.get_column(i).type_code() == df2.get_column(i).type_code(),
                "Columns have deviating types")

    def test_like_nullable(self):
        df2 = DataFrame.like(self.nulldf)
        self.assertTrue(
            isinstance(df2, NullableDataFrame), "DataFrame should be a NullableDataFrame")

        self.assertTrue(df2.columns() == self.df.columns(), "DataFrame should have 10 columns")
        self.assertTrue(df2.is_empty(), "DataFrame should be empty")
        self.assertTrue(
            self.df.get_column_names() == df2.get_column_names(),
            "Columns names do not match")

        for i in range(df2.columns()):
            self.assertTrue(
                self.nulldf.get_column(i).type_code() == df2.get_column(i).type_code(),
                "Columns have deviating types")

    def test_like_uninitialized(self):
        df2 = DataFrame.like(DefaultDataFrame())
        self.assertTrue(
            isinstance(df2, DefaultDataFrame), "DataFrame should be a DefaultDataFrame")

        self.assertTrue(df2.columns() == 0, "DataFrame should have 0 columns")
        self.assertTrue(df2.is_empty(), "DataFrame should be empty")
        self.assertFalse(df2.has_column_names(), "DataFrame should not have column names")
        df2 = DataFrame.like(NullableDataFrame())
        self.assertTrue(
            isinstance(df2, NullableDataFrame), "DataFrame should be a NullableDataFrame")

        self.assertTrue(df2.columns() == 0, "DataFrame should have 0 columns")
        self.assertTrue(df2.is_empty(), "DataFrame should be empty")
        self.assertFalse(df2.has_column_names(), "DataFrame should not have column names")



    #*************************************#
    #           Join operations           #
    #*************************************#



    def test_join_both_keys_specified_default_default(self):
        df1 = DefaultDataFrame(
            IntColumn("A1", [1, 2, 3, 4, 5, 6]),
            StringColumn("B", ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]),
            IntColumn("C", [53, 51, 54, 62, 41, 54]))

        df2 = DefaultDataFrame(
            IntColumn("D", [517, 575, 896, 741, 210, 231]),
            IntColumn("A2", [1, 2, 2, 4, 1, 6]),
            StringColumn("E", ["2018", "2019", "2019", "2020", "2018", "2017"]))

        res = df1.join(df2, "A1", "A2")
        self.assertTrue(
            isinstance(res, DefaultDataFrame), "DataFrame should be of type DefaultDataFrame")

        self.assertTrue(res.columns() == 5, "DataFrame should have 5 columns")
        self.assertTrue(res.rows() == 6, "DataFrame should have 6 rows")
        self.assertTrue(
            ["A1", "B", "C", "D", "E"] == res.get_column_names(), "Column names do not match")

        self.assertTrue(
            res.get_column("A1").type_code() == df1.get_column("A1").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("B").type_code() == df1.get_column("B").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("C").type_code() == df1.get_column("C").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("D").type_code() == df2.get_column("D").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("E").type_code() == df2.get_column("E").type_code(),
            "Column does not match")

        self.assertTrue(
            res.count("A1", "1") == 2, "DataFrame result does not match expected")

        self.assertTrue(
            res.count("A1", "2") == 2, "DataFrame result does not match expected")

        self.assertTrue(
            res.count("A1", "3") == 0, "DataFrame result does not match expected")

        self.assertTrue(
            res.count("A1", "4") == 1, "DataFrame result does not match expected")

        self.assertTrue(
            res.count("A1", "5") == 0, "DataFrame result does not match expected")

        self.assertTrue(
            res.count("A1", "6") == 1, "DataFrame result does not match expected")

    def test_join_one_key_specified_default_nullable(self):
        df1 = DefaultDataFrame(
            IntColumn("A", [1, 2, 3, 4, 5, 6]),
            StringColumn("B", ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]),
            IntColumn("C", [53, 51, 54, 62, 41, 54]))

        df2 = NullableDataFrame(
            NullableIntColumn("D", [517, 575, 896, 741, None, 231]),
            NullableIntColumn("A", [1, 2, None, 4, 1, 6]),
            NullableStringColumn("E", ["2018", "2019", None, "2020", "2018", None]))

        res = df1.join(df2, "A")
        self.assertTrue(
            isinstance(res, NullableDataFrame),
            "DataFrame should be of type NullableDataFrame")

        self.assertTrue(res.columns() == 5, "DataFrame should have 5 columns")
        self.assertTrue(res.rows() == 5, "DataFrame should have 5 rows")
        self.assertTrue(
            ["A", "B", "C", "D", "E"] == res.get_column_names(),
            "Column names do not match")

        self.assertTrue(
            res.get_column("A").type_code() == df1.get_column("A").as_nullable().type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("B").type_code() == df1.get_column("B").as_nullable().type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("C").type_code() == df1.get_column("C").as_nullable().type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("D").type_code() == df2.get_column("D").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("E").type_code() == df2.get_column("E").type_code(),
            "Column does not match")

        self.assertTrue(res.count("A", "1") == 2, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "2") == 1, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "3") == 0, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "4") == 1, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "5") == 0, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "6") == 1, "DataFrame result does not match expected")

    def testJoinNoKeySpecifiedNullableDefault(self):
        df1 = NullableDataFrame(
            NullableIntColumn("D", [517, 575, 896, 741, None, 231]),
            NullableIntColumn("A", [1, 2, None, 4, 1, 6]),
            NullableStringColumn("E", ["2018", "2019", None, "2020", "2018", None]))

        df2 = DefaultDataFrame(
            StringColumn("B", ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]),
            IntColumn("C", [53, 51, 54, 62, 41, 54]),
            IntColumn("A", [1, 2, 3, 4, 5, 6]))

        res = df1.join(df2)
        self.assertTrue(
            isinstance(res, NullableDataFrame),
            "DataFrame should be of type NullableDataFrame")

        self.assertTrue(res.columns() == 5, "DataFrame should have 5 columns")
        self.assertTrue(res.rows() == 5, "DataFrame should have 5 rows")
        self.assertTrue(
            ["D", "A", "E", "B", "C"] == res.get_column_names(),
            "Column names do not match")

        self.assertTrue(
            res.get_column("A").type_code() == df2.get_column("A").as_nullable().type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("B").type_code() == df2.get_column("B").as_nullable().type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("C").type_code() == df2.get_column("C").as_nullable().type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("D").type_code() == df1.get_column("D").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("E").type_code() == df1.get_column("E").type_code(),
            "Column does not match")

        self.assertTrue(res.count("A", "1") == 2, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "2") == 1, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "3") == 0, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "4") == 1, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "5") == 0, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "6") == 1, "DataFrame result does not match expected")

    def test_join_no_key_specified_nullable_nullable(self):
        df1 = NullableDataFrame(
            NullableIntColumn("D", [517, 575, 896, 741, None, 231]),
            NullableIntColumn("A", [1, 2, None, 4, 1, 6]),
            NullableStringColumn("E", ["2018", "2019", None, "2020", "2018", None]))

        df2 = NullableDataFrame(
            NullableStringColumn("B", ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]),
            NullableIntColumn("C", [53, 51, 54, 62, 41, 54]),
            NullableIntColumn("A", [1, None, 3, 4, 5, None]))

        res = df1.join(df2)
        self.assertTrue(
            isinstance(res, NullableDataFrame),
            "DataFrame should be of type NullableDataFrame")

        self.assertTrue(res.columns() == 5, "DataFrame should have 5 columns")
        self.assertTrue(res.rows() == 5, "DataFrame should have 5 rows")
        self.assertTrue(
            ["D", "A", "E", "B", "C"] == res.get_column_names(),
            "Column names do not match")

        self.assertTrue(
            res.get_column("A").type_code() == df2.get_column("A").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("B").type_code() == df2.get_column("B").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("C").type_code() == df2.get_column("C").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("D").type_code() == df1.get_column("D").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("E").type_code() == df1.get_column("E").type_code(),
            "Column does not match")

        self.assertTrue(res.count("A", "1") == 2, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "None") == 2, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "3") == 0, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "4") == 1, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "5") == 0, "DataFrame result does not match expected")
        self.assertTrue(res.count("A", "6") == 0, "DataFrame result does not match expected")

    def test_join_empty_arg(self):
        df1 = DefaultDataFrame(
            IntColumn("A", [517, 575]),
            IntColumn("B", [1, 2]),
            StringColumn("C", ["2018", "2019"]))

        df2 = NullableDataFrame(
            NullableStringColumn("D", ["AAA", "BBB", "CCC"]),
            NullableIntColumn("E", [53, 51, 54]),
            NullableIntColumn("A", [1, 2, 3]))

        df3 = df2.clone()
        df3.clear()
        res = df1.join(df3)
        self.assertTrue(
            isinstance(res, NullableDataFrame),
            "DataFrame should be of type NullableDataFrame")

        self.assertTrue(res.columns() == 5, "DataFrame should have 5 columns")
        self.assertTrue(res.rows() == 0, "DataFrame should have 0 rows")
        self.assertTrue(
            ["A", "B", "C", "D", "E"] == res.get_column_names(),
            "Column names do not match")

        df1.clear()
        res = df2.join(df1)
        self.assertTrue(
            isinstance(res, NullableDataFrame),
            "DataFrame should be of type NullableDataFrame")

        self.assertTrue(res.columns() == 5, "DataFrame should have 5 columns")
        self.assertTrue(res.rows() == 0, "DataFrame should have 0 rows")
        self.assertTrue(
            ["D", "E", "A", "B", "C"] == res.get_column_names(),
            "Column names do not match")

    def test_join_one_key_specified_duplicate_columns(self):
        df1 = NullableDataFrame(
            NullableIntColumn("A", [517, 575, 896, 741, None, 231]),
            NullableIntColumn("B", [1, 2, None, 4, 1, 6]),
            NullableStringColumn("C", ["2018", "2019", None, "2020", "2018", None]))

        df2 = DefaultDataFrame(
            IntColumn("B", [1, 2, 3, 4, 5, 6]),
            StringColumn("A", ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]),
            IntColumn("D", [53, 51, 54, 62, 41, 54]),
            IntColumn("C", [53, 51, 54, 62, 41, 54]))

        res = df1.join(df2, "B")
        self.assertTrue(
            isinstance(res, NullableDataFrame),
            "DataFrame should be of type NullableDataFrame")

        self.assertTrue(res.columns() == 4, "DataFrame should have 4 columns")
        self.assertTrue(res.rows() == 5, "DataFrame should have 5 rows")
        self.assertTrue(
            ["A", "B", "C", "D"] == res.get_column_names(),
            "Column names do not match")

        self.assertTrue(
            res.get_column("A").type_code() == df1.get_column("A").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("B").type_code() == df1.get_column("B").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("C").type_code() == df1.get_column("C").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("D").type_code() == df2.get_column("D").as_nullable().type_code(),
            "Column does not match")

        self.assertTrue(res.count("B", "1") == 2, "DataFrame result does not match expected")
        self.assertTrue(res.count("B", "2") == 1, "DataFrame result does not match expected")
        self.assertTrue(res.count("B", "3") == 0, "DataFrame result does not match expected")
        self.assertTrue(res.count("B", "4") == 1, "DataFrame result does not match expected")
        self.assertTrue(res.count("B", "5") == 0, "DataFrame result does not match expected")
        self.assertTrue(res.count("B", "6") == 1, "DataFrame result does not match expected")

    def test_join_both_key_specified_duplicate_columns(self):
        df1 = NullableDataFrame(
            NullableIntColumn("A", [517, 575, 896, 741, None, 231]),
            NullableIntColumn("B", [1, 2, None, 4, 1, 6]),
            NullableStringColumn("C", ["2018", "2019", None, "2020", "2018", None]))

        df2 = DefaultDataFrame(
            IntColumn("B", [53, 51, 54, 62, 41, 54]),
            StringColumn("A", ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]),
            IntColumn("E", [53, 51, 54, 62, 41, 54]),
            IntColumn("D", [1, 2, 3, 4, 5, 6]),
            IntColumn("C", [53, 51, 54, 62, 41, 54]))

        res = df1.join(df2, "B", "D")
        self.assertTrue(
            isinstance(res, NullableDataFrame),
            "DataFrame should be of type NullableDataFrame")

        self.assertTrue(res.columns() == 4, "DataFrame should have 4 columns")
        self.assertTrue(res.rows() == 5, "DataFrame should have 5 rows")
        self.assertTrue(
            ["A", "B", "C", "E"] == res.get_column_names(),
            "Column names do not match")

        self.assertTrue(
            res.get_column("A").type_code() == df1.get_column("A").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("B").type_code() == df1.get_column("B").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("C").type_code() == df1.get_column("C").type_code(),
            "Column does not match")

        self.assertTrue(
            res.get_column("E").type_code() == df2.get_column("E").as_nullable().type_code(),
            "Column does not match")

        self.assertTrue(res.count("B", "1") == 2, "DataFrame result does not match expected")
        self.assertTrue(res.count("B", "2") == 1, "DataFrame result does not match expected")
        self.assertTrue(res.count("B", "3") == 0, "DataFrame result does not match expected")
        self.assertTrue(res.count("B", "4") == 1, "DataFrame result does not match expected")
        self.assertTrue(res.count("B", "5") == 0, "DataFrame result does not match expected")
        self.assertTrue(res.count("B", "6") == 1, "DataFrame result does not match expected")

    def test_join_fail_no_matching_key(self):
        df1 = NullableDataFrame(
            NullableIntColumn("A", [517, 575]),
            NullableIntColumn("B", [1, 2]),
            NullableStringColumn("C", ["2018", "2019"]))

        df2 = NullableDataFrame(
            NullableStringColumn("D", ["AAA", "BBB"]),
            NullableIntColumn("E", [53, 51]),
            NullableIntColumn("F", [1, None]))

        self.assertRaises(
            DataFrameException, df1.join, df2)

    def test_join_fail_multiple_keys(self):
        df1 = NullableDataFrame(
            NullableIntColumn("A", [517, 575]),
            NullableIntColumn("B", [1, 2]),
            NullableStringColumn("C", ["2018", "2019"]))

        df2 = NullableDataFrame(
            NullableStringColumn("A", ["AAA", "BBB"]),
            NullableIntColumn("E", [53, 51]),
            NullableIntColumn("B", [1, None]))

        self.assertRaises(
            DataFrameException, df1.join, df2)

    def testJ_join_fail_null_arg(self):
        df1 = NullableDataFrame(
            NullableIntColumn("A", [517, 575]),
            NullableIntColumn("B", [1, 2]),
            NullableStringColumn("C", ["2018", "2019"]))

        self.assertRaises(
            DataFrameException, df1.join, None)

    def test_join_fail_invalid_column_name(self):
        df1 = NullableDataFrame(
            NullableIntColumn("A", [517, 575]),
            NullableIntColumn("B", [1, 2]),
            NullableStringColumn("C", ["2018", "2019"]))

        df2 = NullableDataFrame(
            NullableStringColumn("A", ["AAA", "BBB"]),
            NullableIntColumn("D", [53, 51]),
            NullableIntColumn("E", [1, None]))

        self.assertRaises(
            DataFrameException, df1.join, df2, "INVALID")

    def test_join_fail_invalid_column_name_second_Arg(self):
        df1 = NullableDataFrame(
            NullableIntColumn("A1", [517, 575]),
            NullableIntColumn("B", [1, 2]),
            NullableStringColumn("C", ["2018", "2019"]))

        df2 = NullableDataFrame(
            NullableStringColumn("A2", ["AAA", "BBB"]),
            NullableIntColumn("D", [53, 51]),
            NullableIntColumn("E", [1, None]))

        self.assertRaises(
            DataFrameException, df1.join, df2, "A1", "INVALID")

    def test_join_fail_empty_first_column_name(self):
        df1 = NullableDataFrame(
            NullableIntColumn("A1", [517, 575]),
            NullableIntColumn("B", [1, 2]),
            NullableStringColumn("C", ["2018", "2019"]))

        df2 = NullableDataFrame(
            NullableStringColumn("A2", ["AAA", "BBB"]),
            NullableIntColumn("D", [53, 51]),
            NullableIntColumn("E", [1, None]))

        self.assertRaises(
            DataFrameException, df1.join, df2, "", "A2")

    def test_join_fail_empty_second_column_name(self):
        df1 = NullableDataFrame(
            IntColumn("A1", [517, 575]),
            IntColumn("B", [1, 2]),
            StringColumn("C", ["2018", "2019"]))

        df2 = NullableDataFrame(
            NullableStringColumn("A2", ["AAA", "BBB"]),
            NullableIntColumn("D", [53, 51]),
            NullableIntColumn("E", [1, None]))

        self.assertRaises(
            DataFrameException, df1.join, df2, "A1", "")

    def test_join_fail_self_referential(self):
        df1 = NullableDataFrame(
            NullableIntColumn("A1", [517, 575]),
            NullableIntColumn("B", [1, 2]),
            NullableStringColumn("C", ["2018", "2019"]))

        self.assertRaises(
            DataFrameException, df1.join, df1)



    #*************************************#
    #           Merge operation           #
    #*************************************#



    def test_merge(self):
        df1 = DefaultDataFrame(
            ByteColumn(values=[1, 2, 3]),
            ShortColumn(values=[1, 2, 3]),
            IntColumn(values=[1, 2, 3]))

        df1.set_column_names(["c1", "c2", "c3"])

        df2 = DefaultDataFrame(
            CharColumn(values=["a", "b", "c"]),
            FloatColumn(values=[1.0, 2.0, 3.0]),
            DoubleColumn(values=[1.0, 2.0, 3.0]))

        df2.set_column_names(["c4", "c5", "c6"])

        res = DataFrame.merge(df1, df2)
        self.assertTrue(
            isinstance(res, DefaultDataFrame),
            "DataFrame should be of type DefaultDataFrame")

        self.assertTrue(res.rows() == 3, "DataFrame should have 3 rows")
        self.assertTrue(res.columns() == 6, "DataFrame should have 6 columns")
        self.assertTrue(
            ["c1", "c2", "c3", "c4", "c5", "c6"] == res.get_column_names(),
            "Column names should match")

        self.assertTrue(
            res.get_column(0) == df1.get_column(0),
            "Column references do not match")

        self.assertTrue(
            res.get_column(1) == df1.get_column(1),
            "Column references do not match")

        self.assertTrue(
            res.get_column(2) == df1.get_column(2),
            "Column references do not match")

        self.assertTrue(
            res.get_column(3) == df2.get_column(0),
            "Column references do not match")

        self.assertTrue(
            res.get_column(4) == df2.get_column(1),
            "Column references do not match")

        self.assertTrue(
            res.get_column(5) == df2.get_column(2),
            "Column references do not match")

    def test_merge_different_types(self):
        df1 = DefaultDataFrame(
            StringColumn("A", ["AAA", "AAB", "AAC"]),
            FloatColumn("B", [11.11, 22.22, 33.33]),
            CharColumn("C", ["A", "B", "C"]))

        df2 = DefaultDataFrame(
            StringColumn("D", ["BBA", "BBB", "BBC"]),
            IntColumn("E", [10, 11, 12]))

        df3 = NullableDataFrame(
            NullableIntColumn("F", [0, 1, 2]),
            NullableFloatColumn("G", [0.1, 0.2, 0.3]))

        res = DataFrame.merge(df1, df2, df3)
        self.assertTrue(
            isinstance(res, NullableDataFrame),
            "DataFrame should be of type NullableDataFrame")

        self.assertTrue(res.rows() == 3, "DataFrame should have 3 rows")
        self.assertTrue(res.columns() == 7, "DataFrame should have 7 columns")
        self.assertTrue(
            ["A", "B", "C", "D", "E", "F", "G"] == res.get_column_names(),
            "Column names should match")

        self.assertTrue(
            res.get_column(5) == df3.get_column(0),
            "Column references do not match")

        self.assertTrue(
            res.get_column(6) == df3.get_column(1),
            "Column references do not match")

    def test_merge_duplicate_names(self):
        df1 = DefaultDataFrame(
            StringColumn("A", ["AAA", "AAB", "AAC"]),
            FloatColumn("B", [11.11, 22.22, 33.33]),
            CharColumn("C", ["A", "B", "C"]))

        df2 = DefaultDataFrame(
            StringColumn("D", ["BBA", "BBB", "BBC"]),
            IntColumn("B", [10, 11, 12]))

        df3 = NullableDataFrame(
            NullableIntColumn("B", [0, 1, 2]),
            NullableFloatColumn("D", [0.1, 0.2, 0.3]))

        res = DataFrame.merge(df1, df2, df3)
        self.assertTrue(
            isinstance(res, NullableDataFrame),
            "DataFrame should be of type NullableDataFrame")

        self.assertTrue(res.rows() == 3, "DataFrame should have 3 rows")
        self.assertTrue(res.columns() == 7, "DataFrame should have 7 columns")
        self.assertTrue(
            ["A", "B_0", "C", "D_0", "B_1", "B_2", "D_1"] == res.get_column_names(),
            "Column names should match")

        self.assertTrue(
            res.get_column(5) == df3.get_column(0),
            "Column references do not match")

        self.assertTrue(
            res.get_column(6) == df3.get_column(1),
            "Column references do not match")

    def test_merge_one_arg(self):
        df1 = DefaultDataFrame(
            StringColumn("A", ["AAA", "AAB", "AAC"]),
            FloatColumn("B", [11.11, 22.22, 33.33]),
            CharColumn("C", ["A", "B", "C"]))

        res = DataFrame.merge(df1)
        self.assertTrue(res is df1, "DataFrame reference does not match")

    def test_merge_fail_invalid_row_size(self):
        df1 = DefaultDataFrame(
            StringColumn("A", ["AAA", "AAB", "AAC"]),
            FloatColumn("B", [11.11, 22.22, 33.33]),
            CharColumn("C", ["A", "B", "C"]))

        df2 = DefaultDataFrame(
            StringColumn("D", ["BBA", "BBB", "BBC"]),
            IntColumn("B", [10, 11, 12]))

        df3 = NullableDataFrame(
            NullableIntColumn("B", [0, 1, 2, 3]),
            NullableFloatColumn("D", [0.1, 0.2, 0.3, 0.4]))

        self.assertRaises(
            DataFrameException, DataFrame.merge, df1, df2, df3)

    def test_merge_fail_null_arg(self):
        df1 = DefaultDataFrame(
            StringColumn("A", ["AAA", "AAB", "AAC"]),
            FloatColumn("B", [11.11, 22.22, 33.33]),
            CharColumn("C", ["A", "B", "C"]))

        df2 = NullableDataFrame(
            NullableIntColumn("B", [0, 1, 2, 3]),
            NullableFloatColumn("D", [0.1, 0.2, 0.3, 0.4]))

        self.assertRaises(
            DataFrameException, DataFrame.merge, df1, df2, None)



    #***************************************#
    #           Convert operation           #
    #***************************************#



    def test_convert_from_default_to_nullable(self):
        conv = DataFrame.convert_to(self.df, "NullableDataFrame")
        self.assertTrue(
            isinstance(conv, NullableDataFrame),
            "DataFrame should be of type NullableDataFrame")

        self.assertTrue(conv.rows() == 3, "DataFrame should have 3 rows")
        self.assertTrue(conv.columns() == 10, "DataFrame should have 10 columns")
        self.assertTrue(
            self.column_names == conv.get_column_names(),
            "Column names should match")

        self.assertTrue(self.df.get_row(0) == conv.get_row(0), "Rows do not match")
        self.assertTrue(self.df.get_row(1) == conv.get_row(1), "Rows do not match")
        self.assertTrue(self.df.get_row(2) == conv.get_row(2), "Rows do not match")

    def test_convert_from_nullable_to_default(self):
        conv = DataFrame.convert_to(self.nulldf, "DefaultDataFrame")
        self.assertTrue(
            isinstance(conv, DefaultDataFrame),
            "DataFrame should be of type DefaultDataFrame")

        self.assertTrue(conv.rows() == 3, "DataFrame should have 3 rows")
        self.assertTrue(conv.columns() == 10, "DataFrame should have 10 columns")
        self.assertTrue(
            self.column_names == conv.get_column_names(),
            "Column names should match")

        obj = conv.to_array()
        for i in range(len(obj)):
            for j in range(len(obj[i])):
                self.assertTrue(
                    obj[i][j] is not None,
                    "Converted DataFrame should not contain any None values")


if __name__ == "__main__":
    unittest.main()
