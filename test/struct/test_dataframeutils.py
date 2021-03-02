# Copyright (C) 2021 Raven Computing
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



    #*********************************************#
    #           []-operator and slicing           #
    #*********************************************#



    def test_getitem_column_by_index(self):
        for i in range(self.df.columns()):
            col = self.df[i]
            self.assertTrue(col is self.df.get_column(i), "Invalid Column instance")

        for i in range(self.nulldf.columns()):
            col = self.nulldf[i]
            self.assertTrue(col is self.nulldf.get_column(i), "Invalid Column instance")

    def test_getitem_column_by_negative_index(self):
        for i in range(0, self.df.columns()+1, -1):
            col = self.df[i]
            self.assertTrue(
                col is self.df.get_column(i % self.df.columns()),
                "Invalid Column instance")

        for i in range(0, self.nulldf.columns()+1, -1):
            col = self.nulldf[i]
            self.assertTrue(
                col is self.nulldf.get_column(i % self.nulldf.columns()),
                "Invalid Column instance")

    def test_getitem_column_by_invalid_index_exception(self):
        self.assertRaises(DataFrameException, self.df.__getitem__, self.df.columns())
        self.assertRaises(DataFrameException, self.df.__getitem__, (-1 * self.df.columns())-1)
        self.assertRaises(DataFrameException, self.nulldf.__getitem__, self.nulldf.columns())
        self.assertRaises(
            DataFrameException,
            self.nulldf.__getitem__, (-1 * self.nulldf.columns())-1)

    def test_getitem_column_by_name(self):
        for name in self.df.get_column_names():
            col = self.df[name]
            self.assertTrue(col is self.df.get_column(name), "Invalid Column instance")

        for name in self.nulldf.get_column_names():
            col = self.nulldf[name]
            self.assertTrue(col is self.nulldf.get_column(name), "Invalid Column instance")

    def test_getitem_column_by_invalid_name_exception(self):
        self.assertRaises(DataFrameException, self.df.__getitem__, "INVALID_COL")
        self.assertRaises(DataFrameException, self.nulldf.__getitem__, "INVALID_COL")

    def test_getitem_value_by_column_index(self):
        for i in range(self.df.rows()):
            row = self.df.get_row(i)
            for j in range(self.df.columns()):
                val = self.df[j, i]
                self.assertTrue(val == row[j], "Unexpected value")

        for i in range(self.nulldf.rows()):
            row = self.nulldf.get_row(i)
            for j in range(self.nulldf.columns()):
                val = self.nulldf[j, i]
                if row[j] is None:
                    self.assertTrue(val is None, "Unexpected value")
                else:
                    self.assertTrue(val == row[j], "Unexpected value")

    def test_getitem_value_by_column_name(self):
        for i in range(self.df.rows()):
            row = self.df.get_row(i)
            for j in range(self.df.columns()):
                name = self.df.get_column(j).get_name()
                val = self.df[name, i]
                self.assertTrue(val == row[j], "Unexpected value")

        for i in range(self.nulldf.rows()):
            row = self.nulldf.get_row(i)
            for j in range(self.nulldf.columns()):
                name = self.df.get_column(j).get_name()
                val = self.nulldf[name, i]
                if row[j] is None:
                    self.assertTrue(val is None, "Unexpected value")
                else:
                    self.assertTrue(val == row[j], "Unexpected value")

    def test_getitem_value_by_negative_column_and_row_index(self):
        for i in range(0, self.df.rows()+1, -1):
            row = self.df.get_row(i % self.df.rows())
            for j in range(0, self.df.columns()+1, -1):
                val = self.df[j, i]
                self.assertTrue(val == row[j], "Unexpected value")

        for i in range(0, self.nulldf.rows()+1, -1):
            row = self.nulldf.get_row(i % self.nulldf.rows())
            for j in range(0, self.nulldf.columns()+1, -1):
                val = self.nulldf[j, i]
                if row[j] is None:
                    self.assertTrue(val is None, "Unexpected value")
                else:
                    self.assertTrue(val == row[j], "Unexpected value")

    def test_getitem_value_by_invalid_column_and_row_index_exception(self):
        cols = self.df.columns()
        rows = self.df.rows()
        self.assertRaises(DataFrameException, self.df.__getitem__, (cols, 0))
        self.assertRaises(DataFrameException, self.df.__getitem__, (0, rows))
        self.assertRaises(DataFrameException, self.df.__getitem__, (-1-cols, 0))
        self.assertRaises(DataFrameException, self.df.__getitem__, (0, -1-rows))
        cols = self.nulldf.columns()
        rows = self.nulldf.rows()
        self.assertRaises(DataFrameException, self.nulldf.__getitem__, (cols, 0))
        self.assertRaises(DataFrameException, self.nulldf.__getitem__, (0, rows))
        self.assertRaises(DataFrameException, self.nulldf.__getitem__, (-1-cols, 0))
        self.assertRaises(DataFrameException, self.nulldf.__getitem__, (0, -1-rows))

    def test_getitem_filter_by_column_index(self):
        self.df.add_rows(self.df)
        filtered = self.df[2, "1|3"]
        self.assertTrue(filtered is not None,
                        "API violation: Returned DataFrame should not be None")

        self.assertTrue(filtered == self.df.filter(2, "1|3"),
                        "Filtered DataFrame does not match expected")

        self.nulldf.add_rows(self.nulldf)
        filtered = self.nulldf[2, "1|3"]
        self.assertTrue(filtered is not None,
                        "API violation: Returned DataFrame should not be None")

        self.assertTrue(filtered == self.nulldf.filter(2, "1|3"),
                        "Filtered DataFrame does not match expected")

    def test_getitem_filter_by_invalid_column_type_exception(self):
        self.assertRaises(DataFrameException, self.df.__getitem__, ((1, 2), "myregex"))
        self.assertRaises(
            DataFrameException,
            self.df.__getitem__,
            (("byteCol", "shortCol"), "myregex"))

        self.assertRaises(DataFrameException, self.df.__getitem__, (slice(2, 7, 1), "myregex"))
        self.assertRaises(
            DataFrameException,
            self.nulldf.__getitem__,
            (slice(2, 7, 1), "myregex"))

    def test_getitem_filter_by_column_name(self):
        self.df.add_rows(self.df)
        filtered = self.df["booleanCol", "True"]
        self.assertTrue(filtered is not None,
                        "API violation: Returned DataFrame should not be None")

        self.assertTrue(filtered == self.df.filter("booleanCol", "True"),
                        "Filtered DataFrame does not match expected")

        self.nulldf.add_rows(self.nulldf)
        filtered = self.nulldf["booleanCol", "True"]
        self.assertTrue(filtered is not None,
                        "API violation: Returned DataFrame should not be None")

        self.assertTrue(filtered == self.nulldf.filter("booleanCol", "True"),
                        "Filtered DataFrame does not match expected")

    def test_getitem_row(self):
        for i in range(self.df.rows()):
            row = self.df[:, i]
            self.assertTrue(row == self.df.get_row(i), "Row does not match expected list")

        for i in range(self.df.rows()):
            row = self.nulldf[:, i]
            self.assertTrue(row == self.nulldf.get_row(i), "Row does not match expected list")

    def test_getitem_row_slice_columns(self):
        for i in range(self.df.rows()):
            row = self.df[2:7, i]
            self.assertTrue(
                row == self.df.get_columns(cols=(2, 3, 4, 5, 6)).get_row(i),
                "Row does not match expected list")

        for i in range(self.df.rows()):
            row = self.nulldf[1:6, i]
            self.assertTrue(
                row == self.nulldf.get_columns(cols=(1, 2, 3, 4, 5)).get_row(i),
                "Row does not match expected list")

    def test_getitem_dataframe_slice_rows(self):
        cols = self.df.columns()
        rows = self.df.rows()
        for i in range(cols):
            for j in range(rows):
                df = self.df[i:cols, j:rows]
                c = tuple(range(i, cols))
                self.assertTrue(
                    df == self.df.get_columns(cols=c).get_rows(from_index=j, to_index=rows),
                    "DataFrames do not match")

        cols = self.nulldf.columns()
        rows = self.nulldf.rows()
        for i in range(cols):
            for j in range(rows):
                df = self.nulldf[i:cols, j:rows]
                c = tuple(range(i, cols))
                self.assertTrue(
                    df == self.nulldf.get_columns(cols=c).get_rows(from_index=j, to_index=rows),
                    "DataFrames do not match")

    def test_getitem_dataframe_rows_by_index(self):
        self.df.add_rows(self.df)
        df = self.df[2, (0, 2, 5)]
        truth = DataFrame.like(self.df.get_columns(cols=2))
        for i in [0, 2, 5]:
            truth.add_row(self.df.get_columns(cols=2).get_row(i))

        self.assertTrue(df == truth, "DataFrames do not match")

        df = self.df[4, (1, 2, 3)]
        truth = DataFrame.like(self.df.get_columns(cols=4))
        for i in [1, 2, 3]:
            truth.add_row(self.df.get_columns(cols=4).get_row(i))

        self.assertTrue(df == truth, "DataFrames do not match")

        self.nulldf.add_rows(self.nulldf)
        df = self.nulldf["intCol", (0, 2, 5)]
        truth = DataFrame.like(self.nulldf.get_columns(cols="intCol"))
        for i in [0, 2, 5]:
            truth.add_row(self.nulldf.get_columns(cols="intCol").get_row(i))

        self.assertTrue(df == truth, "DataFrames do not match")

        df = self.nulldf["stringCol", (1, 2, 3)]
        truth = DataFrame.like(self.nulldf.get_columns(cols="stringCol"))
        for i in [1, 2, 3]:
            truth.add_row(self.nulldf.get_columns(cols="stringCol").get_row(i))

        self.assertTrue(df == truth, "DataFrames do not match")

    def test_getitem_dataframe_slice_columns_rows_by_index(self):
        self.df.add_rows(self.df)
        cols = self.df.columns()
        rows = self.df.rows()
        for i in range(cols):
            for j in range(rows):
                c = tuple(range(i, cols))
                r = tuple(range(j, rows))
                df = self.df[c, r]
                self.assertTrue(
                    df == self.df.get_columns(cols=c).get_rows(from_index=j, to_index=rows),
                    "DataFrames do not match")

        self.nulldf.add_rows(self.nulldf)
        cols = self.nulldf.columns()
        rows = self.nulldf.rows()
        for i in range(cols):
            for j in range(rows):
                c = tuple(range(i, cols))
                r = tuple(range(j, rows))
                df = self.nulldf[c, r]
                self.assertTrue(
                    df == self.nulldf.get_columns(cols=c).get_rows(from_index=j, to_index=rows),
                    "DataFrames do not match")

    def test_setitem_value_by_column_index(self):
        truth = self.df.clone()
        truth.set_int(2, 0, 15)
        truth.set_int(2, 2, 42)
        truth.set_string(4, 0, "TEST1")
        truth.set_string(4, 2, "TEST2")
        truth.set_boolean(8, 0, False)
        truth.set_boolean(8, 1, True)
        self.df[2, 0] = 15
        self.df[2, 2] = 42
        self.df[4, 0] = "TEST1"
        self.df[4, 2] = "TEST2"
        self.df[8, 0] = False
        self.df[8, 1] = True
        self.assertTrue(self.df == truth, "DataFrames do not match")

    def test_setitem_value_by_negative_column_row_index(self):
        truth = self.df.clone()
        truth.set_int(2, 0, 15)
        truth.set_int(2, 2, 42)
        truth.set_string(4, 0, "TEST1")
        truth.set_string(4, 2, "TEST2")
        truth.set_boolean(8, 0, False)
        truth.set_boolean(8, 1, True)
        self.df[-8, -3] = 15
        self.df[-8, -1] = 42
        self.df[-6, -3] = "TEST1"
        self.df[-6, -1] = "TEST2"
        self.df[-2, -3] = False
        self.df[-2, -2] = True
        self.assertTrue(self.df == truth, "DataFrames do not match")

    def test_setitem_value_by_column_name(self):
        truth = self.nulldf.clone()
        truth.set_int("intCol", 0, 15)
        truth.set_int("intCol", 2, None)
        truth.set_string("stringCol", 0, "TEST1")
        truth.set_string("stringCol", 2, None)
        truth.set_boolean("booleanCol", 0, False)
        truth.set_boolean("booleanCol", 2, None)
        self.nulldf["intCol", 0] = 15
        self.nulldf["intCol", 2] = None
        self.nulldf["stringCol", 0] = "TEST1"
        self.nulldf["stringCol", 2] = None
        self.nulldf["booleanCol", 0] = False
        self.nulldf["booleanCol", 2] = None
        self.assertTrue(self.nulldf == truth, "DataFrames do not match")

    def test_setitem_value_slice_rows_by_column_index(self):
        self.df.add_rows(self.df)
        truth = self.df.clone()
        for i in range(1, 5):
            truth.set_int(2, i, 42)
        for i in range(0, 4):
            truth.set_string(4, i, "TEST")
        for i in range(2, truth.rows()):
            truth.set_boolean(8, i, False)

        self.df[2, 1:5] = 42
        self.df[(4, ), :4] = "TEST"
        self.df[8:9, 2:] = False
        self.assertTrue(self.df == truth, "DataFrames do not match")

    def test_setitem_value_slice_rows_by_column_name(self):
        self.nulldf.add_rows(self.nulldf)
        truth = self.nulldf.clone()
        for i in range(1, 5):
            truth.set_int("intCol", i, 42)
        for i in range(0, 4):
            truth.set_string("stringCol", i, None)
        for i in range(2, truth.rows()):
            truth.set_boolean("booleanCol", i, False)

        self.nulldf["intCol", 1:5] = 42
        self.nulldf["stringCol", :4] = None
        self.nulldf["booleanCol", 2:] = False
        self.assertTrue(self.nulldf == truth, "DataFrames do not match")

    def test_setitem_replace_by_column_index(self):
        self.df.add_rows(self.df)
        truth = self.df.clone()
        truth.replace(2, "1|3", 42)
        truth.replace(5, "a|c", "F")
        truth.replace(8, "True", False)
        self.df[2, "1|3"] = 42
        self.df[5, "a|c"] = "F"
        self.df[8, "True"] = False
        self.assertTrue(self.df == truth, "DataFrames do not match")

    def test_setitem_replace_by_column_name(self):
        self.nulldf.add_rows(self.nulldf)
        truth = self.nulldf.clone()
        truth.replace(0, "2", 42)
        truth.replace(5, "a|D", "F")
        truth.replace(8, "True", None)
        self.nulldf[2, "2"] = 42
        self.nulldf[5, "a|D"] = "F"
        self.nulldf[8, "True"] = None
        self.assertTrue(self.nulldf == truth, "DataFrames do not match")

    def test_setitem_set_column_by_index(self):
        self.df[0] = LongColumn("TEST1", [11, 22, 33])
        self.df[4] = StringColumn("TEST2", ["val1", "val2", "val3"])
        self.df[6] = BooleanColumn(values=[False, False, True])
        self.assertTrue(self.df.columns() == 10, "DataFrame should have 10 columns")
        self.assertTrue(self.df.get_column(0).type_name() == "long", "Column type does not match")
        self.assertTrue(self.df.get_column(0).get_name() == "TEST1", "Column name does not match")
        self.assertTrue(
            self.df.get_column(0).as_array().tolist() == [11, 22, 33],
            "Column values do not match")

        self.assertTrue(
            self.df.get_column(4).type_name() == "string",
            "Column type does not match")

        self.assertTrue(
            self.df.get_column(4).get_name() == "TEST2",
            "Column name does not match")

        self.assertTrue(
            self.df.get_column(4).as_array().tolist() == ["val1", "val2", "val3"],
            "Column values do not match")

        self.assertTrue(
            self.df.get_column(6).type_name() == "boolean",
            "Column type does not match")

        self.assertTrue(
            self.df.get_column(6) is self.df.get_column("floatCol"),
            "Columns do not match")

        self.assertTrue(
            self.df.get_column(6).get_name() == "floatCol", "Column name does not match")

        self.assertTrue(
            self.df.get_column(6).as_array().tolist() == [False, False, True],
            "Column values do not match")

    def test_setitem_set_column_by_name(self):
        self.nulldf["intCol"] = NullableLongColumn("TEST1", [11, 22, None])
        self.nulldf["stringCol"] = NullableStringColumn("TEST2", ["val1", None, "val3"])
        self.nulldf["floatCol"] = NullableBooleanColumn(values=[None, None, True])
        self.assertTrue(self.nulldf.columns() == 10, "DataFrame should have 10 columns")
        self.assertTrue(
            self.nulldf.get_column("intCol").type_name() == "long",
            "Column type does not match")

        self.assertTrue(
            self.nulldf.get_column("intCol").get_name() == "intCol",
            "Column name does not match")

        self.assertTrue(
            self.nulldf.get_column("intCol").as_array().tolist() == [11, 22, None],
            "Column values do not match")

        self.assertTrue(
            self.nulldf.get_column("stringCol").type_name() == "string",
            "Column type does not match")

        self.assertTrue(
            self.nulldf.get_column("stringCol").get_name() == "stringCol",
            "Column name does not match")

        self.assertTrue(
            self.nulldf.get_column("stringCol").as_array().tolist() == ["val1", None, "val3"],
            "Column values do not match")

        self.assertTrue(
            self.nulldf.get_column("floatCol").type_name() == "boolean",
            "Column type does not match")

        self.assertTrue(
            self.nulldf.get_column("floatCol").get_name() == "floatCol",
            "Column name should be None")

        self.assertTrue(
            self.nulldf.get_column("floatCol").as_array().tolist() == [None, None, True],
            "Column values do not match")

    def test_setitem_add_column_by_index(self):
        self.df[10] = LongColumn("TEST1", [11, 22, 33])
        self.df[11] = StringColumn("TEST2", ["val1", "val2", "val3"])
        self.df[12] = BooleanColumn(values=[False, False, True])
        self.assertTrue(self.df.columns() == 13, "DataFrame should have 13 columns")
        self.assertTrue(self.df.get_column(10).type_name() == "long", "Column type does not match")
        self.assertTrue(self.df.get_column(10).get_name() == "TEST1", "Column name does not match")
        self.assertTrue(
            self.df.get_column(10).as_array().tolist() == [11, 22, 33],
            "Column values do not match")

        self.assertTrue(
            self.df.get_column(11).type_name() == "string",
            "Column type does not match")

        self.assertTrue(
            self.df.get_column(11).get_name() == "TEST2",
            "Column name does not match")

        self.assertTrue(
            self.df.get_column(11).as_array().tolist() == ["val1", "val2", "val3"],
            "Column values do not match")

        self.assertTrue(
            self.df.get_column(12).type_name() == "boolean",
            "Column type does not match")

        self.assertTrue(
            self.df.get_column(12).get_name() is None, "Column name should be None")

        self.assertTrue(
            self.df.get_column(12).as_array().tolist() == [False, False, True],
            "Column values do not match")

    def test_setitem_add_column_by_name(self):
        self.nulldf["TEST_A"] = NullableLongColumn("TEST1", [11, 22, None])
        self.nulldf["TEST_B"] = NullableStringColumn("TEST2", ["val1", "val2", None])
        self.nulldf["TEST_C"] = NullableBooleanColumn(values=[False, False, None])
        self.assertTrue(self.nulldf.columns() == 13, "DataFrame should have 13 columns")
        self.assertTrue(
            self.nulldf.get_column("TEST_A").type_name() == "long",
            "Column names do not match")

        self.assertTrue(
            self.nulldf.get_column("TEST_A").get_name() == "TEST_A",
            "Column name do not match")

        self.assertTrue(
            self.nulldf.get_column("TEST_A").as_array().tolist() == [11, 22, None],
            "Column values do not match")

        self.assertTrue(
            self.nulldf.get_column("TEST_B").type_name() == "string",
            "Column names do not match")

        self.assertTrue(
            self.nulldf.get_column("TEST_B").get_name() == "TEST_B",
            "Column names do not match")

        self.assertTrue(
            self.nulldf.get_column("TEST_B").as_array().tolist() == ["val1", "val2", None],
            "Column values do not match")

        self.assertTrue(
            self.nulldf.get_column("TEST_C").type_name() == "boolean",
            "Column type does not match")

        self.assertTrue(
            self.nulldf.get_column("TEST_C").get_name() == "TEST_C",
            "Column names do not match")

        self.assertTrue(
            self.nulldf.get_column("TEST_C").as_array().tolist() == [False, False, None],
            "Column values do not match")

    def test_setitem_set_single_row(self):
        truth = self.df.clone()
        truth.set_row(0, [4, 4, 4, 4, "42", "D", 4.0, 4.0, True, bytearray.fromhex("0004")])
        truth.set_row(1, [6, 6, 6, 6, "66", "F", 6.0, 6.0, False, bytearray.fromhex("0066")])
        truth.set_row(2, [7, 7, 7, 7, "77", "G", 7.0, 7.0, True, bytearray.fromhex("aa77")])
        self.df[:, 0] = [4, 4, 4, 4, "42", "D", 4.0, 4.0, True, bytearray.fromhex("0004")]
        self.df[:, 1] = [6, 6, 6, 6, "66", "F", 6.0, 6.0, False, bytearray.fromhex("0066")]
        self.df[:, 2] = [7, 7, 7, 7, "77", "G", 7.0, 7.0, True, bytearray.fromhex("aa77")]
        self.assertTrue(self.df == truth, "DataFrames do not match")

        truth = self.nulldf.clone()
        truth.set_row(0, [4, None, 4, None, "42", "D", None, 4.0, True, None])
        truth.set_row(1, [6, 6, 6, 6, "66", "F", 6.0, 6.0, False, bytearray.fromhex("0066")])
        truth.set_row(2, [7, 7, 7, None, "77", "G", 7.0, 7.0, None, None])
        self.nulldf[:, 0] = [4, None, 4, None, "42", "D", None, 4.0, True, None]
        self.nulldf[:, 1] = [6, 6, 6, 6, "66", "F", 6.0, 6.0, False, bytearray.fromhex("0066")]
        self.nulldf[:, 2] = [7, 7, 7, None, "77", "G", 7.0, 7.0, None, None]
        self.assertTrue(self.nulldf == truth, "DataFrames do not match")

    def test_setitem_set_multiple_rows(self):
        self.df.add_rows(self.df)
        truth = self.df.clone()
        truth.set_row(1, [4, 4, 4, 4, "42", "D", 4.0, 4.0, True, bytearray.fromhex("0004")])
        truth.set_row(3, [4, 4, 4, 4, "42", "D", 4.0, 4.0, True, bytearray.fromhex("0004")])
        truth.set_row(5, [4, 4, 4, 4, "42", "D", 4.0, 4.0, True, bytearray.fromhex("0004")])
        self.df[:, (1, 3, 5)] = [4, 4, 4, 4, "42", "D", 4.0, 4.0, True, bytearray.fromhex("0004")]
        self.assertTrue(self.df == truth, "DataFrames do not match")

        self.nulldf.add_rows(self.nulldf)
        truth = self.nulldf.clone()
        truth.set_row(1, [4, None, 4, None, "42", "D", None, 4.0, True, None])
        truth.set_row(3, [4, None, 4, None, "42", "D", None, 4.0, True, None])
        truth.set_row(5, [4, None, 4, None, "42", "D", None, 4.0, True, None])
        self.nulldf[:, (1, 3, 5)] = [4, None, 4, None, "42", "D", None, 4.0, True, None]
        self.assertTrue(self.nulldf == truth, "DataFrames do not match")

    def test_setitem_set_multiple_rows_slices(self):
        self.df.add_rows(self.df)
        truth = self.df.clone()
        truth.set_row(1, [4, 4, 4, 4, "42", "D", 4.0, 4.0, True, bytearray.fromhex("0004")])
        truth.set_row(2, [4, 4, 4, 4, "42", "D", 4.0, 4.0, True, bytearray.fromhex("0004")])
        truth.set_row(3, [4, 4, 4, 4, "42", "D", 4.0, 4.0, True, bytearray.fromhex("0004")])
        self.df[:, 1:4] = [4, 4, 4, 4, "42", "D", 4.0, 4.0, True, bytearray.fromhex("0004")]
        self.assertTrue(self.df == truth, "DataFrames do not match")

        self.nulldf.add_rows(self.nulldf)
        truth = self.nulldf.clone()
        truth.set_row(0, [4, None, 4, None, "42", "D", None, 4.0, True, None])
        truth.set_row(2, [4, None, 4, None, "42", "D", None, 4.0, True, None])
        truth.set_row(4, [4, None, 4, None, "42", "D", None, 4.0, True, None])
        self.nulldf[:, ::2] = [4, None, 4, None, "42", "D", None, 4.0, True, None]
        self.assertTrue(self.nulldf == truth, "DataFrames do not match")

    def test_setitem_set_single_row_with_specific_column(self):
        truth = self.df.clone()
        truth.set_int("intCol", 1, 4)
        truth.set_string("stringCol", 1, "TEST")
        truth.set_boolean("booleanCol", 1, True)
        self.df[("intCol", "stringCol", "booleanCol"), 1] = [4, "TEST", True]
        self.assertTrue(self.df == truth, "DataFrames do not match")

        truth = self.nulldf.clone()
        truth.set_int("intCol", 0, 4)
        truth.set_string("stringCol", 0, "TEST")
        truth.set_boolean("booleanCol", 0, None)
        self.nulldf[("intCol", "stringCol", "booleanCol"), 0] = [4, "TEST", None]
        self.assertTrue(self.nulldf == truth, "DataFrames do not match")

    def test_setitem_set_single_row_column_slice(self):
        truth = self.df.clone()
        truth.set_short(1, 1, 42)
        truth.set_int(2, 1, 43)
        truth.set_long(3, 1, 44)
        truth.set_string(4, 1, "TEST")
        self.df[1:5, 1] = [42, 43, 44, "TEST"]
        self.assertTrue(self.df == truth, "DataFrames do not match")

        truth = self.nulldf.clone()
        truth.set_short(1, 2, 42)
        truth.set_long(3, 2, 43)
        truth.set_char(5, 2, "F")
        truth.set_double(7, 2, 4.0)
        truth.set_binary(9, 2, None)
        self.nulldf[1::2, 2] = [42, 43, "F", 4.0, None]
        self.assertTrue(self.nulldf == truth, "DataFrames do not match")

    def test_setitem_set_multiple_rows_dataframe(self):
        self.df.add_rows(self.df)
        replacement = DataFrame.like(self.df)
        replacement.add_row([7, 7, 7, 7, "77", "7", 7.0, 7.0, False, bytearray.fromhex("0077")])
        replacement.add_row([8, 8, 8, 8, "88", "8", 8.0, 8.0, True, bytearray.fromhex("0088")])
        replacement.add_row([9, 9, 9, 9, "99", "9", 9.0, 9.0, False, bytearray.fromhex("0099")])
        truth = self.df.clone()
        truth.set_row(2, replacement.get_row(0))
        truth.set_row(4, replacement.get_row(1))
        truth.set_row(5, replacement.get_row(2))
        self.df[:, (2, 4, 5)] = replacement
        self.assertTrue(self.df == truth, "DataFrames do not match")

        self.nulldf.add_rows(self.nulldf)
        replacement = DataFrame.like(self.nulldf)
        replacement.add_row([7, 7, None, 7, "77", "7", None, 7.0, False, None])
        replacement.add_row([8, None, 8, 8, None, "8", 8.0, None, True, bytearray.fromhex("0088")])
        replacement.add_row([None, 9, 9, None, "99", None, 9.0, None, False, None])
        truth = self.nulldf.clone()
        truth.set_row(0, replacement.get_row(0))
        truth.set_row(2, replacement.get_row(1))
        truth.set_row(4, replacement.get_row(2))
        self.nulldf[:, (0, 2, 4)] = replacement
        self.assertTrue(self.nulldf == truth, "DataFrames do not match")

    def test_setitem_set_multiple_rows_constant_value(self):
        self.df.add_rows(self.df)
        truth = self.df.clone()
        truth.get_columns(("stringCol", "charCol")).set_row(2, ["#", "#"])
        truth.get_columns(("stringCol", "charCol")).set_row(4, ["#", "#"])
        truth.get_columns(("stringCol", "charCol")).set_row(5, ["#", "#"])
        self.df[("stringCol", "charCol"), (2, 4, 5)] = "#"
        self.assertTrue(self.df == truth, "DataFrames do not match")
        truth.get_columns(("stringCol", "charCol")).set_row(2, ["#", "#"])
        truth.get_columns(("stringCol", "charCol")).set_row(4, ["-", "-"])
        truth.get_columns(("stringCol", "charCol")).set_row(5, [";", ";"])
        self.df[("stringCol", "charCol"), 4] = "-"
        self.df[("stringCol", "charCol"), 5] = ";"
        self.assertTrue(self.df == truth, "DataFrames do not match")

    def test_setitem_set_multiple_rows_dataframe_columns_select(self):
        self.df.add_rows(self.df)
        replacement = DataFrame.like(self.df.get_columns(cols=(3, 5, 8)))
        replacement.add_row([7, "7", False])
        replacement.add_row([8, "8", True])
        replacement.add_row([9, "9", False])
        truth = self.df.clone()
        truth.get_columns(cols=(3, 5, 8)).set_row(2, replacement.get_row(0))
        truth.get_columns(cols=(3, 5, 8)).set_row(3, replacement.get_row(1))
        truth.get_columns(cols=(3, 5, 8)).set_row(5, replacement.get_row(2))
        self.df[(3, 5, 8), (2, 3, 5)] = replacement
        self.assertTrue(self.df == truth, "DataFrames do not match")

        self.nulldf.add_rows(self.nulldf)
        replacement = DataFrame.like(self.nulldf.get_columns(cols=("stringCol", "doubleCol")))
        replacement.add_row(["77", 7.0])
        replacement.add_row(["88", 8.0])
        replacement.add_row(["99", 9.0])
        truth = self.nulldf.clone()
        truth.get_columns(cols=("stringCol", "doubleCol")).set_row(0, replacement.get_row(0))
        truth.get_columns(cols=("stringCol", "doubleCol")).set_row(2, replacement.get_row(1))
        truth.get_columns(cols=("stringCol", "doubleCol")).set_row(4, replacement.get_row(2))
        self.nulldf[("stringCol", "doubleCol"), (0, 2, 4)] = replacement
        self.assertTrue(self.nulldf == truth, "DataFrames do not match")

    def test_setitem_set_multiple_rows_dataframe_column_slice(self):
        self.df.add_rows(self.df)
        replacement = DataFrame.like(self.df.get_columns(cols=(2, 4, 6, 8)))
        replacement.add_row([7, "777", 7.0, False])
        replacement.add_row([8, "888", 8.0, True])
        truth = self.df.clone()
        truth.get_columns(cols=(2, 4, 6, 8)).set_row(2, replacement.get_row(0))
        truth.get_columns(cols=(2, 4, 6, 8)).set_row(4, replacement.get_row(1))
        self.df[2::2, (2, 4)] = replacement
        self.assertTrue(self.df == truth, "DataFrames do not match")

        self.nulldf.add_rows(self.nulldf)
        replacement = DataFrame.like(
            self.nulldf.get_columns(cols=("longCol", "stringCol", "charCol")))

        replacement.add_row([77, "77", "A"])
        replacement.add_row([88, "88", "B"])
        replacement.add_row([99, "99", "C"])
        truth = self.nulldf.clone()
        truth.get_columns(
            cols=("longCol", "stringCol", "charCol")).set_row(0, replacement.get_row(0))
        truth.get_columns(
            cols=("longCol", "stringCol", "charCol")).set_row(1, replacement.get_row(1))
        truth.get_columns(
            cols=("longCol", "stringCol", "charCol")).set_row(3, replacement.get_row(2))

        self.nulldf[3:6, (0, 1, 3)] = replacement
        self.assertTrue(self.nulldf == truth, "DataFrames do not match")



if __name__ == "__main__":
    unittest.main()
