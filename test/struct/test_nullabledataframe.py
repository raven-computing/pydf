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
Tests for NullableDataFrame implementation.
"""

import unittest
import math
import struct

from raven.struct.dataframe.core import (DataFrame,
                                         NullableDataFrame,
                                         DataFrameException)

from raven.struct.dataframe.bytecolumn import NullableByteColumn
from raven.struct.dataframe.shortcolumn import NullableShortColumn
from raven.struct.dataframe.intcolumn import NullableIntColumn
from raven.struct.dataframe.longcolumn import NullableLongColumn
from raven.struct.dataframe.floatcolumn import NullableFloatColumn
from raven.struct.dataframe.doublecolumn import NullableDoubleColumn
from raven.struct.dataframe.stringcolumn import NullableStringColumn
from raven.struct.dataframe.charcolumn import NullableCharColumn
from raven.struct.dataframe.booleancolumn import NullableBooleanColumn
from raven.struct.dataframe.binarycolumn import NullableBinaryColumn

# pylint: disable=too-many-lines
# pylint: disable=bad-whitespace, missing-function-docstring
# pylint: disable=consider-using-enumerate, invalid-name

class TestNullableDataFrame(unittest.TestCase):
    """Tests for NullableDataFrame implementation."""

    def assertSequenceAlmostEqual(self, first, second, msg):
        if len(first) != len(second):
            self.fail("Sequences have deviating lengths")

        for i in range(len(first)):
            if first[i] is None or second[i] is None:
                self.assertTrue(first[i] is None and second[i] is None,
                                "Values should both be None")
            else:
                self.assertAlmostEqual(first[i], second[i], places=5, msg=msg)

    def assertDataFrameIsSortedAscend(self):
        self.assertSequenceAlmostEqual(
            [1, 1, 1, 1, "1", "a", 1.0, 1.0, True, bytearray.fromhex("05")],
            self.toBeSorted.get_row(0),
            "Row does not match expected values at row index 0. DataFrame is not sorted correctly")

        self.assertSequenceAlmostEqual(
            [2, 2, 2, 2, "2", "b", 2.0, 2.0, False, bytearray.fromhex("0060")],
            self.toBeSorted.get_row(1),
            "Row does not match expected values at row index 1. DataFrame is not sorted correctly")

        self.assertSequenceAlmostEqual(
            [3, 3, 3, 3, "3", "c", 3.0, 3.0, True, bytearray.fromhex("000070")],
            self.toBeSorted.get_row(2),
            "Row does not match expected values at row index 2. DataFrame is not sorted correctly")

        self.assertSequenceAlmostEqual(
            [None, None, None, None, None, None, None, None, None, None],
            self.toBeSorted.get_row(3),
            ("Row does not match expected values at row index 3 "
             "(Should contain only None). DataFrame is not sorted correctly"))

        self.assertSequenceAlmostEqual(
            [None, None, None, None, None, None, None, None, None, None],
            self.toBeSorted.get_row(4),
            ("Row does not match expected values at row index 4 "
             "(Should contain only None). DataFrame is not sorted correctly"))

    def assertDataFrameIsSortedDescend(self):
        self.assertSequenceAlmostEqual(
            [3, 3, 3, 3, "3", "c", 3.0, 3.0, True, bytearray.fromhex("000070")],
            self.toBeSorted.get_row(0),
            "Row does not match expected values at row index 2. DataFrame is not sorted correctly")

        self.assertSequenceAlmostEqual(
            [2, 2, 2, 2, "2", "b", 2.0, 2.0, False, bytearray.fromhex("0060")],
            self.toBeSorted.get_row(1),
            "Row does not match expected values at row index 1. DataFrame is not sorted correctly")

        self.assertSequenceAlmostEqual(
            [1, 1, 1, 1, "1", "a", 1.0, 1.0, True, bytearray.fromhex("05")],
            self.toBeSorted.get_row(2),
            "Row does not match expected values at row index 0. DataFrame is not sorted correctly")

        self.assertSequenceAlmostEqual(
            [None, None, None, None, None, None, None, None, None, None],
            self.toBeSorted.get_row(3),
            ("Row does not match expected values at row index 3 "
             "(Should contain only None). DataFrame is not sorted correctly"))

        self.assertSequenceAlmostEqual(
            [None, None, None, None, None, None, None, None, None, None],
            self.toBeSorted.get_row(4),
            ("Row does not match expected values at row index 4 "
             "(Should contain only None). DataFrame is not sorted correctly"))

    def setUp(self):
        column_names = [
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

        self.df = NullableDataFrame(
            DataFrame.NullableByteColumn(column_names[0], [10,None,30,None,50]),
            DataFrame.NullableShortColumn(column_names[1], [11,None,31,None,51]),
            DataFrame.NullableIntColumn(column_names[2], [12,None,32,None,52]),
            DataFrame.NullableLongColumn(column_names[3], [13,None,33,None,53]),
            DataFrame.NullableStringColumn(column_names[4], ["10",None,"30",None,"50"]),
            DataFrame.NullableCharColumn(column_names[5], ['a',None,'c',None,'e']),
            DataFrame.NullableFloatColumn(column_names[6], [10.1,None,30.3,None,50.5]),
            DataFrame.NullableDoubleColumn(column_names[7], [11.1,None,31.3,None,51.5]),
            DataFrame.NullableBooleanColumn(column_names[8], [True,None,True,None,True]),
            DataFrame.NullableBinaryColumn(column_names[9], [bytearray.fromhex("05"),
                                                             None,
                                                             bytearray.fromhex("000070"),
                                                             None,
                                                             bytearray.fromhex("0000000090")])
            )

        self.toBeSorted = NullableDataFrame(
            DataFrame.NullableByteColumn(column_names[0], [None,2,1,None,3]),
            DataFrame.NullableShortColumn(column_names[1], [None,2,1,None,3]),
            DataFrame.NullableIntColumn(column_names[2], [None,2,1,None,3]),
            DataFrame.NullableLongColumn(column_names[3], [None,2,1,None,3]),
            DataFrame.NullableStringColumn(column_names[4], [None,"2","1",None,"3"]),
            DataFrame.NullableCharColumn(column_names[5], [None,'b','a',None,'c']),
            DataFrame.NullableFloatColumn(column_names[6], [None,2.0,1.0,None,3.0]),
            DataFrame.NullableDoubleColumn(column_names[7], [None,2.0,1.0,None,3.0]),
            DataFrame.NullableBooleanColumn(column_names[8], [None,False,True,None,True]),
            DataFrame.NullableBinaryColumn(column_names[9], [None,
                                                             bytearray.fromhex("0060"),
                                                             bytearray.fromhex("05"),
                                                             None,
                                                             bytearray.fromhex("000070")])
            )

        self.column_names = column_names
        self.column_types = [NullableByteColumn.TYPE_CODE,
                             NullableShortColumn.TYPE_CODE,
                             NullableIntColumn.TYPE_CODE,
                             NullableLongColumn.TYPE_CODE,
                             NullableFloatColumn.TYPE_CODE,
                             NullableDoubleColumn.TYPE_CODE,
                             NullableStringColumn.TYPE_CODE,
                             NullableCharColumn.TYPE_CODE,
                             NullableBooleanColumn.TYPE_CODE,
                             NullableBinaryColumn.TYPE_CODE]


    def test_constructor_no_args(self):
        test = NullableDataFrame()
        self.assertTrue(test.is_empty(), "NullableDataFrame should be empty")
        self.assertTrue(test.rows() == 0, "NullableDataFrame row count should be 0")
        self.assertTrue(test.columns() == 0, "NullableDataFrame column count should be 0")
        self.assertFalse(
            test.has_column_names(), "NullableDataFrame should not have column names set")
        self.assertTrue(isinstance(test, NullableDataFrame), "Is not NullableDataFrame type")

    def test_constructor_with_columns(self):
        test = NullableDataFrame(
            DataFrame.NullableIntColumn(values=[1,2,3]),
            DataFrame.NullableStringColumn(values=["1","2","3"]),
            DataFrame.NullableByteColumn(values=[1,2,3]))

        self.assertFalse(test.is_empty(), "NullableDataFrame should not be empty")
        self.assertTrue(test.rows() == 3, "NullableDataFrame row count should be 3")
        self.assertTrue(test.columns() == 3, "NullableDataFrame column count should be 3")
        self.assertFalse(
            test.has_column_names(), "NullableDataFrame should not have column names set")
        self.assertTrue(isinstance(test, NullableDataFrame), "Is not NullableDataFrame type")

    def test_constructor_with_labeled_columns(self):
        names = ["myInt","myString","myByte"]
        test = NullableDataFrame(
            DataFrame.NullableIntColumn(names[0], [1,2,3]),
            DataFrame.NullableStringColumn(names[1], ["1","2","3"]),
            DataFrame.NullableByteColumn(names[2], [1,2,3]))

        self.assertFalse(test.is_empty(), "NullableDataFrame should not be empty")
        self.assertTrue(test.rows() == 3, "NullableDataFrame row count should be 3")
        self.assertTrue(test.columns() == 3, "NullableDataFrame column count should be 3")
        self.assertTrue(
            test.has_column_names(), "NullableDataFrame should have column names set")
        self.assertEqual(
            names, test.get_column_names(), "NullableDataFrame column names do not match")
        self.assertTrue(isinstance(test, NullableDataFrame), "Is not NullableDataFrame type")



    #**********************#
    #        Getters       #
    #**********************#



    def test_get_byte_by_index(self):
        self.assertTrue(self.df.get_byte(0, 2) == 30, "Byte at index 2 should be 30")

    def test_get_byte_by_name(self):
        self.assertTrue(self.df.get_byte("byteCol", 2) == 30, "Byte at index 2 should be 30")

    def test_get_short_by_index(self):
        self.assertTrue(self.df.get_short(1, 3) is None, "Short at index 3 should be None")

    def test_get_short_by_name(self):
        self.assertTrue(
            self.df.get_short("shortCol", 3) is None, "Short at index 3 should be None")

    def test_get_int_by_index(self):
        self.assertTrue(self.df.get_int(2, 1) is None, "Int at index 1 should be None")

    def test_get_int_by_name(self):
        self.assertTrue(self.df.get_int("intCol", 1) is None, "Int at index 1 should be None")

    def test_get_long_by_index(self):
        self.assertTrue(self.df.get_long(3, 4) == 53, "Long at index 4 should be 53")

    def test_get_long_by_name(self):
        self.assertTrue(self.df.get_long("longCol", 4) == 53, "Long at index 4 should be 53")

    def test_get_string_by_index(self):
        self.assertTrue(self.df.get_string(4, 0) == "10", "String at index 0 should be \"10\"")

    def test_get_string_by_name(self):
        self.assertTrue(
            self.df.get_string("stringCol", 0) == "10", "String at index 0 should be \"10\"")

    def test_get_char_by_index(self):
        self.assertTrue(self.df.get_char(5, 2) == 'c', "Char at index 2 should be \'c\'")

    def test_get_char_by_name(self):
        self.assertTrue(self.df.get_char("charCol", 2) == 'c', "Char at index 2 should be \'c\'")

    def test_get_float_by_index(self):
        self.assertTrue(self.df.get_float(6, 1) is None, "Float at index 1 should be None")

    def test_get_float_by_name(self):
        self.assertTrue(
            self.df.get_float("floatCol", 1) is None, "Float at index 1 should be None")

    def test_get_double_by_index(self):
        self.assertAlmostEqual(
            self.df.get_double(7, 4), 51.5, places=5, msg="Double at index 4 should be 51.5")

    def test_get_double_by_name(self):
        self.assertAlmostEqual(
            self.df.get_double("doubleCol", 4), 51.5,
            places=5, msg="Double at index 4 should be 51.5")

    def test_get_boolean_by_index(self):
        self.assertTrue(self.df.get_boolean(8, 1) is None, "Boolean at index 1 should be None")

    def test_get_boolean_by_name(self):
        self.assertTrue(
            self.df.get_boolean("booleanCol", 1) is None, "Boolean at index 1 should be None")

    def test_get_binary_by_index(self):
        self.assertTrue(self.df.get_binary(9, 1) is None, "Binary at index 1 should be None")

    def test_get_binary_by_name(self):
        self.assertTrue(
            self.df.get_binary("binaryCol", 0) == bytearray.fromhex('05'),
            "Binary at index 1 should be 0x05")



    #**********************#
    #        Setters       #
    #**********************#



    def test_set_byte_by_index(self):
        self.df.set_byte(0, 2, 35)
        self.assertTrue(self.df.get_byte(0, 2) == 35, "Byte at index 2 should be set to 35")

    def test_set_byte_by_name(self):
        self.df.set_byte("byteCol", 2, 35)
        self.assertTrue(
            self.df.get_byte("byteCol", 2) == 35, "Byte at index 2 should be set to 35")

    def testSet_short_by_index(self):
        self.df.set_short(1, 3, 11)
        self.assertTrue(
            self.df.get_short(1, 3) == 11, "Short at index 3 should be set to 11")

    def test_set_short_by_name(self):
        self.df.set_short("shortCol", 3, 11)
        self.assertTrue(
            self.df.get_short("shortCol", 3) == 11, "Short at index 3 should be set to 11")

    def test_set_int_by_index(self):
        self.df.set_int(2, 1, 11)
        self.assertTrue(
            self.df.get_int(2, 1) == 11, "Int at index 1 should be set to 11")

    def test_set_int_by_name(self):
        self.df.set_int("intCol", 1, 11)
        self.assertTrue(
            self.df.get_int("intCol", 1) == 11, "Int at index 1 should be set to 11")

    def test_set_long_by_index(self):
        self.df.set_long(3, 4, 11)
        self.assertTrue(
            self.df.get_long(3, 4) == 11, "Long at index 4 should be set to 11")

    def test_set_long_by_name(self):
        self.df.set_long("longCol", 4, 11)
        self.assertTrue(
            self.df.get_long("longCol", 4) == 11, "Long at index 4 should be set to 11")

    def test_set_string_by_index(self):
        self.df.set_string(4, 0, "coffee")
        self.assertTrue(
            self.df.get_string(4, 0) == "coffee", "String at index 0 should be set to \"coffee\"")

    def test_set_string_by_name(self):
        self.df.set_string("stringCol", 0, "coffee")
        self.assertTrue(
            self.df.get_string("stringCol", 0) == "coffee",
            "String at index 0 should be set to \"coffee\"")

    def test_set_char_by_index(self):
        self.df.set_char(5, 2, 'T')
        self.assertTrue(self.df.get_char(5, 2) == 'T', "Char at index 2 should be set to \'T\'")

    def test_set_char_by_name(self):
        self.df.set_char("charCol", 2, 'T')
        self.assertTrue(
            self.df.get_char("charCol", 2) == 'T', "Char at index 2 should be set to \'T\'")

    def test_set_float_by_index(self):
        self.df.set_float(6, 1, 11.2)
        self.assertAlmostEqual(
            self.df.get_float(6, 1), 11.2, places=5, msg="Float at index 1 should be 11.2")

    def test_set_float_by_name(self):
        self.df.set_float("floatCol", 1, 11.2)
        self.assertAlmostEqual(
            self.df.get_float("floatCol", 1), 11.2,
            places=5, msg="Float at index 1 should be 11.2")

    def test_set_double_by_index(self):
        self.df.set_double(7, 4, 11.3)
        self.assertAlmostEqual(
            self.df.get_double(7, 4), 11.3, places=5, msg="Double at index 4 should be 11.3")

    def test_set_double_by_name(self):
        self.df.set_double("doubleCol", 4, 11.3)
        self.assertAlmostEqual(
            self.df.get_double("doubleCol", 4), 11.3,
            places=5, msg="Double at index 4 should be 11.3")

    def test_set_boolean_by_index(self):
        self.df.set_boolean(8, 1, True)
        val = self.df.get_boolean(8, 1)
        self.assertTrue(isinstance(val, bool), "Value should be a boolean")
        self.assertTrue(val, "Boolean at index 1 should be set to True")

    def test_set_boolean_by_name(self):
        self.df.set_boolean("booleanCol", 1, True)
        val = self.df.get_boolean("booleanCol", 1)
        self.assertTrue(isinstance(val, bool), "Value should be a boolean")
        self.assertTrue(val, "Boolean at index 1 should be set to True")

    def test_set_binary_by_index(self):
        self.df.set_binary(9, 1, bytearray.fromhex('abcd'))
        val = self.df.get_binary(9, 1)
        self.assertTrue(isinstance(val, bytearray), "Value should be a bytearray")
        self.assertTrue(
            val == bytearray.fromhex('abcd'), "Binary at index 1 should be set to 0xabcd")

    def test_set_binary_by_name(self):
        self.df.set_binary("binaryCol", 1, bytearray.fromhex('abcd'))
        val = self.df.get_binary(9, 1)
        self.assertTrue(isinstance(val, bytearray), "Value should be a bytearray")
        self.assertTrue(
            val == bytearray.fromhex('abcd'), "Binary at index 1 should be set to 0xabcd")



    #**********************************#
    #    Column names and indices      #
    #**********************************#



    def test_get_column_names(self):
        names = self.df.get_column_names()
        self.assertTrue(len(names) == 10, "Array of column names should have length 10")
        self.assertSequenceEqual(
            self.column_names, names, "Column names do not match array content")

    def test_get_column_name(self):
        self.assertEqual(
            "longCol",
            self.df.get_column_name(3),
            "Column name for column at index 3 does not equal \"longCol\"")

        self.assertEqual(
            "longCol",
            self.df.get_column(3).get_name(),
            "Column name for column at index 3 does not equal \"longCol\"")

    def test_get_column_index(self):
        self.assertTrue(
            self.df.get_column_index("longCol") == 3, "Column \"longCol\" is not at index 3")

    def test_set_column_names(self):
        names = ["A","B","C","D","E","F","G","H","I","J"]
        self.df.set_column_names(names)
        self.assertSequenceEqual(
            names,
            self.df.get_column_names(),
            "Column names do not match set names")

        self.assertTrue(self.df.has_column_names(), "Test-DataFrame should have column names set")
        for i in range(self.df.columns()):
            col = self.df.get_column(i)
            self.assertEqual(names[i], col.get_name(), "Column name does not match")

    def test_set_column_names_varargs(self):
        names = ["A","B","C","D","E","F","G","H","I","J"]
        self.df.set_column_names("A","B","C","D","E","F","G","H","I","J")
        self.assertSequenceEqual(
            names,
            self.df.get_column_names(),
            "Column names do not match set names")

        self.assertTrue(
            self.df.has_column_names(), "Test-DataFrame should have column names set")

        for i in range(self.df.columns()):
            col = self.df.get_column(i)
            self.assertEqual(names[i], col.get_name(), "Column name does not match")

    def test_set_column_name(self):
        self.df.set_column_name(3, "NEW_NAME")
        self.assertEqual(
            "NEW_NAME", self.df.get_column_name(3),
            "Column name does not match set name \"NEW_NAME\"")

        self.assertEqual(
            "NEW_NAME", self.df.get_column(3).get_name(),
            "Column name does not match set name \"NEW_NAME\"")

        self.assertTrue(
            self.df.has_column_names(), "Test-DataFrame should have column names set")

    def test_rename_column(self):
        self.df.set_column_name("longCol", "NEW_NAME")
        self.assertEqual(
            "NEW_NAME", self.df.get_column_name(self.df.get_column_index("NEW_NAME")),
            "Column name does not match set name \"NEW_NAME\"")

        self.assertEqual(
            "NEW_NAME", self.df.get_column(self.df.get_column_index("NEW_NAME")).get_name(),
            "Column name does not match set name \"NEW_NAME\"")

        self.assertTrue(
            self.df.has_column_names(),
            "Test-DataFrame should have column names set")

    def test_remove_column_names(self):
        self.df.remove_column_names()
        self.assertFalse(
            self.df.has_column_names(), "Test-DataFrame should not have column names set")

        for i in range(self.df.columns()):
            col = self.df.get_column(i)
            self.assertTrue(col.get_name() is None, "Column should not have a name set")

    def test_has_column_names(self):
        d = NullableDataFrame()
        self.assertFalse(d.has_column_names(), "Empty DataFrame should not have column names set")
        self.assertTrue(self.df.has_column_names(), "Test-DataFrame should have column names set")

        self.df.remove_column_names()
        self.assertFalse(
            self.df.has_column_names(),
            "Test-DataFrame should not have column names set after removal")



    #***************************#
    #           Rows            #
    #***************************#



    def test_get_row(self):
        row = self.df.get_row(0)
        self.assertSequenceAlmostEqual(
            (10,11,12,13,"10","a",10.1,11.1,True,bytearray.fromhex("05")),
            row, "Row does not match set values")

        row = self.df.get_row(1)
        self.assertSequenceAlmostEqual(
            (None,None,None,None,None,None,None,None,None,None),
            row, "Row does not match set values")

    def test_get_rows(self):
        res = self.df.get_rows(1, 3)
        self.assertTrue(res.rows() == 2, "DataFrame should have 2 rows")
        self.assertTrue(res.columns() == 10, "DataFrame should have 10 columns")
        self.assertSequenceAlmostEqual(
            [None,None,None,None,None,None,None,None,None,None],
            res.get_row(0), "Row does not match selected values")

        self.assertSequenceAlmostEqual(
            [30,31,32,33,"30","c",30.3,31.3,True,bytearray.fromhex("000070")],
            res.get_row(1), "Row does not match selected values")

    def test_set_row(self):
        row = (42,42,None,42,"42","A",42.2,None,True,None)
        self.df.set_row(1, row)
        self.assertSequenceAlmostEqual(
            self.df.get_row(1), row, "Row does not match set values")

    def test_add_row(self):
        self.df.add_row((42,42,None,42,"42","A",42.2,None,True,None))
        self.assertTrue(self.df.rows() == 6, "Row count should be 6")
        row = self.df.get_row(5)
        self.assertSequenceAlmostEqual(
            (42,42,None,42,"42","A",42.2,None,True,None),
            row, "Row does not match added values")

    def test_add_row_invalid_char(self):
        self.assertRaises(
            DataFrameException, self.df.add_row, [42,None,42,42,"42","€",42.2,None,True])

    def test_insert_row(self):
        self.df.insert_row(2, (42,42,None,42,"42","A",42.2,None,True,None))
        self.assertTrue(self.df.rows() == 6, "Row count should be 6")
        row = self.df.get_row(2)
        self.assertSequenceAlmostEqual(
            (42,42,None,42,"42",'A',42.2,None,True,None),
            row, "Row does not match inserted values")

    def test_insert_row_zero(self):
        self.df.insert_row(0, (42,42,None,42,"42","A",42.2,None,True,None))
        self.assertTrue(self.df.rows() == 6, "Row count should be 6")
        row = self.df.get_row(0)
        self.assertSequenceAlmostEqual(
            (42,42,None,42,"42",'A',42.2,None,True,None),
            row, "Row does not match inserted values")

    def test_insert_row_end(self):
        self.df.insert_row(5, (42,42,None,42,"42","A",42.2,None,True,None))
        self.assertTrue(self.df.rows() == 6, "Row count should be 6")
        row = self.df.get_row(5)
        self.assertSequenceAlmostEqual(
            (42,42,None,42,"42","A",42.2,None,True,None),
            row, "Row does not match inserted values")

    def test_insert_row_invalid_char(self):
        self.assertRaises(
            DataFrameException, self.df.insert_row, 1,
            [42,42,None,42,"42","€",42.2,None,True])

    def test_remove_row(self):
        self.df.remove_row(1)
        self.assertTrue(self.df.rows() == 4, "Row count should be 4")
        row = self.df.get_row(1)
        self.assertSequenceAlmostEqual(
            (30,31,32,33,"30","c",30.3,31.3,True,bytearray.fromhex('000070')),
            row, "Row does not match expected values")

    def test_remove_rows(self):
        self.df.remove_rows(from_index=1, to_index=3)
        self.assertTrue(self.df.rows() == 3, "Row count should be 3")
        row = self.df.get_row(1)
        self.assertSequenceAlmostEqual(
            (None,None,None,None,None,None,None,None,None,None),
            row, "Row does not match expected values after removal point")

        row = self.df.get_row(0)
        self.assertSequenceAlmostEqual(
            (10,11,12,13,"10","a",10.1,11.1,True,bytearray.fromhex("05")),
            row, "Row does not match expected values before removal point")













    def test_remove_rows_regex_match(self):
        removed = self.df.remove_rows(2, "(1|3)2")
        self.assertTrue(removed == 2, "Remove count should be 2")
        self.assertTrue(self.df.rows() == 3, "Row count should be 3")
        removed = self.df.remove_rows(8, "None")
        self.assertTrue(removed == 2, "Remove count should be 2")
        self.assertTrue(self.df.rows() == 1, "Row count should be 1")
        row = self.df.get_row(0)
        self.assertSequenceAlmostEqual(
            [50,51,52,53,"50","e",50.5,51.5,True,bytearray.fromhex("0000000090")],
            row, "Row does not match remaining values")

    def test_remove_rows_regex_match_by_name(self):
        removed = self.df.remove_rows("intCol", "(1|3)2")
        self.assertTrue(removed == 2, "Remove count should be 2")
        self.assertTrue(self.df.rows() == 3, "Row count should be 3")
        removed = self.df.remove_rows("booleanCol", "None")
        self.assertTrue(removed == 2, "Remove count should be 2")
        self.assertTrue(self.df.rows() == 1, "Row count should be 1")
        row = self.df.get_row(0)
        self.assertSequenceAlmostEqual(
            [50,51,52,53,"50","e",50.5,51.5,True,bytearray.fromhex("0000000090")],
            row, "Row count should be 1")

    def test_remove_rows_null_regex_match(self):
        r = self.df.remove_rows("intCol", "None")
        self.assertTrue(r == 2, "Return value should be 2")
        self.assertTrue(self.df.rows() == 3, "Returned DataFrame should have 3 rows")
        self.assertTrue(self.df.columns() == 10, "Returned DataFrame should have 10 columns")
        self.assertTrue(self.df.get_int("intCol", 0) == 12, "Value should be 12")
        self.assertTrue(self.df.get_int("intCol", 1) == 32, "Value should be 32")
        self.assertTrue(self.df.get_int("intCol", 2) == 52, "Value should be 52")

    def test_add_rows(self):
        df2 = NullableDataFrame(
            NullableByteColumn(values=[11,22]),
            NullableShortColumn(values=[11,22]),
            NullableIntColumn(values=[11,22]),
            NullableLongColumn(values=[11,22]),
            NullableStringColumn(values=["11","22"]),
            NullableCharColumn(values=["A","B"]),
            NullableFloatColumn(values=[11.1,22.2]),
            NullableDoubleColumn(values=[11.1,22.2]),
            NullableBooleanColumn(values=[True,False]),
            NullableBinaryColumn(values=[bytearray.fromhex("11"),bytearray.fromhex("22")]))

        df2.set_column_names(self.column_names)

        self.df.add_rows(df2)
        self.assertTrue(self.df.rows() == 7, "DataFrame should have 7 rows")
        self.assertTrue(self.df.columns() == 10, "DataFrame should have 10 columns")
        self.assertSequenceAlmostEqual(self.df.get_row(5), df2.get_row(0), "Rows do not match")
        self.assertSequenceAlmostEqual(self.df.get_row(6), df2.get_row(1), "Rows do not match")
        df2 = DataFrame.convert_to(df2, "DefaultDataFrame")
        self.df.add_rows(df2)
        self.assertTrue(self.df.rows() == 9, "DataFrame should have 9 rows")
        self.assertTrue(self.df.columns() == 10, "DataFrame should have 10 columns")
        self.assertSequenceAlmostEqual(self.df.get_row(7), df2.get_row(0), "Rows do not match")
        self.assertSequenceAlmostEqual(self.df.get_row(8), df2.get_row(1), "Rows do not match")

    def test_add_rows_shuffled_labels(self):
        names = ["longCol",   # 3
                 "intCol",    # 2
                 "booleanCol",# 8
                 "charCol",   # 5
                 "floatCol",  # 6
                 "shortCol",  # 1
                 "stringCol", # 4
                 "byteCol",   # 0
                 "doubleCol", # 7
                 "binaryCol"  # 9
                ]

        df2 = NullableDataFrame(
            NullableLongColumn(values=[11,22]),
            NullableIntColumn(values=[11,22]),
            NullableBooleanColumn(values=[True,False]),
            NullableCharColumn(values=["A","B"]),
            NullableFloatColumn(values=[11.1,22.2]),
            NullableShortColumn(values=[11,22]),
            NullableStringColumn(values=["11","22"]),
            NullableByteColumn(values=[11,22]),
            NullableDoubleColumn(values=[11.1,22.2]),
            NullableBinaryColumn(values=[bytearray.fromhex("11"),bytearray.fromhex("22")]))

        df2.set_column_names(names)

        self.df.add_rows(df2)
        self.assertTrue(self.df.rows() == 7, "DataFrame should have 7 rows")
        self.assertTrue(self.df.columns() == 10, "DataFrame should have 10 columns")
        self.assertSequenceAlmostEqual(
            self.df.get_row(5),
            [11,11,11,11,"11","A",11.1,11.1,True,bytearray.fromhex("11")], "Rows do not match")

        self.assertSequenceAlmostEqual(
            self.df.get_row(6),
            [22,22,22,22,"22","B",22.2,22.2,False,bytearray.fromhex("22")], "Rows do not match")

        df2 = DataFrame.convert_to(df2, "DefaultDataFrame")
        self.df.add_rows(df2)
        self.assertTrue(self.df.rows() == 9, "DataFrame should have 9 rows")
        self.assertTrue(self.df.columns() == 10, "DataFrame should have 10 columns")
        self.assertSequenceAlmostEqual(
            self.df.get_row(7), [11,11,11,11,"11","A",11.1,11.1,True,bytearray.fromhex("11")],
            "Rows do not match")

        self.assertSequenceAlmostEqual(
            self.df.get_row(8), [22,22,22,22,"22","B",22.2,22.2,False,bytearray.fromhex("22")],
            "Rows do not match")

    def test_add_rows_unlabeled(self):
        df2 = NullableDataFrame(
            NullableByteColumn(values=[11,22]),
            NullableShortColumn(values=[11,22]),
            NullableIntColumn(values=[11,22]),
            NullableLongColumn(values=[11,22]),
            NullableStringColumn(values=["11","22"]),
            NullableCharColumn(values=["A","B"]),
            NullableFloatColumn(values=[11.1,22.2]),
            NullableDoubleColumn(values=[11.1,22.2]),
            NullableBooleanColumn(values=[True,False]),
            NullableBinaryColumn(values=[bytearray.fromhex("11"),bytearray.fromhex("22")]))

        self.df.add_rows(df2)
        self.assertTrue(self.df.rows() == 7, "DataFrame should have 7 rows")
        self.assertTrue(self.df.columns() == 10, "DataFrame should have 10 columns")
        self.assertSequenceAlmostEqual(self.df.get_row(5), df2.get_row(0), "Rows do not match")
        self.assertSequenceAlmostEqual(self.df.get_row(6), df2.get_row(1), "Rows do not match")
        df2 = DataFrame.convert_to(df2, "DefaultDataFrame")
        self.df.add_rows(df2)
        self.assertTrue(self.df.rows() == 9, "DataFrame should have 9 rows")
        self.assertTrue(self.df.columns() == 10, "DataFrame should have 10 columns")
        self.assertSequenceAlmostEqual(self.df.get_row(7), df2.get_row(0), "Rows do not match")
        self.assertSequenceAlmostEqual(self.df.get_row(8), df2.get_row(1), "Rows do not match")

    def test_add_rows_unlabeled_fraction(self):
        df2 = NullableDataFrame(
            NullableByteColumn(values=[11,22]),
            NullableShortColumn(values=[11,22]),
            NullableIntColumn(values=[11,22]))

        self.df.add_rows(df2)
        self.assertTrue(self.df.rows() == 7, "DataFrame should have 7 rows")
        self.assertTrue(self.df.columns() == 10, "DataFrame should have 10 columns")
        self.assertSequenceAlmostEqual(
            self.df.get_row(5),
            [11,11,11,None,None,None,None,None,None,None],
            "Rows do not match")

        self.assertSequenceAlmostEqual(
            self.df.get_row(6),
            [22,22,22,None,None,None,None,None,None,None],
            "Rows do not match")

        df2 = DataFrame.convert_to(df2, "DefaultDataFrame")
        self.df.add_rows(df2)
        self.assertTrue(self.df.rows() == 9, "DataFrame should have 9 rows")
        self.assertTrue(self.df.columns() == 10, "DataFrame should have 10 columns")
        self.assertSequenceAlmostEqual(
            self.df.get_row(7),
            [11,11,11,None,None,None,None,None,None,None],
            "Rows do not match")

        self.assertSequenceAlmostEqual(
            self.df.get_row(8),
            [22,22,22,None,None,None,None,None,None,None],
            "Rows do not match")

    def test_head(self):
        res = self.df.head()
        self.assertTrue(res.equals(self.df), "DataFrames should be equal")
        res = self.df.head(3)
        self.assertTrue(res.equals(self.df.get_rows(0, 3)), "DataFrames should be equal")
        res = self.df.head(9999)
        self.assertTrue(res.equals(self.df), "DataFrames should be equal")
        res = self.df.head(0)
        self.df.clear()
        self.assertTrue(res.equals(self.df), "DataFrames should be equal")

    def test_head_uninitialized(self):
        res = NullableDataFrame().head()
        self.assertTrue(res.rows() == 0, "DataFrame should have 0 rows")
        self.assertTrue(res.columns() == 0, "DataFrame should have 0 columns")
        self.assertFalse(res.has_column_names(), "DataFrame should have no column names")

    def test_tail(self):
        res = self.df.tail()
        self.assertTrue(res.equals(self.df), "DataFrames should be equal")
        res = self.df.tail(3)
        self.assertTrue(res.equals(self.df.get_rows(2, 5)), "DataFrames should be equal")
        res = self.df.tail(9999)
        self.assertTrue(res.equals(self.df), "DataFrames should be equal")
        res = self.df.tail(0)
        self.df.clear()
        self.assertTrue(res.equals(self.df), "DataFrames should be equal")

    def test_tail_uninitialized(self):
        res = NullableDataFrame().tail()
        self.assertTrue(res.rows() == 0, "DataFrame should have 0 rows")
        self.assertTrue(res.columns() == 0, "DataFrame should have 0 columns")
        self.assertFalse(res.has_column_names(), "DataFrame should have no column names")

    def test_head_invalid_arg(self):
        self.assertRaises(DataFrameException, self.df.head, -1)

    def test_tail_invalid_arg(self):
        self.assertRaises(DataFrameException, self.df.tail, -1)



    #***************************#
    #          Columns          #
    #***************************#



    def test_add_column(self):
        col = DataFrame.NullableIntColumn(values=[0,1,2,3,4])
        self.df.add_column(col)
        self.assertTrue(self.df.columns() == 11, "Column count should be 11")
        self.assertTrue(col is self.df.get_column(10), "Column reference should be the same")

    def test_add_column_with_name(self):
        col = DataFrame.NullableIntColumn("INT", [0,1,2,3,4])
        self.df.add_column(col)
        self.assertTrue(self.df.columns() == 11, "Column count should be 11")
        self.assertTrue(col is self.df.get_column(10), "Column reference should be the same")
        self.assertTrue(col is self.df.get_column("INT"), "Column reference should be the same")

    def test_remove_column_by_index(self):
        self.df.remove_column(3)
        self.assertTrue(self.df.columns() == 9, "Column count should be 9")
        self.assertTrue(
            isinstance(self.df.get_column(3), NullableStringColumn),
            "Column after removal point should be of type NullableStringColumn")

        self.assertTrue(
            isinstance(self.df.get_column(2), NullableIntColumn),
            "Column before removal point should be of type NullableIntColumn")

    def test_remove_column_by_name(self):
        self.df.remove_column("longCol")
        self.assertTrue(self.df.columns() == 9, "Column count should be 9")
        self.assertTrue(
            isinstance(self.df.get_column(3), NullableStringColumn),
            "Column after removal point should be of type NullableStringColumn")

        self.assertTrue(
            isinstance(self.df.get_column(2), NullableIntColumn),
            "Column before removal point should be of type NullableIntColumn")

    def test_remove_column_by_reference(self):
        col = self.df.get_column("floatCol")
        res = self.df.remove_column(col)
        self.assertTrue(res, "Column should be removed")
        self.assertTrue(self.df.columns() == 9, "Column count should be 9")
        self.assertTrue(self.df.rows() == 5, "Row count should be 5")
        names = self.df.get_column_names()
        self.assertSequenceAlmostEqual(
            ["byteCol","shortCol","intCol","longCol","stringCol",
             "charCol","doubleCol","booleanCol", "binaryCol"],
            names, "Column names do not match")

    def test_remove_column_by_reference_no_removal(self):
        col = NullableFloatColumn("TEST", self.df.rows())
        res = self.df.remove_column(col)
        self.assertFalse(res, "Column should not be removed")
        self.assertTrue(self.df.columns() == 10, "Column count should be 10")
        self.assertTrue(self.df.rows() == 5, "Row count should be 5")
        names = self.df.get_column_names()
        self.assertSequenceAlmostEqual(self.column_names, names, "Column names do not match")

    def test_insert_column(self):
        col = DataFrame.NullableIntColumn(values=[0,1,2,3,4])
        self.df.insert_column(2, col)
        self.assertTrue(self.df.columns() == 11, "Column count should be 11")
        self.assertTrue(col is self.df.get_column(2), "Column reference should be the same")
        self.assertTrue(
            isinstance(self.df.get_column(3), NullableIntColumn),
            "Column after insertion point should be of type NullableIntColumn")

        self.assertTrue(
            isinstance(self.df.get_column(1), NullableShortColumn),
            "Column before insertion point should be of type NullableShortColumn")

    def test_insert_column_with_name(self):
        col = DataFrame.NullableIntColumn("INT", [0,1,2,3,4])
        self.df.insert_column(2, col)
        self.assertTrue(self.df.columns() == 11, "Column count should be 11")
        self.assertTrue(col is self.df.get_column(2), "Column reference should be the same")
        self.assertTrue(col is self.df.get_column("INT"), "Column reference should be the same")
        self.assertTrue(
            isinstance(self.df.get_column(3), NullableIntColumn),
            "Column after insertion point should be of type NullableIntColumn")

        self.assertTrue(
            isinstance(self.df.get_column(1), NullableShortColumn),
            "Column before insertion point should be of type NullableShortColumn")

    def test_get_column(self):
        col = self.df.get_column(2)
        self.assertTrue(
            isinstance(col, NullableIntColumn),
            "Column at index 2 should be of type NullableIntColumn")

    def test_get_column_by_name(self):
        col = self.df.get_column("stringCol")
        self.assertTrue(
            isinstance(col, NullableStringColumn),
            "Column \"stringCol\" should be of type NullableStringColumn")

    def test_get_columns(self):
        res = self.df.get_columns(cols=(1, 3, 5, 8))
        self.assertTrue(res.columns() == 4, "DataFrame should have 4 columns")
        self.assertTrue(res.rows() == 5, "DataFrame should have 5 rows")
        self.assertTrue(isinstance(res, NullableDataFrame),
                        "DataFrame should be a NullableDataFrame")

        self.assertSequenceAlmostEqual(
            ["shortCol", "longCol", "charCol", "booleanCol"],
            res.get_column_names(),
            "Column names do not match")

        self.assertTrue(
            res.get_column(0) is self.df.get_column(1), "Column references do not match")
        self.assertTrue(
            res.get_column(1) is self.df.get_column(3), "Column references do not match")
        self.assertTrue(
            res.get_column(2) is self.df.get_column(5), "Column references do not match")
        self.assertTrue(
            res.get_column(3) is self.df.get_column(8), "Column references do not match")

    def test_get_columns_by_name(self):
        res = self.df.get_columns(cols=("shortCol", "longCol", "charCol", "booleanCol"))
        self.assertTrue(res.columns() == 4, "DataFrame should have 4 columns")
        self.assertTrue(res.rows() == 5, "DataFrame should have 5 rows")
        self.assertTrue(
            isinstance(res, NullableDataFrame), "DataFrame should be a NullableDataFrame")

        self.assertSequenceAlmostEqual(
            ["shortCol", "longCol", "charCol", "booleanCol"],
            res.get_column_names(),
            "Column names do not match")

        self.assertTrue(
            res.get_column(0) is self.df.get_column(1), "Column references do not match")
        self.assertTrue(
            res.get_column(1) is self.df.get_column(3), "Column references do not match")
        self.assertTrue(
            res.get_column(2) is self.df.get_column(5), "Column references do not match")
        self.assertTrue(
            res.get_column(3) is self.df.get_column(8), "Column references do not match")

    def test_get_columns_by_element_types(self):
        res = self.df.get_columns(types=("short", "long", "char", "boolean"))
        self.assertTrue(res.columns() == 4, "DataFrame should have 4 columns")
        self.assertTrue(res.rows() == 5, "DataFrame should have 5 rows")
        self.assertTrue(
            isinstance(res, NullableDataFrame), "DataFrame should be a NullableDataFrame")

        self.assertSequenceAlmostEqual(
            ["shortCol", "longCol", "charCol", "booleanCol"],
            res.get_column_names(),
            "Column names do not match")

        self.assertTrue(
            res.get_column(0) is self.df.get_column(1), "Column references do not match")
        self.assertTrue(
            res.get_column(1) is self.df.get_column(3), "Column references do not match")
        self.assertTrue(
            res.get_column(2) is self.df.get_column(5), "Column references do not match")
        self.assertTrue(
            res.get_column(3) is self.df.get_column(8), "Column references do not match")

    def test_get_columns_by_element_types_numeric_only(self):
        res = self.df.get_columns(types="number")
        self.assertTrue(res.columns() == 6, "DataFrame should have 6 columns")
        self.assertTrue(res.rows() == 5, "DataFrame should have 5 rows")
        self.assertTrue(
            isinstance(res, NullableDataFrame), "DataFrame should be a NullableDataFrame")

        self.assertSequenceAlmostEqual(
            ["byteCol", "shortCol", "intCol", "longCol", "floatCol", "doubleCol"],
            res.get_column_names(),
            "Column names do not match")

        self.assertTrue(
            res.get_column(0) is self.df.get_column(0), "Column references do not match")
        self.assertTrue(
            res.get_column(1) is self.df.get_column(1), "Column references do not match")
        self.assertTrue(
            res.get_column(2) is self.df.get_column(2), "Column references do not match")
        self.assertTrue(
            res.get_column(3) is self.df.get_column(3), "Column references do not match")
        self.assertTrue(
            res.get_column(4) is self.df.get_column(6), "Column references do not match")
        self.assertTrue(
            res.get_column(5) is self.df.get_column(7), "Column references do not match")

    def test_get_columns_from_empty_dataframe(self):
        self.df.clear()
        res = self.df.get_columns(cols=(0, 2, 5))
        self.assertTrue(res.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(res.rows() == 0, "DataFrame should have 0 rows")
        self.assertTrue(res.capacity() == self.df.capacity(), "Capacity does not match")
        self.assertTrue(
            isinstance(res, NullableDataFrame), "DataFrame should be a NullableDataFrame")

        self.assertTrue(
            res.get_column(0) is self.df.get_column(0), "Column references do not match")
        self.assertTrue(
            res.get_column(1) is self.df.get_column(2), "Column references do not match")
        self.assertTrue(
            res.get_column(2) is self.df.get_column(5), "Column references do not match")
        res = self.df.get_columns(cols=("byteCol", "intCol", "charCol"))
        self.assertTrue(res.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(res.rows() == 0, "DataFrame should have 0 rows")
        self.assertTrue(res.capacity() == self.df.capacity(), "Capacity does not match")
        self.assertTrue(
            isinstance(res, NullableDataFrame), "DataFrame should be a NullableDataFrame")
        self.assertTrue(
            res.get_column(0) is self.df.get_column(0), "Column references do not match")
        self.assertTrue(
            res.get_column(1) is self.df.get_column(2), "Column references do not match")
        self.assertTrue(
            res.get_column(2) is self.df.get_column(5), "Column references do not match")
        res = self.df.get_columns(types=("byte", "int", "char"))
        self.assertTrue(res.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(res.rows() == 0, "DataFrame should have 0 rows")
        self.assertTrue(res.capacity() == self.df.capacity(), "Capacity does not match")
        self.assertTrue(
            isinstance(res, NullableDataFrame), "DataFrame should be a NullableDataFrame")
        self.assertTrue(
            res.get_column(0) is self.df.get_column(0), "Column references do not match")
        self.assertTrue(
            res.get_column(1) is self.df.get_column(2), "Column references do not match")
        self.assertTrue(
            res.get_column(2) is self.df.get_column(5), "Column references do not match")

    def test_set_column(self):
        col = DataFrame.NullableIntColumn(values=[0,1,2,3,4])
        self.df.set_column(3, col)
        col2 = self.df.get_column(3)
        self.assertTrue(col is col2, "References to columns should match")
        self.assertTrue(self.df.columns() == 10, "Column count should be 10")

    def test_set_column_by_name(self):
        col = NullableIntColumn("shouldBeReplaced", [0,1,2,3,4])
        self.df.set_column("longCol", col)
        col2 = self.df.get_column(3)
        name1 = self.df.get_column_name(3)
        name2 = self.df.get_column("longCol").get_name()
        self.assertSequenceAlmostEqual(name1, "longCol", "Column names do not match")
        self.assertSequenceAlmostEqual(name2, "longCol", "Column names do not match")
        self.assertTrue(col is col2, "References to columns should match")
        self.assertTrue(self.df.columns() == 10, "Column count should be 10")

    def test_set_column_by_name_add(self):
        col = NullableIntColumn("shouldBeReplaced", [0,1,2,3,4])
        self.df.set_column("NEWCOL", col)
        col2 = self.df.get_column(self.df.columns()-1)
        name1 = self.df.get_column_name(self.df.columns()-1)
        name2 = self.df.get_column("NEWCOL").get_name()
        self.assertSequenceAlmostEqual(name1, "NEWCOL", "Column names do not match")
        self.assertSequenceAlmostEqual(name2, "NEWCOL", "Column names do not match")
        self.assertTrue(col is col2, "References to columns should match")
        self.assertTrue(self.df.columns() == 11, "Column count should be 11")

    def test_has_column(self):
        self.assertTrue(self.df.has_column("byteCol"), "Column should be present")
        self.assertTrue(self.df.has_column("booleanCol"), "Column should be present")
        self.assertFalse(self.df.has_column("NoByteCol"), "Column should not be present")
        self.assertFalse(self.df.has_column("NoBooleanCol"), "Column should not be present")



    #****************************************************************************************#
    #           Search, Filter, Drop, Include, Exclude, Replace, Factor and Contains         #
    #****************************************************************************************#



    def test_index_of(self):
        i = self.df.index_of(2, "52")
        self.assertTrue(i == 4, "Found index should be 4")
        i = self.df.index_of(2, "nothing")
        self.assertTrue(i == -1, "Returned index should be -1")

    def test_index_of_by_name(self):
        i = self.df.index_of("intCol", "52")
        self.assertTrue(i == 4, "Found index should be 4")
        i = self.df.index_of("intCol", "nothing")
        self.assertTrue(i == -1, "Returned index should be -1")

    def test_index_of_with_start_point(self):
        i = self.df.index_of(2, "52", start_from=2)
        self.assertTrue(i == 4, "Found index should be 4")
        i = self.df.index_of(2, "nothing", 2)
        self.assertTrue(i == -1, "Returned index should be -1")
        i = self.df.index_of(2, "12", 1)
        self.assertTrue(i == -1, "Returned index should be -1")

    def test_index_of_by_name_with_start_point(self):
        i = self.df.index_of("intCol", "52", start_from=2)
        self.assertTrue(i == 4, "Found index should be 4")
        i = self.df.index_of("intCol", "nothing", 2)
        self.assertTrue(i == -1, "Returned index should be -1")
        i = self.df.index_of("intCol", "12", 1)
        self.assertTrue(i == -1, "Returned index should be -1")

    def test_index_of_all(self):
        i = self.df.index_of_all(2, "[1-4]2")
        self.assertTrue(len(i) == 2, "Returned array should have length 2")
        truth = [0,2]
        self.assertSequenceEqual(
            truth, i, "Content of the returned array does not match expected values")

        i = self.df.index_of_all(2, "nothing")
        self.assertTrue(len(i) == 0, "Returned array should be empty")

    def test_index_of_all_by_name(self):
        i = self.df.index_of_all("intCol", "[1-4]2")
        self.assertTrue(len(i) == 2, "Returned array should have length 2")
        truth = [0,2]
        self.assertSequenceEqual(
            truth, i, "Content of the returned array does not match expected values")

        i = self.df.index_of_all("intCol", "nothing")
        self.assertTrue(len(i) == 0, "Returned array should be empty")

    def test_filter(self):
        filtered = self.df.filter(2, "[1-4]2")
        self.assertTrue(
            filtered is not None, "API violation: Returned DataFrame should not be None")

        self.assertFalse(filtered.is_empty(), "Returned DataFrame should not be empty")
        self.assertTrue(
            isinstance(filtered, NullableDataFrame),
            "Returned DataFrame should be of type NullableDataFrame")

        self.assertTrue(filtered.rows() == 2, "Returned DataFrame should have 2 rows")
        self.assertTrue(filtered.columns() == 10, "Returned DataFrame should have 10 columns")
        self.assertTrue(filtered.get_int("intCol", 1) == 32, "Int value should be 32")
        self.assertSequenceAlmostEqual(
            [10,11,12,13,"10","a",10.1,11.1,True,bytearray.fromhex("05")],
            filtered.get_row(0), "Row does not match expected values")

    def test_filter_by_name(self):
        filtered = self.df.filter("intCol", "[1-4]2")
        self.assertTrue(
            filtered is not None, "API violation: Returned DataFrame should not be null")

        self.assertFalse(filtered.is_empty(), "Returned DataFrame should not be empty")
        self.assertTrue(
            isinstance(filtered, NullableDataFrame),
            "Returned DataFrame should be of type NullableDataFrame")

        self.assertTrue(filtered.rows() == 2, "Returned DataFrame should have 2 rows")
        self.assertTrue(filtered.columns() == 10, "Returned DataFrame should have 10 columns")
        self.assertTrue(filtered.get_int("intCol", 1) == 32, "Int value should be 32")
        self.assertSequenceAlmostEqual(
            [10,11,12,13,"10","a",10.1,11.1,True,bytearray.fromhex("05")],
            filtered.get_row(0), "Row does not match expected values")

    def test_filter_no_match(self):
        filtered = self.df.filter(2, "[1-4]2Digit")
        self.assertTrue(
            filtered is not None, "API violation: Returned DataFrame should not be null")

        self.assertTrue(filtered.is_empty(), "Returned DataFrame should be empty")
        self.assertTrue(
            isinstance(filtered, NullableDataFrame),
            "Returned DataFrame should be of type NullableDataFrame")

        self.assertTrue(filtered.rows() == 0, "Returned DataFrame should have 0 rows")
        self.assertTrue(filtered.columns() == 10, "Returned DataFrame should have 10 columns")

    def test_filter_null_regex_match(self):
        filtered = self.df.filter("intCol", "None")
        self.assertTrue(
            filtered is not None, "API violation: Returned DataFrame should not be None")

        self.assertTrue(
            isinstance(filtered, NullableDataFrame),
            "Returned DataFrame should be of type NullableDataFrame")

        self.assertTrue(filtered.rows() == 2, "Returned DataFrame should have 2 rows")
        self.assertTrue(filtered.columns() == 10, "Returned DataFrame should have 10 columns")
        self.assertTrue(filtered.get_int("intCol", 0) is None, "Filtered value should be null")
        self.assertTrue(filtered.get_int("intCol", 1) is None, "Filtered value should be null")

    def test_drop(self):
        filtered = self.df.drop(2, "[1-3]2")
        self.assertTrue(filtered is not None,
                        "API violation: Returned DataFrame should not be None")

        self.assertFalse(filtered.is_empty(), "Returned DataFrame should not be empty")
        self.assertTrue(isinstance(filtered, NullableDataFrame),
                        "Returned DataFrame should be of type NullableDataFrame")

        self.assertTrue(filtered.rows() == 3, "Returned DataFrame should have 3 rows")
        self.assertTrue(filtered.columns() == 10, "Returned DataFrame should have 10 columns")
        self.assertTrue(filtered.get_int("intCol", 2) == 52, "Invalid value")
        self.assertSequenceAlmostEqual(
            [None,None,None,None,None,None,None,None,None,None],
            filtered.get_row(0),
            "Row does not match expected values")
        self.assertSequenceAlmostEqual(
            [None,None,None,None,None,None,None,None,None,None],
            filtered.get_row(1),
            "Row does not match expected values")
        self.assertSequenceAlmostEqual(
            [50,51,52,53,"50","e",50.5,51.5,True,bytearray.fromhex("0000000090")],
            filtered.get_row(2),
            "Row does not match expected values")

    def test_drop_by_name(self):
        filtered = self.df.drop("intCol", "[1-3]2")
        self.assertTrue(filtered is not None,
                        "API violation: Returned DataFrame should not be None")

        self.assertFalse(filtered.is_empty(), "Returned DataFrame should not be empty")
        self.assertTrue(isinstance(filtered, NullableDataFrame),
                        "Returned DataFrame should be of type NullableDataFrame")

        self.assertTrue(filtered.rows() == 3, "Returned DataFrame should have 3 rows")
        self.assertTrue(filtered.columns() == 10, "Returned DataFrame should have 10 columns")

        self.assertTrue(filtered.get_int("intCol", 2) == 52, "Invalid value")
        self.assertSequenceAlmostEqual(
            [None,None,None,None,None,None,None,None,None,None],
            filtered.get_row(0),
            "Row does not match expected values")
        self.assertSequenceAlmostEqual(
            [None,None,None,None,None,None,None,None,None,None],
            filtered.get_row(1),
            "Row does not match expected values")
        self.assertSequenceAlmostEqual(
            [50,51,52,53,"50","e",50.5,51.5,True,bytearray.fromhex("0000000090")],
            filtered.get_row(2),
            "Row does not match expected values")

    def test_drop_everything(self):
        filtered = self.df.drop(2, ".*")
        self.assertTrue(filtered is not None,
                        "API violation: Returned DataFrame should not be None")

        self.assertTrue(filtered.is_empty(), "Returned DataFrame should be empty")
        self.assertTrue(isinstance(filtered, NullableDataFrame),
                        "Returned DataFrame should be of type NullableDataFrame")

        self.assertTrue(filtered.rows() == 0, "Returned DataFrame should have 0 rows")
        self.assertTrue(filtered.columns() == 10, "Returned DataFrame should have 10 columns")

    def test_drop_null_regex_match(self):
        filtered = self.df.drop("intCol", "None")
        self.assertTrue(
            filtered is not None, "API violation: Returned DataFrame should not be None")

        self.assertTrue(
            isinstance(filtered, NullableDataFrame),
            "Returned DataFrame should be of type NullableDataFrame")

        self.assertTrue(filtered.rows() == 3, "Returned DataFrame should have 3 rows")
        self.assertTrue(filtered.columns() == 10, "Returned DataFrame should have 10 columns")
        self.assertTrue(filtered.get_int("intCol", 0) == 12, "Value should be 12")
        self.assertTrue(filtered.get_int("intCol", 1) == 32, "Value should be 32")
        self.assertTrue(filtered.get_int("intCol", 2) == 52, "Value should be 52")

    def test_include(self):
        self.df.include(2, "[1-4]2")
        self.assertTrue(self.df.rows() == 2, "DataFrame should have 2 rows")
        self.assertTrue(self.df.columns() == 10, "DataFrame should have 10 columns")
        self.assertTrue(self.df.get_int("intCol", 1) == 32, "Invalid value")
        self.assertSequenceAlmostEqual(
            [10,11,12,13,"10","a",10.1,11.1,True,bytearray.fromhex("05")],
            self.df.get_row(0),
            "Row does not match expected values")

    def test_include_by_name(self):
        self.df.include("intCol", "[1-4]2")
        self.assertTrue(self.df.rows() == 2, "DataFrame should have 2 rows")
        self.assertTrue(self.df.columns() == 10, "DataFrame should have 10 columns")
        self.assertTrue(self.df.get_int("intCol", 0) == 12, "Invalid value")
        self.assertSequenceAlmostEqual(
            [10,11,12,13,"10","a",10.1,11.1,True,bytearray.fromhex("05")],
            self.df.get_row(0),
            "Row does not match expected values")

    def test_include_null_regex_match(self):
        filtered = self.df.include("intCol", "None")
        self.assertTrue(
            filtered, "API violation: Returned DataFrame should not be None")

        self.assertTrue(
            isinstance(filtered, NullableDataFrame),
            "Returned DataFrame should be of type NullableDataFrame")

        self.assertTrue(filtered.rows() == 2, "Returned DataFrame should have 2 rows")
        self.assertTrue(filtered.columns() == 10, "Returned DataFrame should have 10 columns")
        self.assertTrue(filtered.get_int("intCol", 0) is None, "Value should be None")
        self.assertTrue(filtered.get_int("intCol", 1) is None, "Value should be None")

    def test_exclude(self):
        self.df.exclude(2, "[1-3]2")
        self.assertTrue(self.df.rows() == 3, "DataFrame should have 3 rows")
        self.assertTrue(self.df.columns() == 10, "DataFrame should have 10 columns")
        self.assertTrue(self.df.get_int("intCol", 2) == 52, "Invalid value")
        self.assertSequenceAlmostEqual(
            [None,None,None,None,None,None,None,None,None,None],
            self.df.get_row(0),
            "Row does not match expected values")
        self.assertSequenceAlmostEqual(
            [None,None,None,None,None,None,None,None,None,None],
            self.df.get_row(1),
            "Row does not match expected values")
        self.assertSequenceAlmostEqual(
            [50,51,52,53,"50","e",50.5,51.5,True,bytearray.fromhex("0000000090")],
            self.df.get_row(2),
            "Row does not match expected values")

    def test_exclude_by_name(self):
        self.df.exclude("intCol", "[1-3]2")
        self.assertTrue(self.df.rows() == 3, "DataFrame should have 3 rows")
        self.assertTrue(self.df.columns() == 10, "DataFrame should have 10 columns")
        self.assertTrue(self.df.get_int("intCol", 0) is None, "Invalid value")
        self.assertTrue(self.df.get_int("intCol", 1) is None, "Invalid value")
        self.assertTrue(self.df.get_int("intCol", 2) == 52, "Invalid value")
        self.assertSequenceAlmostEqual(
            [None,None,None,None,None,None,None,None,None,None],
            self.df.get_row(0),
            "Row does not match expected values")
        self.assertSequenceAlmostEqual(
            [None,None,None,None,None,None,None,None,None,None],
            self.df.get_row(1),
            "Row does not match expected values")
        self.assertSequenceAlmostEqual(
            [50,51,52,53,"50","e",50.5,51.5,True,bytearray.fromhex("0000000090")],
            self.df.get_row(2),
            "Row does not match expected values")

    def test_exclude_null_regex_match(self):
        filtered = self.df.exclude("intCol", "None")
        self.assertTrue(
            filtered is not None, "API violation: Returned DataFrame should not be None")

        self.assertTrue(
            isinstance(filtered, NullableDataFrame),
            "Returned DataFrame should be of type NullableDataFrame")

        self.assertTrue(filtered.rows() == 3, "Returned DataFrame should have 3 rows")
        self.assertTrue(filtered.columns() == 10, "Returned DataFrame should have 10 columns")
        self.assertTrue(filtered.get_int("intCol", 0) == 12, "Value should be 12")
        self.assertTrue(filtered.get_int("intCol", 1) == 32, "Value should be 32")
        self.assertTrue(filtered.get_int("intCol", 2) == 52, "Value should be 52")

    def test_replace(self):
        replaced_longs = self.df.replace(3, "(1|2|3)3", 666)
        replaced_strings = self.df.replace(4, "(4|5)0", "TEST")
        replaced_booleans = self.df.replace(8, "None", True)
        self.assertTrue(replaced_longs == 2, "Replaced number should be 2")
        self.assertTrue(replaced_strings == 1, "Replaced number should be 1")
        self.assertTrue(replaced_booleans == 2, "Replaced number should be 2")

        self.assertTrue(self.df.get_long(3, 0) == 666, "Value does not match replaced value")
        self.assertTrue(self.df.get_long(3, 1) is None, "Value does not match replaced value")
        self.assertTrue(self.df.get_long(3, 2) == 666, "Value does not match replaced value")
        self.assertTrue(self.df.get_long(3, 3) is None, "Value does not match replaced value")
        self.assertTrue(self.df.get_long(3, 4) == 53, "Value does not match replaced value")

        self.assertTrue(self.df.get_string(4, 0) == "10", "Value does not match replaced value")
        self.assertTrue(self.df.get_string(4, 1) is None, "Value does not match replaced value")
        self.assertTrue(self.df.get_string(4, 2) == "30", "Value does not match replaced value")
        self.assertTrue(self.df.get_string(4, 3) is None, "Value does not match replaced value")
        self.assertTrue(self.df.get_string(4, 4) == "TEST", "Value does not match replaced value")

        self.assertTrue(self.df.get_boolean(8, 0), "Value does not match replaced value")
        self.assertTrue(self.df.get_boolean(8, 1), "Value does not match replaced value")
        self.assertTrue(self.df.get_boolean(8, 2), "Value does not match replaced value")
        self.assertTrue(self.df.get_boolean(8, 3), "Value does not match replaced value")
        self.assertTrue(self.df.get_boolean(8, 4), "Value does not match replaced value")

    def test_replace_by_name(self):
        replaced_longs = self.df.replace("longCol", "(1|2|3)3", 666)
        replaced_strings = self.df.replace("stringCol", "(4|5)0", "TEST")
        replaced_booleans = self.df.replace("booleanCol", "False", True)
        self.assertTrue(replaced_longs == 2, "Replaced number should be 2")
        self.assertTrue(replaced_strings == 1, "Replaced number should be 1")
        self.assertTrue(replaced_booleans == 0, "Replaced number should be 0")

        self.assertTrue(
            self.df.get_long("longCol", 0) == 666, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_long("longCol", 1) is None, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_long("longCol", 2) == 666, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_long("longCol", 3) is None, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_long("longCol", 4) == 53, "Value does not match replaced value")

        self.assertTrue(
            self.df.get_string("stringCol", 0) == "10", "Value does not match replaced value")
        self.assertTrue(
            self.df.get_string("stringCol", 1) is None, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_string("stringCol", 2) == "30", "Value does not match replaced value")
        self.assertTrue(
            self.df.get_string("stringCol", 3) is None, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_string("stringCol", 4) == "TEST", "Value does not match replaced value")

        self.assertTrue(
            self.df.get_boolean("booleanCol", 0), "Value does not match replaced value")
        self.assertTrue(
            self.df.get_boolean("booleanCol", 1) is None, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_boolean("booleanCol", 2), "Value does not match replaced value")
        self.assertTrue(
            self.df.get_boolean("booleanCol", 3) is None, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_boolean("booleanCol", 4), "Value does not match replaced value")

    def test_replace_lambda(self):
        replaced_longs = self.df.replace(3, replacement=lambda i, v: i)
        replaced_strings = self.df.replace(4, replacement=lambda i, v: "TEST" + str(i))
        replaced_booleans = self.df.replace(8, replacement=lambda i, v: False)
        self.assertTrue(replaced_longs == 5, "Replaced number should be 5")
        self.assertTrue(replaced_strings == 5, "Replaced number should be 5")
        self.assertTrue(replaced_booleans == 5, "Replaced number should be 5")

        self.assertTrue(self.df.get_long(3, 0) == 0, "Value does not match replaced value")
        self.assertTrue(self.df.get_long(3, 1) == 1, "Value does not match replaced value")
        self.assertTrue(self.df.get_long(3, 2) == 2, "Value does not match replaced value")
        self.assertTrue(self.df.get_long(3, 3) == 3, "Value does not match replaced value")
        self.assertTrue(self.df.get_long(3, 4) == 4, "Value does not match replaced value")

        self.assertTrue(self.df.get_string(4, 0) == "TEST0", "Value does not match replaced value")
        self.assertTrue(self.df.get_string(4, 1) == "TEST1", "Value does not match replaced value")
        self.assertTrue(self.df.get_string(4, 2) == "TEST2", "Value does not match replaced value")
        self.assertTrue(self.df.get_string(4, 3) == "TEST3", "Value does not match replaced value")
        self.assertTrue(self.df.get_string(4, 4) == "TEST4", "Value does not match replaced value")

        self.assertFalse(self.df.get_boolean(8, 0), "Value does not match replaced value")
        self.assertFalse(self.df.get_boolean(8, 1), "Value does not match replaced value")
        self.assertFalse(self.df.get_boolean(8, 2), "Value does not match replaced value")
        self.assertFalse(self.df.get_boolean(8, 3), "Value does not match replaced value")
        self.assertFalse(self.df.get_boolean(8, 4), "Value does not match replaced value")

    def test_replace_by_name_lambda(self):
        replaced_longs = self.df.replace("longCol", replacement=lambda i, v: i)
        replaced_strings = self.df.replace("stringCol", replacement=lambda i, v: "TEST" + str(i))
        replaced_booleans = self.df.replace("booleanCol", replacement=lambda i, v: True)
        self.assertTrue(replaced_longs == 5, "Replaced number should be 5")
        self.assertTrue(replaced_strings == 5, "Replaced number should be 5")
        self.assertTrue(replaced_booleans == 2, "Replaced number should be 2")

        self.assertTrue(self.df.get_long("longCol", 0) == 0, "Value does not match replaced value")
        self.assertTrue(self.df.get_long("longCol", 1) == 1, "Value does not match replaced value")
        self.assertTrue(self.df.get_long("longCol", 2) == 2, "Value does not match replaced value")
        self.assertTrue(self.df.get_long("longCol", 3) == 3, "Value does not match replaced value")
        self.assertTrue(self.df.get_long("longCol", 4) == 4, "Value does not match replaced value")

        self.assertTrue(
            self.df.get_string("stringCol", 0) == "TEST0", "Value does not match replaced value")
        self.assertTrue(
            self.df.get_string("stringCol", 1) == "TEST1", "Value does not match replaced value")
        self.assertTrue(
            self.df.get_string("stringCol", 2) == "TEST2", "Value does not match replaced value")
        self.assertTrue(
            self.df.get_string("stringCol", 3) == "TEST3", "Value does not match replaced value")
        self.assertTrue(
            self.df.get_string("stringCol", 4) == "TEST4", "Value does not match replaced value")

        self.assertTrue(
            self.df.get_boolean("booleanCol", 0), "Value does not match replaced value")
        self.assertTrue(
            self.df.get_boolean("booleanCol", 1), "Value does not match replaced value")
        self.assertTrue(
            self.df.get_boolean("booleanCol", 2), "Value does not match replaced value")
        self.assertTrue(
            self.df.get_boolean("booleanCol", 3), "Value does not match replaced value")
        self.assertTrue(
            self.df.get_boolean("booleanCol", 4), "Value does not match replaced value")

    def test_replace_regex_lambda(self):
        replaced_longs = self.df.replace(3, "(1|2|3)3", lambda i, v: 666)
        replaced_strings = self.df.replace(4, "(4|5)0", lambda i, v: "TEST")
        replaced_booleans = self.df.replace(8, "False", lambda i, v: True)
        self.assertTrue(replaced_longs == 2, "Replaced number should be 2")
        self.assertTrue(replaced_strings == 1, "Replaced number should be 1")
        self.assertTrue(replaced_booleans == 0, "Replaced number should be 0")

        self.assertTrue(self.df.get_long(3, 0) == 666, "Value does not match replaced value")
        self.assertTrue(self.df.get_long(3, 1) is None, "Value does not match replaced value")
        self.assertTrue(self.df.get_long(3, 2) == 666, "Value does not match replaced value")
        self.assertTrue(self.df.get_long(3, 3) is None, "Value does not match replaced value")
        self.assertTrue(self.df.get_long(3, 4) == 53, "Value does not match replaced value")

        self.assertTrue(self.df.get_string(4, 0) == "10", "Value does not match replaced value")
        self.assertTrue(self.df.get_string(4, 1) is None, "Value does not match replaced value")
        self.assertTrue(self.df.get_string(4, 2) == "30", "Value does not match replaced value")
        self.assertTrue(self.df.get_string(4, 3) is None, "Value does not match replaced value")
        self.assertTrue(self.df.get_string(4, 4) == "TEST", "Value does not match replaced value")

        self.assertTrue(self.df.get_boolean(8, 0), "Value does not match replaced value")
        self.assertTrue(self.df.get_boolean(8, 1) is None, "Value does not match replaced value")
        self.assertTrue(self.df.get_boolean(8, 2), "Value does not match replaced value")
        self.assertTrue(self.df.get_boolean(8, 3) is None, "Value does not match replaced value")
        self.assertTrue(self.df.get_boolean(8, 4), "Value does not match replaced value")

    def test_replace_by_name_regex_lambda(self):
        replaced_longs = self.df.replace("longCol", "(1|2|3)3", lambda i, v: 666)
        replaced_strings = self.df.replace("stringCol", "(4|5)0", lambda i, v: "TEST")
        replaced_booleans = self.df.replace("booleanCol", "True", lambda i, v: False)
        self.assertTrue(replaced_longs == 2, "Replaced number should be 2")
        self.assertTrue(replaced_strings == 1, "Replaced number should be 1")
        self.assertTrue(replaced_booleans == 3, "Replaced number should be 3")

        self.assertTrue(
            self.df.get_long("longCol", 0) == 666, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_long("longCol", 1) is None, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_long("longCol", 2) == 666, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_long("longCol", 3) is None, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_long("longCol", 4) == 53, "Value does not match replaced value")

        self.assertTrue(
            self.df.get_string("stringCol", 0) == "10", "Value does not match replaced value")
        self.assertTrue(
            self.df.get_string("stringCol", 1) is None, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_string("stringCol", 2) == "30", "Value does not match replaced value")
        self.assertTrue(
            self.df.get_string("stringCol", 3) is None, "Value does not match replaced value")
        self.assertTrue(
            self.df.get_string("stringCol", 4) == "TEST", "Value does not match replaced value")

        self.assertFalse(
            self.df.get_boolean("booleanCol", 0), "Value does not match replaced value")
        self.assertTrue(
            self.df.get_boolean("booleanCol", 1) is None, "Value does not match replaced value")
        self.assertFalse(
            self.df.get_boolean("booleanCol", 2), "Value does not match replaced value")
        self.assertTrue(
            self.df.get_boolean("booleanCol", 3) is None, "Value does not match replaced value")
        self.assertFalse(
            self.df.get_boolean("booleanCol", 4), "Value does not match replaced value")

    def test_replace_dataframe(self):
        df2 = NullableDataFrame(
            NullableIntColumn(values=[44,44,44,44,44]),
            NullableFloatColumn(values=[44.4,44.4,44.4,44.4,44.4]),
            NullableDoubleColumn(values=[44.4,44.4,44.4,44.4,44.4]),
            NullableBooleanColumn(values=[False,True,False,True,False]))

        df2.set_column_names(["TEST1","floatCol","TEST2","booleanCol"])

        replaced = self.df.replace(df=df2)
        self.assertTrue(replaced == 2, "Replace count should be 2")
        self.assertTrue(
            self.df.get_column("floatCol") == df2.get_column("floatCol"),
            "Column reference does not match")

        self.assertTrue(
            self.df.get_column("booleanCol") == df2.get_column("booleanCol"),
            "Column reference does not match")

    def test_replace_dataframe_no_column_names(self):
        df2 = NullableDataFrame(
            NullableIntColumn(values=[44,44,44,44,44]),
            NullableFloatColumn(values=[44.4,44.4,44.4,44.4,44.4]),
            NullableDoubleColumn(values=[44.4,44.4,44.4,44.4,44.4]),
            NullableBooleanColumn(values=[False,True,False,True,False]))

        self.df.remove_column_names()
        replaced = self.df.replace(df=df2)
        self.assertTrue(replaced == 4, "Replace count should be 4")
        self.assertTrue(
            self.df.get_column(0) is df2.get_column(0),
            "Column reference does not match")

        self.assertTrue(
            self.df.get_column(1) is df2.get_column(1),
            "Column reference does not match")

        self.assertTrue(
            self.df.get_column(2) is df2.get_column(2),
            "Column reference does not match")

        self.assertTrue(
            self.df.get_column(3) is df2.get_column(3),
            "Column reference does not match")

    def test_factor(self):
        self.df.set_string(4, 0, self.df.get_string(4, 4))
        self.df.set_char(5, 0, self.df.get_char(5, 4))
        map1 = self.df.factor(4)
        map2 = self.df.factor(5)
        map3 = self.df.factor(8)
        self.assertTrue(len(map1) == 2, "Factor map should have a size of 2")
        self.assertTrue(len(map2) == 2, "Factor map should have a size of 2")
        self.assertTrue(len(map3) == 1, "Factor map should have a size of 1")
        self.assertTrue(
            self.df.get_column(4).type_code() == NullableIntColumn.TYPE_CODE,
            "Column should be an NullableIntColumn")

        self.assertTrue(
            self.df.get_column(5).type_code() == NullableIntColumn.TYPE_CODE,
            "Column should be an NullableIntColumn")

        self.assertTrue(
            self.df.get_column(8).type_code() == NullableIntColumn.TYPE_CODE,
            "Column should be an NullableIntColumn")

        self.assertSequenceAlmostEqual(
            [1,None,2,None,1],
            self.df.get_column(4).as_array(),
            "Column content does not match")

        self.assertSequenceAlmostEqual(
            [1,None,2,None,1],
            self.df.get_column(5).as_array(),
            "Column content does not match")

        self.assertSequenceAlmostEqual(
            [1,None,1,None,1],
            self.df.get_column(8).as_array(),
            "Column content does not match")

    def test_factor_by_name(self):
        self.df.set_string("stringCol", 0, self.df.get_string("stringCol", 4))
        self.df.set_char("charCol", 0, self.df.get_char("charCol", 4))
        map1 = self.df.factor("stringCol")
        map2 = self.df.factor("charCol")
        map3 = self.df.factor("booleanCol")
        self.assertTrue(len(map1) == 2, "Factor map should have a size of 2")
        self.assertTrue(len(map2) == 2, "Factor map should have a size of 2")
        self.assertTrue(len(map3) == 1, "Factor map should have a size of 1")
        self.assertTrue(
            self.df.get_column("stringCol").type_code() == NullableIntColumn.TYPE_CODE,
            "Column should be an NullableIntColumn")

        self.assertTrue(
            self.df.get_column("charCol").type_code() == NullableIntColumn.TYPE_CODE,
            "Column should be an NullableIntColumn")

        self.assertTrue(
            self.df.get_column("booleanCol").type_code() == NullableIntColumn.TYPE_CODE,
            "Column should be an NullableIntColumn")

        self.assertSequenceAlmostEqual(
            [1,None,2,None,1],
            self.df.get_column("stringCol").as_array(),
            "Column content does not match")

        self.assertSequenceAlmostEqual(
            [1,None,2,None,1],
            self.df.get_column("charCol").as_array(),
            "Column content does not match")

        self.assertSequenceAlmostEqual(
            [1,None,1,None,1],
            self.df.get_column("booleanCol").as_array(),
            "Column content does not match")

    def test_factor_numeric_column(self):
        map1 = self.df.factor("byteCol")
        self.assertTrue(not map1, "Factor map should be empty")
        map1 = self.df.factor("shortCol")
        self.assertTrue(not map1, "Factor map should be empty")
        map1 = self.df.factor("intCol")
        self.assertTrue(not map1, "Factor map should be empty")
        map1 = self.df.factor("longCol")
        self.assertTrue(not map1, "Factor map should be empty")
        map1 = self.df.factor("floatCol")
        self.assertTrue(not map1, "Factor map should be empty")
        map1 = self.df.factor("doubleCol")
        self.assertTrue(not map1, "Factor map should be empty")
        self.assertTrue(
            self.df.get_column("byteCol").type_code() == NullableByteColumn.TYPE_CODE,
            "Column should be a NullableByteColumn")
        self.assertTrue(
            self.df.get_column("shortCol").type_code() == NullableShortColumn.TYPE_CODE,
            "Column should be a NullableShortColumn")
        self.assertTrue(
            self.df.get_column("intCol").type_code() == NullableIntColumn.TYPE_CODE,
            "Column should be an NullableIntColumn")
        self.assertTrue(
            self.df.get_column("longCol").type_code() == NullableLongColumn.TYPE_CODE,
            "Column should be a NullableLongColumn")
        self.assertTrue(
            self.df.get_column("floatCol").type_code() == NullableFloatColumn.TYPE_CODE,
            "Column should be a NullableFloatColumn")
        self.assertTrue(
            self.df.get_column("doubleCol").type_code() == NullableDoubleColumn.TYPE_CODE,
            "Column should be a NullableDoubleColumn")

    def test_replace_fail_type(self):
        self.assertRaises(
            DataFrameException, self.df.replace, "longCol", "(1|2|3)3", "NOT_A_LONG")

    def test_replace_lambda_fail_type(self):
        self.assertRaises(
            DataFrameException,
            self.df.replace, "longCol", None, lambda i, v: "NOT_A_LONG")

    def test_replace_regex_lambda_fail_type(self):
        self.assertRaises(
            DataFrameException,
            self.df.replace, "longCol", "(1|2|3)3", lambda i, v: "NOT_A_LONG")

    def test_replace_identity(self):
        count = self.df.replace(3, replacement=lambda i, v: v)
        self.assertTrue(count == 0, "Replacement count should be zero")

    def test_replace_regex_identity(self):
        count = self.df.replace(3, "(1|2|3)3", lambda i, v: v)
        self.assertTrue(count == 0, "Replacement count should be zero")

    def test_contains(self):
        res = self.df.contains(3, "53")
        self.assertTrue(res, "Contains should return true")
        res = self.df.contains(8, "None")
        self.assertTrue(res, "Contains should return true")
        res = self.df.contains(8, "TEST")
        self.assertFalse(res, "Contains should return false")

    def test_contains_by_name(self):
        res = self.df.contains("longCol", "53")
        self.assertTrue(res, "Contains should return true")
        res = self.df.contains("booleanCol", "None")
        self.assertTrue(res, "Contains should return true")
        res = self.df.contains(8, "TEST")
        self.assertFalse(res, "Contains should return false")



    #**************************************************************#
    #           Count, CountUnique and Unique operations           #
    #**************************************************************#



    def test_count(self):
        count = self.df.count(4)
        self.assertTrue(count.rows() == 4, "Count should have 4 rows")
        self.assertTrue(count.columns() == 3, "Count should have 3 columns")
        self.assertTrue(count.sum(1) == self.df.rows(), "Counts sum is incorrect")
        self.assertTrue(
            isinstance(count.get_column(0), NullableStringColumn),
            "Value column should be a NullableStringColumn")

        self.assertTrue(
            isinstance(count.get_column(1), NullableIntColumn),
            "Count column should be an NullableIntColumn")

        self.assertTrue(
            isinstance(count.get_column(2), NullableFloatColumn),
            "Rate column should be a NullableFloatColumn")

        self.assertTrue(
            isinstance(count, NullableDataFrame),
            "Count DataFrame should be a NullableDataFrame")

        for i in range(count.rows()-1):
            self.assertTrue(
                count.get_int("count", i) == 1,
                "Value should have a count of 1")

        self.assertTrue(
            count.get_int("count", count.rows()-1) == 2,
            "None values should have a count of 2")

        count = self.df.count(8)
        self.assertTrue(count.rows() == 2, "Count should have 2 rows")
        self.assertTrue(count.columns() == 3, "Count should have 3 columns")
        self.assertTrue(count.sum(1) == self.df.rows(), "Counts sum is incorrect")
        self.assertTrue(
            isinstance(count.get_column(0), NullableBooleanColumn),
            "Value column should be a NullableBooleanColumn")

    def test_count_by_name(self):
        count = self.df.count("stringCol")
        self.assertTrue(count.rows() == 4, "Count should have 4 rows")
        self.assertTrue(count.columns() == 3, "Count should have 3 columns")
        self.assertTrue(count.sum("count") == self.df.rows(), "Counts sum is incorrect")
        self.assertTrue(
            isinstance(count.get_column(0), NullableStringColumn),
            "Value column should be a NullableStringColumn")

        self.assertTrue(
            isinstance(count.get_column(1), NullableIntColumn),
            "Count column should be an NullableIntColumn")

        self.assertTrue(
            isinstance(count.get_column(2), NullableFloatColumn),
            "Rate column should be a NullableFloatColumn")

        self.assertTrue(
            isinstance(count, NullableDataFrame),
            "Count DataFrame should be a NullableDataFrame")

        for i in range(count.rows()-1):
            self.assertTrue(
                count.get_int("count", i) == 1,
                "Value should have a count of 1")

        self.assertTrue(
            count.get_int("count", count.rows()-1) == 2,
            "None values should have a count of 2")

        count = self.df.count("booleanCol")
        self.assertTrue(count.rows() == 2, "Count should have 2 rows")
        self.assertTrue(count.columns() == 3, "Count should have 3 columns")
        self.assertTrue(count.sum("count") == self.df.rows(), "Counts sum is incorrect")
        self.assertTrue(
            isinstance(count.get_column(0), NullableBooleanColumn),
            "Value column should be a NullableBooleanColumn")

    def test_count_regex(self):
        count = self.df.count(2, "[1-4]2")
        self.assertTrue(count == 2, "Count should be 2")
        count = self.df.count(4, "NothingValid")
        self.assertTrue(count == 0, "Count should be 0")

    def test_count_regex_by_name(self):
        count = self.df.count("intCol", "[1-4]2")
        self.assertTrue(count == 2, "Count should be 2")
        count = self.df.count("stringCol", "NothingValid")
        self.assertTrue(count == 0, "Count should be 0")

    def test_count_null_regex(self):
        count = self.df.count("intCol", "None")
        self.assertTrue(count == 2, "Count should be 2")
        count = self.df.count("stringCol", "Nothing")
        self.assertTrue(count == 0, "Count should be 0")
        count = self.df.count("intCol", "None")
        self.assertTrue(count == 2, "Count should be 2")
        count = self.df.count("stringCol", "None")
        self.assertTrue(count == 2, "Count should be 2")

    def test_count_unique(self):
        count = self.df.count_unique(2)
        self.assertTrue(count == 3, "Unique count should be 3")
        self.df.set_boolean(8, 4, False)
        count = self.df.count_unique(8)
        self.assertTrue(count == 2, "Unique count should be 2")

    def test_count_unique_by_name(self):
        count = self.df.count_unique("intCol")
        self.assertTrue(count == 3, "Unique count should be 3")
        count = self.df.count_unique("booleanCol")
        self.assertTrue(count == 1, "Unique count should be 1")

    def test_unique(self):
        set1 = self.df.unique(2)
        self.assertTrue(len(set1) == 3, "Unique set size should be 3")
        truth_int = {12, 32, 52}
        self.assertTrue(set1 == truth_int, "Sets should be equal")

        set2 = self.df.unique(4)
        self.assertTrue(len(set2) == 3, "Unique set size should be 3")
        truth_string = {"10", "30", "50"}
        self.assertTrue(set2 == truth_string, "Sets should be equal")

        set3 = self.df.unique(8)
        self.assertTrue(len(set3) == 1, "Unique set size should be 1")
        truth_boolean = {True}
        self.assertTrue(set3 == truth_boolean, "Sets should be equal")

        self.df.set_char(5, 4, "a")
        set4 = self.df.unique(5)
        self.assertTrue(len(set4) == 2, "Unique set size should be 2")
        truth_char = {"a", "c"}
        self.assertTrue(set4 == truth_char, "Sets should be equal")

        self.df.set_binary(9, 4, bytearray.fromhex("05"))
        set5 = self.df.unique(9)
        self.assertTrue(len(set5) == 2, "Unique set size should be 2")
        truth_binary = {bytes(bytearray.fromhex("05")), bytes(bytearray.fromhex("000070"))}
        self.assertTrue(set5 == truth_binary, "Sets should be equal")

    def test_unique_by_name(self):
        set1 = self.df.unique("intCol")
        self.assertTrue(len(set1) == 3, "Unique set size should be 3")
        truth_int = {12, 32, 52}
        self.assertTrue(set1 == truth_int, "Sets should be equal")

        set2 = self.df.unique("stringCol")
        self.assertTrue(len(set2) == 3, "Unique set size should be 3")
        truth_string = {"10", "30", "50"}
        self.assertTrue(set2 == truth_string, "Sets should be equal")

        set3 = self.df.unique("booleanCol")
        self.assertTrue(len(set3) == 1, "Unique set size should be 1")
        truth_boolean = {True}
        self.assertTrue(set3 == truth_boolean, "Sets should be equal")

        self.df.set_char("charCol", 4, "a")
        set4 = self.df.unique("charCol")
        self.assertTrue(len(set4) == 2, "Unique set size should be 2")
        truth_char = {"a", "c"}
        self.assertTrue(set4 == truth_char, "Sets should be equal")

        self.df.set_binary("binaryCol", 4, bytearray.fromhex("05"))
        set5 = self.df.unique("binaryCol")
        self.assertTrue(len(set5) == 2, "Unique set size should be 2")
        truth_binary = {bytes(bytearray.fromhex("05")), bytes(bytearray.fromhex("000070"))}
        self.assertTrue(set5 == truth_binary, "Sets should be equal")



    #*******************************************************************#
    #           Difference, Union and Intersection operations           #
    #*******************************************************************#



    def test_difference_columns(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac"]),
            NullableIntColumn("B", [1, 2, 3]),
            NullableIntColumn("E", [1, 2, 3]))

        df2 = NullableDataFrame(
            NullableStringColumn("A", ["bba","bbb","bbc"]),
            NullableFloatColumn("C", [1.1,2.2,3.3]),
            NullableIntColumn("B", [11, 22, 33]),
            NullableCharColumn("D", ["a", "b", "c"]))

        df3 = df1.difference_columns(df2)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(df3.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["E", "C", "D"], df3.get_column_names(), "Columns do not match")

        self.assertTrue(
            df3.get_column("E") is df1.get_column("E"), "Columns reference does not match")
        self.assertTrue(
            df3.get_column("C") is df2.get_column("C"), "Columns reference does not match")
        self.assertTrue(
            df3.get_column("D") is df2.get_column("D"), "Columns reference does not match")

    def test_difference_columns_same_arg(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac"]),
            NullableIntColumn("B", [1, 2, 3]),
            NullableIntColumn("E", [1, 2, 3]))

        df3 = df1.difference_columns(df1)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 0, "DataFrame should have 0 columns")
        self.assertTrue(df3.rows() == 0, "DataFrame should have 0 rows")
        self.assertTrue(df3.get_column_names() is None, "Column names should be empty")

    def test_union_columns(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac"]),
            NullableIntColumn("B", [1, 2, 3]),
            NullableIntColumn("E", [1, 2, 3]))

        df2 = NullableDataFrame(
            NullableStringColumn("A", ["bba","bbb","bbc"]),
            NullableFloatColumn("C", [1.1,2.2,3.3]),
            NullableIntColumn("B", [11, 22, 33]),
            NullableCharColumn("D", ["a", "b", "c"]))

        df3 = df1.union_columns(df2)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 5, "DataFrame should have 5 columns")
        self.assertTrue(df3.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A", "B", "E", "C", "D"],
            df3.get_column_names(),
            "Columns do not match")

        self.assertTrue(
            df3.get_column("A") is df1.get_column("A"), "Columns reference does not match")
        self.assertTrue(
            df3.get_column("B") is df1.get_column("B"), "Columns reference does not match")
        self.assertTrue(
            df3.get_column("E") is df1.get_column("E"), "Columns reference does not match")
        self.assertTrue(
            df3.get_column("C") is df2.get_column("C"), "Columns reference does not match")
        self.assertTrue(
            df3.get_column("D") is df2.get_column("D"), "Columns reference does not match")

    def test_union_columns_same_arg(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac"]),
            NullableIntColumn("B", [1, 2, 3]),
            NullableIntColumn("E", [1, 2, 3]))

        df3 = df1.union_columns(df1)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(df3.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A", "B", "E"],
            df3.get_column_names(),
            "Columns do not match")

        self.assertTrue(
            df3.get_column("A") is df1.get_column("A"), "Columns reference does not match")
        self.assertTrue(
            df3.get_column("B") is df1.get_column("B"), "Columns reference does not match")
        self.assertTrue(
            df3.get_column("E") is df1.get_column("E"), "Columns reference does not match")

    def test_intersection_columns(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac"]),
            NullableIntColumn("B", [1, 2, 3]),
            NullableIntColumn("E", [1, 2, 3]))

        df2 = NullableDataFrame(
            NullableStringColumn("C", ["bba","bbb","bbc"]),
            NullableFloatColumn("A", [1.1,2.2,3.3]),
            NullableIntColumn("D", [11, 22, 33]),
            NullableCharColumn("B", ["a", "b", "c"]))

        df3 = df1.intersection_columns(df2)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 2, "DataFrame should have 2 columns")
        self.assertTrue(df3.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A", "B"], df3.get_column_names(), "Columns do not match")

        self.assertTrue(
            df3.get_column("A") is df1.get_column("A"), "Columns reference does not match")
        self.assertTrue(
            df3.get_column("B") is df1.get_column("B"), "Columns reference does not match")

    def test_intersection_columns_same_arg(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac"]),
            NullableIntColumn("B", [1, 2, 3]),
            NullableIntColumn("E", [1, 2, 3]))

        df3 = df1.intersection_columns(df1)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(df3.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A", "B", "E"], df3.get_column_names(), "Columns do not match")

        self.assertTrue(
            df3.get_column("A") is df1.get_column("A"), "Columns reference does not match")

        self.assertTrue(
            df3.get_column("B") is df1.get_column("B"), "Columns reference does not match")
        self.assertTrue(
            df3.get_column("E") is df1.get_column("E"), "Columns reference does not match")

    def test_difference_columns_invalid_arg(self):
        df1 = DataFrame.Default(
            DataFrame.StringColumn("A", ["aaa", "aab", "aac"]))

        self.assertRaises(
            DataFrameException, self.df.difference_columns, df1)

    def test_difference_columns_empty_arg(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac"]),
            NullableIntColumn("B", [1, 2, 3]),
            NullableIntColumn("E", [1, 2, 3]))

        df2 = NullableDataFrame()
        self.assertRaises(
            DataFrameException, df1.difference_columns, df2)

    def test_union_columns_invalid_arg(self):
        df1 = DataFrame.Default(
            DataFrame.StringColumn("A", ["aaa", "aab", "aac"]))

        self.assertRaises(
            DataFrameException, self.df.union_columns, df1)

    def test_union_columns_empty_arg(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac"]),
            NullableIntColumn("B", [1, 2, 3]),
            NullableIntColumn("E", [1, 2, 3]))

        df2 = NullableDataFrame()
        self.assertRaises(
            DataFrameException, df1.union_columns, df2)

    def test_intersection_columns_invalid_arg(self):
        df1 = DataFrame.Default(
            DataFrame.StringColumn("A", ["aaa", "aab", "aac"]))

        self.assertRaises(
            DataFrameException, self.df.intersection_columns, df1)

    def test_intersection_columns_empty_arg(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac"]),
            NullableIntColumn("B", [1, 2, 3]),
            NullableIntColumn("E", [1, 2, 3]))

        df2 = NullableDataFrame()
        self.assertRaises(
            DataFrameException, df1.intersection_columns, df2)

    def test_difference_rows(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac", "aab"]),
            NullableIntColumn("B", [1, 2, 3, 2]),
            NullableIntColumn("C", [1, None, 3, None]))

        df2 = NullableDataFrame(
            NullableStringColumn("A", ["bba", "aab", "bbc", "aab"]),
            NullableIntColumn("B", [1, 2, 3, 2]),
            NullableIntColumn("C", [1, None, 3, None]))

        df3 = df1.difference_rows(df2)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(df3.rows() == 4, "DataFrame should have 4 rows")
        self.assertSequenceAlmostEqual(
            ["A", "B", "C"], df3.get_column_names(), "Columns do not match")

        self.assertSequenceAlmostEqual(["aaa", 1, 1], df3.get_row(0), "Invalid row")
        self.assertSequenceAlmostEqual(["aac", 3, 3], df3.get_row(1), "Invalid row")
        self.assertSequenceAlmostEqual(["bba", 1, 1], df3.get_row(2), "Invalid row")
        self.assertSequenceAlmostEqual(["bbc", 3, 3], df3.get_row(3), "Invalid row")

    def test_difference_rows_unlabeled(self):
        df1 = NullableDataFrame(
            NullableStringColumn(values=["aaa", "aab", "aac", "aab"]),
            NullableIntColumn(values=[1, 2, 3, 2]),
            NullableIntColumn(values=[1, 2, 3, 2]))

        df2 = NullableDataFrame(
            NullableStringColumn(values=["bba", "aab", "bbc", "aab"]),
            NullableIntColumn(values=[1, 2, 3, 2]),
            NullableIntColumn(values=[1, 2, 3, 2]))

        df3 = df1.difference_rows(df2)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(df3.rows() == 4, "DataFrame should have 4 rows")
        self.assertFalse(df3.has_column_names(), "DataFrame should not have column names")
        self.assertSequenceAlmostEqual(["aaa", 1, 1], df3.get_row(0), "Invalid row")
        self.assertSequenceAlmostEqual(["aac", 3, 3], df3.get_row(1), "Invalid row")
        self.assertSequenceAlmostEqual(["bba", 1, 1], df3.get_row(2), "Invalid row")
        self.assertSequenceAlmostEqual(["bbc", 3, 3], df3.get_row(3), "Invalid row")

    def test_difference_rows_same_arg(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac", "aab"]),
            NullableIntColumn("B", [1, 2, 3, 2]),
            NullableIntColumn("C", [1, 2, 3, 2]))

        df3 = df1.difference_rows(df1)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(df3.rows() == 0, "DataFrame should have 0 rows")
        self.assertSequenceAlmostEqual(
            ["A", "B", "C"], df3.get_column_names(), "Columns do not match")

    def test_union_rows(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac", "aab"]),
            NullableIntColumn("B", [1, 2, 3, 2]),
            NullableIntColumn("C", [1, None, 3, None]))

        df2 = NullableDataFrame(
            NullableStringColumn("A", ["bba", "aab", "bbc"]),
            NullableIntColumn("B", [1, 2, 3]),
            NullableIntColumn("C", [1, None, 3]))

        df3 = df1.union_rows(df2)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(df3.rows() == 5, "DataFrame should have 5 rows")
        self.assertSequenceAlmostEqual(
            ["A", "B", "C"], df3.get_column_names(), "Columns do not match")

        self.assertSequenceAlmostEqual(["aaa", 1, 1], df3.get_row(0), "Invalid row")
        self.assertSequenceAlmostEqual(["aab", 2, None], df3.get_row(1), "Invalid row")
        self.assertSequenceAlmostEqual(["aac", 3, 3], df3.get_row(2), "Invalid row")
        self.assertSequenceAlmostEqual(["bba", 1, 1], df3.get_row(3), "Invalid row")
        self.assertSequenceAlmostEqual(["bbc", 3, 3], df3.get_row(4), "Invalid row")

    def test_union_rows_unlabeled(self):
        df1 = NullableDataFrame(
            NullableStringColumn(values=["aaa", "aab", "aac", "aab"]),
            NullableIntColumn(values=[1, 2, 3, 2]),
            NullableIntColumn(values=[1, 2, 3, 2]))

        df2 = NullableDataFrame(
            NullableStringColumn(values=["bba", "aab", "bbc"]),
            NullableIntColumn(values=[1, 2, 3]),
            NullableIntColumn(values=[1, 2, 3]))

        df3 = df1.union_rows(df2)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(df3.rows() == 5, "DataFrame should have 5 rows")
        self.assertFalse(df3.has_column_names(), "DataFrame should not have column names")
        self.assertSequenceAlmostEqual(["aaa", 1, 1], df3.get_row(0), "Invalid row")
        self.assertSequenceAlmostEqual(["aab", 2, 2], df3.get_row(1), "Invalid row")
        self.assertSequenceAlmostEqual(["aac", 3, 3], df3.get_row(2), "Invalid row")
        self.assertSequenceAlmostEqual(["bba", 1, 1], df3.get_row(3), "Invalid row")
        self.assertSequenceAlmostEqual(["bbc", 3, 3], df3.get_row(4), "Invalid row")

    def test_union_rows_same_arg(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac", "aab"]),
            NullableIntColumn("B", [1, 2, 3, 2]),
            NullableIntColumn("C", [1, None, 3, None]))

        df3 = df1.union_rows(df1)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(df3.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A", "B", "C"], df3.get_column_names(), "Columns do not match")

        self.assertSequenceAlmostEqual(["aaa", 1, 1], df3.get_row(0), "Invalid row")
        self.assertSequenceAlmostEqual(["aab", 2, None], df3.get_row(1), "Invalid row")
        self.assertSequenceAlmostEqual(["aac", 3, 3], df3.get_row(2), "Invalid row")

    def test_intersection_rows(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac", "aab"]),
            NullableIntColumn("B", [1, 2, 3, 2]),
            NullableIntColumn("C", [1, None, 3, None]))

        df2 = NullableDataFrame(
            NullableStringColumn("A", ["bba", "aab", "bbc", "aab"]),
            NullableIntColumn("B", [1, 2, 3, 2]),
            NullableIntColumn("C", [1, None, 3, None]))

        df3 = df1.intersection_rows(df2)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(df3.rows() == 1, "DataFrame should have 1 rows")
        self.assertSequenceAlmostEqual(
            ["A", "B", "C"], df3.get_column_names(), "Columns do not match")

        self.assertSequenceAlmostEqual(["aab", 2, None], df3.get_row(0), "Invalid row")

    def test_intersection_rows_unlabeled(self):
        df1 = NullableDataFrame(
            NullableStringColumn(values=["aaa", "aab", "aac", "aab"]),
            NullableIntColumn(values=[1, 2, 3, 2]),
            NullableIntColumn(values=[1, 2, 3, 2]))

        df2 = NullableDataFrame(
            NullableStringColumn(values=["bba", "aab", "bbc", "aab"]),
            NullableIntColumn(values=[1, 2, 3, 2]),
            NullableIntColumn(values=[1, 2, 3, 2]))

        df3 = df1.intersection_rows(df2)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(df3.rows() == 1, "DataFrame should have 1 rows")
        self.assertFalse(df3.has_column_names(), "DataFrame should not have column names")
        self.assertSequenceAlmostEqual(["aab", 2, 2], df3.get_row(0), "Invalid row")

    def test_intersection_rows_same_arg(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac", "aab"]),
            NullableIntColumn("B", [1, 2, 3, 2]),
            NullableIntColumn("C", [1, None, 3, None]))

        df3 = df1.intersection_rows(df1)
        self.assertTrue(df3.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df3.columns() == 3, "DataFrame should have 3 columns")
        self.assertTrue(df3.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A", "B", "C"], df3.get_column_names(), "Columns do not match")

        self.assertSequenceAlmostEqual(["aaa", 1, 1], df3.get_row(0), "Invalid row")
        self.assertSequenceAlmostEqual(["aab", 2, None], df3.get_row(1), "Invalid row")
        self.assertSequenceAlmostEqual(["aac", 3, 3], df3.get_row(2), "Invalid row")

    def test_difference_rows_empty_arg(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac"]),
            NullableIntColumn("B", [1, 2, 3]),
            NullableIntColumn("E", [1, 2, 3]))

        df2 = NullableDataFrame()
        self.assertRaises(
            DataFrameException, df1.difference_rows, df2)

    def test_union_rows_empty_arg(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac"]),
            NullableIntColumn("B", [1, 2, 3]),
            NullableIntColumn("E", [1, 2, 3]))

        df2 = NullableDataFrame()
        self.assertRaises(
            DataFrameException, df1.union_rows, df2)

    def test_intersection_rows_empty_arg(self):
        df1 = NullableDataFrame(
            NullableStringColumn("A", ["aaa", "aab", "aac"]),
            NullableIntColumn("B", [1, 2, 3]),
            NullableIntColumn("E", [1, 2, 3]))

        df2 = NullableDataFrame()
        self.assertRaises(
            DataFrameException, df1.intersection_rows, df2)



    #****************************************************#
    #                 Convert Column Types               #
    #****************************************************#



    def test_convert_from_bytecolumn(self):
        self.df.add_column(NullableByteColumn("data"))
        self.df.replace("data", replacement=lambda i, v: 0 if i % 2 == 0 else None)
        for _, code in enumerate(self.column_types):
            df2 = self.df.clone().convert("data", code)
            df2 = df2.convert("data", self.df.get_column("data").type_code())
            self.assertTrue(df2.equals(self.df), "Conversion failure")

    def test_convert_from_shortcolumn(self):
        self.df.add_column(NullableShortColumn("data"))
        self.df.replace("data", replacement=lambda i, v: 0 if i % 2 == 0 else None)
        for _, code in enumerate(self.column_types):
            df2 = self.df.clone().convert("data", code)
            df2 = df2.convert("data", self.df.get_column("data").type_code())
            self.assertTrue(df2.equals(self.df), "Conversion failure")

    def test_convert_from_intcolumn(self):
        self.df.add_column(NullableIntColumn("data"))
        self.df.replace("data", replacement=lambda i, v: 0 if i % 2 == 0 else None)
        for _, code in enumerate(self.column_types):
            df2 = self.df.clone().convert("data", code)
            df2 = df2.convert("data", self.df.get_column("data").type_code())
            self.assertTrue(df2.equals(self.df), "Conversion failure")

    def test_convert_from_longcolumn(self):
        self.df.add_column(NullableLongColumn("data"))
        self.df.replace("data", replacement=lambda i, v: 0 if i % 2 == 0 else None)
        for _, code in enumerate(self.column_types):
            df2 = self.df.clone().convert("data", code)
            df2 = df2.convert("data", self.df.get_column("data").type_code())
            self.assertTrue(df2.equals(self.df), "Conversion failure")

    def test_convert_from_floatcolumn(self):
        self.df.add_column(NullableFloatColumn("data"))
        self.df.replace("data", replacement=lambda i, v: 0.0 if i % 2 == 0 else None)
        for _, code in enumerate(self.column_types):
            df2 = self.df.clone().convert("data", code)
            df2 = df2.convert("data", self.df.get_column("data").type_code())
            self.assertTrue(df2.equals(self.df), "Conversion failure")

    def test_convert_from_doublecolumn(self):
        self.df.add_column(NullableDoubleColumn("data"))
        self.df.replace("data", replacement=lambda i, v: 0.0 if i % 2 == 0 else None)
        for _, code in enumerate(self.column_types):
            df2 = self.df.clone().convert("data", code)
            df2 = df2.convert("data", self.df.get_column("data").type_code())
            self.assertTrue(df2.equals(self.df), "Conversion failure")

    def test_convert_from_stringcolumn(self):
        self.df.add_column(NullableStringColumn("data"))
        self.df.replace("data", replacement=lambda i, v: "0" if i % 2 == 0 else None)
        for _, code in enumerate(self.column_types):
            df2 = self.df.clone()
            if code == NullableBinaryColumn.TYPE_CODE:
                df2.replace("data", "0", replacement="00")

            df2 = df2.convert("data", code)
            df2 = df2.convert("data", self.df.get_column("data").type_code())
            # only keep the first character to match the original string
            # when converting back from float and double
            df2.replace("data", replacement=lambda v: v[0] if v is not None else None)
            # change correctly from boolean values
            df2.replace("data", "F", replacement="0")
            self.assertTrue(df2.equals(self.df), "Conversion failure")

    def test_convert_from_charcolumn(self):
        self.df.add_column(NullableCharColumn("data"))
        self.df.replace("data", replacement=lambda i, v: "0" if i % 2 == 0 else None)
        for _, code in enumerate(self.column_types):
            df2 = self.df.clone().convert("data", code)
            df2 = df2.convert("data", self.df.get_column("data").type_code())
            self.assertTrue(df2.equals(self.df), "Conversion failure")

    def test_convert_from_booleancolumn(self):
        self.df.add_column(NullableBooleanColumn("data"))
        self.df.replace("data", replacement=lambda i, v: False if i % 2 == 0 else None)
        for _, code in enumerate(self.column_types):
            df2 = self.df.clone().convert("data", code)
            df2 = df2.convert("data", self.df.get_column("data").type_code())
            self.assertTrue(df2.equals(self.df), "Conversion failure")

    def test_convert_from_binarycolumn(self):
        self.df.add_column(NullableBinaryColumn("data"))
        self.df.replace(
            "data",
            replacement=lambda i, v: bytearray.fromhex("00")
                        if i % 2 == 0
                        else None)

        df2 = self.df.clone().convert("data", NullableByteColumn.TYPE_CODE)
        df2 = df2.convert("data", self.df.get_column("data").type_code())
        self.assertTrue(df2.equals(self.df), "Conversion failure")

        self.df.replace(
            "data",
            replacement=lambda i, v: bytearray.fromhex("0000")
                        if i % 2 == 0
                        else None)

        df2 = self.df.clone().convert("data", NullableShortColumn.TYPE_CODE)
        df2 = df2.convert("data", self.df.get_column("data").type_code())
        self.assertTrue(df2.equals(self.df), "Conversion failure")

        self.df.replace(
            "data",
            replacement=lambda i, v: bytearray.fromhex("00000000")
                        if i % 2 == 0
                        else None)

        df2 = self.df.clone().convert("data", NullableIntColumn.TYPE_CODE)
        df2 = df2.convert("data", self.df.get_column("data").type_code())
        self.assertTrue(df2.equals(self.df), "Conversion failure")

        self.df.replace(
            "data",
            replacement=lambda i, v: bytearray.fromhex("0000000000000000")
                        if i % 2 == 0
                        else None)

        df2 = self.df.clone().convert("data", NullableLongColumn.TYPE_CODE)
        df2 = df2.convert("data", self.df.get_column("data").type_code())
        self.assertTrue(df2.equals(self.df), "Conversion failure")

        self.df.replace(
            "data",
            replacement=lambda i, v: bytearray(struct.pack(">f", 0.0))
                        if i % 2 == 0
                        else None)

        df2 = self.df.clone().convert("data", NullableFloatColumn.TYPE_CODE)
        df2 = df2.convert("data", self.df.get_column("data").type_code())
        self.assertTrue(df2.equals(self.df), "Conversion failure")

        self.df.replace(
            "data",
            replacement=lambda i, v: bytearray(struct.pack(">d", 0.0))
                        if i % 2 == 0
                        else None)

        df2 = self.df.clone().convert("data", NullableDoubleColumn.TYPE_CODE)
        df2 = df2.convert("data", self.df.get_column("data").type_code())
        self.assertTrue(df2.equals(self.df), "Conversion failure")

        self.df.replace(
            "data",
            replacement=lambda i, v: bytearray("0".encode("utf-8"))
                        if i % 2 == 0
                        else None)

        df2 = self.df.clone().convert("data", NullableStringColumn.TYPE_CODE)
        df2 = df2.convert("data", self.df.get_column("data").type_code())
        self.assertTrue(df2.equals(self.df), "Conversion failure")

        self.df.replace(
            "data",
            replacement=lambda i, v: bytearray("0".encode("utf-8"))
                        if i % 2 == 0
                        else None)

        df2 = self.df.clone().convert("data", NullableCharColumn.TYPE_CODE)
        df2 = df2.convert("data", self.df.get_column("data").type_code())
        self.assertTrue(df2.equals(self.df), "Conversion failure")

        self.df.replace(
            "data",
            replacement=lambda i, v: bytearray.fromhex("00")
                        if i % 2 == 0
                        else None)

        df2 = self.df.clone().convert("data", NullableBooleanColumn.TYPE_CODE)
        df2 = df2.convert("data", self.df.get_column("data").type_code())
        self.assertTrue(df2.equals(self.df), "Conversion failure")

        self.df.replace(
            "data",
            replacement=lambda i, v: bytearray.fromhex("00")
                        if i % 2 == 0
                        else None)

        df2 = self.df.clone().convert("data", NullableBinaryColumn.TYPE_CODE)
        self.assertTrue(df2.equals(self.df), "Conversion failure")



    #****************************************#
    #           GroupBy operations           #
    #****************************************#



    def test_group_minimum_by(self):
        df1 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", [None, "b", "c", "b", "b", None]),
            DataFrame.NullableStringColumn("B", [None, "aab", "aac", "aab", "aab", None]),
            DataFrame.NullableFloatColumn("C", [5.5, 2.2, 3.3, 4.4, None, 6.6]),
            DataFrame.NullableStringColumn("D", [None, "bbb", "bbc", "bbb", "bbb", None]),
            DataFrame.NullableIntColumn("E", [5, 2, 3, 4, None, 6]),
            DataFrame.NullableLongColumn("F", [5, 2, 3, 4, None, 6]))

        df2 = df1.group_minimum_by("B")
        self.assertTrue(df2.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df2.columns() == 4, "DataFrame should have 4 columns")
        self.assertTrue(df2.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["B", "C", "E", "F"],
            df2.get_column_names(),
            "Columns do not match")

        df2.sort_by(0)
        df3 = NullableDataFrame(
            DataFrame.NullableStringColumn("B", ["aab", "aac", None]),
            DataFrame.NullableFloatColumn("C", [2.2, 3.3, 5.5]),
            DataFrame.NullableIntColumn("E", [2, 3, 5]),
            DataFrame.NullableLongColumn("F", [2, 3, 5]))

        self.assertTrue(df2.equals(df3), "DataFrames are not equal")

    def test_group_maximum_by(self):
        df1 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", [None, "b", "c", "b", "b", None]),
            DataFrame.NullableStringColumn("B", [None, "aab", "aac", "aab", "aab", None]),
            DataFrame.NullableFloatColumn("C", [5.5, 2.2, 3.3, 4.4, None, 6.6]),
            DataFrame.NullableStringColumn("D", [None, "bbb", "bbc", "bbb", "bbb", None]),
            DataFrame.NullableIntColumn("E", [5, 2, 3, 4, None, 6]),
            DataFrame.NullableLongColumn("F", [5, 2, 3, 4, None, 6]))

        df2 = df1.group_maximum_by("A")
        self.assertTrue(df2.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df2.columns() == 4, "DataFrame should have 4 columns")
        self.assertTrue(df2.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A", "C", "E", "F"],
            df2.get_column_names(),
            "Columns do not match")

        df2.sort_by(0)
        df3 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", ["b", "c", None]),
            DataFrame.NullableFloatColumn("C", [4.4, 3.3, 6.6]),
            DataFrame.NullableIntColumn("E", [4, 3, 6]),
            DataFrame.NullableLongColumn("F", [4, 3, 6]))

        self.assertTrue(df2.equals(df3), "DataFrames are not equal")

    def test_group_average_by(self):
        df1 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", [None, "b", "c", "b", "b", None]),
            DataFrame.NullableStringColumn("B", [None, "aab", "aac", "aab", "aab", None]),
            DataFrame.NullableFloatColumn("C", [5.5, 2.2, 3.3, 4.4, None, 6.6]),
            DataFrame.NullableStringColumn("D", [None, "bbb", "bbc", "bbb", "bbb", None]),
            DataFrame.NullableIntColumn("E", [5, 2, 3, 4, None, 6]),
            DataFrame.NullableLongColumn("F", [5, 2, 3, 4, None, 6]))

        df2 = df1.group_average_by("D")
        self.assertTrue(df2.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df2.columns() == 4, "DataFrame should have 4 columns")
        self.assertTrue(df2.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["D", "C", "E", "F"],
            df2.get_column_names(),
            "Columns do not match")

        df2.sort_by(0)
        df2.round("C", 2)
        df2.round("E", 2)
        df2.round("F", 2)
        df3 = NullableDataFrame(
            DataFrame.NullableStringColumn("D", ["bbb", "bbc", None]),
            DataFrame.NullableDoubleColumn("C", [3.3, 3.3, 6.05]),
            DataFrame.NullableDoubleColumn("E", [3.0, 3.0, 5.5]),
            DataFrame.NullableDoubleColumn("F", [3.0, 3.0, 5.5]))

        self.assertTrue(df2.equals(df3), "DataFrames are not equal")

    def test_group_sum_by(self):
        df1 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", [None, "b", "c", "b", "b", None]),
            DataFrame.NullableStringColumn("B", [None, "aab", "aac", "aab", "aab", None]),
            DataFrame.NullableFloatColumn("C", [5.5, 2.2, 3.3, 4.4, None, 6.6]),
            DataFrame.NullableStringColumn("D", [None, "bbb", "bbc", "bbb", "bbb", None]),
            DataFrame.NullableIntColumn("E", [5, 2, 3, 4, None, 6]),
            DataFrame.NullableLongColumn("F", [5, 2, 3, 4, None, 6]))

        df2 = df1.group_sum_by("A")
        self.assertTrue(df2.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df2.columns() == 4, "DataFrame should have 4 columns")
        self.assertTrue(df2.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A", "C", "E", "F"],
            df2.get_column_names(),
            "Columns do not match")

        df2.sort_by(0)
        df2.round("C", 2)
        df3 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", ["b", "c", None]),
            DataFrame.NullableDoubleColumn("C", [6.6, 3.3, 12.1]),
            DataFrame.NullableDoubleColumn("E", [6.0, 3.0, 11.0]),
            DataFrame.NullableDoubleColumn("F", [6.0, 3.0, 11.0]))

        self.assertTrue(df2.equals(df3), "DataFrames are not equal")

    def test_group_minimum_empty(self):
        df1 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", [None, "b", "c", "b", "b", None]))

        df2 = df1.group_minimum_by("A")
        self.assertTrue(df2.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df2.columns() == 1, "DataFrame should have 1 columns")
        self.assertTrue(df2.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A"], df2.get_column_names(), "Columns do not match")

    def test_group_maximum_empty(self):
        df1 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", [None, "b", "c", "b", "b", None]))

        df2 = df1.group_maximum_by("A")
        self.assertTrue(df2.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df2.columns() == 1, "DataFrame should have 1 columns")
        self.assertTrue(df2.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A"], df2.get_column_names(), "Columns do not match")

    def test_group_average_empty(self):
        df1 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", [None, "b", "c", "b", "b", None]))

        df2 = df1.group_average_by("A")
        self.assertTrue(df2.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df2.columns() == 1, "DataFrame should have 1 columns")
        self.assertTrue(df2.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A"], df2.get_column_names(), "Columns do not match")

    def test_group_sum_empty(self):
        df1 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", [None, "b", "c", "b", "b", None]))

        df2 = df1.group_sum_by("A")
        self.assertTrue(df2.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df2.columns() == 1, "DataFrame should have 1 columns")
        self.assertTrue(df2.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A"], df2.get_column_names(), "Columns do not match")

    def test_group_minimum_only_nulls(self):
        df1 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", [None, "b", "c", "b", "b", None]),
            DataFrame.NullableStringColumn("B", [None, "aab", "aac", "aab", "aab", None]),
            DataFrame.NullableFloatColumn("C", [5.5, 2.2, None, 4.4, None, 6.6]),
            DataFrame.NullableStringColumn("D", [None, "bbb", "bbc", "bbb", "bbb", None]),
            DataFrame.NullableIntColumn("E", [5, 2, None, 4, None, 6]),
            DataFrame.NullableLongColumn("F", [5, 2, None, 4, None, 6]))

        df2 = df1.group_minimum_by("A")
        self.assertTrue(df2.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df2.columns() == 4, "DataFrame should have 4 columns")
        self.assertTrue(df2.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A", "C", "E", "F"],
            df2.get_column_names(),
            "Columns do not match")

        df2.sort_by(0)
        df3 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", ["b", "c", None]),
            DataFrame.NullableFloatColumn("C", [2.2, float("NaN"), 5.5]),
            DataFrame.NullableIntColumn("E", [2, None, 5]),
            DataFrame.NullableLongColumn("F", [2, None, 5]))

        self.assertTrue(df2.equals(df3), "DataFrames are not equal")

    def test_group_maximum_only_nulls(self):
        df1 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", [None, "b", "c", "b", "b", None]),
            DataFrame.NullableStringColumn("B", [None, "aab", "aac", "aab", "aab", None]),
            DataFrame.NullableFloatColumn("C", [5.5, 2.2, None, 4.4, None, 6.6]),
            DataFrame.NullableStringColumn("D", [None, "bbb", "bbc", "bbb", "bbb", None]),
            DataFrame.NullableIntColumn("E", [5, 2, None, 4, None, 6]),
            DataFrame.NullableLongColumn("F", [5, 2, None, 4, None, 6]))

        df2 = df1.group_maximum_by("A")
        self.assertTrue(df2.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df2.columns() == 4, "DataFrame should have 4 columns")
        self.assertTrue(df2.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A", "C", "E", "F"],
            df2.get_column_names(),
            "Columns do not match")

        df2.sort_by(0)
        df3 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", ["b", "c", None]),
            DataFrame.NullableFloatColumn("C", [4.4, float("NaN"), 6.6]),
            DataFrame.NullableIntColumn("E", [4, None, 6]),
            DataFrame.NullableLongColumn("F", [4, None, 6]))

        self.assertTrue(df2.equals(df3), "DataFrames are not equal")

    def test_group_average_only_nulls(self):
        df1 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", [None, "b", "c", "b", "b", None]),
            DataFrame.NullableStringColumn("B", [None, "aab", "aac", "aab", "aab", None]),
            DataFrame.NullableFloatColumn("C", [5.5, 2.2, None, 4.4, None, 6.6]),
            DataFrame.NullableStringColumn("D", [None, "bbb", "bbc", "bbb", "bbb", None]),
            DataFrame.NullableIntColumn("E", [5, 2, None, 4, None, 6]),
            DataFrame.NullableLongColumn("F", [5, 2, None, 4, None, 6]))

        df2 = df1.group_average_by("A")
        self.assertTrue(df2.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df2.columns() == 4, "DataFrame should have 4 columns")
        self.assertTrue(df2.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A", "C", "E", "F"],
            df2.get_column_names(),
            "Columns do not match")

        df2.sort_by(0)
        df2.round("C", 6)
        df2.round("E", 6)
        df2.round("F", 6)
        df3 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", ["b", "c", None]),
            DataFrame.NullableDoubleColumn("C", [3.3, float("NaN"), 6.05]),
            DataFrame.NullableDoubleColumn("E", [3.0, float("NaN"), 5.5]),
            DataFrame.NullableDoubleColumn("F", [3.0, float("NaN"), 5.5]))

        self.assertTrue(df2.equals(df3), "DataFrames are not equal")

    def test_group_sum_only_nulls(self):
        df1 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", [None, "b", "c", "b", "b", None]),
            DataFrame.NullableStringColumn("B", [None, "aab", "aac", "aab", "aab", None]),
            DataFrame.NullableFloatColumn("C", [5.5, 2.2, None, 4.4, None, 6.6]),
            DataFrame.NullableStringColumn("D", [None, "bbb", "bbc", "bbb", "bbb", None]),
            DataFrame.NullableIntColumn("E", [5, 2, None, 4, None, 6]),
            DataFrame.NullableLongColumn("F", [5, 2, None, 4, None, 6]))

        df2 = df1.group_sum_by("A")
        self.assertTrue(df2.is_nullable(), "DataFrame has an invalid type")
        self.assertTrue(df2.columns() == 4, "DataFrame should have 4 columns")
        self.assertTrue(df2.rows() == 3, "DataFrame should have 3 rows")
        self.assertSequenceAlmostEqual(
            ["A", "C", "E", "F"],
            df2.get_column_names(),
            "Columns do not match")

        df2.sort_by(0)
        df2.round("C", 2)
        df2.round("E", 2)
        df2.round("F", 2)
        df3 = NullableDataFrame(
            DataFrame.NullableCharColumn("A", ["b", "c", None]),
            DataFrame.NullableDoubleColumn("C", [6.6, float("NaN"), 12.1]),
            DataFrame.NullableDoubleColumn("E", [6.0, float("NaN"), 11.0]),
            DataFrame.NullableDoubleColumn("F", [6.0, float("NaN"), 11.0]))

        self.assertTrue(df2.equals(df3), "DataFrames are not equal")



    #*******************************************************************************************#
    #           Minimum, Maximum, Average, Median, Sum, absolute, Ceil, Floor, Round            #
    #*******************************************************************************************#



    def test_minimum(self):
        self.assertTrue(self.df.minimum(0) == 10.0, "Computed minimum should be 10")
        self.assertTrue(self.df.minimum(1) == 11.0, "Computed minimum should be 11")
        self.assertTrue(self.df.minimum(2) == 12.0, "Computed minimum should be 12")
        self.assertTrue(self.df.minimum(3) == 13.0, "Computed minimum should be 13")
        self.assertAlmostEqual(
            10.1, self.df.minimum(6), places=5, msg="Computed minimum should be 10.1")
        self.assertAlmostEqual(
            11.1, self.df.minimum(7), places=5, msg="Computed minimum should be 11.1")

    def test_minimum_by_name(self):
        self.assertTrue(self.df.minimum("byteCol") == 10.0, "Computed minimum should be 10")
        self.assertTrue(self.df.minimum("shortCol") == 11.0, "Computed minimum should be 11")
        self.assertTrue(self.df.minimum("intCol") == 12.0, "Computed minimum should be 12")
        self.assertTrue(self.df.minimum("longCol") == 13.0, "Computed minimum should be 13")
        self.assertAlmostEqual(
            10.1, self.df.minimum("floatCol"), places=5, msg="Computed minimum should be 10.1")
        self.assertAlmostEqual(
            11.1, self.df.minimum("doubleCol"), places=5, msg="Computed minimum should be 11.1")

    def test_minimum_with_nan(self):
        self.df.clear()
        self.assertTrue(math.isnan(self.df.minimum("byteCol")), "Computed minimum should be NaN")
        df2 = NullableDataFrame(
            NullableByteColumn("bytes", [None, None, None]),
            NullableShortColumn("shorts", [None, None, None]),
            NullableIntColumn("ints", [None, None, None]),
            NullableLongColumn("longs", [None, None, None]),
            NullableFloatColumn("floats", [None, None, None]),
            NullableDoubleColumn("doubles", [None, None, None]))

        self.assertTrue(math.isnan(df2.minimum("bytes")), "Computed minimum should be NaN")
        self.assertTrue(math.isnan(df2.minimum("shorts")), "Computed minimum should be NaN")
        self.assertTrue(math.isnan(df2.minimum("ints")), "Computed minimum should be NaN")
        self.assertTrue(math.isnan(df2.minimum("longs")), "Computed minimum should be NaN")
        self.assertTrue(math.isnan(df2.minimum("floats")), "Computed minimum should be NaN")
        self.assertTrue(math.isnan(df2.minimum("doubles")), "Computed minimum should be NaN")

    def test_maximum(self):
        self.assertTrue(self.df.maximum(0) == 50.0, "Computed maximum should be 50")
        self.assertTrue(self.df.maximum(1) == 51.0, "Computed maximum should be 51")
        self.assertTrue(self.df.maximum(2) == 52.0, "Computed maximum should be 52")
        self.assertTrue(self.df.maximum(3) == 53.0, "Computed maximum should be 53")
        self.assertAlmostEqual(
            50.5, self.df.maximum(6), places=5, msg="Computed maximum should be 50.5")
        self.assertAlmostEqual(
            51.5, self.df.maximum(7), places=5, msg="Computed maximum should be 51.5")

    def test_maximum_by_name(self):
        self.assertTrue(self.df.maximum("byteCol") == 50.0, "Computed maximum should be 50")
        self.assertTrue(self.df.maximum("shortCol") == 51.0, "Computed maximum should be 51")
        self.assertTrue(self.df.maximum("intCol") == 52.0, "Computed maximum should be 52")
        self.assertTrue(self.df.maximum("longCol") == 53.0, "Computed maximum should be 53")
        self.assertAlmostEqual(
            50.5, self.df.maximum("floatCol"), places=5, msg="Computed minimum should be 10.1")
        self.assertAlmostEqual(
            51.5, self.df.maximum("doubleCol"), places=5, msg="Computed minimum should be 11.1")

    def test_maximum_with_nan(self):
        self.df.clear()
        self.assertTrue(math.isnan(self.df.maximum("byteCol")), "Computed maximum should be NaN")
        df2 = NullableDataFrame(
            NullableByteColumn("bytes", [None, None, None]),
            NullableShortColumn("shorts", [None, None, None]),
            NullableIntColumn("ints", [None, None, None]),
            NullableLongColumn("longs", [None, None, None]),
            NullableFloatColumn("floats", [None, None, None]),
            NullableDoubleColumn("doubles", [None, None, None]))

        self.assertTrue(math.isnan(df2.maximum("bytes")), "Computed maximum should be NaN")
        self.assertTrue(math.isnan(df2.maximum("shorts")), "Computed maximum should be NaN")
        self.assertTrue(math.isnan(df2.maximum("ints")), "Computed maximum should be NaN")
        self.assertTrue(math.isnan(df2.maximum("longs")), "Computed maximum should be NaN")
        self.assertTrue(math.isnan(df2.maximum("floats")), "Computed maximum should be NaN")
        self.assertTrue(math.isnan(df2.maximum("doubles")), "Computed maximum should be NaN")

    def test_average(self):
        self.assertTrue(self.df.average(0) == 30.0, "Computed average should be 30")
        self.assertTrue(self.df.average(1) == 31.0, "Computed average should be 31")
        self.assertTrue(self.df.average(2) == 32.0, "Computed average should be 32")
        self.assertTrue(self.df.average(3) == 33.0, "Computed average should be 33")
        self.assertAlmostEqual(
            30.3, self.df.average(6), places=5, msg="Computed average should be 30.3")
        self.assertAlmostEqual(
            31.3, self.df.average(7), places=5, msg="Computed average should be 31.3")

    def test_average_by_name(self):
        self.assertTrue(self.df.average("byteCol") == 30.0, "Computed average should be 30")
        self.assertTrue(self.df.average("shortCol") == 31.0, "Computed average should be 31")
        self.assertTrue(self.df.average("intCol") == 32.0, "Computed average should be 32")
        self.assertTrue(self.df.average("longCol") == 33.0, "Computed average should be 33")
        self.assertAlmostEqual(
            30.3, self.df.average("floatCol"), places=5, msg="Computed average should be 30.3")
        self.assertAlmostEqual(
            31.3, self.df.average("doubleCol"), places=5, msg="Computed average should be 31.3")

    def test_average_with_nan(self):
        self.df.clear()
        self.assertTrue(math.isnan(self.df.average("byteCol")), "Computed average should be NaN")
        df2 = NullableDataFrame(
            NullableByteColumn("bytes", [None, None, None]),
            NullableShortColumn("shorts", [None, None, None]),
            NullableIntColumn("ints", [None, None, None]),
            NullableLongColumn("longs", [None, None, None]),
            NullableFloatColumn("floats", [None, None, None]),
            NullableDoubleColumn("doubles", [None, None, None]))

        self.assertTrue(math.isnan(df2.average("bytes")), "Computed average should be NaN")
        self.assertTrue(math.isnan(df2.average("shorts")), "Computed average should be NaN")
        self.assertTrue(math.isnan(df2.average("ints")), "Computed average should be NaN")
        self.assertTrue(math.isnan(df2.average("longs")), "Computed average should be NaN")
        self.assertTrue(math.isnan(df2.average("floats")), "Computed average should be NaN")
        self.assertTrue(math.isnan(df2.average("doubles")), "Computed average should be NaN")

    def test_median(self):
        self.assertTrue(self.df.median(0) == 30.0, "Computed median should be 30")
        self.assertTrue(self.df.median(1) == 31.0, "Computed median should be 31")
        self.assertTrue(self.df.median(2) == 32.0, "Computed median should be 32")
        self.assertTrue(self.df.median(3) == 33.0, "Computed median should be 33")
        self.assertAlmostEqual(
            30.3, self.df.median(6), places=5, msg="Computed median should be 30.3")
        self.assertAlmostEqual(
            31.3, self.df.median(7), places=5, msg="Computed median should be 31.3")

        self.df.add_row([127,420,420,420,"42","A",420.2,420.2,True,bytearray.fromhex("00ff")])

        self.assertTrue(self.df.median(0) == 40.0, "Computed median should be 40")
        self.assertTrue(self.df.median(1) == 41.0, "Computed median should be 41")
        self.assertTrue(self.df.median(2) == 42.0, "Computed median should be 42")
        self.assertTrue(self.df.median(3) == 43.0, "Computed median should be 43")
        self.assertAlmostEqual(
            40.4, self.df.median(6), places=5, msg="Computed median should be 40.4")
        self.assertAlmostEqual(
            41.4, self.df.median(7), places=5, msg="Computed median should be 41.4")

    def test_median_by_name(self):
        self.assertTrue(self.df.median("byteCol") == 30.0, "Computed median should be 30")
        self.assertTrue(self.df.median("shortCol") == 31.0, "Computed median should be 31")
        self.assertTrue(self.df.median("intCol") == 32.0, "Computed median should be 32")
        self.assertTrue(self.df.median("longCol") == 33.0, "Computed median should be 33")
        self.assertAlmostEqual(
            30.3, self.df.median("floatCol"), places=5, msg="Computed median should be 30.3")
        self.assertAlmostEqual(
            31.3, self.df.median("doubleCol"), places=5, msg="Computed median should be 31.3")

        self.df.add_row([127,420,420,420,"42","A",420.2,420.2,True,bytearray.fromhex("00ff")])

        self.assertTrue(self.df.median("byteCol") == 40.0, "Computed median should be 40")
        self.assertTrue(self.df.median("shortCol") == 41.0, "Computed median should be 41")
        self.assertTrue(self.df.median("intCol") == 42.0, "Computed median should be 42")
        self.assertTrue(self.df.median("longCol") == 43.0, "Computed median should be 43")
        self.assertAlmostEqual(
            40.4, self.df.median("floatCol"), places=5, msg="Computed median should be 40.4")
        self.assertAlmostEqual(
            41.4, self.df.median("doubleCol"), places=5, msg="Computed median should be 41.4")

    def test_median_with_nan(self):
        self.df.clear()
        self.assertTrue(math.isnan(self.df.median("byteCol")), "Computed median should be NaN")
        df2 = NullableDataFrame(
            NullableByteColumn("bytes", [None, None, None]),
            NullableShortColumn("shorts", [None, None, None]),
            NullableIntColumn("ints", [None, None, None]),
            NullableLongColumn("longs", [None, None, None]),
            NullableFloatColumn("floats", [None, None, None]),
            NullableDoubleColumn("doubles", [None, None, None]))

        self.assertTrue(math.isnan(df2.median("bytes")), "Computed median should be NaN")
        self.assertTrue(math.isnan(df2.median("shorts")), "Computed median should be NaN")
        self.assertTrue(math.isnan(df2.median("ints")), "Computed median should be NaN")
        self.assertTrue(math.isnan(df2.median("longs")), "Computed median should be NaN")
        self.assertTrue(math.isnan(df2.median("floats")), "Computed median should be NaN")
        self.assertTrue(math.isnan(df2.median("doubles")), "Computed median should be NaN")

    def test_sum(self):
        self.assertTrue(self.df.sum(0) == 90.0, "Computed sum should be 90")
        self.assertTrue(self.df.sum(1) == 93.0, "Computed sum should be 93")
        self.assertTrue(self.df.sum(2) == 96.0, "Computed sum should be 96")
        self.assertTrue(self.df.sum(3) == 99.0, "Computed sum should be 99")
        self.assertAlmostEqual(
            90.9, self.df.sum(6), places=5, msg="Computed sum should be 90.9")
        self.assertAlmostEqual(
            93.9, self.df.sum(7), places=5, msg="Computed sum should be 93.9")

    def test_sum_by_name(self):
        self.assertTrue(self.df.sum("byteCol") == 90.0, "Computed sum should be 90")
        self.assertTrue(self.df.sum("shortCol") == 93.0, "Computed sum should be 93")
        self.assertTrue(self.df.sum("intCol") == 96.0, "Computed sum should be 96")
        self.assertTrue(self.df.sum("longCol") == 99.0, "Computed sum should be 99")
        self.assertAlmostEqual(
            90.9, self.df.sum("floatCol"), places=5, msg="Computed sum should be 90.9")
        self.assertAlmostEqual(
            93.9, self.df.sum("doubleCol"), places=5, msg="Computed sum should be 93.9")

    def test_sum_with_nan(self):
        self.df.clear()
        self.assertTrue(math.isnan(self.df.sum("byteCol")), "Computed sum should be NaN")
        df2 = NullableDataFrame(
            NullableByteColumn("bytes", [None, None, None]),
            NullableShortColumn("shorts", [None, None, None]),
            NullableIntColumn("ints", [None, None, None]),
            NullableLongColumn("longs", [None, None, None]),
            NullableFloatColumn("floats", [None, None, None]),
            NullableDoubleColumn("doubles", [None, None, None]))

        self.assertTrue(math.isnan(df2.sum("bytes")), "Computed sum should be NaN")
        self.assertTrue(math.isnan(df2.sum("shorts")), "Computed sum should be NaN")
        self.assertTrue(math.isnan(df2.sum("ints")), "Computed sum should be NaN")
        self.assertTrue(math.isnan(df2.sum("longs")), "Computed sum should be NaN")
        self.assertTrue(math.isnan(df2.sum("floats")), "Computed sum should be NaN")
        self.assertTrue(math.isnan(df2.sum("doubles")), "Computed sum should be NaN")

    def test_minimum_rank(self):
        res1 = self.toBeSorted.minimum(0, 1)
        res2 = self.toBeSorted.minimum(1, 1)
        res3 = self.toBeSorted.minimum(2, 1)
        res4 = self.toBeSorted.minimum(3, 1)
        res5 = self.toBeSorted.minimum(6, 1)
        res6 = self.toBeSorted.minimum(7, 1)
        self.assertTrue(res1.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res2.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res3.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res4.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res5.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res6.rows() == 1, "DataFrame should have 1 row")
        truth = self.toBeSorted.clone().get_rows(2, 3)
        self.assertTrue(res1.equals(truth), "DataFrames should be equal")
        self.assertTrue(res2.equals(truth), "DataFrames should be equal")
        self.assertTrue(res3.equals(truth), "DataFrames should be equal")
        self.assertTrue(res4.equals(truth), "DataFrames should be equal")
        self.assertTrue(res5.equals(truth), "DataFrames should be equal")
        self.assertTrue(res6.equals(truth), "DataFrames should be equal")
        self.assertSequenceAlmostEqual(
            res1.get_column_names(),
            self.toBeSorted.get_column_names(),
            "Column names should be equal")

        res1 = self.toBeSorted.minimum(0, 3)
        res2 = self.toBeSorted.minimum(1, 3)
        res3 = self.toBeSorted.minimum(2, 3)
        res4 = self.toBeSorted.minimum(3, 3)
        res5 = self.toBeSorted.minimum(6, 3)
        res6 = self.toBeSorted.minimum(7, 3)
        self.assertTrue(res1.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res2.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res3.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res4.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res5.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res6.rows() == 3, "DataFrame should have 3 row")
        truth = self.toBeSorted.clone()
        truth.clear()
        truth.add_row(self.toBeSorted.get_row(2))
        truth.add_row(self.toBeSorted.get_row(1))
        truth.add_row(self.toBeSorted.get_row(4))
        self.assertTrue(res1.equals(truth), "DataFrames should be equal")
        self.assertTrue(res2.equals(truth), "DataFrames should be equal")
        self.assertTrue(res3.equals(truth), "DataFrames should be equal")
        self.assertTrue(res4.equals(truth), "DataFrames should be equal")
        self.assertTrue(res5.equals(truth), "DataFrames should be equal")
        self.assertTrue(res6.equals(truth), "DataFrames should be equal")
        self.assertSequenceAlmostEqual(
            res1.get_column_names(),
            self.toBeSorted.get_column_names(),
            "Column names should be equal")

    def test_minimum_rank_by_name(self):
        res1 = self.toBeSorted.minimum("byteCol", 1)
        res2 = self.toBeSorted.minimum("shortCol", 1)
        res3 = self.toBeSorted.minimum("intCol", 1)
        res4 = self.toBeSorted.minimum("longCol", 1)
        res5 = self.toBeSorted.minimum("floatCol", 1)
        res6 = self.toBeSorted.minimum("doubleCol", 1)
        self.assertTrue(res1.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res2.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res3.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res4.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res5.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res6.rows() == 1, "DataFrame should have 1 row")
        truth = self.toBeSorted.clone().get_rows(2, 3)
        self.assertTrue(res1.equals(truth), "DataFrames should be equal")
        self.assertTrue(res2.equals(truth), "DataFrames should be equal")
        self.assertTrue(res3.equals(truth), "DataFrames should be equal")
        self.assertTrue(res4.equals(truth), "DataFrames should be equal")
        self.assertTrue(res5.equals(truth), "DataFrames should be equal")
        self.assertTrue(res6.equals(truth), "DataFrames should be equal")
        self.assertSequenceAlmostEqual(
            res1.get_column_names(),
            self.toBeSorted.get_column_names(),
            "Column names should be equal")

        res1 = self.toBeSorted.minimum("byteCol", 3)
        res2 = self.toBeSorted.minimum("shortCol", 3)
        res3 = self.toBeSorted.minimum("intCol", 3)
        res4 = self.toBeSorted.minimum("longCol", 3)
        res5 = self.toBeSorted.minimum("floatCol", 3)
        res6 = self.toBeSorted.minimum("doubleCol", 3)
        self.assertTrue(res1.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res2.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res3.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res4.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res5.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res6.rows() == 3, "DataFrame should have 3 row")
        truth = self.toBeSorted.clone()
        truth.clear()
        truth.add_row(self.toBeSorted.get_row(2))
        truth.add_row(self.toBeSorted.get_row(1))
        truth.add_row(self.toBeSorted.get_row(4))
        self.assertTrue(res1.equals(truth), "DataFrames should be equal")
        self.assertTrue(res2.equals(truth), "DataFrames should be equal")
        self.assertTrue(res3.equals(truth), "DataFrames should be equal")
        self.assertTrue(res4.equals(truth), "DataFrames should be equal")
        self.assertTrue(res5.equals(truth), "DataFrames should be equal")
        self.assertTrue(res6.equals(truth), "DataFrames should be equal")
        self.assertSequenceAlmostEqual(
            res1.get_column_names(),
            self.toBeSorted.get_column_names(),
            "Column names should be equal")

    def test_minimum_rank_large(self):
        res1 = self.toBeSorted.minimum("byteCol", 15)
        res2 = self.toBeSorted.minimum("shortCol", 15)
        res3 = self.toBeSorted.minimum("intCol", 15)
        res4 = self.toBeSorted.minimum("longCol", 15)
        res5 = self.toBeSorted.minimum("floatCol", 15)
        res6 = self.toBeSorted.minimum("doubleCol", 15)
        self.assertTrue(res1.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res2.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res3.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res4.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res5.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res6.rows() == 3, "DataFrame should have 3 row")
        truth = self.toBeSorted.clone()
        truth.clear()
        truth.add_row(self.toBeSorted.get_row(2))
        truth.add_row(self.toBeSorted.get_row(1))
        truth.add_row(self.toBeSorted.get_row(4))
        self.assertTrue(res1.equals(truth), "DataFrames should be equal")
        self.assertTrue(res2.equals(truth), "DataFrames should be equal")
        self.assertTrue(res3.equals(truth), "DataFrames should be equal")
        self.assertTrue(res4.equals(truth), "DataFrames should be equal")
        self.assertTrue(res5.equals(truth), "DataFrames should be equal")
        self.assertTrue(res6.equals(truth), "DataFrames should be equal")
        self.assertSequenceAlmostEqual(
            res1.get_column_names(),
            self.toBeSorted.get_column_names(),
            "Column names should be equal")

    def test_maximum_rank(self):
        res1 = self.toBeSorted.maximum(0, 1)
        res2 = self.toBeSorted.maximum(1, 1)
        res3 = self.toBeSorted.maximum(2, 1)
        res4 = self.toBeSorted.maximum(3, 1)
        res5 = self.toBeSorted.maximum(6, 1)
        res6 = self.toBeSorted.maximum(7, 1)
        self.assertTrue(res1.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res2.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res3.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res4.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res5.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res6.rows() == 1, "DataFrame should have 1 row")
        truth = self.toBeSorted.clone().get_rows(4, 5)
        self.assertTrue(res1.equals(truth), "DataFrames should be equal")
        self.assertTrue(res2.equals(truth), "DataFrames should be equal")
        self.assertTrue(res3.equals(truth), "DataFrames should be equal")
        self.assertTrue(res4.equals(truth), "DataFrames should be equal")
        self.assertTrue(res5.equals(truth), "DataFrames should be equal")
        self.assertTrue(res6.equals(truth), "DataFrames should be equal")
        self.assertSequenceAlmostEqual(
            res1.get_column_names(),
            self.toBeSorted.get_column_names(),
            "Column names should be equal")

        res1 = self.toBeSorted.maximum(0, 3)
        res2 = self.toBeSorted.maximum(1, 3)
        res3 = self.toBeSorted.maximum(2, 3)
        res4 = self.toBeSorted.maximum(3, 3)
        res5 = self.toBeSorted.maximum(6, 3)
        res6 = self.toBeSorted.maximum(7, 3)
        self.assertTrue(res1.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res2.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res3.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res4.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res5.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res6.rows() == 3, "DataFrame should have 3 row")
        truth = self.toBeSorted.clone()
        truth.clear()
        truth.add_row(self.toBeSorted.get_row(4))
        truth.add_row(self.toBeSorted.get_row(1))
        truth.add_row(self.toBeSorted.get_row(2))
        self.assertTrue(res1.equals(truth), "DataFrames should be equal")
        self.assertTrue(res2.equals(truth), "DataFrames should be equal")
        self.assertTrue(res3.equals(truth), "DataFrames should be equal")
        self.assertTrue(res4.equals(truth), "DataFrames should be equal")
        self.assertTrue(res5.equals(truth), "DataFrames should be equal")
        self.assertTrue(res6.equals(truth), "DataFrames should be equal")
        self.assertSequenceAlmostEqual(
            res1.get_column_names(),
            self.toBeSorted.get_column_names(),
            "Column names should be equal")

    def test_maximum_rank_by_name(self):
        res1 = self.toBeSorted.maximum("byteCol", 1)
        res2 = self.toBeSorted.maximum("shortCol", 1)
        res3 = self.toBeSorted.maximum("intCol", 1)
        res4 = self.toBeSorted.maximum("longCol", 1)
        res5 = self.toBeSorted.maximum("floatCol", 1)
        res6 = self.toBeSorted.maximum("doubleCol", 1)
        self.assertTrue(res1.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res2.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res3.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res4.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res5.rows() == 1, "DataFrame should have 1 row")
        self.assertTrue(res6.rows() == 1, "DataFrame should have 1 row")
        truth = self.toBeSorted.clone().get_rows(4, 5)
        self.assertTrue(res1.equals(truth), "DataFrames should be equal")
        self.assertTrue(res2.equals(truth), "DataFrames should be equal")
        self.assertTrue(res3.equals(truth), "DataFrames should be equal")
        self.assertTrue(res4.equals(truth), "DataFrames should be equal")
        self.assertTrue(res5.equals(truth), "DataFrames should be equal")
        self.assertTrue(res6.equals(truth), "DataFrames should be equal")
        self.assertSequenceAlmostEqual(
            res1.get_column_names(),
            self.toBeSorted.get_column_names(),
            "Column names should be equal")

        res1 = self.toBeSorted.maximum("byteCol", 3)
        res2 = self.toBeSorted.maximum("shortCol", 3)
        res3 = self.toBeSorted.maximum("intCol", 3)
        res4 = self.toBeSorted.maximum("longCol", 3)
        res5 = self.toBeSorted.maximum("floatCol", 3)
        res6 = self.toBeSorted.maximum("doubleCol", 3)
        self.assertTrue(res1.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res2.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res3.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res4.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res5.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res6.rows() == 3, "DataFrame should have 3 row")
        truth = self.toBeSorted.clone()
        truth.clear()
        truth.add_row(self.toBeSorted.get_row(4))
        truth.add_row(self.toBeSorted.get_row(1))
        truth.add_row(self.toBeSorted.get_row(2))
        self.assertTrue(res1.equals(truth), "DataFrames should be equal")
        self.assertTrue(res2.equals(truth), "DataFrames should be equal")
        self.assertTrue(res3.equals(truth), "DataFrames should be equal")
        self.assertTrue(res4.equals(truth), "DataFrames should be equal")
        self.assertTrue(res5.equals(truth), "DataFrames should be equal")
        self.assertTrue(res6.equals(truth), "DataFrames should be equal")
        self.assertSequenceAlmostEqual(
            res1.get_column_names(),#
            self.toBeSorted.get_column_names(),
            "Column names should be equal")

    def test_maximum_rank_large(self):
        res1 = self.toBeSorted.maximum("byteCol", 15)
        res2 = self.toBeSorted.maximum("shortCol", 15)
        res3 = self.toBeSorted.maximum("intCol", 15)
        res4 = self.toBeSorted.maximum("longCol", 15)
        res5 = self.toBeSorted.maximum("floatCol", 15)
        res6 = self.toBeSorted.maximum("doubleCol", 15)
        self.assertTrue(res1.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res2.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res3.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res4.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res5.rows() == 3, "DataFrame should have 3 row")
        self.assertTrue(res6.rows() == 3, "DataFrame should have 3 row")
        truth = self.toBeSorted.clone()
        truth.clear()
        truth.add_row(self.toBeSorted.get_row(4))
        truth.add_row(self.toBeSorted.get_row(1))
        truth.add_row(self.toBeSorted.get_row(2))
        self.assertTrue(res1.equals(truth), "DataFrames should be equal")
        self.assertTrue(res2.equals(truth), "DataFrames should be equal")
        self.assertTrue(res3.equals(truth), "DataFrames should be equal")
        self.assertTrue(res4.equals(truth), "DataFrames should be equal")
        self.assertTrue(res5.equals(truth), "DataFrames should be equal")
        self.assertTrue(res6.equals(truth), "DataFrames should be equal")
        self.assertSequenceAlmostEqual(
            res1.get_column_names(),
            self.toBeSorted.get_column_names(),
            "Column names should be equal")

    def test_minimum_exception(self):
        self.assertRaises(DataFrameException, self.df.minimum, "stringCol")
        self.assertRaises(DataFrameException, self.df.minimum, "binaryCol")

    def test_maximum_exception(self):
        self.assertRaises(DataFrameException, self.df.maximum, "stringCol")
        self.assertRaises(DataFrameException, self.df.maximum, "binaryCol")

    def test_average_exception(self):
        self.assertRaises(DataFrameException, self.df.average, "stringCol")
        self.assertRaises(DataFrameException, self.df.average, "binaryCol")

    def test_median_exception(self):
        self.assertRaises(DataFrameException, self.df.median, "stringCol")

    def test_sum_exception(self):
        self.assertRaises(DataFrameException, self.df.sum, "stringCol")

    def test_minimum_rank_exception(self):
        self.assertRaises(DataFrameException, self.df.minimum, "stringCol", 3)

    def test_minimum_invalid_rank_exception(self):
        self.assertRaises(DataFrameException, self.df.minimum, "intCol", 0)

    def test_maximum_rank_exception(self):
        self.assertRaises(DataFrameException, self.df.maximum, "stringCol", 3)

    def test_maximum_invalid_rank_exception(self):
        self.assertRaises(DataFrameException, self.df.maximum, "intCol", 0)

    def test_absolute(self):
        self.df.set_row(
            2, [-42, -42, -42, -42, "A", "a", -42.12, -42.12, False, bytearray.fromhex("0042")])

        self.df.absolute("byteCol")
        self.df.absolute("shortCol")
        self.df.absolute("intCol")
        self.df.absolute("longCol")
        self.df.absolute("floatCol")
        self.df.absolute("doubleCol")
        self.assertTrue(self.df.get_byte("byteCol", 2) == 42, "Value should be positive")
        self.assertTrue(self.df.get_short("shortCol", 2) == 42, "Value should be positive")
        self.assertTrue(self.df.get_int("intCol", 2) == 42, "Value should be positive")
        self.assertTrue(self.df.get_long("longCol", 2) == 42, "Value should be positive")
        self.assertAlmostEqual(
            self.df.get_float("floatCol", 2), 42.12, places=2, msg="Value should be positive")
        self.assertAlmostEqual(
            self.df.get_double("doubleCol", 2), 42.12, places=2, msg="Value should be positive")

    def test_ceil(self):
        self.df.ceil("intCol")
        self.df.ceil("floatCol")
        self.df.ceil("doubleCol")
        self.assertSequenceAlmostEqual(
            [12, None, 32, None, 52],
            self.df.get_column("intCol").as_array(),
            "Column values are not equal")

        self.assertSequenceAlmostEqual(
            [11, None, 31, None, 51],
            self.df.get_column("floatCol").as_array(),
            "Column values are not equal")

        self.assertSequenceAlmostEqual(
            [12, None, 32, None, 52],
            self.df.get_column("doubleCol").as_array(),
            "Column values are not equal")

    def test_floor(self):
        self.df.floor("intCol")
        self.df.floor("floatCol")
        self.df.floor("doubleCol")
        self.assertSequenceAlmostEqual(
            [12, None, 32, None, 52],
            self.df.get_column("intCol").as_array(),
            "Column values are not equal")

        self.assertSequenceAlmostEqual(
            [10, None, 30, None, 50],
            self.df.get_column("floatCol").as_array(),
            "Column values are not equal")

        self.assertSequenceAlmostEqual(
            [11, None, 31, None, 51],
            self.df.get_column("doubleCol").as_array(),
            "Column values are not equal")

    def test_round(self):
        self.df.set_column("floatCol",
                           NullableFloatColumn(values=[10.2354, None, 30.256, None, 50.515]))

        self.df.set_column("doubleCol",
                           NullableDoubleColumn(values=[10.2354, None, 30.256, None, 50.515]))

        self.df.round("intCol", 2)
        self.df.round("floatCol", 2)
        self.df.round("doubleCol", 2)
        self.assertSequenceAlmostEqual(
            [12, None, 32, None, 52],
            self.df.get_column("intCol").as_array(),
            "Column values are not equal")

        self.assertSequenceAlmostEqual(
            [10.24, None, 30.26, None, 50.52],
            self.df.get_column("floatCol").as_array(),
            "Column values are not equal")

        self.assertSequenceAlmostEqual(
            [10.24, None, 30.26, None, 50.52],
            self.df.get_column("doubleCol").as_array(),
            "Column values are not equal")

        self.df.round("floatCol", 0)
        self.df.round("doubleCol", 0)
        self.assertSequenceAlmostEqual(
            [10.0, None, 30.0, None, 51.0],
            self.df.get_column("floatCol").as_array(),
            "Column values are not equal")

        self.assertSequenceAlmostEqual(
            [10.0, None, 30.0, None, 51.0],
            self.df.get_column("doubleCol").as_array(),
            "Column values are not equal")

    def test_absolute_exception(self):
        self.assertRaises(DataFrameException, self.df.absolute, "stringCol")

    def test_ceil_exception(self):
        self.assertRaises(DataFrameException, self.df.ceil, "stringCol")

    def test_floor_exception(self):
        self.assertRaises(DataFrameException, self.df.floor, "stringCol")

    def test_round_exception(self):
        self.assertRaises(DataFrameException, self.df.round, "stringCol", 2)

    def test_round_invalid_arg_exception(self):
        self.assertRaises(DataFrameException, self.df.round, "floatCol", -1)



    #***************************#
    #           Clip            #
    #***************************#



    def test_clip(self):
        self.df.remove_column("stringCol")
        self.df.remove_column("charCol")
        self.df.remove_column("booleanCol")
        self.df.remove_column("binaryCol")
        self.df.clip("byteCol", 20.0, 40.0)
        self.df.clip("shortCol", 20.0, 40.0)
        self.df.clip("intCol", 20.0, 40.0)
        self.df.clip("longCol", 20.0, 40.0)
        self.df.clip("floatCol", 20.0, 40.0)
        self.df.clip("doubleCol", 20.0, 40.0)
        truth = NullableDataFrame(
            DataFrame.NullableByteColumn("byteCol", [20, None, 30, None, 40]),
            DataFrame.NullableShortColumn("shortCol", [20, None, 31, None, 40]),
            DataFrame.NullableIntColumn("intCol", [20, None, 32, None, 40]),
            DataFrame.NullableLongColumn("longCol", [20 ,None, 33, None, 40]),
            DataFrame.NullableFloatColumn("floatCol", [20.0, None, 30.3, None, 40.0]),
            DataFrame.NullableDoubleColumn("doubleCol", [20.0, None, 31.3, None, 40.0]))

        self.assertTrue(self.df.equals(truth), "DataFrame does not match expected content")

    def test_clip_with_low_unspecified(self):
        self.df.remove_column("stringCol")
        self.df.remove_column("charCol")
        self.df.remove_column("booleanCol")
        self.df.remove_column("binaryCol")
        self.df.clip("byteCol", high=40.0)
        self.df.clip("shortCol", high=40.0)
        self.df.clip("intCol", high=40.0)
        self.df.clip("longCol", high=40.0)
        self.df.clip("floatCol", high=40.0)
        self.df.clip("doubleCol", high=40.0)
        truth = NullableDataFrame(
            DataFrame.NullableByteColumn("byteCol", [10, None, 30, None, 40]),
            DataFrame.NullableShortColumn("shortCol", [11, None, 31, None, 40]),
            DataFrame.NullableIntColumn("intCol", [12, None, 32, None, 40]),
            DataFrame.NullableLongColumn("longCol", [13 ,None, 33, None, 40]),
            DataFrame.NullableFloatColumn("floatCol", [10.1, None, 30.3, None, 40.0]),
            DataFrame.NullableDoubleColumn("doubleCol", [11.1, None, 31.3, None, 40.0]))

        self.assertTrue(self.df.equals(truth), "DataFrame does not match expected content")

    def test_clip_with_high_unspecified(self):
        self.df.remove_column("stringCol")
        self.df.remove_column("charCol")
        self.df.remove_column("booleanCol")
        self.df.remove_column("binaryCol")
        self.df.clip("byteCol", low=20.0)
        self.df.clip("shortCol", low=20.0)
        self.df.clip("intCol", low=20.0)
        self.df.clip("longCol", low=20.0)
        self.df.clip("floatCol", low=20.0)
        self.df.clip("doubleCol", low=20.0)
        truth = NullableDataFrame(
            DataFrame.NullableByteColumn("byteCol", [20, None, 30, None, 50]),
            DataFrame.NullableShortColumn("shortCol", [20, None, 31, None, 51]),
            DataFrame.NullableIntColumn("intCol", [20, None, 32, None, 52]),
            DataFrame.NullableLongColumn("longCol", [20 ,None, 33, None, 53]),
            DataFrame.NullableFloatColumn("floatCol", [20.0, None, 30.3, None, 50.5]),
            DataFrame.NullableDoubleColumn("doubleCol", [20.0, None, 31.3, None, 51.5]))

        self.assertTrue(self.df.equals(truth), "DataFrame does not match expected content")

    def test_clip_invalid_column_arg_exception(self):
        self.assertRaises(DataFrameException, self.df.clip, "stringCol", 20.0, 40.0)

    def test_clip_invalid_range_arg_exception(self):
        self.assertRaises(DataFrameException, self.df.clip, "intCol", 3, 2)



    #*************************#
    #         Sorting         #
    #*************************#



    def test_sort_by_byte(self):
        self.toBeSorted.sort_by("byteCol")
        self.assertDataFrameIsSortedAscend()

    def test_sort_by_short(self):
        self.toBeSorted.sort_by("shortCol")
        self.assertDataFrameIsSortedAscend()

    def test_sort_by_int(self):
        self.toBeSorted.sort_by("intCol")
        self.assertDataFrameIsSortedAscend()

    def test_sort_by_long(self):
        self.toBeSorted.sort_by("longCol")
        self.assertDataFrameIsSortedAscend()

    def test_sort_by_string(self):
        self.toBeSorted.sort_by("stringCol")
        self.assertDataFrameIsSortedAscend()

    def test_sort_by_char(self):
        self.toBeSorted.sort_by("charCol")
        self.assertDataFrameIsSortedAscend()

    def test_sort_by_float(self):
        self.toBeSorted.sort_by("floatCol")
        self.assertDataFrameIsSortedAscend()

    def test_sort_by_double(self):
        self.toBeSorted.sort_by("doubleCol")
        self.assertDataFrameIsSortedAscend()

    def test_sort_by_boolean(self):
        self.toBeSorted.sort_by("booleanCol")
        self.assertFalse(
            self.toBeSorted.get_boolean("booleanCol", 0),
            "Row does not match expected values at row index 0. DataFrame is not sorted correctly")
        self.assertTrue(
            self.toBeSorted.get_boolean("booleanCol", 1),
            "Row does not match expected values at row index 1. DataFrame is not sorted correctly")
        self.assertTrue(
            self.toBeSorted.get_boolean("booleanCol", 2),
            "Row does not match expected values at row index 2. DataFrame is not sorted correctly")
        self.assertTrue(
            self.toBeSorted.get_boolean("booleanCol", 3) is None,
            "Row does not match expected values at row index 3. DataFrame is not sorted correctly")
        self.assertTrue(
            self.toBeSorted.get_boolean("booleanCol", 4) is None,
            "Row does not match expected values at row index 4. DataFrame is not sorted correctly")

    def test_sort_by_binary(self):
        self.toBeSorted.sort_by("binaryCol")
        self.assertDataFrameIsSortedAscend()

    def test_sort_descend_by_byte(self):
        self.toBeSorted.sort_descending_by("byteCol")
        self.assertDataFrameIsSortedDescend()

    def test_sort_descend_by_short(self):
        self.toBeSorted.sort_descending_by("shortCol")
        self.assertDataFrameIsSortedDescend()

    def test_sort_descend_by_int(self):
        self.toBeSorted.sort_descending_by("intCol")
        self.assertDataFrameIsSortedDescend()

    def test_sort_descend_by_long(self):
        self.toBeSorted.sort_descending_by("longCol")
        self.assertDataFrameIsSortedDescend()

    def test_sort_descend_by_string(self):
        self.toBeSorted.sort_descending_by("stringCol")
        self.assertDataFrameIsSortedDescend()

    def test_sort_descend_by_char(self):
        self.toBeSorted.sort_descending_by("charCol")
        self.assertDataFrameIsSortedDescend()

    def test_sort_descend_by_float(self):
        self.toBeSorted.sort_descending_by("floatCol")
        self.assertDataFrameIsSortedDescend()

    def test_sort_descend_by_double(self):
        self.toBeSorted.sort_descending_by("doubleCol")
        self.assertDataFrameIsSortedDescend()

    def test_sort_descend_by_boolean(self):
        self.toBeSorted.sort_descending_by("booleanCol")
        self.assertTrue(
            self.toBeSorted.get_boolean("booleanCol", 0),
            "Row does not match expected values at row index 0. DataFrame is not sorted correctly")
        self.assertTrue(
            self.toBeSorted.get_boolean("booleanCol", 1),
            "Row does not match expected values at row index 1. DataFrame is not sorted correctly")
        self.assertFalse(
            self.toBeSorted.get_boolean("booleanCol", 2),
            "Row does not match expected values at row index 2. DataFrame is not sorted correctly")
        self.assertTrue(
            self.toBeSorted.get_boolean("booleanCol", 3) is None,
            "Row does not match expected values at row index 3. DataFrame is not sorted correctly")
        self.assertTrue(
            self.toBeSorted.get_boolean("booleanCol", 4) is None,
            "Row does not match expected values at row index 4. DataFrame is not sorted correctly")

    def test_sort_descend_by_binary(self):
        self.toBeSorted.sort_descending_by("binaryCol")
        self.assertDataFrameIsSortedDescend()

    def test_sort_ascend_with_nans(self):
        df = NullableDataFrame(
            DataFrame.NullableIntColumn("A", [4, 2, 1, 5, 3]),
            DataFrame.NullableStringColumn("B", ["4", "2", "1", "5", "3"]),
            DataFrame.NullableFloatColumn("C", [None, float("NaN"), 1.0, float("NaN"), 3.0]),
            DataFrame.NullableDoubleColumn("D", [float("NaN"), None, None, 5.0, float("NaN")]))

        df.sort_by("C")
        vals = df.get_column("C").as_array()
        for i, truth in enumerate([1.0, 3.0, float("NaN"), float("NaN"), None]):
            if truth is None:
                self.assertTrue(vals[i] is None, "DataFrame is not sorted correctly")
            elif math.isnan(truth):
                self.assertTrue(math.isnan(vals[i]), "DataFrame is not sorted correctly")
            else:
                self.assertTrue(truth == vals[i], "DataFrame is not sorted correctly")

        df.sort_by("D")
        vals = df.get_column("D").as_array()
        for i, truth in enumerate([5.0, float("NaN"), float("NaN"), None, None]):
            if truth is None:
                self.assertTrue(vals[i] is None, "DataFrame is not sorted correctly")
            elif math.isnan(truth):
                self.assertTrue(math.isnan(vals[i]), "DataFrame is not sorted correctly")
            else:
                self.assertTrue(truth == vals[i], "DataFrame is not sorted correctly")

    def test_sort_ascend_only_nans_and_nulls(self):
        nan = float("NaN")
        df = NullableDataFrame(
            DataFrame.NullableIntColumn("A", [4, 2, 1, 5, 3]),
            DataFrame.NullableStringColumn("B", ["4", "2", "1", "5", "3"]),
            DataFrame.NullableFloatColumn("C", [nan, None, nan, None, nan]),
            DataFrame.NullableDoubleColumn("D", [None, None, nan, nan, nan]))

        df.sort_by("C")
        vals = df.get_column("C").as_array()
        for i in range(3):
            self.assertTrue(math.isnan(vals[i]), "DataFrame is not sorted correctly")
        for i in range(3, 5, 1):
            self.assertTrue(vals[i] is None, "DataFrame is not sorted correctly")

        df.sort_by("D")
        vals = df.get_column("D").as_array()
        for i in range(3):
            self.assertTrue(math.isnan(vals[i]), "DataFrame is not sorted correctly")
        for i in range(3, 5, 1):
            self.assertTrue(vals[i] is None, "DataFrame is not sorted correctly")

    def test_sort_descend_with_nans(self):
        df = NullableDataFrame(
            DataFrame.NullableIntColumn("A", [4, 2, 1, 5, 3]),
            DataFrame.NullableStringColumn("B", ["4", "2", "1", "5", "3"]),
            DataFrame.NullableFloatColumn("C", [4.0, float("NaN"), 1.0, float("NaN"), None]),
            DataFrame.NullableDoubleColumn("D", [float("NaN"), 2.0, None, None, float("NaN")]))

        df.sort_descending_by("C")
        vals = df.get_column("C").as_array()
        for i, truth in enumerate([4.0, 1.0, float("NaN"), float("NaN"), None]):
            if truth is None:
                self.assertTrue(vals[i] is None, "DataFrame is not sorted correctly")
            elif math.isnan(truth):
                self.assertTrue(math.isnan(vals[i]), "DataFrame is not sorted correctly")
            else:
                self.assertTrue(truth == vals[i], "DataFrame is not sorted correctly")

        df.sort_descending_by("D")
        vals = df.get_column("D").as_array()
        for i, truth in enumerate([2.0, float("NaN"), float("NaN"), None, None]):
            if truth is None:
                self.assertTrue(vals[i] is None, "DataFrame is not sorted correctly")
            elif math.isnan(truth):
                self.assertTrue(math.isnan(vals[i]), "DataFrame is not sorted correctly")
            else:
                self.assertTrue(truth == vals[i], "DataFrame is not sorted correctly")

    def test_sort_descend_only_nans_and_nulls(self):
        nan = float("NaN")
        df = NullableDataFrame(
            DataFrame.NullableIntColumn("A", [4, 2, 1, 5, 3]),
            DataFrame.NullableStringColumn("B", ["4", "2", "1", "5", "3"]),
            DataFrame.NullableFloatColumn("C", [None, nan, nan, None, None]),
            DataFrame.NullableDoubleColumn("D", [nan, None, nan, nan, None]))

        df.sort_descending_by("C")
        vals = df.get_column("C").as_array()
        for i in range(2):
            self.assertTrue(math.isnan(vals[i]), "DataFrame is not sorted correctly")
        for i in range(2, 5, 1):
            self.assertTrue(vals[i] is None, "DataFrame is not sorted correctly")

        df.sort_descending_by("D")
        vals = df.get_column("D").as_array()
        for i in range(3):
            self.assertTrue(math.isnan(vals[i]), "DataFrame is not sorted correctly")
        for i in range(3, 5, 1):
            self.assertTrue(vals[i] is None, "DataFrame is not sorted correctly")



    #***************************************#
    #         Resizing and Flushing         #
    #***************************************#



    def test_space_alteration(self):
        #initial row count is 5
        #add 5 rows
        for _ in range(5):
            self.df.add_row([42,42,42,42,"42","A",42.2,42.2,True,bytearray.fromhex("00000080")])

        self.assertTrue(self.df.rows() == 10, "Row count should be 10")
        self.assertTrue(self.df.capacity() == 10, "Capacity should be 10")
        #add another row to trigger resizing
        self.df.add_row([42,42,42,42,"42","A",42.2,42.2,True,bytearray.fromhex("00000080")])
        #one additional row but capacity should have doubled
        self.assertTrue(self.df.rows() == 11, "Row count should be 11")
        self.assertTrue(self.df.capacity() == 20, "Capacity should be 20")

        #add more rows
        for _ in range(10):
            self.df.add_row([42,42,42,42,"42","A",42.2,42.2,True,bytearray.fromhex("00000080")])

        self.assertTrue(self.df.rows() == 21, "Row count should be 21")
        self.assertTrue(self.df.capacity() == 40, "Capacity should be 40")
        #flush back to 21
        self.df.flush()
        self.assertTrue(self.df.rows() == 21, "Row count should be 21")
        self.assertTrue(self.df.capacity() == 21, "Capacity should be 21")
        self.df.add_row([42,42,42,42,"42","A",42.2,42.2,True,bytearray.fromhex("00000080")])
        self.assertTrue(self.df.rows() == 22, "Row count should be 22")
        self.assertTrue(self.df.capacity() == 42, "Capacity should be 42")

        #remove 19 rows which should cause an automatic flush operation
        #with an applied buffer of 4
        self.df.remove_rows(from_index=0, to_index=19)
        self.assertTrue(self.df.rows() == 3, "Row count should be 3")
        self.assertTrue(self.df.capacity() == 7, "Capacity should be 7")

        #add again
        for _ in range(5):
            self.df.add_row([42,42,42,42,"42","A",42.2,42.2,True,bytearray.fromhex("00000080")])

        self.assertTrue(self.df.rows() == 8, "Row count should be 8")
        self.assertTrue(self.df.capacity() == 14, "Capacity should be 14")



    #***************************************#
    #          Equals and HashCode          #
    #***************************************#



    def test_equals_hash_code_contract(self):
        names = ["BYTE", "SHORT", "INT", "LONG", "STRING",
                 "CHAR", "FLOAT", "DOUBLE", "BOOLEAN"]

        test1 = NullableDataFrame(
            NullableByteColumn(values=[1, None, 3]),
            NullableShortColumn(values=[1, 2, 3]),
            NullableIntColumn(values=[1, 2, 3]),
            NullableLongColumn(values=[1, 2, 3]),
            NullableStringColumn(values=["1", "2", "3"]),
            NullableCharColumn(values=["1", "2", "3"]),
            NullableFloatColumn(values=[None, 2.0, 3.0]),
            NullableDoubleColumn(values=[1.0, 2.0, 3.0]),
            NullableBooleanColumn(values=[True, False, True]))

        test2 = NullableDataFrame(
            NullableByteColumn(values=[1, None, 3]),
            NullableShortColumn(values=[1, 2, 3]),
            NullableIntColumn(values=[1, 2, 3]),
            NullableLongColumn(values=[1, 2, 3]),
            NullableStringColumn(values=["1", "2", "3"]),
            NullableCharColumn(values=["1", "2", "3"]),
            NullableFloatColumn(values=[None, 2.0, 3.0]),
            NullableDoubleColumn(values=[1.0, 2.0, 3.0]),
            NullableBooleanColumn(values=[True, False, True]))

        test1.set_column_names(names)
        test2.set_column_names(names)
        self.assertTrue(test1.equals(test2), "Equals method should return true")
        self.assertTrue(test1.hash_code() == test2.hash_code(),
                        "HashCode method should return the same hash code")

        self.assertTrue(test1 == test2, "DataFrames should be equal")
        self.assertTrue(hash(test1) == hash(test2), "Hash code should be equal")

        # change to make unequal
        test1.set_byte("BYTE", 2, 42)
        self.assertFalse(test1.equals(test2), "Equals method should return false")
        self.assertFalse(test1 == test2, "DataFrames should not be equal")



    #***************************************#
    #               Utilities               #
    #***************************************#



    def test_to_array(self):
        self.df.remove_row(4)
        a = self.df.to_array()
        self.assertTrue(isinstance(a, list), "Returned object should be a list")
        col = a[0] # NullableByteColumn
        self.assertTrue(isinstance(col, list), "Column object should be a list")
        self.assertTrue(len(col) == self.df.rows(), "Column list length does not match expected")
        for i, elem in enumerate(col):
            if elem is not None:
                self.assertTrue(isinstance(elem, int), "Invalid column list element type")
            self.assertTrue(self.df.get_byte(0, i) == elem, "Value does not match")

        col = a[1] # NullableShortColumn
        self.assertTrue(isinstance(col, list), "Column object should be a list")
        self.assertTrue(len(col) == self.df.rows(), "Column list length does not match expected")
        for i, elem in enumerate(col):
            if elem is not None:
                self.assertTrue(isinstance(elem, int), "Invalid column list element type")
            self.assertTrue(self.df.get_short(1, i) == elem, "Value does not match")

        col = a[2] # NullableIntColumn
        self.assertTrue(isinstance(col, list), "Column object should be a list")
        self.assertTrue(len(col) == self.df.rows(), "Column list length does not match expected")
        for i, elem in enumerate(col):
            if elem is not None:
                self.assertTrue(isinstance(elem, int), "Invalid column list element type")
            self.assertTrue(self.df.get_int(2, i) == elem, "Value does not match")

        col = a[3] # NullableLongColumn
        self.assertTrue(isinstance(col, list), "Column object should be a list")
        self.assertTrue(len(col) == self.df.rows(), "Column list length does not match expected")
        for i, elem in enumerate(col):
            if elem is not None:
                self.assertTrue(isinstance(elem, int), "Invalid column list element type")
            self.assertTrue(self.df.get_long(3, i) == elem, "Value does not match")

        col = a[4] # NullableStringColumn
        self.assertTrue(isinstance(col, list), "Column object should be a list")
        self.assertTrue(len(col) == self.df.rows(), "Column list length does not match expected")
        for i, elem in enumerate(col):
            if elem is not None:
                self.assertTrue(isinstance(elem, str), "Invalid column list element type")
            self.assertTrue(self.df.get_string(4, i) == elem, "Value does not match")

        col = a[5] # NullableCharColumn
        self.assertTrue(isinstance(col, list), "Column object should be a list")
        self.assertTrue(len(col) == self.df.rows(), "Column list length does not match expected")
        for i, elem in enumerate(col):
            if elem is not None:
                self.assertTrue(isinstance(elem, str), "Invalid column list element type")
            self.assertTrue(self.df.get_char(5, i) == elem, "Value does not match")

        col = a[6] # NullableFloatColumn
        self.assertTrue(isinstance(col, list), "Column object should be a list")
        self.assertTrue(len(col) == self.df.rows(), "Column list length does not match expected")
        for i, elem in enumerate(col):
            if elem is not None:
                self.assertTrue(isinstance(elem, float), "Invalid column list element type")
            self.assertTrue(self.df.get_float(6, i) == elem, "Value does not match")

        col = a[7] # NullableDoubleColumn
        self.assertTrue(isinstance(col, list), "Column object should be a list")
        self.assertTrue(len(col) == self.df.rows(), "Column list length does not match expected")
        for i, elem in enumerate(col):
            if elem is not None:
                self.assertTrue(isinstance(elem, float), "Invalid column list element type")
            self.assertTrue(self.df.get_double(7, i) == elem, "Value does not match")

        col = a[8] # NullableBooleanColumn
        self.assertTrue(isinstance(col, list), "Column object should be a list")
        self.assertTrue(len(col) == self.df.rows(), "Column list length does not match expected")
        for i, elem in enumerate(col):
            if elem is not None:
                self.assertTrue(isinstance(elem, bool), "Invalid column list element type")
            self.assertTrue(self.df.get_boolean(8, i) == elem, "Value does not match")

        col = a[9] # NullableBinaryColumn
        self.assertTrue(isinstance(col, list), "Column object should be a list")
        self.assertTrue(len(col) == self.df.rows(), "Column list length does not match expected")
        for i, elem in enumerate(col):
            if elem is not None:
                self.assertTrue(isinstance(elem, bytearray), "Invalid column list element type")
            self.assertTrue(self.df.get_binary(9, i) == elem, "Value does not match")

    def test_to_array_from_uninitialized(self):
        df = NullableDataFrame()
        a = df.to_array()
        self.assertTrue(a is None, "Returned value should be None")



if __name__ == "__main__":
    unittest.main()
