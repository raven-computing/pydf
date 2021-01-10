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
Tests for Column construction and static functions.
"""

import unittest

import numpy as np

from raven.struct.dataframe.bytecolumn import ByteColumn, NullableByteColumn
from raven.struct.dataframe.shortcolumn import ShortColumn, NullableShortColumn
from raven.struct.dataframe.intcolumn import IntColumn, NullableIntColumn
from raven.struct.dataframe.longcolumn import LongColumn, NullableLongColumn
from raven.struct.dataframe.floatcolumn import FloatColumn, NullableFloatColumn
from raven.struct.dataframe.doublecolumn import DoubleColumn, NullableDoubleColumn
from raven.struct.dataframe.stringcolumn import StringColumn, NullableStringColumn
from raven.struct.dataframe.charcolumn import CharColumn, NullableCharColumn
from raven.struct.dataframe.booleancolumn import BooleanColumn, NullableBooleanColumn
from raven.struct.dataframe.binarycolumn import BinaryColumn, NullableBinaryColumn

import raven.struct.dataframe.column as column

# pylint: disable=invalid-name, missing-function-docstring

class TestColumn(unittest.TestCase):
    """Tests for Column construction and static functions."""



    #***************************************#
    #              Constructors             #
    #***************************************#



    def test_construct_bytecolumn(self):
        col = ByteColumn(values=[11, 22, 33, 44, 55])
        self.assertTrue(col.type_code() == ByteColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "byte")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_bytecolumn(self):
        col = ByteColumn("colname", [11, 22, 33, 44, 55])
        self.assertTrue(col.type_code() == ByteColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "byte")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_bytecolumn(self):
        values = np.array([11, 22, 33, 44, 55], dtype=np.int8)
        col = ByteColumn("colname", values)
        self.assertTrue(col.type_code() == ByteColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "byte")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_shortcolumn(self):
        col = ShortColumn(values=[11, 22, 33, 44, 55])
        self.assertTrue(col.type_code() == ShortColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "short")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_shortcolumn(self):
        col = ShortColumn("colname", [11, 22, 33, 44, 55])
        self.assertTrue(col.type_code() == ShortColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "short")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_shortcolumn(self):
        values = np.array([11, 22, 33, 44, 55], dtype=np.int16)
        col = ShortColumn("colname", values)
        self.assertTrue(col.type_code() == ShortColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "short")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_intcolumn(self):
        col = IntColumn(values=[11, 22, 33, 44, 55])
        self.assertTrue(col.type_code() == IntColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "int")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_intcolumn(self):
        col = IntColumn("colname", [11, 22, 33, 44, 55])
        self.assertTrue(col.type_code() == IntColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "int")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_intcolumn(self):
        values = np.array([11, 22, 33, 44, 55], dtype=np.int32)
        col = IntColumn("colname", values)
        self.assertTrue(col.type_code() == IntColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "int")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_longcolumn(self):
        col = LongColumn(values=[11, 22, 33, 44, 55])
        self.assertTrue(col.type_code() == LongColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "long")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_longcolumn(self):
        col = LongColumn("colname", [11, 22, 33, 44, 55])
        self.assertTrue(col.type_code() == LongColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "long")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_longcolumn(self):
        values = np.array([11, 22, 33, 44, 55], dtype=np.int64)
        col = LongColumn("colname", values)
        self.assertTrue(col.type_code() == LongColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "long")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_floatcolumn(self):
        col = FloatColumn(values=[11.1, 22.2, 33.3, 44.4, 55.5])
        self.assertTrue(col.type_code() == FloatColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "float")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_floatcolumn(self):
        col = FloatColumn("colname", values=[11.1, 22.2, 33.3, 44.4, 55.5])
        self.assertTrue(col.type_code() == FloatColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "float")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_floatcolumn(self):
        values = np.array([11.1, 22.2, 33.3, 44.4, 55.5], dtype=np.float32)
        col = FloatColumn("colname", values)
        self.assertTrue(col.type_code() == FloatColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "float")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_doublecolumn(self):
        col = DoubleColumn(values=[11.1, 22.2, 33.3, 44.4, 55.5])
        self.assertTrue(col.type_code() == DoubleColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "double")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_doublecolumn(self):
        col = DoubleColumn("colname", values=[11.1, 22.2, 33.3, 44.4, 55.5])
        self.assertTrue(col.type_code() == DoubleColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "double")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_doublecolumn(self):
        values = np.array([11.1, 22.2, 33.3, 44.4, 55.5], dtype=np.float64)
        col = DoubleColumn("colname", values)
        self.assertTrue(col.type_code() == DoubleColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "double")
        self.assertFalse(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_stringcolumn(self):
        col = StringColumn(values=["AAA", "AAB", "AAC", "AAD", "AAE"])
        self.assertTrue(col.type_code() == StringColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "string")
        self.assertFalse(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_stringcolumn(self):
        col = StringColumn("colname", values=["AAA", "AAB", "AAC", "AAD", "AAE"])
        self.assertTrue(col.type_code() == StringColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "string")
        self.assertFalse(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_stringcolumn(self):
        values = np.array(["AAA", "AAB", "AAC", "AAD", "AAE"], dtype=np.object)
        col = StringColumn("colname", values)
        self.assertTrue(col.type_code() == StringColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "string")
        self.assertFalse(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_charcolumn(self):
        col = CharColumn(values=["A", "B", "C", "D", "E"])
        self.assertTrue(col.type_code() == CharColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "char")
        self.assertFalse(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_charcolumn(self):
        col = CharColumn("colname", values=["A", "B", "C", "D", "E"])
        self.assertTrue(col.type_code() == CharColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "char")
        self.assertFalse(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_charcolumn(self):
        # specify as ASCII-values
        #                   A   B   C   D   E
        values = np.array([65, 66, 67, 68, 69], dtype=np.uint8)
        col = CharColumn("colname", values)
        self.assertTrue(col.type_code() == CharColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "char")
        self.assertFalse(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_booleancolumn(self):
        col = BooleanColumn(values=[True, False, True, False, True])
        self.assertTrue(col.type_code() == BooleanColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "boolean")
        self.assertFalse(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_booleancolumn(self):
        col = BooleanColumn("colname", values=[True, False, True, False, True])
        self.assertTrue(col.type_code() == BooleanColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "boolean")
        self.assertFalse(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_booleancolumn(self):
        values = np.array([True, False, True, False, True], dtype=np.bool)
        col = BooleanColumn("colname", values)
        self.assertTrue(col.type_code() == BooleanColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "boolean")
        self.assertFalse(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_binarycolumn(self):
        col = BinaryColumn(values=[bytearray.fromhex("0101"),
                                   bytearray.fromhex("0202"),
                                   bytearray.fromhex("0303"),
                                   bytearray.fromhex("0404"),
                                   bytearray.fromhex("0505")])

        self.assertTrue(col.type_code() == BinaryColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "binary")
        self.assertFalse(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_binarycolumn(self):
        col = BinaryColumn("colname",
                           values=[bytearray.fromhex("0101"),
                                   bytearray.fromhex("0202"),
                                   bytearray.fromhex("0303"),
                                   bytearray.fromhex("0404"),
                                   bytearray.fromhex("0505")])

        self.assertTrue(col.type_code() == BinaryColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "binary")
        self.assertFalse(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_binarycolumn(self):
        values = np.array([bytearray.fromhex("01"),
                           bytearray.fromhex("0202"),
                           bytearray.fromhex("0303ff"),
                           bytearray.fromhex("0404"),
                           bytearray.fromhex("0505ff")], dtype=np.object)

        col = BinaryColumn("colname", values)
        self.assertTrue(col.type_code() == BinaryColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "binary")
        self.assertFalse(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_nullablebytecolumn(self):
        col = NullableByteColumn(values=[11, None, 33, 44, None])
        self.assertTrue(col.type_code() == NullableByteColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "byte")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_nullablebytecolumn(self):
        col = NullableByteColumn("colname", [None, 22, 33, 44, None])
        self.assertTrue(col.type_code() == NullableByteColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "byte")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_nullablebytecolumn(self):
        values = np.array([None, 22, 33, None, 55], dtype=np.object)
        col = NullableByteColumn("colname", values)
        self.assertTrue(col.type_code() == NullableByteColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "byte")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_nullableshortcolumn(self):
        col = NullableShortColumn(values=[11, None, 33, None, 55])
        self.assertTrue(col.type_code() == NullableShortColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "short")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_nullableshortcolumn(self):
        col = NullableShortColumn("colname", [None, 22, 33, 44, None])
        self.assertTrue(col.type_code() == NullableShortColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "short")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_nullableshortcolumn(self):
        values = np.array([None, 22, 33, None, 55], dtype=np.object)
        col = NullableShortColumn("colname", values)
        self.assertTrue(col.type_code() == NullableShortColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "short")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_nullableintcolumn(self):
        col = NullableIntColumn(values=[11, None, 33, 44, None])
        self.assertTrue(col.type_code() == NullableIntColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "int")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_nullableintcolumn(self):
        col = NullableIntColumn("colname", [None, 22, 33, 44, None])
        self.assertTrue(col.type_code() == NullableIntColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "int")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_nullableintcolumn(self):
        values = np.array([None, 22, None, 44, None], dtype=np.object)
        col = NullableIntColumn("colname", values)
        self.assertTrue(col.type_code() == NullableIntColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "int")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_nullablelongcolumn(self):
        col = NullableLongColumn(values=[None, 22, 33, None, 55])
        self.assertTrue(col.type_code() == NullableLongColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "long")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_nullablelongcolumn(self):
        col = NullableLongColumn("colname", [11, None, 33, 44, None])
        self.assertTrue(col.type_code() == NullableLongColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "long")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_nullablelongcolumn(self):
        values = np.array([None, 22, None, 44, None], dtype=np.object)
        col = NullableLongColumn("colname", values)
        self.assertTrue(col.type_code() == NullableLongColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "long")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_nullablefloatcolumn(self):
        col = NullableFloatColumn(values=[None, 22.2, 33.3, None, None])
        self.assertTrue(col.type_code() == NullableFloatColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "float")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_nullablefloatcolumn(self):
        col = NullableFloatColumn("colname", values=[11.1, None, 33.3, None, 55.5])
        self.assertTrue(col.type_code() == NullableFloatColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "float")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_nullablefloatcolumn(self):
        values = np.array([11.1, None, 33.3, None, 55.5], dtype=np.object)
        col = NullableFloatColumn("colname", values)
        self.assertTrue(col.type_code() == NullableFloatColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "float")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_nullabledoublecolumn(self):
        col = NullableDoubleColumn(values=[11.1, None, 33.3, 44.4, None])
        self.assertTrue(col.type_code() == NullableDoubleColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "double")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_nullabledoublecolumn(self):
        col = NullableDoubleColumn("colname", values=[11.1, None, 33.3, 44.4, None])
        self.assertTrue(col.type_code() == NullableDoubleColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "double")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_nullabledoublecolumn(self):
        values = np.array([11.1, None, 33.3, None, 55.5], dtype=np.object)
        col = NullableDoubleColumn("colname", values)
        self.assertTrue(col.type_code() == NullableDoubleColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "double")
        self.assertTrue(col.is_nullable())
        self.assertTrue(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_nullablestringcolumn(self):
        col = NullableStringColumn(values=["AAA", None, "AAC", "", None])
        self.assertTrue(col.type_code() == NullableStringColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "string")
        self.assertTrue(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_nullablestringcolumn(self):
        col = NullableStringColumn("colname", values=[None, None, "", "AAD", "AAE"])
        self.assertTrue(col.type_code() == NullableStringColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "string")
        self.assertTrue(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_nullablestringcolumn(self):
        values = np.array([None, "AAB", "AAC", "AAD", None], dtype=np.object)
        col = NullableStringColumn("colname", values)
        self.assertTrue(col.type_code() == NullableStringColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "string")
        self.assertTrue(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_nullablecharcolumn(self):
        col = NullableCharColumn(values=["A", None, "C", None, "E"])
        self.assertTrue(col.type_code() == NullableCharColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "char")
        self.assertTrue(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_nullablecharcolumn(self):
        col = NullableCharColumn("colname", values=[None, "B", "C", None, None])
        self.assertTrue(col.type_code() == NullableCharColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "char")
        self.assertTrue(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_nullablecharcolumn(self):
        values = np.array(["A", None, "C", None, "E"], dtype=np.object)
        col = NullableCharColumn("colname", values)
        self.assertTrue(col.type_code() == NullableCharColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "char")
        self.assertTrue(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_nullablebooleancolumn(self):
        col = NullableBooleanColumn(values=[True, None, None, False, True])
        self.assertTrue(col.type_code() == NullableBooleanColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "boolean")
        self.assertTrue(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_nullablebooleancolumn(self):
        col = NullableBooleanColumn("colname", values=[None, False, True, None, True])
        self.assertTrue(col.type_code() == NullableBooleanColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "boolean")
        self.assertTrue(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_nullablebooleancolumn(self):
        values = np.array([True, None, None, False, None], dtype=np.object)
        col = NullableBooleanColumn("colname", values)
        self.assertTrue(col.type_code() == NullableBooleanColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "boolean")
        self.assertTrue(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_nullablebinarycolumn(self):
        col = NullableBinaryColumn(values=[bytearray.fromhex("0101"),
                                           bytearray.fromhex("0202"),
                                           None,
                                           bytearray.fromhex("0404"),
                                           None])

        self.assertTrue(col.type_code() == NullableBinaryColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "binary")
        self.assertTrue(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() is None)
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_nullablebinarycolumn(self):
        col = NullableBinaryColumn("colname",
                                   values=[None,
                                           bytearray.fromhex("0202"),
                                           bytearray.fromhex("0303"),
                                           bytearray.fromhex("0404"),
                                           None])

        self.assertTrue(col.type_code() == NullableBinaryColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "binary")
        self.assertTrue(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)

    def test_construct_named_numpy_nullablebinarycolumn(self):
        values = np.array([bytearray.fromhex("01"),
                           None,
                           bytearray.fromhex("0303ff"),
                           None,
                           bytearray.fromhex("0505ff")], dtype=np.object)

        col = NullableBinaryColumn("colname", values)
        self.assertTrue(col.type_code() == NullableBinaryColumn.TYPE_CODE)
        self.assertTrue(col.type_name() == "binary")
        self.assertTrue(col.is_nullable())
        self.assertFalse(col.is_numeric())
        self.assertTrue(col.get_name() == "colname")
        self.assertTrue(col.capacity() == 5)



    #********************************************#
    #              Utility Functions             #
    #********************************************#


    def test_static_like(self):
        col1 = IntColumn("myCol", [11, 22, 33, 44, 55])
        col2 = column.Column.like(col1, length=15)
        self.assertTrue(isinstance(col2, IntColumn))
        self.assertTrue(col2.get_name() == col1.get_name())
        self.assertTrue(col2.capacity() == 15)

    def test_static_like_zero_length(self):
        col1 = IntColumn("myCol", [11, 22, 33, 44, 55])
        col2 = column.Column.like(col1)
        self.assertTrue(isinstance(col2, IntColumn))
        self.assertTrue(col2.get_name() == col1.get_name())
        self.assertTrue(col2.capacity() == 0)

    def test_static_like_none_arg(self):
        col1 = column.Column.like(col=None)
        self.assertTrue(col1 is None)

    def test_static_of_type(self):
        col2 = column.Column.of_type(IntColumn.TYPE_CODE, length=15)
        self.assertTrue(isinstance(col2, IntColumn))
        self.assertTrue(col2.capacity() == 15)

    def test_static_of_type_zero_length(self):
        col2 = column.Column.of_type(IntColumn.TYPE_CODE)
        self.assertTrue(isinstance(col2, IntColumn))
        self.assertTrue(col2.capacity() == 0)

    def test_static_of_type_none_type(self):
        col2 = column.Column.of_type(type_code=None)
        self.assertTrue(col2 is None)
