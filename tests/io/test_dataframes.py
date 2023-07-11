# Copyright (C) 2022 Raven Computing
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
Tests for DataFrame serialization and file I/O implementation.
"""

import os

import unittest

from raven.struct.dataframe import (DataFrame,
                                    DefaultDataFrame,
                                    NullableDataFrame)

# pylint: disable=invalid-name, missing-function-docstring

class TestDataFramesIO(unittest.TestCase):
    """Tests for DataFrame serialization and file I/O implementation."""

    DIR_TEST_RESOURCES = os.path.dirname(__file__)
    FILE_DEFAULT = os.path.join(os.path.dirname(__file__), "test_default.df")
    FILE_NULLABLE = os.path.join(os.path.dirname(__file__), "test_nullable.df")

    column_names = None
    df_default = None
    truth = None
    truth_compressed = None

    df_nullable = None
    truth_nullable = None
    truth_nullable_compressed = None

    truth_base64 = None
    truth_nullable_base64 = None

    def setUp(self):
        TestDataFramesIO.column_names = [
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

        TestDataFramesIO.df_default = DefaultDataFrame(
            DataFrame.ByteColumn(values=[10, 20, 30, 40, 50]),
            DataFrame.ShortColumn(values=[11, 21, 31, 41, 51]),
            DataFrame.IntColumn(values=[12, 22, 32, 42, 52]),
            DataFrame.LongColumn(values=[13, 23, 33, 43, 53]),
            DataFrame.StringColumn(values=["10", "20", "30", "40", "50"]),
            DataFrame.CharColumn(values=["a", "b", "c", "d", "e"]),
            DataFrame.FloatColumn(values=[10.1, 20.2, 30.3, 40.4, 50.5]),
            DataFrame.DoubleColumn(values=[11.1, 21.2, 31.3, 41.4, 51.5]),
            DataFrame.BooleanColumn(values=[True, False, True, False, True]),
            DataFrame.BinaryColumn(values=[
                bytearray.fromhex("0102030405"),
                bytearray.fromhex("0504030201"),
                bytearray.fromhex("0502010203"),
                bytearray.fromhex("0201040503"),
                bytearray.fromhex("0301020504")
            ]))

        # expected uncompressed
        TestDataFramesIO.truth = bytearray.fromhex(
            ("7b763a323b64000000050000000a62797465436f6c0073686f7274436f6c00696e74436f6"
             "c006c6f6e67436f6c00737472696e67436f6c0063686172436f6c00666c6f6174436f6c00"
             "646f75626c65436f6c00626f6f6c65616e436f6c0062696e617279436f6c0001020304050"
             "8060709137d0a141e2832000b0015001f002900330000000c00000016000000200000002a"
             "00000034000000000000000d00000000000000170000000000000021000000000000002b0"
             "00000000000003531300032300033300034300035300061626364654121999a41a1999a41"
             "f266664221999a424a000040263333333333334035333333333333403f4ccccccccccd404"
             "4b333333333334049c00000000000a8000000050102030405000000050504030201000000"
             "050502010203000000050201040503000000050301020504"))

        # expected compressed
        TestDataFramesIO.truth_compressed = bytearray.fromhex(
            ("6466ab2eb332b24e616060600562aea4ca9254e7fc1c86e28"
             "cfca2121023330f4ce5e4e7a583c54b8a3221ace48cc42210"
             "9d96939f085692925f9a9403d69c949f9f939a98076666e62"
             "5165582588c4ccc2cac1c6cec9cc2b55c22721a460cdc0ca2"
             "0cf20c9a0cc6407b7980580c881580580b884d182080174a8"
             "b43694528ad0da54d0d0d188c0c188c0d184c0c184c0d1812"
             "939253521d1567ce725c08c49fd2d29c806c272f060607356"
             "330703085d2f63e6780e0ac83cb6608dff300d8c015a08000"
             "3b15c460656166620433988062200613230b2b98c1ccc8c4c"
             "a020011103dae"))

        # expected Base64
        TestDataFramesIO.truth_base64 = (
            "ZGarLrMysk5hYGBgBWKupMqSVOf8HIbijPyiEhAjMw9M5eTnpYPFS4oyIaz"
            "kjMQiEJ2Wk58IVpKSX5qUA9aclJ+fk5qYB2Zm5iUWVYJYjEzMLKwcbOycwr"
            "VcInIaRgzcDKIM8gyaDMZAe3mAWAyIFYBYC4hNGCCAF0qLQ2lFKK0NpU0ND"
            "RiMDBiMDRhMDBhMDRgSk5JTUh0VZ85yXAjEn9LSnIBsJy8GBgc1YzBwMIXS"
            "9j5ngOCsg8tmCN/zANjAFaCAADsVxGBlYWZiBDOYgGIgBhMjCyuYwczIxMo"
            "CABEQPa4=")



        #*************************************************************#
        #                                                             #
        #                 Data for NullableDataFrame                  #
        #                                                             #
        #*************************************************************#



        TestDataFramesIO.df_nullable = NullableDataFrame(
            DataFrame.NullableByteColumn(values=[10, None, None, 0, 50]),
            DataFrame.NullableShortColumn(values=[11, 21, None, 0, None]),
            DataFrame.NullableIntColumn(values=[12, None, 32, 0, None]),
            DataFrame.NullableLongColumn(values=[None, None, 33, 0, 53]),
            DataFrame.NullableStringColumn(values=[
                "ABCD", "2!\"0,.", None, "", "#5{=0>}"]),
            DataFrame.NullableCharColumn(values=[",", "b", None, "d", "?"]),
            DataFrame.NullableFloatColumn(values=[10.1, None, 0.0, None, 50.5]),
            DataFrame.NullableDoubleColumn(values=[None, 0.0, 0.0, None, 51.5]),
            DataFrame.NullableBooleanColumn(values=[True, None, False, None, True]),
            DataFrame.NullableBinaryColumn(values=[
                bytearray.fromhex("00"),
                bytearray.fromhex("0504030201"),
                None,
                bytearray.fromhex("020104054a0503"),
                None
            ]))

        TestDataFramesIO.truth_nullable = bytearray.fromhex(
            ("7b763a323b6e000000050000000a62797465436f6c0073686"
             "f7274436f6c00696e74436f6c006c6f6e67436f6c00737472"
             "696e67436f6c0063686172436f6c00666c6f6174436f6c006"
             "46f75626c65436f6c00626f6f6c65616e436f6c0062696e61"
             "7279436f6c000a0b0c0d0e110f10121400000003d6eacd7d0"
             "a00000032000b00150000000000000000000c000000000000"
             "0020000000000000000000000000000000000000000000000"
             "0000000000000000021000000000000000000000000000000"
             "354142434400322122302c2e00000023357b3d303e7d002c6"
             "200643f4121999a000000000000000000000000424a000000"
             "0000000000000000000000000000000000000000000000000"
             "00000000000004049c0000000000088000000010000000005"
             "05040302010000000000000007020104054a050300000000"))

        TestDataFramesIO.truth_nullable_compressed = bytearray.fromhex(
            ("6466ab2eb332b2ce636060600562aea4ca9254e7fc1"
             "c86e28cfca2121023330f4ce5e4e7a583c54b8a3221"
             "ace48cc422109d96939f085692925f9a9403d69c949"
             "f9f939a98076666e625165582585cdc3cbc7c82fc02"
             "4222407b98afbd3a5bcb0564183170338832c0000f9"
             "45660c00f14d1f8a68e4ece2e0c468a4a063a7a40ae"
             "b269b5ad815d2d834e12438abda3e2cc59c86a9dbc0"
             "8180e040e9e07c0740710338218acac2ccc4c8c5059"
             "76264616562f5666101b00df48306a"))

        TestDataFramesIO.truth_nullable_base64 = (
            ("ZGarLrMyss5jYGBgBWKupMqSVOf8HIbijPyiEhAjMw9M5e"
             "TnpYPFS4oyIazkjMQiEJ2Wk58IVpKSX5qUA9aclJ+fk5qY"
             "B2Zm5iUWVYJYXNw8vHyC/AJCIkB7mK+9OlvLBWQYMXAziD"
             "LAAA+UVmDADxTR+KaOTs4uDEaKSgY6ekCusmm1rYFdLYNO"
             "EkOKvaPizFnIap28CBgOBA6eB8B0BxAzghisrCzMTIxQWX"
             "YmRhZWL1ZmEBsA30gwag=="))

        TestDataFramesIO.df_default.set_column_names(TestDataFramesIO.column_names)
        TestDataFramesIO.df_nullable.set_column_names(TestDataFramesIO.column_names)

    def test_serialization_default(self):
        b = DataFrame.serialize(TestDataFramesIO.df_default)
        self.assertTrue(
            TestDataFramesIO.truth == b, "Serialized Dataframe does not match expected bytes")

    def test_serialization_default_compress(self):
        b = DataFrame.serialize(TestDataFramesIO.df_default, compress=True)
        self.assertTrue(
            TestDataFramesIO.truth_compressed == b,
            "Serialized Dataframe does not match expected bytes")

    def test_serialization_nullable(self):
        b = DataFrame.serialize(TestDataFramesIO.df_nullable)
        self.assertTrue(
            TestDataFramesIO.truth_nullable == b,
            "Serialized Dataframe does not match expected bytes")

    def test_serialization_nullable_compressed(self):
        b = DataFrame.serialize(TestDataFramesIO.df_nullable, compress=True)
        self.assertTrue(
            TestDataFramesIO.truth_nullable_compressed == b,
            "Serialized Dataframe does not match expected bytes")

    def test_deserialization_default(self):
        res = DataFrame.deserialize(TestDataFramesIO.truth)
        self.assertFalse(res.is_empty(), "DataFrame should not be empty")
        self.assertTrue(res.rows() == 5, "DataFrame row count should be 5")
        self.assertTrue(res.columns() == 10, "DataFrame column count should be 10")
        self.assertTrue(res.has_column_names(), "DataFrame should have column names set")
        self.assertTrue(
            isinstance(res, DefaultDataFrame), "DataFrame should be of type DefaultDataFrame")

        self.assertTrue(res.equals(TestDataFramesIO.df_default), "DataFrame differs in content")

    def test_deserialization_default_compressed(self):
        res = DataFrame.deserialize(TestDataFramesIO.truth_compressed)
        self.assertFalse(res.is_empty(), "DataFrame should not be empty")
        self.assertTrue(res.rows() == 5, "DataFrame row count should be 5")
        self.assertTrue(res.columns() == 10, "DataFrame column count should be 10")
        self.assertTrue(res.has_column_names(), "DataFrame should have column names set")
        self.assertTrue(
            isinstance(res, DefaultDataFrame), "DataFrame should be of type DefaultDataFrame")

        self.assertTrue(res.equals(TestDataFramesIO.df_default), "DataFrame differs in content")

    def test_deserialization_nullable(self):
        res = DataFrame.deserialize(TestDataFramesIO.truth_nullable)
        self.assertFalse(res.is_empty(), "DataFrame should not be empty")
        self.assertTrue(res.rows() == 5, "DataFrame row count should be 5")
        self.assertTrue(res.columns() == 10, "DataFrame column count should be 10")
        self.assertTrue(res.has_column_names(), "DataFrame should have column names set")
        self.assertTrue(
            isinstance(res, NullableDataFrame), "DataFrame should be of type NullableDataFrame")

        self.assertTrue(res.equals(TestDataFramesIO.df_nullable), "DataFrame differs in content")

    def test_deserialization_nullable_compressed(self):
        res = DataFrame.deserialize(TestDataFramesIO.truth_nullable_compressed)
        self.assertFalse(res.is_empty(), "DataFrame should not be empty")
        self.assertTrue(res.rows() == 5, "DataFrame row count should be 5")
        self.assertTrue(res.columns() == 10, "DataFrame column count should be 10")
        self.assertTrue(res.has_column_names(), "DataFrame should have column names set")
        self.assertTrue(
            isinstance(res, NullableDataFrame), "DataFrame should be of type NullableDataFrame")

        self.assertTrue(res.equals(TestDataFramesIO.df_nullable), "DataFrame differs in content")

    def test_to_base64_default(self):
        s = DataFrame.to_base64(TestDataFramesIO.df_default)
        df = DataFrame.from_base64(s)
        self.assertTrue(df == TestDataFramesIO.df_default, "Dataframe does not match original")

    def test_to_base64_nullable(self):
        s = DataFrame.to_base64(TestDataFramesIO.df_nullable)
        df = DataFrame.from_base64(s)
        self.assertTrue(df == TestDataFramesIO.df_nullable, "Dataframe does not match original")

    def test_to_base64_string_default(self):
        s = DataFrame.to_base64(TestDataFramesIO.df_default)
        self.assertTrue(
            TestDataFramesIO.truth_base64 == s,
            "Serialized Dataframe does not match expected Base64 string")

    def test_to_base64_string_nullable(self):
        s = DataFrame.to_base64(TestDataFramesIO.df_nullable)
        self.assertTrue(
            TestDataFramesIO.truth_nullable_base64 == s,
            "Serialized Dataframe does not match expected Base64 string")

    def test_from_base64_string_default(self):
        res = DataFrame.from_base64(TestDataFramesIO.truth_base64)
        self.assertFalse(res.is_empty(), "DataFrame should not be empty")
        self.assertTrue(res.rows() == 5, "DataFrame row count should be 5")
        self.assertTrue(res.columns() == 10, "DataFrame column count should be 10")
        self.assertTrue(res.has_column_names(), "DataFrame should have column names set")
        self.assertTrue(
            isinstance(res, DefaultDataFrame), "DataFrame should be of type DefaultDataFrame")

        self.assertTrue(res.equals(TestDataFramesIO.df_default), "DataFrame differs in content")

    def test_from_base64_string_nullable(self):
        res = DataFrame.from_base64(TestDataFramesIO.truth_nullable_base64)
        self.assertFalse(res.is_empty(), "DataFrame should not be empty")
        self.assertTrue(res.rows() == 5, "DataFrame row count should be 5")
        self.assertTrue(res.columns() == 10, "DataFrame column count should be 10")
        self.assertTrue(res.has_column_names(), "DataFrame should have column names set")
        self.assertTrue(
            isinstance(res, NullableDataFrame), "DataFrame should be of type NullableDataFrame")

        self.assertTrue(res.equals(TestDataFramesIO.df_nullable), "DataFrame differs in content")

    def test_serial_deserial_default(self):
        b = DataFrame.serialize(TestDataFramesIO.df_default, compress=False)
        res = DataFrame.deserialize(b)
        self.assertTrue(res.equals(TestDataFramesIO.df_default), "DataFrames are not equal")

    def test_serial_deserial_default_compressed(self):
        b = DataFrame.serialize(TestDataFramesIO.df_default, compress=True)
        res = DataFrame.deserialize(b)
        self.assertTrue(res.equals(TestDataFramesIO.df_default), "DataFrames are not equal")

    def test_serial_deserial_nullable(self):
        b = DataFrame.serialize(TestDataFramesIO.df_nullable, compress=False)
        res = DataFrame.deserialize(b)
        self.assertTrue(res.equals(TestDataFramesIO.df_nullable), "DataFrames are not equal")

    def test_serial_deserial_nullable_compressed(self):
        b = DataFrame.serialize(TestDataFramesIO.df_nullable, compress=True)
        res = DataFrame.deserialize(b)
        self.assertTrue(res.equals(TestDataFramesIO.df_nullable), "DataFrames are not equal")

    def stress_test_default(self):
        df = DataFrame.copy(TestDataFramesIO.df_default)
        for _ in range(df.columns()):
            col = df.get_column(0)
            df.remove_column(0)
            df.add_column(col)
            b = DataFrame.serialize(df)
            df = DataFrame.deserialize(b)

        self.assertTrue(
            df.equals(TestDataFramesIO.df_default), "DataFrame does not match original")

        df2 = DataFrame.copy(TestDataFramesIO.df_default)
        for _ in range(df2.rows()):
            df2.remove_row(0)
            df.remove_row(0)
            b = DataFrame.serialize(df)
            df = DataFrame.deserialize(b)
            self.assertTrue(df.equals(df2), "DataFrame does not match changed object")

    def stress_test_nullable(self):
        df = DataFrame.copy(TestDataFramesIO.df_nullable)
        for _ in range(df.columns()):
            col = df.get_column(0)
            df.remove_column(0)
            df.add_column(col)
            b = DataFrame.serialize(df)
            df = DataFrame.deserialize(b)

        self.assertTrue(
            df.equals(TestDataFramesIO.df_nullable), "DataFrame does not match original")

        df2 = DataFrame.copy(TestDataFramesIO.df_nullable)
        for _ in range(df2.rows()):
            df2.remove_row(0)
            df.remove_row(0)
            b = DataFrame.serialize(df)
            df = DataFrame.deserialize(b)
            self.assertTrue(df.equals(df2), "DataFrame does not match changed object")

    def test_file_read_default(self):
        if not os.path.exists(TestDataFramesIO.FILE_DEFAULT):
            self.fail("Test resource '{}' was not found".format(TestDataFramesIO.FILE_DEFAULT))

        df = DataFrame.read(TestDataFramesIO.FILE_DEFAULT)
        self.assertTrue(df.equals(TestDataFramesIO.df_default), "DataFrames do not match")

    def test_file_read_nullable(self):
        if not os.path.exists(TestDataFramesIO.FILE_NULLABLE):
            self.fail("Test resource '{}' was not found".format(TestDataFramesIO.FILE_NULLABLE))

        df = DataFrame.read(TestDataFramesIO.FILE_NULLABLE)
        self.assertTrue(df.equals(TestDataFramesIO.df_nullable), "DataFrames do not match")

    def test_file_read_multiple_files_in_dir(self):
        if not os.path.exists(TestDataFramesIO.DIR_TEST_RESOURCES):
            self.fail(
                ("Test resource directory '{}' was not found")
                .format(TestDataFramesIO.DIR_TEST_RESOURCES))

        files = DataFrame.read(TestDataFramesIO.DIR_TEST_RESOURCES)
        self.assertTrue(isinstance(files, dict), "Returned object should be of type dict")
        self.assertTrue(len(files) == 2, "Returned dict should have 2 elements")
        self.assertTrue(
            files["test_default"].equals(DataFrame.read(TestDataFramesIO.FILE_DEFAULT)),
            "DataFrames do not match")

        self.assertTrue(
            files["test_nullable"].equals(DataFrame.read(TestDataFramesIO.FILE_NULLABLE)),
            "DataFrames do not match")


if __name__ == "__main__":
    unittest.main()
