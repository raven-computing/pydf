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
Tests reading and writing of CSV files.
"""

import os

import unittest

import raven.io.dataframe.csvfiles as csv

from raven.struct.dataframe import (DataFrame,
                                    DefaultDataFrame,
                                    NullableDataFrame)

# pylint: disable=invalid-name, missing-function-docstring, bad-whitespace

class TestCSV(unittest.TestCase):
    """Tests reading and writing of CSV files."""

    FILE_DEFAULT   = os.path.join(os.path.dirname(__file__), "test.csv")
    FILE_NULLABLE  = os.path.join(os.path.dirname(__file__), "test_nullable.csv")
    FILE_NOHEADER  = os.path.join(os.path.dirname(__file__), "test_noheader.csv")
    FILE_MALFORMED = os.path.join(os.path.dirname(__file__), "test_nullable_malformed.csv")

    DF_DEFAULT             = None
    DF_DEFAULT_AS_STRING   = None
    DF_NULLABLE            = None
    DF_NULLABLE_AS_STRING  = None
    DF_MALFORMED           = None
    DF_MALFORMED_AS_STRING = None

    def setUp(self):
        TestCSV.DF_DEFAULT = DefaultDataFrame(
            DataFrame.IntColumn("AttrA", [1, 2, 3]),
            DataFrame.DoubleColumn("AttrB", [1.1, 2.2, 3.3]),
            DataFrame.StringColumn("AttrC", ["C1", "C2", "C,3"]))

        TestCSV.DF_DEFAULT_AS_STRING = DefaultDataFrame(
            DataFrame.StringColumn("AttrA", ["1","2","3"]),
            DataFrame.StringColumn("AttrB", ["1.1","2.2","3.3"]),
            DataFrame.StringColumn("AttrC", ["C1","C2","C,3"]))

        TestCSV.DF_NULLABLE = NullableDataFrame(
            DataFrame.NullableIntColumn("AttrA", [None, 2, 3]),
            DataFrame.NullableDoubleColumn("AttrB", [1.1, None, 3.3]),
            DataFrame.NullableStringColumn("AttrC", ["C1", "C2", None]))

        TestCSV.DF_NULLABLE_AS_STRING = NullableDataFrame(
            DataFrame.NullableStringColumn("AttrA", [None,"2","3"]),
            DataFrame.NullableStringColumn("AttrB", ["1.1",None,"3.3"]),
            DataFrame.NullableStringColumn("AttrC", ["C1","C2",None]))

        TestCSV.DF_MALFORMED = NullableDataFrame(
            DataFrame.NullableIntColumn("AttrA", [11, 22, 33, None]),
            DataFrame.NullableDoubleColumn("AttrB", [None, None, 3.3, None]),
            DataFrame.NullableDoubleColumn("AttrC", [None, 2.2, None, 4.4]),
            DataFrame.NullableStringColumn("AttrD", [None, None, None, None]))

        TestCSV.DF_MALFORMED_AS_STRING = NullableDataFrame(
            DataFrame.NullableStringColumn("AttrA", ["11", "22", "33", None]),
            DataFrame.NullableStringColumn("AttrB", [None, None, "3.3", None]),
            DataFrame.NullableStringColumn("AttrC", [None, "2.2", None, "4.4"]),
            DataFrame.NullableStringColumn("AttrD", [None, None, None, None]))


    def test_file_read(self):
        if not os.path.exists(TestCSV.FILE_DEFAULT):
            self.fail("Test resource '{}' was not found".format(TestCSV.FILE_DEFAULT))

        df = csv.read(TestCSV.FILE_DEFAULT)
        self.assertTrue(TestCSV.DF_DEFAULT_AS_STRING == df, "DataFrames do not match")

    def test_file_read_with_types(self):
        if not os.path.exists(TestCSV.FILE_DEFAULT):
            self.fail("Test resource '{}' was not found".format(TestCSV.FILE_DEFAULT))

        df = csv.read(TestCSV.FILE_DEFAULT, types=("int", "double", "string"))
        self.assertTrue(TestCSV.DF_DEFAULT == df, "DataFrames do not match")

    def test_file_read__no_header(self):
        if not os.path.exists(TestCSV.FILE_DEFAULT):
            self.fail("Test resource '{}' was not found".format(TestCSV.FILE_NOHEADER))

        df = csv.read(TestCSV.FILE_NOHEADER, header=False)
        # Remove column names to match expected
        TestCSV.DF_DEFAULT_AS_STRING.remove_column_names()
        self.assertTrue(TestCSV.DF_DEFAULT_AS_STRING == df, "DataFrames do not match")

    def test_file_read_with_types_no_header(self):
        if not os.path.exists(TestCSV.FILE_DEFAULT):
            self.fail("Test resource '{}' was not found".format(TestCSV.FILE_NOHEADER))

        df = csv.read(TestCSV.FILE_NOHEADER, header=False, types=("int", "double", "string"))

        # Remove column names to match expected
        TestCSV.DF_DEFAULT.remove_column_names()
        self.assertTrue(TestCSV.DF_DEFAULT == df, "DataFrames do not match")

    def test_file_read_nullable(self):
        if not os.path.exists(TestCSV.FILE_NULLABLE):
            self.fail("Test resource '{}' was not found".format(TestCSV.FILE_NULLABLE))

        df = csv.read(TestCSV.FILE_NULLABLE)
        self.assertTrue(TestCSV.DF_NULLABLE_AS_STRING == df, "DataFrames do not match")

    def test_file_read_with_types_nullable(self):
        if not os.path.exists(TestCSV.FILE_NULLABLE):
            self.fail("Test resource '{}' was not found".format(TestCSV.FILE_NULLABLE))

        df = csv.read(TestCSV.FILE_NULLABLE, types=("int", "double", "string"))
        self.assertTrue(TestCSV.DF_NULLABLE == df, "DataFrames do not match")

    def test_file_read_nullable_malformed(self):
        if not os.path.exists(TestCSV.FILE_MALFORMED):
            self.fail("Test resource '{}' was not found".format(TestCSV.FILE_MALFORMED))

        df = csv.read(TestCSV.FILE_MALFORMED)
        self.assertTrue(TestCSV.DF_MALFORMED_AS_STRING == df, "DataFrames do not match")

    def test_file_read_with_types_nullable_malformed(self):
        if not os.path.exists(TestCSV.FILE_MALFORMED):
            self.fail("Test resource '{}' was not found".format(TestCSV.FILE_MALFORMED))

        df = csv.read(TestCSV.FILE_MALFORMED, types=("int", "double", "double", "string"))
        self.assertTrue(TestCSV.DF_MALFORMED == df, "DataFrames do not match")
