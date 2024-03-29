# Copyright (C) 2023 Raven Computing
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
Provides an implementation for StringColumn and NullableStringColumn
"""

import numpy as np

import raven.struct.dataframe.core as dataframe
import raven.struct.dataframe.column as column
import raven.struct.dataframe._columnutils as utils

class StringColumn(column.Column):
    """A Column holding string values (str).
    This implementation DOES NOT support null values or empty strings.
    """

    TYPE_CODE = 5
    DEFAULT_VALUE = "n/a"

    def __init__(self, name=None, values=None):
        """Constructs a new StringColumn.

        The constructed Column will have the specified name or is unlabeled
        if the specified name is None or empty.
        The constructed Column has the content of the specified list
        or numpy array. If the argument specifying the Column values is
        an int, then the constructed Column is initialized with the given
        length and all Column entries are set to default values.

        Args:
            name: The name of the StringColumn as a string
            values: The content of the StringColumn.
                Must be a list or numpy array with dtype object, or an int
        """
        if values is None:
            values = np.empty(0, dtype=object)

        if isinstance(values, list):
            for i, value in enumerate(values):
                self._check_type(value)
                if not value:
                    values[i] = StringColumn.DEFAULT_VALUE

            values = np.array(values, dtype=object)

        elif isinstance(values, np.ndarray):
            if values.dtype != "object":
                raise dataframe.DataFrameException(
                    ("Invalid argument array. Expected "
                     "string array (object) but found {}".format(values.dtype)))

            for i, value in enumerate(values):
                if not isinstance(value, str):
                    raise dataframe.DataFrameException(
                        ("Invalid element in argument array. Expected "
                         "string (str) but found {}".format(values.dtype)))

                if not value:
                    values[i] = StringColumn.DEFAULT_VALUE

        elif isinstance(values, int):
            values = np.empty(values, dtype=object)
            for i in range(values.shape[0]):
                values[i] = StringColumn.DEFAULT_VALUE

        else:
            raise dataframe.DataFrameException(
                ("Invalid argument array. Expected "
                 "list or numpy array but found {}".format(type(values))))

        super().__init__(name, values)

    def set_value(self, index, value):
        """Sets the string value at the specified index

        Args:
            index: The index of the string value to set
            value: The string value to set at the specified position.
                Must be a string

        Raises:
            ValueError: If the specified index is out of bounds or of the
                object provided is of the wrong type
        """
        self._check_bounds(index)
        self._check_type(value)
        if not value:
            self._values[index] = StringColumn.DEFAULT_VALUE
        else:
            self._values[index] = value

    def _check_type(self, value):
        if not isinstance(value, str):
            raise dataframe.DataFrameException(
                ("Invalid argument. Expected "
                 "string (str) but found {}".format(type(value))))

    def type_code(self):
        return StringColumn.TYPE_CODE

    def type_name(self):
        return "string"

    def is_nullable(self):
        return False

    def is_numeric(self):
        return False

    def get_default_value(self):
        return StringColumn.DEFAULT_VALUE

    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=invalid-name
    def convert_to(self, typecode):
        converted = None
        if typecode == utils.type_code_byte_column():
            vals = np.empty([self._values.shape[0]], dtype=np.int8)
            for i, x in np.ndenumerate(self._values):
                vals[i] = int(np.int8(x))

            converted = dataframe.DataFrame.ByteColumn(values=vals)
        elif typecode == utils.type_code_short_column():
            vals = np.empty([self._values.shape[0]], dtype=np.int16)
            for i, x in np.ndenumerate(self._values):
                vals[i] = int(np.int16(x))

            converted = dataframe.DataFrame.ShortColumn(values=vals)
        elif typecode == utils.type_code_int_column():
            vals = np.empty([self._values.shape[0]], dtype=np.int32)
            for i, x in np.ndenumerate(self._values):
                vals[i] = int(np.int32(x))

            converted = dataframe.DataFrame.IntColumn(values=vals)
        elif typecode == utils.type_code_long_column():
            vals = np.empty([self._values.shape[0]], dtype=np.int64)
            for i, x in np.ndenumerate(self._values):
                vals[i] = int(np.int64(x))

            converted = dataframe.DataFrame.LongColumn(values=vals)
        elif typecode == StringColumn.TYPE_CODE:
            converted = self.clone()
        elif typecode == utils.type_code_float_column():
            vals = np.empty([self._values.shape[0]], dtype=np.float32)
            for i, x in np.ndenumerate(self._values):
                vals[i] = float(np.float32(x))

            converted = dataframe.DataFrame.FloatColumn(values=vals)
        elif typecode == utils.type_code_double_column():
            vals = np.empty([self._values.shape[0]], dtype=np.float64)
            for i, x in np.ndenumerate(self._values):
                vals[i] = float(np.float64(x))

            converted = dataframe.DataFrame.DoubleColumn(values=vals)
        elif typecode == utils.type_code_char_column():
            vals = np.empty([self._values.shape[0]], dtype=np.uint8)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = ord(x[0])
                else:
                    vals[i] = utils.default_value_char_column()

            converted = dataframe.DataFrame.CharColumn(values=vals)
        elif typecode == utils.type_code_boolean_column():
            values_true = {"true", "t", "1", "yes", "y", "on"}
            values_false = {"false", "f", "0", "no", "n", "off"}
            vals = np.empty([self._values.shape[0]], dtype=bool)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    x = x.lower()
                    is_true = x in values_true
                    is_false = x in values_false
                    if not is_true and not is_false:
                        raise dataframe.DataFrameException(
                            ("Invalid boolean string: '{}'".format(self._values[i])))

                    vals[i] = is_true
                else:
                    vals[i] = False

            converted = dataframe.DataFrame.BooleanColumn(values=vals)
        elif typecode == utils.type_code_binary_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = bytearray.fromhex(x)
                else:
                    vals[i] = bytearray(b'\x00')

            converted = dataframe.DataFrame.BinaryColumn(values=vals)
        elif typecode == utils.type_code_nullable_byte_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = int(np.int8(x))
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableByteColumn(values=vals)
        elif typecode == utils.type_code_nullable_short_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = int(np.int16(x))
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableShortColumn(values=vals)
        elif typecode == utils.type_code_nullable_int_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = int(np.int32(x))
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableIntColumn(values=vals)
        elif typecode == utils.type_code_nullable_long_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = int(np.int64(x))
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableLongColumn(values=vals)
        elif typecode == NullableStringColumn.TYPE_CODE:
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(vals):
                vals[i] = x

            converted = NullableStringColumn(values=vals)
        elif typecode == utils.type_code_nullable_float_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = float(np.float32(x))
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableFloatColumn(values=vals)
        elif typecode == utils.type_code_nullable_double_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = float(np.float64(x))
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableDoubleColumn(values=vals)
        elif typecode == utils.type_code_nullable_char_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = ord(x[0])
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableCharColumn(values=vals)
        elif typecode == utils.type_code_nullable_boolean_column():
            values_true = {"true", "t", "1", "yes", "y", "on"}
            values_false = {"false", "f", "0", "no", "n", "off"}
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    x = x.lower()
                    is_true = x in values_true
                    is_false = x in values_false
                    if not is_true and not is_false:
                        raise dataframe.DataFrameException(
                            ("Invalid boolean string: '{}'".format(self._values[i])))

                    vals[i] = is_true
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableBooleanColumn(values=vals)
        elif typecode == utils.type_code_nullable_binary_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = bytearray.fromhex(x)
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableBinaryColumn(values=vals)
        else:
            raise dataframe.DataFrameException(
                "Unknown column type code: {}".format(typecode))

        # pylint: disable=protected-access
        converted._name = self._name
        return converted

    def _create_array(self, size=0):
        return np.empty(size, dtype=object)

class NullableStringColumn(column.Column):
    """A Column holding nullable string values.
    Any values not explicitly set are considered None.
    """

    TYPE_CODE = 14

    def __init__(self, name=None, values=None):
        """Constructs a new NullableStringColumn.

        The constructed Column will have the specified name or is unlabeled
        if the specified name is None or empty.
        The constructed Column has the content of the specified list
        or numpy array. If the argument specifying the Column values is
        an int, then the constructed Column is initialized with the given
        length and all Column entries are set to default values.

        Args:
            name: The name of the NullableStringColumn as a string
            values: The content of the NullableStringColumn.
                Must be a list or numpy array with dtype object, or an int
        """
        if values is None:
            values = np.empty(0, dtype=object)

        if isinstance(values, list):
            for value in values:
                self._check_type(value)

            values = np.array(values, dtype=object)

        elif isinstance(values, np.ndarray):
            if values.dtype != "object":
                raise dataframe.DataFrameException(
                    ("Invalid argument array. Expected "
                     "string array (object) but found {}".format(values.dtype)))

            for value in values:
                self._check_type(value)

        elif isinstance(values, int):
            values = np.empty(values, dtype=object)

        else:
            raise dataframe.DataFrameException(
                ("Invalid argument array. Expected "
                 "list or numpy array but found {}".format(type(values))))

        super().__init__(name, values)

    def _check_type(self, value):
        if value is not None and not isinstance(value, str):
            raise dataframe.DataFrameException(
                ("Invalid argument. Expected "
                 "string (str) but found {}".format(type(value))))

    def type_code(self):
        return NullableStringColumn.TYPE_CODE

    def type_name(self):
        return "string"

    def is_nullable(self):
        return True

    def is_numeric(self):
        return False

    def get_default_value(self):
        return None

    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=invalid-name
    def convert_to(self, typecode):
        converted = None
        if typecode == utils.type_code_byte_column():
            vals = np.empty([self._values.shape[0]], dtype=np.int8)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = int(np.int8(x))
                else:
                    vals[i] = 0

            converted = dataframe.DataFrame.ByteColumn(values=vals)
        elif typecode == utils.type_code_short_column():
            vals = np.empty([self._values.shape[0]], dtype=np.int16)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = int(np.int16(x))
                else:
                    vals[i] = 0

            converted = dataframe.DataFrame.ShortColumn(values=vals)
        elif typecode == utils.type_code_int_column():
            vals = np.empty([self._values.shape[0]], dtype=np.int32)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = int(np.int32(x))
                else:
                    vals[i] = 0

            converted = dataframe.DataFrame.IntColumn(values=vals)
        elif typecode == utils.type_code_long_column():
            vals = np.empty([self._values.shape[0]], dtype=np.int64)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = int(np.int64(x))
                else:
                    vals[i] = 0

            converted = dataframe.DataFrame.LongColumn(values=vals)
        elif typecode == StringColumn.TYPE_CODE:
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = str(x)
                else:
                    vals[i] = StringColumn.DEFAULT_VALUE

            converted = StringColumn(values=vals)
        elif typecode == utils.type_code_float_column():
            vals = np.empty([self._values.shape[0]], dtype=np.float32)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = float(np.float32(x))
                else:
                    vals[i] = 0.0

            converted = dataframe.DataFrame.FloatColumn(values=vals)
        elif typecode == utils.type_code_double_column():
            vals = np.empty([self._values.shape[0]], dtype=np.float64)
            for i, x in np.ndenumerate(self._values):
                if x is not None:
                    vals[i] = float(np.float64(x))
                else:
                    vals[i] = 0.0

            converted = dataframe.DataFrame.DoubleColumn(values=vals)
        elif typecode == utils.type_code_char_column():
            vals = np.zeros([self._values.shape[0]], dtype=np.uint8)
            ord_default = ord(utils.default_value_char_column())
            for i, x in np.ndenumerate(self._values):
                if x:
                    vals[i] = ord(x[0])
                else:
                    vals[i] = ord_default

            converted = dataframe.DataFrame.CharColumn(values=vals)
        elif typecode == utils.type_code_boolean_column():
            values_true = {"true", "t", "1", "yes", "y", "on"}
            values_false = {"false", "f", "0", "no", "n", "off"}
            vals = np.empty([self._values.shape[0]], dtype=bool)
            for i, x in np.ndenumerate(self._values):
                if x:
                    x = x.lower()
                    is_true = x in values_true
                    is_false = x in values_false
                    if not is_true and not is_false:
                        raise dataframe.DataFrameException(
                            ("Invalid boolean string: '{}'".format(self._values[i])))

                    vals[i] = is_true
                else:
                    vals[i] = False

            converted = dataframe.DataFrame.BooleanColumn(values=vals)
        elif typecode == utils.type_code_binary_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x:
                    vals[i] = bytearray.fromhex(x)
                else:
                    vals[i] = bytearray(b'\x00')

            converted = dataframe.DataFrame.BinaryColumn(values=vals)
        elif typecode == utils.type_code_nullable_byte_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                vals[i] = int(np.int8(x)) if x is not None else None

            converted = dataframe.DataFrame.NullableByteColumn(values=vals)
        elif typecode == utils.type_code_nullable_short_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                vals[i] = int(np.int16(x)) if x is not None else None

            converted = dataframe.DataFrame.NullableShortColumn(values=vals)
        elif typecode == utils.type_code_nullable_int_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                vals[i] = int(np.int32(x)) if x is not None else None

            converted = dataframe.DataFrame.NullableIntColumn(values=vals)
        elif typecode == utils.type_code_nullable_long_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                vals[i] = int(np.int64(x)) if x is not None else None

            converted = dataframe.DataFrame.NullableLongColumn(values=vals)
        elif typecode == NullableStringColumn.TYPE_CODE:
            converted = self.clone()
        elif typecode == utils.type_code_nullable_float_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x:
                    vals[i] = float(np.float32(x))
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableFloatColumn(values=vals)
        elif typecode == utils.type_code_nullable_double_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x:
                    vals[i] = float(np.float64(x))
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableDoubleColumn(values=vals)
        elif typecode == utils.type_code_nullable_char_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x:
                    vals[i] = str(x)[0]
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableCharColumn(values=vals)
        elif typecode == utils.type_code_nullable_boolean_column():
            values_true = {"true", "t", "1", "yes", "y", "on"}
            values_false = {"false", "f", "0", "no", "n", "off"}
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x:
                    x = x.lower()
                    is_true = x in values_true
                    is_false = x in values_false
                    if not is_true and not is_false:
                        raise dataframe.DataFrameException(
                            ("Invalid boolean string: '{}'".format(self._values[i])))

                    vals[i] = is_true
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableBooleanColumn(values=vals)
        elif typecode == utils.type_code_nullable_binary_column():
            vals = np.empty([self._values.shape[0]], dtype=object)
            for i, x in np.ndenumerate(self._values):
                if x:
                    vals[i] = bytearray.fromhex(x)
                else:
                    vals[i] = None

            converted = dataframe.DataFrame.NullableBinaryColumn(values=vals)
        else:
            raise dataframe.DataFrameException(
                "Unknown column type code: {}".format(typecode))

        # pylint: disable=protected-access
        converted._name = self._name
        return converted

    def _create_array(self, size=0):
        return np.empty(size, dtype=object)
