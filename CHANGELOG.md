#### 1.1.3
* Fixed erroneous setting of row items with \[\]-operator when the specified row index is out of bounds but within column capacity.
* Fixed error resulting from inclusion of capacity buffer range in unique() method.
* Improved performance of absolute() method.
* Improved performance of ceil() method.
* Improved performance of floor() method.

#### 1.1.2
* Changed loop in remove\_column\_names() method.
* Fixed incorrect reading of CSV files with blank (empty) lines.
* Improved performance of get\_row() method.
* Improved performance of get\_rows() method.
* Improved performance of set\_row() method.
* Improved performance of add\_row() method.

#### 1.1.1
* Changed info string to standard representation for uninitialized DataFrames in info() method.
* Changed hashable internal representation of numpy arrays in all Columns by calling tolist() instead of tobytes() function.
* Changed hash code computation of (Nullable)BinaryColumns by taking hex string representation of bytearray content.
* Changed super calls in DefaultDataFrame and NullableDataFrame classes and removed function arguments.
* Fixed differing hash codes for some equal NullableDataFrames.
* Fixed some column conversion errors.
* Improved call stack representation involving DataFrameExceptions by accurately propagating nested Exceptions.
* Refactored column implementations and moved import dependencies of columns to \_columnutils module.
* Upgraded pylint version to 2.7.2 in requirements.txt

#### 1.1.0
* Added []-operator implementation via getitem and setitem functions.
* Added the ability to read and write multiple DataFrame files in one directory with a single call to read() and write() functions
* Added type check in read\_file() function in dataframes io module
* Changed DataFrame.memory\_usage() method and replaced internal for-loop
* Fixed type error resulting in DataFrameException in BooleanColumn.\_check\_type() method when using np.bool\_ dtype
* Improved some exception messages
* Improved performance of copy operations by preallocating list of Columns and directly constructing the DataFrame copy with it

#### 1.0.3
* Fixed invalid ASCII range check in NullableCharColumn
* Improved performance in Column._create_array() implementations by only utilizing numpy functions
* Improved performance of unique() and count_unique() methods
* Improved error messages in NullableCharColumn for invalid types

#### 1.0.2
* Fixed invalid construction of BinaryColumns with predefined column length
* Improved performance for column value access by using getitem() and setitem() methods
* Improved performance in DataFrame.index_of() and DataFrame.index_of_all() methods by removing redundant bounds checks
* Improved performance in DataFrame.filter() method by preallocating columns and setting rows instead of adding them
* Improved performance in include(), drop(), factor(), _set_typed_value(), _replace_by_match() and _remove_rows_by_match() in DataFrame class by using direct column access
* Improved performance in Column._remove() by calling setitem() method directly

#### 1.0.1
* Fixed missing flush instruction in to_array() method
* Fixed representation for char columns in to_array() method
* Fixed internal function name typo in _replace_by_dataframe()
* Improved error handling for invalid row item types when reading CSV files

#### 1.0.0 
* Open source release

