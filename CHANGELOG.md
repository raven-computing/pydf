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

