This is the official implementation of the DataFrame specification provided by Raven Computing.

## Getting Started

Install via:
```
pip install raven-pydf
```

After installation you can use the entire DataFrame API by importing one class:
```python
from raven.struct.dataframe import DataFrame

# read a DataFrame file into memory
df = DataFrame.read("/path/to/myFile.df")

# show the first 10 rows on stdout
print(df.head(10))
```
Alternatively, you can import all concrete Column types directly, for example:
```python
from raven.struct.dataframe import (DefaultDataFrame,
                                    IntColumn,
                                    FloatColumn,
                                    StringColumn)

# create a DataFrame with 3 columns and 3 rows
df = DefaultDataFrame(
        IntColumn("A", [1, 2, 3]),
        FloatColumn("B", [4.4, 5.5, 6.6]),
        StringColumn("C", ["cat", "dog", "horse"]))

print(df)
```

## Compatibility

This library requires **Python3.7** or higher.

Internally, this library uses [Numpy](https://github.com/numpy/numpy) for array operations. The minimum required version is v1.19.0

## Documentation

The unified documentation is available [here](https://www.raven-computing.com/docs/dataframe?language=python).

Additional features implemented in Python are documented in the [Wiki](https://github.com/raven-computing/pydf/wiki).

## License

This library is licensed under the Apache License Version 2 - see the [LICENSE](https://github.com/raven-computing/pydf/blob/master/LICENSE) for details.
