# Loading and Manipulating Data with Pandas

In this guide, we will walk through the steps to load a CSV file using
the **Pandas library** in Python, summarize the data, and perform some
basic manipulations, including converting certain columns to their
logarithmic values (using the **Numpy library**) and creating new
variables based on calculations from existing ones.

**Note**: Pandas and Numpy are two of the main libraries when working
with data. They are widely known and we will be using them all the time
in this class.

## 1. Loading the Data

First, we need to import the necessary libraries and load the data from
the CSV file. Here’s how to do it:

``` python
import numpy as np
import pandas as pd

# Load the data
data = pd.read_csv('./data/capm.csv', index_col=0)
```

Here, we use the `read_csv` function to load the data, and (optionally)
define the first column (the date in this case) as the index by using
`index_col=0`.

The data is now stored in the variable `data` which, in this case, is a
Pandas **DataFrame**.

Also, it is good practice to import all the libraries that are needed at
the beginning of the file.

## 2. Summarizing the Data

After loading the data, it’s essential to understand its structure and
contents. We can use the following methods:

- **View the first few rows** of the DataFrame:

``` python
print(data.head())
```

                  SANDP   FORD         GE  MICROSOFT     ORACLE  USTB3M
    Date                                                               
    Jan-02  1130.199951  15.30  37.150002  31.855000  17.260000    1.68
    Feb-02  1106.729980  14.88  38.500000  29.170000  16.620001    1.76
    Mar-02  1147.390015  16.49  37.400002  30.155001  12.800000    1.83
    Apr-02  1076.920044  16.00  31.549999  26.129999  10.040000    1.75
    May-02  1067.140015  17.65  31.139999  25.455000   7.920000    1.76

- **Get basic information** about the DataFrame:

``` python
print(data.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 194 entries, Jan-02 to Feb-18
    Data columns (total 6 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   SANDP      194 non-null    float64
     1   FORD       194 non-null    float64
     2   GE         194 non-null    float64
     3   MICROSOFT  194 non-null    float64
     4   ORACLE     194 non-null    float64
     5   USTB3M     194 non-null    float64
    dtypes: float64(6)
    memory usage: 10.6+ KB
    None

- **Summary statistics** can be obtained using:

``` python
print(data.describe())
```

                 SANDP        FORD          GE   MICROSOFT      ORACLE      USTB3M
    count   194.000000  194.000000  194.000000  194.000000  194.000000  194.000000
    mean   1468.897883   11.430361   26.593866   34.754433   25.972474    1.241495
    std     472.527116    3.552879    7.384621   15.044991   12.038702    1.570625
    min     729.570007    1.870000    8.510000   16.150000    7.860000    0.010000
    25%    1127.207459    8.527500   20.719999   25.867500   13.950000    0.080000
    50%    1323.525024   11.740000   26.950000   28.350000   22.710000    0.480000
    75%    1854.725037   14.167500   32.697501   38.265001   36.474999    1.725000
    max    2816.449951   17.650000   41.400002   95.010002   51.590000    5.160000

These commands will help us understand the data types, the number of
entries, and basic statistics like mean, standard deviation, and
quartiles.

## 3. Basic Data Manipulation

### 3.1. Converting Columns to Logarithmic Scale

To convert specific columns to their logarithmic values, we can use the
`numpy` library for the logarithmic transformation. For example, if we
want to convert the `SANDP` column to its logarithmic form:

``` python
# Convert SANDP column to logarithmic scale using Numpy
data['log_SANDP'] = np.log(data['SANDP'])
print(data.head())
```

                  SANDP   FORD         GE  MICROSOFT     ORACLE  USTB3M  log_SANDP
    Date                                                                          
    Jan-02  1130.199951  15.30  37.150002  31.855000  17.260000    1.68   7.030150
    Feb-02  1106.729980  14.88  38.500000  29.170000  16.620001    1.76   7.009165
    Mar-02  1147.390015  16.49  37.400002  30.155001  12.800000    1.83   7.045245
    Apr-02  1076.920044  16.00  31.549999  26.129999  10.040000    1.75   6.981860
    May-02  1067.140015  17.65  31.139999  25.455000   7.920000    1.76   6.972737

Every column in the CSV or Excel file, is now a variable in the pandas
`DataFrame`. Here we are creating a new variable, the log of `SANDP` and
storing it as another variable in our data set (stored in `data`).

In the example above, we **get** the `SANDP` column or variable with
`data['SANDP']`, which is just the way to get an item from a Python
dictionary `dict` (see explanation at the end [Understanding Python
Dictionaries](#understanding-python-dictionaries)).

Then we use `data['SANDP']` to calculate the `log` and store that in a
new column or variable, namely `data['log_SANDP']`. **Note** that
`log_SANDP` is just an arbitrary name (you can give whatever name you
want, just avoid using spaces when naming new variables).

### 3.2. Creating New Variables

To summarize, we can create new columns based on calculations from
existing columns. For instance, let’s create a new column that
represents the difference between the `MICROSOFT` and `ORACLE` stock
prices:

``` python
# Create a new column for the difference between MICROSOFT and ORACLE
data['MSFT_ORCL_diff'] = data['MICROSOFT'] - data['ORACLE']
print(data.head())
```

                  SANDP   FORD         GE  MICROSOFT     ORACLE  USTB3M  \
    Date                                                                  
    Jan-02  1130.199951  15.30  37.150002  31.855000  17.260000    1.68   
    Feb-02  1106.729980  14.88  38.500000  29.170000  16.620001    1.76   
    Mar-02  1147.390015  16.49  37.400002  30.155001  12.800000    1.83   
    Apr-02  1076.920044  16.00  31.549999  26.129999  10.040000    1.75   
    May-02  1067.140015  17.65  31.139999  25.455000   7.920000    1.76   

            log_SANDP  MSFT_ORCL_diff  
    Date                               
    Jan-02   7.030150       14.595000  
    Feb-02   7.009165       12.549999  
    Mar-02   7.045245       17.355001  
    Apr-02   6.981860       16.089999  
    May-02   6.972737       17.535000  

## 4. Conclusion

At this point, you have successfully loaded the data, summarized it, and
performed basic manipulations. Later on, we will further explore the
dataset by applying various statistical analyses or visualizations as
needed.

### Example of a Simple Calculation

If you want to calculate the mean of the `log_SANDP` column, you can do
so as follows:

``` python
mean_sandp = data['SANDP'].mean()
print(f'Mean of SANDP: {mean_sandp}')
```

    Mean of SANDP: 1468.8978825206184

Or, if we want to calculate the mean for all columns, excluding the
date, we can use a `for loop`:

``` python
# Calculate the mean of all numeric columns excluding the first column (Date)
means = {}
columns = data.columns # this is a List with the columns names
print(columns)

# Exclude the first column (Date)
for column in columns[1:]:  # This excludes the first column
    if pd.api.types.is_numeric_dtype(data[column]): # optional check
        means[column] = data[column].mean()

# Print the means for all numeric columns
for column, mean_value in means.items():
    print(f'Mean of {column:<15}: {round(mean_value, 4):>10}')
```

    Index(['SANDP', 'FORD', 'GE', 'MICROSOFT', 'ORACLE', 'USTB3M', 'log_SANDP',
           'MSFT_ORCL_diff'],
          dtype='object')
    Mean of FORD           :    11.4304
    Mean of GE             :    26.5939
    Mean of MICROSOFT      :    34.7544
    Mean of ORACLE         :    25.9725
    Mean of USTB3M         :     1.2415
    Mean of log_SANDP      :     7.2445
    Mean of MSFT_ORCL_diff :      8.782

Alternatively, we can use this easier approach:

``` python
# Calculate the mean of all numeric columns excluding the first column (Date)
means = data.iloc[:, 1:].mean()  # Excludes the first column and calculates the mean

# Print the means for all numeric columns
for column, mean_value in means.items():
    print(f'Mean of {column}: {round(mean_value, 4)}')
```

    Mean of FORD: 11.4304
    Mean of GE: 26.5939
    Mean of MICROSOFT: 34.7544
    Mean of ORACLE: 25.9725
    Mean of USTB3M: 1.2415
    Mean of log_SANDP: 7.2445
    Mean of MSFT_ORCL_diff: 8.782

------------------------------------------------------------------------

## Understanding Python Dictionaries

A **Python dictionary** is a built-in data type that allows you to store
data in key-value pairs. Each key in a dictionary is unique, and it maps
to a specific value. This makes dictionaries very useful for quickly
accessing data based on a specific identifier (the key).

### Example 1: Creating a Simple Dictionary

``` python
# Creating a simple dictionary
person = {
    'name': 'Alice',
    'age': 30,
    'city': 'New York'
}

# Accessing a value using its key
print(person['name'])  # Output: Alice
```

    Alice

In this example, we create a dictionary called `person` with three
key-value pairs. We can access the value associated with the key
`'name'` using the syntax `person['name']`.

### Example 2: Modifying a Dictionary

``` python
# Modifying a value in the dictionary
person['age'] = 31  # Update the age
print(person)  # Output: {'name': 'Alice', 'age': 31, 'city': 'New York'}
```

    {'name': 'Alice', 'age': 31, 'city': 'New York'}

Here, we update the value associated with the key `'age'`. The
dictionary allows us to easily modify values without changing the
structure.

## Pandas DataFrames as Dictionaries

A **Pandas DataFrame** can be thought of as a dictionary. Each column in
a DataFrame can be seen as a key-value pair, where the key is the column
name, and the value is a list (or array-like structure) of the column’s
data.

### Key Points:

- **Columns as Keys**: In a DataFrame, each column name acts as a key,
  and the data within that column acts as the associated value.
- **Accessing Columns**: You can access a specific column (which is
  similar to a list) using the following notation:

``` python
# Accessing a column in a DataFrame
sandp_column = data['SANDP']  # This retrieves the 'SANDP' column
print(sandp_column)
```

    Date
    Jan-02    1130.199951
    Feb-02    1106.729980
    Mar-02    1147.390015
    Apr-02    1076.920044
    May-02    1067.140015
                 ...     
    Oct-17    2583.209961
    Nov-17    2645.100098
    Dec-17    2683.729980
    Jan-18    2816.449951
    Feb-18    2715.219971
    Name: SANDP, Length: 194, dtype: float64

- **Columns as Lists**: Each column in a DataFrame behaves like a list,
  allowing you to perform list-like operations, such as slicing and
  indexing (**Note**: we covered Python list in the previous Python
  introduction).

### Example of Accessing DataFrame Elements

``` python
# Accessing the first five elements of the 'SANDP' column
first_five_sandp = data['SANDP'][:5]
print(first_five_sandp)
```

    Date
    Jan-02    1130.199951
    Feb-02    1106.729980
    Mar-02    1147.390015
    Apr-02    1076.920044
    May-02    1067.140015
    Name: SANDP, dtype: float64

In this example, we access the first five elements of the `SANDP` column
using list notation, which is similar to how we would access elements in
a Python list.

**In summary**, a Python dictionary is a collection of key-value pairs,
and Pandas DataFrames can be thought of as dictionaries where each
column is a key, and the data within those columns are the values. This
structure allows for efficient data manipulation and retrieval, making
DataFrames a powerful tool for data analysis in Python.

------------------------------------------------------------------------

## Pandas `iloc`

`iloc` is an indexing method in Pandas that allows you to select data
from a DataFrame by its integer location (index positions). It stands
for “integer location” and is primarily used for positional indexing,
which means you can access rows and columns based on their integer
positions rather than their labels.

### Key Features of `iloc`

1.  **Positional Indexing**: You can access rows and columns using their
    integer positions. For example, `data.iloc[0]` accesses the first
    row of the DataFrame, and `data.iloc[:, 0]` accesses the first
    column.

2.  **Slicing**: You can use slicing to get a range of rows or columns.
    For example, `data.iloc[0:5]` retrieves the first five rows.

3.  **Single Value Access**: You can access a specific value by
    providing both row and column indices. For example,
    `data.iloc[0, 1]` retrieves the value in the first row and the
    second column.

4.  **Boolean Indexing**: You can also use boolean arrays (lists of
    `True` and/or `False`) to filter data when using `iloc`.

### Basic Syntax

The basic syntax for using `iloc` is:

``` python
data.iloc[row_index, column_index]
```

- `row_index`: The integer index (or slice) for the rows you want to
  select.
- `column_index`: The integer index (or slice) for the columns you want
  to select.

### Examples of Using `iloc`

Here are some examples to illustrate how to use `iloc` in Pandas:

#### Example 1: Accessing Rows

``` python
# Sample DataFrame
data = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
})

# Access the first row
first_row = data.iloc[0]
print(first_row)
```

    A    1
    B    5
    C    9
    Name: 0, dtype: int64

#### Example 2: Accessing Specific Rows and Columns

``` python
# Access the value in the second row and first column
value = data.iloc[1, 0]
print(value)  # Output: 2
```

    2

#### Example 3: Slicing Rows

``` python
# Access the first two rows
first_two_rows = data.iloc[0:2]
print(first_two_rows)
```

       A  B   C
    0  1  5   9
    1  2  6  10

#### Example 4: Accessing Multiple Rows and Columns

``` python
# Access the first two rows and the first two columns
subset = data.iloc[0:2, 0:2]
print(subset)
```

       A  B
    0  1  5
    1  2  6

### Important Notes

- **Zero-Based Indexing**: `iloc` uses zero-based indexing, meaning the
  first element has an index of `0`.
- **Error Handling**: If you try to access an index that is out of
  bounds, Pandas will raise an `IndexError`.
- **Not Label-Based**: Unlike `loc`, which is label-based, `iloc`
  strictly uses integer indices.

### Conclusion

`iloc` is a powerful tool in Pandas for data selection and manipulation
based on integer positions. It is especially useful when you want to
access data without worrying about the column names or row indices,
making it a flexible option for data analysis and exploration.
