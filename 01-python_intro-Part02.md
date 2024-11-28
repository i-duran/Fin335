# Introduction to Python (Part 2)

## Python Lists and Basic Operations

Lists are one of the most versatile and commonly used data structures in
Python. They allow you to store multiple items in a single variable.

### Creating Lists

You can create a list by enclosing comma-separated values in square
brackets:

```python
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
```

#### Accessing List Elements

List elements are indexed starting from 0. You can access individual
elements using square brackets:

```python
numbers = [10, 1, 3, 4, -5]
num1 = numbers[-1] # stores the last element of the list (-5) in num1
num2 = numbers[0] # stores the first element of the list (10) in num2
num3 = numbers[3] # stores '4' in num3

print("num1:", num1)
print("num2:", num2)
print("num3:", num3)
```

#### 1. Adding Elements

Use `append()` to add an element to the end of the list

```python
fruits = ["apple", "banana"]
fruits.append("cherry")
print(fruits)  # Output: ['apple', 'banana', 'cherry']

numbers = [1, 2, 3]
numbers.append(4)
print(numbers)  # Output: [1, 2, 3, 4]
```

#### 2. Removing Elements

Use `remove()` to remove a specific element:

```python
fruits = ["apple", "banana", "cherry"]
fruits.remove("banana")
print(fruits)  # Output: ['apple', 'cherry']

numbers = [1, 2, 3, 4]
numbers.remove(3)
print(numbers)  # Output: [1, 2, 4]
```

#### 3. List Length

Use the `len()` function to get the number of elements in a list:

```python
fruits = ["apple", "banana", "cherry"]
print(len(fruits))  # Output: 3

numbers = [1, 2, -3, 4, 2, 10]
print(len(numbers))  # Output: 6
```

#### 4. Slicing

You can extract a portion of a list using slicing:

```python
numbers = [0, 1, 2, 3, 4, 5]
print(numbers[1:4])  # Output: [1, 2, 3]
print(numbers[:3])   # Output: [0, 1, 2]
print(numbers[2:])   # Output: [2, 3, 4, 5]
```

## Basic Math Operations

Now, let’s we explore simple mathematical operations in Python. These
operations include addition, subtraction, multiplication, and division.

### Addition

```python
# Adding two numbers in Python
number1 = 5
number2 = 3
sum = number1 + number2
print("Sum:", sum) # Output: Sum: 8
```

### Subtraction

```python
# Subtracting two numbers in Python
number1 = 5
number2 = 3
difference = number1 - number2
print("Difference:", difference) # Output: Difference: 2
```

### Multiplication


```python
# Multiplying two numbers in Python
number1 = 5
number2 = 3
product = number1 * number2
print("Product:", product) # Output: Product: 15
```

### Division

```python
number1 = 10
number2 = 2
quotient = number1 / number2
print("Quotient:", quotient) # Output: Quotient: 5.0
```

### Raising to a Power

``` python
base = 2
exponent = 3
result = base ** exponent
print(f"{base} raised to the power of {exponent} is {result}") # Output: 2 raised to the power of 3 is 8
```

Remember that
`print(f"{base} raised to the power of {exponent} is {result}")` uses an
f-string to format the output, which allows to include variables
directly in the string in between curly braces.


## Data Representation Using Nested Lists

Nested lists are lists within lists, allowing for more complex data
representation. Here’s an example of a nested list:

```python
# Create a nested list representing a data table
# with 4 columns and 6 rows (including the header)
data = [
    ["Name", "Age", "City", "Occupation"],
    ["Alice", 28, "New York", "Engineer"],
    ["Bob", 35, "San Francisco", "Designer"],
    ["Charlie", 42, "London", "Manager"],
    ["Diana", 31, "Paris", "Artist"],
    ["Ethan", 25, "Tokyo", "Teacher"]
]

# Accessing data
print(f"The second person's name is: {data[2][0]}")
print(f"The third person's age is: {data[3][1]}")

# Modifying data
data[4][2] = "Berlin"  # Change Diana's city to Berlin

# Adding a new row
data.append(["Frank", 39, "Sydney", "Doctor"])

# Printing the entire table
print("\nData Table:")
for row in data:
    print(f"{row[0]:<10} {row[1]:<5} {row[2]:<15} {row[3]:<10}")
```

    The second person's name is: Bob
    The third person's age is: 42

    Data Table:
    Name       Age   City            Occupation
    Alice      28    New York        Engineer  
    Bob        35    San Francisco   Designer  
    Charlie    42    London          Manager   
    Diana      31    Berlin          Artist    
    Ethan      25    Tokyo           Teacher   
    Frank      39    Sydney          Doctor    

Explanation:

1.  We created a nested list called `data`. Each inner list represents a
    row, and each element in the inner list represents a column.

2.  The first row `["Name", "Age", "City", "Occupation"]` serves as the
    header.

3.  We demonstrated how to access specific elements using indexing. For
    example, `data[2][0]` accesses the name in the second row (remember,
    indexing starts at 0).

4.  We show how to modify data by changing Diana’s city from Paris to
    Berlin.

5.  We append a new row to the list using the `append()` method.

6.  Finally, we print the entire table using a formatted string. The
    `:<10` syntax in the f-string ensures that each column is aligned
    and has a minimum width of 10 characters (or 5 for the Age column).

This example showcases how nested lists can be used to represent tabular
data and how to perform common operations on such data structures.

> [!note]
> In practice, however, we will be using later on (in the next
tutorial) more advanced data structures like **Pandas DataFrames** and
**Numpy Arrays** for handling tabular data in Python. **Pandas** and
**Numpy** are what we call Python **libraries**. These libraries make it
much easier to manipulate and perform operations with data.
