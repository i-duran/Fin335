# Introduction to Python

## Creating variables

We store data in what we call variables. Variables are 'names' that we can reference for later use.

### 1. **General Naming Rules**

-  **Allowed Characters**: Variable names can include letters (both uppercase and lowercase), digits, and underscores (`_`).
-  **Starting Character**: A variable name must start with a letter or an underscore, not a digit.
-  **Case Sensitivity**: Variable names are case-sensitive (`myVar` and `myvar` are different variables).
-  **No Special Characters**: Avoid using special characters (like `@`, `$`, `%`, etc.) in variable names.

### 2. **Naming Conventions**

-  **Snake Case**: Use lowercase letters and underscores to separate words. This is the most common convention in Python.

    ```python
    my_variable_name = 10
    total_count = 5
    ```

-  **Camel Case**: This is less common in Python but may be seen in some contexts, especially in class names.

    ```python
    myVariableName = 10  # Not recommended for variables
    ```

-  **Pascal Case**: Typically used for class names in Python (we will not use `classes` in this course).

    ```python
    class MyClass:
    pass
    ```

### 3. **Descriptive Names**

-  Choose meaningful and descriptive names that convey the purpose of the variable.

    ```python
    user_age = 25
    total_price = 99.99
    ```

### 4. **Avoiding Reserved Keywords**

-  Do not use Python reserved keywords as variable names (e.g., `if`, `else`, `while`, `for`, `def`, etc.).

### 5. **Constants**

-  For constants, use all uppercase letters with underscores separating words.

    ```python
    MAX_CONNECTIONS = 100
    PI = 3.14
    ```

### 6. **Avoid One-Letter Names**

-  Avoid using single-letter variable names except for counters or iterators (like `i`, `j`, `k` in loops). Use descriptive names for clarity.

### Example

Here’s a simple example that incorporates these conventions:

```python
# Constants
MAX_USERS = 100

# Function to calculate the average age of users
def calculate_average_age(user_ages):
    total_age = sum(user_ages)
    average_age = total_age / len(user_ages)
    return average_age

# Main execution
ages = [25, 30, 35, 40]
avg_age = calculate_average_age(ages)
print(f"The average age is: {avg_age}")
```

### Explanation

Sure! Let’s break down the concepts of lists and the `print` function in Python, as used in the example provided.

#### 1. **The `print` Function**

The `print` function in Python is used to output data to the console or the screen. It can display strings (which is the name for text), numbers, and other objects. Here are some key features of the `print` function:

-  **Basic Usage**: You can pass one or more items to the `print` function, and it will display them in the console.

    ```python
    print("Hello, World!")  # Output: Hello, World!
    ```

-  **Multiple Arguments**: You can print multiple items by separating them with commas. Python automatically adds a space between them.

    ```python
    name = "Alice"
    age = 30
    print(name, "is", age, "years old.")  # Output: Alice is 30 years old.
    ```

-  **String Formatting**: You can format strings to include variables in a more readable way. In the example, we used an f-string (formatted string literal) to include the average age in the output.

    ```python
    avg_age = 32
    print(f"The average age is {avg_age}")  # Output: The average age is 32
    ```

-  **Custom Separator and End Character**: You can customize how items are separated and what character is printed at the end of the output.

    ```python
    print("Hello", "World", sep="-")  # Output: Hello-World
    print("Hello", end="!")            # Output: Hello!
    ```

### 2. **Lists in Python**

A **list** is a built-in data structure in Python that allows you to store a collection of items. Here are some key points about lists:

-  **Definition**: A list is defined using square brackets `[]`, and items within the list are separated by commas.
  
    ```python
    ages = [25, 30, 35, 40]
    names = ["Alice", "Bob", "Charlie"]
    ```

-  **Ordered**: Lists maintain the order of items. The first item has an index of `0`, the second item has an index of `1`, and so on.

-  **Mutable**: Lists are mutable, which means you can change their content (add, remove, or modify items) after they have been created.

    ```python
    ages = [25, 30, 35, 40]
    ages[1] = 32  # Changes the second element from 30 to 32
    ```

-  **Heterogeneous**: Lists can contain items of different data types (e.g., integers, strings, other lists).

#### Example of List Operations

Here are some basic operations you can perform on lists:

```python
# Creating a list
my_list = [1, 2, 3, 4, 5]

# Accessing elements
print(my_list[0])  # Output: 1 (first element)
print(my_list[2])  # Output: 3 (third element)

# Modifying elements
my_list[1] = 20   # Changes the second element to 20
print(my_list)    # Output: [1, 20, 3, 4, 5]

# Adding elements
my_list.append(6)  # Adds 6 to the end of the list
print(my_list)     # Output: [1, 20, 3, 4, 5, 6]

# Removing elements
my_list.remove(20)  # Removes the element 20 from the list
print(my_list)      # Output: [1, 3, 4, 5, 6]
```

### Putting It All Together

In the example provided earlier, we used a list to store ages and then calculated the average age using a function. Finally, we printed the result to the console:

```python
ages = [25, 30, 35, 40, 67, 80]  # A list containing ages
avg_age = calculate_average_age(ages)  # Calculate average using the function
print(f"The new average age is {avg_age}")  # Print the result
```

-  **`ages`**: This is a list containing four integers representing the ages of users.
-  **`print(f"The average age is {avg_age}")`**: This line uses the `print` function to display the average age calculated by the `calculate_average_age` function, formatted nicely within a string.

This combination of lists and the `print` function is fundamental in Python and allows you to handle collections of data and display results effectively.

