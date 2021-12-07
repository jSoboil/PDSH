# Chapter 2:
# This outlines techniques for effectively loading, storing, and manipulating
# in-memory data in Python.
import numpy as np

result = 0
for i in range(100):
 result += i
result

# Exploring integers
L = list(range(100))
type(L[0])

# Fixed Type Arrays in Python:
import array
L = list(range(10))
A = array.array('i', L)
A
# Here 'i' is a type code indicating the contents are integers.

# Much more useful, however, is the ndarray object of the NumPy package. While 
# Python’s array object provides efficient storage of array-based data, NumPy 
# adds to this efficient operations on that data. We explore these operations in
# later sections; here we demonstrate several ways of creating a NumPy array.

## Creating Arrays from Python Lists:
# We can use np.array to create arrays from Python lists. For example, the 
# integer array below.
np.array([1, 2, 3, 4, 5])

# Unlike Python lists, NumPy is constrained to arrays that all contain the same 
# type. If types do not match, NumPy will upcast if possible (here, integers are
# upcast to floating point). For example...
np.array([3.14, 4, 2, 3])

# If we want to explicitly set the data type of the resulting array, we can use
# the dtype keyword.
np.array([1, 2, 3, 4], dtype = 'float32')

# Finally, unlike Python lists, NumPy arrays can explicitly be multidimensional; 
# here’s one way of initializing a multidimensional array using a list of lists.
# The nested lists result in multidimensional arrays...
np.array([range(i, i + 3) for i in [2, 4, 6]])
# The inner lists are treated as rows of the resulting two-dimensional array.

## Creating Arrays from Scratch:
# Especially for larger arrays, it is more efficient to create arrays from scratch
# using routines built into NumPy. Below are several examples.

# 1. A length-10 integer array filled with zeros
np.zeros(10, dtype = int)

# 2. A 3x5 floating-point array filled with 1s
np.ones((3, 5), dtype = float)

# 3. A 3x5 array filled with 3.14
np.full((3, 5), 3.14)

# 4. An array filled with a linear sequence starting at 0, ending at 20, stepping 
# by 2 (which is similar to the standard range function)
np.arange(0, 20, 2)

# 5. An array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)

# 6. A 3x3 array of uniformly distributed random values between 0, 1
np.random.random((3, 3))

# 7. A 3x3 array of normally distributed random values with mean 0 and sd 1
np.random.normal(0, 1, (3, 3))

# 8. A 3x3 I matrix
np.eye(3)

# The Basics of NumPy Arrays:
# This section presents several examples using NumPy array manipulation to access
# data and subarrays, and to split, reshape, and join the arrays.

## NumPy Attributes:
# We’ll start by defining three random arrays: a one-dimensional, two-dimensional, 
# and three-dimensional array. We’ll use NumPy’s random number generator, which
# we will seed with a set value in order to ensure that the same random arrays
# are generated each time this code is run
np.random.seed(0)

# One-dimensional array
x_1 = np.random.randint(10, size = 6)
# Two-dimensional array
x_2 = np.random.randint(10, size = (3, 4))
# Three-dimensional array
x_3 = np.random.randint(10, size = (3, 4, 5))

# Each array has attributes ndim (the number of dimensions), shape (the size of
# each dimension), and size (the total size of the array).
print("x_3 ndim: ", x_3.ndim)
print("x_3 shape: ", x_3.shape)
print("x_3 size: ", x_3.size)

## Array Indexing - Accessing Single Elements:
# Much the same as standard R. But important to note that unlike Python lists, 
# NumPy arrays have a fixed type. This means, for example, that if you attempt 
# to insert a floating-point value to an integer array, the value will be 
# silently truncated. Don’t be caught unaware by this behavior!
x_1[0] = 3.14159 # this is silently truncated!

## Reshaping of Arrays:
# Another useful type of operation is reshaping of arrays. The most flexible way
# of doing this is with the reshape() method. For example, if you want to put 
# the numbers 1 through 9 in a 3×3 grid, you can do the following.
grid = np.arange(1, 10).reshape((3, 3))
print(grid)
# Note that for this to work, the size of the initial array must match the size
# of the reshaped array. Where possible, the reshape method will use a no-copy
# view of the initial array, but with noncontiguous memory buffers this is not 
# always the case.

# Another common reshaping pattern is the conversion of a one-dimensional array 
# into a two-dimensional row or column matrix. You can do this with the reshape
# method, or more easily by making use of the newaxis keyword within a slice 
# operation.
x = np.array([1, 2, 3])
# Reshape into ropw vector via reshape
x.reshape((1, 3))
# Row vector via newaxis
x[np.newaxis, :]
# Column vector via reshape
x.reshape((3, 1))
# Column vector via newaxis
x[ : , np.newaxis]

## Array Concatenation and Splitting:
### Concatenation of arrays:
