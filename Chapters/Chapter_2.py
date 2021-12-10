#################
### Chapter 2 ###
#################
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

#### Fixed Type Arrays in Python
import array
L = list(range(10))
A = array.array('i', L)
A
# Here 'i' is a type code indicating the contents are integers.

# But much more useful is the ndarray object of the NumPy package. While 
# Python’s array object provides efficient storage of array-based data, NumPy 
# adds to this efficient operations on that data. We explore these operations in
# later sections; here we demonstrate several ways of creating a NumPy array.

##### Creating Arrays from Python Lists
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

##### Creating Arrays from Scratch
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

#### The Basics of NumPy Arrays
# This section presents several examples using NumPy array manipulation to access
# data and subarrays, and to split, reshape, and join the arrays.

##### NumPy Attributes
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

##### Array Indexing - Accessing Single Elements
# Much the same as standard R. But important to note that unlike Python lists, 
# NumPy arrays have a fixed type. This means, for example, that if you attempt 
# to insert a floating-point value to an integer array, the value will be 
# silently truncated. Don’t be caught unaware by this behavior!
x_1[0] = 3.14159 # this is silently truncated!

##### Reshaping of Arrays
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

#### Array Concatenation and Splitting
##### Concatenation of arrays
# Concatenation, or joining of two arrays in NumPy, is primarily accomplished
# through the routines np.concatenate, np.vstack, and np.hstack. np.concatenate 
# takes a tuple or list of arrays as its first argument, as we can see here.
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])

# You can also concatenate more than two arrays at once.
z = [99, 99, 99]
print(np.concatenate([x, y, z]))

###### Splitting of arrays
# The opposite of concatenation is splitting, which is implemented by the 
# functions np.split, np.hsplit, and np.vsplit. For each of these, we can pass a 
# list of indices giving the split points.
x = [1, 2, 3, 99, 99, 3, 2, 1]
x_1, x_2, x_3 = np.split(x, [3, 5])
print(x_1, x_2, x_3)

#### Computation on NumPy Arrays - Universal Functions
##### The Slowness of Loops
# The relative sluggishness of Python generally manifests itself in situations
# where many small operations are being repeated — for instance, looping over 
# arrays to operate on each element. For example, imagine we have an array of 
# values and we’d like to compute the reciprocal of each. A straightforward 
# approach might look like this below...
np.random.seed(0)
def compute_reciprocals(values):
 output = np.empty(len(values))
 for i in range(len(values)):
  output[i] = 1.0 / values[i]
 return output

values = np.random.randint(1, 10, size = 5)
compute_reciprocals(values)
# This implementation probably feels fairly natural to someone from, say, a C or
# Java background. But if we measure the execution time of this code for a large
# input, we see that this operation is very slow, perhaps surprisingly so!

# Each time the reciprocal is computed, Python first examines the object’s type
# and does a dynamic lookup of the correct function to use for that type. If we
# were working in compiled code instead, this type specification would be known
# before the code executes and the result could be computed much more 
# efficiently.

#### Introducing UFuncs
# Ah... good ol' vectorisation. Also a strategy in R. Instead of iterating over
# individual values, we can treat the object as a vector and do the following...
print(1.0 / values)

# Vectorized operations in NumPy are implemented via ufuncs, whose main purpose
# is to quickly execute repeated operations on values in NumPy arrays. Ufuncs 
# are extremely flexible — before we saw an operation between a scalar and an 
# array, but we can also operate between two arrays like so
np.arange(5) / np.arange(1, 6)

# And ufunc operations are not limited to one-dimensional arrays — they can act
# on multidimensional arrays as well. See below.
x = np.arange(9).reshape((3, 3))
2 ** x

#### Exploring NumPy’s UFuncs
# Ufuncs exist in two flavors: unary ufuncs, which operate on a single input, 
# and binary ufuncs, which operate on two inputs. We’ll see examples of both 
# these types of functions here.

##### Array arithmetic
# The standard addition, subtraction, multiplication, and division can all be 
# used...
x = np.arange(4)
print("x =", x)
print("x =", x - 5)
print("x =", x + 5)
print("x =", x * 2)
print("x / 2", x / 2)
print("x // 2", x // 2) # floor division

# There is also a unary ufunc for negation, a ** operator for exponentiation, 
# and a % operator for modulus.
print("-x ", -x)
print("x ** 2", x ** 2)
print("x % 2", x % 2)

#### Aggregations - Min, Max, and Everything in Between
##### Example - What Is the Average Height of US Presidents?
# Aggregates available in NumPy can be extremely useful for summarizing a set of
# values. As a simple example, let’s consider the heights of all US presidents. 
# This data is available in the file president_heights.csv, which is a simple 
# comma-separated list of labels and values.
from scipy import special
import pandas as pd

# We can use the pandas package to read and extract the data.
data = pd.read_csv('data/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)
# Now that we have this data array, we can compute a variety of summary 
# statistics.
print("The mean height is: ", heights.mean())
print("Standard deviation of heights is: ", heights.std())
print("The minimum height is: ", heights.min())
print("The maximum height is: ", heights.max())
# We may also wish to compute quantiles.
print("The 25th percentile is: ", np.percentile(heights, 25))
print("The median: ", np.median(heights))
print("The 75th percentile is: ", np.percentile(heights, 75))

# Sometimes it’s more useful to see a visual representation of this data, which
# we can accomplish using tools in Matplotlib. See below for an example.
import matplotlib.pyplot as plt
import seaborn; seaborn.set() # set plot style
# Create plot
plt.hist(heights)
plt.title("Height Distribution of US Presidents")
plt.xlabel("height(cm)")
plt.ylabel("number")
plt.show() # Note: in Rstudio, you need to call plt.show() to make plot appear

#### Computation on Arrays - Broadcasting
# Another means of vectorising operations is to use NumPy’s broadcasting 
# functionality. Broadcasting is simply a set of rules for applying binary 
# ufuncs (addition, subtraction, multiplication, etc.) on arrays of different 
# sizes.

##### Introducing Broadcasting
# Recall that for arrays of the same size, binary operations are performed on an
# element-by-element basis. Broadcasting allows these types of binary operations
# to be performed on arrays of different sizes — for example, we can just as 
# easily add a scalar (think of it as a zero - dimensional array) to an array.
a = np.array([0, 1, 2])
a + 5

# We can similarly extend this to arrays of higher dimension. Observe the result
# when we add a one-dimensional array to a two-dimensional array, below.
M = np.ones((3, 3))
M + a

#### Rules of Broadcasting

# Rule 1: If the two arrays differ in their number of dimensions, the shape of 
# the one with fewer dimensions is padded with ones on its leading (left) side.

# Rule 2: If the shape of the two arrays does not match in any dimension, the 
# array with shape equal to 1 in that dimension is stretched to match the other 
# shape.

# Rule 3: If in any dimension the sizes disagree and neither is equal to 1, an 
# error is raised.  

##### Broadcasting example 1
# Let’s look at adding a two-dimensional array to a one-dimensional array, below.
M = np.ones((2, 3))
a =np.arange(3)
# Let’s consider an operation on these two arrays. The shapes of the arrays are
M.shape = (2, 3)
a.shape = (3, )
# We see by rule 1 that the array a has fewer dimensions, so we pad it on the
# left with one
# M.shape -> (2, 3)
# a.shape -> (1, 3)
# By rule 2, we now see that the first dimension disagrees, so we stretch this 
# dimension to match:
# M.shape -> (2, 3)
# a.shape -> (2, 3)

# The shapes match, and we see that the final shape will be (2, 3):
M + a

##### Broadcasting example 2
# Let’s take a look at an example where both arrays need to be broadcast.
a = np.arange(3).reshape((3, 1))
b = np.arange(3)
# Rule 1 says we must pad the shape of b with ones...
# a.shape -> (3, 1)
# b.shape -> (1, 3)
# And rule 2 tells us that we upgrade each of these ones to match the 
# corresponding size of the other array.
# a.shape -> (3, 3)
# b.shape -> (3, 3)

# Because the result matches, these shapes are compatible. We can see this below.
a + b

##### Broadcasting example 3
# Now let’s take a look at an example in which the two arrays are not compatible.
M = np.ones((3, 2))
a = np.arange(3)

# This is just a slightly different situation than in the first example - the 
# matrix M is transposed. How does this affect the calculation? The shapes of 
# the arrays are...
M.shape 
a.shape
# Again, rule 1 tells us that we must pad the shape of a with ones.
# M.shape -> (3, 2)
# a.shape -> (1, 3)
# By rule 2, the first dimension of a is stretched to match that of M.
# M.shape -> (3, 2)
# a.shape -> (3, 3)
# Now we hit rule 3—the final shapes do not match, so these two arrays are 
# incompatible, as we can observe by attempting this operation.
M + a

# If right-side padding is what you’d like, you can do this explicitly by 
# reshaping the array.
a[:, np.newaxis].shape
M + a[:, np.newaxis]

# Also note that while we’ve been focusing on the + operator here, these 
# broadcasting rules apply to any binary ufunc. For example, here is the 
# logaddexp(a, b) function, which computes log(exp(a) + exp(b)) with more 
# precision than the naive approach.
np.logaddexp(M, a[:, np.newaxis])

#### Broadcasting in Practice
##### Centering an array
# We saw that ufuncs allow a NumPy user to remove the need to explicitly write 
# slow Python loops. Broadcasting extends this ability. One commonly seen 
# example is centering an array of data. Imagine you have an array of 10 
# observations, each of which consists of 3 values. Using the standard 
# convention, we’ll store this in a 10×3 array.
X = np.random.random((10, 3))
# We can compute the mean of each feature using the mean aggregate across the 
# first dimension.
X_mean = X.mean(0)
X_mean

# Now, we can center the X array by subtracting the mean (this is a broadcasting 
# operation).
X_centred = X - X_mean
# To double-check that we’ve done this correctly, we can check that the centered 
# array has near zero mean.
X_centred.mean(0)
# To within-machine precision, the mean is now zero.

###### Plotting a two-dimensional function
# One place that broadcasting is very useful is in displaying images based on 
# two- dimensional functions. If we want to define a function z = f(x, y), 
# broadcasting can be used to compute the function across the grid.
x = np.linspace(0, 5, 50) # x has 50 steps from 0 to 5
y = np.linspace(0, 5, 50) # y has 50 steps from 0 to 5
# Write func...
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
# We’ll use Matplotlib to plot this two-dimensional array.
plt.clf()
plt.imshow(z, origin = "lower", extent = [0, 5, 0, 5], cmap = "viridis")
plt.show()

#### Comparisons, Masks, and Boolean Logic
##### Example: Counting Rainy Days
# Imagine you have a series of data that represents the amount of precipitation
# each day for a year in a given city. For example, here we’ll load the daily 
# rainfall statistics for the city of Seattle in 2014, using Pandas.
import pandas as pd
# Use Pandas to extract rainfall inches as a NumPy array.
rainfall = pd.read_csv("data/Seattle2014.csv")['PRCP'].values
inches = rainfall / 254 # 1/10mm -> inches
inches.shape
# The array contains 365 values, giving daily rainfall in inches from January 1
# to December 31, 2014.

# As a first quick visualization, let’s look at the histogram of rainy days.
plt.clf()
plt.hist(inches, 40)
plt.show()
# This doesn’t do a good job of conveying some information we’d like to see: for
# example, how many rainy days were there in the year? What is the average 
# precipitation on those rainy days? How many days were there with more than 
# half an inch of rain?

##### Comparison Operators as ufuncs
# Using +, -, *, /, and others on arrays leads to element-wise operations. NumPy
# also implements comparison operators such as < (less than) and > (greater than)
# as element-wise ufuncs. The result of these comparison operators is always an
# array with a Boolean data type. All six of the standard comparison operations
# are available. For example
x = np.array([1, 2, 3, 4, 5])
x < 3
x > 3
x <= 3
x >= 3
x != 3
x == 3

# It is also possible to do an element-by-element comparison of two arrays, and 
# to include compound expressions.
(2 * x) == (x ** 2)
# As in the case of arithmetic operators, the comparison operators are 
# implemented as ufuncs in NumPy; for example, when you write x < 3, internally
# NumPy uses np.less(x, 3).

#### Working with Boolean Arrays




















