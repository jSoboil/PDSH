# Chapter 2:
# Note: before starting Python in R, you have to use the reticulate::repl_python() 
# command, which loads a Python interpreter.

# Chapter 2 outlines techniques for effectively loading, storing, and manipulating
# in-memory data in Python.
import numpy as np

result = 0
for i in range(100):
 result += i
print(result)
