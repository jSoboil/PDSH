## Visualization with Matplotlib
### Simple Scatter Plots
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

# 1st example...
plt.clf()
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.plot(x, y, 'o', color = "black")
plt.show()

#### Setting Styles
#  Here we will set the classic style, which ensures that the plots we create use
# the classic Matplotlib style.
plt.style.use("classic")

##### Plotting from an IPython notebook
# Note that, although this does not work in Rstudio, it is important to use these
# commands when use a python notebook...

# %matplotlib notebook will lead to interactive plots embedded within the 
# notebook

# %matplotlib inline will lead to static images of your plot embedded in the 
# notebook

# Alternatively in Rstudio, one can important the matplotlib_inline library...

# Remember that it needs to be done only once per kernel/session. See below
# for another example...
plt.clf()
x = np.linspace(0, 10, 100)
# Plot...
fig = plt.figure()
plt.plot(x, np.sin(x), "-")
plt.plot(x, np.cos(x), "--")
plt.show()

#### Saving Figures to File
# You can save a figure using the savefig() command. For example, to save the 
# previous figure as a PNG file, you can run this.
fig.savefig("img/my_fig.png")
# Although the book recommends using the IPython Image object to display the 
# contents of this file when working in a Python notebook, we can't in Rstudio.
# But we can do the following...
plt.show("img/my_fig.png")

## Two Interfaces for the Price of One
# A potentially confusing feature of Matplotlib is its dual interfaces: a 
# convenient MATLAB-style state-based interface, and a more powerful 
# object-oriented interface. We’ll quickly highlight the differences between the 
# two here.

### MATLAB-style interface
# Matplotlib was originally written as a Python alternative for MATLAB users, and
# much of its syntax reflects that fact. The MATLAB-style tools are contained in
# the pyplot (plt) interface. For example, the following code will probably look
# quite familiar to MATLAB users.
plt.figure() # create a plot figure
# create the first two panels and set current axis
plt.subplot(2, 1, 1) # (rows, columns, panel number)
plt.plot(x, np.sin(x))
# create second panel and set current axis
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.show()
# It’s important to note that this interface is stateful: it keeps track of the 
# “current” figure and axes, which are where all plt commands are applied. You 
# can get a reference to these using the plt.gcf() (get current figure) and 
# plt.gca() (get current axes) routines.

# While this stateful interface is fast and convenient for simple plots, it is 
# easy to run into problems. For example, once the second panel is created, how 
# can we go back and add something to the first? This is possible within the 
# MATLAB-style interface, but a bit clunky. Fortunately, there is a better way.

### Object-oriented interface
# The object-oriented interface is available for these more complicated 
# situations, and for when you want more control over your figure. Rather than 
# depending on some notion of an “active” figure or axes, in the object-oriented
# interface the plotting func‐ tions are methods of explicit Figure and Axes 
# objects. To re-create the previous plot using this style of plotting, you might
# do the following...

# First create a grid of plots
# ax will be am array of two Axes objects
fig, ax = plt.subplots(2)
# Call plot() method on the appropriate object
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x));
plt.show()
# In most cases, the difference is as small as switching plt.plot() to ax.plot(),
# but there are a few gotchas.

## Simple Line Plots
# Let's visualise a simple y = f(x) function.
# In a Python notebook we would add...
# %matplotlib inline/notebook

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import numpy as np

plt.clf()
# For all Matplotlib plots, we start by creating a figure and an axes. In their 
# simplest form, a figure and axes can be created as follows
fig = plt.figure()
ax = plt.axes()
plt.show()

# Once we have created an axes, we can use the ax.plot function to plot some 
# data. Let’s start with a simple sinusoid.
x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x));
plt.show()

# If we want to create a single figure with multiple lines, we can simply call 
# the plot function multiple times, as follows...
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x));
plt.show()

### Adjusting the Plot: Line Colors and Styles
# The plt.plot() function takes additional arguments that can be used to specify
# these. To adjust the colour, you can use the color keyword, which accepts a 
# string argument representing virtually any imaginable colour. The colour can 
# be specified in a variety of ways.
plt.plot(x, np.sin(x - 0), color = "blue") # by name
plt.plot(x, np.sin(x - 1), color = "g") # by short colour code (rgbcmyk)
plt.plot(x, np.sin(x - 2), color = "0.75") # by grayscale between 0 and 1
plt.plot(x, np.sin(x - 3), color = "#FFDD44") # by Hex code
plt.plot(x, np.sin(x - 4), color = (1.0,0.2,0.3)) # by RGB tuple, values 0 and 1
plt.plot(x, np.sin(x - 5), color = "chartreuse") # by HTML colour names
plt.show()
# Note that if no colour is specified, Matplotlib will automatically cycle 
# through a set of default colors for multiple lines.

# Similarly, you can adjust the line style using the linestyle keyword.
plt.clf()
plt.plot(x, x + 0, linestyle = "solid")
plt.plot(x, x + 1, linestyle = "dashed")
plt.plot(x, x + 2, linestyle = "dashdot")
plt.plot(x, x + 3, linestyle = "dotted")
plt.show()

### Adjusting the Plot: Axes Limits
















