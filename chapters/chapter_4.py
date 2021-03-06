## Visualization with Matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

### 1st example...
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
# Similar to R, the most basic way to adjust axis limits is to use the 
# plt.xlim() and plt.ylim() methods.
plt.clf()
plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)
plt.show()
# If you’d like either axis to be displayed in reverse, you can simply reverse
# the order of the arguments
plt.clf()
plt.plot(x, np.sin(x))
plt.xlim(10, 0)
plt.ylim(1.2, -1.2)
plt.show()

# A useful related method is plt.axis() (note here the potential confusion 
# between axes with an e, and axis with an i). The plt.axis() method allows you 
# to set the x and y limits with a single call, by passing a list that specifies 
# [xmin, xmax, ymin, ymax].
plt.clf()
plt.plot(x, np.sin(x))
# I find this method preferable...
plt.axis([-1, 11, -1.5, 1.5]);
plt.show()
# The plt.axis() method goes even beyond this, allowing you to do things like 
# automatically tighten the bounds around the current plot.
plt.clf()
plt.plot(x, np.sin(x))
plt.axis("tight")
plt.show()
# It also allows even higher-level specifications, such as ensuring an equal 
# aspect ratio so that on your screen, one unit in x is equal to one unit in y.
plt.clf()
plt.plot(x, np.sin(x))
plt.axis('equal');
plt.show()

### Labeling Plots
# Titles and axis labels are the simplest such labels—there are methods that can
# be used to quickly set them.
plt.clf()
plt.plot(x, np.sin(x))
plt.title("A Sine Curve")
# Very similar to tidyverse ggplot, but instead of a pipe it's like base R...
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()
# You can adjust the position, size, and style of these labels using optional 
# arguments to the function.

# When multiple lines are being shown within a single axes, it can be useful to 
# create a plot legend that labels each line type. It is done via the 
# plt.legend() method. Though there are several valid ways of using this, but 
# this is probably the easiest.
plt.clf()
plt.plot(x, np.sin(x), "-g", label = "sin(x)")
plt.plot(x, np.cos(x), ":b", label = "cos(x)")
plt.axis("tight")
plt.legend(fontsize = 6, loc = 3);
plt.show()

### Simple Scatter Plots
# The primary difference of plt.scatter from plt.plot is that it can be used to 
# create scatter plots where the properties of each individual point (size, face 
# color, edge color, etc.) can be individually controlled or mapped to data.

# Let’s show this by creating a random scatter plot with points of many colors 
# and sizes.
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colours = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.clf()
plt.scatter(x, y, c = colours, s = sizes, alpha = 0.3, cmap = "viridis")
plt.colorbar(); #show colour scale
plt.show()

# For a better example, we can use the Iris data from Scikit-Learn.
from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T
# Plot...
plt.clf()
plt.scatter(features[0], features[1], alpha = 0.2, s = 100 * features[3], c = iris.target, cmap = "viridis")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1]);
plt.show()
plt.clf()

### Visualizing Errors
#### Basic Errorbars
# A basic errorbar can be created with a single Matplotlib function call.
plt.clf()
plt.style.use("seaborn-whitegrid")
x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)
plt.errorbar(x, y, yerr = dy, fmt = ".k")
plt.show()
# In addition to these basic options, the errorbar function has many options to
# fine-tune the outputs.
plt.errorbar(x, y, yerr = dy, fmt = "o", color = "black", ecolor = "lightgray", elinewidth = 3, capsize = 0);
plt.show()

#### Continuous Errors
# In some situations it is desirable to show errorbars on continuous quantities.
# Though Matplotlib does not have a built-in convenience routine for this type 
# of application, it’s relatively easy to combine primitives like plt.plot and 
# plt.fill_between for a useful result.

# Here we’ll perform a simple Gaussian process regression (GPR), using the 
# Scikit-Learn API.
from sklearn.gaussian_process import GaussianProcessRegressor

# # define the model and draw some data
model = lambda x: x * np.sin(x)
x_data = np.array([1, 3, 5, 6, 8])
y_data = model(x_data)
# Compute the Guassian process fit
gp = GaussianProcessRegressor(random_state = 100)
gp.fit(x_data[ : , np.newaxis], y_data)

x_fit = np.linspace(0, 10, 1000)
y_fit, MSE = gp.predict(x_fit[ : , np.newaxis], return_std = True)
dy_fit = 2 * np.sqrt(MSE) # 2 * sigma ~ 95% confidence region

# We now have xfit, yfit, and dyfit, which sample the continuous fit to our 
# data. We can use the plt.fill_between function with a light color to 
# visualize this continuous error.
plt.clf()
plt.plot(x_data, y_data, "or")
plt.plot(x_fit, y_fit, "-", color = "gray")
plt.fill_between(x_fit, y_fit - dy_fit, y_fit + dyfit, color = "gray", alpha = 0.2)
plt.xlim(0, 10);
plt.show()

### Density and Contour Plots
#### Visualizing a Three-Dimensional Function
# We’ll start by demonstrating a contour plot using a function z = f (x, y), 
# using the following particular choice for f.
def f(x, y): return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

# A contour plot can be created with the plt.contour function. It takes three 
# argu‐ ments: a grid of x values, a grid of y values, and a grid of z values. 
# The x and y values represent positions on the plot, and the z values will be 
# represented by the contour levels. Perhaps the most straightforward way to 
# prepare such data is to use the np.meshgrid function, which builds 
# two-dimensional grids from one-dimensional arrays.
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
# Now let’s look at this with a standard line-only contour plot
plt.clf()
plt.contour(X, Y, Z, colors = "black")
plt.show()
# Notice that by default when a single color is used, negative values are 
# represented by dashed lines, and positive values by solid lines. 
# Alternatively, you can color-code the lines by specifying a colormap with the 
# cmap argument.
plt.clf()
plt.contour(X, Y, Z, 20, cmap = "RdGy")
plt.show()
# Our plot is looking nicer, but the spaces between the lines may be a bit 
# distracting. We can change this by switching to a filled contour plot using 
# the plt.contourf() function (notice the f at the end), which uses largely the 
# same syntax as plt.con tour(). Additionally, we’ll add a plt.colorbar() 
# command, which automatically creates an additional axis with labeled color 
# information for the plot.
plt.clf()
plt.contour(X, Y, Z, 20, cmap = "RdGy");
plt.colorbar();
plt.show()
# One potential issue with this plot is that it is a bit “splotchy.” That is, 
# the color steps are discrete rather than continuous, which is not always what
# is desired. o handle this is to use the plt.imshow() function, which 
# interprets a two-dimensional grid of data as an image.
plt.clf()
plt.imshow(Z, extent = [0, 5, 0, 5], origin = "lower", cmap = "RdGy")
plt.colorbar()
plt.axis("scaled")
plt.show()
# Finally, it can sometimes be useful to combine contour plots and image plots.
# We’ll use a partially transparent background image (with transparency set via
# the alpha parameter) and over-plot contours with labels on the contours 
# themselves (using the plt.clabel() function).
plt.clf()
contours = plt.contour(X, Y, Z, 3, colors = "black")
plt.clabel(contours, inline = True, fontsize = 8)
plt.imshow(Z, extent = [0, 5, 0, 5], origin = "lower", cmap = "RdGy")
plt.colorbar()
plt.show()

### Histograms, Binnings, and Density
# The hist() function has many options to tune both the calculation and the 
# display; here’s an example of a more customized histogram.
data = np.random.randn(1000)
plt.clf()
plt.hist(data, bins = 30, stacked = True, alpha = 0.5, histtype = "stepfilled", color = "steelblue", edgecolor = "none");
plt.show()
?plt.hist
# As a further example of multi-plots...
x_1 = np.random.normal(0, 0.8, 1000)
x_2 = np.random.normal(-2, 1, 1000)
x_3 = np.random.normal(3, 2, 1000)
kwargs = dict(histtype = "stepfilled", alpha = 0.3, stacked = True, bins = 40)

plt.clf()
plt.hist(x_1, **kwargs)
plt.hist(x_2, **kwargs)
plt.hist(x_3, **kwargs)
plt.show()

# Tired of plots that I can learn on the fly... moving on to chapter 5.

# End file
