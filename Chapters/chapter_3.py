### Data Manipulation with Pandas
#### Data Manipulation with Pandas
# In this chapter, we will focus on the mechanics of using Series, DataFrame, and 
# related structures effectively.
import pandas as pd
import numpy as np

##### Introducing Pandas Objects
# At the very basic level, Pandas objects can be thought of as enhanced versions 
# of NumPy structured arrays in which the rows and columns are identified with 
# labels rather than simple integer indices.

###### The Pandas Series Object
# A Pandas Series is a one-dimensional array of indexed data. It can be created 
# from a list or array as below.
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data

# The Series wraps both a sequence of values and a sequence of indices, which we 
# can access with the values and index attributes.
data.values
# The index is an array-like object of type pd.Index
data.index
# Like with a NumPy array, data can be accessed by the associated index via the 
# familiar Python square-bracket notation.
data[1]
data[1:3]

####### Series as generalized NumPy array
# The Pandas Series has an explicitly defined index associated with the values. 
# This explicit index definition gives the Series object additional capabilities.
# For example, the index need not be an integer, but can consist of values of any
# desired type. For example, if we wish, we can use strings as an index as is 
# shown below.
data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ["a", "b", "c", "d"])
data
data["b"]

# Many of these things are very similar to R, I', not going to go through them in
# detail.

##### Merge and Join operations
###### Example: US States Data
pop = pd.read_csv("data/state-population.csv")
areas = pd.read_csv("data/state-areas.csv")
abbrevs = pd.read_csv("data/state-abbrevs.csv")

print(pop.head()); print(areas.head()); print(abbrevs.head())
# Given this information, say we want to compute a relatively straightforward 
# result - rank US states and territories by their 2010 population density. We 
# clearly have the data here to find this result, but we’ll have to combine the 
# datasets to get it.

# We’ll start with a many-to-one merge that will give us the full state name 
# within the population DataFrame. We want to merge based on the state/region 
# column of pop, and the abbreviation column of abbrevs. We’ll use how = 'outer' 
# to make sure no data is thrown away due to mismatched labels.
merged_data = pd.merge(pop, abbrevs, how = "outer", left_on = "state/region", right_on = "abbreviation")
merged_data = merged_data.drop("abbreviation", 1)
merged_data.head()
# Let’s double-check whether there were any mismatches here, which we can do by 
# looking for rows with nulls.
merged_data.isnull().any()
# Some of the population info is null; let’s figure out which these are!
merged_data[merged_data["population"].isnull()].head()
# It appears that all the null population values are from Puerto Rico prior to the
# year 2000; this is likely due to this data not being available from the original
# source.

# More importantly, we see also that some of the new state entries are also null, 
# which means that there was no corresponding entry in the abbrevs key! Let’s 
# figure out which regions lack this match.
merged_data.loc[merged_data["state"].isnull(), "state/region"].unique()

# We can quickly infer the issue: our population data includes entries for Puerto 
# Rico (PR) and the United States as a whole (USA), while these entries do not 
# appear in the state abbreviation key. We can fix these quickly by filling in 
# appropriate entries.
merged_data.loc[merged_data["state/region"] == "PR", "state"] = "Puerto Rico"
merged_data.loc[merged_data["state/region"] == "USA", "state"] = "United States"
merged_data.isnull().any()
# No more nulls in the state column - we’re all set!

# Now we can merge the result with the area data using a similar procedure. 
# Examining our results, we will want to join on the state column in both.
data_final = pd.merge(merged_data, areas, on = "state", how = "left")
data_final.head()

# Again, let’s check for nulls to see if there were any mismatches.
data_final.isnull().any()
# There are nulls in the area column; we can take a look to see which regions
# were ignored here.
data_final["state"][data_final["area (sq. mi)"].isnull()].unique()
# We see that our areas DataFrame does not contain the area of the United States
# as a whole. We could insert the appropriate value (using the sum of all state
# areas, for instance), but in this case we’ll just drop the null values because 
# the population density of the entire United States is not relevant to our 
# current discussion.
data_final.dropna(inplace = True)
data_final.head()

# To answer the question of interest, let’s first select the portion of the data
# corresponding with the year 2000, and the total population. We’ll use the 
# query() function to do this quickly.
data_2010 = data_final.query("year == 2010 & ages == 'total'")
data_2010.head()
# Now let’s compute the population density and display it in order. We’ll start
# by reindexing our data on the state, and then compute the result.
data_2010.set_index("state", inplace = True)
density = data_2010["population"] / data_2010["area (sq. mi)"]

density.sort_values(ascending = False, inplace = True)
density.head()
# The result is a ranking of US states plus Washington, DC, and Puerto Rico in 
# order of their 2010 population density, in residents per square mile.

# We can also check the end of the list.
density.tail()

##### Aggregation and Grouping
###### Example: Birthrate Data
# Let’s take a look at the freely available data on births in the United States, 
# provided by the Centers for Disease Control (CDC).
births = pd.read_csv("data/births.csv")
# Taking a look at the data, we see that it’s relatively simple—it contains the number of
# births grouped by date and gender.
births.head()

# We can start to understand this data a bit more by using a pivot table. Let’s 
# add a decade column, and take a look at male and female births as a function 
# of decade.
births["decade"] = 10 * (births["year"] // 10)
births.pivot_table("births", index = "decade", columns = "gender", aggfunc = "sum")

# We immediately see that male births outnumber female births in every decade. 
# To see this trend a bit more clearly, we can use the built-in plotting tools 
# in Pandas to visual‐ ize the total number of births by year.
plt.clf()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
births.pivot_table("births", index = "year", columns = "gender", aggfunc = "sum").plot()
plt.ylabel("Total births per year")
plt.show()
plt.clf() # render plot for Rstudio
plt.clf() # clear settings
# With a simple pivot table and plot() method, we can immediately see the annual
# trend in births by gender. By eye, it appears that over the past 50 years male
# births have outnumbered female births by around 5%.

###### Further data exploration
# There are a few more interest‐ ing features we can pull out of this dataset 
# using the Pandas tools covered up to this point. We must start by cleaning the 
# data a bit, removing outliers caused by mistyped dates (e.g., June 31st) or 
# missing values (e.g., June 99th). One easy way to remove these all at once is 
# to cut outliers; we’ll do this via a robust sigma-clipping operation.
quartiles = np.percentile(births["births"], [25, 50, 75])
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[0])
# With this we can use the query() method to filter out rows with births outside
# these values.
births = births.query("(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)")
# Next we set the day column to integers; previously it had been a string 
# because some columns in the dataset contained the value 'null'.
births["day"] = births["day"].astype(int)
births["day"].dtype
# Finally, we can combine the day, month, and year to create a Date index. This 
# allows us to quickly compute the weekday corresponding to each row.
births.index = pd.to_datetime(10000 * births.year + 100 * births.month + births.day, format = "%Y%m%d")
births["dayofweek"] = births.index.dayofweek
# Using this we can plot births by weekday for several decades
import matplotlib as mpl
births.pivot_table("births", index = "dayofweek", columns = "decade", aggfunc = "mean").plot()
plt.gca().set_xticklabels(["Mon", "Tues", "Wed", "Thurs", "Fri", "Sat", "Sun"])
plt.ylabel("mean births by day")
plt.show() # render plot for Rstudio
plt.clf() # clear settings
# According to the data we have here, births are slightly less common on weekends
# than on weekdays! Note that the 1990s and 2000s are missing because the CDC
# data contains only the month of birth starting in 1989.

# Another interesting view is to plot the mean number of births by the day of the
# year. Let’s first group the data by month and day separately.
births_by_date = births.pivot_table("births", [births.index.month, births.index.day])
births_by_date.head()
# The result is a multi-index over months and days. To make this easily 
# plottable, let’s turn these months and days into a date by associating them 
# with a dummy year variable (making sure to choose a leap year so February 29th
# is correctly handled!).
import datetime as dt
births_by_date.index =  [dt.datetime(2012, month, day) for (month, day) in births_by_date.index]
births_by_date.head()

# Focusing on the month and day only, we now have a time series reflecting the 
# average number of births by date of the year. From this, we can use the plot 
# method to plot the data. It reveals some interesting trends...
plt.clf()
fig, ax = plt.subplots(figsize = (12, 4))
births_by_date.plot(ax = ax)
plt.show() # render plot for Rstudio
plt.clf() # clear settings

#### Vectorized String Operations
# One strength of Python is its relative ease in handling and manipulating 
# string data. Pandas builds on this and provides a comprehensive set of 
# vectorized string operations that become an essential piece of the type of 
# munging required when one is working with (read: cleaning up) real-world data.

##### Introducing Pandas String Operations





