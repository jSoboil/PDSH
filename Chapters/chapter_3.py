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

