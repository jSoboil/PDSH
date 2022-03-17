## Machine Learning
### Introducing Scikit-Learn
# Scikit-Learn is  package that provides efficient versions of a large number of 
# common algorithms.

#### Scikit-Learn’s Estimator API
##### Basics of the API
# Most commonly, the steps in using the Scikit-Learn estimator API are as follows

# 1. Choose a class of model by importing the appropriate estimator class from 
# Scikit- Learn.

# 2. Choose model hyperparameters by instantiating this class with desired 
# values.

# 3. Arrange data into a features matrix and target vector.

# 4. Fit the model to your data by calling the fit() method of the model 
# instance.

# 5. Apply the model to new data

##### Application: Exploring Handwritten Digits
# To demonstrate these principles, let’s consider one piece of the optical 
# character recognition problem: the identification of handwritten digits. In 
# the wild, this problem involves both locating and identifying characters in an 
# image. Here we’ll take a shortcut and use Scikit-Learn’s set of preformatted 
# digits, which is built into the library.

###### Loading and visualizing the digits data
from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape
# The images data is a three-dimensional array: 1,797 samples, each consisting of
# an 8×8 grid of pixels. Let’s visualize the first hundred of these .
import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize = (8, 8), subplot_kw = {"xticks":[], "yticks":[]}, gridspec_kw = dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat): 
 ax.imshow(digits.images[i], cmap = "binary", interpolation = "nearest")
 ax.text(0.05, 0.05, str(digits.target[i]), transform = ax.transAxes, color = "green");
plt.show()

# To work with this data within Scikit-Learn, we need a two-dimensional, 
# [n_samples, n_features] representation. We can accomplish this by treating 
# each pixel in the image as a feature—that is, by flattening out the pixel 
# arrays so that we have a length-64 array of pixel values representing each 
# digit. Additionally, we need the target array, which gives the previously 
# determined label for each digit. These two quantities are built into the 
# digits dataset under the data and target attributes, respectively.
X = digits.data
X.shape
y = digits.target
y.shape
# We see here that there are 1,797 samples and 64 features.

###### Unsupervised learning: Dimensionality reduction
# We’d like to visualize our points within the 64-dimensional parameter space, 
# but it’s difficult to effectively visualize points in such a high-dimensional 
# space. Instead we’ll reduce the dimensions to 2, using an unsupervised method. 
# Here, we’ll make use of a manifold learning algorithm called Isomap, and 
# transform the data to two dimensions.
from sklearn.manifold import Isomap
iso = Isomap(n_components = 2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape
# We see that the projected data is now two-dimensional. Let’s plot this data to 
# see if we can learn anything from its structure.
plt.clf()
plt.scatter(data_projected[:, 0], data_projected[:, 1], c = digits.target, 
  edgecolor = "none", alpha = 0.5, cmap = plt.get_cmap("nipy_spectral", 10))
plt.colorbar(label = "digit label", ticks = range(10))
plt.clim(-0.5, 9.5);
plt.show()
# Overall, the different groups appear to be fairly well separated in the 
# parameter space: this tells us that even a very straightforward supervised 
# classification algorithm should perform suitably on this data. Let’s give it a 
# try.

###### Classification on digits
# Let’s apply a classification algorithm to the digits. We will split the data 
# into a training and test set, and fit a Gaussian naive Bayes model.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
yhat_model = model.predict(X_test)
# Now that we have predicted our model, we can gauge its accuracy by comparing 
# the true values of the test set to the predictions.
from sklearn.metrics import accuracy_score
accuracy_score(y_test, yhat_model)
# With even this extremely simple model, we find about 80% accuracy for 
# classification of the digits! However, this single number doesn’t tell us 
# where we’ve gone wrong — one nice way to do this is to use the confusion 
# matrix, which we can compute with Scikit-Learn and plot with Seaborn.
from sklearn.metrics import confusion_matrix
c_mat = confusion_matrix(y_test, yhat_model)

import seaborn as sns
plt.clf()
sns.set()
sns.heatmap(c_mat, square = True, annot = True, cbar = False, 
                                                 cmap = "ocean")
plt.xlabel("predicted vale")
plt.ylabel("observed value")
plt.show()
plt.clf()
# This shows us where the mislabeled points tend to be: for example, a large 
# number of twos here are misclassified as either ones or eights.

# Another way to gain intuition into the characteristics of the model is to 
# plot the inputs again, with their predicted labels.
fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                                     subplot_kw={'xticks':[], 'yticks':[]},
                                     gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
 ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
 ax.text(0.05, 0.05, str(yhat_model[i]),
        transform=ax.transAxes,
        color='green' if (y_test[i] == yhat_model[i]) else 'red')
plt.show()

##### Hyperparameters and Model Validation
###### Thinking About Model Validation
# In principle, model validation is very simple: after choosing a model and its
# hyper‐parameters, we can estimate how effective it is by applying it to some 
# of the training data and comparing the prediction to the known value.

######## Model validation the right way - Holdout sets
# We can get a better sense of a model’s performance using what’s known as a 
# holdout set; that is, we hold back some subset of the data from the training 
# of the model, and then use this holdout set to check the model performance. 
# We can do this splitting using the train_test_split utility in Scikit-Learn.
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 1)

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# split the data with 50% in each set
X_1, X_2, y_1, y_2 = train_test_split(X, y, random_state = 0, train_size = 0.5)

# fit the model on one set of data
model.fit(X_1, y_1)

# evaluate the model on the second set of data
y_2_model = model.predict(X_2)
from sklearn.metrics import accuracy_score
accuracy_score(y_2, y_2_model)

##### Model validation via cross-validation
# One disadvantage of using a holdout set for model validation is that we have 
# lost a portion of our data to the model training. One way to address this is 
# to use cross-validation — that is, to do a sequence of fits where each subset
# of the data is used both as a training set and as a validation set.

# Here we do two validation trials, alternately using each half of the data as a
# holdout set. Using the split data from before, we could implement it like 
# this
y_2_model = model.fit(X_1, y_1).predict(X_2)
y_1_model = model.fit(X_2, y_2).predict(X_1)
accuracy_score(y_1, y_1_model), accuracy_score(y_2, y_2_model)
# What comes out are two accuracy scores, which we could combine (by, say, 
# taking the mean) to get a better measure of the global model performance. 
# This particular form of cross-validation is a two-fold cross-validation—one 
# in which we have split the data into two sets and used each in turn as a 
# validation set. We could expand on this idea to use even more trials, and 
# more folds in the data.

# Here we split the data into five groups, and use each of them in turn to 
# evaluate the model fit on the other 4/5 of the data. This would be rather 
# tedious to do by hand, and so we can use Scikit-Learn’s cross_val_score 
# convenience routine to do it succinctly.
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, cv = 5)
# Repeating the validation across different subsets of the data gives us an 
# even better idea of the performance of the algorithm.

# And classic LOOCV...
from sklearn.model_selection import LeaveOneOut
scores = cross_val_score(model, X, y, cv = LeaveOneOut())
scores
# Because we have 150 samples, the leave-one-out cross-validation yields scores 
# for 150 trials, and the score indicates either successful (1.0) or 
# unsuccessful (0.0) prediction. Taking the mean of these gives an estimate of 
# the error rate...
scores.mean()

#### Validation in Practice - Grid Search
# Scikit-Learn provides automated tools to find the par‐ ticular model that 
# maximizes the validation score. Below is an example of using grid search to 
# find the optimal polynomial model. We will explore a three-dimensional grid 
# of model features — namely, the polynomial degree, the flag telling us 
# whether to fit the intercept, and the flag telling us whether to normalise 
# the problem. We can set this up using Scikit-Learn’s GridSearchCV 
# meta-estimator.
import numpy as np
from sklearn.model_selection import GridSearchCV
param_grid = {"polynomialfeatures__degree": np.arange(21),
              "linearregression__fit_intercept": [True, False],
              "linearregression__normalize": [True, False]}

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree = 2, **kwargs): 
 return make_pipeline(PolynomialFeatures(degree), 
LinearRegression(**kwargs))

def make_data(N, err = 1.0, rseed = 1):
 # randomly sample the data
 rng = np.random.RandomState(rseed)
 X = rng.rand(N, 1) ** 2
 y = 10 - 1. / (X.ravel() + 0.1)
 if err > 0:
  y += err * rng.randn(N)
 return X, y

X, y = make_data(40)

grid = GridSearchCV(PolynomialRegression(), param_grid, cv = 7)
grid.fit(X, y)
grid.best_params_

# Finally, if we wish, we can use the best model and show the fit to our data 
# using code from before...
model = grid.best_estimator_

# plot
plt.clf()
plt.scatter(X.ravel(), y)
lim = plt.axis()
X_test = np.linspace(-0.1, 1.1, 500)[:, None]
y_test = model.fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test);
plt.axis(lim);
plt.show()
plt.clf()

#### Feature Engineering
# The previous sections outline the fundamental ideas of machine learning, but 
# all of the examples assume that you have numerical data in a tidy, [n_samples, 
# n_fea tures] format. In the real world, data rarely comes in such a form. 
# With this in mind, one of the more important steps in using machine learning 
# in practice is feature engi‐ neering—that is, taking whatever information you 
# have about your problem and turning it into numbers that you can use to build
# your feature matrix.

##### Categorical Features
# One common type of non-numerical data is categorical data. For example, 
# imagine housing prices data, and along with numerical features like “price” 
# and “rooms,” you also have “neighborhood” information. For example, your data 
# might look something like this...
data = [
 {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
 {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
 {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
 {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
 ]
# ... and so you might be tempted to encode this data with a straightforward 
# numerical mapping, such as
# {'Queen Anne': 1, 'Fremont': 2, 'Wallingford': 3};

# It turns out that this is not generally a useful approach in Scikit-Learn, as 
# the package’s models make the fundamental assumption that numerical features 
# reflect algebraic quantities.

# In this case, one proven technique is to use one-hot (binary) encoding, which 
# effectively creates extra columns indicating the presence or absence of a 
# category with a value of 1 or 0, respectively. When your data comes as a list 
# of dictionaries, Scikit-Learn’s DictVector izer will do this for you
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False, dtype = int)
vec.fit_transform(data)
# The neighborhood column has been expanded into three separate columns, 
# representing the three neighborhood labels, and that each row has a 1 in the 
# column associated with its neighborhood. With these categorical features thus 
# encoded, you can proceed as normal with fitting a Scikit-Learn model.

# To see the meaning of each column, you can inspect the feature names...
vec.get_feature_names_out()
# However, there is a disadvantage of this approach, as it this can greatly 
# increase the size of your dataset. Nevertheless, because the encoded data 
# contains mostly zeros, a sparse output can be a very efficient solution...
vec = DictVectorizer(sparse = True, dtype = int)
vec.fit_transform(data) 

##### Text Features
# Another common need in feature engineering is to convert text to a set of 
# representa‐ tive numerical values. For example, most automatic mining of 
# social media data relies on some form of encoding the text as numbers. One of
# the simplest methods of encoding data is by word counts - you take each 
# snippet of text, count the occurrences of each word within it, and put the 
# results in a table.

# For example, consider the following set of three phrases
sample = ["problem of evil",
          "evil queen",
          "horizon problem"]
# For a vectorization of this data based on word count, we could construct a 
# column representing the word “problem,” the word “evil,” the word “horizon,” 
# and so on. While doing this by hand would be possible, we can avoid the 
# tedium by using Scikit-Learn’s CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(sample)
X
# The result is a sparse matrix recording the number of times each word appears;
# it is easier to inspect if we convert this to a DataFrame with labeled 
# columns.
import pandas as pd
pd.DataFrame(X.toarray(), columns = vec.get_feature_names_out())
# However, the raw word counts lead to features that put too much weight on 
# words that appear very frequently, and this can be suboptimal in some 
# classification algorithms. One approach to fix this is known as term 
# frequency–inverse document frequency (TF–IDF), which weights the word counts 
# by a measure of how often they appear in the documents. The syntax for 
# computing these features is similar to the previous example
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns = vec.get_feature_names_out())

##### Image Features
# Another common need is to suitably encode images for machine learning 
# analysis.

##### Derived Features
# Another useful type of feature is one that is mathematically derived from 
# some input features. For example, one could convert a linear regression into 
# a polynomial regression not by changing the model, but by transforming the 
# input! This is sometimes known as basis function regression.

# For example, this data clearly cannot be well described by a straight line
x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.clf()
plt.scatter(x, y);
plt.show()
plt.clf()

from sklearn.linear_model import LinearRegression
X = x[:, np.newaxis]
model = LinearRegression().fit(X = X, y = y)
y_hat = model.predict(X)
plt.scatter(x, y)
plt.plot(X, y_hat)
plt.show()

# It’s clear that we need a more sophisticated model to describe the 
# relationship between x and y. We can do this by transforming the data, adding
# extra columns of features to drive more flexibility in the model. For example,
# we can add polynomial features to the data this way
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 3, include_bias = False)
X_2 = poly.fit_transform(X)
print(X_2)
# The derived feature matrix has one column representing x, and a second column 
# rep‐ resenting x^2, and a third column representing x^3. Computing a linear 
# regression on this expanded input gives a much closer fit to our data
model = LinearRegression().fit(X_2, y)
y_hat = model.predict(X_2)

plt.clf()
plt.scatter(x, y)
plt.plot(x, y_hat)
plt.show()
plt.clf()

# Although the boomk emphasises the need for preprocessing of inputs, this is
# very tricky as it leads to a change in interpretation of the results, 
# sometimes making it nonsensical for real-world application...

##### Imputation of Missing Data
# Another common need in feature engineering is handling missing data. For 
# example, we might have a dataset that looks like this...
from numpy import nan
X = np.array([[nan,0, 3],
              [3, 7, 9], 
              [3, 5, 2], 
              [4, nan,6], 
              [8, 8, 1]])
y = np.array([14, 16, -1,  8, -5])
# For a baseline imputation approach, using the mean, median, or most frequent 
# value, Scikit-Learn provides the Imputer class.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "mean")
X_2 = imputer.fit_transform(X)
X_2

##### Feature Pipelines
# This is quite cool... with any of the preceding examples, it can quickly 
# become tedious to do the transformations by hand, especially if you wish to 
# string together multiple steps. For example, we might want a processing 
# pipeline that looks something like this

# 1. Impute missing values using the mean
# 2. Transform features to quadratic
# 3. Fit a linear regression

# To streamline this type of processing pipeline, Scikit-Learn provides a 
# pipeline object, which can be used as follows.
from sklearn.pipeline import make_pipeline
model = make_pipeline(SimpleImputer(strategy = "mean"),
                      PolynomialFeatures(degree = 2),
                      LinearRegression())
# This pipeline looks and acts like a standard Scikit-Learn object, and will 
# apply all the specified steps to any input data.
model.fit(X, y)
print(y)
print(model.predict(X))
# All the steps of the model are applied automatically.

### In Depth: Naive Bayes Classification
# Naive Bayes models are a group of extremely fast and simple classification 
# algorithms that are often suitable for very high-dimensional datasets. 
# Because they are so fast and have so few tunable parameters, they end up 
# being very useful as a quick-and-dirty baseline for a classification problem.

#### Bayesian Classification
# Naive Bayes classifiers are built on Bayesian classification methods. These 
# rely on Bayes’s theorem

# P(L \mid features) = \frac{P(features \mid L)P(L)}{P(features)}

# If we are trying to decide between two labels — let’s call them L1 and L2 — 
# then one way to make this decision is to compute the ratio of the posterior 
# probabilities for each label. All we need now is some model by which we can 
# compute P features Li for each label. Such a model is called a generative 
# model.

# Specifying this generative model for each label is the main piece of the 
# training of such a Bayesian classifier. The general version of such a 
# training step is a very difficult task, but we can make it simpler through 
# the use of some simplifying assumptions about the form of this model.

# This is where the “naive” in “naive Bayes” comes in - if we make very naive 
# assumptions about the generative model for each label, we can find a rough 
# approximation of the generative model for each class, and then proceed with 
# the Bayesian classification. Different types of naive Bayes classifiers rest 
# on different naive assumptions about the data. We begin with the standard 
# imports.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#### Gaussian Naive Bayes
# Perhaps the easiest naive Bayes classifier to understand is Gaussian naive 
# Bayes.
from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers = 2, random_state = 2, cluster_std = 1.5)
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = "RdBu")
plt.show()
plt.clf()

# The procedure is implemented in Scikit-Learn’s sklearn.naive_bayes.GaussianNB
# estimator.
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
# Now let’s generate some new data and predict the label...
rng = np.random.RandomState(0)
X_new = [-6, -14] + [14, 18] * rng.rand(2000, 2)
y_new = model.predict(X_new)

# Now we can plot this new data to get an idea of where the decision boundary is
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = "RdBu")
lim = plt.axis()
plt.scatter(X_new[:, 0], X_new[:, 1], c = y_new, s = 20, cmap = "RdBu", 
                                                            alpha = 0.1)
plt.axis(lim);
plt.show()
plt.clf()
# We see a slightly curved boundary in the classifications — in general, the 
# boundary in Gaussian naive Bayes is quadratic.

# A nice piece of this Bayesian formalism is that it naturally allows for 
# probabilistic classification, which we can compute using the predict_proba 
# method.
y_prob = model.predict_proba(X_new)
y_prob[-8: ].round(2)
# The columns give the posterior probabilities of the first and second label, 
# respectively.

#### Multinomial Naive Bayes
# Another useful example is multinomial naive Bayes, where the features are 
# assumed to be generated from a simple multinomial distribution.

##### Example: Classifying text
# Let’s download the data and take a look at the target names.
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
data.target_names
# For simplicity, we will select just a few of these categories, and download 
# the training and testing set.
categories = ["talk.religion.misc", "soc.religion.christian", "sci.space",
              "comp.graphics"]
train_data = fetch_20newsgroups(subset = "train", categories = categories)
test_data = fetch_20newsgroups(subset = "test", categories = categories)
# Here is a representative entry from the data...
print(train_data.data[5])

# In order to use this data for machine learning, we need to be able to convert 
# the content of each string into a vector of numbers. For this we will use the
# TF–IDF vectoriser, , and create a pipeline that attaches it to a multinomial 
# naive Bayes classifier.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# With this pipeline, we can apply the model to the training data, and predict 
# labels for the test data.
model.fit(train_data.data, train_data.target)
labels = model.predict(test_data.data)
# Now that we have predicted the labels for the test data, we can evaluate them 
# to learn about the performance of the estimator. For example, here is the 
# confusion matrix between the true and predicted labels for the test data.
from sklearn.metrics import confusion_matrix
plt.clf()
c_mat = confusion_matrix(test_data.target, labels)
sns.heatmap(c_mat.T, square = True, annot = True, fmt = "d", cbar = False, 
            xticklabels = train_data.target_names, 
            yticklabels = train_data.target_names)
            
plt.xlabel("true label")
plt.ylabel("predicted label");
plt.show()
plt.clf()
# Evidently, even this very simple classifier can successfully separate space 
# talk from computer talk, but it gets confused between talk about religion and 
# talk about Christianity.

# The very cool thing here is that we now have the tools to determine the 
# category for any string, using the predict() method of this pipeline. Here’s a
# quick utility function that will return the prediction for a single string.
def predict_category(s, train = train_data, model = model):
 pred = model.predict([s])
 return train_data.target_names[pred[0]]

# Let’s try it out.
predict_category("sending a payload to the ISS")
predict_category("discussing islam vs atheism")
predict_category("determining the screen resolution")

### In Depth: Linear Regression
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

#### Simple Linear Regression
# A straight-line fit is a model of the form y = ax + b where a is commonly 
# known as the slope, and b is commonly known as the intercept.

# Consider the following data, which is scattered about a line with a slope of 
# 2 and an intercept of –5.
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.clf()
plt.scatter(x, y);
plt.show()
plt.clf()

# We can use Scikit-Learn’s LinearRegression estimator to fit this data and 
# construct the best-fit line.
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept = True)

model.fit(x[:, np.newaxis], y)

x_fit = np.linspace(0, 10, 1000)
y_fit = model.predict(x_fit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(x_fit, y_fit)
plt.show()
plt.clf()

# The slope and intercept of the data are contained in the model’s fit 
# parameters, which in Scikit-Learn are always marked by a trailing underscore. 
# Here the relevant parameters are coef_ and intercept_.
print("Model slope: ", model.coef_[0])
print("Model intercept:", model.intercept_)

#### Basis Function Regression
# The idea is to take our multidimensional linear model

# y = a_0 + a_1 * x_1 + a_2 * x_2 + a_3 * x_3 ...

# and build the x_1, x_2, x_3, and so on from our single-dimensional input x. 
# That is, we let x_n = f n x , where f_n() is some function that transforms 
# our data.

# For example, if f_n(x) = x^n, the model becomes a polynomial regression.

# y = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 ...

###### Polynomial basis functions
# This polynomial projection is useful enough that it is built into Scikit-Learn,
# using the PolynomialFeatures transformer.
from sklearn.preprocessing import PolynomialFeatures
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias = False)
poly.fit_transform(x[:, None])
# We see here that the transformer has converted our one-dimensional array into 
# a three-dimensional array by taking the exponent of each value. This new, 
# higher- dimensional data representation can then be plugged into a linear 
# regression.

# Let’s make a 7th-degree polynomial model in this way.
from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(7), 
                           LinearRegression())
# With this transform in place, we can use the linear model to fit much more 
# complicated relationships between x and y. For example, here is a sine wave 
# with noise.
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.rand(50)

poly_model.fit(x[:, np.newaxis], y)
# there's some code missing here for some reason...
y_hat = poly_model.predict(x_fit[:, np.newaxis])

plt.clf()
plt.scatter(x, y)
plt.plot(x_fit, y_fit)
plt.show()

##### Gaussian basis functions
# Tne useful pattern is to fit a model that is not a sum of polynomial bases, 
# but a sum of Gaussian bases. These Gaussian basis functions are not built into
# Scikit-Learn, but we can write a custom transformer that will create them.
from sklearn.base import BaseEstimator, TransformerMixin

class GaussianFeatures(BaseEstimator, TransformerMixin):
 """Uniformly spaced Gaussian features for one-dimension output"""
 
 def __init__(self, N, width_factor = 2.0):
  self.N = N
  self.width_factor = width_factor
  
 @staticmethod
 def _gauss_basis(x, y, width, axis = None):
  arg = (x - y) / width
  return np.exp(-0.5 * np.sum(arg ** 2, axis))
 
 def fit(self, X, y = None):
  # create N centres spread along the data range
  self.centers_ = np.linspace(X.min(), X.max(), self.N)
  self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
  return self
 
 def transform(self, X):
  return self._gauss_basis(X[:, :, np.newaxis], self.centers_, 
                           self.width_, axis = 1)

# Use function for new model...
gauss_model = make_pipeline(GaussianFeatures(20),
                            LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
y_hat = gauss_model.predict(x_fit[:, np.newaxis])
# Plot...
plt.clf()
plt.scatter(x, y)
plt.plot(x_fit, y_fit)
plt.xlim(0, 10);
plt.show()
plt.clf()

#### Regularization
##### Ridge regression (L2 regularization)
# Perhaps the most common form of regularization is known as ridge regression or
# L2 regularization, sometimes also called Tikhonov regularization. This 
# proceeds by penalising the sum of squares (2-norms) of the model coefficients;
# in this case, the penalty on the model fit would be

# P = \alpha \sum_{n = 1}^{N} \theta_{n}^2

# where \alpha is a free parameter that controls the strength of the penalty. 
# This type of penalized model is built into Scikit-Learn with the Ridge 
# estimator.
def basis_plot(model, title = None):
 fig, ax = plt.subplots(2, sharex = True)
 model.fit(x[:, np.newaxis], y)
 ax[0].scatter(x, y)
 ax[0].plot(x_fit, model.predict(x_fit[:, np.newaxis]))
 ax[0].set(xlabel = 'x', ylabel = 'y', ylim = (-1.5, 1.5))
 
 if title:
  ax[0].set_title(title)
  ax[1].plot(model.steps[0][1].centers_,
             model.steps[1][1].coef_)
  ax[1].set(xlabel = 'basis location',
            ylabel = 'coefficient',
            xlim = (0, 10))
 
from sklearn.linear_model import Ridge
model = make_pipeline(GaussianFeatures(30), Ridge(alpha = 0.1))
basis_plot(model, title = "Ridge Regression")
plt.show()
plt.clf()

##### Lasso regularization (L1)
# Another very common type of regularization is known as lasso, and involves 
# penalising the sum of absolute values (1-norms) of regression coefficients.

# P = \alpha \sum_{n = 1}^{N} | \theta_{n}^2 |

# Though this is conceptually very similar to ridge regression, the results can 
# differ surprisingly: for example, due to geometric reasons lasso regression 
# tends to favor sparse models where possible.
from sklearn.linear_model import Lasso
model = make_pipeline(GaussianFeatures(30), Lasso(alpha = 0.001))
basis_plot(model, title = "Lasso Regression")
plt.show();
plt.clf()
# As with ridge regularization, the \alpha parameter tunes the strength of the 
# penalty, and should be determined via, for example, cross-validation.

###### Example: Predicting Bicycle Traffic
# Let’s take a look at whether we can predict the number of bicycle trips across
# Seattle’s Fremont Bridge based on weather, season, and other factors. We will 
# join the bike data with another dataset, and try to determine the extent to 
# which weather and seasonal factors — temperature, precipitation, and daylight 
# hours — affect the volume of bicycle traffic through this corridor.

# Let’s start by loading the two datasets, indexing by date.
import pandas as pd
counts = pd.read_csv("data/Fremont_Bridge_Bicycle_Counter.csv", 
                     index_col = "Date", parse_dates = True)
weather = pd.read_csv("data/BicycleWeather.csv", index_col = "DATE", 
                      parse_dates = True)
# Next we will compute the total daily bicycle traffic, and put this in its own 
# DataFrame...
daily = counts.resample("d").agg("sum")
daily["Total"] = daily.sum(axis = 1)
daily = daily[["Total"]] # remove other columns
# We saw previously that the patterns of use generally vary from day to day; 
# let’s account for this in our data by adding binary columns that indicate the 
# day of the week...
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for i in range(7):
 daily[days[i]] = (daily.index.day_of_week == i).astype(float)

# Similarly, we might expect riders to behave differently on holidays; let’s add
# an indicator of this too.
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays("2012", "2016")
daily = daily.join(pd.Series(1, index = holidays, name = "holiday"))
daily["holiday"].fillna(0, inplace = True)

# We also might suspect that the hours of daylight would affect how many people 
# ride; let’s use the standard astronomical calculation to add this information.
import datetime as dt
def hours_of_daylight(date, axis=23.44, latitude=47.61):
 """Compute the hours of daylight for the given date"""
 days = (date - pd.datetime(2000, 12, 21)).days
 m = (1. - np.tan(np.radians(latitude)) 
      * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
 return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

daily["daylight_hrs"] = list(map(hours_of_daylight, daily.index))
plt.clf()
daily[['daylight_hrs']].plot();
plt.show()
plt.clf()

# We can also add the average temperature and total precipitation to the data. 
# In addition to the inches of precipitation, let’s add a flag that indicates 
# whether a day is dry (has zero precipitation).

# temperatures are in 1/10 deg C; convert to C
weather["TMIN"] /= 10
weather["TMAX"] /= 10
weather['Temp (C)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])

# precip is in 1/10 mm; convert to inches
weather["PRCP"] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(float)

daily = daily.join(weather[["PRCP", "Temp (C)", "dry day"]])

# Finally, let’s add a counter that increases from day 1, and measures how many 
# years have passed. This will let us measure any observed annual increase or 
# decrease in daily crossings.
daily["annual"] = (daily.index - daily.index[0]).days / 365.

# Now our data is in order, and we can take a look at it.
daily.head()
daily = daily.fillna(0)
# With this in place, we can choose the columns to use, and fit a linear 
# regression model to our data. We will set fit_intercept = False, because the 
# daily flags essentially operate as their own day-specific intercepts.
column_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", 
                "holiday", "daylight_hrs", "PRCP", "dry day", "Temp (C)", 
                "annual"]
X = daily[column_names]
y = daily["Total"]

model = LinearRegression(fit_intercept = False)
model.fit(X, y)
daily["predicted"] = model.predict(X)

# Finally, we can compare the total and predicted bicycle traffic visually...
plt.clf()
daily[["Total", "predicted"]].plot(alpha = 0.5);
plt.show()
plt.clf()

# It is evident that we have missed some key features, especially during the 
# summer time. Either our features are not complete (i.e., people decide 
# whether to ride to work based on more than just these) or there are some 
# nonlinear relationships that we have failed to take into account (e.g., 
# perhaps people ride less at both high and low temperatures). Nevertheless, 
# our rough approximation is enough to give us some insights, and we can take a 
# look at the coefficients of the linear model to estimate how much each 
# feature contributes to the daily bicycle count...
params = pd.Series(model.coef_, index = X.columns)
params

# These numbers are difficult to interpret without some measure of their 
# uncertainty. We can compute these uncertainties quickly using bootstrap 
# resamplings of the data.
from sklearn.utils import resample
np.random.seed(1)
err = np.std([model.fit(*resample(X, y)).coef_
              for i in range(1000)], 0)
# With these errors estimated, let’s again look at the results.
print(pd.DataFrame({"effect": params.round(0),
                    "error": err.round(0)}))

# We first see that there is a relatively stable trend in the weekly baseline - 
# there are many more riders on weekdays than on weekends and holidays. We see 
# that for each additional hour of daylight, 1007 ± 28 more people choose to 
# ride; a temperature increase of one degree Celsius discourages 47 ± 12 people;
# a dry day means an average of 352 ± 194 more riders; and each inch of 
# precipitation means 5776 ± 459 more people leave their bike at home.

### In-Depth: Support Vector Machines
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# use seaborn plotting defaults
import seaborn as sns; sns.set()

#### Motivating Support Vector Machines
# Here we will consider instead discriminative classification: rather than 
# modeling each class, we simply find a line or curve (in two dimensions) or 
# manifold (in multiple dimensions) that divides the classes from each other.
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples = 50, centers = 2, 
                  random_state = 0, cluster_std = 0.60)
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = "autumn")
plt.show()
plt.clf()
# A linear discriminative classifier would attempt to draw a straight line 
# separating the two sets of data, and thereby create a model for 
# classification. For two-dimensional data like that shown here, this is a task 
# we could do by hand. But immediately we see a problem - there is more than 
# one possible dividing line that can perfectly discriminate between the two 
# classes!
x_fit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = "autumn")
plt.plot([0.6], [2.1], "x", color = "red", 
         markeredgewidth = 2, markersize = 10)
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
 plt.plot(x_fit, m * x_fit + b, "-k")
plt.xlim(-1, 3.5);
plt.show()
plt.clf()
# Evidently the simple intuition of “drawing a line between classes” is not 
# enough, and we need to think a bit deeper.

##### Support Vector Machines: Maximizing the Margin
# Support vector machines offer one way to improve on this. Rather than simply
# drawing a zero-width line between the classes, we can draw around each line 
# a margin of some width, up to the nearest point. Here is an example of how 
# this might look...
x_fit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = "autumn")

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
 y_hat = m * x_fit + b
 plt.plot(x_fit, y_hat, "-k")
 plt.fill_between(x_fit, y_hat - d, y_hat + d, edgecolor = "none", 
                  color = "#AAAAAA", alpha = 0.4)
plt.xlim(-1, 3.5);
plt.show()
plt.clf()
# In support vector machines, the line that maximizes this margin is the one 
# we will choose as the optimal model. Hence, SVMs are an example of such a 
# maximum margin estimator.

##### Fitting a support vector machine
# For this example, we will use a linear kernel.
from sklearn.svm import SVC
model = SVC(kernel = "linear", C = 1E10)
model.fit(X, y)
# To better visualize what’s happening, let’s create a convenience function 
# that will plot SVM decision boundaries for us...
def plot_svc_decision_function(model, ax = None, plot_support = True):
 """Plot the decision function for a two-dimensional SVC"""
 if ax is None:
     ax = plt.gca()
 # else
 xlim = ax.get_xlim()
 ylim = ax.get_ylim()
  
 # create grid to evaluate model
 x = np.linspace(xlim[0], xlim[1], 30)
 y = np.linspace(ylim[0], ylim[1], 30)
 Y, X = np.meshgrid(y, x)
 xy = np.vstack([X.ravel(), Y.ravel()]).T
 P = model.decision_function(xy).reshape(X.shape)
 
 # plot decision boundary and margins
 ax.contour(X, Y, P, colors='k',
            levels = [-1, 0, 1], alpha = 0.5, 
            linestyles=['--', '-', '--'])
             
 # plot support vectors
 if plot_support: 
     ax.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s = 300, linewidth = 1, facecolors = 'none');
 ax.set_xlim(xlim)
 ax.set_ylim(ylim)

plt.clf()
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = "autumn")
plot_svc_decision_function(model)
plt.show()
# This is the dividing line that maximizes the margin between the two sets of 
# points. These points are the pivotal elements of this fit, and are known as 
# the support vectors, and give the algorithm its name.

# In Scikit-Learn, the identity of these points is stored in the 
# support_vectors_ attribute of the classifier...
model.support_vectors_

# A key to this classifier’s success is that for the fit, only the position of 
# the support vectors matter; any points further from the margin that are on 
# the correct side do not modify the fit.

# We can see this, for example, if we plot the model learned from the first 60
# points and first 120 points of this dataset.
plt.clf()

def plot_svm(N = 10, ax = None):
 X, y = make_blobs(n_samples = 200, centers = 2,
                   random_state = 0, cluster_std = 0.60)
 X = X[:N]
 y = y[:N]
 model = SVC(kernel = 'linear', C = 1E10)
 model.fit(X, y)
 
 ax = ax or plt.gca()
 ax.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'autumn')
 ax.set_xlim(-1, 4)
 ax.set_ylim(-1, 6)
 plot_svc_decision_function(model, ax)


fig, ax = plt.subplots(1, 2, figsize = (16, 6))
fig.subplots_adjust(left = 0.0625, right = 0.95, wspace = 0.1)

fig, ax = plt.subplots(1, 2, figsize = (16, 6))
fig.subplots_adjust(left = 0.0625, right = 0.95, wspace = 0.1) 
for axi, N in zip(ax, [60, 120]):
     plot_svm(N, axi)
     axi.set_title('N = {0}'.format(N))
plt.show()
plt.clf()

# You can use IPython’s interactive widgets to view this feature of the SVM 
# model interactively...
from ipywidgets import interact, fixed
interact(plot_svm, N = [10, 200], ax = fixed(None))
# Note: this only works with Jupyter Widgets...

##### Beyond linear boundaries: Kernel SVM
# Where SVM becomes extremely powerful is when it is combined with kernels. 
#To motivate the need for kernels, let’s look at some data that is not 
# linearly separable.
from sklearn.datasets import make_circles
X, y = make_circles(100, factor = 0.1, noise = 0.1)
clf = SVC(kernel = "linear").fit(X, y)

plt.clf()
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = "autumn")
plot_svc_decision_function(clf, plot_support = False)
plt.show()
# It is clear that no linear discrimination will ever be able to separate 
# this data.

# As a solution, we might project the data into a higher dimension such that 
# a linear separator would be sufficient. For example, one simple projection 
# we could use would be to compute a radial basis function centered on the 
# middle clump.
r = np.exp(-(X ** 2).sum(1))

# We can visualize this extra data dimension using a three-dimensional plot.
from mpl_toolkits import mplot3d

def plot_3D(elev = 30, azim = 30, X = X, y = y):
 ax = plt.subplot(projection = '3d')
 ax.scatter3D(X[:, 0], X[:, 1], r, c = y, s = 50, cmap='autumn')
 ax.view_init(elev = elev, azim = azim)
 ax.set_xlabel('x')
 ax.set_ylabel('y')
 ax.set_zlabel('r')

plot_3D()
plt.show()
plt.clf()
# We can see that with this additional dimension, the data becomes trivially 
# linearly separable, by drawing a separating plane at, say, r=0.7.

# In general, however, the need to make such a choice is a problem - we would 
# like to somehow automatically find the best basis functions to use. In 
# Scikit-Learn, we can apply kernelized SVM simply by changing our linear 
# kernel to an RBF (radial basis function) kernel, using the kernel model 
# hyperparameter.
clf = SVC(kernel = "rbf", C = 1E6)
clf.fit(X, y)

# plot...
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = "autumn")
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], 
            s = 300, lw = 1, facecolors = "none");
plt.show()
plt.clf()
# Using this kernelized support vector machine, we learn a suitable nonlinear 
# decision boundary.

##### Tuning the SVM: Softening margins
# But what if your data has some amount of overlap? For example, you may have 
# data like this...
X, y = make_blobs(n_samples = 100, centers = 2, random_state = 0, cluster_std = 1.2)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = "autumn")
plt.show()
plt.clf()

# To handle this case, the SVM implementation has a bit of a fudge-factor that 
# “softens” the margin; that is, it allows some of the points to creep into the 
# margin if that allows a better fit. The hardness of the margin is controlled by
# a tuning parameter, most often known as C. For very large C, the margin is 
# hard, and points cannot lie in it. For smaller C, the margin is softer, and 
# can grow to encompass some points.

# Below gives a visual picture of how a changing C parameter affects the final 
# fit, via the softening of the margin...
X, y = make_blobs(n_samples = 100, centers = 2, random_state = 0, 
                  cluster_std = 0.8)
fig, ax = plt.subplots(1, 2, figsize = (16, 6))
fig.subplots_adjust(left = 0.0625, right = 0.95, wspace = 0.1)

for axi, C in zip(ax, [10.0, 0.1]):
 model = SVC(kernel = "linear", C = C).fit(X, y)
 axi.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = "autumn")
 plot_svc_decision_function(model, axi)
 axi.scatter(model.support_vectors_[:, 0],
             model.support_vectors_[:, 1], 
             s = 300, lw = 1, facecolors = "none");
 axi.set_title("C = {0:0.1f}".format(C), size = 14)
plt.show()
plt.clf()
# Remember, the optimal value of the C parameter will depend on your dataset, and
# should be tuned via cross-validation or a similar procedure.

###### Example: Face Recognition
# As an example of support vector machines in action, let’s take a look at the 
# facial recognition problem. We will use the Labeled Faces in the Wild dataset, 
# which consists of several thousand collated photos of various public figures.
# A fetcher for the dataset is built into Scikit-Learn.
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person = 60)
print(faces.target_names)
print(faces.images.shape)

fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
 axi.imshow(faces.images[i], cmap = "bone")
 axi.set(xticks = [], yticks = [], 
         xlabel = faces.target_names[faces.target[i]])
plt.show()
plt.clf()

# We could proceed by simply using each pixel value as a feature, but often it 
# is more effective to use some sort of preprocessor to extract more meaningful 
# features; here we will use a principal component analysis to extract 150 
# fundamental components to feed into our support vector machine classifier.

# We can do this most straightforwardly by packaging the preprocessor and the 
# classifier into a single pipeline.
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

pca = PCA(whiten = True, svd_solver = "randomized", random_state = 42)
svc = SVC(kernel = "rbf")
model = make_pipeline(pca, svc)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target,
                                                    random_state = 42)
# Finally, we can use a grid search cross-validation to explore combinations of 
# parameters. Here we will adjust C (which controls the margin hardness) and 
# gamma (which controls the size of the radial basis function kernel), and 
# determine the best model...
from sklearn.model_selection import GridSearchCV
param_grid = {"svc__C": [1, 5, 10, 50],
              "svc__gamma": [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
grid.fit(X_train, y_train)
print(grid.best_params_)
# Now with this cross-validated model, we can predict the labels for the test 
# data, which the model has not yet seen.
model = grid.best_estimator_
y_hat = model.predict(Xtest)

## In-Depth: Decision Trees and Random Forests
# Random forests are an example of an ensemble method, a method that relies on 
# aggregating the results of an ensemble of simpler estimators.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#### Creating a decision tree
# Consider the following two-dimensional data, which has one of four class labels
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples = 300, centers = 4,
                  random_state = 0, cluster_std = 1.0)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = "rainbow")
plt.show()

# This process of fitting a decision tree to our data can be done in Scikit-Learn
# with the DecisionTreeClassifier estimator.
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X, y)
# Let’s write a quick utility function to help us visualize the output of the 
# classifier
def visualize_classifier(model, X, y, ax=None, cmap='rainbow'): 
 ax = ax or plt.gca()
 
 # Plot the training points
 ax.scatter(X[:, 0], X[:, 1], c = y, s = 30, cmap = cmap,
            clim = (y.min(), y.max()), zorder = 3)
 ax.axis('tight')
 ax.axis('off')
 xlim = ax.get_xlim()
 ylim = ax.get_ylim()
 # fit the estimator
 model.fit(X, y)
 xx, yy = np.meshgrid(np.linspace(*xlim, num = 200),
                      np.linspace(*ylim, num = 200))
 Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
 # Create a color plot with the results
 n_classes = len(np.unique(y))
 contours = ax.contourf(xx, yy, Z, alpha = 0.3,
                        levels = np.arange(n_classes + 1) - 0.5,
                        cmap = cmap, clim = (y.min(), y.max()),
                        zorder = 1)
 ax.set(xlim = xlim, ylim = ylim);

# Now we can examine what the decision tree classification looks like...
visualize_classifier(DecisionTreeClassifier(), X, y)
plt.show()

##### Decision trees and overfitting
# it is very easy to go too deep in the tree, and thus to fit details of the 
# particular data rather than the overall properties of the distributions they 
# are drawn from.

##### Ensembles of Estimators: Random Forest
# Multiple overfitting estimators can be combined to reduce the effect of this 
# overfitting. This is called called bagging. We can do this type of bagging 
# classification manually using Scikit-Learn’s Bagging Classifier meta-estimator.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier()
bag = BaggingClassifier(tree, n_estimators = 100, max_samples = 0.8, 
                        random_state = 1)

bag.fit(X, y)
visualize_classifier(bag, X, y)
plt.show()
plt.clf()

# In Scikit-Learn, such an optimized ensemble of randomized decision trees is 
# implemented in the RandomForestClassifier estimator, which takes care of all 
# the randomization automatically. All you need to do is select a number of 
# estimators, and it will very quickly (in parallel, if desired) fit the 
# ensemble of trees.
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100, random_state = 0)
visualize_classifier(model, X, y);
plt.show()
plt.clf()

#### Random Forest Regression
# Random forests can also be made to work in the case of regression (that is, 
# continuous rather than categorical variables).

# Consider the following data, drawn from the combination of a fast and slow 
# oscillation.
rng = np.random.RandomState(42)
x = 10 * rng.rand(200)

def model(x, sigma = 0.3):
 fast_oscillation = np.sin(5 * x)
 slow_oscillation = np.sin(0.5 * x)
 error = sigma * rng.randn(len(x))
 
 return slow_oscillation + fast_oscillation + error

y = model(x)
plt.errorbar(x, y, 0.3, fmt = "o")
plt.show()
plt.clf()

# Using the random forest regressor, we can find the best-fit curve as follows
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(200)
forest.fit(x[:, None], y)

x_fit = np.linspace(0, 10, 1000)
y_hat = forest.predict(x_fit[:, None])
y_true = model(x_fit, sigma = 0)

plt.errorbar(x, y, 0.3, fmt = "o", alpha = 0.5)
plt.plot(x_fit, y_fit, "-r")
plt.plot(x_fit, y_true, "-k", alpha = 0.5)
plt.show()
plt.clf()

###### Example: Random Forest for Classifying Digits
from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()

# Plot data...
fig = plt.figure(figsize = (6, 6)) 
# figure size in inches
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05,
                    wspace = 0.05)
# plot the digits: each image is 8x8 pixels
for i in range(64):
 ax = fig.add_subplot(8, 8, i + 1, xticks = [], yticks = [])
 ax.imshow(digits.images[i], cmap = plt.cm.binary, interpolation = 'nearest')
 # label the image with the target value
 ax.text(0, 7, str(digits.target[i]))
plt.show()
plt.clf()

# We can quickly classify the digits using a random forest as follows
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, 
                                                    random_state = 0)
model = RandomForestClassifier(n_estimators = 1000)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# We can take a look at the classification report for this classifier
from sklearn import metrics
print(metrics.classification_report(y_hat, y_test))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_hat)
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()
plt.clf()

### In Depth: Principal Component Analysis
# PCA is fundamentally a dimensionality reduction algorithm.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#### Introducing Principal Component Analysis
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis("equal");
plt.show()
plt.clf()

# In principal component analysis, one quantifies variable relationships by 
# finding a list of the principal axes in the data, and using those axes to 
# describe the dataset. Using Scikit-Learn’s PCA estimator, we can compute this 
# as follows.
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(X)
# The fit learns some quantities from the data, most importantly the “components”
# and “explained variance”...
print(pca.components_)
print(pca.explained_variance_)

# To see what these numbers mean, let’s visualize them as vectors over the input
# data, using the “components” to define the direction of the vector, and the 
# “explained variance” to define the squared-length of the vector...
def draw_vector(v_0, v_1, ax = None):
 ax = ax or plt.gca()
 arrowprops = dict(arrowstyle = "->",
                   linewidth = 2,
                   shrinkA = 0, shrinkB = 0)
 ax.annotate("", v_1, v_0, arrowprops = arrowprops)

# plot data
plt.scatter(X[:, 0], X[:, 1], alpha = 0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
 v = vector * 3 * np.sqrt(length)
 draw_vector(pca.mean_, pca.mean_ + v)
plt.axis("equal");
plt.show()
plt.clf()
# These vectors represent the principal axes of the data, and the length shown is
# an indication of how “important” that axis is in describing the distribution 
# of the data — more precisely, it is a measure of the variance of the data when 
# projected onto that axis. The projection of each data point onto the principal
# axes are the “principal components” of the data.

# This transformation from data axes to principal axes is as an affine 
# transformation, which basically means it is composed of a translation, 
# rotation, and uniform scaling.

##### PCA as dimensionality reduction
# Using PCA for dimensionality reduction involves zeroing out one or more of the 
# smallest principal components, resulting in a lower-dimensional projection of 
# the data that preserves the maximal data variance.

# Here is an example of using PCA as a dimensionality reduction transform.
pca = PCA(n_components = 1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape: ", X.shape)
print("original shape: ", X_pca.shape)

# The transformed data has been reduced to a single dimension. To understand the 
# effect of this dimensionality reduction, we can perform the inverse transform 
# of this reduced data and plot it along with the original data...
X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha = 0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha = 0.8)
plt.axis('equal');
plt.show()
plt.clf()

# The "information"" along the least important principal axis or axes is removed,
# leaving only the component(s) of the data with the highest variance.

# But the book doesn't add that this has no deeper understanding, i.e. causality!
# Hence, PCA can be dangerous if not done carefully, especially in medical field.

#### In-Depth: Manifold Learning
# While PCA is flexible, fast, and easily interpretable, it does not perform so 
# well when there are nonlinear relationships within the data. To address this 
# deficiency, we can turn to a class of methods known as manifold learning — a 
# class of unsupervised estimators that seeks to describe datasets as 
# low-dimensional manifolds embedded in high-dimensional spaces.
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() 
import numpy as np

##### Manifold Learning: “HELLO”
# To make these concepts more clear, let’s start by generating some 
# two-dimensional data that we can use to define a manifold. Here is a function 
# that will create data in the shape of the word “HELLO”.
def make_hello(N = 1000, rseed = 42):
 # Make a plot with "HELLO" text; save as png
 fig, ax = plt.subplots(figsize = (4, 1))
 fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
 ax.axis("off")
 ax.text(0.5, 0.4,  "HELLO", va = "center", ha = "center", weight = "bold", 
         size = 85)
 fig.savefig("img/hello.png")
 plt.close(fig)
 
 # Open PNG and draw random points on it...
 from matplotlib.image import imread
 data = imread("img/hello.png")[::-1, :, 0].T
 rng = np.random.RandomState(rseed)
 X = rng.rand(4 * N, 2)
 i, j = (X * data.shape).astype(int).T
 mask = (data[i, j] < 1)
 X = X[mask]
 X[:, 0] *= (data.shape[0] / data.shape[1])
 X = X[:N]
 return X[np.argsort(X[:, 0])]

# Let’s call the function and visualize the resulting data...
X = make_hello(1000)
colourise = dict(c = X[:, 0], cmap = plt.cm.get_cmap("rainbow", 5))
plt.scatter(X[:, 0], X[:, 1], **colourise)
plt.axis("equal")
plt.show()
plt.clf()

###### Multidimensional Scaling (MDS)
# Looking at data like this, we can see that the particular choice of x and y 
# values of the dataset are not the most fundamental description of the data; we 
# can scale, shrink, or rotate the data, and the “HELLO” will still be apparent. 
# For example, if we use a rotation matrix to rotate the data, the x and y values
# change, but the data is still fundamentally the same.
def rotate(X, angle):
 theta = np.deg2rad(angle)
 R = [[np.cos(theta), np.sin(theta)],
      [-np.sin(theta), np.cos(theta)]]
 return np.dot(X, R)

X_2 = rotate(X, 20) + 5
plt.scatter(X_2[:, 0], X_2[:, 1], **colourise)
plt.axis("equal")
plt.show()
plt.clf()

# This tells us that the x and y values are not necessarily fundamental to the 
# relationships in the data. What is fundamental, in this case, is the distance 
# between each point and the other points in the dataset.

# A common way to represent this is to use a distance matrix: for N points, we 
# construct an N × N array such that entry i, j contains the distance between 
# point i and point j. Let’s use Scikit-Learn’s efficient pair wise_distances 
# function to do this for our original data...
from sklearn.metrics import pairwise_distances
D = pairwise_distances(X)
D.shape
# As promised, for our N = 1,000 points, we obtain a 1,000×1,000 matrix, which 
# can be visualized as...
plt.imshow(D, zorder = 2, cmap = "Blues", interpolation = "nearest")
plt.colorbar();
plt.show()
plt.clf()
# If we similarly construct a distance matrix for our rotated and translated 
# data, we see that it is the same...
D_2 = pairwise_distances(X_2)
np.allclose(D, D_2)

# This distance matrix gives us a representation of our data that is invariant to
# rotations and translations, but the visualization of the matrix is not entirely
# intuitive. 

# While computing this distance matrix from the (x, y) coordinates is 
# straight‐forward, transforming the distances back into x and y coordinates is 
# rather difficult. This is exactly what the multidimensional scaling algorithm 
# aims to do - given a distance matrix between points, it recovers a 
# D-dimensional coordinate representation of the data. Let’s see how it works 
# for our distance matrix, using the precomputed dissimilarity to specify that 
# we are passing a distance matrix...
from sklearn.manifold import MDS
model = MDS(n_components = 2, dissimilarity = "precomputed", random_state = 1)
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1], **colourise)
plt.axis("equal");
plt.show()
plt.clf()
# The MDS algorithm recovers one of the possible two-dimensional coordinate 
# representations of our data, using only the N × N distance matrix describing 
# the relationship between the data points.

##### MDS as Manifold Learning
# The usefulness of this becomes more apparent when we consider the fact that 
# distance matrices can be computed from data in any dimension. So, for example
def random_projection(X, dimension = 3, rseed = 42):
 assert dimension >= X.shape[1]
 rng = np.random.RandomState(rseed)
 C = rng.randn(dimension, dimension)
 e, V = np.linalg.eigh(np.dot(C, C.T))
 return np.dot(X, V[:X.shape[1]])

X_3 = random_projection(X, 3)
X_3.shape
# Let’s visualize these points to see what we’re working with...
from mpl_toolkits import mplot3d
ax = plt.axes(projection = "3d")
ax.scatter3D(X_3[:, 0], X_3[:, 1], X_3[:, 2],
             **colourise)
ax.view_init(azim = 70, elev = 50)
plt.show()
plt.clf()

# We can now ask the MDS estimator to input this three-dimensional data, compute 
# the distance matrix, and then determine the optimal two-dimensional embedding 
# for this distance matrix. The result recovers a representation of the original 
# data.
model = MDS(n_components = 2, random_state = 1)
out_3 = model.fit_transform(X_3)
plt.scatter(out_3[:, 0], out_3[:, 1], **colourise)
plt.axis("equal");
plt.show()
plt.clf()

# This is essentially the goal of a manifold learning estimator - given 
# high-dimensional embedded data, it seeks a low-dimensional representation of 
# the data that preserves certain relationships within the data. In the case of 
# MDS, the quantity preserved is the distance between every pair of points.

##### Nonlinear Embeddings: Where MDS Fails
# Where MDS breaks down is when the embedding is nonlinear—that is, when it goes 
# beyond this simple set of operations. Consider the following embedding, which 
# takes the input and contorts it into an “S” shape in three dimensions.
def make_hello_s_curve(X):
 t = (X[:, 0] - 2) * 0.75 * np.pi
 x = np.sin(t)
 y = X[:, 1]
 z = np.sign(t) * (np.cos(t) - 1)
 return np.vstack((x, y, z)).T

X_S = make_hello_s_curve(X)
# This is again three-dimensional data, but we can see that the embedding is 
# much more complicated...
from mpl_toolkits import mplot3d
ax = plt.axes(projection = "3d")
ax.scatter3D(X_S[:, 0], X_S[:, 1], X_S[:, 2], 
             **colourise);
plt.show()
plt.clf()
# If we try a simple MDS algorithm on this data, it is not able to “unwrap” this 
# nonlinear embedding, and we lose track of the fundamental relationships in the 
# embedded manifold.
from sklearn.manifold import MDS
model = MDS(n_components = 2, random_state = 2)
out_S = model.fit_transform(X_S)
plt.scatter(out_S[:, 0], out_S[:, 1], **colourise)
plt.axis("equal");
plt.show()
plt.clf()
# The best two-dimensional linear embedding does not unwrap the S-curve, but 
# instead throws out the original y-axis.

##### Nonlinear Manifolds: Locally Linear Embedding
# We can see that the source of the problem is that MDS tries to preserve 
# distances between faraway points when constructing the embedding. But what if 
# we instead modified the algorithm such that it only preserves distances 
# between nearby points? The resulting embedding would be closer to what we want.

# Rather than preserving all distances, it instead tries to preserve only the 
# distances between neighboring points - in this case, the nearest 100 neighbors 
# of each point.

# LLE comes in a number of flavors; here we will use the modified LLE algorithm 
# to recover the embedded two-dimensional manifold. In general, modified LLE does
# better than other flavors of the algorithm at recovering well-defined manifolds
# with very little distortion.
from sklearn.manifold import LocallyLinearEmbedding
model = LocallyLinearEmbedding(n_neighbors = 100, n_components = 2, 
                               method = "modified", eigen_solver = "dense")
out = model.fit_transform(X_S)
# plot data
fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colourise)
ax.set_ylim(0.15, -0.15);
plt.show()
plt.clf()
# The result remains somewhat distorted compared to our original manifold, but 
# captures the essential relationships in the data!

##### Example: Isomap on Faces
# One place manifold learning is often used is in understanding the relationship 
# between high-dimensional data points. Here let’s apply Isomap on some faces 
# data.
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person = 30)
faces.data.shape

# plot data
fig, ax = plt.subplots(4, 8, subplot_kw = dict(xticks = [], yticks = []))
for i, axi in enumerate(ax.flat):
 axi.imshow(faces.images[i], cmap = "gray")
plt.show();
plt.clf()
# We would like to plot a low-dimensional embedding of the 2,914-dimensional data
# to learn the fundamental relationships between the images. One useful way to 
# start is to compute a PCA, and examine the explained variance ratio, which 
# will give us an idea of how many linear features are required to describe the 
# data.
from sklearn.decomposition import PCA
model = PCA(n_components = 100, svd_solver = "randomized", whiten = True).fit(faces.data)
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel("n components")
plt.ylabel("cumulative variance");
plt.show()
plt.clf()
# We see that for this data, nearly 100 components are required to preserve 90% 
# of the variance. This tells us that the data is intrinsically very high
# dimensional — it can’t be described linearly with just a few components.

# When this is the case, nonlinear manifold embeddings like LLE and Isomap can be
# helpful. We can compute an Isomap embedding on these faces using the same 
# pattern shown before...
from sklearn.manifold import Isomap
model = Isomap(n_components = 2)
proj = model.fit_transform(faces.data)
proj.shape

# The output is a two-dimensional projection of all the input images. To get a 
# better idea of what the projection tells us, let’s define a function that will 
# output image thumbnails at the locations of the projections.
from matplotlib import offsetbox
def plot_components(data, model, images = None, ax = None, 
                    thumb_frac = 0.05, cmap = "gray"):
    ax = ax or plt.gca()
    
    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], ".k")
    
    if images is not None:
     min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
     shown_images = np.array([2 * proj.max(0)])
     for i in range(data.shape[0]):
      dist = np.sum((proj[i] - shown_images) ** 2, 1)
      if np.min(dist) < min_dist_2:
       # don't show points that are too close
       continue
      
      shown_images = np.vstack([shown_images, proj[i]])
      imagebox = offsetbox.AnnotationBbox(
        offsetbox.OffsetImage(images[i], cmap = cmap),
                                         proj[i])
      ax.add_artist(imagebox)

# Call function and plot...
fig, ax = plt.subplots(figsize = (10, 10))
plot_components(faces.data, model = Isomap(n_components = 2),
                images = faces.images[:, ::2, ::2]);
plt.show()
plt.clf()

### In Depth: k-Means Clustering
# Here we will move on to another class of unsupervised machine learning 
# models - clustering algorithms.

# Many clustering algorithms are available in Scikit-Learn and elsewhere, but 
# perhaps the simplest to understand is an algorithm known as k-means 
# clustering, which is implemented in sklearn.cluster.KMeans. We begin with 
# the standard imports.
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # for plot style
import numpy as np

#### Intro to k-Means
# First, let’s generate a two-dimensional dataset containing four distinct 
# blobs. To emphasise that this is an unsupervised algorithm, we will leave 
# the labels out of the visualisation.
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples = 300, centers = 4, 
                       cluster_std = 0.60, random_state = 0)
plt.scatter(X[:, 0], X[:, 1], s = 50)
plt.show()
plt.clf()
# By eye, it is relatively easy to pick out the four clusters. The k-means 
# algorithm does this automatically, and in Scikit-Learn uses the typical 
# estimator API.
from sklearn.cluster import KMeans
kMeans = KMeans(n_clusters = 4)
kMeans.fit(X)
y_hat = kMeans.predict(X)
# plot results
plt.scatter(X[:, 0], X[:, 1], c = y_hat, s = 50, cmap = "viridis")
centers = kMeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = "black", s = 200, alpha = 0.5)
plt.show()
plt.clf()

#### k-Means Algorithm - EM
# Expectation–maximization (E–M) is a powerful algorithm that comes up in a 
# variety of contexts within data science. k-means is a particularly simple 
# and easy-to- understand application of the algorithm.

# In short, the expectation–maximization approach consists of the following 
# procedure

# 1. Guess some cluster centers
# 2. Repeat until converged
#  a. E-Step - assign points to the nearest cluster center
#  b. M-Step - set the cluster centers to the mean

# Here the “E-step” or “Expectation step” is so named because it involves 
# updating our expectation of which cluster each point belongs to. The 
# “M-step” or “Maximization step” is so named because it involves maximizing 
# some fitness function that defines the location of the cluster centers — in 
# this case, that maximization is accomplished by taking a simple mean of the 
# data in each cluster.

# The k-means algorithm is simple enough that we can write it in a few lines 
# of code. The following is a very basic implementation
from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed = 2):
 # 1. Randomly choose clusters
 rng = np.random.RandomState(rseed)
 i = rng.permutation(X.shape[0])[:n_clusters]
 centers = X[i]
 
 while True:
  # 2a. Assign labels based on closest center
  labels = pairwise_distances_argmin(X, centers)
  # 2b. Find new centers from means of points
  new_centers = np.array([X[labels == i].mean(0)
                          for i in range(n_clusters)])
  # 2c. Check for convergence
  if np.all(centers == new_centers):
   break
  centers = new_centers
  
 return centers, labels

centers, labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c = labels, s = 50, 
            cmap = "viridis");
plt.show()
# Most well-tested implementations will do a bit more than this under the hood,
# but the preceding function gives the gist of the expectation–maximization 
# approach.

# NB! Careful about non-linearity

##### Example 1: k-Means on digits
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape
# The clustering can be performed as we did before.
kMeans = KMeans(n_clusters = 10, random_state = 0)
clusters = kMeans.fit_predict(digits.data)
kMeans.cluster_centers_.shape
# The result is 10 clusters in 64 dimensions. Notice that the cluster centers 
# themselves are 64-dimensional points, and can themselves be interpreted as 
# the “typical” digit within the cluster. Let’s see what these cluster centers
# look like.
fig, ax = plt.subplots(2, 5, figsize = (8, 3))
centers = kMeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
 axi.set(xticks = [], yticks = [])
 axi.imshow(center, interpolation = "nearest", cmap = plt.cm.binary)
plt.show()
plt.clf()
# Because k-means knows nothing about the identity of the cluster, the 0–9 
# labels may be permuted. We can fix this by matching each learned cluster 
# label with the true labels found in them.
from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(10):
 mask = (clusters == i)
 labels[mask] = mode(digits.target[mask])[0]
# Now we can check how accurate our unsupervised clustering was in finding 
# similar digits within the data.
from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)
# With just a simple k-means algorithm, we discovered the correct grouping for 
# 80% of the input digits! Let’s check the confusion matrix for this...
from sklearn.metrics import confusion_matrix
c_mat = confusion_matrix(digits.target, labels)
sns.heatmap(c_mat.T, square = True, annot = True, fmt = "d", cbar = False,
            xticklabels = digits.target_names, 
            yticklabels = digits.target_names)
plt.xlabel("true label")
plt.ylabel("y_hat label");
plt.show()
plt.clf()

# Just for fun, let’s try to push this even further. We can use the 
# t-distributed stochastic neighbor embedding (t-SNE) algorithm to preprocess 
# the data before performing k-means. t-SNE is a non‐linear embedding 
# algorithm that is particularly adept at preserving points within clusters. 
# Let’s see how it does.
from sklearn.manifold import TSNE

# Project the data
tsne = TSNE(n_components = 2, init = "pca", random_state = 0)
digits_proj = tsne.fit_transform(digits.data)

# Compute clusters
kMeans = KMeans(n_clusters = 10, random_state = 0)
clusters = kMeans.fit_predict(digits_proj)

# Permute labels
labels = np.zeros_like(clusters)
for i in range(10):
 mask = (clusters == i)
 labels[mask] = mode(digits.target[mask])[0]

# Compute accuracy
accuracy_score(digits.target, labels)
# That’s roughly 94% classification accuracy without using the labels. This is
# the power of unsupervised learning when used carefully - it can extract 
# information from the dataset that it might be difficult to do by hand or by
# eye.

##### Example 2: k-means for color compression
from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")
ax = plt.axes(xticks = [], yticks = [])
ax.imshow(china);
plt.show()
plt.clf()
# The image itself is stored in a three-dimensional array of size (height, 
# width, RGB), containing red/blue/green contributions as integers from 0 to 
# 255.
china.shape
# One way we can view this set of pixels is as a cloud of points in a 3D colour
# space. We will reshape the data to [n_samples x n_features], and rescale the
# colours so that they lie between 0 and 1.
data = china / 255.0 # use 0, ..., 1 scale.
data = data.reshape(427 * 640, 3)
data.shape
# We can visualise these pixels in this color space, using a subset of 10,000 
# pixels for efficiency.
def plot_pixels(data, title, colors = None, N = 10000):
 if colors is None:
  colors = data
  
 # choose a random subset
 rng = np.random.RandomState(0)
 i = rng.permutation(data.shape[0])[:N]
 colors = colors[i]
 R, G, B = data[i].T
 
 fig, ax = plt.subplots(1, 2, figsize = (16, 6))
 ax[0].scatter(R, G, color = colors, marker = ".")
 ax[0].set(xlabel = "Red", ylabel = "Green", xlim = (0, 1), ylim = (0, 1))
 
 ax[1].scatter(R, B, color = colors, marker = ".")
 ax[1].set(xlabel = "Red", ylabel = "Blue", xlim = (0, 1), ylim = (0, 1))
 
 fig.suptitle(title, size = 20);
 
plot_pixels(data, title = "Input color space: 16 million possible colors")
plt.show()
plt.clf()
# Now let’s reduce these 16 million colors to just 16 colors, using a k-means 
# clustering across the pixel space. Because we are dealing with a very large 
# dataset, we will use the mini batch k-means, which operates on subsets of 
# the data to compute the result much more quickly than the standard k-means 
# algorithm
from sklearn.cluster import MiniBatchKMeans
kMeans = MiniBatchKMeans(16)
kMeans.fit(data)
new_colors = kMeans.cluster_centers_[kMeans.predict(data)]

plot_pixels(data, colors = new_colors, 
            title = "Reduced color space: 16 colors")
plt.show()
plt.clf()
# The result is a recoloring of the original pixels, where each pixel is 
# assigned the color of its closest cluster center. Plotting these new colors 
# in the image space rather than the pixel space shows us the effect of this
china_recolored = new_colors.reshape(china.shape)

fig, ax = plt.subplots(1, 2, figsize = (16, 6), 
                       subplot_kw = dict(xticks = [], yticks = []))

fig.subplots_adjust(wspace = 0.05)
ax[0].imshow(china)
ax[0].set_title("Original Image", size = 16)
ax[1].imshow(china_recolored)
ax[1].set_title("16-color Image", size = 16);
plt.show()
plt.clf()
# Some detail is certainly lost in the rightmost panel, but the overall image 
# is still easily recognizable. This image on the right achieves a compression
# factor of around 1 million!

### In Depth: Gaussian Mixture Models
