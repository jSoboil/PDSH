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
