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
