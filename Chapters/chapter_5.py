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
