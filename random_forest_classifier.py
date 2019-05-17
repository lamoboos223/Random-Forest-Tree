# Random Forest Classifier

# Importing the libraries

from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the datasets

datasets = pd.read_csv('Social_Network_Ads.csv')
dataset_lama = pd.read_csv('lama_data.csv')
# features, ignoring the userID column
X = datasets.iloc[:, [2, 3]].values
x_lama = dataset_lama.iloc[:, [2, 3]].values
# class
Y = datasets.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
# doing the cross validation by spliting the dataset into 70% training and 30% testing
X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X, Y, test_size=0.30, random_state=0)

# Feature Scaling
# to make the learning step faster by rounding the features values a little bit
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)
x_lama = sc_X.fit_transform(x_lama)
# Fitting the classifier into the Training set

classifier = RandomForestClassifier(
    n_estimators=200, criterion='entropy', random_state=0)
classifier.fit(X_Train, Y_Train)

# Predicting the test set results
Y_Pred = classifier.predict(X_Test)
y_lama = classifier.predict(x_lama)
print("Lama purchace is= ", y_lama)
# Making the Confusion Matrix

cm = confusion_matrix(Y_Test, Y_Pred)

# Visualising the Training set results

X_Set, Y_Set = X_Train, Y_Train
# X1 = Gender, X2 = Age
X1, X2 = np.meshgrid(np.arange(start=X_Set[:, 0].min() - 1, stop=X_Set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_Set[:, 1].min() - 1, stop=X_Set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Random Forest Classifier (Training set) - Scatter Plot')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results

X_Set, Y_Set = X_Test, Y_Test
X1, X2 = np.meshgrid(np.arange(start=X_Set[:, 0].min() - 1, stop=X_Set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_Set[:, 1].min() - 1, stop=X_Set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Random Forest Classifier (Test set) - Scatter Plot')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
