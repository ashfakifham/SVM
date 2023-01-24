# Python machine learning exercise
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import decomposition  # for pca
from sklearn import preprocessing  # module and method to convert categorical to discrete numbers
from sklearn.model_selection import train_test_split  # module and method for test data splitting
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC

with open("C:/Users/User/Documents/IRIS.csv") as file:
    data = pd.read_csv(file)
    # Opening a file this way closes the document
    # keeps a record of the file data only
    # which is better for memory optimization
setosa = data['species'] == 'Iris-setosa'  #creates a series, a list with index
versicolor = data['species'] == 'Iris-versicolor'
verginica = data['species'] == 'Iris-verginica'

print("Describing an Iris Category series/list")
print()
print(data[versicolor].describe())

plt.scatter(data['sepal_length'],data['petal_width'])  #showing the overall spread of sepal vs petal in the data
plt.show()


print(data.shape)
print()
print(data.head(3))
print()
print(type(data))
print()
print(data.keys())
print()
print(data.info())
print()
print(data.describe())
print()
print(data["species"].value_counts())
print()
print()

ax = plt.subplots(1, 1, figsize=(7, 7))
sns.countplot(data=data, x='species')

plt.title("Iris Species count")

plt.show()  # getting a count plot of the number of unique observations available in the data

data["species"].value_counts().plot.pie(figsize=(7, 7))  # easy to get the unique values
# and count the values to plot a graph
plt.show() #gives a nice pie chart easily

plt.scatter(data['sepal_length'],data['sepal_width'])


plt.show()

#



correlation_data = data.iloc[:, 0:3] # for correlation, we only pass in the numerical input values
C = correlation_data.corr()

sns.heatmap(data=C, mask=np.zeros_like(C), cmap=sns.diverging_palette(220, 10, as_cmap=True), linewidths=0.5)

plt.show()


sns.boxplot(data=data)
plt.show()
# this is used to convert label fields into numerical for analysis
le = preprocessing.LabelEncoder()  # creating a label encoder object, to call its function which converts
# the label data into numbers

data['species'] = le.fit_transform(data['species'])  # overwrite the labels with numerical values
# pass the field to be transformed into the function
print(data.head(3))

# create an array of X and Y values
# These values are passed into the PCA for being computed

X = data.iloc[:, 0:3].values
Y = data.iloc[:, 4].values

fig8 = plt.figure(1, figsize=(8, 10))
plt.clf()

pca = decomposition.PCA(n_components=3)  # create a PCA object from decomposition based on number of input variables
# then pass the input variables into the pca object to be fit
pca.fit(X)  # fit the model to X
X = pca.transform(X)  # apply dimensionality reduction to X

ax = plt.axes(projection='3d')
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[Y == label, 0].mean(),
              X[Y == label, 1].mean() + 1.5,
              X[Y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

y = np.choose(Y, [1, 2, 0])
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
plt.show()


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)  # import the function from sklearn and split the data

# Running the model with a different split of data
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size=0.2)

svm = SVC(kernel = 'rbf', random_state = 1)
svm.fit(X_train, Y_train) #fit the model according to the training data
Y_pred= svm.predict(X_test)  # perform classifications on the new/test data
svm_acc= svm.score(X_test, Y_test) # gives the mean accuracy of the model after the initial fitting is done using the test data
svm_metric = metrics.accuracy_score(Y_test, Y_pred)

cm = metrics.confusion_matrix(Y_test,Y_pred)
print("The confusion matrix is:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= svm.classes_)
disp.plot()
plt.show()

print()
print("SVM accuracy is: ", svm_acc)
print()
print("SVM Metric is: ", svm_metric)
print(Y_pred)

