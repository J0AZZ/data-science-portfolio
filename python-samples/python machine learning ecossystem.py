import numpy as np
import pandas as pd
import pymssql

#-----------------------NUMPY-------------------------#
num_array = np.array([1, 2, 3])
str_array = np.array(['a', 'b', 'c'])

zeros_matrix = np.zeros((3, 5))
ones_matrix = np.ones((3, 5))
identity_matrix = np.identity(5)

#random values
rndm = np.random.randn(3, 5)

#range originated matrix
new_array = np.arange(15).reshape(3, 5)

#advanced indexing
arr = np.arange(15).reshape(5, 3)
##[[rows],[columns]]
new_arr = arr[[0, 2, 4], [0, 0, 0]]
##select even items
new_arr_2 = arr[arr % 2 == 0]
##select items that are greater than 5
new_arr_3 = arr[arr > 5]

#operations on ndarrays
num_array + 10
num_array + np.array([4, 5, 6])
np.dot(num_array, np.random.randn(3,1))

#separate integer and fractional parts
np.modf(rndm)

#transpose a matrix
arr.T

#solve linear systems
A = np.array([[7, 5, -3], [3, -5, 2], [5, 3, -7]])
B = np.array([16, -8, 0])
x = np.linalg.solve(A, B)
##check solution
np.allclose(np.dot(A, x), B)
#-------------------------------------------------------------------------------------#

#-----------------------------------PANDAS-------------------------------------#
#generate data frame from dictionary array
d = [{"city": "Niterói", "state": "RJ"},
     {"city": "Osasco", "state": "SP"},
     {"city": "Recife", "state": "PE"}]
pd.DataFrame(d)

#read data from csv file
city_data = pd.read_csv(filepath_or_buffer="worldcities.csv")
##show ten first samples
city_data.head(n=10)
##show five last samples
city_data.tail(n=5)

#read information from database (MSSQL)
server = ""     #address of database's server
user = ""       #username for database's server
password = ""   #password for the above user
database = ""   #database in which the table is present
conn = pymssql.connect(server=server, user=user, password=password, database=database)
query = "select * from some_table"
df = pd.read_sql(query, conn)

#values attribute
df_2 = pd.DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"])
df2_values = df_2.values

#descriptive statistics
columns_numeric = city_data['lat', 'long', 'pop']
city_data[columns_numeric].mean()
city_data[columns_numeric].sum()
city_data[columns_numeric].count()
city_data[columns_numeric].median()
city_data[columns_numeric].quantile(0.8)
##most important statistics for numerical data
city_data[columns_numeric].describe()

#generate random sample from data frame
cd1 = city_data.sample(3)
cd2 = city_data.sample(3)

#concat two data frames
new_df = pd.concat(cd1, cd2)
#-------------------------------------------------------------------------------------#

#-----------------------------------SCIKIT-LEARN-------------------------------------#
from sklearn import datasets, linear_model
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


diabetes = datasets.load_diabetes()
X = diabetes.data
Y = diabetes.target
feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

#split data into train and test
y_train = diabetes.target[:310]
x_train = diabetes.data[:310]
y_test = diabetes.target[310:]
x_test = diabetes.data[310:]

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

estimator = GridSearchCV(lasso, dict(alpha=alphas))
estimator.fit(x_train, y_train)

best_score = estimator.best_score_
best_estimator = estimator.best_estimator_

estimator.predict(x_test)
#-------------------------------------------------------------------------------------#

#-----------------------------------THEANO-------------------------------------#
import theano.tensor as tensor
from theano import function
x = tensor.dscalar('x')
y = tensor.dscalar('y')
z = (x + y)/2
#defining a symbolic function
f = function([x, y], z)
f(5, 4)
#-------------------------------------------------------------------------------------#

#-----------------------------------KERAS-------------------------------------#
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout

cancer = load_breast_cancer()
x_train = [:340]
y_train = [:340]
x_test = [340:]
y_test = [340:]

#fully connected hidden and output layers, 30 inputs
model = Sequential()
model.add(Dense(15, input_dim=30, activation='relu'))
model.add(Dense(i, activation='sigmoid'))

#compiling the sequential model
##binary_crossentropy is commonly used in binary classification problems
##rmsprop is an upgrade of the normal gradient descent algorithm
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#fit the model
##for 100 observations and batch_size=10, each epoch will be composed by 10 iterations of backpropagation
model.fit(x_train, y_train, epochs=20, batch_size=50)

#evaluate the model
predictions = model.predict_classes(x_test)
print('Accuracy: ', metrics.accuracy_score(y_true=y_test, y_pred=predictions))
print(metrics.classification_report(y_true=y_test, y_pred=predictions))score