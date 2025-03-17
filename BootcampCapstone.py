# Practical Machine Learning 
### Project: Model Selection

# Abstract

# Part 1: Business and Data Understanding

### Q. Identify a suitable dataset for your canditate question(s)

Go have a look at any of these websites:

* https://www.kaggle.com/datasets 
* https://datasetsearch.research.google.com/
* https://data.gov.uk/

Find an interesting looking data set related to your problem domain and get a copy of it


### Q. Define the problem domain

# This data is on Diabetes Health Indicators and was published in 2015.
# The reports focus on different indicators such as "High BP" and "High Chol", and relates this to
# a diagnosis of either "No Diabetes", "Pre-Diabetes", or "Diabetes" which is coded
# as 0, 1 and 2 respectivly. 

# This project will look at the data present in the report and construct a Machine Learning model that predicts 
# "Diabetes" based on other metric data included in the data set such as "High Chol", or "PhysActivity". 
# This will allow us to predict whether Diabetes is present based on other data in future patients.


# First off, I am importing then looking at the first few rows of the data set:

import pandas as pd
path = ("diabetes.csv")
data = pd.read_csv(path)
data.head()

### Q. Identify candidate questions for your machine learning project

# Data has imported correctly. 
# As above my machine learning model will hopefully be able to predict indcidens of Diabetes based on values in other metrics.
# My inputs (x) will be all column names except "Diabetes_012" and my outputs (y) will be = "Diabetes_012".
# It is important to note that the data within the output column is scored from 0, 1, 2, based on whether there
# is no diagnosis of Diabates, whether PreDiabetes is present, or whether there is a diagnosis of Diabates.

### Q. Generate a descriptive statistics report for the columns in your dataset

## Now it might be helpful to understand the data set from a statistical perspective, data.describe() can provide this:
data.describe()

# Generate correlation heatmap, boxplot, and any other graphs

## It is also helpful to see if there are any correlations at all within the data set.This is easiest to understand
## through visualisations so lets create one of those aswell. 

correlations = data.corr() 
correlations
import seaborn as sns
sns.heatmap(correlations, cmap="coolwarm")

## How about with a different correlation model?

# This shows correlation but is perhaps a bit small and it is hard to distinguish each value, a score and a larger map would help with this:

plt.figure(figsize=(20, 8))
heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)

# I will also give a title to the heatmap so I know what I am looking at.
# Pad defines the distance of the title from the top of the heatmap.

heatmap.set_title('Diabetes Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

# I will now change the colour scheme back to 'coolwarm' as it is intuitive that red is danger and
# helps visualse a greater physical health risk or each domain.
# I will also make the title a bit bigger for ease of reading.

plt.figure(figsize=(20, 8))
heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True, cmap='coolwarm')
heatmap.set_title('Diabetes Correlation Heatmap', fontdict={'fontsize':18}, pad=12);

# In English not Python- rationale why you choose to select or drop columns

# So aside from the below addressing null values it is important to look and see if there are any 
# columns that don't make sense or fit with the data. So the majority of the variables are measured in a binary
# way, for example: If high cholesterol is present a score of 1 in the column is indicated, if it isn't present, 
# 0 is scored. However, for some columns such as Age or BMI these variables are recorded with integers like "43".
# This would skew the model, so for now I will drop these columns to ensure that data makes sense with each other.

# lets drop these columns.
data.drop(['BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth', 'Education', 'Income'], axis=1, inplace=True)
data.head()

# Part 2: Data Preparation

### Q. Discuss the following types of missing data and how they would be handled in reference to your dataset where applicable.
*	Missing completely at random (MCAR)
*	Missing at random (MAR)
*	Missing Not at Random (MNAR)

# null values- first column wise, then row wise


# So null values either MCAR, MAR or MNAR are important to address both at the training end of the ML model, but
# also at model review stage. So now, I will first check for null values in the data set:

data.info()

# There doesn't seem to be any null values present, lets drop any null values just in case.

data.dropna()

# optional: normalize
# As the data within this set is now binary (and thus normalized), we should check for outliers, 
# in order to prevent our model being skewewd, this can also help with normalizing data. 
# min/max will help here as we know our max is 1 and our min is 0.

data.min()
data.max()

# Part 3: Model Selection

### Q. Use the cheat sheet below to choose the algorithm/estimator suitable for building a model to address your candidate question(s)

* https://scikit-learn.org/stable/tutorial/machine_learning_map/

# Now i will begin training my model. First I have to split the data and assign my x and y, which I had already decided above.
# the value test_size=0.2 means that a random 0.2 of the data will be used to test the model after training to ensure accuracy.

from sklearn.model_selection import train_test_split

y = data.loc[:,["Diabetes_012"]]
x = data.iloc[:,2:8]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,random_state=42)



# I will now check to see that x and y training and test data are the domains that they should be

ytest

ytrain

xtest

xtrain

# Now it is important to calculate z score.
# Z-score is the number of standard deviations away from the mean that a value holds
# and is useful in figuring out the accuracy of the model. In this instance it cal help with any anomaly detection
# just in case something went wrong in training/testing.

zmean = xtrain.describe().T['mean']
zstd = xtrain.describe().T['std']
xtrain_norm = (xtrain-zmean)/zstd
xtest_norm = (xtest-zmean)/zstd 

xtrain_norm.head()

# So now I have to import the ML model. As mentioned above, the data in the "Diabates_012" column is either
# 0, 1 or 2 based on whether the individual is not diabetic, prediabetic or has Diabetes respectivly. As this 
# value is discrete it makes sense to use a classifier model.
# I will first use a Random Forest Classifier and then a a Decision Tree Classifier and compare their validity prior to 
# deploying the most accurate one. 

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain_norm, ytrain)

# Now I have my model, lets output some predictions
model_predictions = model.predict(xtest_norm)
print(model_predictions)

ytest["Diabetes_012"]

# Part 4: Model Evaluation

### Q. Identify which of the statistical measures below are suitable for the evaluation of your model.

Classification Metrics:
* Accuracy
* Precision
* Recall
* F1 Score

Regression Metrics:
    
* Mean absolute error (MAE)
* Root mean squared error (RMSE)
* Relative absolute error (RAE)
* Relative squared error (RSE)
* Mean Zero One Error (MZOE)
* Coefficient of determination

 

# So with my first model it is important to check for mistakes
p = model.predict(xtrain_norm)
mistakes = ytrain["Diabetes_012"]-p
df = pd.DataFrame( { 'Diabetes_012': ytrain['Diabetes_012'],

'Predictions': p,

'mistakes': mistakes

})
mistakes = (ytest["Diabetes_012"]-model_predictions)
print(mistakes)

# This is really difficult to visualise over so many rows and so a visualisation helps with this.
# It is reassuring to see no negative results as this would be impossible!

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ytest.index,mistakes)
plt.show()

# Now I can use mean squared error and mean absolute error to check the predictions of the model, both are printed below. 
from sklearn.metrics import mean_squared_error, mean_absolute_error
modelmse = mean_squared_error(model_predictions, ytest)
print("Mean Squared Error:", + modelmse)
modelmae = mean_absolute_error(model_predictions, ytest)
print("Mean Absolute Error", + modelmae)

# Now I will look at accuracy


# Now I have the MSE and MAE for the Random Forest Classifier I will try the Decision Tree Classifier
from sklearn import tree
model2 = tree.DecisionTreeClassifier()
model2.fit(xtrain_norm, ytrain)

model2_predictions = model2.predict(xtest_norm)
print(model2_predictions)
ytest["Diabetes_012"]

p = model2.predict(xtrain_norm)
mistakes = ytrain["Diabetes_012"]-p
df = pd.DataFrame( { 'Diabetes_012': ytrain['Diabetes_012'],

'Predictions': p,

'mistakes': mistakes

})
mistakes = (ytest["Diabetes_012"]-model2_predictions)
print(mistakes)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ytest.index,mistakes)
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
modelmse = mean_squared_error(model2_predictions, ytest)
print("Mean Squared Error:", + modelmse)
modelmae = mean_absolute_error(model2_predictions, ytest)
print("Mean Absolute Error", + modelmae)
