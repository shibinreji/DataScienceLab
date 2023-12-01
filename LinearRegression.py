# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
df = datasets.load_diabetes()

# Extract feature names
df['feature_names']

# Load the features (X) and target variable (y) from the dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Check the shape of the feature matrix
diabetes_X.shape

# Check the shape of the target variable
diabetes_y.shape

# Select a single feature (age) for the demonstration
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Check the shape of the modified feature matrix
diabetes_X.shape

# Split the data into training and testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create a linear regression model
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions on the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# Print the coefficients of the linear regression model
print("Coefficients: \n", regr.coef_)

# Print the mean squared error of the predictions
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# Print the coefficient of determination (R^2 score)
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot the actual vs. predicted values
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
plt.xlabel("Age")
plt.ylabel("Diabetes Progression")
plt.show()
