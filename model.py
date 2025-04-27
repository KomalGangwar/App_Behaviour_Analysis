import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import time

# Loading the dataset
dataset = pd.read_csv('new_appdata10.csv')

# Data Preprocessing
response = dataset['enrolled']  # Extracting the response variable
dataset = dataset.drop(columns = 'enrolled')  # Dropping the response column

# Splitting the data into train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset, response, test_size = 0.2, random_state = 0)

# Storing user identifiers
train_identifier = x_train['user']
test_identifier = x_test['user']

# Dropping 'user' column as it's not a feature
x_train = x_train.drop(columns = 'user')
x_test = x_test.drop(columns = 'user')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

# Scaling the features
x_train2 = pd.DataFrame(sc_x.fit_transform(x_train))
x_test2 = pd.DataFrame(sc_x.transform(x_test))

# Restoring the column names and indices after scaling
x_train2.columns = x_train.columns.values
x_test2.columns = x_test.columns.values
x_train2.index = x_train.index.values
x_test2.index = x_test.index.values

# Assigning back the scaled features
x_train, x_test = x_train2, x_test2

# Model Building using Logistic Regression
from sklearn.linear_model import LogisticRegression
cls = LogisticRegression(random_state = 0, penalty = "l1", solver = 'saga')
cls.fit(x_train, y_train)  # Training the model
y_pred = cls.predict(x_test)  # Making predictions

# Generating the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)

# Creating a DataFrame for confusion matrix and displaying it
df_cm = pd.DataFrame(cm, index = ['No Enrollment', 'Enrollment'], columns = ['No Enrollment', 'Enrollment'])
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
plt.show()

# Displaying model metrics: Accuracy, Precision, Recall, F1 Score
def evaluate_model_metrics(y_true, y_pred):
    print("Accuracy: %0.4f" % accuracy_score(y_true, y_pred))
    print("Precision: %0.4f" % precision_score(y_true, y_pred))
    print("Recall: %0.4f" % recall_score(y_true, y_pred))
    print("F1 Score: %0.4f" % f1_score(y_true, y_pred))

evaluate_model_metrics(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = cls, X = x_train, y = y_train, cv = 10)
print("Logistic Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

# Analyzing the model coefficients
pd.concat([
    pd.DataFrame(dataset.drop(columns = 'user').columns, columns = ["Features"]),
    pd.DataFrame(np.transpose(cls.coef_), columns = ["Coefficients"])
], axis = 1)

# Hyperparameter Tuning using Grid Search

## Grid Search (Round 1)
from sklearn.model_selection import GridSearchCV

# Define regularization methods and hyperparameter space
penalty = ['l1', 'l2']
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
params = dict(C=C, penalty=penalty)

# Perform GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(estimator = cls, param_grid = params, scoring = "accuracy", cv = 10, n_jobs = -1)
t0 = time.time()
grid_search = grid_search.fit(x_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

# Best accuracy and hyperparameters after first round of grid search
rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
print(rf_best_accuracy, rf_best_parameters)

## Grid Search (Round 2)
C = [0.1, 0.5, 0.9, 1, 2, 5]  # Adjusted values of C
params = dict(C=C, penalty=penalty)

# Second round of grid search
grid_search = GridSearchCV(estimator = cls, param_grid = params, scoring = "accuracy", cv = 10, n_jobs = -1)
t0 = time.time()
grid_search = grid_search.fit(x_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

# Best accuracy and hyperparameters after second round of grid search
rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
print(rf_best_accuracy, rf_best_parameters)

# Final Results: Combining predicted and actual results
final_results = pd.concat([y_test, test_identifier], axis=1)
final_results['predicted_results'] = y_pred
final_results = final_results[['user', 'enrolled', 'predicted_results']].reset_index(drop=True)

# Final results with user ID, actual and predicted enrollment status
print(final_results)