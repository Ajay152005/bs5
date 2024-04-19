#importing necessary libraries
#importing additional libraries for hyperparameter tuning and Visualization
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_decision_regions
import joblib

iris = datasets.load_iris()
X = iris.data
y = iris.target
#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)

#define a range of neighbors to search over
param_grid = {'n_neighbors': range(1, 11)}


knn = KNeighborsClassifier()
#Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

#get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters: ", best_params)


#standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection using SelectKBest and ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=3)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

#train the classifier with selected features
knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])
knn.fit(X_train_selected, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test_selected)

#calculate additional evaluation metrics
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

print("Classification Report: ")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("Precision:", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)
# #load the iris dataset


# # Use the best model for prediction
# best_knn = grid_search.best_estimator_
# y_pred = best_knn.predict(X_test)

# #initialize the KNN classifier

# # Calculate accuray 
# accuracy = metrics.accuracy_score(y_test, y_pred)
# print("Accuracy: ", accuracy)

# # #train the classifier
# # knn.fit(X_train, y_train)

# # #predict the labels for the test set
# # y_pred = knn.predict(X_test)

# #Calculate the accuracy of the classifier
# # accuracy = metrics.accuracy_score(y_test, y_pred)

# # print("Accuracy:", accuracy)

# #visualize confusion matrix
# cm = metrics.confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=iris.target_name, yticklabels=iris.target_names)
# plt.xlable('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

#feature importance analysis
feature_importances = knn.feature_importances_
print("Feature Importances: ")
for i, importance in enumerate(feature_importances):
    print(f"Feature {i+1}: {importance}")

#save the trained model to disk
joblib.dump(knn, 'knn_model.pkl')

#simple user interface for predicting new instances
def predicted_new_instance(model, scaler, selector):
    sepal_length = float(input("Enter sepal length (cm): "))
    sepal_width = float(input("Enter sepal width (cm): "))
    petal_length = float(input("Enter petal length (cm): "))
    petal_width = float(input("Enter petal width (cm): "))

    #Standardize and select features 
    instance = [[sepal_length, sepal_width, petal_length, petal_width]]
    instance_scaled = scaler.transform(instance)
    instance_selected = selector.transform(instance_scaled)

    #predict the label 
    prediction = model.predict(instance_selected)
    print("Predicted Class:", iris.target_names[prediction[0]])

#example usuage of the user interface
print('n--- Predict New Instance ---')
predicted_new_instance(knn, scaler, selector)

# cross-validation for model evaluation
cv_scores = cross_val_score(knn, X_train_selected, y_train, cv=5)
print("Cross-Validation Scores: ", cv_scores)
print("Mean CV Accuracy: ", cv_scores.mean())
#Visualize decision boundaries

plt.figure(figsize=(10, 8))
plot_decision_regions(X_train_selected, y_train, clf= knn, legends=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundaries')
plt.show()

# Option to load a pre-trained model for prediction
def load_model_and_predict(instance):
    loaded_model = joglib.load('knn_model.pkl')
    instance_scaled = scaler.transform(instance)
    instance_selected = selector.transform(instance_scaled)
    prediction = loaded_model.predict(instance_selected)
    return prediction

#example usage of loading pre-trained model for prediction
new_instances = [[5.1, 3.5, 1.4, 0.2]] #example new instance
print('\n --- Loading Pre-trained Model for prediction ---')
print("predicted class: ", iris.target_names[load_model_and_predict(new_instance)[0]])
