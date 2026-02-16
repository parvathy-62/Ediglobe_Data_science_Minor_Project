import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#--------------------
# Load the dataset
df=pd.read_csv("heart.csv");
#first 5 rows
print(df.head()); 
#Dtataset info
print(df.info());
#Basic statistics
print(df.describe());
#-------------------------

#----------------------------------------
#Exploratory Data Analysis (EDA)

#Target distribution
sns.countplot(x='target', data=df)
plt.title("Heart Disease Distribution")
plt.show()

#Age vs Target
sns.boxplot(x='target', y='age', data=df)
plt.show()

#Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
#----------------------------------------------------

#--------------------------
#Data Preprocessing

#Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

#Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#-----------------------------------

#----------------------------------------
#Model Building

#Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

#Neural Network (Simple MLP)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(max_iter=1000)
mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)
#-------------------------------------------------

#--------------------------------------------------------------
#Model Evaluation
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
#---------------------------------------------------------------

#Feature Importance
#----------------------------------
#For Random Forest:
importances = rf.feature_importances_

for i, v in enumerate(importances):
    print(f"Feature: {df.columns[i]}, Importance: {v}")
#--------------------------------------------------------

#----------------------------------------------------------
#Visualize feature importance
from sklearn.metrics import accuracy_score #to evaluate model performance

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_lr))#accuracy score is seen as 0.85 which is good for a logistic regression model

print("Decision Tree Accuracy:",
      accuracy_score(y_test, y_pred_dt))#accuracy score is seen as 0.75 which is not good for a decision tree model, it may be overfitting the training data.

print("Random Forest Accuracy:",
      accuracy_score(y_test, y_pred_rf))#accuracy score is seen as 0.90 which is good for a random forest model, it may be generalizing well to unseen data.

print("Neural Network Accuracy:",
      accuracy_score(y_test, y_pred_mlp))#accuracy score is seen as 0.88 which is good for a neural network model, it may be capturing complex patterns in the data.

#-------------------------------------------------------------

#------------------------------------------------------------------
#Model Tuning and Validation
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20]
}
#a grid search is performed to find the best hyperparameters for the random forest model, which includes the number of trees (n_estimators) and the maximum depth of the trees (max_depth). 
#The grid search will evaluate different combinations of these parameters using cross-validation to determine which combination yields the best performance on the training data.

grid_rf = GridSearchCV(RandomForestClassifier(), 
                       param_grid, 
                       cv=5)

grid_rf.fit(X_train, y_train)
#After fitting the grid search, the best parameters for the random forest model are printed, along with the tuned accuracy score on the test set. Additionally, the ROC-AUC score is calculated to evaluate the model's performance in distinguishing between classes.
#  A confusion matrix and ROC curve are also visualized to further assess the model's performance. Finally, cross-validation scores are printed to evaluate the model's stability across different subsets of the data.
print("Best Parameters for Random Forest:", grid_rf.best_params_)

best_rf = grid_rf.best_estimator_
#The tuned accuracy score for the random forest model is calculated using the best estimator obtained from the grid search, and the ROC-AUC score is computed to evaluate the model's ability to distinguish between classes. 
# The confusion matrix and ROC curve are visualized to further assess the model's performance, and cross-validation scores are printed to evaluate the model's stability across different subsets of the data.
print("Tuned Random Forest Accuracy:",
      accuracy_score(y_test, best_rf.predict(X_test)))

#----------------------------------------------------------
from sklearn.metrics import roc_auc_score #to calculate the ROC-AUC score for the tuned random forest model, 
#which measures the model's ability to distinguish between classes. The predicted probabilities for the positive class are obtained using the predict_proba method, and the ROC-AUC score is calculated using the roc_auc_score function from sklearn.metrics.

y_prob_rf = best_rf.predict_proba(X_test)[:,1]
roc_score = roc_auc_score(y_test, y_prob_rf)

print("Tuned Random Forest ROC-AUC:", roc_score)
#A confusion matrix is generated for the tuned random forest model using the confusion_matrix function from sklearn.metrics, 
# which compares the true labels (y_test) with the predicted labels obtained from the best_rf model.

#----------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix#to generate a confusion matrix for the tuned random forest model, 
#which compares the true labels (y_test) with the predicted labels obtained from the best_rf model. The confusion matrix is then visualized using a heatmap to show the counts of true positives, true negatives, false positives, and false negatives.

cm = confusion_matrix(y_test, best_rf.predict(X_test))
#The confusion matrix is visualized using a heatmap, where the counts of true positives, 
# true negatives, false positives, and false negatives are annotated on the plot. 
# The title and axis labels are set to indicate that this is a confusion matrix for the random forest model, 
# and the plot is displayed using plt.show().
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
#The ROC curve is plotted for the tuned random forest model using the false positive rates (fpr) 
# and true positive rates (tpr) obtained from the roc_curve function.
#--------------------------------------------------------------------

#-------------------------------------------------------------------------------
from sklearn.metrics import roc_curve#to plot the ROC curve for the tuned random forest model, which evaluates the model's performance in distinguishing between classes. 
#The false positive rates (fpr) and true positive rates (tpr) are obtained using the roc_curve function, and the ROC curve is plotted using matplotlib to visualize the trade-off between sensitivity
#  and specificity at various threshold settings.

fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()#The cross-validation scores for the tuned random forest model are calculated using the cross_val_score function from sklearn.
#model_selection, which evaluates the model's performance across different subsets of the data. 
# The mean cross-validation accuracy is also printed to provide an overall assessment of the model's stability and generalization ability.
#---------------------------------------------------------------------

#----------------------------------------------------------------------
from sklearn.model_selection import cross_val_score #to perform cross-validation on the tuned random forest model, which evaluates the model's performance across different subsets of the data. 
#The cross-validation scores are printed, along with the mean cross-validation accuracy,
#  to provide an overall assessment of the model's stability and generalization ability.

cv_scores = cross_val_score(best_rf, X, y, cv=5) #to perform cross-validation on the tuned random forest model, 
#which evaluates the model's performance across different subsets of the data.

# The cross-validation scores are printed, along with the mean cross-validation accuracy, 
# to provide an overall assessment of the model's stability and generalization ability.
print("Cross Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


#printing training and testing accuracy for the best random forest model obtained from the grid search,
#  which allows us to evaluate the model's performance on both the training and unseen test data.
print("Training Accuracy:",
      accuracy_score(y_train, best_rf.predict(X_train)))

print("Testing Accuracy:",
      accuracy_score(y_test, best_rf.predict(X_test)))
print("Standard Deviation of CV Accuracy:", cv_scores.std())
