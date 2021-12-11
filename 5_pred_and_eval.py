import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.naive_bayes import GaussianNB
import pandas as pd
train_selected = pd.read_csv("train_selected_KW_Bonferroni.csv")
test = pd.read_csv("test.csv")

####################################
# Apply preprocessing to test data #
####################################
# The scaling does not change Gaussian NB results

#############
# Apply SFS #
#############
N_feature = 10
model = svm.SVC()

# Train & test data
X_train, y_train = train_selected.drop(
    columns=["type"]), train_selected["type"]
X_test, y_test = test.drop(
    columns=["type"]), test["type"]

sfs = SFS(model, k_features=N_feature, forward=True,
          floating=False, scoring='accuracy', cv=3)
sfs.fit(X_train, y_train)
X_train_SFS = sfs.transform(X_train)
#X_test_SFS = sfs.transform(X_test)
selected_columns = list(sfs.k_feature_names_)
X_test_SFS = X_test[selected_columns]

#############
# Fit model #
#############

model = svm.SVC()
model.fit(X_train_SFS, y_train)
y_prediction = model.predict(X_test_SFS)

##########################
# Print and save results #
##########################

print(accuracy_score(y_test, y_prediction))
# Overfit par SFS donc attendu qu'on perde un peu, c'est raisonnable
pred_df = pd.DataFrame()
pred_df["type"] = y_prediction
pred_df.to_csv("predictions_test_data.csv", index=False)


######################
# Confusion matrices #
######################

# test data
plot_confusion_matrix(model, X_test_SFS, y_test)
plt.savefig("images/confusion_matrices/confusion_matrix_test",
            bbox_inches="tight",
            )
# train data
plot_confusion_matrix(model, X_train_SFS, y_train)
plt.savefig("images/confusion_matrices/confusion_matrix_train",
            bbox_inches="tight",
            )
