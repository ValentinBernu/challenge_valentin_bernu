import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

# Get data
train_selected = pd.read_csv("train_selected_KW_Bonferroni.csv")
print(train_selected.shape)

# Normalization of columns
# for feature in train_selected.columns:
#     if feature != 'type':
#         train_selected[feature] = (train_selected[feature]-train_selected[feature].mean()) / \
#             train_selected[feature].std()

# Scaling of columns
for feature in train_selected.columns:
    if feature != 'type':
        train_selected[feature] = (train_selected[feature]-train_selected[feature].min()) / \
            (train_selected[feature].max()-train_selected[feature].min())

########################
# Classifier benchmark #
########################

# Label encode the features
train_selected.loc[train_selected["type"]
                   == "Ovarian_Tumor_Serous", "type"] = 1
train_selected.loc[train_selected["type"] ==
                   "Ovarian_Tumor_Endometrioid", "type"] = 2
train_selected.loc[train_selected["type"]
                   == "Ovarian_Tumor_ClearCel", "type"] = 3
train_selected.loc[train_selected["type"]
                   == "Ovarian_Tumor_Mucinous", "type"] = 4
train_selected.type = train_selected.type.astype('int')

# Models dict to benchmark
model_dict = {}
model_dict["Lo"] = LogisticRegression(
    multi_class="multinomial")
model_dict["RF"] = RandomForestClassifier()
model_dict["GNB"] = GaussianNB()
model_dict["SVM"] = svm.SVC()
model_dict["DS"] = DummyClassifier(strategy="stratified")
model_dict["DMF"] = DummyClassifier(strategy="most_frequent")


# Compute benchmark with cross validation and 10 iterations for each
N_ITER = 10
benchmark_df = pd.DataFrame()
for model_name, model in model_dict.items():
    res = []
    X, y = train_selected.drop(columns=["type"]), train_selected["type"]
    for i in range(N_ITER):
        clf = model.fit(X, y)
        res.append(np.mean(cross_val_score(
            clf, X, y, cv=3, scoring="accuracy")))
    benchmark_df[model_name] = res

# Plot the benchmark
sns.boxplot(data=benchmark_df)
plt.xlabel("Models")
plt.ylabel("Average cross validated accuracy")
plt.title("Accuracy benchmark of models, {} iterations".format(N_ITER))
plt.savefig("images/benchmarks/Benchmark {} iterations".format(N_ITER),
            bbox_inches="tight",
            )
