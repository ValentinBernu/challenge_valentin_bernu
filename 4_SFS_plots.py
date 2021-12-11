from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from numpy.core.numeric import normalize_axis_tuple
from scipy import stats
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

train_selected = pd.read_csv("train_selected_KW_Bonferroni.csv")
X, y = train_selected.drop(columns=["type"]), train_selected["type"]

# # Scaling of columns
# for feature in train_selected.columns:
#     if feature != 'type':
#         train_selected[feature] = (train_selected[feature]-train_selected[feature].min()) / \
#             (train_selected[feature].max()-train_selected[feature].min())


def plot_SFS(label, model, n_feat):
    """[Plot Sequential feature slection]
    X axis : Nb feature selected
    Y axis : Accuracy commputed with CV

    Args:
        model ([sklearn classifier]): [Model to select feature for]
        n_feat ([int]): [Number max of feature to select]
    """
    clf = model
    sfs = SFS(clf,
              k_features=n_feat,
              forward=True,
              floating=False,
              scoring='accuracy',
              cv=3)

    sfs = sfs.fit(X, y)
    fig1 = plot_sfs(sfs.get_metric_dict(),
                    kind='std_dev',
                    figsize=(6, 4))
    plt.ylim([0.8, 1])
    plt.title('{} - Sequential Forward Selection (w. StdDev)'.format(label))
    plt.grid()
    plt.savefig("images/SFS_plots/SFS with scaling, {}, {} features".format(label, n_feat),
                bbox_inches="tight",
                )
    plt.clf()


# Only plot SFS for the weakest learners
plot_SFS("SVM", svm.SVC(), 20)
plot_SFS("MNB", MultinomialNB(), 20)
