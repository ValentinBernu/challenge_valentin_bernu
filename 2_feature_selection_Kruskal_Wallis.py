from scipy import stats
import pandas as pd

train = pd.read_csv("train.csv")
train = train.drop(["Unnamed: 0", "samples"], axis=1)

#######################
# Kruskal Wallis test #
#######################

# Bonferroni correction treshold
bonferoni_treshold = 0.05 / len(train.columns)

cat1 = train.loc[train["type"] == "Ovarian_Tumor_Serous"]
cat2 = train.loc[train["type"] == "Ovarian_Tumor_Endometrioid"]
cat3 = train.loc[train["type"] == "Ovarian_Tumor_ClearCel"]
cat4 = train.loc[train["type"] == "Ovarian_Tumor_Mucinous"]

column_to_keep = []
for column in train.columns:
    if stats.kruskal(cat1[column], cat2[column], cat3[column], cat4[column]).pvalue \
            < bonferoni_treshold:  # Bonferoni correction yielded to much features
        column_to_keep.append(column)

train_selected = train[column_to_keep]
print("train shape = {}".format(train.shape))
print("train_selected shape = {}".format(train_selected.shape))

train_selected.to_csv("train_selected_KW_Bonferroni.csv", index=False)
