from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pandas as pd

#########################
# Tumor type processing #
#########################
# Get df from csv
train = pd.read_csv("train.csv")

# One hot encode tumors
train_ohe = pd.get_dummies(train, columns=["type"])

# Rename one hot encoded columns to simpler names
tumor_columns_list = train_ohe.columns[-4:]
tumor_name_list = [e[19:] for e in tumor_columns_list]
for i, column in enumerate(tumor_columns_list):
    train_ohe = train_ohe.rename(columns={column: tumor_name_list[i]})

# Update the list of tumor type names
tumor_columns_list = tumor_name_list

# Drop unamed and sample columns
train_ohe = train_ohe.drop(["Unnamed: 0", "samples"], axis=1)
train_ohe["type"] = train["type"]

######################
# Tumor distribution #
######################
nb_Serous = len(train_ohe.loc[train_ohe["Serous"] == 1])
nb_Mucinous = len(train_ohe.loc[train_ohe["Mucinous"] == 1])
nb_Endometrioid = len(train_ohe.loc[train_ohe["Endometrioid"] == 1])
nb_ClearCel = len(train_ohe.loc[train_ohe["ClearCel"] == 1])
n_tot = nb_Serous + nb_ClearCel + nb_Endometrioid + nb_Mucinous

print("Nb {} = {}".format("Serous", nb_Serous))
print("Nb {} = {}".format("Mucinous", nb_Mucinous))
print("Nb {} = {}".format("Endometrioid", nb_Endometrioid))
print("Nb {} = {}".format("ClearCel", nb_ClearCel))
print("Nb {} = {}".format("tot", n_tot))

#################################
# Gene expression distributions #
#################################
# Plot 10 random histograms & save in a file
# Plot with a smoothing to observe if the distribution is Gaussian
columns = train_ohe.columns
for i in range(10):
    column = random.choice(columns)
    plt.title("Histogram of: {} ".format(column))
    sns.histplot(train_ohe, stat="count", x=column, kde=True)
    plt.savefig("images/random_hists/{} hist".format(column),
                bbox_inches="tight",
                )
    plt.clf()

#############################
# Are the features normal ? #
#############################

# get pvalues
p_values = []
for i in range(2000):
    column = random.choice(columns)
    if column != "type":
        p_values.append(stats.shapiro(train_ohe[column]).pvalue)

# Plot the p values
sns.histplot(p_values, stat="count", bins=100)
#plt.axvline(x=0.05, c='r')
plt.xlabel("pvalues")
plt.title("Histogram of pvalues of Shapiro test (rd sample)")
plt.savefig("images/pvalues_shapiro/pvalues_shapiro",
            bbox_inches="tight")

# Apply FDR correction to pvalues
pval_FDR_corrected = fdrcorrection(p_values)

# Print % of gaussian distribution
list_bool = pval_FDR_corrected[0].tolist()
percentage_gaussian = len([e for e in list_bool if e == True]) / len(list_bool)
print("percentage_gaussian features = {}".format(percentage_gaussian))

#####################################
# Plot correlations of the features #
#####################################
# Loop to test 10 features randomly 10 times
# to accounts for the very high number of columns
for i in range(10):
    # Keep OHE tumor types in each matrices
    columns_selected = tumor_columns_list[:]
    # Add 10 randomly selected columns
    columns_selected += random.choices(train_ohe.columns, k=100)
    selected_train_ohe = train_ohe[columns_selected]
    plt.figure(figsize=(30, 20))
    sns.heatmap(
        selected_train_ohe.corr().round(1),
        vmin=-1,
        vmax=1,
        annot=False,
        cmap="coolwarm",
    )
    plt.title("Pearson correlation matrice (rd sample)", size=40)
    plt.savefig("images/correlation_matrices/correlation_matrices_{}".format(i),
                bbox_inches="tight")
    plt.clf()
