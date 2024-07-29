import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from stepwise_regression import stepwise_regression
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score
from scipy import stats

seed = 50

path = 'winequality-white.csv'

data_df = pd.read_csv(path,header=0, delimiter=';')
data_df = data_df.drop('density', axis = 1)

quality_count = data_df['quality'].value_counts()
print(quality_count)

# found it difficult to get accurate predictions when there were so many classes as they were likely to be placed in a class close to it
# changed the quality metric to have only 3 classes: low (1), medium (2), and high (3)
# data_df['quality'] = data_df['quality'].apply(lambda x: 0 if x in [3, 4, 5] else (1 if x in [6] else 2))
data_df['quality'] = data_df['quality'].apply(lambda x: 0 if x in [3, 4, 5] else 1)
# data_df['fixed acidity'] = np.log(data_df['fixed acidity'])
# data_df['quality'] = data_df['quality'].apply(lambda x: x - 3) # subtracting 3 so classes are 0 - 6 rather than 3 - 9; change back later
# print(data_df)
# data_df = data_df.drop('quality', axis = 1)

# Define a function to remove outliers for each class
def remove_outliers_per_class(df, class_column, predictors, threshold=3):
    cleaned_data = []
    for class_label, group in df.groupby(class_column):
        for predictor in predictors:
            z_scores = stats.zscore(group[predictor])
            outliers = group[(z_scores > threshold) | (z_scores < -threshold)]
            group = group.drop(outliers.index)
        cleaned_data.append(group)
    return pd.concat(cleaned_data)

# X_headers = data_df.columns[0:11]
X_headers = data_df.columns[0:10] # because I removed a column: density

# Define the list of predictor variables
predictors = X_headers

# Remove outliers for each class separately
data_df = remove_outliers_per_class(data_df, 'quality', predictors)
# Check the cleaned DataFrame
# print(cleaned_df)

data = data_df.to_numpy()

# print('data', data.shape)

# X = data[:, 0:11]
# y = data[:, 11]
X = data[:, 0:10] # because I removed a column: density
y = data[:, 10] # because I removed a column: density

X = StandardScaler().fit_transform(X)

# PCA decreases accuracy so not using it
# # Initialize PCA with desired number of components
# pca = PCA(n_components=10)
# # Fit PCA model to data
# pca.fit(X)
# # Transform data to new feature space
# X = pca.transform(X)

# X_df = data_df.iloc[:,0:10]
# y_df = data_df.iloc[:,10]

# def stepwise_selection(X_df, y_df, initial_list=[], threshold_in=0.01, threshold_out = 0.05, verbose=True):
#     included = list(initial_list)
#     while True:
#         changed=False
#         # forward step
#         excluded = list(set(X_df.columns)-set(included))
#         new_pval = pd.Series(index=excluded)
#         for new_column in excluded:
#             model = sm.OLS(y, sm.add_constant(pd.DataFrame(X_df[included+[new_column]]))).fit()
#             new_pval[new_column] = model.pvalues[new_column]
#         best_pval = new_pval.min()
#         if best_pval < threshold_in:
#             best_feature = new_pval.idxmin()
#             included.append(best_feature)
#             changed=True
#             if verbose:
#                 print('Add  {:30} p-value {:.6}'.format(best_feature, best_pval))

#         # backward step
#         model = sm.OLS(y_df, sm.add_constant(pd.DataFrame(X_df[included]))).fit()
#         # use all coefs except intercept
#         pvalues = model.pvalues.iloc[1:]
#         worst_pval = pvalues.max() # null if pvalues is empty
#         if worst_pval > threshold_out:
#             changed=True
#             worst_feature = pvalues.argmax()
#             included.remove(worst_feature)
#             if verbose:
#                 print('Drop {:30} p-value {:.6}'.format(worst_feature, worst_pval))
#         if not changed:
#             break
#     return included

# # Usage example
# selected_features = stepwise_selection(X_df, y_df)
# # Print the selected features
# print("Selected features:", selected_features)
# selected features are: ['alcohol', 'volatile acidity', 'residual sugar', 'pH', 'chlorides', 'free sulfur dioxide', 'sulphates', 'citric acid']

# include only the chosen features in X
# X = data_df.loc[:, ['alcohol', 'volatile acidity', 'residual sugar', 'free sulfur dioxide', 'fixed acidity', 'sulphates']].to_numpy()
X_headers = data_df.columns[0:10]

# Exploratory Data Analysis
# fixed_acidity = data[:,0]
# volatile_acidity = data[:,1]
# citric_acidity = data[:,2]
# residual_sugar = data[:,3]
# chlorides = data[:,4]
# free_sulfur_dioxide = data[:,5]
# total_sulfur_dioxide = data[:,6]
# pH = data[:,7]
# sulphates = data[:,8]
# alcohol = data[:,9]
# quality = data[:,10]

# add_features = fixed_acidity+volatile_acidity+citric_acidity+residual_sugar+chlorides+total_sulfur_dioxide

# import matplotlib.pyplot as plt

# # Assuming your DataFrame is named 'data_df'
# plt.figure(figsize=(10, 6))
# plt.scatter(quality, alcohol, color='blue', alpha=0.5)
# plt.title('Fixed Acidity vs Quality')
# plt.xlabel('Fixed Acidity')
# plt.ylabel('Quality')
# plt.grid(True)
# plt.show()

# print(np.unique(quality))

# summary
# summary = data_df.describe()
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# print(summary)

# histogram
# plt.hist(quality, bins=7, color='green', edgecolor='black')
# plt.xlabel('Quality')
# plt.ylabel('Count')
# plt.title('Histogram of Response Variable: Quality')
# plt.show()

# correlation plot
# corr = data_df.corr()
# plt.figure(figsize=(8, 8))
# ax = sns.heatmap(corr, annot=True, cmap='seismic', fmt=".3f", annot_kws={"size": 10})  # Adjust the fontsize here
# ax.figure.tight_layout()
# # Adjusting the size of the axes and text
# plt.tick_params(axis='both', which='major', labelsize=10)
# plt.title("Correlation Heatmap Across All Variables", fontsize=14)
# plt.show()

# VIF test - used to test for multicollinearity between predictors
# vif = pd.DataFrame()
# vif['Feature'] = X_headers
# vif['VIF'] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
# print(vif)

# high vif: residual sugar 12.644064; density 28.232546; alcohol 7.706957
# both residual sugar and alcohol are highly correlated with density (according to heat map) so I tried removing density
# no  high VIF's after I remove density; therefore, I will remove density for my analysis

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

smote = SMOTE(sampling_strategy='auto', random_state=seed)
X_train, y_train = smote.fit_resample(X_train, y_train)

unique_values, counts = np.unique(y_train, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"{value}: {count}")

########################################### K Fold CV #####################################################################################

from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Define the models with their respective parameter grids for tuning
# models = {
#     "KNN": (KNeighborsClassifier(), {"n_neighbors": np.arange(1, 21)}),
#     "Linear SVM": (SVC(kernel="linear"), {"C": np.logspace(-3, 3, 7)}),
#     "RBF SVM": (SVC(kernel="rbf"), {"C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)}),
#     "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100, 150, 200, 250]}),
#     "Gradient Boosting Machine": (GradientBoostingClassifier(), {"n_estimators": [50, 100, 150, 200, 250]}),
#     "Naive Bayes": (GaussianNB(), {}),
#     "Neural Networks": (MLPClassifier(max_iter=1000), {"hidden_layer_sizes": [(50,), (100,), (150,)], "alpha": np.logspace(-5, 0, 6)}),
#     "XGBoost": (xgb.XGBClassifier(), {"n_estimators": [50, 100, 150, 200, 250]})
# }

# models = {
#     "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100, 150, 200, 250], "max_depth": [None, 5, 10, 15]}),
#     "Gradient Boosting Machine": (GradientBoostingClassifier(), {"n_estimators": [50, 100, 150, 200, 250], "learning_rate": [0.01, 0.1, 0.5], "max_depth": [3, 5, 7]}),
#     "XGBoost": (xgb.XGBClassifier(), {"n_estimators": [50, 100, 150, 200, 250], "learning_rate": [0.01, 0.1, 0.5], "max_depth": [3, 5, 7], "reg_alpha": [0, 0.1, 1], "reg_lambda": [0, 0.1, 1]})
# }

# {'Random Forest': {'max_depth': None, 'n_estimators': 150}, 'Gradient Boosting Machine': {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 100}, 'XGBoost': {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 150, 'reg_alpha': 0, 'reg_lambda': 0.1}}

# # # final models
models = {
    "Multiple Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "KNN": KNeighborsClassifier(3),
    "Linear SVM": SVC(kernel="linear",C=10.0, random_state=seed),
    "RBF SVM": SVC(kernel="rbf", C=10, gamma=10, random_state=seed),
    "Random Forest": RandomForestClassifier(max_depth=None, n_estimators=150, random_state=seed),
    "Gradient Boosting Machine": GradientBoostingClassifier(learning_rate=0.1, max_depth=7, n_estimators=100, random_state=seed),
    "Naive Bayes": GaussianNB(),
    "Neural Networks": MLPClassifier(max_iter=1000, alpha=0.0001, hidden_layer_sizes=(150,), random_state=seed),
    "XGBoost": xgb.XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=150, reg_alpha=0, reg_lambda=0.1, random_state=seed),
}

# Perform k-fold cross-validation for parameter tuning
# best_params = {}
# for model_name, (model, param_grid) in models.items():
#     print(f"Performing parameter tuning for {model_name}...")
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#     # need to balance the training data set (it's okay if test data is not balanced)
#     # smote = SMOTE(sampling_strategy='auto')
#     # X_train, y_train = smote.fit_resample(X_train, y_train)
#     clf = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
#     clf.fit(X_train, y_train)
#     best_params[model_name] = clf.best_params_


# print(best_params)
# Perform Monte Carlo cross-validation for each model using best parameters
# num_iterations = 1
# results = []
# for i in range(num_iterations):
#     print(f"Iteration {i + 1}/{num_iterations}")
#     for model_name, model in models.items():
#         print(f"Evaluating {model_name}...")
#         clf = model
#         clf.fit(X_train, y_train)
#         if model_name == "Random Forest":
#             feature_importances = clf.feature_importances_
#             feature_importance_dict = dict(zip(X_headers, feature_importances))
#             sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
#             for feature, importance in sorted_feature_importance:
#                 print(f"Feature: {feature}, Importance: {importance}")
#         test_accuracy = clf.score(X_test, y_test)
#         if hasattr(clf, 'predict_proba'):  # Check if the model has predict_proba method
#             y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of positive class for AUC-ROC
#             auc_roc = roc_auc_score(y_test, y_proba)
#         else:
#             auc_roc = None
#         results.append({'Model': model_name, 'Accuracy': test_accuracy, 'AUC-ROC': auc_roc})

# # Create a DataFrame to store results
# results_df = pd.DataFrame(results)

# # Print the DataFrame
# print(results_df)


# #######################################################################################################################################


# #################################### try combining the three best models using stacking ###############################################

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np

# After already doing parameter tuning
rf_models = [RandomForestClassifier(max_depth=20, n_estimators=250, random_state=42),
             RandomForestClassifier(max_depth=20, n_estimators=200, random_state=50),
             RandomForestClassifier(max_depth=20, n_estimators=200, random_state=58)]

xgb_models = [XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=250, random_state=42),
              XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=250, random_state=50),
              XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=250, random_state=58)]

# Define meta-model
meta_model = XGBClassifier(random_state=42, learning_rate=0.1, max_depth=3, reg_alpha=0.1, reg_lambda=0.1)

# Create stacking classifier
stacking_model = StackingClassifier(
    estimators=[('rf{}'.format(model.random_state), model) for model in rf_models] +
               [('xgb{}'.format(model.random_state), model) for model in xgb_models],
    final_estimator=meta_model
)

# Fit the stacking model with updated base models
stacking_model.fit(X_train, y_train)

# Evaluate the stacking model
accuracy_stacking = stacking_model.score(X_test, y_test)
print("Stacking Model Accuracy:", accuracy_stacking)



# ########################################################################################################################

# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier
# import numpy as np

# # After already doing parameter tuning
# rf_models = [RandomForestClassifier(max_depth=20, n_estimators=250, random_state=43),
#              RandomForestClassifier(max_depth=20, n_estimators=200, random_state=51),
#              RandomForestClassifier(max_depth=20, n_estimators=200, random_state=59)]

# xgb_models = [XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=250, random_state=43),
#               XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=250, random_state=51),
#               XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=250, random_state=59)]

# # Define meta-model
# meta_model = XGBClassifier(random_state=42, learning_rate=0.1, max_depth=3, reg_alpha=0.1, reg_lambda=0.1)

# # Create voting classifier for blending
# blending_model = VotingClassifier(
#     estimators=[('rf{}'.format(model.random_state), model) for model in rf_models] +
#                [('xgb{}'.format(model.random_state), model) for model in xgb_models],
#     voting='soft'
# )

# # Fit the blending model
# blending_model.fit(X_train, y_train)

# # Evaluate the blending model
# accuracy_blending = blending_model.score(X_test, y_test)
# print("Blending Model Accuracy:", accuracy_blending)

# ###########################################################################################################

# # Combine stacking and blending into a super learner
# super_learner = VotingClassifier(
#     estimators=[('stacking', stacking_model), ('blending', blending_model)],
#     voting='soft'
# )

# # Fit the super learner model
# super_learner.fit(X_train, y_train)

# # Evaluate the super learner model
# accuracy_super_learner = super_learner.score(X_test, y_test)
# print("Super Learner Model Accuracy:", accuracy_super_learner)

###########################################################################################################
names = ["RF", "XGB", "Stacking"]

classifiers = [
    RandomForestClassifier(max_depth=None, n_estimators=150, random_state=seed),
    xgb.XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=150, reg_alpha=0, reg_lambda=0.1, random_state=seed),
    StackingClassifier(estimators=[('rf{}'.format(model.random_state), model) for model in rf_models] +
               [('xgb{}'.format(model.random_state), model) for model in xgb_models],final_estimator=meta_model)
]

# iterate over classifiers
results_small = []
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    auc_roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    class_report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    digits = ['0', '1']
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(confusion, annot=True, fmt='d', cmap="viridis", xticklabels=digits, yticklabels=digits, annot_kws={"size": 10})  # Adjust the fontsize here
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title(name + ' Confusion Matrix', fontsize=14)  # Adjust the fontsize here
    plt.show()
    results_small.append({'model': name, 'accuracy': score, 'AUC ROC': auc_roc})
    print(confusion)
    print(name)
    print(class_report)

df_small = pd.DataFrame(results_small)
print('Accuracy and AUC ROC by Classifier \n', df_small)

# try combining models to see where you can get improved performance

# clustering to see if there are patterns that are not obvious