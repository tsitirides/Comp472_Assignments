import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.neural_network import MLPClassifier


# 4a
def train_base_dt(train_data_features, training_labels, test_data_features):
    # trains, predicts  (predict) and returns trained dt and predictions
    base_dt = DecisionTreeClassifier()
    base_dt.fit(train_data_features, training_labels)
    test_predictions = base_dt.predict(test_data_features)

    return base_dt, test_predictions


# 4b
def train_top_dt(train_data_features, training_labels, test_data_features):
    # sets hyperparameters to use, performs exhaustive search using gridsearch
    # grid search is then applied to training data to find the best model
    # gets the best dt model found, predictions and best parameters
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 9, None],
        'min_samples_split': [2, 8, 12]
    }
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid_search.fit(train_data_features, training_labels)
    top_dt = grid_search.best_estimator_
    test_predictions = top_dt.predict(test_data_features)
    return top_dt, test_predictions, grid_search.best_params_


# 4c
def train_base_mlp(train_data_features, training_labels, test_data_features):
    # takes training and testing data, fits the model with below params (rest are default)
    # then makes predictions on the data and returns the trained model base_mlp and the predictions test_predictions
    base_mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd')
    base_mlp.fit(train_data_features, training_labels)
    test_predictions = base_mlp.predict(test_data_features)
    return base_mlp, test_predictions


# 4d
def train_top_mlp(train_data_features, training_labels, test_data_features):
    # like base_mlp but uses the best hyperparameters
    # returns the best model, predictions and best parameters
    param_grid = {
        'activation': ['logistic', 'tanh', 'relu'],
        'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
        'solver': ['adam', 'sgd']
    }
    grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(train_data_features, training_labels)
    top_mlp = grid_search.best_estimator_
    test_predictions = top_mlp.predict(test_data_features)
    return top_mlp, test_predictions, grid_search.best_params_


# Abalone Dataset
def abalone_data_set():
    filename = "abalone-performance.txt"
    # Abalone.csv

    abalone = pd.read_csv('abalone.csv')
    original_categories = abalone['Type'].astype(
        'category').cat.categories.tolist()  # gives M,F,I with index so we can plot later

    # i. convert to hot vectors
    abalone_hot_vectors = pd.get_dummies(abalone, columns=['Type'])

    # ii. convert to categories
    abalone['Type'] = pd.Categorical(abalone['Type']).codes

    # 2 Plot

    sex_counts = abalone['Type'].value_counts(normalize=True) * 100
    sex_counts.plot(kind='bar')
    plt.title('Abalone Sex Distribution (%)')
    plt.xlabel('Sex')
    plt.ylabel('Percentage')
    plt.xticks(ticks=[0, 1, 2], labels=original_categories)
    plt.savefig('abalone-classes.png')  # gif doesnt work so png
    # plt.show()

    # 3 Split data set

    abalone_x = abalone.drop('Type', axis=1)  # or use abalone if using categories
    abalone_y = abalone['Type']

    train_abalone_x, test_abalone_x, train_abalone_y, test_abalone_y = train_test_split(abalone_x, abalone_y)
    # 4
    # a.
    base_dt_abalone, test_predictions_base_dt_abalone = train_base_dt(train_abalone_x, train_abalone_y, test_abalone_x)
    append_to_file(filename, "Base-DT", test_abalone_y, test_predictions_base_dt_abalone)

    plt.figure(figsize=(30, 10))
    plot_tree(base_dt_abalone, max_depth=4, filled=True, feature_names=train_abalone_x.columns,
              class_names=['F', 'I', 'M'])
    plt.show()

    # b.
    top_dt_abalone, test_predictions_top_dt_abalone, top_dt_params_abalone = train_top_dt(train_abalone_x,
                                                                                          train_abalone_y,
                                                                                          test_abalone_x)
    append_to_file(filename, "Top-DT", test_abalone_y, test_predictions_top_dt_abalone,
                   best_params=top_dt_params_abalone)

    plt.figure(figsize=(20, 50))
    plot_tree(top_dt_abalone, filled=True, feature_names=train_abalone_x.columns,
              class_names=['F', 'I', 'M'])
    plt.title('Penguins Top Decision Tree')
    plt.show()
    # c. Base MLP
    base_mlp_abalone, test_predictions_base_mlp_abalone = train_base_mlp(train_abalone_x, train_abalone_y,
                                                                         test_abalone_x)
    append_to_file(filename, "Base-MLP", test_abalone_y, test_predictions_base_mlp_abalone)

    # d. Top MLP
    top_mlp_abalone, test_predictions_top_mlp_abalone, top_mlp_params_abalone = train_top_mlp(train_abalone_x,
                                                                                              train_abalone_y,
                                                                                              test_abalone_x)
    append_to_file(filename, "Top-MLP", test_abalone_y, test_predictions_top_mlp_abalone,
                   best_params=top_mlp_params_abalone)

    # model_performance_evaluation(train_abalone_x, train_abalone_y, test_abalone_x, test_abalone_y, train_base_dt, "Base-DT", "performance_base_dt.txt")
    # model_performance_evaluation(train_abalone_x, train_abalone_y, test_abalone_x, test_abalone_y, train_top_dt, "Top-DT", "performance_top_dt.txt")
    # model_performance_evaluation(train_abalone_x, train_abalone_y, test_abalone_x, test_abalone_y, train_base_mlp, "Base-MLP", "performance_base_mlp.txt")
    # model_performance_evaluation(train_abalone_x, train_abalone_y, test_abalone_x, test_abalone_y, train_top_mlp, "Top-MLP", "performance_top_mlp.txt")


# ==============================================================================
# Q1 Loads Dataset for penguins, converts to 1-hot vectors and categories, plots percentages (Q2)
# splits dataset
# ==============================================================================
def penguins_data_set():
    filename = "penguin-performance.txt"
    # penguins.csv
    penguins = pd.read_csv('penguins.csv')
    # 1

    # i. convert to hot vectors
    penguins_hot_vectors = pd.get_dummies(penguins, columns=['island', 'sex'])

    # ii. convert to categories
    penguins['island'] = pd.Categorical(penguins['island']).codes
    penguins['sex'] = pd.Categorical(penguins['sex']).codes

    # 2 Plot

    species_counts = penguins['species'].value_counts(normalize=True) * 100
    species_counts.plot(kind='bar')
    plt.title('Penguin Species Distribution (%)')
    plt.xlabel('Species')
    plt.ylabel('Percentage')
    plt.savefig('penguin-classes.png')  # Saving as PNG because GIF is not directly supported
    # plt.show()

    penguins_x = penguins.drop('species', axis=1)
    penguins_y = penguins['species']

    train_data_features, test_data_features, training_labels, test_results = train_test_split(penguins_x, penguins_y)

    # test_results is the actual results, we will use it to compare later

    # 4

    # a. Base Decision Tree

    base_dt, test_predictions_base_dt = train_base_dt(train_data_features, training_labels, test_data_features)
    append_to_file(filename, "Base-DT", test_results, test_predictions_base_dt)

    plt.figure(figsize=(20, 50))
    plot_tree(base_dt, filled=True, feature_names=train_data_features.columns,
              class_names=training_labels.unique().tolist())
    plt.title('Penguins Base Decision Tree')
    plt.show()

    # b. Top Decision Tree

    top_dt, test_predictions_top_dt, top_dt_params = train_top_dt(train_data_features, training_labels,
                                                                  test_data_features)
    append_to_file(filename, "Top-DT", test_results, test_predictions_top_dt, best_params=top_dt_params)

    plt.figure(figsize=(20, 50))
    plot_tree(top_dt, filled=True, feature_names=train_data_features.columns,
              class_names=training_labels.unique().tolist())
    plt.title('Penguins Top Decision Tree')
    plt.show()

    # c. Base-MLP

    base_mlp, test_predictions_base_mlp = train_base_mlp(train_data_features, training_labels, test_data_features)
    append_to_file(filename, "Base-MLP", test_results, test_predictions_base_mlp)

    # d. Top-MLP

    top_mlp, test_predictions_top_mlp, top_mlp_params = train_top_mlp(train_data_features, training_labels,
                                                                      test_data_features)
    append_to_file(filename, "Top-MLP", test_results, test_predictions_top_mlp, best_params=top_mlp_params)


# commented because it almost blows up my pc when I run it
# model_performance_evaluation(train_data_features, training_labels, test_data_features, test_results, train_base_dt, "Base-DT", "performance_base_dt.txt")
# model_performance_evaluation(train_data_features, training_labels, test_data_features, test_results, train_top_dt, "Top-DT", "performance_top_dt.txt")
# model_performance_evaluation(train_data_features, training_labels, test_data_features, test_results, train_base_mlp, "Base-MLP", "performance_base_mlp.txt")
# model_performance_evaluation(train_data_features, training_labels, test_data_features, test_results, train_top_mlp, "Top-MLP", "performance_top_mlp.txt")


# ==============================================================================
# Q5: Appends information for each datasets respective models
# ==============================================================================
def append_to_file(filename, model_name, y_true, test_predictions, best_params=None):
    with open(filename, 'a') as f:
        # A. Model Description
        f.write("-----" * 10 + "\n")
        f.write(f"Model: {model_name}\n")
        if best_params:
            f.write(f"(A)\nBest Hyperparameters: {best_params}\n")
        else:
            f.write(f"(A)\nBest Hyperparameters: None\n")

        # B. Confusion Matrix
        cm = confusion_matrix(y_true, test_predictions)
        f.write("(B)\nConfusion Matrix:\n")
        f.write(str(cm))

        # C. Precision, Recall, F1-measure
        report = classification_report(y_true, test_predictions)
        f.write("\n(C)\nClassification Report:\n")
        f.write(report)

        # D. Accuracy, Macro F1 and Weighted F1
        accuracy = accuracy_score(y_true, test_predictions)
        macro_f1 = f1_score(y_true, test_predictions, average='macro')
        weighted_f1 = f1_score(y_true, test_predictions, average='weighted')
        f.write(f"(D)\nAccuracy: {accuracy:.4f}\n")
        f.write(f"Macro-average F1: {macro_f1:.4f}\n")
        f.write(f"Weighted-average F1: {weighted_f1:.4f}\n\n")


# =============================================================
# Q6: Run model 5 times and prints results for each
# =============================================================

def model_performance_evaluation(train_data_features, training_labels, test_data_features, test_results, model,
                                 model_name, filename):
    accuracies = []
    macro_avg = []
    weighted_avg = []

    # run 5 times
    for i in range(5):
        results = model(train_data_features, training_labels, test_data_features)
        model, test_predictions = results[:2]  # we don't need best params for this part

        # evaluate current model
        accuracy = accuracy_score(test_results, test_predictions)
        macro_f1 = f1_score(test_results, test_predictions, average='macro')
        weighted_f1 = f1_score(test_results, test_predictions, average='weighted')

        # append each score to respective list
        accuracies.append(accuracy)
        macro_avg.append(macro_f1)
        weighted_avg.append(weighted_f1)

    # average and variance
    accuracy_avg = np.mean(accuracies)
    accuracy_var = np.var(accuracies)

    macro_f1_avg = np.mean(macro_avg)
    macro_f1_var = np.var(macro_avg)

    weighted_f1_avg = np.mean(weighted_avg)
    weighted_f1_var = np.var(weighted_avg)

    # create performance file
    with open(filename, 'a') as f:
        f.write(f"Dataset: {filename}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Average Accuracy: {accuracy_avg:.4f}, Variance: {accuracy_var:.4f}\n")
        f.write(f"Average Macro F1: {macro_f1_avg:.4f}, Variance: {macro_f1_var:.4f}\n")
        f.write(f"Average Weighted F1: {weighted_f1_avg:.4f}, Variance: {weighted_f1_var:.4f}\n")
        f.write("-----" * 10 + "\n\n")


if __name__ == '__main__':
    penguins_data_set()
    abalone_data_set()
