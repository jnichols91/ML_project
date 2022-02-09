#!/usr/bin/env python3

# Written by Justin Nichols

import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# These are the different algorithms we'll be using.
from surprise import NMF, SVD, SVDpp, KNNBasic, KNNWithMeans, KNNWithZScore, CoClustering
# These are higher level functions for cross-validation.
from surprise.model_selection import cross_validate, GridSearchCV
# These are for data reading and formatting.
from surprise import Reader, Dataset
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QRadioButton, QHBoxLayout,
                             QFormLayout, QGroupBox, QPushButton)
from PyQt5 import QtCore
from random import sample

####------------User Interface-----------####

# finds the button that is checked by the user
def find_checked_radiobutton(radiobuttons):
    for button in radiobuttons:
        if button.isChecked():
            checked_radiobutton = button.text()
            return checked_radiobutton

def get_movies(movie_list, n):
    ixMovies = sample(range(movie_list.shape[0]), n)
    return movie_list.iloc[ixMovies, :]

def createWindow(movies, label):

    app = QApplication([])
    window = QWidget()
    window.setWindowTitle('Movie Ratings')

    layout = QFormLayout()
    group_list = []
    for i in range(movies.shape[0]):
        group = QGroupBox()
        movie = QHBoxLayout()
        button = QRadioButton('None')
        button.setChecked(True)
        movie.addWidget(button)
        for j in range(5):
            movie.addWidget(QRadioButton(str(j+0.5)))
            movie.addWidget(QRadioButton(str(j+1)))
        group.setLayout(movie)
        layout.addRow(movies.iloc[i, 1], group)
        layout.setLabelAlignment(QtCore.Qt.AlignCenter)
        group_list.append(group)
    next_button = QPushButton(label)
    layout.addRow('', next_button)
    window.setLayout(layout)
    window.show()
    next_button.clicked.connect(lambda:window.close())
    app.exec_()

    return [find_checked_radiobutton(group.findChildren(QRadioButton)) for group in group_list]

def get_user_ratings(data, final_movies):
    # read in available movies
    num_movies = round(len(data['movieId'].unique())/3)
    movies_list = final_movies[final_movies['movieId'].isin(data['movieId'])]
    user_movies = get_movies(movies_list, num_movies)
    user_ratings = createWindow(user_movies.iloc[:int(num_movies/2), :], 'Next Page ---->')
    user_ratings = user_ratings +  createWindow(user_movies.iloc[int(num_movies/2):, :], 'Get Your Recommendations!')

    full_user = pd.DataFrame({'movieId' : user_movies['movieId'],
                              'rating' : user_ratings})
    return full_user

def print_predictions(results, label):

    app = QApplication([])
    window = QWidget()
    window.setWindowTitle('Your Predicted Movies by Rating')

    layout = QFormLayout()
    for i in range(results.shape[0]):
        label1 = QLabel(str(results.iloc[i, 1]))
        label2 = QLabel(str(round(results.iloc[i, 2], 2)))
        layout.addRow(label1, label2)
        layout.setLabelAlignment(QtCore.Qt.AlignRight)
    next_button = QPushButton(label)
    layout.addRow('', next_button)
    window.setLayout(layout)
    window.show()
    next_button.clicked.connect(lambda:window.close())
    app.exec_()

####-------------------------------------####


####----------Cross-Validation-----------####
    # 1. Perform cross-validation for parameter selection.
    # 2. Construct model with optimal parameters.
    # 3. Record cross-validated estimates and standard errors for RMSE and MAE.

def cv(data, method, param_grid, folds = 5):

  # Grid-search cross-validation.
  start = time.time()
  cv = GridSearchCV(method, param_grid, measures=['rmse', 'mae'], cv=folds)
  cv.fit(data)

  # Construct optimal model and record optimal parameters.
  model = cv.best_estimator['rmse']
  params = cv.best_params['rmse']

  # Grab RMSE and MAE estimates and standard errors.
  index = np.argmin(cv.cv_results['mean_test_rmse'])
  cv_rmse = {
      'est' : cv.cv_results['mean_test_rmse'][index],
      'se'  : cv.cv_results['std_test_rmse'][index]
  }
  cv_mae = {
      'est' : cv.cv_results['mean_test_mae'][index],
      'se'  : cv.cv_results['std_test_mae'][index]
  }
  vals = [i for t in [(x['est'], x['se']) for x in [cv_rmse, cv_mae]] for i in t]
  finish = time.time()

  results = {
      'results' : cv.cv_results,
      'model'   : model,
      'params'  : params,
      'cv_rmse' : cv_rmse,
      'cv_mae'  : cv_mae,
      'vals'    : vals,
      'time'    : finish - start
  }

  return results

####-----------------------------------####


####--------Different Algorithms--------####

def knn(data, results):
    #### KNNBasic ####
    knn_param_grid = {
        'k': [k for k in range(1, 41)], # Default = 40
        'verbose' : [False] # Not really a tuning parameter, but leave it False.
    }
    knn_results = cv(data, KNNBasic, knn_param_grid, folds = 5)
    results.loc['KNN', :'MAE se'] = knn_results['vals']
    results.loc['KNN', 'time'] = knn_results['time']

    ## Plot RMSE vs. k with 1SE error bars.
    plt.figure()
    plt.errorbar(knn_results['results']['param_k'], knn_results['results']['mean_test_rmse'],
                 yerr = knn_results['results']['std_test_rmse'], color='steelblue',
                 ecolor='red', elinewidth=0.5)
    plt.plot(knn_results['params']['k'], np.min(knn_results['results']['mean_test_rmse']), 'bo')
    plt.xlabel('k')
    plt.ylabel('RMSE')
    plt.title('KNNBasic Parameter Tuning')
    plt.ion()
    plt.pause(0.01)
    plt.show()

    knn_results['params']
    return knn_results

def knn_means(data, results):
    #### KNNWithMeans ####
    knnmeans_param_grid = {
        'k': [k for k in range(1, 41)], # Default = 40
        'verbose' : [False] # Not really a tuning parameter, but leave it False.
    }
    knnmeans_results = cv(data, KNNWithMeans, knnmeans_param_grid, folds = 5)
    results.loc['KNNMeans', :'MAE se'] = knnmeans_results['vals']
    results.loc['KNNMeans', 'time'] = knnmeans_results['time']

    ## Plot RMSE vs. k with 1SE error bars.
    plt.figure()
    plt.errorbar(knnmeans_results['results']['param_k'], knnmeans_results['results']['mean_test_rmse'],
                 yerr = knnmeans_results['results']['std_test_rmse'], color='steelblue',
                 ecolor='red', elinewidth=0.5)
    plt.plot(knnmeans_results['params']['k'], np.min(knnmeans_results['results']['mean_test_rmse']), 'bo')
    plt.xlabel('k')
    plt.ylabel('RMSE')
    plt.title('KNNWithMeans Parameter Tuning')
    plt.ion()
    plt.pause(0.01)
    plt.show()

    knnmeans_results['params']
    return knnmeans_results

def knn_zscore(data, results):
    #### KNNWithZScore ####
    knnzscore_param_grid = {
        'k': [k for k in range(1, 41)], # Default = 40
        'verbose' : [False] # Not really a tuning parameter, but leave it False.
    }
    knnzscore_results = cv(data, KNNWithZScore, knnzscore_param_grid, folds = 5)
    results.loc['KNNZScore', :'MAE se'] = knnzscore_results['vals']
    results.loc['KNNZScore', 'time'] = knnzscore_results['time']

    ## Plot RMSE vs. k with 1SE error bars.
    plt.figure()
    plt.errorbar(knnzscore_results['results']['param_k'], knnzscore_results['results']['mean_test_rmse'],
                  yerr = knnzscore_results['results']['std_test_rmse'], color='steelblue',
                  ecolor='red', elinewidth=0.5)
    plt.plot(knnzscore_results['params']['k'], np.min(knnzscore_results['results']['mean_test_rmse']), 'bo')
    plt.xlabel('k')
    plt.ylabel('RMSE')
    plt.title('KNNWithZScore Parameter Tuning')
    plt.ion()
    plt.pause(0.01)
    plt.show()

    knnzscore_results['params']
    return knnzscore_results

def svd(data, results):
    #### SVD ####
    svd_param_grid = {
        'n_epochs': [20, 40, 60],        # Default = 20
        'lr_all': [0.010, 0.015, 0.020], # Default = 0.005
        'reg_all': [0.05, 0.10, 0.20]    # Default = 0.02
    }
    svd_results = cv(data, SVD, svd_param_grid, folds = 5)
    results.loc['SVD', :'MAE se'] = svd_results['vals']
    results.loc['SVD', 'time'] = svd_results['time']

    svd_results['params']
    return svd_results

def svdpp(data, results):
    #### SVDpp ####
    svdpp_param_grid = {
        'n_epochs': [20, 40],     # Default = 20
        'lr_all': [0.015, 0.020], # Default = 0.007
        'reg_all': [0.10, 0.20]   # Default = 0.02
    }
    svdpp_results = cv(data, SVDpp, svdpp_param_grid, folds = 5)
    results.loc['SVDpp', :'MAE se'] = svdpp_results['vals']
    results.loc['SVDpp', 'time'] = svdpp_results['time']

    svdpp_results['params']
    return svdpp_results

def nmf(data, results):
    #### NMF ####
    nmf_param_grid = {
        'n_epochs': [50, 60, 70],   # Default = 50
        'reg_pu': [0.03, 0.06, 0.09], # Default = 0.06
        'reg_qi': [0.03, 0.06, 0.09]    # Default = 0.06
    }
    nmf_results = cv(data, NMF, nmf_param_grid, folds = 5)
    results.loc['NMF', :'MAE se'] = nmf_results['vals']
    results.loc['NMF', 'time'] = nmf_results['time']

    nmf_results['params']
    return nmf_results

def clust(data, results):
    #### CoClustering ####
    clust_param_grid = {
        'n_epochs' : [20, 40, 60],             # Default = 20
        'n_cltr_u' : [i for i in range(1, 7)], # Default = 3
        'n_cltr_i' : [i for i in range(1, 7)]  # Default = 3
    }
    clust_results = cv(data, CoClustering, clust_param_grid, folds = 5)
    results.loc['CoClust', :'MAE se'] = clust_results['vals']
    results.loc['CoClust', 'time'] = clust_results['time']

    clust_results['params']
    return clust_results

####-----------------------------------####


####----------Grid Search Analysis-----------####

def analysis(test_data):
    ## Table of results.

    reader = Reader(rating_scale = (0.5, 5.0))
    data = Dataset.load_from_df(test_data, reader)
    results = pd.DataFrame(
        None,
        ['SVD', 'SVDpp', 'NMF', 'KNN', 'KNNMeans', 'KNNZScore', 'CoClust'],
        ['MSE est', 'MSE se', 'MAE est', 'MAE se', 'time']
    )
    print(results)
    knn_results = knn(data, results)
    print(results)
    knnmeans_results = knn_means(data, results)
    print(results)
    knnzscore_results = knn_zscore(data, results)
    print(results)
    svd_results = svd(data, results)
    print(results)
    svdpp_results = svdpp(data, results)
    print(results)
    nmf_results = nmf(data, results)
    print(results)
    clust_results = clust(data, results)
    print (results)

    results.sort_values(by = ['MSE est'], inplace = True)
    plt.figure()
    plt.errorbar(list(results.index), results['MSE est'], yerr = results['MSE se'],
                 marker = 'o', color = 'steelblue', linestyle = '', ecolor = 'red',
                 elinewidth = 0.5)
    plt.xlabel('Method')
    plt.ylabel('Estimated RMSE')
    plt.title('Final Results')
    plt.ion()
    plt.pause(0.01)
    plt.show()

    return results

####-----------------------------------####


####----------Predict User Ratings---------####

def predict_user(combined_ratings, new_userId, user_iids):
    ## Data formatting.
    # The data has to be of a specific type to be usable with the `surprise` library.
    reader = Reader(rating_scale = (0.5, 5.0))
    data = Dataset.load_from_df(combined_ratings, reader)

    unique_ids = combined_ratings['itemID'].unique()

    movies_to_predict = np.setdiff1d(unique_ids, user_iids)

    knn_fit = KNNWithZScore(k=23).fit(data.build_full_trainset())

    user_recs = []
    for iid in movies_to_predict:
        user_recs.append((iid, knn_fit.predict(uid=new_userId, iid=iid).est))

    user_predictions = pd.DataFrame(user_recs,
                                    columns=['movieId', 'prediction']).sort_values('prediction', ascending=False)

    return user_predictions

####-----------------------------------####


def main():
    ## Read in data.
    full_data = pd.read_csv('../data/filtered_data.csv')
    final_movies = pd.read_csv('../data/final_movies.csv')

    # For now, we're just going to look at a small subset of our data.
    users = pd.unique(full_data['userId'])
    new_userId = users.max() + 1
    random.seed(1)
    ix = random.sample(list(users), 6000)
    test_data = full_data[full_data['userId'].isin(ix)]

    user_ratings = get_user_ratings(test_data, final_movies)
    user_iids = user_ratings['movieId']
    user_ratings.drop(user_ratings[user_ratings['rating'] == 'None'].index, inplace = True)
    user_ratings['userId'] = new_userId
    user_ratings = user_ratings.reindex(columns=test_data.columns)

    print('Number of ratings in test data : {:,}'.format(test_data.shape[0]))

    combined_ratings = pd.concat([test_data, user_ratings], axis=0)
    combined_ratings.columns = ['userID', 'itemID', 'rating']

    user_predictions = predict_user(combined_ratings, new_userId, user_iids)

    user_predictions = pd.merge(user_predictions, final_movies, on='movieId', how='left')
    user_predictions = user_predictions.reindex(columns=['movieId', 'title', 'prediction'])
    print_predictions(user_predictions.iloc[0:10, :], 'Done!')

    #results = analysis(test_data)


if __name__ == '__main__':
    main()





















#
