import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

import pickle
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

import xgboost as xgb
from xgboost import XGBRegressor

#-------------------------------------------------------------------------------
'''
    Constants and global variables
'''
DEFAULT_NAME = 'CENSUS'
TRAIN_FILE = 'adult_numeric_train.csv'
SEARCHES = 30    # number of points to search


## Setting up data to be accessible in all functions
arr1 = pd.read_csv(TRAIN_FILE)

x = arr1.iloc[:,1:]
y = arr1.iloc[:,0]

x_train,x_val,y_train,y_val = train_test_split(x,y,
                                                test_size=0.2,random_state=0)


## Dimensions and initial points to search
default_parameters = [9e-2, 4, 0.9, 0.9, 30, 0.09, 5]

dimensions = [
        Real(1e-6,1e-1, prior="log-uniform",name="lr"),
        Integer(2, 15, name="maxdep"),
        Real(0.7, 1, name="subsamp"),
        Real(0.6, 1, name="colsamp"),
        Integer(4, 100, name="n_est"),
        Real(1e-2,10, prior="log-uniform",name="gam"),
        Real(0.5, 30, prior="log-uniform",name="lamb"),
]

#-------------------------------------------------------------------------------

def createAUCCurve(test_labels,predictions):
    fpr, tpr, thresholds = roc_curve(test_labels,predictions)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

#-------------------------------------------------------------------------------

@use_named_args(dimensions=dimensions)
def Get_XGB_Loss(lr, maxdep, subsamp, colsamp, n_est, gam, lamb):

    model = XGBRegressor(learning_rate = lr, max_depth = maxdep,
                        subsample = subsamp, colsample_bytree = colsamp,
                        n_estimators = n_est, gamma = gam, reg_lambda = lamb)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    #import pdb; pdb.set_trace()
    loss = mean_squared_error(y_val, y_pred)

    print(loss)
    return loss

#-------------------------------------------------------------------------------

def Best_Model(param_list):
    lr = param_list[0]
    maxdep = param_list[1]
    subsamp = param_list[2]
    colsamp = param_list[3]
    n_est = param_list[4]
    gam = param_list[5]
    lamb = param_list[6]

    model = XGBRegressor(learning_rate = lr, max_depth = maxdep,
                        subsample = subsamp, colsample_bytree = colsamp,
                        n_estimators = n_est, gamma = gam, reg_lambda = lamb)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    createAUCCurve(y_val,y_pred)

    filename =  "./Models/{}_XGB.sav".format(DEFAULT_NAME)
    pickle.dump(model, open(filename,'wb'))

#-------------------------------------------------------------------------------

def optimizeXGB():

    best = gp_minimize(func=Get_XGB_Loss,
                       dimensions = dimensions,
                       acq_func='EI',
                       n_calls=SEARCHES,
                       x0 = default_parameters,
                       random_state=42)

    print(best.x)
    print(best.fun)
    Best_Model(best.x)

#-------------------------------------------------------------------------------

def main():
    optimizeXGB()


if __name__ == '__main__':
    main()
