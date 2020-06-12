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

from sklearn.svm import SVR

#-------------------------------------------------------------------------------
'''
    Constants and global variables
'''
DEFAULT_NAME = 'CENSUS'
TRAIN_FILE = 'adult_numeric_train.csv'
SEARCHES = 15   # number of points to search


## Setting up data to be accessible in all functions
arr1 = pd.read_csv(TRAIN_FILE).to_numpy()

x = arr1[:,1:]
y = arr1[:,0]

x_train,x_val,y_train,y_val = train_test_split(x,y,
                                                test_size=0.2,random_state=0)


## Dimensions and initial points to search
default_parameters = [1.0, 0.1]

dimensions = [
        Real(0.01, 2.0, prior="log-uniform", name="C"),
        Real(0.005, 0.5, prior="log-uniform", name="epsilon")
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
    plt.savefig("RF_AUC.png")

#-------------------------------------------------------------------------------

@use_named_args(dimensions=dimensions)
def Get_SVM_Loss(C, epsilon):

    model = SVR(C=C, epsilon=epsilon)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    #import pdb; pdb.set_trace()
    loss = mean_squared_error(y_val, y_pred)

    print(loss)
    return loss

#-------------------------------------------------------------------------------

def Best_Model(param_list):
    C = param_list[0]
    epsilon = param_list[1]


    model = SVR(C=C, epsilon=epsilon)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    createAUCCurve(y_val,y_pred)

    filename =  "./Models/{}_SVM.sav".format(DEFAULT_NAME)
    pickle.dump(model, open(filename,'wb'))

#-------------------------------------------------------------------------------

def optimizeSVM():

    best = gp_minimize(func=Get_SVM_Loss,
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
    optimizeSVM()


if __name__ == '__main__':
    main()
