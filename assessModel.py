import numpy as np
from numpy import vstack
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import mean_squared_error, auc, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.inspection import plot_partial_dependence
import shap

from sklearn.ensemble import RandomForestRegressor
import xgboost

TEST_FILE = "adult_numeric_test.csv"
MODEL_FILE = "./Models/CENSUS_XGB.sav"

'''
TO-DO: implement functions for AUC, confusion matrix, specficity and
    sensitivity, something similar to risk category vs. outcome (w quartiles)

'''

arr1 = pd.read_csv(TEST_FILE)
x = arr1.iloc[:,1:]
y = arr1.iloc[:,0]

#------------------------------------------------------------------------------

def Create_AUC_Curve(y_pred, y_obs=y):

    fpr, tpr, thresholds = roc_curve(y_obs, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

#------------------------------------------------------------------------------

def Create_PR_Curve(y_pred, y_obs=y):

    prec, recall, _ = precision_recall_curve(y_obs, y_pred)
    cm_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    plt.show()

#------------------------------------------------------------------------------

def Create_CM(y_pred, y_obs=y):

    cm = confusion_matrix(y_obs, y_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()

#------------------------------------------------------------------------------
'''
    TODO: Divide the predicted outcomes into k percentiles, get a crude mean of
    each percentile then plot it.
'''

def Calibration_Plot(y_pred, y_obs=y, k=10):

    df = pd.DataFrame()
    df['Observed'] = y_obs
    df['Pred'] = y_pred
    bin_labels = range(k)
    df['Category'] = pd.qcut(y_pred, k, precision=0, labels=bin_labels)

    pred_mean_list = []
    obs_mean_list = []

    for i in bin_labels:
        i_df = df.loc[df['Category'] == i]
        pred_mean = i_df['Pred'].mean()
        pred_mean_list.append(pred_mean)
        obs_mean = i_df['Observed'].mean()
        obs_mean_list.append(obs_mean)

    # Calibration Plot
    plt.title("Predicted Mean vs. Observed Mean")
    plt.scatter(pred_mean_list, obs_mean_list)
    xlimit = min((max(pred_mean_list) + 0.1), 1.0)
    ylimit = min((max(obs_mean_list) + 0.1), 1.0)
    plt.plot([0, xlimit], [0, ylimit],'r--')
    plt.ylabel('Observed Mean')
    plt.xlabel('Predicted Mean')
    plt.show()

    plt.clf()

    # Risk Category vs. Mean Plots
    plt.title("Risk Category vs. Predicted Prob of Money")
    plt.bar(bin_labels, pred_mean_list)
    plt.ylabel('Mean Predicted Prob')
    plt.xlabel('Risk Category')
    plt.show()

    plt.clf()

    plt.title("Risk Category vs. Observed Prob of Money")
    plt.bar(bin_labels, obs_mean_list)
    plt.ylabel('Mean Observed Prob')
    plt.xlabel('Risk Category')
    plt.show()



#------------------------------------------------------------------------------

def Shap_Assess(model, type='tree'):

    if type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif type == 'nn':
        pass
    elif type == 'other':
        explainer = shap.KernelExplainer(model.predict, shap.sample(x,50))
    else:
        raise ValueError('type variable must be one of: tree, nn, or other')

    shap_values = explainer.shap_values(x)

    shap.summary_plot(shap_values, x)

    for name in x.columns:
        shap.dependence_plot(name, shap_values, x, interaction_index=None)

#------------------------------------------------------------------------------
'''
    Loads in a specified model and gives information about it.
'''

def Assess_Model():
    model = pickle.load(open(MODEL_FILE, 'rb'))

    y_pred = model.predict(x)
    Create_AUC_Curve(y_pred)
    Create_PR_Curve(y_pred)
    Calibration_Plot(y_pred, 10)

    Shap_Assess(model, type='Tree')

#------------------------------------------------------------------------------

if __name__ == '__main__':
    Assess_Model()
