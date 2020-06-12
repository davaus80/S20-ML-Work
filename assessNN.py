import assessModel
import createNN
import pandas as pd
import numpy as np

import torch
import shap
from torch.utils.data import Dataset, DataLoader, random_split

#-------------------------------------------------------------------------------
'''
    These constants are all that need to be modified to allow the program to run
'''

# Locations of files or data
TRAIN_FNAME = 'adult_numeric_train.csv'
TEST_FNAME = 'adult_numeric_test.csv'
PARAM_FILE = 'Param_DF.csv'
PARAM_ROW = 14 # Which row of PARAM_FILE contains the desired parameters
PATH = './Models/CENSUS_BEST_NN.pt' # PATH to model state_dict

# Loading in Data
x_train = pd.read_csv(TEST_FNAME).iloc[:,1:]

arr1 = pd.read_csv(TEST_FNAME)
x_test = arr1.iloc[:,1:]
y_test = arr1.iloc[:,0]

# Misc.
NUM_CATS = 10 # Number of groups to split for calibration plot
SAMP_SIZE = 200 # Sample size to give to SHAP as background

#-------------------------------------------------------------------------------
'''
    Loads in the MLP model at PATH and outputs an ROC plot, PR plot, Calibration
    plot, Feature Importance plot, and several PDP plots
'''

def main():

    # 1. Loading in Model

    param_df = pd.read_csv(PARAM_FILE)
    row = param_df.iloc[PARAM_ROW, :]

    drop = row['drop']
    fc1 = row['fc1'].astype(int)
    fc2 = row['fc2'].astype(int)

    model = createNN.MLP(fc1out=fc1, fc2out=fc2, drop = drop)
    model.load_state_dict(torch.load(PATH))

    # 2. Loading in Data

    testset = createNN.NewDataset(TEST_FNAME)
    testloader = DataLoader(testset,
                                batch_size = BATCH_SIZE,
                                num_workers=0)
    model.eval()

    # 3. Getting predicted val;ues from test set

    y_pred = torch.Tensor()

    for i, batch in enumerate(testloader):
        x_data, y_data = batch
        preds = model(x_data).flatten()
        y_pred = torch.cat((y_pred, preds), dim=0)

    y_pred = y_pred.detach().numpy()

    # 4. Call Assessment Functions

    assessModel.Create_AUC_Curve(y_pred, y_test)
    assessModel.Create_PR_Curve(y_pred, y_test)
    assessModel.Calibration_Plot(y_pred, y_test, NUM_CATS)

    # 5. SHAP Assessment

    background = x_train.to_numpy()[np.random.choice(x_train.shape[0], SAMP_SIZE, replace=False)]
    background = torch.from_numpy(background).float()
    e = shap.DeepExplainer(model, background)

    x_tensor = torch.from_numpy(x_test.to_numpy()).float()

    print("Calculating Shap Values, this may take a while...")

    shap_values = e.shap_values(x_tensor)

    shap.summary_plot(shap_values, x_tensor)

    for i in range(len(x_train.columns)):
        shap.dependence_plot(i, shap_values, x_test,
                            interaction_index=None, feature_names=x_train.columns)


if __name__ == '__main__':
    main()
