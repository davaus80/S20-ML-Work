import numpy as np
from numpy import vstack
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn.init import xavier_uniform_

from sklearn.metrics import mean_squared_error, auc, roc_curve
from hyperopt import hp, tpe, fmin

from datetime import datetime

#-------------------------------------------------------------------------------
'''
    =============================== README =====================================

    This program searches the hyperparameter space using HyperOpt and prints the
    accuracy of the best found set of hyperparameters. It also saves the model
    every epoch to a NN Saved Models folder and saves the error to /runs/nn
    for later plotting (both must be in the same directory as this file).

    To customize this program for your problem:

    - Save your data to a csv file with the outcome in the first column
    - Update global constants (just below this message)
        - Update FNAME with the location of your csv file
        - Update INPUT_DIM with the number of features
        - OUTPUT_DIM should be 1 unless you're doing multi-class classification
    - Update parameter space to search, found in Optimize_NN() near the
      bottom of the file.
    - If your problem is a regression problem not restricted to (0,1), you will
      need to change the final accuracy measurement from ROC to something else.

'''
# GLOBAL CONSTANTS

randomState = 42
np.random.seed(randomState)

# Path to Data CSV file
FNAME = 'adult_numeric.csv'

# These constants affect the training and structure of the network
# Input_dim should be the number of features, output_dim shouldbe 1 for
# binary classification. Batch_size and epoch can be manually tuned.
BATCH_SIZE = 200
NUM_EPOCHS = 15
INPUT_DIM = 13
OUTPUT_DIM = 1

# These values are deprecated and no longer serve any purpose
# They serve as a backup in case something goes wrong with the
# hyperopt hyperparameter tuning since they allow easier manual tuning.

FC1_OUTCHANNELS = 14
FC2_OUTCHANNELS = 8
LEARNING_RATE = 3e-4
DROPOUT = 0.3

#------------------------------------------------------------------------
'''
    A class implementing a neural network
'''

class MLP(nn.Module):

    ''' Initializes the network

    Args:
        fc1out: number of nodes in first hidden layer
        fc2out: number of nodes in second hidden layer
        drop: dropout values
    '''
    def __init__(self,fc1out=FC1_OUTCHANNELS,fc2out=FC2_OUTCHANNELS,drop=DROPOUT):
        super(MLP, self).__init__()
        self.name = "NN"
        self.hidden1 = nn.Linear(INPUT_DIM, fc1out)
        xavier_uniform_(self.hidden1.weight)
        self.hidden2 = nn.Linear(fc1out, fc2out)
        xavier_uniform_(self.hidden2.weight)
        self.hidden3 = nn.Linear(fc2out, OUTPUT_DIM)
        xavier_uniform_(self.hidden3.weight)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.drop3 = nn.Dropout(drop)

    ''' The feed forward function

    Args:
        x: input tensor
    '''
    def forward(self, x):
        x = F.relu(self.hidden1(self.drop1(x)))
        x = F.relu(self.hidden2(self.drop2(x)))
        x = torch.sigmoid(self.hidden3(self.drop3(x)))
        return x

#------------------------------------------------------------------------
'''
    A class for storing data
'''

class NewDataset(Dataset):
    ''' Initializes the dataset

    Args:
        filename: the location of the csv file from which to import data
    '''
    def __init__(self, filename):
        df = pd.read_csv(filename)
        self.x = df.values[:, 1:].astype('float32')
        self.y = df.values[:,0].astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]

    def get_splits(self, n_test = 0.15):
        test_size = round(n_test * len(self.x))
        train_size = len(self.x) - test_size
        return random_split(self, [train_size, test_size])

#------------------------------------------------------------------------

def get_model_name(name, dim1, dim2, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "./NN Saved Models/{}_{}x{}_bs{}_lr{:.5f}_epoch{}.pt".format(name,
                                                   dim1,
                                                   dim2,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

#---------------------------------------------------------------------------

'''
    Trains a model using the training data in trainloader
    Takes in variables representing the loss function, number of epochs,
    batch size and device to run on.

    Args:
        trainloader: DataLoader with training data
        valloader: DataLoader with validation data
        model: model to be trained

        Others are model features(dim1 = fc1out of model)

    Effects:
        Saves a model to file every epoch
        Saves results to tensorboard
'''
def train_net(trainloader, valloader, model, num_epochs=NUM_EPOCHS,
            learn_rate=LEARNING_RATE, bs=BATCH_SIZE,
            device='cpu', dim1=FC1_OUTCHANNELS, dim2=FC2_OUTCHANNELS):

    logdir = "runs/NN/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(logdir)

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        val_running_loss = 0.0

        model.train()
        for i, data in enumerate(trainloader, 0):
            ## Get input data and label
            x_train, y_train = data

            # Clear gradient and get output
            optimizer.zero_grad()
            y_pred = model(x_train)

            # Get loss from predicted output, get gradients wrt params and then
            # update params
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:
                writer.add_scalar('training loss',
                                running_loss/50,
                                epoch * len(trainloader) + i)

                running_loss = 0.0

        #Get validation loss
        model.eval()
        for i, data in enumerate(valloader, 0):
            features, labels = data

            with torch.no_grad():
                y_pred = model(features)
                loss = loss_fn(y_pred, labels)

                val_running_loss += loss.item()
                if i % 10 == 9:
                    writer.add_scalar('validation loss',
                                    val_running_loss/9,
                                    epoch * len(valloader) + i)

                    val_running_loss = 0.0

        # Save the current model (checkpoint) to a file
        model_path = get_model_name(model.name,dim1, dim2, bs, learn_rate, epoch)
        torch.save(model.state_dict(), model_path)

    #print('Finished Training')

#------------------------------------------------------------------------
'''
    Gets the average loss for the model on the validation set
'''

def get_val_loss(valloader, model, device='cpu'):
    loss_fn = nn.MSELoss()

    running_loss = 0.0
    model.eval()
    for i, data in enumerate(valloader, 0):
        features, labels = data

        with torch.no_grad():
            y_pred = model(features)
            loss = loss_fn(y_pred, labels)

            running_loss += loss.item()

    avg_loss = running_loss / len(valloader)

    return avg_loss

#------------------------------------------------------------------------

'''
    Returns the ROC accuracy of the model on the validation sample
'''
def eval_net(valloader, model, rough_avg=False):
    model.eval()
    loss_fn = nn.MSELoss()
    preds_list, actual_list = list(), list()

    with torch.no_grad():
        for i, data in enumerate(valloader):
            x_test, y_test = data
            y_pred = model(x_test)

            y_pred = y_pred.detach().numpy()
            actual = y_test.numpy()
            actual = actual.reshape(len(actual), 1)

            preds_list.append(y_pred)
            actual_list.append(actual)

        preds_list, actual_list = vstack(preds_list), vstack(actual_list)
        fpr, tpr, thresholds = roc_curve(actual_list, preds_list)
        accuracy = auc(fpr, tpr)

        return accuracy

#------------------------------------------------------------------------
'''
    Given an input row and a model, it returns the predicted value for the row
    Effects:
        None
'''
def predict(row, model):
    row = Tensor([row])
    y_pred = model(row)
    y_pred = y_pred.detach().numpy()
    return y_pred

#-------------------------------------------------------------------------------
'''
    Trains a NN using the given hyperparameters (a dictionary object) and
    returns the average validation loss (if assess_final is false) or
    returns the accuracy and saves the model
'''

def Run_NN(space, assess_final=False):

    lr = space['lr']
    dropout = space['dropout']
    fc1out = space['fc1']
    fc2out = space['fc2']
    batch_size = BATCH_SIZE

    # 1. Set up dataloaders

    data = NewDataset(FNAME)
    learnset, testset = data.get_splits()
    trainset, valset = data.get_splits()

    trainloader = DataLoader(trainset,
                                batch_size = batch_size,
                                shuffle = True,
                                num_workers=0)
    valloader = DataLoader(valset,
                                batch_size = batch_size,
                                shuffle = True,
                                num_workers=0)
    testloader = DataLoader(testset,
                                batch_size = batch_size,
                                shuffle = True,
                                num_workers=0)


    # 2. Create and train model

    model = MLP(fc1out=fc1out, fc2out=fc2out, drop = dropout)
    train_net(trainloader, valloader, model, learn_rate= lr,
                bs=batch_size, dim1=fc1out, dim2=fc2out)

    # 3. Return Validation Loss
    if assess_final:
        accuracy = eval_net(valloader, model)
        return accuracy
    else:
        loss = get_val_loss(valloader, model)
        return loss

#------------------------------------------------------------------------

def optimizeNN():
    param_space = {
            'lr': hp.loguniform('lr', -8, -3),
            'dropout': hp.uniform('dropout', 0, 0.4),
            'fc1': hp.choice('fc1', range(1,20,5)),
            'fc2': hp.choice('fc2', range(1,20,5)),
    }

    best = fmin(fn=Run_NN, space = param_space,
                algo=tpe.suggest, max_evals=40)

    print(best)
    print(Run_NN(best, assess_final=True))

#-------------------------------------------------------------------------------

def main():
    optimizeNN()


if __name__ == '__main__':
    main()
