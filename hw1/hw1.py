"""
# **Download Data**


If the Google drive links are dead, you can download data from [kaggle](https://www.kaggle.com/c/ml2021spring-hw1/data), and upload data manually to the workspace.
"""

tr_path = 'covid.train.csv'  # path to training data
tt_path = 'covid.test.csv'   # path to testing data

import gdown
gdown.download('http://drive.google.com/u/1/uc?id=19CCyCgJrUxtvgZF53vnctJiOJ23T5mqF', "covid.train.csv")
gdown.download('http://drive.google.com/u/1/uc?id=1CE240jLm2npU-tdz81-oVKEF3T2yfT1O', "covid.test.csv")

"""# **Import Some Packages**"""

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

"""# **Some Utilities**

You do not need to modify this part.
"""

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda:1' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.5, 1.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

"""# **Preprocess**

We have three kinds of datasets:
* `train`: for training
* `dev`: for validation
* `test`: for testing (w/o target value)

## **Dataset**

The `COVID19Dataset` below does:
* read `.csv` files
* extract features
* split `covid.train.csv` into train/dev sets
* normalize features

Finishing `TODO` below might make you pass medium baseline.
"""

class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                path,
                mode='train',
                feats=[0]):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]
            
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 5 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 5 == 0]
            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        global glo_data
        # Normalize features (you may remove this part to see what will happen)
        self.data = (self.data - glo_data.mean(dim=0, keepdim=True)) / glo_data.std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

"""## **DataLoader**

A `DataLoader` loads data from a given `Dataset` into batches.

"""

def prep_dataloader(path, mode, batch_size, n_jobs=0, feats=[0]):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = COVID19Dataset(path, mode=mode, feats=feats)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader

"""# **Deep Neural Network**

`NeuralNet` is an `nn.Module` designed for regression.
The DNN consists of 2 fully-connected layers with ReLU activation.
This module also included a function `cal_loss` for calculating loss.

"""

class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.hidden_size = 512
        self.batchnorm = nn.BatchNorm1d(self.hidden_size)
        self.dropout_input = nn.Dropout(0.8)
        self.dropout = 0.8
        self.ensemble_size = 10
        self.input = nn.Linear(input_dim, self.hidden_size)
        self.nets = torch.nn.ModuleList([nn.Sequential(
        nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 4, self.hidden_size // 8),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 8, 1)
        ) for i in range(self.ensemble_size)])
        self.criterion = nn.MSELoss(reduction='mean')
        self.output = nn.Linear(self.ensemble_size, 1)
    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        input_ = self.dropout_input(self.batchnorm(self.input(x)))
        out = torch.cat([self.nets[i](input_) for i in range(self.ensemble_size)], dim = 1)
        return self.output(out).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L2 regularization here
        return self.criterion(pred, target)

"""# **Train/Dev/Test**

## **Training**
"""

def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        tr_mse = 0
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            tr_mse += mse_loss.detach().cpu().item()

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        loss_record['train'].append(tr_mse / len(tr_set.dataset) * config["batch_size"])
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

"""## **Validation**"""

def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss

"""## **Testing**"""

def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds

"""# **Setup Hyper-parameters**

`config` contains hyper-parameters for training and the path to save your model.
"""

device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
target_only = False                   # TODO: Using 40 states & 2 tested_positive features

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 300000,                # maximum number of epochs
    'batch_size': 15,               # mini-batch size for dataloader
    'optimizer': 'Adam',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0004,                # learning rate of SGD
        #'momentum': 0.9              # momentum for SGD
        'weight_decay': 0.00007
    },
    'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}

"""# **Load data and model**"""

glo_data = []
# TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
feats = [40, 41, 42, 43, 47, 51, 56, 57, 58, 59, 60, 61, 65, 69, 74, 75, 76, 77, 78, 79, 83, 87, 92]#40, 41, 42, 43, 51, 56, 57, 58, 59, 60, 61, 69, 74, 75, 76, 77, 78, 79, 87, 92]
#feats = [40, 41, 42, 43, 57, 58, 59, 60, 61, 75, 76, 77, 78, 79, 92]
#feats = [42, 43, 57, 60, 61, 75, 78, 79]
#feats = [43, 57, 58, 60, 75, 78]
with open(tr_path, "r") as fp:
    glo_data = list(csv.reader(fp))
    glo_data = np.array(glo_data[1:])[:, 1:].astype(float)[:, :-1]

with open(tt_path, "r") as fp:
    tmp = np.array(list(csv.reader(fp))[1:])[:, 1:].astype(float)
    glo_data = np.concatenate([glo_data, tmp], axis=0)
glo_data = torch.FloatTensor(glo_data[:, feats])

tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], feats=feats)
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], feats=feats)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], feats=feats)

model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device

"""# **Start Training!**"""

model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

plot_learning_curve(model_loss_record, title='deep model')

del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)  # Show prediction on the validation set

"""# **Testing**
The predictions of your model on testing set will be stored at `pred.csv`.
"""

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, 'pred.csv')         # save prediction file to pred.csv

"""# **Hints**

## **Simple Baseline**
* Run sample code

## **Medium Baseline**
* Feature selection: 40 states + 2 `tested_positive` (`TODO` in dataset)

## **Strong Baseline**
* Feature selection (what other features are useful?)
* DNN architecture (layers? dimension? activation function?)
* Training (mini-batch? optimizer? learning rate?)
* L2 regularization
* There are some mistakes in the sample code, can you find them?

# **Reference**
This code is completely written by Heng-Jui Chang @ NTUEE.  
Copying or reusing this code is required to specify the original author. 

E.g.  
Source: Heng-Jui Chang @ NTUEE (https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb)
"""
