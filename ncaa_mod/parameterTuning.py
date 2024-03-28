import os
import pandas as pd
from scipy.stats import zscore
import pandas as pd
import io
import requests
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import csv
import matplotlib.pyplot as plt
import torch.cuda.amp as amp
import copy
import pickle
import cleaning as c

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False

PATH = 'data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = c.read_to_one_frame(PATH)
df = c.clean_data(df)

X_columns = df.columns[2:-1]
X = pd.get_dummies(df[X_columns])
a14Target = df['POSTSEASON']

X_train_a14, X_temp_a14, y_train_a14, y_temp_a14 = train_test_split(a14Features, a14Target, test_size=0.3, random_state=42)
X_val_a14, X_test_a14, y_val_a14, y_test_a14 = train_test_split(X_temp_a14, y_temp_a14, test_size=0.5, random_state=42)


def scaleXData(train, val, test, scalerName):
    if scalerName == "Robust":
        scaler = RobustScaler()
    elif scalerName == "Standard":
        scaler = StandardScaler()
    elif scalerName == "MinMax":
        scaler = MinMaxScaler()
    elif scalerName == "Norm":
        scaler = Normalizer()
    elif scalerName == "MaxAbs":
        scaler = MaxAbsScaler()
    elif scalerName == "None":
        return scalerName, X_train_a14, X_val_a14, X_test_a14

    X_train_a14 = scaler.fit_transform(train)
    X_val_a14 = scaler.transform(val)
    X_test_a14 = scaler.transform(test)

    return scaler, X_train_a14, X_val_a14, X_test_a14


def trainModel(model, optimizer, scheduler, train_loader, val_loader, early_stopper, n_epochs=100, device=device):
    MSE = nn.MSELoss()
    MAE = nn.L1Loss()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss_mse = MSE(y_pred, y)
            loss_mae = MAE(y_pred, y)
            loss = loss_mse + loss_mae
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss_mse = MSE(y_pred, y)
                loss_mae = MAE(y_pred, y)
                loss = loss_mse + loss_mae
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        if early_stopper(model, val_loss):
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    if early_stopper.restore_best_weights:
        model.load_state_dict(early_stopper.best_model)

    return model, val_loss


class MyNeuralNetWorth(nn.Module):
    def __init__(self, input_size, depth, hiddenLayerNeuronSizes, dropoutLayerRates, activationFunctions):
        super(MyNeuralNetWorth, self).__init__()

        layers = []
        cur_size = input_size
        for size, rate, activation in zip(hiddenLayerNeuronSizes, dropoutLayerRates, activationFunctions):
            layers.append(nn.Linear(cur_size, size))
            layers.append(activation)
            layers.append(nn.Dropout(rate))
            layers.append(nn.BatchNorm1d(size))
            cur_size = size

        layers.append(nn.Linear(cur_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def save_hyperparameters_and_loss(trial, trial_number, val_loss, scaler, sizes, rates, activations):
    csv_file_path = f'{trial_number}.csv'

    hyperparams = {
        'lr': trial.params['lr'],
        'batch_size': trial.params['batch_size'],
        'early_stopping_patience': trial.params['early_stopping_patience'],
        'n_epochs': trial.params['n_epochs'],
        'optimizer_name': trial.params['optimizer'],
        'shuflleTrain': trial.params['shuflleTrain'],
        'shuflleVal': trial.params['shuflleVal'],
        'scheduler_name': trial.params['scheduler'],
        'scalerName' : trial.params['scalerName'],
        'scaler' : scaler,
        'val_loss': val_loss,
        'sizes' : sizes,
        'rates' : rates,
        'activations' : activations,
    }

    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=hyperparams.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(hyperparams)
    scaler_path = f"{trial_number}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)


def int2Activation(num):
    if num == 0:
        return nn.ELU()
    elif num == 1:
        return nn.Hardshrink()
    elif num == 2:
        return nn.Hardsigmoid()
    elif num == 3:
        return nn.Hardtanh()
    elif num == 4:
        return nn.Hardswish()
    elif num == 5:
        return nn.LeakyReLU()
    elif num == 6:
        return nn.LogSigmoid()
    elif num == 7:
        return nn.PReLU()
    elif num == 8:
        return nn.ReLU()
    elif num == 9:
        return nn.ReLU6()
    elif num == 10:
        return nn.RReLU()
    elif num == 11:
        return nn.SELU()
    elif num == 12:
        return nn.CELU()
    elif num == 13:
        return nn.GELU()
    elif num == 14:
        return nn.Mish()
    elif num == 15:
        return nn.Softplus()
    elif num == 16:
        return nn.Softshrink()
    elif num == 17:
        return nn.Softsign()
    elif num == 18:
        return nn.Tanh()
    elif num == 19:
        return nn.Tanhshrink()
    

def load_scaler_for_best_trial(best_trial_number):
    scaler_path = f"{best_trial_number}.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler




import optuna
import shutil

def objective(trial, X_train_a14, X_val_a14, X_test_a14,y_train_a14 ,y_val_a14):
    try:
        input_size = X_train_a14.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'NAdam', 'RMSprop', 'Adamax', 'Rprop'])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        early_stopping_patience = trial.suggest_int('early_stopping_patience', 5, 30)
        n_epochs = trial.suggest_int('n_epochs', 1000, 1000)
        modelDepth = trial.suggest_int('modelDepth', 1, 10)

        sizes = []
        rates = []
        activations = []
        for i in range(1, modelDepth+1):
            sizes.append(trial.suggest_int(f'neuronSize{i}', 8, 512))
            rates.append(trial.suggest_float(f'dropoutRate{i}', 0.1, 0.9))
            activations.append(int2Activation(trial.suggest_int(f'activationFunction{i}', 0, 19)))

        train, val, test = X_train_a14, X_val_a14, X_test_a14
        shuflleTrain = trial.suggest_categorical('shuflleTrain', [True, False])
        shuflleVal = trial.suggest_categorical('shuflleVal', [True, False])

        scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts'])
        scalerName = trial.suggest_categorical('scalerName', ["Robust","Standard", "MinMax", "Norm", "MaxAbs", "None"])

        scaler, X_train_a14, X_val_a14, X_test_a14 = scaleXData(train, val, test, scalerName)

        X_train_tensor_a14 = torch.FloatTensor(X_train_a14).to(device)
        y_train_tensor_a14 = torch.FloatTensor(y_train_a14.values).view(-1, 1).to(device)
        X_val_tensor_a14 = torch.FloatTensor(X_val_a14).to(device)
        y_val_tensor_a14 = torch.FloatTensor(y_val_a14.values).view(-1, 1).to(device)

        train_loader = DataLoader(TensorDataset(X_train_tensor_a14, y_train_tensor_a14), batch_size=batch_size, shuffle=shuflleTrain)
        val_loader = DataLoader(TensorDataset(X_val_tensor_a14, y_val_tensor_a14), batch_size=batch_size, shuffle=shuflleVal)

        model = MyNeuralNetWorth(input_size=input_size,depth=modelDepth,hiddenLayerNeuronSizes=sizes,dropoutLayerRates=rates,activationFunctions=activations).to(device)
        if optimizer_name == 'Adam':
            adamamsgrad = trial.suggest_categorical('adamamsgrad', [True, False])
            adamfused = trial.suggest_categorical('adamfused', [True, False])
            beta1 = trial.suggest_float('adam_beta1', 0.5, 0.999)
            beta2 = trial.suggest_float('adam_beta2', 0.5, 0.999)
            weight_decay = trial.suggest_float('adam_weight_decay', 1e-10, 1e-3, log=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=adamamsgrad, fused=adamfused, betas=(beta1, beta2), weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            adamWamsgrad = trial.suggest_categorical('adamWamsgrad', [True, False])
            adamWfused = trial.suggest_categorical('adamWfused', [True, False])
            beta1 = trial.suggest_float('adamW_beta1', 0.5, 0.999)
            beta2 = trial.suggest_float('adamW_beta2', 0.5, 0.999)
            weight_decay = trial.suggest_float('adamW_weight_decay', 1e-10, 1e-3, log=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=adamWamsgrad, fused=adamWfused, betas=(beta1, beta2), weight_decay=weight_decay)
        elif optimizer_name == 'NAdam':
            betas = (trial.suggest_float('nadam_beta1', 0.5, 0.999),
                trial.suggest_float('nadam_beta2', 0.5, 0.999))
            weight_decay = trial.suggest_float('nadam_weight_decay', 1e-10, 1e-3, log=True)
            optimizer = torch.optim.NAdam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        elif optimizer_name == 'RMSprop':
            alpha = trial.suggest_float('rmsprop_alpha', 0.8, 0.999)
            momentum = trial.suggest_float('rmsprop_momentum', 0.0, 1.0)
            weight_decay = trial.suggest_float('rmsprop_weight_decay', 1e-10, 1e-3, log=True)
            rmspropcentered = trial.suggest_categorical('rmspropcentered', [True, False])
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, centered=rmspropcentered, momentum=momentum, alpha=alpha, weight_decay=weight_decay)
        elif optimizer_name == 'Adamax':
            betas = (trial.suggest_float('adamax_beta1', 0.5, 0.999),
                trial.suggest_float('adamax_beta2', 0.5, 0.999))
            weight_decay = trial.suggest_float('adamax_weight_decay', 1e-10, 1e-3, log=True)
            optimizer = torch.optim.Adamax(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        elif optimizer_name == 'Rprop':
            etas = (trial.suggest_float('rprop_eta1', 0.1, 0.5), trial.suggest_float('rprop_eta2', 1.1, 2.0))
            step_sizes = (trial.suggest_float('rprop_step_size_min', 1e-6, 1e-2), trial.suggest_float('rprop_step_size_max', 1e-2, 1))
            optimizer = torch.optim.Rprop(model.parameters(), lr=lr, etas=etas, step_sizes=step_sizes)


        if scheduler_name == 'StepLR':
            step_size = trial.suggest_int('step_size', 1, 100)
            gamma = trial.suggest_float('gamma', 0.1, 0.99)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'ReduceLROnPlateau':
            factor = trial.suggest_float('factor', 0.1, 0.5)
            patience = trial.suggest_int('patience', 1, 10)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
        elif scheduler_name == 'CosineAnnealingLR':
            T_max = trial.suggest_int('T_max', 1, 100)
            eta_min = trial.suggest_float('eta_min', 1e-6, 1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_name == 'CosineAnnealingWarmRestarts':
            T_0 = trial.suggest_int('T_0', 1, 100)
            T_mult = trial.suggest_int('T_mult', 1, 100)
            eta_min = trial.suggest_float('eta_min', 1e-6, 1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)


        early_stopper = EarlyStopping(patience=early_stopping_patience)

        trained_model, val_loss = trainModel(model, optimizer, scheduler, train_loader, val_loader, early_stopper, n_epochs=n_epochs, device=device)
        save_hyperparameters_and_loss(trial, trial.number, val_loss, scaler, sizes, rates, activations)
        trial_number = trial.number
        torch.save(model.state_dict(), f"/content/drive/MyDrive/NewAssignment4/a2models/a2model_{trial_number}.pt")

        return val_loss
    except Exception as e:
        print(f"Trial failed with error: {e}")
        trial.report(float('inf'), step=0)
        raise optuna.TrialPruned()


def main():
    if not os.path.exists('model_checkpoints'):
        os.makedirs('model_checkpoints')

    tpe_sampler = optuna.samplers.TPESampler(
        consider_prior=True,
        prior_weight=1.0,
        consider_magic_clip=True,
        consider_endpoints=False,
        n_startup_trials=50,
        n_ei_candidates=24,
        multivariate=False,
        group=False,
        warn_independent_sampling=True,
    )
    successive_halving_pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource='auto',
        reduction_factor=4,
        min_early_stopping_rate=0,
        bootstrap_count=0,
    )
    study = optuna.create_study(direction='minimize', sampler=tpe_sampler, pruner=successive_halving_pruner)
    study.optimize(lambda trial: objective(trial, X_train_a14, X_val_a14, X_test_a14, y_train_a14, y_val_a14), n_trials=10000)
    print('Number of finished trials: ', len(study.trials))
    best_trial = study.best_trial
    print(f"Best trial parameters: {best_trial.params}")
    input_size = X_train_a14.shape[1]
    sizes = []
    rates = []
    activations = []
    for i in range(1, best_trial.params['modelDepth']+1):
        sizes.append(best_trial.params[f'neuronSize{i}'])
        rates.append(best_trial.params[f'dropoutRate{i}'])
        activations.append(best_trial.params[f'activationFunction{i}'])
    best_model = MyNeuralNetWorth(
        input_size=input_size,
        depth=best_trial.params['modelDepth'],
        hiddenLayerNeuronSizes=sizes,
        dropoutLayerRates=rates,
        activationFunctions=activations,
    )
    best_model_path = f"{best_trial.number}.pt"
    best_model.load_state_dict(torch.load(best_model_path))
    torch.save(f"{best_trial.number}.pt", f"{best_trial.number}.pt")
    shutil.copy(f'{best_trial.number}.csv', f"{best_trial.number}.csv")
    shutil.copy(f"{best_trial.number}.pkl", f"{best_trial.number}.pkl")
    best_scaler = load_scaler_for_best_trial(best_trial.number)
    return best_model, best_scaler

best_model, best_scaler = main()


def predictMissingValues(model, scaler, features, missingIndicies):
    features_missing_scaled = scaler.transform(features.loc[missingIndicies])
    features_missing_tensor = torch.FloatTensor(features_missing_scaled)
    model.eval()
    with torch.no_grad():
        predictions = model(features_missing_tensor).cpu().numpy().flatten()
    return predictions







