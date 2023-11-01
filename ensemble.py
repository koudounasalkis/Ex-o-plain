import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from helper import *
from preprocessing import *
from posterior_utils import *
from spectral_metric import *
from FM_utils_final import *
from loader import *
import pickle

import warnings
warnings.filterwarnings("ignore")


# preprocessing
def build_targets(target_samples,ss_targets):
    n_targets = target_samples[0].shape[1]
    n_outputs = 2 * n_targets + (n_targets * (n_targets - 1)) // 2

    out_targets = np.zeros((len(target_samples), n_outputs))
    with tqdm(enumerate(target_samples), total=len(target_samples)) as pbar:
        for pos,val in pbar:
            val = ss_targets.transform(val)
            means = val.mean(axis=0)
            flat_conv = np.cov(val.T)[np.triu_indices(n_targets)]

            out_targets[pos] = np.concatenate([ means, flat_conv ])
    return out_targets


class InverseTransform:
    def __init__(self):
        pass
    def fit(self, X):
        pass
    def transform(self, X, y=None):
        return 1/X
    def fit_transform(self, X, y=None):
        return self.transform(X)
    


def preprocess_data(training_spectra, training_aux, training_targets_raw, valid_spectra, valid_aux, valid_targets_raw, test_spectra=None, test_aux=None, drop_radius=False):

    ss_spectra = StandardScaler()
    ss_aux = make_pipeline(
        InverseTransform(),
        PolynomialFeatures(2, include_bias=False),
        StandardScaler()
    )
    ss_targets = MinMaxScaler()

    if drop_radius:
        training_targets_raw = [ training_targets_raw[i][:, 1:] for i in range(len(training_targets_raw)) ]
        valid_targets_raw = [ valid_targets_raw[i][:, 1:] for i in range(len(valid_targets_raw)) ]

    std_aug_spectra = ss_spectra.fit_transform(training_spectra)
    std_aug_aux = ss_aux.fit_transform(training_aux)

    ss_targets.fit(np.vstack(training_targets_raw))

    if valid_spectra.shape[0] > 0:
        std_valid_spectra = ss_spectra.transform(valid_spectra)
        std_valid_aux = ss_aux.transform(valid_aux)
    else:
        std_valid_spectra = np.array([])
        std_valid_aux = np.array([])

    # fit only, will use later
    ss_targets = MinMaxScaler()
    ss_targets.fit(np.vstack(training_targets_raw))

    training_targets = build_targets(training_targets_raw, ss_targets)
    if valid_spectra.shape[0] > 0:
        valid_targets = build_targets(valid_targets_raw, ss_targets)
    else:
        valid_targets = np.array([])

    if test_spectra is None or test_aux is None:
        return (std_aug_spectra, std_aug_aux, training_targets), (std_valid_spectra, std_valid_aux, valid_targets), ss_targets
    else:
        test_spectra = ss_spectra.transform(test_spectra)
        test_aux = ss_aux.transform(test_aux)
        return (std_aug_spectra, std_aug_aux, training_targets), (std_valid_spectra, std_valid_aux, valid_targets), (test_spectra, test_aux), ss_targets

class MC_Convtrainer(nn.Module):
    def __init__(self, aux_number, num_targets, p, filters):
        super(MC_Convtrainer, self).__init__()
        self.num_targets = num_targets
        self.aux_number = aux_number
        self.p = p
        self.filters = [1]+filters
        self.convs = []
        for i in range(len(self.filters)-1):
            self.convs.append(nn.Conv1d(self.filters[i], self.filters[i+1], kernel_size=3, padding='same'))
        self.convs = nn.ModuleList(self.convs)
        
        self.fc1_enc = nn.LazyLinear(500)
        self.fc2_enc = nn.Linear(500, 100)

        self.deconvs = []
        for i in range(len(self.filters)-1, 0, -1):
            self.deconvs.append(nn.ConvTranspose1d(self.filters[i], self.filters[i-1], kernel_size=3, padding=1))
        self.deconvs = nn.ModuleList(self.deconvs)
        
        self.fc_head = nn.Linear(100, num_targets)
        
        self.dropout = nn.Dropout(p)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x_sp, x_aux):
        x = self.encode(x_sp, x_aux)
        x = self.dropout(x)
        x = self.fc_head(x)
        return x
    
    def encode(self, x_sp, x_aux):
        x_sp = x_sp.reshape(x_sp.shape[0], 1, -1)
        for conv in self.convs:
            x_sp = self.relu(conv(x_sp))
        x_sp = x_sp.reshape(x_sp.shape[0], -1)
        x = torch.hstack((x_sp, x_aux))
        x = self.dropout(x)
        x = self.relu(self.fc1_enc(x))
        x = self.dropout(x)
        x = self.relu(self.fc2_enc(x))
        return x

def train(model, optimizer, criterion, dataloader):
    train_loss = 0
    for batch_idx, (x_batch_sp, x_batch_aux, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(x_batch_sp, x_batch_aux)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    return train_loss/len(dataloader)

# using ind to make the previous code work
def train_model(train_data, n_epochs, seed):
    std_aug_spectra, std_aug_aux, training_targets = train_data

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(std_aug_spectra).float().cuda(), 
        torch.from_numpy(std_aug_aux).float().cuda(), 
        torch.from_numpy(training_targets).float().cuda()
        )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    aux_num = std_aug_aux.shape[1]
    num_targets = training_targets.shape[1]
    
    torch.manual_seed(seed)
    
    model = MC_Convtrainer(aux_num, num_targets, 0, filters).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    train_losses = []
    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            train_loss = train(model, optimizer, criterion, train_dataloader) # train_dataloader
            train_losses.append(train_loss)
            pbar.set_postfix({'train_loss': train_loss})

    return model


def train_ensemble(train_data, n_estimators, n_epochs):
    models = []
    for i in range(n_estimators):
        std_aug_spectra, std_aug_aux, training_targets = train_data # sample with replacement

        np.random.seed(i)
        ndx = np.random.choice(std_aug_spectra.shape[0], std_aug_spectra.shape[0], replace=True)
        std_aug_spectra = std_aug_spectra[ndx]
        std_aug_aux = std_aug_aux[ndx]
        training_targets = training_targets[ndx]
        models.append(train_model((std_aug_spectra, std_aug_aux, training_targets), n_epochs, seed=i))
    
    return models

def train_validate(train_data, valid_data, n_estimators, n_epochs, ss_targets, ind, seed=42, verbose=False, drop_radius=False):

    std_aug_spectra, std_aug_aux, training_targets = train_data
    std_valid_spectra, std_valid_aux, valid_targets = valid_data

    models = train_ensemble(train_data, n_estimators, n_epochs)

    ## select the corresponding GT for the validation data, and in the correct order.
    index= np.arange(len(ind))
    valid_index = index[~ind]

    instances = N_samples
    num_targets = training_targets.shape[1]
    n_outputs = int(((9+8*num_targets)**.5-3)//2)
    
    y_valid_distribution_all_models = np.zeros((len(models), len(std_valid_spectra), instances, n_outputs))
    y_valid_distribution = np.zeros((len(std_valid_spectra), instances, n_outputs))

    y_pred_valid_all_models = np.stack([ model(
        torch.from_numpy(std_valid_spectra).float().cuda(),
        torch.from_numpy(std_valid_aux).float().cuda()
    ).cpu().detach().numpy() for model in models ])

    y_pred_valid = y_pred_valid_all_models.mean(axis=0)
    print("compressed prediction", y_pred_valid.shape)

    for pt in range(len(std_valid_spectra)):
        for model_id, model in enumerate(models):
            mean = y_pred_valid_all_models[model_id, pt, :n_outputs]
            cov = np.zeros((n_outputs, n_outputs))
            cov[np.triu_indices(n_outputs, 0)] = y_pred_valid_all_models[model_id, pt, n_outputs:]
            cov = cov + cov.T - np.diag(np.diag(cov))

            y_valid_distribution_all_models[model_id, pt] = ss_targets.inverse_transform(np.random.multivariate_normal(mean, cov, size=instances))

        mean = y_pred_valid[pt, :n_outputs]
        cov = np.zeros((n_outputs, n_outputs))
        cov[np.triu_indices(n_outputs, 0)] = y_pred_valid[pt, n_outputs:]
        cov = cov + cov.T - np.diag(np.diag(cov))
        y_valid_distribution[pt] = ss_targets.inverse_transform(np.random.multivariate_normal(mean, cov, size=instances))

    # tr1 = y_valid_distribution
    # weight takes into account the importance of each point in the tracedata. for now we just assume them to be equally weighted
    weights1 = np.ones((y_valid_distribution.shape[0],y_valid_distribution.shape[1]))/np.sum(np.ones(y_valid_distribution.shape[1]) )

    trace_GT = h5py.File(tracedata_path,"r")

    planet_names = np.array([ f"Planet_train{pl_idx+1}" for pl_idx in range(len(targets_trace)) ])[mask]
    # posterior_scores = []
    models_scores = np.zeros((len(models), len(valid_index)))
    ensemble_scores = np.zeros(len(valid_index))

    np.int = np.int64

    bounds_matrix = default_prior_bounds()
    if drop_radius:
        bounds_matrix = bounds_matrix[1:]
    with tqdm(enumerate(valid_index), total=len(valid_index)) as bar:
        for idx, pl_idx in bar:
            planet_name = planet_names[pl_idx]
            tr_GT = trace_GT[planet_name]['tracedata'][()]
            weights_GT = trace_GT[planet_name]['weights'][()]

            if drop_radius:
                tr_GT = tr_GT[:, 1:]
            ## there are cases without ground truth, we will skip over them for this baseline
            ## but every example in leaderboard and final evaluation set will have a complementary ground truth
            if np.isnan(tr_GT).sum() == 1:
                continue
            # compute posterior loss
            for i, model in enumerate(models):
                # score = compute_posterior_loss(tr1[idx], weights1[idx], tr_GT, weights_GT, bounds_matrix)

                # score for model i, pt idx
                score = compute_posterior_loss(y_valid_distribution_all_models[i, idx], weights1[idx], tr_GT, weights_GT, bounds_matrix)
                models_scores[i, idx] = score
            ensemble_scores[idx] = compute_posterior_loss(y_valid_distribution[idx], weights1[idx], tr_GT, weights_GT, bounds_matrix)
            # posterior_scores.append(score)
    # avg_posterior_score = np.mean(posterior_scores)
    print(models_scores.shape)
    print(models_scores.mean(axis=1))
    print("ensemble", ensemble_scores.mean())
    return models, models_scores, ensemble_scores

def evaluate(model, criterion, dataloader):
    valid_loss = 0
    with torch.no_grad():
        for batch_idx, (x_batch_sp, x_batch_aux, target) in enumerate(dataloader):
            output = model(x_batch_sp, x_batch_aux)
            loss = criterion(output, target)
            valid_loss += loss.item()
    return valid_loss/len(dataloader)

if __name__ == "__main__":
    SEED = 42
    threshold = 0.8 ## for train valid split
    N = 6766        ## train on the first 5000 data instances, remember only some examples are labelled, others are unlabelled!
    
    batch_size = 64
    lr = 1e-3
    epochs = 50 * 5
    filters = [32,32,32,64,64,64]
    dropout = 0.1
    N_samples = 5000

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    spectral_training_data, aux_training_data, targets_trace = load_raw_training_data()
    spec_matrix = to_observed_matrix(spectral_training_data,aux_training_data)

    mask = get_supervised_mask(targets_trace)
    spectra, noise = get_spectrum(spec_matrix[mask])
    aux = get_aux(aux_training_data[mask])
    targets = get_targets(targets_trace, mask)

    n_epochs = 100

    drop_radius = True


    for n_estimators in [5]:
        
        print(f"seed {SEED}, n_estimators {n_estimators}")
        
        train_data, valid_data, ind =  train_valid_split(spectra, aux, noise, targets, train_size=.8, seed=SEED)
        training_spectra, training_noise, training_aux, training_targets_raw = train_data
        valid_spectra, valid_noise, valid_aux, valid_targets_raw = valid_data

        std_train_data, std_valid_data, ss_targets = preprocess_data(training_spectra, training_aux, training_targets_raw, valid_spectra, valid_aux, valid_targets_raw, drop_radius=drop_radius)

        model, posterior_scores, ensemble_scores = train_validate(std_train_data, std_valid_data, n_estimators, n_epochs, ss_targets, ind, seed=SEED, verbose=False, drop_radius=drop_radius)

        print()

        with open(f"posteriors-{n_estimators}est.pkl", "wb") as f:
            pickle.dump((model, posterior_scores, ensemble_scores, valid_aux), f)
