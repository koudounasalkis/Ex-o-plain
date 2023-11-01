import os
import h5py
import pandas as pd
from helper import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler

base_path = "/data1/fgiobergia/ariel"
training_path = os.path.join(base_path, "TrainingData")
test_path = os.path.join(base_path, "TestData")
training_GT_path = os.path.join(training_path, "Ground Truth Package")
tracedata_path = os.path.join(training_GT_path, "Tracedata.hdf5")

def load_raw_training_data():

    spectral_training_data = h5py.File(os.path.join(training_path,'SpectralData.hdf5'),"r")
    aux_training_data = pd.read_csv(os.path.join(training_path,'AuxillaryTable.csv'))
    soft_label_data = pd.read_csv(os.path.join(training_GT_path, 'FM_Parameter_Table.csv'))

    trace_GT = h5py.File(tracedata_path)
    planetlist = [p for p in trace_GT.keys()]
    trace = trace_GT[planetlist[0]]['tracedata'][:] 
    weights = trace_GT[planetlist[0]]['weights'][:]

    print("NUM PLANETS: ", len(planetlist))
    print("TRACE: ", trace.shape)
    print("WEIGHTS: ", weights.shape)

    targets_trace = h5py.File(tracedata_path,"r")

    return spectral_training_data, aux_training_data, targets_trace

def get_supervised_mask(targets_trace):
    mask = [ targets_trace[f"Planet_train{i}"]["tracedata"].shape != () for i in range(1, len(targets_trace)+1) ] # True => has GT, False: no GT
    return np.array(mask)

def get_targets(targets_trace, mask):
    targets = [ targets_trace[f"Planet_train{i+1}"]["tracedata"][:] for i, k in enumerate(mask) if k ]
    return targets

def get_spectrum(spec_matrix):
    noise = spec_matrix[:,:,2]
    ## We will incorporate the noise profile into the observed spectrum by treating the noise as Gaussian noise.
    spectra = spec_matrix[:,:,1]

    return spectra, noise

def get_aux(aux_data):
    return aux_data.values[:, 1:] # remove planetId (1st column)


def train_valid_split(spectra, aux, noise, targets, train_size=.8, seed=42):

    np.random.seed(seed)

    # generate mask to select training and validation data
    ind = np.zeros(len(spectra), dtype=bool)
    ind[:int(len(spectra)*train_size)] = True
    np.random.shuffle(ind)

    training_spectra, training_aux, training_noise = spectra[ind], aux[ind], noise[ind]
    valid_spectra, valid_aux, valid_noise = spectra[~ind], aux[~ind], noise[~ind]

    # "targets" is not a numpy array (different # samples for each planet) -- use the following for the split
    training_targets_raw = [ targets[i] for i, k in enumerate(ind) if k ]
    valid_targets_raw = [ targets[i] for i, k in enumerate(ind) if not k ]

    return (training_spectra, training_noise, training_aux, training_targets_raw), \
           (valid_spectra, valid_noise, valid_aux, valid_targets_raw), ind
