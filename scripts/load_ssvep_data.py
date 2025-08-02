import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
import yaml


with open('../config/config_vars.yaml', 'r') as file:
    config_vars = yaml.safe_load(file)

base_path = config_vars['data']

train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
validation_df = pd.read_csv(os.path.join(base_path, 'validation.csv'))
test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))

label_to_index = {
    "Left": 0,
    "Right": 1,
    "Forward":2,
    "Backward":3
}


ssvep_channels = ['PO8', 'C4', 'FZ','C3','Time','AccX' ,  'AccY' ,  'AccZ' , 'Gyro1'  ,'Gyro2'  ,'Gyro3']

def load_trial_data(row, base_path='.'):
    id_num = row['id']
    if id_num <= 4800:
        dataset = 'train'
    elif id_num <= 4900:
        dataset = 'validation'
    else:
        dataset = 'test'

    eeg_path = f"{base_path}/{row['task']}/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
    eeg_data = pd.read_csv(eeg_path)

    trial_num = int(row['trial'])
    samples_per_trial = 1750 if row['task'] == 'SSVEP' else 2250
    start_idx = (trial_num - 1) * samples_per_trial
    end_idx = start_idx + samples_per_trial - 1
    return eeg_data.iloc[start_idx:end_idx + 1]

def load_all_trials(df, base_path, channels, start, end):
    trials = []
    for i in range(start, end):
        row = df.iloc[i]
        trial_df = load_trial_data(row, base_path)[channels].to_numpy()
        trials.append(trial_df)

    trials = np.stack(trials)
    return trials



class SequenceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return input sequence and label
        return self.data[idx], self.labels[idx]



def ssvep_dataloaders():

    train_x = load_all_trials(train_df, base_path, ssvep_channels,2400,4800)
    val_x = load_all_trials(validation_df, base_path, ssvep_channels,50,100)
    test_x = load_all_trials(test_df, base_path, ssvep_channels,50,100)

    train_labels=pd.read_csv(base_path+"train.csv")
    train_labels = train_labels[0:2400]

    val_labels=pd.read_csv(base_path+"validation.csv")
    val_labels = val_labels[0:50]


    train_labels = train_labels.iloc[:,-1]
    val_labels = val_labels.iloc[:,-1]

    train_labels = [label_to_index[label] for label in train_labels]
    val_labels = [label_to_index[label] for label in val_labels]

    train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)  # shape: [N, 1]
    val_labels = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1)

    train_x = torch.from_numpy(train_x)
    val_x = torch.from_numpy(val_x)
    train_dataset = SequenceDataset(train_x, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = SequenceDataset(val_x, val_labels)
    val_loader=DataLoader(val_dataset, batch_size=32, shuffle=True)

    return train_loader,val_loader
