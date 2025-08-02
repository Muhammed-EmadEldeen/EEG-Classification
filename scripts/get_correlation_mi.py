import pandas as pd
import os
import yaml

with open('../config/config_vars.yaml', 'r') as file:
    config_vars = yaml.safe_load(file)

base_path = config_vars['data']



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

def load_all_trials_df(df, base_path, start, end):
    trials = []
    for i in range(start, end):
        row = df.iloc[i]
        trial_df = load_trial_data(row, base_path)
        trials.append(trial_df)

    return trials


def main():
    train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    validation_df = pd.read_csv(os.path.join(base_path, 'validation.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
    train_x = load_all_trials_df(train_df, base_path,0,2400)
    val_x = load_all_trials_df(validation_df, base_path,0,50)
    test_x = load_all_trials_df(test_df, base_path,0,50)

    num_samples = len(train_x)
    columns = train_x[0].columns
    correlation_table = pd.DataFrame(0.0, index=columns, columns=columns)

    for df in train_x:
        for col1 in columns:
            for col2 in columns:
                corr = df[col1].corr(df[col2])
                if pd.notnull(corr):
                    correlation_table.loc[col1, col2] += corr

    correlation_table /= num_samples

    print(correlation_table.round(3))

if __name__ == "__main__":
    main()
