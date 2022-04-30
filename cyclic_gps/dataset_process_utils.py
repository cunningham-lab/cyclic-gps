import pandas as pd
import numpy as np
import torch
import pickle

PATH_TO_NPY = "../numpy_arrays/"


def load_CO2(path):
    data_np = pd.read_csv(path, comment='#',names=['year','month','decimal date','average','interpolated','trend','mysterycolumn1','mysterycolumn2'],header=0).to_numpy().astype(np.double)
    co2_data = torch.from_numpy(data_np)
    all_ts = co2_data[:, 2]
    all_xs = co2_data[:, 3].unsqueeze(-1)

    ts_standardized = 12*(all_ts-all_ts.min()) # one unit of time = one sample on average
    xs_standardized = all_xs-torch.mean(all_xs)
    xs_standardized = xs_standardized/torch.std(xs_standardized)

    all_ts = ts_standardized
    all_xs = xs_standardized

    train_ts = torch.cat([all_ts[:262], all_ts[502:-28]], dim=0) 
    train_xs = torch.cat([all_xs[:262], all_xs[502:-28]], dim=0) 

    return all_ts, all_xs, train_ts, train_xs


def load_BART(path, dtype=torch.double, load_tensor=False, save=False):
    if load_tensor:
        #'torch_BART_data.pkl'
        with open(PATH_TO_NPY + 'torch_BART_data_unit_time.pkl', 'rb') as f:
            save_dict = pickle.load(f)
        all_ts = save_dict['all_ts']
        all_xs = save_dict['all_xs']
        train_ts = save_dict['train_ts']
        train_xs = save_dict['train_xs']
        if all_xs.dtype != dtype:
            all_ts = all_ts.type(dtype)
            all_xs = all_xs.type(dtype)
            train_ts = train_ts.type(dtype)
            train_xs = train_xs.type(dtype)
        return all_ts, all_xs, train_ts, train_xs
    else:
        df = pd.read_csv(path, names=['day', 'hour', 'origin', 'destination', 'trip count'], header=None, index_col=None)
        if dtype == torch.double or dtype == torch.float64:
            df = df.astype({'hour':'int64', 'trip count': 'int64'})
        else:
            df = df.astype({'hour':'int32', 'trip count': 'int32'})
        print("full data shape: {}".format(df.shape))
        days = pd.date_range(start='1/1/2011', end='1/31/2011').format()
        #days = ['2011-01-01','2011-01-02','2011-01-03','2011-01-04','2011-01-05', '2011-01-06','2011-01-07','2011-01-08','2011-01-09','2011-01-10']
        df = df[df["day"].isin(days)]
        print("selected data shape: {}".format(df.shape))
        embr_arrivals = torch.zeros(size=[len(days) * 24], dtype=dtype) #embarcadero arrivals
        for i, day in enumerate(days):
            print(i)
            for hr in range(24):
                dest_df = df.loc[(df['day'] == day) & (df['hour'] == hr) & (df["destination"] == "EMBR")]
                for j in range(dest_df.shape[0]):
                    embr_arrivals[24 * i + hr] += dest_df['trip count'].iloc[j]
        all_ts = torch.arange(len(days) * 24, dtype=torch.double)
        #all_ts = all_ts * 10
        all_xs = embr_arrivals - torch.mean(embr_arrivals)
        all_xs = all_xs/torch.std(all_xs)
        all_xs = all_xs.unsqueeze(-1)

        train_ts = all_ts[:len(all_ts)//2]
        train_xs = all_xs[:len(all_ts)//2]

        if save:
            save_dict = {"all_ts": all_ts, "all_xs": all_xs, "train_ts": train_ts, "train_xs": train_xs}
            with open(PATH_TO_NPY + 'torch_BART_data_unit_time.pkl', 'wb') as f:
                pickle.dump(save_dict, f)
            
        return all_ts, all_xs, train_ts, train_xs


    
