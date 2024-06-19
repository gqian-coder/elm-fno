import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from modulus.models.mlp.fully_connected import FullyConnected
from modulus.models.fno import FNO
import modulus

class pred_FNO(modulus.Module):
    def __init__(self, fno_, in_features=10, out_features=10):
        super(pred_FNO, self).__init__()

        self.fno = fno_
        self.fc  = FullyConnected(in_features=in_features, out_features=out_features) 

    def forward(self, x):
        x1 = self.fno(x)
        x2 = self.fc(x1)
        return x2 


def loadBesElmData(filename, ts_range):
    file = h5py.File(filename, 'r')
    events_name = list(file.keys())
    n_events    = len(events_name)
    dset_       = file[events_name[0]]
    n_channels  = np.array(dset_['signals']).shape[0]
    ts_elm  = 0
    for i in range(n_events):
        dset_  = file[events_name[i]]
        print(i, events_name[i])
        labels = np.array(dset_['labels'])
        elm_id = np.where(labels==1)[0]
        ts_elm += elm_id.max() - elm_id.min() + 1 
    print("total timesteps of ELM signals: {}".format(ts_elm))
    
    # save the 0.5*ts_range timestep data before and after elm signals
    signals = np.zeros([n_channels, ts_elm+n_events*ts_range]) 
    events_ = np.zeros(n_events+1)
    intervl = int(ts_range/2)
    init_ts = 0

    for i in range(n_events):
        dset_  = file[events_name[i]]
        siga_  = np.array(dset_['signals'])
        label  = np.array(dset_['labels'])
        elm_id = np.where(label==1)[0]
        min_ts, max_ts = elm_id.min(), elm_id.max()
        ts     = max_ts - min_ts 
        elm_ed = min(label.shape[0], max_ts+intervl+1)
        elm_st = max(0, min_ts-intervl)
        elm_dr = elm_ed - elm_st 
        print(ts, min_ts, max_ts, init_ts, init_ts+ts+ts_range, elm_st, elm_ed)
        print(siga_.shape, label.shape)
        signals[:, init_ts:init_ts+elm_dr] = siga_[:, elm_st:elm_ed] 
        events_[i] = init_ts
        init_ts += elm_dr 
        print(i, init_ts)
    events_[-1] = init_ts
    signals = signals[:, :init_ts+1]
    signals.tofile('signals_elm_std_10ts.bin') 
    events_.tofile('event_sep_std_10ts.bin')

class BESDataset(Dataset):
    # test_ratio: means % percentage of total data
    #             > 0 means taking from the begining
    #             < 0 means taking from the back
    def __init__(self, datafile, sepfile, n_channels, obsv_ts, pred_ts, test_ratio, normaliser):
        self.dataset = np.fromfile(datafile).astype('float32')
        self.datasep = np.fromfile(sepfile).astype('int')
        print(self.datasep[-1], self.dataset.shape)
        pesudo_dlen  = int(self.dataset.shape[0]/n_channels)
        nx = ny = int(np.sqrt(n_channels))
        self.dataset = np.reshape(self.dataset, [nx, ny, pesudo_dlen])[:, :self.datasep[-1]+1]
        self.obsv_ts = obsv_ts
        self.pred_ts = pred_ts
        self.nx      = nx
        self.ny      = ny
        n_events     = len(self.datasep)-1
        if (test_ratio>0):
            start_event = 0
            end_event   = int(test_ratio * n_events)
        else:
            start_event = int((1.0+test_ratio) * n_events)
            end_event   = n_events 
        dlen = self.__get_len__(test_ratio)
        print("total data slices: {}, occupying {} GB mem".format(dlen, dlen*n_channels*(obsv_ts+pred_ts)*4/1e9))
        
        data_mean = np.mean(self.dataset)
        data_std  = np.std(self.dataset)
        print("mean and std across events: ", data_mean, data_std)
        dmax = np.max(self.dataset)
        dmin = np.min(self.dataset)
        print("min and max across events: ", dmin, dmax)

        if (pred_ts==1):
            self.data_obsv = np.zeros([dlen, obsv_ts, nx, ny]).astype('float32')
            self.data_pred = np.zeros([dlen, 1      , nx, ny]).astype('float32')
            idx = 0
            for i in range(start_event, end_event):
                wd   = self.datasep[i+1]
                dp   = self.datasep[i]
                # normalize the data by min and max values per channel, per each event
                #dmax = np.max(self.dataset[:,:, dp:wd])
                #dmin = np.min(self.dataset[:,:, dp:wd])
                #print(dmax, dmin)
                while (dp+obsv_ts+1<wd):
                    #print(i, idx, wd, dp)
                    dp_cut = dp+obsv_ts
                    if (normaliser == 'normalization'):
                        self.data_obsv[idx, :, :, :] = (np.moveaxis(self.dataset[:,:, dp:dp_cut], [0,1,2], [1,2,0]) - dmin) / (dmax-dmin) - 0.5
                        self.data_pred[idx, 0, :, :] = (self.dataset[:,:, dp_cut] - dmin) / (dmax-dmin) - 0.5 
                    else:
                        self.data_obsv[idx, :, :, :] = (np.moveaxis(self.dataset[:,:, dp:dp_cut], [0,1,2], [1,2,0]) - data_mean) / data_std 
                        self.data_pred[idx, 0, :, :] = (self.dataset[:,:, dp_cut] - data_mean) / data_std 
                    dp  += 1
                    idx += 1
        else:
            self.data_obsv = np.zeros([dlen, obsv_ts, nx, ny]).astype('float32')
            self.data_pred = np.zeros([dlen, 1      , nx, ny]).astype('float32')
            idx = 0
            for i in range(start_event, end_event):
                wd   = self.datasep[i+1]
                dp   = self.datasep[i]
                # normalize the data by min and max values per each event
                dmax = np.max(self.dataset[:,:, dp:wd])
                dmin = np.min(self.dataset[:,:, dp:wd])
                #print(dmax, dmin)
                while (dp+obsv_ts+pred_ts<wd):
                    #print(i, idx, wd, dp)
                    dp_cut = dp+obsv_ts
                    if (normaliser == 'normalization'):
                        self.data_obsv[idx, :, :, :] = (np.moveaxis(self.dataset[:,:, dp:dp_cut], [0,1,2], [1,2,0]) - dmin) / (dmax-dmin) - 0.5
                        self.data_pred[idx, :, :, :] = (self.dataset[:,:, dp_cut+pred_ts-1] - dmin) / (dmax-dmin) - 0.5
                    else:
                        self.data_obsv[idx, :, :, :] = (np.moveaxis(self.dataset[:,:, dp:dp_cut], [0,1,2], [1,2,0]) - data_mean) / data_std
                        self.data_pred[idx, :, :, :] = (self.dataset[:,:, dp_cut+pred_ts-1] - data_mean) / data_std
                    dp  += 1
                    idx += 1


    # data: [t0,...,t_obsv], [t{pred},...,t_{obsv+pred}], [t_{2*pred},...,t_{obsv+2*pred}]
    # target: [t_{obsv+1},...,t_{t_obsv+pred}], [t_{obsv+pred+1},...,t_{obsv+2*pred}], [t_{2*pred+1},...,t_{3*pred}]
    def __getitem__(self, index):
        data   = torch.from_numpy(self.data_obsv[index,:,:,:]).type('torch.FloatTensor')
        target = torch.from_numpy(self.data_pred[index,:,:,:]).type('torch.FloatTensor')
    
        return {'data': data, 'target': target}
    
    def __len__(self):
        return self.data_obsv.shape[0]

    def __get_len__(self, test_ratio):
        data_len = 0
        n_events = len(self.datasep)-1

        if (test_ratio>0):
            start_event = 0
            end_event   = int(test_ratio * n_events)
        else:
            start_event = int((1.0+test_ratio) * n_events)
            end_event   = n_events

        for i in range(start_event, end_event):
            # ignore the extra timesteps (<pred_ts) at the end
            data_len += int(np.floor(self.datasep[i+1]-self.datasep[i] - self.obsv_ts))
        return data_len
'''
loadBesElmData('../d3d_bes_elm_dataset/labeled_elm_events.hdf5', 20)
obsv_ts, pred_ts = 5, 1
n_channels = 64
dataset    = BESDataset("signals_elm_10ts.bin", "event_sep_10ts.bin", n_channels, obsv_ts, pred_ts)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)
print("number of batches: ", enumerate(dataloader))
for i_batch, d_batch in enumerate(dataloader):
    data, pred = d_batch['data'], d_batch['target']
    print(data.shape, pred.shape)
    break
'''
