'''
Based on https://github.com/lindermanlab/elk/blob/main/experiments/fig3/eigenworms.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        datafile: str = "neuralrde",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.datafile = datafile
        self.classification = True
        if datafile == "neuralrde": #MF: not using this datafile
            self.train_file = "neuralrde_split/eigenworms_train.pkl"
            self.val_file = "neuralrde_split/eigenworms_val.pkl"
            self.test_file = "neuralrde_split/eigenworms_test.pkl"
        elif datafile == "lem": #MF: not using this datafile
            self.train_file = "lem_split/eigenworms_train.pkl"
            self.val_file = "lem_split/eigenworms_val.pkl"
            self.test_file = "lem_split/eigenworms_test.pkl"
        elif datafile.startswith("eigenworms"):
            datafile_seed = datafile.split("_")[1]
            self.train_file_X = f"Eigenworms/{datafile_seed}/X_train.pkl"
            self.train_file_y = f"Eigenworms/{datafile_seed}/y_train.pkl"
            self.val_file_X = f"Eigenworms/{datafile_seed}/X_val.pkl"
            self.val_file_y = f"Eigenworms/{datafile_seed}/y_val.pkl"
            self.test_file_X = f"Eigenworms/{datafile_seed}/X_test.pkl"
            self.test_file_y = f"Eigenworms/{datafile_seed}/y_test.pkl"
        elif datafile.startswith("scp1"):
            datafile_seed = datafile.split("_")[1]
            self.train_file_X = f"SelfRegulationSCP1/{datafile_seed}/X_train.pkl"
            self.train_file_y = f"SelfRegulationSCP1/{datafile_seed}/y_train.pkl"
            self.val_file_X = f"SelfRegulationSCP1/{datafile_seed}/X_val.pkl"
            self.val_file_y = f"SelfRegulationSCP1/{datafile_seed}/y_val.pkl"
            self.test_file_X = f"SelfRegulationSCP1/{datafile_seed}/X_test.pkl"
            self.test_file_y = f"SelfRegulationSCP1/{datafile_seed}/y_test.pkl"
        elif datafile.startswith("scp2"):
            datafile_seed = datafile.split("_")[1]
            self.train_file_X = f"SelfRegulationSCP2/{datafile_seed}/X_train.pkl"
            self.train_file_y = f"SelfRegulationSCP2/{datafile_seed}/y_train.pkl"
            self.val_file_X = f"SelfRegulationSCP2/{datafile_seed}/X_val.pkl"
            self.val_file_y = f"SelfRegulationSCP2/{datafile_seed}/y_val.pkl"
            self.test_file_X = f"SelfRegulationSCP2/{datafile_seed}/X_test.pkl"
            self.test_file_y = f"SelfRegulationSCP2/{datafile_seed}/y_test.pkl"
        elif datafile.startswith("motor"):
            datafile_seed = datafile.split("_")[1]
            self.train_file_X = f"MotorImagery/{datafile_seed}/X_train.pkl"
            self.train_file_y = f"MotorImagery/{datafile_seed}/y_train.pkl"
            self.val_file_X = f"MotorImagery/{datafile_seed}/X_val.pkl"
            self.val_file_y = f"MotorImagery/{datafile_seed}/y_val.pkl"
            self.test_file_X = f"MotorImagery/{datafile_seed}/X_test.pkl"
            self.test_file_y = f"MotorImagery/{datafile_seed}/y_test.pkl"
        elif datafile.startswith("heartbeat"):
            datafile_seed = datafile.split("_")[1]
            self.train_file_X = f"Heartbeat/{datafile_seed}/X_train.pkl"
            self.train_file_y = f"Heartbeat/{datafile_seed}/y_train.pkl"
            self.val_file_X = f"Heartbeat/{datafile_seed}/X_val.pkl"
            self.val_file_y = f"Heartbeat/{datafile_seed}/y_val.pkl"
            self.test_file_X = f"Heartbeat/{datafile_seed}/X_test.pkl"
            self.test_file_y = f"Heartbeat/{datafile_seed}/y_test.pkl"
        elif datafile.startswith("ethanol"):
            datafile_seed = datafile.split("_")[1]
            self.train_file_X = f"EthanolConcentration/{datafile_seed}/X_train.pkl"
            self.train_file_y = f"EthanolConcentration/{datafile_seed}/y_train.pkl"
            self.val_file_X = f"EthanolConcentration/{datafile_seed}/X_val.pkl"
            self.val_file_y = f"EthanolConcentration/{datafile_seed}/y_val.pkl"
            self.test_file_X = f"EthanolConcentration/{datafile_seed}/X_test.pkl"
            self.test_file_y = f"EthanolConcentration/{datafile_seed}/y_test.pkl"
        else:
            raise RuntimeError()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if not self.datafile.startswith("neuralrde"):
            with open(self.train_file_X, "rb") as f:
                x_train = pickle.load(f)
                #print(x_train.shape, type(x_train))
                x_train = x_train.tolist()
                x_train = np.array(x_train) # (nseq, nsequence, ninp) it includes timestep as the first index in the last dim (165, 17984, 7) 
                x_train = x_train[:, :, 1:] # remove the timestep
                x_train = torch.from_numpy(x_train)
                print(x_train.shape, type(x_train)) 
                # torch.Size([165, 17984, 6]) eigenworms
                # torch.Size([392, 896, 6]) scp1
                # torch.Size([266, 1152, 7]) scp2
                # torch.Size([286, 405, 61]) heartbeat
                # torch.Size([366, 1751, 2]) ethanol
                # torch.Size([264, 3000, 63]) motor
                # print(x_train[:10,0,:])
            with open(self.train_file_y, "rb") as f:
                y_train = pickle.load(f)
                y_train = np.array(y_train.tolist())  # (nseq, nclass) (165, 5)
                if self.classification:
                    y_train = [np.where(i==1)[0][0] for i in y_train]
                y_train = np.array(y_train)
                y_train = torch.tensor(y_train)
                print(y_train.shape, type(y_train)) #torch.Size([165])
            with open(self.val_file_X, "rb") as f:
                x_val = pickle.load(f)
                x_val = x_val.tolist()
                x_val = np.array(x_val)
                x_val = x_val[:, :, 1:]
                x_val = torch.from_numpy(x_val)
            with open(self.val_file_y, "rb") as f:
                y_val = pickle.load(f)
                y_val = np.array(y_val.tolist())
                if self.classification:
                    y_val = [np.where(i==1)[0][0] for i in y_val]
                y_val = np.array(y_val)
                y_val = torch.tensor(y_val)
            with open(self.test_file_X, "rb") as f:
                x_test = pickle.load(f)
                x_test = x_test.tolist()
                x_test = np.array(x_test)
                x_test = x_test[:, :, 1:]
                x_test = torch.from_numpy(x_test)
            with open(self.test_file_y, "rb") as f:
                y_test = pickle.load(f)
                y_test = np.array(y_test.tolist())
                if self.classification:
                    y_test = [np.where(i==1)[0][0] for i in y_test]
                y_test = np.array(y_test)
                y_test = torch.tensor(y_test)
            self._train_dataset = TensorDataset(x_train, y_train)
            self._val_dataset = TensorDataset(x_val, y_val)
            self._test_dataset = TensorDataset(x_test, y_test)
        else:
            with open(self.train_file, "rb") as f:
                if self.datafile == "neuralrde":
                    x, y = pickle.load(f)
                    self._train_dataset = TensorDataset(x, y)
                elif self.datafile == "lem":
                    self._train_dataset = pickle.load(f)
                else:
                    raise RuntimeError()
            with open(self.val_file, "rb") as f:
                if self.datafile == "neuralrde":
                    x, y = pickle.load(f)
                    self._val_dataset = TensorDataset(x, y)
                elif self.datafile == "lem":
                    self._val_dataset = pickle.load(f)
                else:
                    raise RuntimeError()
            with open(self.test_file, "rb") as f:
                if self.datafile == "neuralrde":
                    x, y = pickle.load(f)
                    self._test_dataset = TensorDataset(x, y)
                elif self.datafile == "lem":
                    self._test_dataset = pickle.load(f)
                else:
                    raise RuntimeError()
        print("LEN TRAIN DATASET", len(self._train_dataset))
        print("LEN VAL DATASET", len(self._val_dataset))
        print("LEN TEST DATASET", len(self._test_dataset))

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
        return test_dataloader

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch