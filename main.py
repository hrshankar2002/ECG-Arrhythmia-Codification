import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import rc
from mlxtend.plotting import plot_confusion_matrix
from pylab import rcParams
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torchmetrics import ConfusionMatrix
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
from lib.arff2pandas import arff2pandas as a2p
from src.data.make_dataset import create_dataset
from src.model.model import Decoder, Encoder
from src.model.predict_model import predict, predict_new
from src.visualization.visualize import plot_prediction

classes = ['Normal', 'R on T', 'PVC', 'SP']

CLASS_NORMAL = 1
CLASS_R_ON_T = 2
CLASS_PVC = 3
CLASS_SP = 4
# CLASS_UB = 5

NORMAL_MODEL_PATH = "models/normal_model.pth"
PVC_MODEL_PATH = "models/pvc_model.pth"
R_ON_T_MODEL_PATH = "models/r_on_t_model.pth"
SP_MODEL_PATH = "models/sp_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


with open("data/ECG5000_TRAIN.arff") as f:
    train = a2p.load(f)
with open("data/ECG5000_TEST.arff") as f:
    test = a2p.load(f)

df = pd.concat([train, test]).sample(frac=1.0)

# input_df = df[df.target == str(CLASS_SP)].drop(labels='target', axis=1)
y_true = df[df['target'] != '5']['target']

input_df = df.drop(labels='target', axis=1)
input_dataset, seq_len, n_features = create_dataset(input_df)

normal_model = RecurrentAutoencoder(seq_len, n_features, embedding_dim=120).to(device)
r_on_t_model = RecurrentAutoencoder(seq_len, n_features, embedding_dim=120).to(device)
pvc_model = RecurrentAutoencoder(seq_len, n_features, embedding_dim=120).to(device)
sp_model = RecurrentAutoencoder(seq_len, n_features, embedding_dim=120).to(device)

normal_model = torch.load(NORMAL_MODEL_PATH, map_location=torch.device("cpu"))
r_on_t_model = torch.load(R_ON_T_MODEL_PATH, map_location=torch.device("cpu"))
pvc_model = torch.load(PVC_MODEL_PATH, map_location=torch.device("cpu"))
sp_model = torch.load(SP_MODEL_PATH, map_location=torch.device("cpu"))








# accuracy calculation
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def predn(input_dataset):
    with torch.no_grad():
        
        y_pred=[]
        for seq_true in tqdm(input_dataset):
            
            loss = []
    
            _, pred_losses_normal = predict_new(normal_model, [seq_true])
            _, pred_losses_rt = predict_new(r_on_t_model, [seq_true])
            _, pred_losses_pvc = predict_new(pvc_model, [seq_true])
            _, pred_losses_sp = predict_new(sp_model, [seq_true])

            loss_normal = torch.mean(torch.tensor(pred_losses_normal)).item()
            loss_rt = torch.mean(torch.tensor(pred_losses_rt)).item()
            loss_pvc = torch.mean(torch.tensor(pred_losses_pvc)).item()
            loss_sp = torch.mean(torch.tensor(pred_losses_sp)).item()
                
            loss.append(loss_normal)
            loss.append(loss_rt)
            loss.append(loss_pvc)
            loss.append(loss_sp)
            
            min_index = loss.index(min(loss))
            
            if min_index == 0:
                y_pred.append(CLASS_NORMAL)
            elif min_index == 1:
                y_pred.append(CLASS_R_ON_T)
            elif min_index == 2:
                y_pred.append(CLASS_PVC)
            else:
                y_pred.append(CLASS_SP)
    return y_pred


normal_model = normal_model.eval()
r_on_t_model = r_on_t_model.eval()
pvc_model = pvc_model.eval()
sp_model = sp_model.eval()

y_pred = predn(input_dataset)
y_true = y_true.values
y_true = [int(element) for element in y_true]

acc = accuracy_fn(torch.tensor(y_true), torch.tensor(y_pred))
print(f"Accuracy: {acc}%")



# confusion matrix
# y_pred = (torch.tensor(y_pred))
# y_true = (torch.tensor(y_true))

# y_true = y_true.numpy()
# y_pred = y_pred.numpy()

# cm = confusion_matrix(y_true, y_pred)

# fig, ax = plot_confusion_matrix(
#     conf_mat=cm, 
#     class_names=classes,
#     figsize=(5,5))
# plt.show()





# Visualization
# fig, axs = plt.subplots(nrows=4, ncols=6, sharey=True, sharex=True, figsize=(22, 8))
# for i, data in enumerate(input_dataset[:6]):
#     plot_prediction(data, normal_model, title="Normal", ax=axs[0, i])
#     plot_prediction(data, r_on_t_model, title="R on T", ax=axs[1, i])
#     plot_prediction(data, pvc_model, title="PVC", ax=axs[2, i])
#     plot_prediction(data, sp_model, title="SP", ax=axs[3, i])
# plt.show()

# Loss comparison
# _, pred_losses_normal = predict(normal_model, input_dataset[:300])
# _, pred_losses_rt = predict(r_on_t_model, input_dataset[:300])
# _, pred_losses_pvc = predict(pvc_model, input_dataset[:300])
# _, pred_losses_sp = predict(sp_model, input_dataset[:300])

# loss_normal = torch.mean(torch.tensor(pred_losses_normal)).item()
# loss_rt = torch.mean(torch.tensor(pred_losses_rt)).item()
# loss_pvc = torch.mean(torch.tensor(pred_losses_pvc)).item()
# loss_sp = torch.mean(torch.tensor(pred_losses_sp)).item()

# print(f"Normal: {loss_normal}")
# print(f"R on T: {loss_rt}")
# print(f"PVC: {loss_pvc}")
# print(f"SP: {loss_sp}")