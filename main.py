import matplotlib.pyplot as plt
import pandas as pd
import torch
from mlxtend.plotting import plot_confusion_matrix
from pylab import rcParams
from sklearn.metrics import confusion_matrix
from torch import nn
from tqdm.auto import tqdm

from src.data.make_dataset import create_dataset
from src.model.model import Decoder, Encoder
from src.model.predict_model import predict, predict_new
from src.visualization.visualize import plot_prediction

classes = ["N", "L", "R"]

CLASS_N = 0
CLASS_L = 1
CLASS_R = 2

N_MODEL_PATH = "checkpoints/chkpt_v2/n_model.pth"
L_MODEL_PATH = "checkpoints/chkpt_v2/l_model.pth"
R_MODEL_PATH = "checkpoints/chkpt_v2/r_model.pth"

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


df = pd.read_csv("notebooks/t.csv")

input_df = df.drop(labels="target", axis=1)
input_dataset, seq_len, n_features = create_dataset(input_df)

n_model = RecurrentAutoencoder(seq_len, n_features, embedding_dim=120).to(device)
l_model = RecurrentAutoencoder(seq_len, n_features, embedding_dim=120).to(device)
r_model = RecurrentAutoencoder(seq_len, n_features, embedding_dim=120).to(device)

n_model = torch.load(N_MODEL_PATH, map_location=torch.device("cpu"))
l_model = torch.load(L_MODEL_PATH, map_location=torch.device("cpu"))
r_model = torch.load(R_MODEL_PATH, map_location=torch.device("cpu"))


# accuracy calculation
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def predn(input_dataset):
    with torch.no_grad():

        y_pred = []
        for seq_true in tqdm(input_dataset):

            loss = []

            _, pred_losses_n = predict_new(n_model, [seq_true])
            _, pred_losses_l = predict_new(l_model, [seq_true])
            _, pred_losses_r = predict_new(r_model, [seq_true])

            loss_n = torch.mean(torch.tensor(pred_losses_n)).item()
            loss_l = torch.mean(torch.tensor(pred_losses_l)).item()
            loss_r = torch.mean(torch.tensor(pred_losses_r)).item()

            loss.append(loss_n)
            loss.append(loss_l)
            loss.append(loss_r)

            min_index = loss.index(min(loss))

            if min_index == 0:
                y_pred.append(CLASS_N)
            elif min_index == 1:
                y_pred.append(CLASS_L)
            elif min_index == 2:
                y_pred.append(CLASS_R)
    return y_pred


n_model = n_model.eval()
l_model = l_model.eval()
r_model = r_model.eval()

y_true = df["target"]
y_pred = predn(input_dataset)
y_true = y_true.values
y_true = [int(element) for element in y_true]

acc = accuracy_fn(torch.tensor(y_true), torch.tensor(y_pred))
print(f"Accuracy: {acc}%")


# # confusion matrix
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


# # Visualization
# fig, axs = plt.subplots(nrows=5, ncols=6, sharey=True, sharex=True, figsize=(22, 8))
# for i, data in enumerate(input_dataset[:6]):
#     plot_prediction(data, n_model, title="Normal Beat", ax=axs[0, i])
#     plot_prediction(data, l_model, title="Left Bundle Branch Block", ax=axs[1, i])
#     plot_prediction(data, r_model, title="Right Bundle Branch Block", ax=axs[2, i])
#     plot_prediction(data, v_model, title="Premature Ventricular Contraction", ax=axs[3, i])
#     plot_prediction(data, p_model, title="Paced Beat", ax=axs[4, i])
# plt.show()


# # Loss comparison
# _, pred_losses_n = predict(n_model, input_dataset[:300])
# _, pred_losses_l = predict(l_model, input_dataset[:300])
# _, pred_losses_r = predict(r_model, input_dataset[:300])
# _, pred_losses_v = predict(v_model, input_dataset[:300])
# _, pred_losses_p = predict(p_model, input_dataset[:300])

# loss_n = torch.mean(torch.tensor(pred_losses_n)).item()
# loss_l = torch.mean(torch.tensor(pred_losses_l)).item()
# loss_r = torch.mean(torch.tensor(pred_losses_r)).item()
# loss_v = torch.mean(torch.tensor(pred_losses_v)).item()
# loss_p = torch.mean(torch.tensor(pred_losses_p)).item()

# print(f"Normal Beat: {loss_n}")
# print(f"Left Bundle Branch Block: {loss_l}")
# print(f"Right Bundle Branch Block: {loss_r}")
# print(f"Premature Ventricular Contraction: {loss_v}")
# print(f"Paced Beat: {loss_p}")
