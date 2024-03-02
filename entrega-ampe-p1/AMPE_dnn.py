import torch.nn as nn
import torch.nn.functional as F

class My_DNN(nn.Module):
    def __init__(self):
        super(My_DNN, self).__init__()
        # Definir la primera capa fully-connected
        self.fc1 = nn.Linear(784, 128)
        # Definir la segunda capa fully-connected
        self.fc2 = nn.Linear(128, 64)
        # Definir la tercera capa fully-connected
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Aplicar la primera capa y la función de activación ReLU
        x = F.relu(self.fc1(x))
        # Aplicar la segunda capa y la función de activación ReLU
        x = F.relu(self.fc2(x))
        # Aplicar la tercera capa
        x = self.fc3(x)
        # Aplicar la función LogSoftMax a la salida
        return F.log_softmax(x, dim=1)
