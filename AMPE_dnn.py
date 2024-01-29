# Importamos las bibliotecas necesarias
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch import optim
from time import time

# Definición del modelo de red neuronal
class My_DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

# Función para entrenar el modelo
def train(model, trainloader, criterion, optimizer, epochs=15):
    time0 = time()
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"Epoch {e} - Training loss: {running_loss/len(trainloader)}")
    print(f"\nTraining Time (in minutes) = {(time()-time0)/60}")

# Función para guardar los pesos del modelo
def save_model_weights(model, path='./my_weights.pt'):
    torch.save(model.state_dict(), path)

# Función principal
def main():
    # Transformaciones para el conjunto de datos
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Carga de los conjuntos de datos
    trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Creación del modelo, definición de la función de pérdida y el optimizador
    model = My_DNN()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    # Entrenamiento del modelo
    train(model, trainloader, criterion, optimizer)

    # Guardar los pesos del modelo
    save_model_weights(model)

if __name__ == '__main__':
    main()
