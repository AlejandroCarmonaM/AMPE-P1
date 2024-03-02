import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import sys
import time
from AMPE_dnn import My_DNN  # Asumiendo que My_DNN está en AMPE dnn.py

# Paso 0: Comprobación de que argumentos sean números enteros

if(len(sys.argv) != 2):
    print("Uso: ./entrenamiento.py num_epochs")
    exit(1)
if (not sys.argv[1].isdigit()):
    print("El argumento debe ser un número entero")
    exit(1)

# Paso 1: Preparar los DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Paso 2: Cargar la Red Neuronal
model = My_DNN()

# Paso 3: Definir la Función de Pérdida y el Optimizador
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

# Paso 4: Proceso de Entrenamiento

epochs = int(sys.argv[1])
start_time = time.time()
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

# Paso 5: Guardar los Pesos Entrenados
torch.save(model.state_dict(), f'pesos-{epochs}.pt')
end_time = time.time()

# Paso 6: Imprimir el tiempo de entrenamiento
print(f"Tiempo de entrenamiento para {epochs} épocas: {end_time - start_time} segundos")
