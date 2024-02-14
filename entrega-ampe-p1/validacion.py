import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import sys
import time
from AMPE_dnn import My_DNN  # Asumiendo que My_DNN está en AMPE dnn.py

# Paso 0: Comprobación de que argumentos sean números enteros

if(len(sys.argv) != 2):
    print("Uso: ./entrenamiento.py fichero.pt")
    exit(1)

#Comprobar que la extensión del fichero sea .pt
if not sys.argv[1].endswith('.pt'):
    print("El fichero debe tener extensión .pt")
    exit(1)

# Paso 1: Preparar los DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
valset = datasets.MNIST('./data', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# Paso 2: Cargar la Red Neuronal
model = My_DNN()

# Paso 3: Cargar los Pesos Entrenados
model.load_state_dict(torch.load(sys.argv[1]))

# Paso 4: Proceso de Validación

correct_count, all_count = 0, 0
for images,labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if (true_label == pred_label):
                correct_count += 1
            all_count += 1
print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))