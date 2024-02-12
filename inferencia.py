import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import sys
import time
from PIL import Image
import numpy as np
from AMPE_dnn import My_DNN  # Asumiendo que My_DNN está en AMPE dnn.py

if __name__ == "__main__":

    
    if(len(sys.argv) != 3):
        print("Uso: ./inferencia.py ruta-a-imagen fichero.pt")
        exit(1)

    if not sys.argv[1].endswith('.png') and not sys.argv[1].endswith('.jpg'):
        print("No has indicado la ruta a una imagen.\nUso: ./entrenamiento.py ruta-a-imagen fichero.pt")
        exit(1)

    # Comprobar que la extensión del fichero sea .pt
    if not sys.argv[2].endswith('.pt'):
        print("No has indicado la ruta a un fichero .pt\nEl fichero debe tener extensión .pt\nUso: ./entrenamiento.py ruta-a-imagen fichero.pt")
        exit(1)

    # Paso 1: Cargar la Red Neuronal
    model = My_DNN()

    # Paso 2: Cargar los Pesos Entrenados
    model.load_state_dict(torch.load(sys.argv[2]))

    # Paso 3: 
    image = Image.open(sys.argv[1]).convert('L')
    #imprime la info de la imagen
    print(image)

    # Transformaciones para la imagen
    transformaciones = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    imagen = transformaciones(image)
    
    # Paso 4: Proceso de Inferencia
    with torch.no_grad():
        logps = model(imagen.view(1, 784))
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        print(f"El dígito es: {pred_label}")