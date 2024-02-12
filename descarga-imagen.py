import torchvision.datasets as datasets
from PIL import Image
import random

# Descargar el dataset MNIST
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

# Obtener 10 índices aleatorios
indices = random.sample(range(len(mnist_trainset)), 10)

for idx in indices:
    # Acceder a una imagen y su etiqueta
    image, label = mnist_trainset[idx]
    
    # No es necesario convertir a PIL Image, ya que ya es una imagen PIL
    
    # Guardar la imagen
    image_path = f"./digit_{label}_{idx}.png"
    image.save(image_path)
    
    print(f"Imagen guardada en: {image_path}")
