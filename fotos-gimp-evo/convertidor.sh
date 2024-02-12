#!/bin/bash

# Crear un directorio para las imágenes resultantes si aún no existe
mkdir -p converted_images

# Bucle a través de todos los archivos de imagen en el directorio actual
for img in *.jpg *.png; do
  # Asegurarse de que el bucle no intente procesar si no hay archivos
  if [ -e "$img" ]; then
    # Nombre del archivo de destino, preservando la extensión original
    dst="converted_images/$(basename "$img")"
    # Usar ImageMagick para cambiar el tamaño a 28x28 y convertir a escala de grises
    convert "$img" -resize 28x28! -colorspace Gray "$dst"
    echo "Convertido: $img -> $dst"
  fi
done

echo "Conversión completada."
