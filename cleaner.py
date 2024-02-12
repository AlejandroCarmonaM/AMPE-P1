import argparse
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

def enhanced_preprocess_image(image_path, output_path, threshold=200, contrast_factor=2, edge_enhance_factor=1.5):
    # Cargar la imagen y convertir a escala de grises
    img = Image.open(image_path).convert('L')
    
    # Ajustar el contraste
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    
    # Aplicar umbral para claros y oscuros
    img = img.point(lambda p: 255 if p > threshold else p-20 if p-20 > 0 else 0)
    
    # Aplicar filtro de realce de bordes
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    # Invertir colores para coincidir con MNIST
    img = ImageOps.invert(img)
    
    # Opcional: Ajustar nuevamente el contraste y la nitidez después de la inversión
    img = ImageEnhance.Contrast(img).enhance(edge_enhance_factor)  # Ajustar al gusto

    # Guardar o retornar la imagen procesada
    img.save(output_path)
    
    return img


def main():
    # Crear el analizador de argumentos
    parser = argparse.ArgumentParser(description='Preprocess images for MNIST-like format.')
    
    # Añadir argumentos
    parser.add_argument('input_path', type=str, help='Input image file path')
    parser.add_argument('output_path', type=str, help='Output image file path')
    parser.add_argument('--threshold', type=int, default=200, help='Threshold for converting light colors to white and darkening others')
    
    # Parsear los argumentos
    args = parser.parse_args()
    
    # Llamar a la función de preprocesamiento con los argumentos
    enhanced_preprocess_image(args.input_path, args.output_path, args.threshold)

if __name__ == '__main__':
    main()
