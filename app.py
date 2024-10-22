from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

app = Flask(__name__)

# URL de Google Drive del modelo
url_modelo = 'https://drive.google.com/uc?id=1Z2nbxLZ04TVfupKLNdKg1LTgJS3C5cuz'

# Descargar el modelo desde Google Drive y guardarlo en un archivo local
def descargar_modelo(url):
    output = 'EmotionAI.h5'
    gdown.download(url, output, quiet=False)
    return tf.keras.models.load_model(output)

# Cargar el modelo al iniciar la aplicación
modelo = descargar_modelo(url_modelo)

# Ruta para la página principal
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No se subió ningún archivo"

        file = request.files['file']

        if file.filename == '':
            return "No se seleccionó ningún archivo"

        # Procesar la imagen
        if file:
            image = Image.open(file).convert('L')
            image = image.resize((96, 96))
            image_array = np.array(image) / 255.0

            # Preprocesar la imagen
            image_array = np.expand_dims(image_array, axis=0)
            image_array = np.expand_dims(image_array, axis=-1)
            image_array = np.repeat(image_array, 3, axis=-1)  # Convertir a 3 canales

            # Realizar la predicción
            prediccion = modelo.predict(image_array)
            emocion = np.argmax(prediccion)

            # Imprimir la predicción y el índice
            print(f"Predicción: {prediccion}, Índice de emoción: {emocion}, Total de clases: {len(clases_emociones)}")

            # Emociones (ajusta según tu dataset)
            clases_emociones = ['Feliz', 'Triste', 'Enojado', 'Sorprendido', 'Neutral']
            
            # Verificar que el índice de emoción sea válido
            if emocion < len(clases_emociones):
                emocion_predicha = clases_emociones[emocion]
            else:
                emocion_predicha = "Emoción desconocida"

            # Mostrar el resultado
            return render_template('index.html', emocion=emocion_predicha)

    return render_template('index.html')

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
