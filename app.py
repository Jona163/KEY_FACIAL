from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Ruta del modelo
MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/EmotionAI.h5"  # Asegúrate de que este sea el archivo .h5
model = load_model(MODEL_PATH)

def preprocess_image(image):
    # Convertir la imagen a escala de grises, cambiar tamaño y normalización
    image = image.convert("L").resize((96, 96))
    image = np.array(image)
    image = image / 255.0  # Normalización
    image = np.expand_dims(image, axis=-1)
    return np.expand_dims(image, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    # Procesar la imagen subida
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))
    preprocessed_image = preprocess_image(img)
    
    # Hacer predicción con el modelo
    preds = model.predict(preprocessed_image)[0]
    
    # Devolver las coordenadas de los puntos faciales
    return jsonify({'keypoints': preds.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
