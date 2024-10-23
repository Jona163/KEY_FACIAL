from flask import Flask, request, render_template
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib
import os

app = Flask(__name__)

# Ruta donde se guardarán las imágenes procesadas
OUTPUT_FOLDER = 'static'
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'  # Ruta al predictor de dlib

# Verificar si la carpeta de salida existe, si no, crearla
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Cargar el detector de caras y el predictor de puntos faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file part'
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Guardar el archivo subido
        original_image_path = os.path.join(OUTPUT_FOLDER, file.filename)
        file.save(original_image_path)

        # Procesar la imagen
        image = cv2.imread(original_image_path)

        # Detección de rostros
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_image)
        print(f"Número de rostros detectados: {len(faces)}")  # Verificar rostros detectados

        for face in faces:
            # Predecir puntos faciales
            landmarks = predictor(gray_image, face)
            print(f"Puntos faciales detectados: {landmarks.num_parts}")  # Verificar puntos detectados

            # Dibujar puntos faciales para ojos y boca en la imagen original
            for n in range(36, 48):  # Puntos de los ojos (36-47)
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Puntos verdes para los ojos
            for n in range(48, 68):  # Puntos de la boca (48-67)
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)  # Puntos rojos para la boca

        # Guardar la imagen con los puntos faciales marcados
        processed_image_path = os.path.join(OUTPUT_FOLDER, 'result.png')
        cv2.imwrite(processed_image_path, image)

        # Renderizar la plantilla con las imágenes
        return render_template('index.html', original_image=f'/{original_image_path}', processed_image=f'/{processed_image_path}')

if __name__ == '__main__':
    app.run(debug=True)
