from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Crear la aplicación FastAPI
app = FastAPI()

# Clase para modelar los datos de entrada usando Pydantic
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Entrenar el modelo de perceptrón
def train_model():
    # Generar un conjunto de datos de clasificación simple
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo Perceptrón
    model = Perceptron()
    model.fit(X_train, y_train)
    return model

# Entrenar el modelo
model = train_model()

# Ruta para predecir usando el modelo
@app.post("/predict")
def predict(data: InputData):
    # Convertir los datos de la entrada a un arreglo NumPy
    input_data = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])

    # Realizar la predicción
    prediction = model.predict(input_data)

    # Devolver la predicción como JSON
    return {"prediction": int(prediction[0])}
