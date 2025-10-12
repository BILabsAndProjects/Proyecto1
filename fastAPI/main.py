from typing import Optional
from DataModel import DataModel
from DataModel import DataModelRetrain
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import cloudpickle 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import shutil
import json


app = FastAPI()

# Configurar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/model-info")
def get_model_info():
    """Retorna información sobre el modelo actual incluyendo el timestamp del último entrenamiento"""
    info_file = "assets/model_info.json"
    
    # Si existe el archivo de info, leerlo
    if os.path.exists(info_file):
        try:
            with open(info_file, "r") as f:
                info = json.load(f)
                return info
        except Exception as e:
            print(f"Error al leer model_info.json: {e}")
    
    # Si no existe o hay error, retornar info por defecto
    return {
        "model_timestamp": None,
        "model_name": "pipeline.cloudpkl (original)"
    }


@app.post("/predict")
def make_predictions(dataModel: DataModel):
    # dataModel.textos ya es una lista
    textos = dataModel.textos

    model_path = "assets/latest_model.cloudpkl"
    if not os.path.exists(model_path):
        # Fallback al modelo original si latest no existe
        model_path = "assets/pipeline.cloudpkl"
    
    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)

    probabilities = model.predict_proba(textos)
    classes = model.classes_

    # usar argmax ya que es mas facil que obtener predict
    pred_indices = probabilities.argmax(axis=1)

    result = [
        {
            "prediction": int(classes[pred_idx]),
            "probabilities": {int(cls): float(prob) for cls, prob in zip(classes, probs)}
        }
        for pred_idx, probs in zip(pred_indices, probabilities)
    ]
    return result


@app.post("/retrain")
def retrain(dataModel: DataModelRetrain):
    archived_models_dir = "assets/archived_models"
    os.makedirs(archived_models_dir, exist_ok=True)
    
    # hacer backup del pipeline con el modelo actual actual antes de reentrenar
    latest_model_path = "assets/latest_model.cloudpkl"
    original_model_path = "assets/pipeline.cloudpkl"
    
    if os.path.exists(latest_model_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{archived_models_dir}/model_{timestamp}.cloudpkl"
        shutil.copy2(latest_model_path, backup_path)
        print(f"Modelo anterior guardado en: {backup_path}")
    elif os.path.exists(original_model_path):
        shutil.copy2(original_model_path, latest_model_path)
    
    df = pd.DataFrame()
    df["textos"] = dataModel.textos
    df["labels"] = dataModel.labels
    
    df_og = pd.read_excel(r"assets/Datos_proyecto.xlsx")
    df_unified = pd.concat([df, df_og])

    model_to_load = latest_model_path if os.path.exists(latest_model_path) else original_model_path
    with open(model_to_load, "rb") as f:
        model = cloudpickle.load(f)

    X = df_unified['textos']
    y = df_unified['labels']
    X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    classifier = model.named_steps["classifier"]
    classifier.set_params(class_weight='balanced')
    
    model.fit(X_train_text, y_train)
    y_pred = model.predict(X_test_text)
    
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    
    # metricas por clase para tirar mas info
    precision_per_class = precision_score(y_test, y_pred, average=None, labels=[1, 3, 4])
    recall_per_class = recall_score(y_test, y_pred, average=None, labels=[1, 3, 4])

    conf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 3, 4])
    
    # guardar el nuevo modelo como latest_model
    with open("assets/latest_model.cloudpkl", "wb") as f:
        cloudpickle.dump(model, f)
    
    # actualizar el archivo de datos históricos con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_unified.to_excel("assets/Datos_proyecto.xlsx", index=False)
    df_unified.to_csv(f"{archived_models_dir}/datos_historicos_{timestamp}.csv", index=False)
    
    model_info = {
        "model_timestamp": timestamp,
        "model_name": "latest_model.cloudpkl",
        "last_retrain_date": datetime.now().isoformat()
    }
    with open("assets/model_info.json", "w") as f:
        json.dump(model_info, f)
    
    print(f"Nuevo modelo guardado como latest_model.cloudpkl")
    print(f"Datos históricos guardados con timestamp: {timestamp}")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classes": [1, 3, 4],
        "precision_per_class": [float(x) for x in precision_per_class],
        "recall_per_class": [float(x) for x in recall_per_class],
        "confusion_matrix": conf_matrix.tolist(),
        "model_timestamp": timestamp,
        "model_saved_as": "latest_model.cloudpkl"
    }