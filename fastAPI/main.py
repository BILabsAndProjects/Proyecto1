from typing import Optional
from DataModel import DataModel
from DataModel import DataModelRetrain
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import cloudpickle 
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


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


@app.post("/predict")
def make_predictions(dataModel: DataModel):
    # dataModel.textos ya es una lista
    textos = dataModel.textos

    with open("assets/pipeline.cloudpkl", "rb") as f:
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
    df = pd.DataFrame()
    df["textos"] = dataModel.textos
    df["labels"] = dataModel.labels
    
    df_og = pd.read_excel(r"assets/Datos_proyecto.xlsx")
    df_unified = pd.concat([df, df_og])

    with open("assets/pipeline.cloudpkl", "rb") as f:
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
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)

    
    with open("assets/pipeline.cloudpkl", "wb") as f:
       cloudpickle.dump(model, f)

    df_unified.to_csv("assets/datos_historicos.csv", index=False)

    # {
    #     "precision": 0.98,           
    #     "recall": 0.98,            
    #     "f1_score": 0.97,
    #     "precision_per_class": [0.90, 0.80, 0.85],
    #     "recall_per_class": [0.88, 0.78, 0.83]
    # }
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "precision_per_class": [float(x) for x in precision_per_class],
        "recall_per_class": [float(x) for x in recall_per_class]
    }
