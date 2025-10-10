from typing import Optional
from DataModel import DataModel
from DataModel import DataModelRetrain
from fastapi import FastAPI
import pandas as pd
import cloudpickle 
from sklearn.metrics import precision_score, recall_score, f1_score

app = FastAPI()


@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

"""
@app.post("/predict")
def make_predictions(dataModel: DataModel):

    df = pd.DataFrame(dataModel.textos, columns=dataModel.columns())
    df.columns = dataModel.columns()
    with open("assets/pipeline.cloudpkl", "rb") as f:
        model = cloudpickle.load(f)
    result = model.predict(df)
    result = result.tolist() if hasattr(result, "tolist") else [result]
    result = [str(x) if not isinstance(x, (str, int, float, bool)) else x for x in result]

    return {"prediction": result}
"""
### Preguntas JP
# Se puede usar cloudpickle?
# El pipeline incluye pasos de duplicados, vacios y así o se ignoran?
# el pipeline es sobre todo el df o solo sobre los splits train/test?
# para el segundo endpoint, las métricas se calculan solo de train o se debe generar un split de test también?


@app.post("/predict")
def make_predictions(dataModel: DataModel):
    # dataModel.textos ya es una lista
    textos = dataModel.textos

    with open("assets/pipeline.cloudpkl", "rb") as f:
        model = cloudpickle.load(f)

    result = model.predict(textos)

    # Convertir a formato JSON-safe
    result = result.tolist() if hasattr(result, "tolist") else [result]
    result = [str(x) if not isinstance(x, (str, int, float, bool)) else x for x in result]

    #print("➡️ Predicciones:", result)  # Debug

    return {"prediction": result}

@app.post("/retrain")
def make_predictions(dataModel: DataModelRetrain):
    df = pd.DataFrame()
    df["textos"] = dataModel.textos
    df["labels"] = dataModel.labels
    
    df_og = pd.read_excel(r"assets/Datos_proyecto.xlsx")
    df_unified = pd.concat([df, df_og])

    with open("assets/pipeline.cloudpkl", "rb") as f:
        model = cloudpickle.load(f)

    X = df_unified['textos']
    y = df_unified['labels']

    model.fit(X, y)
    y_pred = model.predict(X)

    precision = precision_score(y, y_pred, average="weighted")
    recall = recall_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")

    #with open("assets/pipeline.cloudpkl", "wb") as f:
    #    cloudpickle.dump(model, f)

    #df_unified.to_csv("assets/datos_historicos.csv", index=False)


    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }