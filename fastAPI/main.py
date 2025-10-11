from typing import Optional
from DataModel import DataModel
from DataModel import DataModelRetrain
from fastapi import FastAPI
import pandas as pd
import cloudpickle 
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


app = FastAPI()


@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}


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
    X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    model.fit(X_train_text, y_train)
    y_pred = model.predict(X_test_text)
    

    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    #with open("assets/pipeline.cloudpkl", "wb") as f:
    #    cloudpickle.dump(model, f)

    #df_unified.to_csv("assets/datos_historicos.csv", index=False)


    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }



"""
@app.post("/predict")
def make_predictions(dataModel: DataModel):
    df = dataModel.textos
    with open("assets/pipeline_final.cloudpkl", "rb") as f:
        model = cloudpickle.load(f)
    result = model.predict(df)
    result = result.tolist() if hasattr(result, "tolist") else [result]
    result = [str(x) if not isinstance(x, (str, int, float, bool)) else x for x in result]

    return {"prediction": result}

@app.post("/retrain")
def make_predictions(dataModel: DataModelRetrain):
    df = pd.DataFrame()
    df["textos"] = dataModel.textos
    df["labels"] = dataModel.labels
    
    df_og = pd.read_excel(r"assets/Datos_proyecto.xlsx")
    df_unified = pd.concat([df, df_og])

    with open("assets/pipeline_final.cloudpkl", "rb") as f:
        model = cloudpickle.load(f)

    model.fit(df_unified)

    X = df_unified['textos']
    y = df_unified['labels']
    y_pred = model.predict(X)

    precision = precision_score(y, y_pred, average="macro")
    recall = recall_score(y, y_pred, average="macro")
    f1 = f1_score(y, y_pred, average="macro")

    #with open("assets/pipeline.cloudpkl", "wb") as f:
    #    cloudpickle.dump(model, f)

    #df_unified.to_csv("assets/datos_historicos.csv", index=False)


    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

"""