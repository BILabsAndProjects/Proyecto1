from typing import Optional
from DataModel import DataModel
from fastapi import FastAPI
import pandas as pd
import cloudpickle 

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