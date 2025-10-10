from pydantic import BaseModel
from typing import List

class DataModel(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    textos: List[str]


#Esta función retorna los nombres de las columnas correspondientes con el modelo exportado en joblib.
    def columns(self):
        return ["textos"]

class DataModelRetrain(BaseModel):
    textos: List[str]
    labels: List[int]

    def columns(self):
        return ["textos", "labels"]