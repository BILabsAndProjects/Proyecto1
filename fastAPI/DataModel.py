from pydantic import BaseModel
from typing import List

#Representación para endpoint de predict
class DataModel(BaseModel):
    textos: List[str]
    def columns(self):
        return ["textos"]
#Representación para endpoint de retrain 
class DataModelRetrain(BaseModel):
    textos: List[str]
    labels: List[int]

    def columns(self):
        return ["textos", "labels"]