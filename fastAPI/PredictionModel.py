from sklearn.pipeline import Pipeline
import cloudpickle

class Model:

    def __init__(self,columns):
        with open('assets/pipeline.cloudpkl', 'rb') as f:
            self.model = cloudpickle.load(f)

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
