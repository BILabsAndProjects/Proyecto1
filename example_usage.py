import cloudpickle
import pandas as pd
from TextClassifierPipeline import TextPreprocessorTransform, create_text_classification_pipeline

# Crear datos de ejemplo similares a tu caso de uso
sample_data = [
    "La educación de calidad es fundamental para el desarrollo sostenible de los países",
    "Los servicios de salud mental requieren mayor atención y recursos",
    "La pobreza extrema afecta a millones de personas en todo el mundo",
    "Las escuelas necesitan mejor infraestructura y materiales educativos",
    "El acceso universal a la atención médica es un derecho humano básico"
]

sample_labels = [4, 3, 1, 4, 3]  # ODS 4 (Educación), ODS 3 (Salud), ODS 1 (Pobreza)

# Crear el transformador (similar a tu PlayerAggregatorTransform)
preprocessor = TextPreprocessorTransform()

# Aplicar transformación a los datos
processed_texts = preprocessor.fit_transform(sample_data)

print("Textos originales:")
for i, text in enumerate(sample_data):
    print(f"{i+1}. {text}")

print("\nTextos procesados:")
for i, text in enumerate(processed_texts):
    print(f"{i+1}. {text}")

# Crear el pipeline completo
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('vectorizer_and_classifier', create_text_classification_pipeline().steps[1:])  # Tomar vectorizer y classifier
])

# Simplificar el pipeline para que sea más similar a tu ejemplo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Pipeline simplificado
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('vectorize', CountVectorizer(ngram_range=(1, 1))),
    ('classify', LogisticRegression(max_iter=1000, random_state=42))
])

# Entrenar el pipeline
pipeline.fit(sample_data, sample_labels)

print("\nResultado del entrenamiento:")
predictions = pipeline.predict(sample_data)
print(f"Predicciones: {predictions}")

# Serializar el pipeline para uso futuro (similar a tu ejemplo)
import TextClassifierPipeline
cloudpickle.register_pickle_by_value(TextClassifierPipeline)

# Serialización del pipeline
with open('pipeline.cloudpkl', mode='wb') as file:
    cloudpickle.dump(pipeline, file)

print("\nPipeline guardado exitosamente como 'pipeline.cloudpkl'")

# Cargar y probar el pipeline
with open('pipeline.cloudpkl', mode='rb') as file:
    loaded_pipeline = cloudpickle.load(file)

# Probar con nuevos datos
new_texts = [
    "Los estudiantes necesitan acceso a tecnología educativa",
    "La atención primaria de salud debe ser fortalecida"
]

new_predictions = loaded_pipeline.predict(new_texts)
probabilities = loaded_pipeline.predict_proba(new_texts)

print("\nPrueba con nuevos textos:")
for i, (text, pred, probs) in enumerate(zip(new_texts, new_predictions, probabilities)):
    confidence = max(probs)
    print(f"Texto: {text}")
    print(f"Predicción: ODS {pred} (Confianza: {confidence:.3f})")
    print()
