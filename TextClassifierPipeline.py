import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import cloudpickle

class TextPreprocessorTransform(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para preprocesamiento de texto.
    Basado en las secciones 6 y 7 del notebook proyecto.ipynb
    """
    
    def __init__(self):
        # Descargar recursos de NLTK si es necesario
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Inicializar stemmer y stopwords en español
        self.stemmer = SnowballStemmer('spanish')
        self.stop_words = set(stopwords.words('spanish'))
        
        # Agregar stopwords adicionales comunes en español
        additional_stopwords = {
            'ser', 'estar', 'tener', 'hacer', 'poder', 'decir', 'ir', 'ver', 
            'dar', 'saber', 'querer', 'llegar', 'pasar', 'deber', 'poner', 
            'parecer', 'quedar', 'creer', 'hablar', 'llevar', 'dejar', 'seguir',
            'encontrar', 'llamar', 'venir', 'pensar', 'salir', 'volver', 'tomar',
            'conocer', 'vivir', 'sentir', 'tratar', 'mirar', 'contar', 'empezar',
            'esperar', 'buscar', 'existir', 'entrar', 'trabajar', 'escribir',
            'perder', 'producir', 'ocurrir', 'entender', 'pedir', 'recibir',
            'recordar', 'terminar', 'permitir', 'aparecer', 'conseguir', 'comenzar',
            'servir', 'sacar', 'necesitar', 'mantener', 'resultar', 'leer',
            'caer', 'cambiar', 'presentar', 'crear', 'abrir', 'considerar',
            'oír', 'acabar', 'convertir', 'ganar', 'formar', 'traer', 'partir',
            'morir', 'aceptar', 'realizar', 'suponer', 'comprender', 'lograr'
        }
        self.stop_words.update(additional_stopwords)
    
    def fit(self, X, y=None):
        """Método fit requerido por sklearn (no hace nada en este caso)"""
        return self
    
    def transform(self, X):
        """
        Transforma los textos aplicando preprocesamiento completo
        """
        if isinstance(X, pd.Series):
            texts = X.tolist()
        elif isinstance(X, list):
            texts = X
        else:
            texts = [str(x) for x in X]
        
        processed_texts = []
        
        for text in texts:
            if pd.isna(text) or text == '':
                processed_texts.append('')
                continue
            
            # Aplicar todas las transformaciones de preprocesamiento
            processed_text = self._preprocess_text(str(text))
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def _preprocess_text(self, text):
        """
        Aplica el preprocesamiento completo a un texto individual
        """
        # 1. Convertir a minúsculas
        text = text.lower()
        
        # 2. Eliminar caracteres especiales y números, mantener solo letras y espacios
        text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
        
        # 3. Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        # 4. Tokenizar
        tokens = word_tokenize(text, language='spanish')
        
        # 5. Eliminar stopwords y palabras muy cortas
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # 6. Aplicar stemming
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        # 7. Unir tokens de vuelta en texto
        processed_text = ' '.join(tokens)
        
        return processed_text

def create_text_classification_pipeline():
    """
    Crea el pipeline completo de clasificación de texto basado en el mejor modelo
    del análisis en las secciones 6 y 7 del notebook.
    
    Returns:
        Pipeline: Pipeline de sklearn con preprocesamiento y modelo
    """
    
    # Crear el pipeline con los mejores componentes identificados
    pipeline = Pipeline([
        ('preprocessor', TextPreprocessorTransform()),
        ('vectorizer', CountVectorizer(
            ngram_range=(1, 1),  # Unigramas obtuvieron el mejor rendimiento
            max_features=None,   # Sin límite de features
            lowercase=False,     # Ya se hace en el preprocessor
            token_pattern=r'\b\w+\b'  # Patrón de tokens
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='liblinear',  # Buen solver para problemas multiclase
            multi_class='ovr'    # One-vs-Rest para multiclase
        ))
    ])
    
    return pipeline

def save_pipeline(pipeline, filename='text_classification_pipeline.cloudpkl'):
    """
    Guarda el pipeline usando cloudpickle
    
    Args:
        pipeline: Pipeline entrenado
        filename: Nombre del archivo donde guardar
    """
    # Registrar el módulo para cloudpickle
    cloudpickle.register_pickle_by_value(__name__)
    
    # Guardar el pipeline
    with open(filename, 'wb') as file:
        cloudpickle.dump(pipeline, file)
    
    print(f"Pipeline guardado exitosamente en: {filename}")

def load_pipeline(filename='text_classification_pipeline.cloudpkl'):
    """
    Carga el pipeline desde archivo
    
    Args:
        filename: Nombre del archivo a cargar
        
    Returns:
        Pipeline cargado
    """
    with open(filename, 'rb') as file:
        pipeline = cloudpickle.load(file)
    
    print(f"Pipeline cargado exitosamente desde: {filename}")
    return pipeline

# Ejemplo de uso
if __name__ == "__main__":
    # Crear el pipeline
    pipeline = create_text_classification_pipeline()
    
    # Datos de ejemplo para demostrar el uso
    sample_texts = [
        "La educación es fundamental para el desarrollo de los países",
        "Los servicios de salud mental requieren mayor atención",
        "La pobreza afecta a millones de personas en el mundo"
    ]
    
    sample_labels = [4, 3, 1]  # ODS 4 (Educación), ODS 3 (Salud), ODS 1 (Pobreza)
    
    # Entrenar el pipeline (ejemplo)
    pipeline.fit(sample_texts, sample_labels)
    
    # Hacer predicciones
    predictions = pipeline.predict(sample_texts)
    probabilities = pipeline.predict_proba(sample_texts)
    
    print("Predicciones:", predictions)
    print("Probabilidades:", probabilities)
    
    # Guardar el pipeline
    save_pipeline(pipeline, 'example_pipeline.cloudpkl')
    
    # Cargar y probar
    loaded_pipeline = load_pipeline('example_pipeline.cloudpkl')
    test_predictions = loaded_pipeline.predict(sample_texts)
    print("Predicciones del pipeline cargado:", test_predictions)
