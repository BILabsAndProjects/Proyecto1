import nltk
import os

print("Descargando datos de NLTK...")

try:
    nltk.download('stopwords', quiet=False)
    nltk.download('punkt', quiet=False)
    nltk.download('wordnet', quiet=False)
    nltk.download('omw-1.4', quiet=False)
    nltk.download('punkt_tab', quiet=False)
    
    from nltk.corpus import stopwords
    print(f"Stopwords en español: {len(stopwords.words('spanish'))} palabras")
    
    print("Datos de NLTK descargados y verificados exitosamente")
except Exception as e:
    print(f"Error al descargar datos de NLTK: {e}")
    import traceback
    traceback.print_exc()
