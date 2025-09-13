import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import cloudpickle
from TextClassifierPipeline import TextPreprocessorTransform, create_text_classification_pipeline, save_pipeline

def load_and_prepare_data():
    """
    Carga y prepara los datos del proyecto
    """
    # Cargar los datos (ajustar la ruta según sea necesario)
    try:
        df = pd.read_excel('Datos_proyecto.xlsx')
    except FileNotFoundError:
        print("Archivo Datos_proyecto.xlsx no encontrado. Usando datos de ejemplo.")
        # Crear datos de ejemplo si no se encuentra el archivo
        df = pd.DataFrame({
            'texto': [
                "La educación de calidad es esencial para el desarrollo sostenible",
                "Los servicios de salud mental necesitan mayor inversión",
                "La pobreza extrema afecta a millones de personas",
                "Las escuelas rurales carecen de recursos básicos",
                "El acceso a atención médica es un derecho fundamental",
                "Las familias en situación de vulnerabilidad requieren apoyo"
            ],
            'labels': [4, 3, 1, 4, 3, 1]
        })
    
    return df

def main():
    """
    Función principal que entrena y guarda el pipeline
    """
    print("=== ENTRENAMIENTO DEL PIPELINE DE CLASIFICACIÓN DE TEXTO ===\n")
    
    # 1. Cargar datos
    print("1. Cargando datos...")
    df = load_and_prepare_data()
    print(f"   - Datos cargados: {len(df)} registros")
    print(f"   - Distribución de clases:")
    print(df['labels'].value_counts().sort_index())
    
    # 2. Preparar datos para entrenamiento
    print("\n2. Preparando datos para entrenamiento...")
    X = df['texto']  # Ajustar nombre de columna según tus datos
    y = df['labels']
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"   - Datos de entrenamiento: {len(X_train)}")
    print(f"   - Datos de prueba: {len(X_test)}")
    
    # 3. Crear pipeline
    print("\n3. Creando pipeline...")
    pipeline = create_text_classification_pipeline()
    print("   - Pipeline creado exitosamente")
    print("   - Componentes del pipeline:")
    for i, (name, component) in enumerate(pipeline.steps):
        print(f"     {i+1}. {name}: {type(component).__name__}")
    
    # 4. Entrenar pipeline
    print("\n4. Entrenando pipeline...")
    pipeline.fit(X_train, y_train)
    print("   - Entrenamiento completado")
    
    # 5. Evaluar modelo
    print("\n5. Evaluando modelo...")
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"   - F1-Score (macro): {f1:.4f}")
    print("\n   - Reporte de clasificación:")
    print(classification_report(y_test, y_pred, 
                              target_names=[f"ODS {i}" for i in sorted(y.unique())]))
    
    # 6. Entrenar en todo el dataset
    print("\n6. Entrenando modelo final en todo el dataset...")
    final_pipeline = create_text_classification_pipeline()
    final_pipeline.fit(X, y)
    print("   - Modelo final entrenado")
    
    # 7. Guardar pipeline
    print("\n7. Guardando pipeline...")
    
    # Registrar el módulo para cloudpickle
    import TextClassifierPipeline
    cloudpickle.register_pickle_by_value(TextClassifierPipeline)
    
    # Guardar el pipeline
    filename = 'pipeline.cloudpkl'
    with open(filename, 'wb') as file:
        cloudpickle.dump(final_pipeline, file)
    
    print(f"   - Pipeline guardado como: {filename}")
    
    # 8. Probar carga del pipeline
    print("\n8. Probando carga del pipeline...")
    with open(filename, 'rb') as file:
        loaded_pipeline = cloudpickle.load(file)
    
    # Hacer una predicción de prueba
    test_texts = [
        "La educación es fundamental para el desarrollo",
        "Los servicios de salud requieren mejoras",
        "La pobreza afecta a las comunidades vulnerables"
    ]
    
    predictions = loaded_pipeline.predict(test_texts)
    probabilities = loaded_pipeline.predict_proba(test_texts)
    
    print("   - Pipeline cargado exitosamente")
    print("   - Predicciones de prueba:")
    for i, (text, pred, probs) in enumerate(zip(test_texts, predictions, probabilities)):
        print(f"     Texto {i+1}: ODS {pred} (confianza: {max(probs):.3f})")
    
    print("\n=== PROCESO COMPLETADO EXITOSAMENTE ===")
    print(f"Pipeline final guardado en: {filename}")
    
    return final_pipeline, filename

if __name__ == "__main__":
    pipeline, filename = main()
