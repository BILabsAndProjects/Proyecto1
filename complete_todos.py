# Script para completar los TODOs del proyecto
# Este script implementa las características TF-IDF y 1-3-gramas, 
# entrena modelos y compara sus rendimientos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# Función para cargar y preparar los datos (asumiendo que ya están procesados)
def load_processed_data():
    """
    Esta función debe ser adaptada para cargar los datos ya procesados del notebook.
    Por ahora, incluyo el código que debería estar en el notebook.
    """
    # Aquí deberías cargar normalized_df del notebook
    # normalized_df = pd.read_pickle('processed_data.pkl')  # ejemplo
    pass

# TODO 1: Features con TF-IDF
def create_tfidf_features(X_train_text, X_test_text):
    """
    Crea características TF-IDF a partir del texto tokenizado.
    """
    print("=== CREANDO CARACTERÍSTICAS TF-IDF ===")
    
    # Crear vectorizador TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,  # Limitar a las 5000 características más importantes
        min_df=2,          # Ignorar términos que aparecen en menos de 2 documentos
        max_df=0.95,       # Ignorar términos que aparecen en más del 95% de documentos
        stop_words=None,   # Ya removimos stopwords en el preprocesamiento
        ngram_range=(1, 1) # Solo unigramas para TF-IDF
    )
    
    # Ajustar y transformar datos de entrenamiento
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    
    # Transformar datos de prueba
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    
    print(f"Forma de matriz TF-IDF de entrenamiento: {X_train_tfidf.shape}")
    print(f"Forma de matriz TF-IDF de prueba: {X_test_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

# TODO 2: Features con 1-3-gramas
def create_ngram_features(X_train_text, X_test_text):
    """
    Crea características con n-gramas (1-3-gramas) a partir del texto tokenizado.
    """
    print("\n=== CREANDO CARACTERÍSTICAS 1-3-GRAMAS ===")
    
    # Crear vectorizador de n-gramas
    ngram_vectorizer = CountVectorizer(
        max_features=5000,   # Limitar a las 5000 características más importantes
        min_df=2,           # Ignorar términos que aparecen en menos de 2 documentos
        max_df=0.95,        # Ignorar términos que aparecen en más del 95% de documentos
        ngram_range=(1, 3), # 1-gramas, 2-gramas y 3-gramas
        stop_words=None     # Ya removimos stopwords en el preprocesamiento
    )
    
    # Ajustar y transformar datos de entrenamiento
    X_train_ngram = ngram_vectorizer.fit_transform(X_train_text)
    
    # Transformar datos de prueba
    X_test_ngram = ngram_vectorizer.transform(X_test_text)
    
    print(f"Forma de matriz n-gramas de entrenamiento: {X_train_ngram.shape}")
    print(f"Forma de matriz n-gramas de prueba: {X_test_ngram.shape}")
    
    return X_train_ngram, X_test_ngram, ngram_vectorizer

# TODO 3-5: Entrenar modelos con características TF-IDF
def train_models_tfidf(X_train_tfidf, X_test_tfidf, y_train, y_test):
    """
    Entrena Naive Bayes, Regresión Logística y Random Forest con características TF-IDF.
    """
    print("\n=== ENTRENANDO MODELOS CON TF-IDF ===")
    
    results_tfidf = {}
    
    # Naive Bayes
    print("\nEntrenando Naive Bayes...")
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_tfidf, y_train)
    nb_preds = nb_model.predict(X_test_tfidf)
    
    results_tfidf['Naive Bayes'] = {
        'model': nb_model,
        'predictions': nb_preds,
        'f1_score': f1_score(y_test, nb_preds, average='macro'),
        'precision': precision_score(y_test, nb_preds, average='macro'),
        'recall': recall_score(y_test, nb_preds, average='macro')
    }
    
    print(f"Naive Bayes F1-score: {results_tfidf['Naive Bayes']['f1_score']:.4f}")
    
    # Regresión Logística
    print("\nEntrenando Regresión Logística...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    lr_preds = lr_model.predict(X_test_tfidf)
    
    results_tfidf['Logistic Regression'] = {
        'model': lr_model,
        'predictions': lr_preds,
        'f1_score': f1_score(y_test, lr_preds, average='macro'),
        'precision': precision_score(y_test, lr_preds, average='macro'),
        'recall': recall_score(y_test, lr_preds, average='macro')
    }
    
    print(f"Regresión Logística F1-score: {results_tfidf['Logistic Regression']['f1_score']:.4f}")
    
    # Random Forest
    print("\nEntrenando Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_tfidf, y_train)
    rf_preds = rf_model.predict(X_test_tfidf)
    
    results_tfidf['Random Forest'] = {
        'model': rf_model,
        'predictions': rf_preds,
        'f1_score': f1_score(y_test, rf_preds, average='macro'),
        'precision': precision_score(y_test, rf_preds, average='macro'),
        'recall': recall_score(y_test, rf_preds, average='macro')
    }
    
    print(f"Random Forest F1-score: {results_tfidf['Random Forest']['f1_score']:.4f}")
    
    return results_tfidf

# TODO 6-8: Entrenar modelos con características de n-gramas
def train_models_ngram(X_train_ngram, X_test_ngram, y_train, y_test):
    """
    Entrena Naive Bayes, Regresión Logística y Random Forest con características de n-gramas.
    """
    print("\n=== ENTRENANDO MODELOS CON 1-3-GRAMAS ===")
    
    results_ngram = {}
    
    # Naive Bayes
    print("\nEntrenando Naive Bayes...")
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_ngram, y_train)
    nb_preds = nb_model.predict(X_test_ngram)
    
    results_ngram['Naive Bayes'] = {
        'model': nb_model,
        'predictions': nb_preds,
        'f1_score': f1_score(y_test, nb_preds, average='macro'),
        'precision': precision_score(y_test, nb_preds, average='macro'),
        'recall': recall_score(y_test, nb_preds, average='macro')
    }
    
    print(f"Naive Bayes F1-score: {results_ngram['Naive Bayes']['f1_score']:.4f}")
    
    # Regresión Logística
    print("\nEntrenando Regresión Logística...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_ngram, y_train)
    lr_preds = lr_model.predict(X_test_ngram)
    
    results_ngram['Logistic Regression'] = {
        'model': lr_model,
        'predictions': lr_preds,
        'f1_score': f1_score(y_test, lr_preds, average='macro'),
        'precision': precision_score(y_test, lr_preds, average='macro'),
        'recall': recall_score(y_test, lr_preds, average='macro')
    }
    
    print(f"Regresión Logística F1-score: {results_ngram['Logistic Regression']['f1_score']:.4f}")
    
    # Random Forest
    print("\nEntrenando Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_ngram, y_train)
    rf_preds = rf_model.predict(X_test_ngram)
    
    results_ngram['Random Forest'] = {
        'model': rf_model,
        'predictions': rf_preds,
        'f1_score': f1_score(y_test, rf_preds, average='macro'),
        'precision': precision_score(y_test, rf_preds, average='macro'),
        'recall': recall_score(y_test, rf_preds, average='macro')
    }
    
    print(f"Random Forest F1-score: {results_ngram['Random Forest']['f1_score']:.4f}")
    
    return results_ngram

# TODO 9: Graficar F1 scores
def plot_f1_scores(results_tfidf, results_ngram):
    """
    Crea gráficos comparativos de F1-scores para todos los modelos y conjuntos de características.
    """
    print("\n=== CREANDO GRÁFICOS DE F1-SCORES ===")
    
    # Preparar datos para el gráfico
    models = ['Naive Bayes', 'Logistic Regression', 'Random Forest']
    tfidf_scores = [results_tfidf[model]['f1_score'] for model in models]
    ngram_scores = [results_ngram[model]['f1_score'] for model in models]
    
    # Crear el gráfico
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, tfidf_scores, width, label='TF-IDF', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, ngram_scores, width, label='1-3-gramas', alpha=0.8, color='lightcoral')
    
    # Personalizar el gráfico
    ax.set_xlabel('Modelos', fontsize=12)
    ax.set_ylabel('F1-Score (Macro)', fontsize=12)
    ax.set_title('Comparación de F1-Scores por Modelo y Tipo de Características', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    plt.show()
    
    # Crear tabla de resultados detallados
    print("\n=== TABLA DE RESULTADOS DETALLADOS ===")
    
    results_df = pd.DataFrame({
        'Modelo': models + models,
        'Características': ['TF-IDF'] * 3 + ['1-3-gramas'] * 3,
        'F1-Score': tfidf_scores + ngram_scores,
        'Precision': [results_tfidf[model]['precision'] for model in models] + 
                    [results_ngram[model]['precision'] for model in models],
        'Recall': [results_tfidf[model]['recall'] for model in models] + 
                 [results_ngram[model]['recall'] for model in models]
    })
    
    print(results_df.round(4))
    
    # Gráfico de métricas múltiples
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['F1-Score', 'Precision', 'Recall']
    
    for i, metric in enumerate(metrics):
        tfidf_values = [results_tfidf[model][metric.lower().replace('-', '_')] for model in models]
        ngram_values = [results_ngram[model][metric.lower().replace('-', '_')] for model in models]
        
        bars1 = axes[i].bar(x - width/2, tfidf_values, width, label='TF-IDF', alpha=0.8, color='skyblue')
        bars2 = axes[i].bar(x + width/2, ngram_values, width, label='1-3-gramas', alpha=0.8, color='lightcoral')
        
        axes[i].set_xlabel('Modelos')
        axes[i].set_ylabel(metric)
        axes[i].set_title(f'Comparación de {metric}')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models, rotation=45)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar in bars1 + bars2:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def print_detailed_results(results_tfidf, results_ngram, y_test):
    """
    Imprime reportes detallados de clasificación para todos los modelos.
    """
    print("\n" + "="*80)
    print("REPORTES DETALLADOS DE CLASIFICACIÓN")
    print("="*80)
    
    for feature_type, results in [("TF-IDF", results_tfidf), ("1-3-gramas", results_ngram)]:
        print(f"\n{'='*20} RESULTADOS CON {feature_type.upper()} {'='*20}")
        
        for model_name, model_results in results.items():
            print(f"\n--- {model_name} ---")
            print(f"F1-Score: {model_results['f1_score']:.4f}")
            print(f"Precision: {model_results['precision']:.4f}")
            print(f"Recall: {model_results['recall']:.4f}")
            print("\nReporte de clasificación:")
            print(classification_report(y_test, model_results['predictions']))

# Función principal para ejecutar todos los TODOs
def main():
    """
    Función principal que ejecuta todos los TODOs en secuencia.
    
    NOTA: Esta función necesita ser adaptada para usar los datos del notebook.
    Debes cargar X_train_text, X_test_text, y_train, y_test desde tu notebook.
    """
    print("INICIANDO IMPLEMENTACIÓN DE TODOs")
    print("="*50)
    
    # IMPORTANTE: Aquí debes cargar tus datos del notebook
    # Por ejemplo:
    # X_train_text, X_test_text, y_train, y_test = load_data_from_notebook()
    
    print("NOTA: Debes cargar los datos del notebook antes de ejecutar este script.")
    print("Asegúrate de tener las variables: X_train_text, X_test_text, y_train, y_test")
    
    # Ejemplo de cómo usar las funciones:
    """
    # 1. Crear características TF-IDF
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = create_tfidf_features(X_train_text, X_test_text)
    
    # 2. Crear características de n-gramas
    X_train_ngram, X_test_ngram, ngram_vectorizer = create_ngram_features(X_train_text, X_test_text)
    
    # 3. Entrenar modelos con TF-IDF
    results_tfidf = train_models_tfidf(X_train_tfidf, X_test_tfidf, y_train, y_test)
    
    # 4. Entrenar modelos con n-gramas
    results_ngram = train_models_ngram(X_train_ngram, X_test_ngram, y_train, y_test)
    
    # 5. Crear gráficos comparativos
    results_df = plot_f1_scores(results_tfidf, results_ngram)
    
    # 6. Imprimir resultados detallados
    print_detailed_results(results_tfidf, results_ngram, y_test)
    
    print("\n" + "="*50)
    print("TODOS LOS TODOs COMPLETADOS EXITOSAMENTE!")
    print("="*50)
    """

if __name__ == "__main__":
    main()
