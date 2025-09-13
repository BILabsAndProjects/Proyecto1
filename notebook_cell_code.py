# =============================================================================
# IMPLEMENTACIÓN DE TODOs - COPIAR ESTE CÓDIGO AL NOTEBOOK
# =============================================================================

# TODO: features con tf-idf
print("=== CREANDO CARACTERÍSTICAS TF-IDF ===")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 1)
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

print(f"Forma de matriz TF-IDF de entrenamiento: {X_train_tfidf.shape}")
print(f"Forma de matriz TF-IDF de prueba: {X_test_tfidf.shape}")

# TODO: features con 1-3-gramas
print("\n=== CREANDO CARACTERÍSTICAS 1-3-GRAMAS ===")
ngram_vectorizer = CountVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 3)
)

X_train_ngram = ngram_vectorizer.fit_transform(X_train_text)
X_test_ngram = ngram_vectorizer.transform(X_test_text)

print(f"Forma de matriz n-gramas de entrenamiento: {X_train_ngram.shape}")
print(f"Forma de matriz n-gramas de prueba: {X_test_ngram.shape}")

# TODO: para cada conjunto de datos (tf-idf y 1-3-gramas):
results_tfidf = {}
results_ngram = {}

# =============================================================================
# MODELOS CON TF-IDF
# =============================================================================
print("\n=== ENTRENANDO MODELOS CON TF-IDF ===")

# TODO: obtener preds del modelo naive bayes
print("\nEntrenando Naive Bayes con TF-IDF...")
nb_tfidf = MultinomialNB(alpha=1.0)
nb_tfidf.fit(X_train_tfidf, y_train)
nb_preds_tfidf = nb_tfidf.predict(X_test_tfidf)

results_tfidf['Naive Bayes'] = {
    'predictions': nb_preds_tfidf,
    'f1_score': f1_score(y_test, nb_preds_tfidf, average='macro'),
    'precision': precision_score(y_test, nb_preds_tfidf, average='macro'),
    'recall': recall_score(y_test, nb_preds_tfidf, average='macro')
}

print(f"Naive Bayes F1-score: {results_tfidf['Naive Bayes']['f1_score']:.4f}")

# TODO obtener preds del modelo regresion logistica
print("\nEntrenando Regresión Logística con TF-IDF...")
lr_tfidf = LogisticRegression(max_iter=1000, random_state=42)
lr_tfidf.fit(X_train_tfidf, y_train)
lr_preds_tfidf = lr_tfidf.predict(X_test_tfidf)

results_tfidf['Logistic Regression'] = {
    'predictions': lr_preds_tfidf,
    'f1_score': f1_score(y_test, lr_preds_tfidf, average='macro'),
    'precision': precision_score(y_test, lr_preds_tfidf, average='macro'),
    'recall': recall_score(y_test, lr_preds_tfidf, average='macro')
}

print(f"Regresión Logística F1-score: {results_tfidf['Logistic Regression']['f1_score']:.4f}")

# TODO: obtener preds de random forest
print("\nEntrenando Random Forest con TF-IDF...")
rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_tfidf.fit(X_train_tfidf, y_train)
rf_preds_tfidf = rf_tfidf.predict(X_test_tfidf)

results_tfidf['Random Forest'] = {
    'predictions': rf_preds_tfidf,
    'f1_score': f1_score(y_test, rf_preds_tfidf, average='macro'),
    'precision': precision_score(y_test, rf_preds_tfidf, average='macro'),
    'recall': recall_score(y_test, rf_preds_tfidf, average='macro')
}

print(f"Random Forest F1-score: {results_tfidf['Random Forest']['f1_score']:.4f}")

# =============================================================================
# MODELOS CON 1-3-GRAMAS
# =============================================================================
print("\n=== ENTRENANDO MODELOS CON 1-3-GRAMAS ===")

# TODO: obtener preds del modelo naive bayes
print("\nEntrenando Naive Bayes con 1-3-gramas...")
nb_ngram = MultinomialNB(alpha=1.0)
nb_ngram.fit(X_train_ngram, y_train)
nb_preds_ngram = nb_ngram.predict(X_test_ngram)

results_ngram['Naive Bayes'] = {
    'predictions': nb_preds_ngram,
    'f1_score': f1_score(y_test, nb_preds_ngram, average='macro'),
    'precision': precision_score(y_test, nb_preds_ngram, average='macro'),
    'recall': recall_score(y_test, nb_preds_ngram, average='macro')
}

print(f"Naive Bayes F1-score: {results_ngram['Naive Bayes']['f1_score']:.4f}")

# TODO obtener preds del modelo regresion logistica
print("\nEntrenando Regresión Logística con 1-3-gramas...")
lr_ngram = LogisticRegression(max_iter=1000, random_state=42)
lr_ngram.fit(X_train_ngram, y_train)
lr_preds_ngram = lr_ngram.predict(X_test_ngram)

results_ngram['Logistic Regression'] = {
    'predictions': lr_preds_ngram,
    'f1_score': f1_score(y_test, lr_preds_ngram, average='macro'),
    'precision': precision_score(y_test, lr_preds_ngram, average='macro'),
    'recall': recall_score(y_test, lr_preds_ngram, average='macro')
}

print(f"Regresión Logística F1-score: {results_ngram['Logistic Regression']['f1_score']:.4f}")

# TODO: obtener preds de random forest
print("\nEntrenando Random Forest con 1-3-gramas...")
rf_ngram = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_ngram.fit(X_train_ngram, y_train)
rf_preds_ngram = rf_ngram.predict(X_test_ngram)

results_ngram['Random Forest'] = {
    'predictions': rf_preds_ngram,
    'f1_score': f1_score(y_test, rf_preds_ngram, average='macro'),
    'precision': precision_score(y_test, rf_preds_ngram, average='macro'),
    'recall': recall_score(y_test, rf_preds_ngram, average='macro')
}

print(f"Random Forest F1-score: {results_ngram['Random Forest']['f1_score']:.4f}")

# TODO: graficar f1 score
print("\n=== CREANDO GRÁFICOS DE F1-SCORES ===")

# Preparar datos para el gráfico
models = ['Naive Bayes', 'Logistic Regression', 'Random Forest']
tfidf_scores = [results_tfidf[model]['f1_score'] for model in models]
ngram_scores = [results_ngram[model]['f1_score'] for model in models]

# Crear el gráfico principal
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

# Crear tabla de resultados
results_df = pd.DataFrame({
    'Modelo': models + models,
    'Características': ['TF-IDF'] * 3 + ['1-3-gramas'] * 3,
    'F1-Score': tfidf_scores + ngram_scores,
    'Precision': [results_tfidf[model]['precision'] for model in models] + 
                [results_ngram[model]['precision'] for model in models],
    'Recall': [results_tfidf[model]['recall'] for model in models] + 
             [results_ngram[model]['recall'] for model in models]
})

print("\n=== TABLA DE RESULTADOS DETALLADOS ===")
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

# Reportes detallados de clasificación
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

print("\n" + "="*50)
print("TODOS LOS TODOs COMPLETADOS EXITOSAMENTE!")
print("="*50)
