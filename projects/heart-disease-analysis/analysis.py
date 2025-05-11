import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, GridSearchCV
import os


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data(file_path):

    df = pd.read_csv(file_path)
    
    print(f"Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas")
    
    column_names = {
        'age': 'Edad',
        'sex': 'Sexo',
        'cp': 'Tipo de Dolor Torácico',
        'trestbps': 'Presión Arterial en Reposo',
        'chol': 'Colesterol',
        'fbs': 'Azúcar en Sangre en Ayunas > 120 mg/dl',
        'restecg': 'Resultados Electrocardiográficos',
        'thalach': 'Frecuencia Cardíaca Máxima',
        'exang': 'Angina Inducida por Ejercicio',
        'oldpeak': 'Depresión ST',
        'slope': 'Pendiente del Segmento ST',
        'ca': 'Número de Vasos Principales',
        'thal': 'Talasemia',
        'target': 'Enfermedad Cardíaca'
    }
    
    df.rename(columns=column_names, inplace=True)
    
    sex_mapping = {0: 'Mujer', 1: 'Hombre'}
    cp_mapping = {0: 'Angina Típica', 1: 'Angina Atípica', 2: 'Dolor No Anginoso', 3: 'Asintomático'}
    fbs_mapping = {0: 'No', 1: 'Sí'}
    restecg_mapping = {0: 'Normal', 1: 'Anormalidad ST-T', 2: 'Hipertrofia'}
    exang_mapping = {0: 'No', 1: 'Sí'}
    slope_mapping = {0: 'Ascendente', 1: 'Plana', 2: 'Descendente'}
    thal_mapping = {1: 'Normal', 2: 'Defecto Fijo', 3: 'Defecto Reversible'}
    target_mapping = {0: 'No', 1: 'Sí'}
    
    df_visual = df.copy()
    df_visual['Sexo'] = df_visual['Sexo'].map(sex_mapping)
    df_visual['Tipo de Dolor Torácico'] = df_visual['Tipo de Dolor Torácico'].map(cp_mapping)
    df_visual['Azúcar en Sangre en Ayunas > 120 mg/dl'] = df_visual['Azúcar en Sangre en Ayunas > 120 mg/dl'].map(fbs_mapping)
    df_visual['Resultados Electrocardiográficos'] = df_visual['Resultados Electrocardiográficos'].map(restecg_mapping)
    df_visual['Angina Inducida por Ejercicio'] = df_visual['Angina Inducida por Ejercicio'].map(exang_mapping)
    df_visual['Pendiente del Segmento ST'] = df_visual['Pendiente del Segmento ST'].map(slope_mapping)
    df_visual['Talasemia'] = df_visual['Talasemia'].map(thal_mapping)
    df_visual['Enfermedad Cardíaca'] = df_visual['Enfermedad Cardíaca'].map(target_mapping)
    
    return df, df_visual

def explore_data(df):
    print("\n=== Información del Dataset ===")
    print(df.info())
    
    print("\n=== Estadísticas Descriptivas ===")
    print(df.describe().round(2))
    
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\n=== Valores Faltantes ===")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo hay valores faltantes en el dataset.")
    
    print("\n=== Distribución de la Variable Objetivo ===")
    target_counts = df['Enfermedad Cardíaca'].value_counts()
    print(target_counts)
    print(f"Proporción: {target_counts / len(df)}")
    
    return None

def create_visualizations(df, output_dir='images/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sns.set_style("whitegrid")
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Sexo', hue='Enfermedad Cardíaca', data=df, palette='viridis')
    plt.title('Distribución de Enfermedad Cardíaca por Sexo', fontweight='bold')
    plt.xlabel('Sexo')
    plt.ylabel('Número de Pacientes')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}heart_disease_by_gender.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 7))
    sns.histplot(data=df, x='Edad', hue='Enfermedad Cardíaca', bins=15, multiple='stack', palette='viridis')
    plt.title('Distribución de Enfermedad Cardíaca por Edad', fontweight='bold')
    plt.xlabel('Edad')
    plt.ylabel('Número de Pacientes')
    plt.tight_layout()
    plt.savefig(f"{output_dir}heart_disease_by_age.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(14, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().round(2)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='viridis', 
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlación entre Variables', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sns.boxplot(x='Enfermedad Cardíaca', y='Colesterol', data=df, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Colesterol vs Enfermedad Cardíaca', fontweight='bold')
    axes[0, 0].set_xlabel('Enfermedad Cardíaca')
    axes[0, 0].set_ylabel('Colesterol (mg/dl)')
    
    sns.boxplot(x='Enfermedad Cardíaca', y='Frecuencia Cardíaca Máxima', data=df, ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_title('Frecuencia Cardíaca Máxima vs Enfermedad Cardíaca', fontweight='bold')
    axes[0, 1].set_xlabel('Enfermedad Cardíaca')
    axes[0, 1].set_ylabel('Frecuencia Cardíaca Máxima')
    
    sns.boxplot(x='Enfermedad Cardíaca', y='Presión Arterial en Reposo', data=df, ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title('Presión Arterial vs Enfermedad Cardíaca', fontweight='bold')
    axes[1, 0].set_xlabel('Enfermedad Cardíaca')
    axes[1, 0].set_ylabel('Presión Arterial (mm Hg)')
    
    sns.boxplot(x='Enfermedad Cardíaca', y='Depresión ST', data=df, ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title('Depresión ST vs Enfermedad Cardíaca', fontweight='bold')
    axes[1, 1].set_xlabel('Enfermedad Cardíaca')
    axes[1, 1].set_ylabel('Depresión ST')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}numerical_variables_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 7))
    ax = sns.countplot(x='Tipo de Dolor Torácico', hue='Enfermedad Cardíaca', data=df, palette='viridis')
    plt.title('Tipo de Dolor Torácico vs Enfermedad Cardíaca', fontweight='bold')
    plt.xlabel('Tipo de Dolor Torácico')
    plt.ylabel('Número de Pacientes')
    plt.xticks(rotation=0)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}chest_pain_vs_heart_disease.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizaciones guardadas en el directorio: {output_dir}")
    
    return None

def prepare_data_for_modeling(df):
    df_model = df.copy()
    
    X = df_model.drop('Enfermedad Cardíaca', axis=1)
    y = df_model['Enfermedad Cardíaca']
    
    if y.nunique() > 2:
        y = y.map(lambda x: 1 if x == 'Sí' else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def build_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
    }
    
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor puntuación de validación cruzada: {grid_search.best_score_:.4f}")
    
    return best_model

def evaluate_model(model, X_test, y_test, output_dir='images/'):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print("\n=== Resultados de la Evaluación ===")
    print(f"Exactitud: {accuracy:.4f}")
    print("\nMatriz de Confusión:")
    print(conf_matrix)
    print("\nInforme de Clasificación:")
    print(class_report)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Sí'],
                yticklabels=['No', 'Sí'])
    plt.title('Matriz de Confusión', fontweight='bold')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig(f"{output_dir}confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Área bajo la curva = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC', fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    if hasattr(model, 'feature_importances_'):
        feature_names = ['Edad', 'Sexo', 'Tipo de Dolor Torácico', 'Presión Arterial en Reposo', 
                        'Colesterol', 'Azúcar en Sangre en Ayunas > 120 mg/dl', 
                        'Resultados Electrocardiográficos', 'Frecuencia Cardíaca Máxima',
                        'Angina Inducida por Ejercicio', 'Depresión ST', 
                        'Pendiente del Segmento ST', 'Número de Vasos Principales', 'Talasemia']
        
        importance_df = pd.DataFrame({
            'Característica': feature_names,
            'Importancia': model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('Importancia', ascending=False)
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Importancia', y='Característica', data=importance_df, palette='viridis')
        plt.title('Importancia de Características', fontweight='bold')
        plt.xlabel('Importancia Relativa')
        plt.ylabel('Característica')
        
        for i, v in enumerate(importance_df['Importancia']):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return accuracy, conf_matrix, class_report

def main():
    print("===  Análisis de enfermedades cardíacas ===\n")
    
    df, df_visual = load_data('data/heart.csv')
    
    explore_data(df_visual)
    
    create_visualizations(df_visual)
    
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(df)
    
    model = build_model(X_train, y_train)
    
    accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
    
    print("\n=== Análisis completado  ===")
    print(f"Exactitud del modelo: {accuracy:.4f}")
    
    return None

if __name__ == "__main__":
    main()