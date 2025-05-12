"""
Análisis de Factores de Deserción y Éxito Académico en Estudiantes Universitarios
================================================================================
Dataset: students_dropout_academic_success.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    print(f"Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas")
    
    column_names = {
        'Marital status': 'Estado_Civil',
        'Application mode': 'Modo_Aplicacion',
        'Application order': 'Orden_Aplicacion',
        'Course': 'Curso',
        'Daytime/evening attendance': 'Asistencia_Dia_Noche',
        'Previous qualification': 'Calificacion_Previa',
        'Previous qualification (grade)': 'Nota_Calificacion_Previa',
        'Nacionality': 'Nacionalidad',
        "Mother's qualification": 'Calificacion_Madre',
        "Father's qualification": 'Calificacion_Padre',
        "Mother's occupation": 'Ocupacion_Madre',
        "Father's occupation": 'Ocupacion_Padre',
        'Admission grade': 'Nota_Admision',
        'Displaced': 'Desplazado',
        'Educational special needs': 'Necesidades_Educativas_Especiales',
        'Debtor': 'Deudor',
        'Tuition fees up to date': 'Matricula_Al_Dia',
        'Gender': 'Genero',
        'Scholarship holder': 'Becario',
        'Age at enrollment': 'Edad_Inscripcion',
        'International': 'Internacional',
        'Curricular units 1st sem (credited)': 'Unidades_Acreditadas_1er_Sem',
        'Curricular units 1st sem (enrolled)': 'Unidades_Inscritas_1er_Sem',
        'Curricular units 1st sem (evaluations)': 'Evaluaciones_1er_Sem',
        'Curricular units 1st sem (approved)': 'Unidades_Aprobadas_1er_Sem',
        'Curricular units 1st sem (grade)': 'Nota_1er_Sem',
        'Curricular units 1st sem (without evaluations)': 'Unidades_Sin_Eval_1er_Sem',
        'Curricular units 2nd sem (credited)': 'Unidades_Acreditadas_2do_Sem',
        'Curricular units 2nd sem (enrolled)': 'Unidades_Inscritas_2do_Sem',
        'Curricular units 2nd sem (evaluations)': 'Evaluaciones_2do_Sem',
        'Curricular units 2nd sem (approved)': 'Unidades_Aprobadas_2do_Sem',
        'Curricular units 2nd sem (grade)': 'Nota_2do_Sem',
        'Curricular units 2nd sem (without evaluations)': 'Unidades_Sin_Eval_2do_Sem',
        'Unemployment rate': 'Tasa_Desempleo',
        'Inflation rate': 'Tasa_Inflacion',
        'GDP': 'PIB',
        'Target': 'Estado_Academico'
    }
    
    existing_columns = {}
    for old_col, new_col in column_names.items():
        if old_col in df.columns:
            existing_columns[old_col] = new_col
    
    df.rename(columns=existing_columns, inplace=True)
    
    if 'Estado_Academico' in df.columns:
        print("\nValores únicos en Estado_Academico:")
        print(df['Estado_Academico'].value_counts())
    else:
        print("\nLa columna 'Target' o 'Estado_Academico' no está presente en el dataset.")
    
    return df

def explore_data(df):
    print("\n=== Información del Dataset ===")
    print(df.info())
    
    print("\n=== Estadísticas Descriptivas (Variables Numéricas) ===")
    print(df.describe().round(2))
    
    missing_values = df.isnull().sum()
    print("\n=== Valores Faltantes ===")
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("No hay valores faltantes en el dataset.")
    
    if 'Estado_Academico' in df.columns:
        print("\n=== Distribución de Estado Académico ===")
        target_counts = df['Estado_Academico'].value_counts()
        print(target_counts)
        print(f"Proporción: {(target_counts / len(df)).round(3)}")
    
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
    if df['Estado_Academico'].dtype == 'object':
        estado_counts = df['Estado_Academico'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        ax = estado_counts.plot(kind='bar', color=colors)
    else:
        ax = sns.countplot(x='Estado_Academico', data=df, palette='viridis')
    
    plt.title('Distribución de Estado Académico', fontweight='bold')
    plt.xlabel('Estado Académico')
    plt.ylabel('Número de Estudiantes')
    plt.xticks(rotation=0)
    
    for i, v in enumerate(df['Estado_Academico'].value_counts()):
        ax.text(i, v + 20, str(v), ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}academic_status_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    if 'Genero' in df.columns:
        ct = pd.crosstab(df['Genero'], df['Estado_Academico'])
        ct_pct = ct.div(ct.sum(axis=1), axis=0)
        
        ax = ct_pct.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('Distribución de Estado Académico por Género (%)', fontweight='bold')
        plt.xlabel('Género (0=Femenino, 1=Masculino)')
        plt.ylabel('Porcentaje')
        plt.xticks(rotation=0)
        plt.legend(title='Estado Académico')
        
        for c in ax.containers:
            labels = [f'{v.get_height():.1%}' if v.get_height() > 0.05 else '' for v in c]
            ax.bar_label(c, labels=labels, label_type='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}gender_academic_status.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    plt.figure(figsize=(12, 7))
    if 'Edad_Inscripcion' in df.columns:
        sns.histplot(data=df, x='Edad_Inscripcion', hue='Estado_Academico', bins=15, multiple='stack', palette='viridis')
        plt.title('Distribución de Estado Académico por Edad', fontweight='bold')
        plt.xlabel('Edad al momento de inscripción')
        plt.ylabel('Número de Estudiantes')
        plt.tight_layout()
        plt.savefig(f"{output_dir}age_academic_status.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    if 'Deudor' in df.columns:
        deudor_ct = pd.crosstab(df['Deudor'], df['Estado_Academico'])
        deudor_ct_pct = deudor_ct.div(deudor_ct.sum(axis=1), axis=0)
        deudor_ct_pct.plot(kind='bar', stacked=True, ax=axes[0, 0], colormap='viridis')
        axes[0, 0].set_title('Impacto de ser Deudor en el Estado Académico', fontweight='bold')
        axes[0, 0].set_xlabel('Deudor (0=No, 1=Sí)')
        axes[0, 0].set_ylabel('Porcentaje')
        axes[0, 0].set_xticklabels(['No', 'Sí'])
        axes[0, 0].legend(title='Estado Académico')
    
    if 'Becario' in df.columns:
        beca_ct = pd.crosstab(df['Becario'], df['Estado_Academico'])
        beca_ct_pct = beca_ct.div(beca_ct.sum(axis=1), axis=0)
        beca_ct_pct.plot(kind='bar', stacked=True, ax=axes[0, 1], colormap='viridis')
        axes[0, 1].set_title('Impacto de ser Becario en el Estado Académico', fontweight='bold')
        axes[0, 1].set_xlabel('Becario (0=No, 1=Sí)')
        axes[0, 1].set_ylabel('Porcentaje')
        axes[0, 1].set_xticklabels(['No', 'Sí'])
        axes[0, 1].legend(title='Estado Académico')
    
    if 'Tasa_Desempleo' in df.columns:
        sns.boxplot(x='Estado_Academico', y='Tasa_Desempleo', data=df, ax=axes[1, 0], palette='viridis')
        axes[1, 0].set_title('Tasa de Desempleo vs Estado Académico', fontweight='bold')
        axes[1, 0].set_xlabel('Estado Académico')
        axes[1, 0].set_ylabel('Tasa de Desempleo')
    
    if 'PIB' in df.columns:
        sns.boxplot(x='Estado_Academico', y='PIB', data=df, ax=axes[1, 1], palette='viridis')
        axes[1, 1].set_title('PIB vs Estado Académico', fontweight='bold')
        axes[1, 1].set_xlabel('Estado Académico')
        axes[1, 1].set_ylabel('PIB')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}economic_factors_academic_status.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    if 'Nota_1er_Sem' in df.columns:
        sns.boxplot(x='Estado_Academico', y='Nota_1er_Sem', data=df, ax=axes[0], palette='viridis')
        axes[0].set_title('Notas del 1er Semestre vs Estado Académico', fontweight='bold')
        axes[0].set_xlabel('Estado Académico')
        axes[0].set_ylabel('Nota Promedio 1er Semestre')
    
    if 'Unidades_Aprobadas_1er_Sem' in df.columns:
        sns.boxplot(x='Estado_Academico', y='Unidades_Aprobadas_1er_Sem', data=df, ax=axes[1], palette='viridis')
        axes[1].set_title('Unidades Aprobadas 1er Semestre vs Estado Académico', fontweight='bold')
        axes[1].set_xlabel('Estado Académico')
        axes[1].set_ylabel('Unidades Aprobadas')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}academic_performance_status.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(14, 12))
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if 'Curso' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['Curso'])
    
    corr_matrix = numeric_df.corr().round(2)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='viridis', 
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlación entre Variables', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizaciones guardadas en el directorio: {output_dir}")
    
    return None

def prepare_data_for_modeling(df):
    if 'Estado_Academico' in df.columns:
        df_model = df.copy()
        
        if df_model['Estado_Academico'].dtype == 'object':
            category_map = {
                'Dropout': 0,
                'Graduate': 1,
                'Enrolled': 2
            }
            if set(df_model['Estado_Academico'].unique()).issubset(set(category_map.keys())):
                df_model['Estado_Academico'] = df_model['Estado_Academico'].map(category_map)
            else:
                df_model['Estado_Academico'] = pd.factorize(df_model['Estado_Academico'])[0]
        
        X = df_model.drop('Estado_Academico', axis=1)
        y = df_model['Estado_Academico']
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ] if len(categorical_features) > 0 else [
                ('num', numeric_transformer, numeric_features)
            ]
        )
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        return X_train, X_test, y_train, y_test, model, df_model['Estado_Academico'].unique()
    else:
        print("Error: No se encontró la variable objetivo 'Estado_Academico'.")
        return None, None, None, None, None, None

def build_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    
    print("Modelo entrenado correctamente.")
    
    return model

def evaluate_model(model, X_test, y_test, classes, output_dir='images/'):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print("\n=== Resultados de la Evaluación ===")
    print(f"Exactitud: {accuracy:.4f}")
    print("\nMatriz de Confusión:")
    print(conf_matrix)
    print("\nInforme de Clasificación:")
    print(class_report)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Matriz de Confusión', fontweight='bold')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig(f"{output_dir}confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    if hasattr(model[-1], 'feature_importances_'):

        feature_names = []
        
        numeric_features = X_test.select_dtypes(include=['int64', 'float64']).columns
        for feature in numeric_features:
            feature_names.append(feature)
        
        categorical_features = X_test.select_dtypes(include=['object']).columns
        if len(categorical_features) > 0:
            for feature in categorical_features:
                unique_values = X_test[feature].unique()
                for value in unique_values:
                    feature_names.append(f"{feature}_{value}")
        
        if len(feature_names) == len(model[-1].feature_importances_):
            importance_df = pd.DataFrame({
                'Característica': feature_names,
                'Importancia': model[-1].feature_importances_
            })
            
            importance_df = importance_df.sort_values('Importancia', ascending=False)
            
            top_features = importance_df.head(15)
            
            plt.figure(figsize=(12, 10))
            ax = sns.barplot(x='Importancia', y='Característica', data=top_features, palette='viridis')
            plt.title('15 Características Más Importantes', fontweight='bold')
            plt.xlabel('Importancia Relativa')
            plt.ylabel('Característica')
            
            for i, v in enumerate(top_features['Importancia']):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("\n=== Características Más Importantes ===")
            print(top_features)
    
    return accuracy, conf_matrix, class_report

def main():
    print("=== Iniciando análisis de deserción y éxito académico ===\n")
    
    df = load_data('data/students_dropout_academic_success.csv')
    
    explore_data(df)
    
    create_visualizations(df)
    
    X_train, X_test, y_train, y_test, model, classes = prepare_data_for_modeling(df)
    
    if model is not None:
        trained_model = build_model(X_train, y_train, model)
        
        accuracy, conf_matrix, class_report = evaluate_model(trained_model, X_test, y_test, classes)
        
        print("\n=== Análisis completado exitosamente ===")
        print(f"Exactitud del modelo: {accuracy:.4f}")
    else:
        print("\n=== No se pudo completar el análisis debido a errores en los datos ===")
    
    return None

if __name__ == "__main__":
    main()