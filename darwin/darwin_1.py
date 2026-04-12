"""
Se evalúan los 9 algoritmos BASE (sin mejora) sobre las 25 tareas del dataset DARWIN
y se devuelve una tabla con Accuracy, Sensibilidad, Especificidad y F1 por tarea/algoritmo.

Uso:
    from evaluar_modelos_darwin import evaluar_todos_los_modelos
    df_resultados = evaluar_todos_los_modelos(df)
    df_resultados.to_csv("resultados_darwin.csv", index=False)
"""

import numpy as np # np
import pandas as pd
from sklearn.model_selection import train_test_split # train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder # StandardScaler y LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, classification_report
) # ACC score, CM, F1 y CR
from sklearn.linear_model import LogisticRegression # LR
from sklearn.tree import DecisionTreeClassifier # DT
from sklearn.ensemble import RandomForestClassifier # RF
from sklearn.neighbors import KNeighborsClassifier # kNN
from sklearn.svm import SVC # SVM
from sklearn.naive_bayes import GaussianNB # GNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # LDA
from sklearn.neural_network import MLPClassifier # MLP
import xgboost as xgb # XGB


# ---------------------------------------------------------------------------
# Se extraen las columnas de una tarea específica
# ---------------------------------------------------------------------------
def analizar_tarea(df, num_tarea): # funcion analizar_tarea
    """
    Replica exacta de la función del notebook.
    Devuelve un DataFrame con las columnas de la tarea + 'class'.
    """
    suffix = str(num_tarea) # sufijo a str
    cols_tarea = [] # lista vacia
    for col in df.columns: 
        if col.endswith(suffix):   # sufijo
            posible_prefijo = col[:-(len(suffix))] # se elimina el sufijo
            if not posible_prefijo[-1:].isdigit(): # se revisa el sufijo anterior
                cols_tarea.append(col)

    if not cols_tarea:
        return None # si no se encuentra la tarea se detiene

    return df[cols_tarea + ['class']].copy()  # df final


# ---------------------------------------------------------------------------
# LVQ  
# ---------------------------------------------------------------------------
def _entrenar_lvq(X, Y, n_prototipos_por_clase=2, learning_rate=0.01, epochs=100): # funcionar _entrenar_lvq
    clases = np.unique(Y) # etiquetas
    prototipos, labels_p = [], [] # listas
    for c in clases:
        X_clase = X[Y == c] # clase c
        indices = np.random.choice(len(X_clase), n_prototipos_por_clase, replace=False) # se eligen al azar los puntos
        for idx in indices: # asignacion
            prototipos.append(X_clase[idx])
            labels_p.append(c)
    prototipos = np.array(prototipos) # matriz
    for _ in range(epochs):
        for i in range(len(X)):
            x_i, y_i = X[i], Y[i] # se recorre cada ejemplo
            distancias = np.linalg.norm(prototipos - x_i, axis=1) # cálculo de las distancias
            g = np.argmin(distancias) # se encuentra el índice g
            if labels_p[g] == y_i: # cuando la clase del prototipo coincide con la muestra
                prototipos[g] += learning_rate * (x_i - prototipos[g]) # se acerca el prototipo a la muestra
            else:
                prototipos[g] -= learning_rate * (x_i - prototipos[g]) # se aleja el prototipo de la muestra
        learning_rate *= 0.95 # se reduce la tasa del aprendizaje
    return prototipos, labels_p


def _predecir_lvq(X, prototipos, labels_p):
    preds = [] # lista resultados
    for x_i in X:
        g = np.argmin(np.linalg.norm(prototipos - x_i, axis=1)) # prototipo mas cerca del punto x_i
        preds.append(labels_p[g]) # se asigna la clase del prototipo mas cercano a x_i
    return np.array(preds) # array final


# ---------------------------------------------------------------------------
# Diccionario de modelos (version estándar)
# ---------------------------------------------------------------------------
def _get_modelos_base(): # función _get_modelos_base
    return {
        "LR":  LogisticRegression(max_iter=1000), # LR
        "XGB": xgb.XGBClassifier( # XGB
                    n_estimators=100, learning_rate=0.1, max_depth=3,
                    use_label_encoder=False, eval_metric="logloss",
                    verbosity=0
               ),
        "DT":  DecisionTreeClassifier(random_state=42), # DT
        "RF":  RandomForestClassifier(n_estimators=100, random_state=42), # RF
        "kNN": KNeighborsClassifier(n_neighbors=5), # kNN
        "SVM": SVC(kernel="rbf", probability=True, random_state=42), # SVM
        "GNB": GaussianNB(), # GNB
        "LDA": LinearDiscriminantAnalysis(), # LDA
        "MLP": MLPClassifier( # MLP
                    hidden_layer_sizes=(100, 50), activation="relu",
                    solver="adam", max_iter=1000, random_state=42,
                    learning_rate_init=0.001
               ),
        
        "LVQ": None, # LVQ
    }


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def evaluar_todos_los_modelos(df, test_size=0.2, random_state=42, verbose=True): # evaluar_todos_los_modelos
    """
    Se evaluan los 9 algoritmos BASE sobre las 25 tareas del dataset DARWIN.

    Parámetros
    ----------
    df           : DataFrame completo (features + columna 'class')
    test_size    : proporción para test (default 0.2)
    random_state : semilla (default 42)
    verbose      : progreso tarea a tarea

    Retorna
    -------
    df_resultados : DataFrame con columnas
                    [Tarea, Algoritmo, Accuracy, Sensibilidad, Especificidad, F1]
    """
    registros = [] # resultados

    for num_tarea in range(1, 26):

        # --- Preparación de datos ---
        df_tarea = analizar_tarea(df, num_tarea) # analizar_tarea
        if df_tarea is None: # si la tarea no existe se imprime un aviso
            if verbose:
                print(f"[!] Tarea {num_tarea}: no se encontraron columnas, se omite.")
            continue

        X = df_tarea.drop("class", axis=1) # variables
        Y = df_tarea["class"] # class

        le = LabelEncoder() # codificador de etiqueras
        Y = le.fit_transform(Y) # codificación del vector de etiquetas

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state, stratify=Y
        ) # división de datos

        scaler = StandardScaler() # escalado de datos
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        if verbose: # información sobre las variables y registro de las tareas
            print(f"\n{'─'*55}")
            print(f"  Tarea {num_tarea:2d}  |  {X.shape[1]} variables  "
                  f"|  train={len(X_train)}  test={len(X_test)}")
            print(f"{'─'*55}")

        modelos = _get_modelos_base()

        for nombre, modelo in modelos.items():

            try:
                # --- Entrenamiento y predicción ---
                if nombre == "LVQ":  # modelo
                    prot, labs = _entrenar_lvq(X_train_sc, Y_train) # entrenamiento
                    Y_pred = _predecir_lvq(X_test_sc, prot, labs) # predicción
                else: # cuando el modelo es otro flujo liberias ML
                    modelo.fit(X_train_sc, Y_train)
                    Y_pred = modelo.predict(X_test_sc)

                # --- Métricas ---
                acc = accuracy_score(Y_test, Y_pred) # acc
                f1  = f1_score(Y_test, Y_pred, zero_division=0) # f1

                cm = confusion_matrix(Y_test, Y_pred) # cm
                tn, fp, fn, tp = cm.ravel() # tn, fp, fn y tp

                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # fórmula sensibilidad
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # fórmula erspecificidad

                registros.append({ # registros
                    "Tarea":         num_tarea, # tarea
                    "Algoritmo":     nombre, # algoritmo
                    "Accuracy":      round(acc,  4), # acc
                    "Sensibilidad":  round(sens, 4), # sensibilidad
                    "Especificidad": round(spec, 4), # especificidad
                    "F1":            round(f1,   4), # F1
                })

                if verbose: # se imprimen las métricas si el modo detallado esta activo
                    print(f"  {nombre:<5}  Acc={acc:.3f}  "
                          f"Sens={sens:.3f}  Spec={spec:.3f}  F1={f1:.3f}")

            except Exception as e: # excepción del try
                if verbose:
                    print(f"  {nombre:<5}  ERROR: {e}") # se imprime el algoritmo que fallo y el mensaje de error
                registros.append({ # se rellena el algoritmo y las métricas con none
                    "Tarea": num_tarea, "Algoritmo": nombre,
                    "Accuracy": None, "Sensibilidad": None,
                    "Especificidad": None, "F1": None,
                })

    df_resultados = pd.DataFrame(registros) # tabla final 
    return df_resultados # salida de la función


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tabla_pivot(df_resultados, metrica="Accuracy"):
    """
    Se devuelve una tabla pivotada: filas=Tarea, columnas=Algoritmo.
    metrica: 'Accuracy' | 'Sensibilidad' | 'Especificidad' | 'F1'
    """
    return df_resultados.pivot(index="Tarea", columns="Algoritmo", values=metrica).round(4)


def mejor_algoritmo_por_tarea(df_resultados, metrica="Accuracy"):
    """
    Se devuelve, por cada tarea, el algoritmo con la mejor métrica indicada.
    """
    idx = df_resultados.groupby("Tarea")[metrica].idxmax()
    return df_resultados.loc[idx, ["Tarea", "Algoritmo", metrica]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------
if __name__ == "__main__": # si el archivo se lanza directamente
    from ucimlrepo import fetch_ucirepo # importación

    print("Cargando dataset DARWIN...")
    darwin = fetch_ucirepo(id=732) # descarga de datos
    df = pd.concat([darwin.data.features, darwin.data.targets], axis=1) # se une X e Y

    print("Evaluando modelos (esto puede tardar varios minutos)...\n")
    resultados = evaluar_todos_los_modelos(df, verbose=True) # se llama a la función principal

    # Se guarda el CSV
    resultados.to_csv("resultados_darwin_base.csv", index=False)
    print("\n✔ Resultados guardados en 'resultados_darwin_base.csv'")

    # Vista rápida
    print("\n=== TABLA ACCURACY (pivot) ===")
    print(tabla_pivot(resultados, "Accuracy").to_string())

    print("\n=== MEJOR ALGORITMO POR TAREA (Accuracy) ===")
    print(mejor_algoritmo_por_tarea(resultados).to_string(index=False))