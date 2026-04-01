"""
Se evalúan los 9 algoritmos BASE (sin mejora) sobre las 25 tareas del dataset DARWIN
y se devuelve una tabla con Accuracy, Sensibilidad, Especificidad y F1 por tarea/algoritmo.

Uso:
    from evaluar_modelos_darwin import evaluar_todos_los_modelos
    df_resultados = evaluar_todos_los_modelos(df)
    df_resultados.to_csv("resultados_darwin.csv", index=False)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


# ---------------------------------------------------------------------------
# Función: se extrae las columnas de una tarea específica
# ---------------------------------------------------------------------------
def analizar_tarea(df, num_tarea):
    """
    Replica exacta de la función del notebook.
    Devuelve un DataFrame con las columnas de la tarea + 'class'.
    """
    suffix = str(num_tarea)
    cols_tarea = []
    for col in df.columns:
        if col.endswith(suffix):
            posible_prefijo = col[:-(len(suffix))]
            if not posible_prefijo[-1:].isdigit():
                cols_tarea.append(col)

    if not cols_tarea:
        return None

    return df[cols_tarea + ['class']].copy()


# ---------------------------------------------------------------------------
# LVQ  
# ---------------------------------------------------------------------------
def _entrenar_lvq(X, Y, n_prototipos_por_clase=2, learning_rate=0.01, epochs=100):
    clases = np.unique(Y)
    prototipos, labels_p = [], []
    for c in clases:
        X_clase = X[Y == c]
        indices = np.random.choice(len(X_clase), n_prototipos_por_clase, replace=False)
        for idx in indices:
            prototipos.append(X_clase[idx])
            labels_p.append(c)
    prototipos = np.array(prototipos)
    for _ in range(epochs):
        for i in range(len(X)):
            x_i, y_i = X[i], Y[i]
            distancias = np.linalg.norm(prototipos - x_i, axis=1)
            g = np.argmin(distancias)
            if labels_p[g] == y_i:
                prototipos[g] += learning_rate * (x_i - prototipos[g])
            else:
                prototipos[g] -= learning_rate * (x_i - prototipos[g])
        learning_rate *= 0.95
    return prototipos, labels_p


def _predecir_lvq(X, prototipos, labels_p):
    preds = []
    for x_i in X:
        g = np.argmin(np.linalg.norm(prototipos - x_i, axis=1))
        preds.append(labels_p[g])
    return np.array(preds)


# ---------------------------------------------------------------------------
# Diccionario de modelos (version estándar)
# ---------------------------------------------------------------------------
def _get_modelos_base():
    return {
        "LR":  LogisticRegression(max_iter=1000),
        "XGB": xgb.XGBClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=3,
                    use_label_encoder=False, eval_metric="logloss",
                    verbosity=0
               ),
        "DT":  DecisionTreeClassifier(random_state=42),
        "RF":  RandomForestClassifier(n_estimators=100, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "GNB": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "MLP": MLPClassifier(
                    hidden_layer_sizes=(100, 50), activation="relu",
                    solver="adam", max_iter=1000, random_state=42,
                    learning_rate_init=0.001
               ),
        
        "LVQ": None,
    }


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def evaluar_todos_los_modelos(df, test_size=0.2, random_state=42, verbose=True):
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
    registros = []

    for num_tarea in range(1, 26):

        # --- Preparación de datos ---
        df_tarea = analizar_tarea(df, num_tarea)
        if df_tarea is None:
            if verbose:
                print(f"[!] Tarea {num_tarea}: no se encontraron columnas, se omite.")
            continue

        X = df_tarea.drop("class", axis=1)
        Y = df_tarea["class"]

        le = LabelEncoder()
        Y = le.fit_transform(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state, stratify=Y
        )

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        if verbose:
            print(f"\n{'─'*55}")
            print(f"  Tarea {num_tarea:2d}  |  {X.shape[1]} variables  "
                  f"|  train={len(X_train)}  test={len(X_test)}")
            print(f"{'─'*55}")

        modelos = _get_modelos_base()

        for nombre, modelo in modelos.items():

            try:
                # --- Entrenamiento y predicción ---
                if nombre == "LVQ":
                    prot, labs = _entrenar_lvq(X_train_sc, Y_train)
                    Y_pred = _predecir_lvq(X_test_sc, prot, labs)
                else:
                    modelo.fit(X_train_sc, Y_train)
                    Y_pred = modelo.predict(X_test_sc)

                # --- Métricas ---
                acc = accuracy_score(Y_test, Y_pred)
                f1  = f1_score(Y_test, Y_pred, zero_division=0)

                cm = confusion_matrix(Y_test, Y_pred)
                tn, fp, fn, tp = cm.ravel()

                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                registros.append({
                    "Tarea":         num_tarea,
                    "Algoritmo":     nombre,
                    "Accuracy":      round(acc,  4),
                    "Sensibilidad":  round(sens, 4),
                    "Especificidad": round(spec, 4),
                    "F1":            round(f1,   4),
                })

                if verbose:
                    print(f"  {nombre:<5}  Acc={acc:.3f}  "
                          f"Sens={sens:.3f}  Spec={spec:.3f}  F1={f1:.3f}")

            except Exception as e:
                if verbose:
                    print(f"  {nombre:<5}  ERROR: {e}")
                registros.append({
                    "Tarea": num_tarea, "Algoritmo": nombre,
                    "Accuracy": None, "Sensibilidad": None,
                    "Especificidad": None, "F1": None,
                })

    df_resultados = pd.DataFrame(registros)
    return df_resultados


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tabla_pivot(df_resultados, metrica="Accuracy"):
    """
    Devuelve una tabla pivotada: filas=Tarea, columnas=Algoritmo.
    metrica: 'Accuracy' | 'Sensibilidad' | 'Especificidad' | 'F1'
    """
    return df_resultados.pivot(index="Tarea", columns="Algoritmo", values=metrica).round(4)


def mejor_algoritmo_por_tarea(df_resultados, metrica="Accuracy"):
    """
    Devuelve, por cada tarea, el algoritmo con la mejor métrica indicada.
    """
    idx = df_resultados.groupby("Tarea")[metrica].idxmax()
    return df_resultados.loc[idx, ["Tarea", "Algoritmo", metrica]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ucimlrepo import fetch_ucirepo

    print("Cargando dataset DARWIN...")
    darwin = fetch_ucirepo(id=732)
    df = pd.concat([darwin.data.features, darwin.data.targets], axis=1)

    print("Evaluando modelos (esto puede tardar varios minutos)...\n")
    resultados = evaluar_todos_los_modelos(df, verbose=True)

    # Guardar CSV
    resultados.to_csv("resultados_darwin_base.csv", index=False)
    print("\n✔ Resultados guardados en 'resultados_darwin_base.csv'")

    # Vista rápida
    print("\n=== TABLA ACCURACY (pivot) ===")
    print(tabla_pivot(resultados, "Accuracy").to_string())

    print("\n=== MEJOR ALGORITMO POR TAREA (Accuracy) ===")
    print(mejor_algoritmo_por_tarea(resultados).to_string(index=False))