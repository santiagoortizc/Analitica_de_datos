import os
import io
import joblib
import numpy as np
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures


def load_model_from_path(path: str):
    """Load a joblib model from disk path."""
    return joblib.load(path)


def load_model_from_bytes(uploaded_file):
    """Load a joblib model from an uploaded file (BytesIO)."""
    if uploaded_file is None:
        return None
    try:
        # joblib can load from a file-like object
        return joblib.load(uploaded_file)
    except Exception as e:
        st.error(f"No se pudo cargar el modelo desde el archivo subido: {e}")
        return None


def predict_birth_rate(model, year: int, degree: int = 2):
    poly = PolynomialFeatures(degree=degree)
    # PolynomialFeatures has no learned params, fit_transform on the single sample is fine
    X_poly = poly.fit_transform(np.array([[year]]))
    pred = model.predict(X_poly)[0]
    return float(pred)


def main():
    st.set_page_config(
        page_title="Population model — Predict Birth Rate", layout="centered")
    st.title("Consumir modelo: Predicción de tasa de natalidad")

    st.markdown(
        "Este pequeño demo carga `modelo/model.joblib` (guardado desde el notebook) y predice la tasa de natalidad para un año dado usando un polinomio de grado 2.")

    col1, col2 = st.columns([2, 1])

    with col1:
        year = st.slider("Año a predecir", min_value=2022,
                         max_value=2030, value=2025, step=1)
        use_uploaded = st.checkbox(
            "Subir archivo .joblib en lugar del modelo en disco", value=False)
        uploaded = None
        if use_uploaded:
            uploaded = st.file_uploader("Sube el archivo model.joblib", type=[
                                        "joblib", "pkl"], accept_multiple_files=False)

    with col2:
        st.write("\n")
        st.write("Ruta por defecto:")
        default_path = os.path.join(os.path.dirname(
            __file__), "modelo", "model.joblib")
        st.code(default_path)

    model = None
    if use_uploaded and uploaded is not None:
        model = load_model_from_bytes(uploaded)
    else:
        if os.path.exists(default_path):
            try:
                model = load_model_from_path(default_path)
            except Exception as e:
                st.error(f"Error cargando modelo desde {default_path}: {e}")
        else:
            st.warning(
                f"No se encontró el archivo de modelo en la ruta por defecto: {default_path}.\nPuedes subirlo con la casilla 'Subir archivo'.")

    if model is None:
        st.info(
            "Aún no hay un modelo cargado. Carga `model.joblib` o sube uno para predecir.")
        return

    if st.button("Predecir tasa de natalidad"):
        try:
            pred = predict_birth_rate(model, year, degree=2)
            # Aplicar las mismas restricciones del notebook si lo desea
            pred_clipped = float(np.clip(pred, 0, 100))

            st.success(
                f"Tasa de natalidad estimada (por 1000 hab.) para {year}: {pred_clipped:.3f}")

            st.markdown("---")
            st.write("Detalles:")
            st.write(f"Predicción bruta: {pred:.6f}")
            st.write("Nota: el notebook original aplicaba un recorte entre 10 y 20 y restricciones de tendencia. Aquí mostramos la predicción del modelo cargado.")

        except Exception as e:
            st.error(f"Error al predecir: {e}")


if __name__ == "__main__":
    main()
