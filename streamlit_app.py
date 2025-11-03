import os
import io
import joblib
import numpy as np
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import matplotlib.pyplot as plt


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


def predict_from_bundle(regressor, poly, last_year, last_pop, start_year=None, end_year=2035):
    if start_year is None:
        start_year = last_year + 1
    years = np.array(list(range(start_year, end_year + 1))).reshape(-1, 1)
    preds = regressor.predict(poly.transform(years))
    df = pd.DataFrame({
        'Year': years.flatten(),
        'Population': preds
    })
    # calcular crecimiento natural aproximado
    cur = last_pop
    ng = []
    for p in df['Population']:
        ng.append((p - cur) / cur * 100)
        cur = p
    df['Natural_Growth'] = ng
    return df


def load_model_with_fallback(default_dir: str = 'modelo'):
    """Intenta cargar la instancia completa primero; si falla, carga el bundle y devuelve (pop_model, bundle_dict).
    Devuelve (pop_model_or_None, bundle_or_None, error_message_or_None)
    """
    inst_path = os.path.join(default_dir, 'population_model.joblib')
    bundle_path = os.path.join(default_dir, 'population_model_bundle.joblib')

    # 1) intentar cargar instancia completa
    if os.path.exists(inst_path):
        try:
            inst = joblib.load(inst_path)
            return inst, None, None
        except Exception as e:
            # anotar error y tratar de cargar bundle
            inst_error = e
    else:
        inst_error = None

    # 2) intentar cargar bundle
    if os.path.exists(bundle_path):
        try:
            bundle = joblib.load(bundle_path)
            return None, bundle, None
        except Exception as e:
            return None, None, f"Error cargando bundle: {e} (inst_error: {inst_error})"

    # ninguno existe
    return None, None, f"No se encontró ni '{inst_path}' ni '{bundle_path}'. Inst error: {inst_error}"


def main():
    st.set_page_config(page_title="Population projection", layout="centered")
    st.title("📈 Proyección poblacional")

    st.markdown(
        "Carga el modelo de proyección poblacional entrenado con los datos históricos de población, mortalidad y natalidad")

    col1 = st.columns([2, 1])

    with col1:
        start_year = st.number_input(
            "Año inicio de proyección (opcional)", min_value=2022, value=2022)
        end_year = st.number_input(
            "Año fin de proyección", min_value=2022, value=2035)

    # Cargar modelo: solo desde disco (instancia o bundle)
    pop_model, bundle, load_error = load_model_with_fallback(
        os.path.join(os.path.dirname(__file__), 'modelo'))

    if pop_model is None and bundle is None:
        st.warning(f"No se pudo cargar un modelo: {load_error}")
        return

    # Generar proyecciones a partir de la fuente disponible
    try:
        if pop_model is not None:
            df_proj = pop_model.predict(
                start_year=start_year if start_year is not None else None, end_year=int(end_year))
            metrics = getattr(pop_model, 'test_metrics', None)
        else:
            # bundle expected to contain 'regressor','poly','last_year','last_pop'
            reg = bundle.get('regressor')
            poly = bundle.get('poly')
            last_year = int(bundle.get('last_year'))
            last_pop = float(bundle.get('last_pop'))
            df_proj = predict_from_bundle(
                reg, poly, last_year, last_pop, start_year=start_year if start_year is not None else None, end_year=int(end_year))
            metrics = None

    except Exception as e:
        st.error(f"Error generando proyecciones: {e}")
        return

    st.subheader("Proyecciones")
    st.write(df_proj)

    # Mostrar métricas si existen
    if metrics:
        st.subheader("Métricas de validación (conjunto test)")
        st.write(metrics)

    # Gráfica: intentar usar datos históricos si están disponibles
    data_path = os.path.join(os.path.dirname(
        __file__), 'Data', 'total_population.csv')
    try:
        df_pop_raw = pd.read_csv(data_path)
        countries = df_pop_raw['economy'].tolist()
        country = st.selectbox('País para comparar histórico', countries,
                               index=countries.index('COL') if 'COL' in countries else 0)

        years_cols = [c for c in df_pop_raw.columns if c.startswith('YR')]
        hist_vals = df_pop_raw[df_pop_raw['economy'] ==
                               country][years_cols].values.flatten().astype(float)
        hist_years = [int(c.replace('YR', '')) for c in years_cols]
        df_hist = pd.DataFrame({'Year': hist_years, 'Population': hist_vals})

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0.0)  # Fondo de la figura translúcido
        ax.patch.set_alpha(0.0)   # Fondo del área de ploteo translúcido

        # Agregar cuadrícula
        ax.grid(True, linestyle='--', alpha=0.7, color='gray')

        ax.plot(df_hist['Year'], df_hist['Population'],
                marker='o', label=f'Histórico ({country})')
        ax.plot(df_proj['Year'], df_proj['Population'],
                marker='o', linestyle='--', label='Proyección')

        # Configurar etiquetas y colores
        ax.set_xlabel('Año', color='white')
        ax.set_ylabel('Población', color='white')
        ax.tick_params(colors='white')

        # Formato del eje Y sin notación científica
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

        # Configurar leyenda con texto blanco
        legend = ax.legend()
        plt.setp(legend.get_texts(), color='black')

        st.pyplot(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0.0)  # Fondo de la figura translúcido
        ax.patch.set_alpha(0.0)   # Fondo del área de ploteo translúcido

        # Agregar cuadrícula
        ax.grid(True, linestyle='--', alpha=0.7, color='gray')

        ax.plot(df_proj['Year'], df_proj['Population'],
                marker='o', linestyle='--', label='Proyección')

        # Configurar etiquetas y colores
        ax.set_xlabel('Año', color='white')
        ax.set_ylabel('Población', color='white')
        ax.tick_params(colors='white')

        # Formato del eje Y sin notación científica
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

        # Configurar leyenda con texto blanco
        legend = ax.legend()
        plt.setp(legend.get_texts(), color='white')

        st.pyplot(fig)


if __name__ == "__main__":
    main()
