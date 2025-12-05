import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Car Prediction", page_icon="🎯", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
DF_TRAIN_PATH = MODEL_DIR / "df_train.pkl"
ONE_HOT_ENCODERS_PATH = MODEL_DIR / "one_hot_encoders.pkl"
MEDIANS_PATH = MODEL_DIR / "medians.pkl"


@st.cache_resource
def load_data():
    """Загружаем pipeline через pickle"""

    with open(PIPELINE_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    with open(DF_TRAIN_PATH, 'rb') as f:
        df_train = pickle.load(f)
    with open(ONE_HOT_ENCODERS_PATH, 'rb') as f:
        one_hot_encoders = pickle.load(f)
    with open(ONE_HOT_ENCODERS_PATH, 'rb') as f:
        one_hot_encoders = pickle.load(f)
    with open(MEDIANS_PATH, 'rb') as f:
        medians = pickle.load(f)
    return pipeline, df_train, feature_names, one_hot_encoders, medians


def first_part_to_float(x):
    first_w = str(x).split()[0]
    try:
        return float(first_w)
    except ValueError:
        return 0.0

def to_int(x):
    try:
        return int(x)
    except ValueError:
        return 0 

def prepare_features(df, feature_names):
    """Приводим данные к формату обучения модели."""
    df_proc = df.copy()
    df_proc = df_proc.drop(columns=['torque', 'name'])

    #Приводим к float
    df_proc['mileage'] = df_proc['mileage'].apply(first_part_to_float)
    df_proc['engine'] = df_proc['engine'].apply(first_part_to_float)
    df_proc['max_power'] = df_proc['max_power'].apply(first_part_to_float)

    #заолняем пропуски медианой
    df_proc["mileage"] = df_proc["mileage"].fillna(MEDIANS["mileage"])
    df_proc["engine"] = df_proc["engine"].fillna(MEDIANS["engine"])
    df_proc["max_power"] = df_proc["max_power"].fillna(MEDIANS["max_power"])
    df_proc["seats"] = df_proc["seats"].fillna(MEDIANS["seats"])

    #приводим к int
    df_proc['seats'] = df_proc['seats'].apply(to_int)
    df_proc['engine'] = df_proc['engine'].apply(to_int)

    #Кадируем категориальные признаки
    categorical_features = ONE_HOT_ENCODERS.keys()
    for feature in categorical_features:
        one_hot_enc = ONE_HOT_ENCODERS[feature]
        df_proc_enc = pd.DataFrame(one_hot_enc.transform(df_proc[[feature]])) 
        df_proc_enc.columns = one_hot_enc.get_feature_names_out([feature])
        df_proc = df_proc.join(df_proc_enc)
    df_proc = df_proc.drop(columns=categorical_features)

    return df_proc[feature_names]


# Загружаем модель
try:
    PIPELINE, DF_TRAIN, FEATURE_NAMES, ONE_HOT_ENCODERS, MEDIANS = load_data()
    MODEL = PIPELINE.named_steps['mpdel']
    DF_TRAIN_NUM = DF_TRAIN.select_dtypes(include='number')
except Exception as e:
    st.error(f"Ошибка загрузки данных: {e}")
    st.stop()


# --- Основной интерфейс ---
st.title("Предсказание стоимости машин")

with st.spinner('## Подождите, данные обрабатываются...'):
    # --- Графики EDA ---
    st.subheader("Графики и таблицы EDA")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Показать графики и таблдицы EDA"):

            st.subheader("Статистика числовых признаков")
            st.dataframe(DF_TRAIN.describe(), use_container_width=True)

            st.subheader("Статистика категориальных признаков")
            st.dataframe(DF_TRAIN.describe(include="object"), use_container_width=True)

            p = sns.pairplot(DF_TRAIN_NUM, height=2, corner=True)
            p.figure.suptitle("Попарное распределение числовых признаков на train")
            st.pyplot(p.figure)

            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(DF_TRAIN_NUM.corr(), annot = True, ax=ax)
            ax.set_title("Тепловая карта корреляции Пирсона для тренировочного набора данных\n")
            st.pyplot(fig)
            plt.close(fig)


    st.subheader("Графики обученной модели")
    # --- График весов обученной модели---
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Показать графики обученной модели"):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.barh(FEATURE_NAMES, MODEL.coef_)
            ax.set_xlabel("Веса")
            ax.set_ylabel("Признак")
            ax.set_title("График весов обученной модели\n")
            st.pyplot(fig)
            plt.close(fig)

st.subheader("Предсказание цен машин по csv")
# Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is None:
    st.info("Загрузите CSV файл для начала работы")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file)
with st.spinner('## Подождите, данные обрабатываются...'):
    try:
        features = prepare_features(df, FEATURE_NAMES)
        predictions = PIPELINE.predict(features)
        df['prediction'] = predictions
    except Exception as e:
        st.error(f"Ошибка при обработке данных: {e}")
        st.stop()


    # --- Метрики ---
    st.subheader("Результаты")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Всего машин", len(df))
    with col2:
        avg_prob = df['prediction'].mean()
        st.metric("Средняя цена", f"{avg_prob:.1f}")


    st.subheader("Таблица с резщультатами")
    st.dataframe(df, use_container_width=True)

    st.subheader("Скачать результаты")

    result_csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Скачать CSV с результатом",
        data=result_csv,
        file_name="result_csv.csv",
        mime="text/csv"
    )
st.success("Данные обработаны")