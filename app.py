# ──────────────────────────────────────────────────────────────
# Листинг 1. streamlit_app.py – интерактивный дашборд по датасету риса
# Автор: Дубов Артём • Вариант 7
# ──────────────────────────────────────────────────────────────

import os, warnings, pathlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from catboost import CatBoostClassifier
import shap
from streamlit_shap import st_shap   # pip install streamlit-shap

warnings.filterwarnings("ignore")

DATA_PATH   = "./data.csv"   # файл с исходными данными
TARGET_COL  = "Class"        # название таргета

# ──────────────────────────────────────────────────────────────
# Конфигурация страницы
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Дубов_Артём_Вар7_Rice",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ──────────────────────────────────────────────────────────────
# Листинг 2. Загрузка и кэширование данных
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not pathlib.Path(path).exists():
        st.error(f"Файл {path} не найден"); st.stop()
    df = pd.read_csv(path, low_memory=False)
    return df

# ──────────────────────────────────────────────────────────────
# Листинг 3. Обучение модели CatBoost + SHAP (кэш результата)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def train_model(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1,
        loss_function="Logloss", eval_metric="Accuracy",
        verbose=False, random_seed=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = dict(
        accuracy = accuracy_score(y_test, preds),
        recall   = recall_score(y_test, preds, average="macro"),
        f1       = f1_score(y_test, preds, average="macro"),
        cm       = confusion_matrix(y_test, preds, labels=np.unique(y))
    )

    # SHAP-значения
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return model, X_test, y_test, metrics, shap_values, X.columns

# ──────────────────────────────────────────────────────────────
# Листинг 4. Страница «Обзор»
# ──────────────────────────────────────────────────────────────
def OverviewPage():
    st.header("📊 Обзор набора данных")
    df = load_data(DATA_PATH)

    # Основные сведения
    st.markdown(f"**Размер:** {df.shape[0]} строк × {df.shape[1]} столбцов")
    st.dataframe(df.head(), use_container_width=True)

    # Пропуски по столбцам
    na_counts = df.isna().sum()
    if na_counts.sum() > 0:
        fig = px.bar(
            na_counts[na_counts > 0],
            orientation='v',
            labels={'value':'Количество пропусков','index':'Признак'},
            title="Число пропусков по признакам"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Тепловая карта NAN
        st.subheader("Карта пропусков")
        fig2 = px.imshow(df.isna(), aspect="auto", color_continuous_scale="Reds",
                         labels=dict(color="Пропуск"),
                         title="Структура пропусков (True = NaN)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.success("🚀 Пропущенных значений нет.")

# ──────────────────────────────────────────────────────────────
# Листинг 5. Страница «Признаки»
# ──────────────────────────────────────────────────────────────
def FeaturesPage():
    st.header("🧮 Исследование признаков")
    df = load_data(DATA_PATH)

    numeric_cols = (
        df.drop(columns=[TARGET_COL])
          .select_dtypes(include=["int64","float64"])
          .columns.tolist()
    )
    if len(numeric_cols) < 2:
        st.error("Недостаточно числовых признаков для визуализации"); return

    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Ось X", numeric_cols, index=0)
    with col2:
        y_feature = st.selectbox("Ось Y", numeric_cols, index=1)

    # Диаграмма рассеяния
    fig = px.scatter(
        df, x=x_feature, y=y_feature, color=df[TARGET_COL].astype(str),
        title=f"Взаимосвязь {x_feature} vs {y_feature}",
        opacity=0.8
    )
    st.plotly_chart(fig, use_container_width=True)

    # Корреляционная матрица
    st.subheader("Корреляционная матрица")
    corr = df[numeric_cols].corr().round(2)
    fig_corr = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        title="Матрица корреляций"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# Листинг 6. Страница «Модель CatBoost»
# ──────────────────────────────────────────────────────────────
def ModelPage():
    st.header("🤖 Обучение модели CatBoost + интерпретация")
    df = load_data(DATA_PATH)

    model, X_test, y_test, metrics, shap_values, feature_names = train_model(df)

    # Метрики
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    mcol2.metric("Recall",   f"{metrics['recall']:.3f}")
    mcol3.metric("F1-score", f"{metrics['f1']:.3f}")

    # Confusion Matrix
    st.subheader("Матрица ошибок")
    cm = metrics["cm"]
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Предсказано", y="Истинно", color="Частота"),
        x=np.unique(y_test), y=np.unique(y_test),
        title="Confusion Matrix"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Исправление: создаем фигуру явно
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Бар важности признаков (без изменений)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.tight_layout()
    st.pyplot(fig2)


# ──────────────────────────────────────────────────────────────
# Листинг 7. Навигация между страницами (st.navigation)
# ──────────────────────────────────────────────────────────────
def main():
    pages = [
        st.Page(OverviewPage,  title="Обзор",      icon="📊"),
        st.Page(FeaturesPage,  title="Признаки",   icon="🧮"),
        st.Page(ModelPage,     title="Модель",     icon="🤖"),
    ]
    pg = st.navigation(pages, position="sidebar", expanded=True)
    pg.run()

if __name__ == "__main__":
    main()
