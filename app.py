import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# Настройка страницы
st.set_page_config(
    page_title="2023-фгииб-пи1б_7_Дубов_Артём_Анатольевич_датасет_рис",
    layout="wide"
)
st.title("ИвановИ.И._Группа7_RiceDataset")

# Загрузка и краткое описание датасета
@st.cache_data
def load_data(path="data.csv"):
    df = pd.read_csv(path)
    return df

df = load_data()

# Разделение данных и обучение модели
X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    bootstrap_type="Bernoulli",
    subsample=0.8,
    random_seed=42,
    verbose=0
)
model.fit(X_train, y_train)

# Оценка точности модели
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average="macro")

# Интерпретация модели через SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Организация в табы для "страниц"
tab1, tab2, tab3 = st.tabs(["Описание и метрики", "Матрица ошибок и SHAP", "Распределение признаков"])

with tab1:
    st.markdown("**Краткое описание набора данных:**\n"
                f"- Количество образцов: {df.shape[0]}\n"
                f"- Количество признаков: {df.shape[1]-1} (без целевой переменной)\n"
                f"- Признаки: {', '.join(df.columns.drop('Class'))}\n"
                f"- Типы данных: {df.dtypes.value_counts().to_dict()}")
    
    st.subheader("Точность модели")
    st.write(f"- Accuracy: **{acc:.4f}**")
    st.write(f"- Recall (macro): **{recall:.4f}**")

with tab2:
    # Параллельное размещение графиков в колонках
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Матрица ошибок (Confusion Matrix)")
        cm = confusion_matrix(y_test, y_pred)
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        st.pyplot(fig1)
    
    with col2:
        st.subheader("SHAP: Важность признаков (bar plot)")
        st.image("1.png")

    
    with col3:
        st.subheader("SHAP: Распределение влияния признаков (dot plot)")
        st.image("2.png", use_column_width=True)


with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Гистограмма признака Area**")
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        sns.histplot(data=df, x="Area", kde=True, ax=ax4, bins=30, color="skyblue")
        st.pyplot(fig4)
    
    with col2:
        st.markdown("**Диаграмма разброса Area vs. MajorAxisLength**")
        fig5, ax5 = plt.subplots(figsize=(4, 3))
        sns.scatterplot(data=df, x="Area", y="MajorAxisLength", hue="Class", palette="Set1", s=20, ax=ax5)
        st.pyplot(fig5)
