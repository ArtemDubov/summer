import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

st.set_page_config(
    page_title="2023-фгииб-пи1б_7_Дубов_Артём_Анатольевич_датасет_рис",
    layout="wide",
    initial_sidebar_state="expanded"  # Добавляем боковую панель для навигации
)

st.markdown("""
    <style>
    .main {background-color: #f0f8ff;}  /* Светлый фон */
    h1 {color: #2e8b57;}  /* Цвет заголовка */
    .stTabs [data-baseweb="tab"] {background-color: #add8e6; border-radius: 5px;}  /* Стиль табов */
    </style>
    """, unsafe_allow_html=True)

st.title("2023пи1б_7_Дубов_Артём_Анатольевич_датасет_рис")

with st.sidebar:
    st.header("Навигация")
    st.markdown("Используйте табы ниже для просмотра разделов.")

@st.cache_data
def load_data(path="data.csv"):
    df = pd.read_csv(path)
    return df

df = load_data()

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

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average="macro")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

tab1, tab2, tab3 = st.tabs(["📊 Описание и метрики", "🔍 Матрица ошибок и SHAP", "📈 Распределение признаков"])

with tab1:
    st.markdown("### Краткое описание набора данных")
    st.info(f"- Количество образцов: {df.shape[0]}\n"
            f"- Количество признаков: {df.shape[1]-1} (без целевой переменной)\n"
            f"- Признаки: {', '.join(df.columns.drop('Class'))}\n"
            f"- Типы данных: {df.dtypes.value_counts().to_dict()}")  
    
    st.markdown("### Точность модели")
    st.success(f"- Accuracy: **{acc:.4f}**\n- Recall (macro): **{recall:.4f}**") 

with tab2:
    st.markdown("### Анализ модели")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Матрица ошибок")
        cm = confusion_matrix(y_test, y_pred)
        fig1, ax1 = plt.subplots(figsize=(5, 4)) 
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax1)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        st.pyplot(fig1)
    
    with col2:
        st.subheader("SHAP: Важность признаков")
        st.image("1.png", use_container_width=True, caption="Bar plot")  # Добавили подпись
    
    with col3:
        st.subheader("SHAP: Распределение влияния")
        st.image("2.png", use_container_width=True, caption="Dot plot")

with tab3:
    st.markdown("### Визуализация данных")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Гистограмма Area**")
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        sns.histplot(data=df, x="Area", kde=True, ax=ax4, bins=30, color="lightgreen")  # Изменили цвет
        st.pyplot(fig4)
    
    with col2:
        st.markdown("**Scatter Plot: Area vs. MajorAxisLength**")
        fig5, ax5 = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df, x="Area", y="MajorAxisLength", hue="Class", palette="coolwarm", s=30, ax=ax5)  # Изменили палитру и размер точек
        st.pyplot(fig5)

# Футер с дополнительной информацией
st.markdown("---")
st.caption("Разработано Дубовым Артёмом Анатольевичем | 2023 | Датасет по рису")  # Добавили футер
