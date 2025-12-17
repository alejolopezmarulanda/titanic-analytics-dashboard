import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from src.data_loader import load_data
from src.model import train_model
from styles.loader import load_css

# -------------------------------------------------
# CONFIGURACIÓN (SIEMPRE PRIMERO)
# -------------------------------------------------
st.set_page_config(
    page_title="Titanic Analytics",
    layout="wide"
)

# -------------------------------------------------
# CARGAR CSS
# -------------------------------------------------
load_css()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
col_logo, col_title, col_social = st.columns([1, 4, 2])

with col_logo:
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        st.image(str(logo_path), width=90)
    else:
        st.markdown("### 🚢 Titanic Analytics")


with col_title:
    st.markdown("""
    <h2 class="header-title">Titanic Analytics</h2>
    <p class="header-subtitle">Interactive Data Dashboard</p>
    """, unsafe_allow_html=True)

with col_social:
    st.markdown("""
    <div class="social-links">   
        <a href="https://www.linkedin.com/in/alejandro-lopez-marulanda-0754571b9" target="_blank">
            LinkedIn
        </a>
        <a href="https://github.com/alejolopezmarulanda/titanic-analytics-dashboard" target="_blank">
            GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# CARGAR DATOS
# -------------------------------------------------
df = load_data()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.markdown("## 🎛️ Filtros")

sex_filter = st.sidebar.multiselect(
    "Sexo",
    df['Sex'].unique(),
    df['Sex'].unique()
)

pclass_filter = st.sidebar.multiselect(
    "Clase",
    sorted(df['Pclass'].unique()),
    sorted(df['Pclass'].unique())
)

df_filtered = df[
    (df['Sex'].isin(sex_filter)) &
    (df['Pclass'].isin(pclass_filter))
]

# -------------------------------------------------
# KPIs
# -------------------------------------------------
st.markdown("### 📊 Indicadores clave")

k1, k2, k3 = st.columns(3)
k1.metric("Pasajeros", len(df_filtered))
k2.metric("Sobrevivientes", int(df_filtered['Survived'].sum()))
k3.metric(
    "Tasa de supervivencia",
    f"{df_filtered['Survived'].mean()*100:.1f}%"
)

# -------------------------------------------------
# GRÁFICOS
# -------------------------------------------------
st.markdown("### 📈 Análisis")

c1, c2 = st.columns(2)

with c1:
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df_filtered, x="Survived", hue="Sex", ax=ax1)
    st.pyplot(fig1)

with c2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df_filtered, x="Survived", y="Age", ax=ax2)
    st.pyplot(fig2)

# -------------------------------------------------
# MACHINE LEARNING
# -------------------------------------------------
st.markdown("---")
st.markdown("### 🤖 Predicción de Supervivencia")

model = train_model(df)

c1, c2, c3 = st.columns(3)

with c1:
    pclass = st.selectbox("Clase", [1, 2, 3])
with c2:
    age = st.slider("Edad", 0, 80, 30)
with c3:
    fare = st.slider("Tarifa", 0, 500, 50)

if st.button("Predecir"):
    pred = model.predict([[pclass, age, fare]])
    st.success("✅ Sobrevive" if pred[0] else "❌ No sobrevive")
