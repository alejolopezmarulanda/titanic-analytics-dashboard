from sklearn.linear_model import LogisticRegression
import streamlit as st

@st.cache_resource
def train_model(df):
    df_ml = df[['Survived', 'Pclass', 'Age', 'Fare']].dropna()
    X = df_ml[['Pclass', 'Age', 'Fare']]
    y = df_ml['Survived']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model
