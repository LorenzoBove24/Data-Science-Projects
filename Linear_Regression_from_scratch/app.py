import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- 1. CLASSE LINEAR REGRESSION FROM SCRATCH ---
class RegressioneLineare():
    def __init__(self):
       self.coefficienti = None
       self.y_predicted = None
    
    def fit(self, X, y): 
        final_X = X.copy()
        final_X.insert(0, 'ones', 1)    
        X_mat = final_X.to_numpy()  
        y_arr = y.to_numpy()
        coefficienti = np.linalg.inv(np.matmul(X_mat.T, X_mat)) @ np.matmul(X_mat.T, y_arr)
        self.coefficienti = coefficienti     
    
    def predict(self, test_data):    
        mat_testdata = test_data.copy()
        mat_testdata.insert(0, 'ones', 1)      
        mat_testdata = mat_testdata.to_numpy()
        y_predicted = mat_testdata @ self.coefficienti   
        self.y_predicted = y_predicted
        return self.y_predicted
    
    def score(self, y_test):
        y_test_arr = y_test.to_numpy()
        y_mean = y_test_arr.mean()
        SST = np.sum((y_test_arr - y_mean)**2)
        SSE = np.sum((y_test_arr - self.y_predicted)**2)
        R_square = 1 - (SSE/SST)
        return R_square


# --- 2. CONFIGURAZIONE INTERFACCIA STREAMLIT ---
st.set_page_config(page_title="Linear Regression From Scratch", page_icon="📈")
st.title("📈 Linear Regression From Scratch")
st.markdown("""
This project demonstrates the implementation of a **Linear Regression** model built entirely from scratch, using exclusively algebraic operations with the `numpy` library.
""")

# --- 3. CARICAMENTO DATI ---
# Cacheiamo la funzione così Streamlit non ricarica il CSV a ogni click
@st.cache_data
def load_data():
    # Trova il percorso assoluto della cartella in cui si trova app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Unisce il percorso della cartella al nome del file
    file_path = os.path.join(current_dir, "SOCR-HeightWeight.csv")
    
    return pd.read_csv(file_path)

try:
    dataset = load_data()
    final_df = dataset.drop("Index", axis=1)
    
    feature = final_df[["Weight(Pounds)"]]
    target = final_df["Height(Inches)"]

    # --- 4. ADDESTRAMENTO DEL MODELLO ---
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

    modello = RegressioneLineare()
    modello.fit(X_train, y_train)
    y_pred = modello.predict(X_test)
    r2 = modello.score(y_test)

    # Mostriamo una piccola preview del dataset
    st.subheader("📊 Anteprima del Dataset (Flickr30k)")
    st.dataframe(final_df.head(), use_container_width=True)

    # --- 5. RISULTATI E GRAFICO ---
    st.subheader("⚙️ Risultati del Modello")
    st.info(f"**R² Score:** {r2:.4f} \n\n*Ottiene lo stesso identico risultato della libreria Scikit-Learn!*")

    st.subheader("📉 Grafico della Regressione")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_test, y_test, alpha=0.4, label='Dati reali')
    ax.plot(X_test, y_pred, color='red', linewidth=2, label='Linea di regressione (My Model)')
    ax.set_xlabel("Weight (Pounds)")
    ax.set_ylabel("Height (Inches)")
    ax.set_title("Plot of my Linear Regression model")
    ax.legend()
    
    # Renderizza il grafico di matplotlib su Streamlit
    st.pyplot(fig)

    # --- 6. DEMO INTERATTIVA ---
    st.subheader("🔮 Prova il modello: Fai una previsione!")
    st.write("Sposta lo slider per inserire un peso e vedere l'altezza prevista dal modello calcolato da zero.")
    
    min_weight = float(feature.min().iloc[0])
    max_weight = float(feature.max().iloc[0])
    mean_weight = float(feature.mean().iloc[0])
    
    user_input = st.slider("Seleziona il Peso (Pounds)", min_value=min_weight, max_value=max_weight, value=mean_weight)
    
    # Prepariamo l'input per la classe (ha bisogno di un DataFrame per il copy e l'insert)
    input_df = pd.DataFrame({"Weight(Pounds)": [user_input]})
    predicted_height = modello.predict(input_df)[0]
    
    st.success(f"L'altezza calcolata per **{user_input:.2f} lbs** è di **{predicted_height:.2f} inches**.")

except FileNotFoundError:
    st.error("⚠️ Errore: File `SOCR-HeightWeight.csv` non trovato. Assicurati di aver inserito il dataset nella stessa cartella di questa app.")