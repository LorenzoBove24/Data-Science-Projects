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
    st.subheader("📊 Preview of Data ")
    st.dataframe(final_df.head(), use_container_width=True)

    # --- 5. RISULTATI E GRAFICO ---
    st.subheader("⚙️ Results of the Model")
    st.info(f"**R² Score:** {r2:.4f} \n\n*Obtains the same exact result as the Scikit-Learn library!*")

    st.subheader("📉 Regression Plot")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_test, y_test, alpha=0.4, label='Real Data')
    ax.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line (My Model)')
    ax.set_xlabel("Weight (Pounds)")
    ax.set_ylabel("Height (Inches)")
    ax.set_title("Plot of my Linear Regression model")
    ax.legend()
    
    # Renderizza il grafico di matplotlib su Streamlit
    st.pyplot(fig)

    # --- 6. DEMO INTERATTIVA: MODIFICA I PARAMETRI ---
    st.subheader("🎛️ Interactive Demo: Modify the Line!")
    st.write("Play with the y-intercept and slope to see how the coefficients change the line. Try to overlay it perfectly on the data!")
    
    # Estraiamo i parametri ottimali calcolati dal tuo modello from scratch
    # coefficienti[0] è l'intercetta (perché abbiamo aggiunto la colonna di 1)
    # coefficienti[1] è la pendenza (il peso della feature)
    ottimo_b0 = float(modello.coefficienti[0]) 
    ottimo_b1 = float(modello.coefficienti[1]) 
    
    col1, col2 = st.columns(2)
    with col1:
        # Slider per l'intercetta (lo facciamo variare attorno al valore ottimale)
        user_b0 = st.slider("Y-Intercept (b0)", min_value=ottimo_b0 - 20.0, max_value=ottimo_b0 + 20.0, value=ottimo_b0, step=0.5)
    with col2:
        # Slider per la pendenza
        user_b1 = st.slider("Slope (b1)", min_value=ottimo_b1 - 1.0, max_value=ottimo_b1 + 1.0, value=ottimo_b1, step=0.01)
        
    # Calcoliamo i punti della retta personalizzata (y = b0 + b1*x)
    y_custom = user_b0 + (user_b1 * X_test["Weight(Pounds)"])
    
    # Calcoliamo l'errore per dare un feedback visivo immediato
    mse_custom = np.mean((y_test - y_custom)**2)
    mse_ottimo = np.mean((y_test - y_pred)**2)
    
    st.info(f"**Mean Squared Error (MSE) of your line:** {mse_custom:.2f}  \n*(The minimum MSE calculated by the algorithm is {mse_ottimo:.2f})*")

    # --- 7. GRAFICO DINAMICO ---
    st.subheader("📉 Interactive Regression Plot")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Disegna i dati storici
    ax.scatter(X_test, y_test, alpha=0.4, label='Real Data (Test set)')
    
    # Disegna la retta ottimale del tuo modello (tratteggiata per confronto)
    ax.plot(X_test, y_pred, color='red', linestyle='--', linewidth=2, label='Optimal Line (Model)')
    
    # Disegna la retta governata dagli slider
    ax.plot(X_test, y_custom, color='lime', linewidth=3, label='Your Interactive Line')
    
    ax.set_xlabel("Weight (Pounds)")
    ax.set_ylabel("Height (Inches)")
    ax.set_title("Interactive Linear Regression")
    ax.legend()
    
    # Mostra il grafico aggiornato in tempo reale
    st.pyplot(fig)

except FileNotFoundError:
    st.error("⚠️ Error: File `SOCR-HeightWeight.csv` not found. Make sure you have placed the dataset in the same folder as this app.")