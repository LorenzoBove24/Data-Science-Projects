import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris
import os # <--- Aggiungi questo import fondamentale

# --- 1. SETUP INIZIALE E GESTIONE PERCORSI ---
# Troviamo la cartella dove si trova QUESTO file app.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Creiamo i percorsi completi per i file .pkl
# (Così funzionerà sia sul tuo PC che su Streamlit Cloud)
path_scaler = os.path.join(current_dir, 'scaler.pkl')
path_rf = os.path.join(current_dir, 'modello_rf.pkl')
path_knn = os.path.join(current_dir, 'modello_knn.pkl')

iris_names = load_iris().target_names 

st.write("""
# 🌸 Iris Flower Classifier
Questa app usa modelli **già addestrati** (Random Forest e KNN) per predire la specie del fiore.
""")

# --- CARICAMENTO SCALER CON PERCORSO SICURO ---
try:
    scaler = joblib.load(path_scaler)
except FileNotFoundError:
    st.error(f"Errore: Manca il file 'scaler.pkl'. Il sistema lo cercava qui: {path_scaler}")
    st.stop()

# --- 2. INPUT LATERALE ---
st.sidebar.header('1. Inserisci le misure')

def user_input_features():
    # Creiamo gli slider
    sepal_length = st.sidebar.slider('Lunghezza Sepalo (cm)', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Larghezza Sepalo (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Lunghezza Petalo (cm)', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Larghezza Petalo (cm)', 0.1, 2.5, 0.2)
    
    # Creiamo il DataFrame con i TUOI nomi variabili originali
    data = {
        'SepalLengthCm': sepal_length,
        'SepalWidthCm': sepal_width,
        'PetalLengthCm': petal_length,
        'PetalWidthCm': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('I parametri che hai inserito:')
st.write(df_input)

# --- 3. SCELTA E CARICAMENTO MODELLO ---
st.sidebar.header('2. Scegli il Modello')
model_choice = st.sidebar.selectbox("Algoritmo", ["Random Forest", "KNN"])

@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

if model_choice == "Random Forest":
    clf = load_model(path_rf) # <--- Usa la variabile path_rf
else:
    clf = load_model(path_knn) # <--- Usa la variabile path_knn

if clf is None:
    st.error("Errore nel caricamento del modello. Controlla i percorsi.")
    st.stop()
else:
    st.sidebar.success(f"✅ Modello {model_choice} caricato!")

# --- 4. PREDIZIONE ---

if model_choice == "KNN":
    # Se usiamo KNN, dobbiamo scalare i dati perché è stato addestrato su dati scalati
    # (Assicurati di aver caricato lo scaler come detto prima)
    input_dati = scaler.transform(df_input)
else:
    # Se usiamo Random Forest (e l'hai addestrata sui dati originali),
    # passiamo i dati "puri" così come arrivano dagli slider.
    # Usiamo .values per sicurezza sui nomi delle colonne
    input_dati = df_input.values

# Ora passiamo la variabile giusta (input_dati) al modello
prediction = clf.predict(input_dati)
prediction_proba = clf.predict_proba(input_dati)
# --- 5. VISUALIZZAZIONE RISULTATI ---
st.markdown("---")
st.subheader('Risultato della Predizione:')

col1, col2 = st.columns([1, 2])

with col1:
    # Mostra il nome della specie in grande e colorato
    # iris_names è un array, prediction restituisce un array di indici (es. [0])
    specie_predetta = iris_names[prediction][0].upper()
    
    if specie_predetta == "SETOSA":
        st.success(f"### 🌸 {specie_predetta}")
    elif specie_predetta == "VERSICOLOR":
        st.info(f"### 🌺 {specie_predetta}")
    else:
        st.warning(f"### 🌼 {specie_predetta}")

with col2:
    st.write("Confidenza del modello:")
    proba_df = pd.DataFrame(prediction_proba, columns=iris_names)
    st.bar_chart(proba_df.T)

import seaborn as sns
import matplotlib.pyplot as plt

# --- 6. VISUALIZZAZIONE GRAFICA AVANZATA ---
st.markdown("---")
st.subheader("📊 Dove si colloca il tuo fiore?")
st.write("I grafici mostrano la distribuzione delle misure nel dataset originale. La **linea rossa** indica il valore che hai inserito.")

# 1. Prepariamo il DataFrame originale per i grafici
iris_data = load_iris()
df_source = pd.DataFrame(iris_data.data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
# Aggiungiamo la colonna specie per colorare i grafici
df_source['Specie'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

# 2. Creiamo una griglia 2x2 per i grafici
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# Appiattiamo l'array degli assi per iterarci facilmente (da [[ax1, ax2], [ax3, ax4]] a [ax1, ax2, ax3, ax4])
axes_flat = axes.flatten()

# Lista delle colonne da plottare
columns_to_plot = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# 3. Ciclo per creare i 4 grafici
for i, col_name in enumerate(columns_to_plot):
    ax = axes_flat[i]
    
    # Disegna l'istogramma (distribuzione) colorato per specie
    sns.histplot(data=df_source, x=col_name, hue='Specie', kde=True, ax=ax, palette='Set2', element="step")
    
    # Prende il valore inserito dall'utente per questa variabile
    user_value = df_input[col_name].values[0]
    
    # Aggiunge la linea verticale rossa
    ax.axvline(user_value, color='red', linestyle='--', linewidth=2, label='Il tuo fiore')
    
    # Pulizia grafico
    ax.set_title(col_name)
    ax.set_xlabel('')
    ax.legend([],[], frameon=False) # Nascondiamo la legenda interna per pulizia (opzionale)

# Aggiusta il layout per non sovrapporre le scritte
plt.tight_layout()

# Mostra il grafico in Streamlit
st.pyplot(fig)