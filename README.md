# Hose Price Competition

<!-- FIGURA 1: Contenuto vs layer -->
<p align="center">
  <img src="Home price/Plots/flag.PNG" alt="Ricostruzione del contenuto al variare del layer (conv1_2 → conv5_2)" width="900">
</p>

## Obiettivo
Costruire un modello di Machine Learning in grado di predire il prezzo di vendita di una casa.

## Dataset
Il dataset è costruito come indicato nel file presente nella cartella data e presenta alcune criticità:

- Dimensione ridotta del campione.

- Distribuzione concentrata nella fascia di prezzo 100k–500k.

- Eterogeneità delle feature: numeriche, con range molto diversi di valori, categoriche.

## Strategia

- È stato scelto XGBoost come modello principale, puntando soprattutto sulla qualità dei dati per migliorare le performance.

- Feature engineering: creazione di variabili più rappresentative (ad es. numero totale di bagni, superficie totale). È stata inoltre introdotta una flag LuxuryHome per aiutare il modello a gestire meglio il range > 500k.

- Analisi delle correlazioni tra feature ed eliminazione di variabili fortemente collineari per ridurre rumore e varianza.

- Correzione della skewness: trasformazioni logaritmiche sulle feature con skewness elevata (ad es. > 5).

Tutti i passaggi di analisi dati sono documentati nel codice/notebook del progetto.

## Risultati

- RMSE medio (tra predetto e reale) < 10% del prezzo della casa.

- Top 3.1% nella leaderboard della competizione Kaggle.

<!-- FIGURA 1: prezzi predetti vs prezzi veri -->
<p align="center">
  <img src="Home price/Plots/Figura_1.png" alt="prezzi predetti vs prezzi veri" width="900">
</p>

---

# Prerequisiti:

- Python 3.9+

- pip aggiornato

- (Opzionale) GPU NVIDIA con driver aggiornati

  ---

# Guida d'utilizzo

> Esegui i comandi dalla cartella del progetto (es. `C:\NTS`).  
> Le immagini di input vanno in `data/`, l’output viene salvato in `results/`.

## 1) Preparazione ambiente

### Windows (PowerShell)
```powershell
D:
cd House price
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux (Bash)
```
cd ~/House price
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
# 2) Inserisci i dati

I file del dataset li trovi all'interno della repository data

#3) Esecuzione
Avvio con i default

Se i file sopra esistono:
```
python .\src\core.py
```

## Contatti

- Email: <eugenioquaglia@gmail.com>
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/eugenio-quaglia-86114a372/)
- Portfolio: [Portfolio](https://github.com/EugeQuaglia/Portfolio/tree/main?tab=readme-ov-file)
