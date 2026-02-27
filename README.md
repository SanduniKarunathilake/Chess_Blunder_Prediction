# ♟️ Chess Blunder Prediction

A machine learning project that predicts chess blunders based on board positions and player ELO ratings, with a full-stack web interface powered by the Stockfish chess engine.

---

## 📁 Project Structure

```
Blunder_chess/
├── ML/
│   ├── src/
│   │   ├── features.py        # Feature extraction
│   │   ├── preprocessing.py   # Data preprocessing
│   │   ├── train.py           # Model training
│   │   ├── predict.py         # Prediction logic
│   │   └── elo.py             # ELO rating utilities
│   ├── models/
│   │   ├── blunder_model.pkl
│   │   └── chess_opening_model.pkl
│   ├── data/raw/
│   │   ├── positions.csv
│   │   └── sample_game.pgn
│   ├── blunder.ipynb
│   ├── app.py
│   └── requirements.txt
├── backend/
│   ├── app.py
│   ├── engine/
│   │   └── stockfish/         # ⚠️ Add Stockfish binary here (see setup below)
│   ├── model/
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── blunder.html
│   ├── elo.html
│   ├── script.js
│   └── style.css
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/SanduniKarunathilake/Chess_Blunder_Prediction.git
cd Chess_Blunder_Prediction
```

### 2. ⚠️ Download Stockfish (Required)

The Stockfish chess engine binary is **not included** in this repository due to its large file size. You must download it manually.

**Steps:**
1. Go to the official Stockfish download page: https://stockfishchess.org/download/
2. Download the Windows version: `stockfish-windows-x86-64-avx2.exe`
3. Place the file in this exact path inside the project:

```
backend/engine/stockfish/stockfish-windows-x86-64-avx2.exe
```

> **Note:** If you are on Linux or Mac, download the appropriate binary for your OS and update the engine path in `backend/app.py` accordingly.

---

### 3. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Install ML Dependencies

```bash
cd ML
pip install -r requirements.txt
```

### 5. Run the Backend

```bash
cd backend
python app.py
```

### 6. Open the Frontend

Open `frontend/index.html` in your browser, or serve it with a local server:

```bash
cd frontend
python -m http.server 8080
```

Then visit: `http://localhost:8080`

---

## 🤖 ML Model

The machine learning model predicts whether a chess move is a **blunder** based on:

- Board position features
- Player ELO rating
- Move history
- Opening classification

### Training the Model

```bash
cd ML
python src/train.py
```

### Running Predictions

```bash
cd ML
python src/predict.py
```

Or explore the full pipeline in the Jupyter notebook:

```bash
jupyter notebook blunder.ipynb
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | Python, scikit-learn, pandas |
| Chess Engine | Stockfish |
| Backend | Python (Flask) |
| Frontend | HTML, CSS, JavaScript |

---

## 📌 Requirements

- Python 3.x
- Stockfish chess engine (downloaded separately — see setup above)
- pip packages listed in `requirements.txt`

---

## 👩‍💻 Author

**Sanduni Karunathilake**  
[GitHub](https://github.com/SanduniKarunathilake)
