# DNA-Promoter-Prediction
A Flask-based web app that predicts DNA promoter sequences using a Random Forest Classifier trained on sequence features.

# 🚀 Features
- **DNA Sequence Input**: Paste any sequence for analysis  
- **ML-Powered Predictions**: Uses **Scikit-Learn RandomForestClassifier**  
- **Web Interface**: Simple Flask app UI  
- **API Support**: Get predictions via JSON

#   Installation
1. Clone repo:
   ```bash
   git clone https://github.com/devansh1331/DNA-Promoter-Prediction.git
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the Flask app:
   ```bash
   python app.py
   
# ⚙️ How It Works
This project takes DNA sequences and extracts relevant features like nucleotide composition, dinucleotide frequency, and GC content. These features are then passed into a **Random Forest Classifier**, which predicts whether the given sequence is a **promoter** or **non-promoter**.


