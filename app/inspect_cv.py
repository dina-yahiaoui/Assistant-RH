# app/inspect_cv.py
from pathlib import Path
import pandas as pd

# app/ → racine du projet (Assistant-RH)
BASE_DIR = Path(__file__).resolve().parent.parent

csv_path = BASE_DIR / "data" / "cv" / "resumes.csv"

print("CSV path:", csv_path)

# On suppose que c'est un CSV normal (séparateur = virgule)
df = pd.read_csv(csv_path)

print("Colonnes :", df.columns.tolist())
print(df.head(3))
