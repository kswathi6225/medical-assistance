import pandas as pd
import os

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "../data/medical_text_structured/medicine_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

def get_medicine_info(medicine_name):
    if not medicine_name:
        return None

    result = df[df["Name"].str.lower() == medicine_name.lower()]

    if result.empty:
        return None

    row = result.iloc[0]

    return {
        "name": row["Name"],
        "category": row["Category"],
        "dosage_form": row["Dosage Form"],
        "strength": row["Strength"],
        "manufacturer": row["Manufacturer"],
        "indication": row["Indication"],
        "classification": row["Classification"]
    }
