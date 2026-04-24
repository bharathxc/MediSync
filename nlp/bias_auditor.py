"""
Bias Auditing Engine using Presidio (Traditional ML).
Evaluates if the NER model systematically misses names for certain demographic groups.
"""
import os
import json
import pandas as pd
from nlp.hybrid_pipeline import HybridDeidentifier
from config import DATA_DIR
from typing import List, Dict


class BiasAuditor:
    def __init__(self):
        self.deidentifier = HybridDeidentifier()
        test_data_path = os.path.join(DATA_DIR, "test_clinical_notes.json")
        if os.path.exists(test_data_path):
            self.dataset_path = test_data_path
        else:
            self.dataset_path = None
    
    def load_dataset(self) -> List[Dict]:
        if not self.dataset_path:
            return self._generate_synthetic_dataset()
        
        with open(self.dataset_path, "r") as f:
            return json.load(f)
    
    def _generate_synthetic_dataset(self) -> List[Dict]:
        """Generate synthetic test data if no dataset exists."""
        return [
            {"id": "1", "text": "Patient John Smith was admitted to Mayo Clinic.", "patient_name": "John Smith", "demographic_group": "White Male"},
            {"id": "2", "text": "Patient Maria Garcia treated at St. Mary's Hospital.", "patient_name": "Maria Garcia", "demographic_group": "Hispanic Female"},
            {"id": "3", "text": "Patient Wei Chen discharged from General Hospital.", "patient_name": "Wei Chen", "demographic_group": "Asian Male"},
            {"id": "4", "text": "Patient Aisha Mohammed admitted for observation.", "patient_name": "Aisha Mohammed", "demographic_group": "Middle Eastern Female"},
            {"id": "5", "text": "Patient Kwame Okonkwo seen at Regional Medical.", "patient_name": "Kwame Okonkwo", "demographic_group": "Black Male"},
            {"id": "6", "text": "Patient Priya Patel treated in the ICU.", "patient_name": "Priya Patel", "demographic_group": "Indian Female"},
            {"id": "7", "text": "Patient Mohammad Hassan in room 302.", "patient_name": "Mohammad Hassan", "demographic_group": "Middle Eastern Male"},
            {"id": "8", "text": "Patient Tao Lin transferred to main campus.", "patient_name": "Tao Lin", "demographic_group": "Asian Female"},
        ]

    def evaluate_model(self):
        """Runs the traditional NER model over the synthetic dataset."""
        dataset = self.load_dataset()
        results = []

        for item in dataset:
            text = item["text"]
            patient_name = item["patient_name"]
            demo_group = item["demographic_group"]

            trad_result = self.deidentifier.traditional_pass(text)
            flagged = trad_result["flagged_entities"]
            
            name_detected = False
            for entity in flagged:
                if entity["entity"] == "PERSON" and (entity["text"] in patient_name or patient_name in entity["text"]):
                    name_detected = True
                    break

            results.append({
                "id": item["id"],
                "demographic_group": demo_group,
                "patient_name": patient_name,
                "detected": name_detected
            })

        df = pd.DataFrame(results)
        
        summary = df.groupby('demographic_group')['detected'].agg(['count', 'sum']).reset_index()
        summary['detection_rate'] = summary['sum'] / summary['count']
        summary = summary.rename(columns={'count': 'total_samples', 'sum': 'true_positives'})
        
        return df, summary


if __name__ == "__main__":
    print("Running Bias Auditor...")
    auditor = BiasAuditor()
    df, summary = auditor.evaluate_model()
    print("\n--- Fairness Metrics ---")
    print(summary.to_string(index=False))