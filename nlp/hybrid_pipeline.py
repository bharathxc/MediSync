"""
Hybrid Pipeline: Traditional NLP (Presidio/spaCy) for high recall,
followed by GenAI (Ollama) for high precision refinement.
"""
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from langchain_core.messages import SystemMessage, HumanMessage
import json

from config import get_llm_deterministic, NLP_DEID_SYSTEM_PROMPT


class HybridDeidentifier:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.llm = get_llm_deterministic()

    def traditional_pass(self, text: str) -> dict:
        """Step 1: Use Presidio (spaCy backend) to detect PII."""
        results = self.analyzer.analyze(text=text, language='en')
        
        operators = {
            "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
            "PERSON": OperatorConfig("replace", {"new_value": "[REDACTED_NAME]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[REDACTED_PHONE]"}),
            "DATE_TIME": OperatorConfig("replace", {"new_value": "[REDACTED_DATE]"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "[REDACTED_HOSPITAL]"}),
        }

        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators
        )
        
        flagged_entities = []
        for result in results:
            flagged_entities.append({
                "entity": result.entity_type,
                "text": text[result.start:result.end],
                "score": result.score
            })

        return {
            "anonymized_text": anonymized_result.text,
            "flagged_entities": flagged_entities,
            "original_text": text
        }

    def genai_refinement_pass(self, original_text: str, traditional_result: dict) -> str:
        """Step 2: Use LLM to refine the redaction, preventing medical terminology from being redacted."""
        if not traditional_result["flagged_entities"]:
            return original_text
            
        prompt = f"""Raw Clinical Text: 
{original_text}

Entities flagged by traditional NLP model:
{json.dumps(traditional_result['flagged_entities'], indent=2)}

Please output ONLY the properly refined redacted text, followed by your REASONING.
"""

        messages = [
            SystemMessage(content=NLP_DEID_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def process_note(self, text: str):
        trad_result = self.traditional_pass(text)
        final_text = self.genai_refinement_pass(text, trad_result)
        
        reasoning = ""
        if "REASONING:" in final_text:
            parts = final_text.split("REASONING:")
            final_text = parts[0].strip()
            reasoning = parts[1].strip()
            
        return {
            "traditional_redaction": trad_result["anonymized_text"],
            "flagged_entities": trad_result["flagged_entities"],
            "hybrid_redaction": final_text,
            "reasoning": reasoning
        }


if __name__ == "__main__":
    test_text = "Patient John Smith was admitted to Mayo Clinic on 2024-05-12. Diagnosed with Parkinson's disease. Call 555-123-4567."
    print("Testing Pipeline...")
    pipeline = HybridDeidentifier()
    result = pipeline.process_note(test_text)
    print("\n--- Traditional Redaction ---")
    print(result['traditional_redaction'])
    print("\n--- Hybrid GenAI Redaction ---")
    print(result['hybrid_redaction'])
    print(f"\nReasoning: {result['reasoning']}")