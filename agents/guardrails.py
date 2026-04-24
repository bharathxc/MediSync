"""
MediSync AI — Agent Guardrails
PII masking and grounding enforcement for healthcare data privacy.
"""
import re
from typing import Tuple


PII_PATTERNS = [
    (r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', '[REDACTED_SSN]', 'SSN'),
    (r'\b[1-9][A-Za-z][A-Za-z0-9]\d[-\s]?[A-Za-z][A-Za-z0-9]\d[-\s]?[A-Za-z]{2}\d{2}\b',
     '[REDACTED_MEDICARE_ID]', 'Medicare ID'),
    (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[REDACTED_CC]', 'Credit Card'),
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', 'Email'),
    (r'(?:\(\d{3}\)\s?|\b\d{3}[-.\s])\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]', 'Phone'),
    (r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b',
     '[REDACTED_DOB]', 'Date of Birth'),
]


def mask_pii(text: str) -> Tuple[str, list[dict]]:
    """Scan text for PII patterns and replace with redaction tokens."""
    detections = []
    masked = text

    for pattern, replacement, pii_type in PII_PATTERNS:
        matches = re.findall(pattern, masked)
        if matches:
            for match in matches:
                detections.append({
                    "type": pii_type,
                    "original_length": len(match),
                    "replacement": replacement
                })
            masked = re.sub(pattern, replacement, masked)

    return masked, detections


def format_grounding_context(context_chunks: list[str], plan_type: str = None) -> str:
    """Format retrieved context with grounding instructions for the LLM."""
    header = "=" * 60 + "\n"
    header += "RETRIEVED CONTEXT DOCUMENTS (Use ONLY these to answer)\n"
    header += "=" * 60 + "\n\n"

    if plan_type:
        header += f"[Member's Plan: {plan_type}]\n\n"

    formatted = header
    for i, chunk in enumerate(context_chunks, 1):
        formatted += f"--- Document Chunk {i} ---\n{chunk}\n\n"

    formatted += "=" * 60 + "\n"
    formatted += ("INSTRUCTION: Answer ONLY using the context above. "
                   "If the information is not present, say: "
                   "'I do not have enough information to confirm your coverage. "
                   "Please contact Optum support for assistance.'\n")
    formatted += "=" * 60

    return formatted


def validate_response_grounding(response: str, context_chunks: list[str]) -> dict:
    """Basic post-generation grounding check."""
    hallucination_markers = [
        "I think", "I believe", "probably", "might be",
        "generally speaking", "in most cases", "typically"
    ]

    flags = []
    for marker in hallucination_markers:
        if marker.lower() in response.lower():
            flags.append(f"Potential hedging language detected: '{marker}'")

    response_amounts = set(re.findall(r'\$[\d,]+(?:\.\d{2})?', response))
    context_text = " ".join(context_chunks)
    context_amounts = set(re.findall(r'\$[\d,]+(?:\.\d{2})?', context_text))

    ungrounded_amounts = response_amounts - context_amounts
    if ungrounded_amounts:
        flags.append(f"Dollar amounts not found in context: {ungrounded_amounts}")

    return {
        "is_grounded": len(flags) == 0,
        "flags": flags,
        "confidence": max(0, 1.0 - (len(flags) * 0.2))
    }