"""
============================================================
PII Scrubber — HIPAA-Compliant Data Sanitization
utils/pii_scrubber.py
============================================================
Regex-based PII detection + SHA-256 hashing.
Preserves data traceability while removing sensitive info.
"""
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ScrubStats:
    """Audit trail for PII scrubbing."""
    items_found: int = 0
    items_hashed: int = 0
    patterns_matched: List[str] = field(default_factory=list)


class PIIScrubber:
    """HIPAA-compliant PII redaction engine."""
    
    # Regex patterns (conservative — avoid false positives)
    PATTERNS = {
        "nhs_id": r"\b(?:[A-Z]{2})\s?(?:\d{2})\s?(?:\d{2})\s?(?:\d{2})\s?(?:\d{2})\b",
        "patient_id": r"\b(?:PT|PAT|MRN)[_-]?(?:\d{6,12})\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.\s]?)?\(?(?:\d{3})\)?[-.\s]?(?:\d{3})[-.\s]?(?:\d{4})\b",
        "ipv4": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "ssn": r"\b(?:\d{3}[-]?\d{2}[-]?\d{4})\b",
        "date_of_birth": r"\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])[-/](?:\d{4}|\d{2})\b",
        "person_name": r"\b(?:[A-Z][a-z]+\s+)+[A-Z][a-z]+\b",  # Capitalized names (conservative)
    }
    
    def __init__(self, hash_salt: str = "tp53_secure_salt_2026"):
        self.hash_salt = hash_salt
        self.stats = ScrubStats()
    
    def hash_value(self, value: str) -> str:
        """SHA-256 hash with salt (traceable but anonymized)."""
        combined = f"{value}:{self.hash_salt}".encode('utf-8')
        return hashlib.sha256(combined).hexdigest()[:16]
    
    def scrub_text(self, text: str, mode: str = "hash") -> Tuple[str, ScrubStats]:
        """
        Scrub PII from text.
        
        Args:
            text: Raw text possibly containing PII
            mode: "hash" (replace with hash), "redact" (replace with [REDACTED]), "remove" (delete)
        
        Returns:
            (scrubbed_text, audit_stats)
        """
        if not isinstance(text, str):
            return text, self.stats
        
        scrubbed = text
        local_stats = ScrubStats()
        
        for pattern_name, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, scrubbed, re.IGNORECASE)
            for match in matches:
                original = match.group()
                
                if mode == "hash":
                    replacement = f"[{self.hash_value(original)}]"
                elif mode == "redact":
                    replacement = f"[REDACTED_{pattern_name.upper()}]"
                else:  # remove
                    replacement = ""
                
                scrubbed = scrubbed.replace(original, replacement, 1)
                local_stats.items_found += 1
                local_stats.items_hashed += 1
                
                if pattern_name not in local_stats.patterns_matched:
                    local_stats.patterns_matched.append(pattern_name)
        
        self.stats = local_stats
        return scrubbed, local_stats
    
    def scrub_dict(self, data: Dict, skip_keys: List[str] = None) -> Tuple[Dict, ScrubStats]:
        """Recursively scrub dict values (e.g., JSON from LLM)."""
        if skip_keys is None:
            skip_keys = ["agent", "type", "confidence", "score", "status"]
        
        scrubbed = {}
        for key, value in data.items():
            if key in skip_keys:
                scrubbed[key] = value
            elif isinstance(value, str):
                scrubbed[key], _ = self.scrub_text(value, mode="hash")
            elif isinstance(value, dict):
                scrubbed[key], _ = self.scrub_dict(value, skip_keys)
            elif isinstance(value, list):
                scrubbed[key] = [
                    self.scrub_text(v, mode="hash")[0] if isinstance(v, str) else v
                    for v in value
                ]
            else:
                scrubbed[key] = value
        
        return scrubbed, self.stats
    
    def scrub_json_string(self, json_str: str) -> Tuple[str, ScrubStats]:
        """Parse JSON, scrub values, return JSON string."""
        try:
            data = json.loads(json_str)
            scrubbed, stats = self.scrub_dict(data)
            return json.dumps(scrubbed, indent=2), stats
        except json.JSONDecodeError:
            # Fallback to text scrubbing if not valid JSON
            return self.scrub_text(json_str, mode="hash")
    
    def reset_stats(self):
        """Reset audit counters."""
        self.stats = ScrubStats()


# Global instance
_scrubber = PIIScrubber()

def scrub(text: str, mode: str = "hash") -> str:
    """Convenience function for one-off scrubbing."""
    scrubbed, _ = _scrubber.scrub_text(text, mode=mode)
    return scrubbed

def scrub_dict(data: Dict, skip_keys: List[str] = None) -> Dict:
    """Convenience function for dict scrubbing."""
    scrubbed, _ = _scrubber.scrub_dict(data, skip_keys=skip_keys)
    return scrubbed
