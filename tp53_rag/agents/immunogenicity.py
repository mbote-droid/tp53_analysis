"""
============================================================
Agent #2: Immunogenicity & TME Predictor
agents/immunogenicity.py
============================================================
Evaluates tumor microenvironment (TME) impact of TP53 mutations.
Predicts immune infiltration status (hot/cold), checkpoint blockade response.
Uses CIBERSORT/ESTIMATE-style immune scoring (mock implementation).
"""

import json
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

import logging
log = logging.getLogger(__name__)


@dataclass
class ImmunePrediction:
    """Structured immune prediction output."""
    mutation: str
    immune_status: str  # immune-hot, immune-cold, intermediate
    predicted_tme_type: str  # inflamed, excluded, desert
    checkpoint_response_likelihood: float  # 0-1
    neoantigen_burden: str  # low, moderate, high
    immune_infiltration_score: float  # ESTIMATE/CIBERSORT-like, 0-1
    t_cell_estimated_fraction: float  # 0-1
    macrophage_m1_score: float  # 0-1
    immune_checkpoint_genes: Dict[str, float]  # {PDL1: 0.75, CTLA4: 0.65, ...}
    synergy_with_icp: List[str]  # ["anti-PD-1", "anti-CTLA4", ...]
    confidence: float  # 0-1
    clinical_recommendation: str


class ImmunogenicityPredictor:
    """Tumor microenvironment & immunotherapy response predictor."""
    
    # TP53 mutation immune impact database
    MUTATION_IMMUNE_PROFILE = {
        "R175H": {
            "immune_status": "immune-hot",
            "neoantigen_burden": "high",
            "checkpoint_response": 0.75,
            "t_cell_fraction": 0.35,
        },
        "R248W": {
            "immune_status": "immune-hot",
            "neoantigen_burden": "high",
            "checkpoint_response": 0.70,
            "t_cell_fraction": 0.32,
        },
        "R273H": {
            "immune_status": "intermediate",
            "neoantigen_burden": "moderate",
            "checkpoint_response": 0.55,
            "t_cell_fraction": 0.25,
        },
        "R249S": {
            "immune_status": "immune-hot",
            "neoantigen_burden": "high",
            "checkpoint_response": 0.68,
            "t_cell_fraction": 0.33,
        },
        "R282W": {
            "immune_status": "immune-cold",
            "neoantigen_burden": "low",
            "checkpoint_response": 0.30,
            "t_cell_fraction": 0.15,
        },
        "G245S": {
            "immune_status": "intermediate",
            "neoantigen_burden": "moderate",
            "checkpoint_response": 0.50,
            "t_cell_fraction": 0.20,
        },
        "Y220C": {
            "immune_status": "immune-cold",
            "neoantigen_burden": "low",
            "checkpoint_response": 0.25,
            "t_cell_fraction": 0.12,
        },
    }
    
    def __init__(self):
        self._lock = threading.Lock()
        self._audit_log = Path("logs/immunogenicity.log")
        self._audit_log.parent.mkdir(parents=True, exist_ok=True)
    
    def predict(
        self,
        mutation: str,
        cancer_type: Optional[str] = None,
        vaf: Optional[float] = None,
    ) -> Dict:
        """
        Predict immune response to TP53 mutation.
        
        Args:
            mutation: Mutation name (e.g., "R175H")
            cancer_type: Optional cancer context (e.g., "colorectal")
            vaf: Optional variant allele frequency
        
        Returns:
            JSON-serializable immune prediction
        """
        # Get mutation profile or use defaults
        profile = self.MUTATION_IMMUNE_PROFILE.get(mutation, {})
        
        if not profile:
            # Default for non-hotspot
            profile = {
                "immune_status": "intermediate",
                "neoantigen_burden": "low",
                "checkpoint_response": 0.40,
                "t_cell_fraction": 0.18,
            }
        
        # Adjust based on VAF (higher VAF = more clonal = more immunogenic)
        checkpoint_response = profile.get("checkpoint_response", 0.5)
        if vaf and vaf > 40:
            checkpoint_response += 0.1  # clonal mutations more immunogenic
        checkpoint_response = min(checkpoint_response, 1.0)
        
        # Immune checkpoint gene expression estimates
        if profile.get("immune_status") == "immune-hot":
            checkpoint_genes = {
                "PD-L1": 0.75,
                "PD-L2": 0.60,
                "CTLA-4": 0.65,
                "LAG-3": 0.55,
                "TIM-3": 0.50,
                "TIGIT": 0.45,
            }
        elif profile.get("immune_status") == "immune-cold":
            checkpoint_genes = {
                "PD-L1": 0.15,
                "PD-L2": 0.10,
                "CTLA-4": 0.20,
                "LAG-3": 0.15,
                "TIM-3": 0.10,
                "TIGIT": 0.08,
            }
        else:  # intermediate
            checkpoint_genes = {
                "PD-L1": 0.45,
                "PD-L2": 0.35,
                "CTLA-4": 0.40,
                "LAG-3": 0.30,
                "TIM-3": 0.25,
                "TIGIT": 0.20,
            }
        
        # Recommended immunotherapy
        icp_synergies = []
        if checkpoint_genes.get("PD-L1", 0) > 0.40:
            icp_synergies.append("anti-PD-1")
            icp_synergies.append("anti-PD-L1")
        if checkpoint_genes.get("CTLA-4", 0) > 0.40:
            icp_synergies.append("anti-CTLA-4")
        if len(icp_synergies) > 0:
            icp_synergies.append("combination")
        
        # Clinical recommendation
        if checkpoint_response > 0.70:
            recommendation = f"High likelihood of response to checkpoint blockade. Consider anti-PD-1/PD-L1 monotherapy or combination with anti-CTLA-4."
        elif checkpoint_response > 0.50:
            recommendation = f"Moderate likelihood. Combination checkpoint blockade recommended. Monitor for resistance mutations."
        else:
            recommendation = f"Low likelihood of standard immunotherapy response. Consider alternative: targeted therapy, PARP inhibitors, or synthetic lethality approaches."
        
        # Build prediction
        prediction = ImmunePrediction(
            mutation=mutation,
            immune_status=profile.get("immune_status", "intermediate"),
            predicted_tme_type="inflamed" if "hot" in profile.get("immune_status", "") else "desert",
            checkpoint_response_likelihood=checkpoint_response,
            neoantigen_burden=profile.get("neoantigen_burden", "moderate"),
            immune_infiltration_score=profile.get("t_cell_fraction", 0.2) + 0.15,
            t_cell_estimated_fraction=profile.get("t_cell_fraction", 0.2),
            macrophage_m1_score=0.35 if "hot" in profile.get("immune_status", "") else 0.10,
            immune_checkpoint_genes=checkpoint_genes,
            synergy_with_icp=icp_synergies,
            confidence=0.85,
            clinical_recommendation=recommendation,
        )
        
        # Audit
        self._audit(f"predicted:{mutation} → {prediction.immune_status}")
        
        return {
            "prediction": asdict(prediction),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": f"{mutation} predicted as {prediction.immune_status} with {checkpoint_response:.1%} checkpoint response likelihood",
        }
    
    def _audit(self, msg: str):
        try:
            with self._lock:
                entry = json.dumps({
                    "ts": datetime.now().isoformat(),
                    "event": msg,
                }) + "\n"
                with open(self._audit_log, "a", encoding="utf-8") as f:
                    f.write(entry)
        except Exception as e:
            log.warning(f"Audit log failed: {e}")


# Global instance
_predictor = ImmunogenicityPredictor()

def predict_immune_response(mutation: str, cancer_type: Optional[str] = None, vaf: Optional[float] = None) -> Dict:
    """Convenience function."""
    return _predictor.predict(mutation, cancer_type=cancer_type, vaf=vaf)
