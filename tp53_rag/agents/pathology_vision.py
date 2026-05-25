"""
Agent #15 — Pathology Vision Agent
Processes H&E slide images using UNI foundation model
Correlates findings with TP53 mutation data

CLOUD FIX: torch and timm are imported lazily inside _load_model()
only when actually needed — not at module level.
This prevents Streamlit Cloud from crashing on startup when torch
is not installed.
"""
from pathlib import Path
from typing import Dict, Any, Optional
from utils.logger import log

# Tissue classification labels
TISSUE_CLASSES = [
    "Tumor", "Stroma", "Inflammatory", "Necrosis",
    "Normal epithelium", "Mucus", "Smooth muscle", "Adipose"
]

# TP53 mutation → tissue correlation (from published literature)
MUTATION_TISSUE_CORRELATION = {
    "R248W": {"Colorectal": 0.90, "Lung": 0.75},
    "R273H": {"Colorectal": 0.88, "Bladder": 0.72},
    "R175H": {"Breast": 0.85, "Ovarian": 0.75},
    "R249S": {"Liver": 0.95, "Esophageal": 0.65},
    "G245S": {"Sarcoma": 0.80, "Brain": 0.55},
}


class PathologyVisionAgent:
    """
    Agent #15 — Pathology slide analysis.
    Uses UNI foundation model for tissue embedding.
    Falls back to ResNet if UNI unavailable.
    Falls back gracefully (model=None) if torch not installed.
    """

    _cached_model = None  # module-level cache

    def __init__(self, rag_chain=None):
        self.rag_chain = rag_chain
        self.device = "cpu"
        self.model = None

        if PathologyVisionAgent._cached_model is None:
            self._load_model()
            PathologyVisionAgent._cached_model = self.model
        else:
            self.model = PathologyVisionAgent._cached_model
            log.info("Pathology model loaded from cache")

    def _load_model(self):
        """
        Load UNI or ResNet fallback.
        torch and timm imported here — lazy, not at module level.
        If torch not installed (Streamlit Cloud), self.model stays None
        and process_slide() returns a graceful error.
        """
        try:
            # Lazy import — only runs when model is actually needed
            import timm
            # UNI model — best option
            self.model = timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True
            )
            self.model.eval()
            log.info("UNI pathology model loaded")
        except Exception as e:
            log.warning(f"UNI not available: {e}. Using ResNet fallback.")
            try:
                import timm
                self.model = timm.create_model(
                    "resnet50", pretrained=True, num_classes=len(TISSUE_CLASSES)
                )
                self.model.eval()
                log.info("ResNet fallback loaded")
            except ImportError:
                log.warning("timm/torch not installed — pathology agent disabled on this deployment")
                self.model = None
            except Exception as e2:
                log.error(f"No model available: {e2}")
                self.model = None

    def process_slide(
        self,
        image_path: str,
        mutation_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Process a pathology slide image.
        Returns graceful error if model not available (cloud deployment).
        """
        if not self.model:
            return {
                "error": "Pathology model unavailable in this environment. Run locally for full functionality.",
                "success": False
            }

        try:
            # Lazy imports inside method — safe for cloud
            import torch
            import torchvision.transforms as transforms
            from PIL import Image

            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image = image.resize((224, 224))

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            tensor = transform(image).unsqueeze(0)

            # Extract features
            with torch.no_grad():
                features = self.model(tensor)

            # Classify tissue
            probs = torch.softmax(features, dim=1).squeeze().tolist()
            if len(probs) == len(TISSUE_CLASSES):
                classifications = sorted(
                    zip(TISSUE_CLASSES, probs),
                    key=lambda x: x[1], reverse=True
                )
            else:
                classifications = [("Tissue features extracted", 1.0)]

            # Correlate with TP53 mutations
            correlations = []
            if mutation_data:
                mutations = mutation_data.get("mutations", [])
                for mut in mutations[:3]:
                    aa_change = mut.get("amino_acid_change", "")
                    if aa_change in MUTATION_TISSUE_CORRELATION:
                        correlations.append({
                            "mutation": aa_change,
                            "cancer_correlations": MUTATION_TISSUE_CORRELATION[aa_change]
                        })

            # Get Gemma 4 narration
            narration = self._get_narration(classifications, correlations, mutation_data)

            return {
                "success": True,
                "tissue_classifications": [
                    {"tissue": t, "probability": round(p, 3)}
                    for t, p in classifications[:5]
                ],
                "top_tissue": classifications[0][0] if classifications else "Unknown",
                "mutation_correlations": correlations,
                "llm_narration": narration,
                "image_processed": str(image_path),
            }

        except ImportError:
            return {
                "error": "torch/torchvision not installed — run locally for pathology analysis",
                "success": False
            }
        except Exception as e:
            log.error(f"Slide processing failed: {e}")
            return {"error": str(e), "success": False}

    def _get_narration(self, classifications, correlations, mutation_data):
        """Get Gemma 4 clinical narration of findings."""
        if not self.rag_chain:
            top = classifications[0][0] if classifications else "unknown tissue"
            return (
                f"Pathology analysis identified predominant {top} tissue. "
                f"{'Correlates with detected TP53 mutations.' if correlations else ''}"
            )
        try:
            top_tissues = ", ".join([c[0] for c in classifications[:3]])
            mut_summary = ", ".join([c["mutation"] for c in correlations])
            question = (
                f"A pathology slide shows: {top_tissues}. "
                f"The patient has TP53 mutations: {mut_summary}. "
                f"Provide clinical interpretation of these combined findings. "
                f"What malignancy is most likely? What are the implications?"
            )
            result = self.rag_chain.query(
                question=question,
                agent_type="clinical_interpretation"
            )
            return result["answer"]
        except Exception as e:
            log.warning(f"Narration failed: {e}")
            return "Narration unavailable — RAG offline."