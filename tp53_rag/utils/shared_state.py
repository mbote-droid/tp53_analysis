"""
Shared state manager for multi-agent communication.
Replaces text-passing between agents with direct state sharing.
"""
import threading
from typing import Any, Dict, Optional
from datetime import datetime

class SharedAgentState:
    """
    Thread-safe shared state for all 14 agents.
    Agents read/write directly — no text serialization.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "session_id":      None,
            "accession":       None,
            "sequence":        None,
            "mutations":       [],
            "orfs":            [],
            "domains":         [],
            "phylo_tree":      None,
            "embeddings":      {},   # agent_name → embedding vector
            "agent_outputs":   {},   # agent_name → structured result
            "pipeline_data":   {},
            "timestamp":       None,
        }

    def update(self, key: str, value: Any):
        with self._lock:
            self._state[key] = value
            self._state["timestamp"] = datetime.now().isoformat()

    def get(self, key: str, default=None) -> Any:
        with self._lock:
            return self._state.get(key, default)

    def update_agent_output(self, agent_name: str, output: Dict):
        with self._lock:
            self._state["agent_outputs"][agent_name] = output

    def get_all_outputs(self) -> Dict:
        with self._lock:
            return dict(self._state["agent_outputs"])

    def reset(self):
        with self._lock:
            self._state["agent_outputs"] = {}
            self._state["embeddings"] = {}

# Global singleton
shared_state = SharedAgentState()
