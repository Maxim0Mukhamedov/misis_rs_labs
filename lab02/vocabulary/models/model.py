from dataclasses import dataclass

@dataclass
class Model:
    model_name: str
    model_temerature: float

    def to_ollama(self):
        return {
            "model": self.model_name,
            "options": {"temperature": self.model_temerature},
        }
