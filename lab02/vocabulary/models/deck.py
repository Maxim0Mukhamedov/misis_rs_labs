from dataclasses import dataclass

@dataclass
class Deck:
    first_language: str
    second_language: str
    topic: str
    word_count: int
