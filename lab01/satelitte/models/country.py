from dataclasses import dataclass
from typing import Optional


@dataclass
class CountryResponse:
    name: Optional[str] = None
    description: Optional[str] = None
