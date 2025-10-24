import requests

import satelitte.config
from satelitte.models import CountryResponse


def get_ai_country_description(latitude: float, longitude: float, model: str) -> CountryResponse:
    params_country = {
        "lat": latitude,
        "lon": longitude,
        "format": "json",
    }

    response = requests.get(
        satelitte.config.COUNTRY_URL,
        params=params_country,
        headers={"User-Agent": "Mozilla/5.0"},
    ).json()

    if response.get("address"):
        country = response["address"]["country"]
    else:
        return CountryResponse("", "МКС над водой или с пингвинами.")

    params_ai = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0},
        "eval_count": 5,
        "prompt": f"Describe {country}. Without links and format, only plain text.",
    }

    country_description = requests.post(
        satelitte.config.OLLAMA_URL, json=params_ai
    ).json()["response"]

    return CountryResponse(country, country_description)
