import os

OLLAMA_URL = ""

ISS_URL = ""

COUNTRY_URL = ""


def set_config():
    import satelitte.config

    satelitte.config.OLLAMA_URL = os.getenv("OLLAMA_URL", "")
    satelitte.config.ISS_URL = os.getenv("ISS_URL", "")
    satelitte.config.COUNTRY_URL = os.getenv("COUNTRY_URL", "")
