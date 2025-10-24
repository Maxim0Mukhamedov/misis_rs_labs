import os
import json


OLLAMA_URL = ""

PROMPT = ""

TG_BOT_API_TOKEN = ""

BOT_MESSAGES = {}


def set_config():
    import vocabulary.config

    vocabulary.config.OLLAMA_URL = os.getenv("OLLAMA_URL", "")
    vocabulary.config.TG_BOT_API_TOKEN = os.getenv("TG_BOT_API_TOKEN", "")
    with open('./static/prompt.json', 'r') as f:
        vocabulary.config.PROMPT = json.load(f)["prompt"]
    with open('./static/messages.json', 'r') as f:
        vocabulary.config.BOT_MESSAGES = json.load(f)
