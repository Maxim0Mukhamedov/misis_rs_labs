import os
import json

_BOT_MESSAGES_PATH = "./static/messages.json"

USERS_DATA_PATH = "./data/u.data"
FILMS_DATA_PATH = "./data/u.item"

TG_BOT_API_TOKEN = ""
BOT_MESSAGES = {}
RATING_FORMAT = r"^\s*(\d+)[,\s]+(\d+)\s*$"
NUM_RECOMMENDATION = 3

def set_config():
    import filtering.config as cfg
    cfg.TG_BOT_API_TOKEN = os.getenv("TG_BOT_API_TOKEN", "")

    try:
        with open(_BOT_MESSAGES_PATH, "r", encoding="utf-8") as f:
            cfg.BOT_MESSAGES = json.load(f)
    except Exception:
        cfg.BOT_MESSAGES = {}
