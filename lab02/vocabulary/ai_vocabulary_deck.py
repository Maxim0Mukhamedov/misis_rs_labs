import aiohttp
from dataclasses import asdict

import vocabulary.config
from vocabulary.models import Deck, Model


async def get_ai_vocabulary_deck(deck: Deck, model: Model):
    ollama_request_params = {
        "stream": False,
    }
    ollama_request_params.update(model.to_ollama())
    ollama_request_params.update({"prompt": vocabulary.config.PROMPT.format(**asdict(deck))})

    async with aiohttp.ClientSession() as session:
        async with session.post(vocabulary.config.OLLAMA_URL, json=ollama_request_params) as response:
            response_json = await response.json()

            return response_json["response"]
