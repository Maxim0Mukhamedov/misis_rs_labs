import requests

import satelitte.config
from satelitte.models import PositionResponse


def get_position() -> PositionResponse:
    satelite_position = requests.get(satelitte.config.ISS_URL).json()

    iss_position = satelite_position["iss_position"]
    longitude = float(iss_position["longitude"])
    latitude = float(iss_position["latitude"])

    return PositionResponse(latitude, longitude)
