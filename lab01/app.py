from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import satelitte.ai_country_description
import satelitte.config
import satelitte.position

satelitte.config.set_config()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
def read_index(request: Request, response_class=HTMLResponse):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/coo")
def get_coordinates() -> dict:
    response = satelitte.position.get_position()
    return {"latitude": response.latitude, "longitude": response.longitude}


@app.get("/des")
def get_description(latitude: float, longitude: float, model: str) -> dict:
    response = satelitte.ai_country_description.get_ai_country_description(latitude, longitude, model)
    return {"name": response.name, "description": response.description}
