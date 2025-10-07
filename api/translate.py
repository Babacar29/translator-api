import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Charger les variables depuis le fichier .env
load_dotenv()

app = FastAPI()

HF_API_URL = os.getenv("HF_API_URL")
HF_API_KEY = os.getenv("HF_API_KEY")

@app.post("/api/translate")
async def translate(request: Request):
    body = await request.json()
    text = body.get("text", "")
    target_lang = body.get("target_lang", "wol_Latn")

    if not text:
        return JSONResponse({"error": "Le champ 'text' est requis"}, status_code=400)

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": text,
        "parameters": {
            # Exemple : wol_Latn, fra_Latn, eng_Latn
            "forced_bos_token_id": target_lang
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(HF_API_URL, headers=headers, json=payload)
        data = response.json()

    if isinstance(data, list) and "generated_text" in data[0]:
        translation = data[0]["generated_text"]
        return JSONResponse({"translated": translation})
    else:
        return JSONResponse({"error": data}, status_code=500)
