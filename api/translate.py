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
        # Try to parse JSON from the upstream response. If that fails,
        # fall back to the raw text so we can return a useful error payload
        try:
            data = response.json()
        except Exception:
            # response.content may be bytes; use .text for a string fallback
            data = response.text

    # If the upstream returned a successful JSON list with generated_text, use it
    if response.status_code == 200 and isinstance(data, list) and "generated_text" in data[0]:
        translation = data[0]["generated_text"]
        return JSONResponse({"translated": translation})

    # Otherwise return a helpful error payload including upstream status and body
    status = response.status_code if isinstance(response.status_code, int) else 500
    return JSONResponse({"error": data, "upstream_status": status}, status_code=max(status, 500))
