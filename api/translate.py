from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# CORS for browser requests (dev-friendly: allow all; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # includes OPTIONS preflight
    allow_headers=["*"]
)

# Charger le modèle depuis Hugging Face
model_name = "bilalfaye/nllb-200-distilled-600M-wo-fr-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.post("/translate")
async def translate(req: TranslationRequest):
    inputs = tokenizer(req.text, return_tensors="pt")
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(req.target_lang)
    )
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return {"translated": translated_text}
