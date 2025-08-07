from fastapi import FastAPI, Request, Header
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import os, requests, io, json
from datetime import datetime

app = FastAPI()

MODEL = "naver-clova-ix/donut-base-finetuned-rvlcdip"
processor = DonutProcessor.from_pretrained(MODEL)
model = VisionEncoderDecoderModel.from_pretrained(MODEL)

PAPERLESS_TOKEN = os.environ["PAPERLESS_TOKEN"]
PAPERLESS_SECRET = os.environ["PAPERLESS_SECRET"]
HEADERS = {"Authorization": f"Token {PAPERLESS_TOKEN}"}
BASE_URL = "http://paperless:8000"

@app.post("/webhook")
async def process_webhook(request: Request, x_hook_secret: str = Header(None)):
    if x_hook_secret != PAPERLESS_SECRET:
        return {"error": "unauthorized"}

    data = await request.json()
    doc_id = data.get("id")
    if not doc_id:
        return {"error": "missing doc ID"}

    doc_url = f"{BASE_URL}/api/documents/{doc_id}/download/"
    r = requests.get(doc_url, headers=HEADERS)
    image = Image.open(io.BytesIO(r.content)).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values
    decoder_input_ids = processor.tokenizer("<s_rvlcdip>", add_special_tokens=False, return_tensors="pt").input_ids
    output = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
    result = processor.batch_decode(output, skip_special_tokens=True)[0]
    print("Donut output:", result)

    # Save raw JSON output
    try:
        raw_json_path = f"/data/raw_donut_doc_{doc_id}.json"
        with open(raw_json_path, "w") as f:
            f.write(result)
    except Exception as e:
        print("Failed to write raw JSON:", e)

    try:
        result = result.replace("'", """)
        parsed = json.loads(result)
        invoice = parsed.get("invoice", {})
        payload = {
            "title": invoice.get("vendor", "Receipt"),
            "correspondent": invoice.get("vendor"),
            "document_type": "Receipt",
            "date": invoice.get("date")
        }

        patch_url = f"{BASE_URL}/api/documents/{doc_id}/"
        patch = requests.patch(patch_url, headers=HEADERS, json=payload)
        print(f"Updated doc {doc_id}: {patch.status_code}")
        return {"status": "success"}

    except Exception as e:
        print("Failed to parse/update:", e)
        return {"status": "error", "detail": str(e)}
