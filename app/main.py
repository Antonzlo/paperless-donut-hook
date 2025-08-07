from fastapi import FastAPI, Request, Header, HTTPException
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from pydantic import BaseModel
import os, io, json, logging
from datetime import datetime
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Pydantic models for validation
class WebhookData(BaseModel):
    id: int

class DocumentUpdate(BaseModel):
    title: str
    correspondent: str = None
    document_type: str = "Receipt"
    date: str = None

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
        logger.warning("Unauthorized webhook attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")

    data = await request.json()
    # Validate webhook data
    try:
        webhook_data = WebhookData(**data)
        doc_id = webhook_data.id
    except Exception as e:
        logger.error(f"Invalid webhook data: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook data")

    doc_url = f"{BASE_URL}/api/documents/{doc_id}/download/"
    async with httpx.AsyncClient() as client:
        r = await client.get(doc_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        image = Image.open(io.BytesIO(r.content)).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values
    decoder_input_ids = processor.tokenizer("<s_rvlcdip>", add_special_tokens=False, return_tensors="pt").input_ids
    output = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
    result = processor.batch_decode(output, skip_special_tokens=True)[0]
    logger.info(f"Donut output for doc {doc_id}: {result[:100]}...")

    # Save raw JSON output
    try:
        raw_json_path = f"/data/raw_donut_doc_{doc_id}.json"
        with open(raw_json_path, "w") as f:
            f.write(result)
        logger.info(f"Saved raw output to {raw_json_path}")
    except Exception as e:
        logger.error(f"Failed to write raw JSON for doc {doc_id}: {e}")

    try:
        result = result.replace("'", "\"")
        parsed = json.loads(result)
        invoice = parsed.get("invoice", {})
        payload = {
            "title": invoice.get("vendor", "Receipt"),
            "correspondent": invoice.get("vendor"),
            "document_type": "Receipt",
            "date": invoice.get("date")
        }

        patch_url = f"{BASE_URL}/api/documents/{doc_id}/"
        async with httpx.AsyncClient() as client:
            patch = await client.patch(patch_url, headers=HEADERS, json=payload, timeout=30)
        logger.info(f"Updated doc {doc_id}: {patch.status_code}")
        return {"status": "success"}

    except Exception as e:
        logger.error(f"Failed to parse/update doc {doc_id}: {e}")
        return {"status": "error", "detail": str(e)}
