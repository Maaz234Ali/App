import os
import json
import logging
import uvicorn
import nest_asyncio
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
import pytesseract
import io
import requests
import fitz  # PyMuPDF for handling PDFs
from firebase_admin import credentials, storage
import firebase_admin
import openai

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize nest_asyncio
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI()

# Firebase Admin SDK initialization
if not firebase_admin._apps:
    firebase_json = os.getenv("FIREBASE_CREDENTIALS")
    if not firebase_json:
        raise HTTPException(status_code=500, detail="FIREBASE_CREDENTIALS not set in environment variables")

    firebase_config = json.loads(firebase_json)
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred, {'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET", "")})

# Tesseract configuration
pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY", "")
if not openai.api_key:
    raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment variables")

class ReportRequest(BaseModel):
    file_paths: List[str]

def generate_signed_url(file_path: str) -> str:
    """Generate a signed URL for a Firebase Storage file."""
    try:
        bucket = storage.bucket()
        blob = bucket.blob(file_path)
        return blob.generate_signed_url(timedelta(minutes=15))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating signed URL: {str(e)}")

def download_file_from_firebase(firebase_path: str) -> bytes:
    """Download a file from Firebase Storage."""
    try:
        signed_url = generate_signed_url(firebase_path)
        response = requests.get(signed_url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF, using OCR if needed."""
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_text = ""
    for page in pdf_document:
        extracted_text += page.get_text("text") + "\n"
        if not extracted_text.strip():
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            extracted_text += pytesseract.image_to_string(img, lang="eng") + "\n"
    return extracted_text.strip()

def summarize_text(text: str) -> str:
    """Summarize the extracted text using OpenAI with a structured prompt."""
    if not text.strip():
        return "No text provided for summarization."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert medical report analyst."},
                {"role": "user", "content": f"""
                    Medical Report:
                    {text}
                    
                    Tasks:
                    1. Extract all patient details, test names, results, and reference ranges.
                    2. Include all dates found in the report.
                    3. Preserve original data format, ensuring nothing is left out.
                    4. Summarize findings in a structured JSON format.

                    Output format (structured JSON):
                    {{
                        "Patient Details": {{...}},
                        "Test Results": [...],
                        "Dates": [...],
                        "Summary": "...",
                        "Analysis": "..."
                    }}
                """}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"].strip()
    except openai.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in summarizing text: {str(e)}")

@app.post("/summarize_reports")
async def summarize_reports(request: ReportRequest):
    summaries = []
    for path in request.file_paths:
        file_bytes = download_file_from_firebase(path)
        text = extract_text_from_pdf(file_bytes) if path.endswith(".pdf") else ""
        logging.info(f"Extracted text from {path}: {text[:500]}...")
        summaries.append(summarize_text(text))
    return {"summary": "\n\n".join(summaries)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
