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
        signed_url = blob.generate_signed_url(timedelta(minutes=15))
        return signed_url
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


def convert_pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """Convert PDF pages to images."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        return [Image.open(io.BytesIO(page.get_pixmap().tobytes("png"))) for page in pdf_document]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting PDF to images: {str(e)}")


def extract_text_from_images(images: List[Image.Image]) -> str:
    """Extract text from images using OCR."""
    return "\n".join(pytesseract.image_to_string(img) for img in images)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF by converting it to images."""
    images = convert_pdf_to_images(pdf_bytes)
    return extract_text_from_images(images)


def extract_text_from_image(image: Image.Image) -> str:
    """Extract text from a single image."""
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in OCR: {str(e)}")


def summarize_text(text: str) -> str:
    """Summarize the extracted text using OpenAI."""
    if not text.strip():
        return "No text provided for summarization."
    
    try:
        response = openai.ChatCompletion.create(  # Updated method
            model="gpt-3.5-turbo",  # Specify the correct model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following medical report:\n{text}"}
            ],
            max_tokens=200
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in summarizing text: {str(e)}")


@app.post("/summarize_reports")
async def summarize_reports(request: ReportRequest):
    summaries = []
    for path in request.file_paths:
        file_bytes = download_file_from_firebase(path)
        if path.endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
        else:
            image = Image.open(io.BytesIO(file_bytes))
            text = extract_text_from_image(image)
        summaries.append(summarize_text(text))
    return {"summary": "\n\n".join(summaries)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
