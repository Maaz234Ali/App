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
    # Firebase credentials URL from GitHub
    firebase_json_url = "https://raw.githubusercontent.com/Maaz234Ali/App/main/firebase.json"
    
    try:
        # Download the firebase.json file
        response = requests.get(firebase_json_url)
        response.raise_for_status()
        
        # Load the JSON content
        firebase_config = response.json()
        
        # Use the credentials from the downloaded JSON
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred, {'storageBucket': 'login-cb7d4.appspot.com'})
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download firebase.json: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download firebase.json: {str(e)}")

# Tesseract configuration
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# OpenAI API Key
openai.api_key = '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'  # Replace with your OpenAI API key

class ReportRequest(BaseModel):
    file_paths: List[str]

def generate_signed_url(file_path: str) -> str:
    """Generate a signed URL for a Firebase Storage file."""
    bucket = storage.bucket()
    blob = bucket.blob(file_path)
    signed_url = blob.generate_signed_url(timedelta(minutes=15))
    return signed_url

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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Summarize medical reports."},
                      {"role": "user", "content": text}],
            max_tokens=200
        )
        return response.choices[0].message["content"].strip()
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
