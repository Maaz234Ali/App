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
    """Generate a signed URL to access the file from Firebase Storage."""
    bucket = storage.bucket()
    blob = bucket.blob(file_path)
    return blob.generate_signed_url(timedelta(minutes=15))

def download_file_from_firebase(firebase_path: str) -> bytes:
    """Download the file from Firebase Storage using a signed URL."""
    signed_url = generate_signed_url(firebase_path)
    response = requests.get(signed_url)
    if response.status_code == 200:
        return response.content
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to download file from Firebase.")

def extract_text_from_file(file_bytes: bytes, file_extension: str) -> str:
    """Extract text from PDFs and image files using OCR."""
    extracted_text = ""

    if file_extension == '.pdf':
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        for page in pdf_document:
            page_text = page.get_text("text")
            extracted_text += page_text + "\n"
            
            # If no text is found, perform OCR on the page image
            if not page_text.strip():
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                extracted_text += pytesseract.image_to_string(img, lang="eng") + "\n"

    elif file_extension in ['.png', '.jpg', '.jpeg']:
        img = Image.open(io.BytesIO(file_bytes))
        extracted_text = pytesseract.image_to_string(img, lang="eng")
    
    return extracted_text.strip()

def summarize_text(text: str) -> str:
    """Generate a text-based summary of the extracted text using OpenAI GPT-4."""
    if not text.strip():
        return "No text provided for summarization."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert medical report analyst. Provide a concise, well-structured text summary of the given medical report."},
                {"role": "user", "content": f"""
                    Medical Report:
                    {text}

                    Tasks:
                    1. Summarize the report concisely in text format.
                    2. Highlight key patient details, important test results, and any significant findings.
                    3. Use bullet points for readability.
                    4. Analyse the full report and give advice to the patient on thier report result

               
                """}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message['content'].strip()

    except Exception as e:
        return f"Error in summarizing text: {str(e)}"

@app.post("/summarize_reports")
async def summarize_reports(report_request: ReportRequest):
    """Process PDF and image reports, extract text, and generate summaries."""
    summaries = []

    for file_path in report_request.file_paths:
        file_bytes = download_file_from_firebase(file_path)
        
        # Get file extension
        file_extension = file_path.lower().split('.')[-1]
        file_extension = f".{file_extension}"  # Convert to ".pdf", ".png", etc.

        # Extract text based on file type
        extracted_text = extract_text_from_file(file_bytes, file_extension)
        
        logging.info(f"Extracted text from {file_path}: {extracted_text[:500]}...")

        # Summarize extracted text
        summary = summarize_text(extracted_text)
        summaries.append(summary)

    return {"summary": "\n\n".join(summaries)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
