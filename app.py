import nest_asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import pytesseract
from PIL import Image
import io
import requests
import fitz  # PyMuPDF for handling PDFs
from firebase_admin import credentials, storage
import firebase_admin
from datetime import timedelta
import logging
import os
from typing import List

# Apply nest_asyncio to handle asynchronous tasks
nest_asyncio.apply()

# Initialize FastAPI
app = FastAPI()

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:\Users\Lenovo\Desktop\Heroku\login-cb7d4-firebase-adminsdk-apvga-d9cac60178.json")
    firebase_admin.initialize_app(cred, {'storageBucket': 'login-cb7d4.appspot.com'})

# Path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Set your OpenAI API key from an environment variable for security
openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure OPENAI_API_KEY is set in your environment

class ReportRequest(BaseModel):
    file_paths: List[str]

def generate_signed_url(file_path: str) -> str:
    """Generate a signed URL for a file in Firebase Storage."""
    logging.info(f"Generating signed URL for file path: {file_path}")
    bucket = storage.bucket()
    blob = bucket.blob(file_path)
    signed_url = blob.generate_signed_url(timedelta(minutes=15))  # Increased expiration time to 15 minutes
    logging.info(f"Generated signed URL: {signed_url}")
    return signed_url

def download_file_from_firebase(firebase_path: str) -> bytes:
    """Download file content from Firebase Storage."""
    signed_url = generate_signed_url(firebase_path)
    response = requests.get(signed_url)

    if response.status_code == 200:
        return response.content
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to download file from Firebase.")

def convert_pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """Convert PDF pages to images using PyMuPDF."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()  # Convert each page to an image
            img = Image.open(io.BytesIO(pix.tobytes("png")))  # Convert to PIL Image
            images.append(img)

        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting PDF to images: {str(e)}")

def extract_text_from_images(images: List[Image.Image]) -> str:
    """Extract text from a list of images using OCR."""
    try:
        extracted_text = ""
        for image in images:
            extracted_text += pytesseract.image_to_string(image) + "\n"
        return extracted_text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in extracting text from images: {str(e)}")

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Convert PDF pages to images and perform OCR."""
    images = convert_pdf_to_images(pdf_bytes)  # Convert PDF to images
    text = extract_text_from_images(images)  # Extract text from images
    return text

def extract_text_from_image(image: Image.Image) -> str:
    """Extract text from an image using OCR."""
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error in extracting text from image: {str(e)}"

def summarize_text(text: str) -> str:
    """Summarize the extracted text using OpenAI."""
    if not text.strip():
        return "No text provided for summarization."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Analyzing Medical Reports."},
                      {"role": "user", "content": f"Summarize all the data in the reports. Analyze the reports and provide an overall summary. Determine if the patient is normal or not based on the cumulative information:\n\n{text}"}],
            max_tokens=200
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error in summarizing text: {str(e)}"

logging.basicConfig(level=logging.INFO)

@app.post("/summarize_reports")
async def summarize_reports(report_request: ReportRequest):
    summaries = []

    for file_path in report_request.file_paths:
        # Download the file from Firebase
        file_bytes = download_file_from_firebase(file_path)

        # Check file type and extract text
        if file_path.endswith('.pdf'):
            text = extract_text_from_pdf(file_bytes)  # Now this extracts text by converting the PDF to images
        else:
            # Assuming image if not a PDF
            image = Image.open(io.BytesIO(file_bytes))
            text = extract_text_from_image(image)

        # Summarize the extracted text
        summary = summarize_text(text)
        summaries.append(summary)

    # Combine summaries
    combined_summary = "\n\n".join(summaries)

    # Print to console for debugging
    print(f"Generated summary: {combined_summary}")

    return {"summary": combined_summary}

if __name__ == "__main__":
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
