import os
import fitz  # PyMuPDF
from PIL import Image
import io
import logging
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import base64
import itertools
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def pdf_to_images(pdf_path, dpi=300):
    """
    Convert PDF pages to high-resolution images, skipping first 2 pages.

    Args:
        pdf_path (str): Path to the PDF file
        dpi (int): Resolution for the output images

    Returns:
        list: List of PIL Image objects
    """
    try:
        doc = fitz.open(pdf_path)
        images = []

        # Skip first 2 pages (index 0 and 1)
        start_page = 2

        # Process remaining pages
        for page_num in range(start_page, len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)

        return images
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise

def call_gemini_api(image):
    """
    Call the Gemini API to extract text from an image.

    Args:
        image (PIL.Image): Image to process

    Returns:
        str: Extracted text
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    
    # Convert PIL Image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        "contents": [{
            "parts": [
                {
                    "text": """Extract only the handwritten questions and their answers from this image. Format the output as follows:
1. Start each question with 'Question X' where X is the question number
2. For each answer option, start with the appropriate marker (A), B), Ⅲ, Ⅳ, etc.)
3. Preserve all line breaks and indentation
4. Keep the exact text as written, including any special characters or symbols
5. Do not add any additional text or formatting
6. Skip any printed text or non-handwritten content"""
                },
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_str
                    }
                }
            ]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            content = result['candidates'][0]['content']
            if 'parts' in content and len(content['parts']) > 0:
                return content['parts'][0]['text']
        
        raise ValueError("Invalid response format from Gemini API")
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"API Response: {e.response.text}")
        raise

def format_qa_text(text):
    """
    Format the extracted text to ensure proper question-answer structure.
    
    Args:
        text (str): Raw extracted text
        
    Returns:
        str: Formatted question-answer text
    """
    # First, clean up the text by removing any "no handwritten content" messages
    text = '\n'.join([line for line in text.split('\n') 
                     if not any(msg in line.lower() for msg in 
                              ['no handwritten', 'unable to find', 'sorry', 'no content'])])

    lines = text.split('\n')
    formatted_lines = []
    current_question = None
    current_answer = []
    last_label = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts a new question
        if 'Question' in line or line.startswith('Q'):
            # If we have a previous question, add it to formatted lines
            if current_question:
                formatted_lines.append('')  # Add blank line between questions
                formatted_lines.append(current_question)
                formatted_lines.extend(current_answer)
                formatted_lines.append('')  # Add another blank line for spacing
            
            # Start new question
            current_question = line
            current_answer = []
            last_label = None
        # Check if line is an answer option
        elif any(line.startswith(prefix) for prefix in ['A)', 'B)', 'Ⅲ', 'Ⅳ', 'Ⅱ', 'Ⅰ', ')', '①', '②', '③', '④']):
            if current_answer:  # Add blank line before new answer option
                formatted_lines.append('')
            
            # Handle continuation of subquestions across pages
            if last_label and not any(line.startswith(prefix) for prefix in ['Question', 'Q']):
                # If we have a last label and this line doesn't start with a question number,
                # it's likely a continuation of the previous question
                if last_label.endswith('A') and line.startswith('B'):
                    # If last was A and this is B, it's a continuation
                    current_question = f"Question {current_question.split()[-1]}" if 'Question' in current_question else current_question
                elif last_label.endswith('B') and line.startswith('C'):
                    # If last was B and this is C, it's a continuation
                    current_question = f"Question {current_question.split()[-1]}" if 'Question' in current_question else current_question
                # Add more conditions for other letter sequences if needed
            
            current_answer.append(line)
            last_label = line[0]  # Store the label for next iteration
        # Regular answer text
        else:
            if not current_question and current_answer:
                # If we have an answer but no question, append to previous question
                formatted_lines.extend(current_answer)
                current_answer = []
            current_answer.append(line)
    
    # Add the last question
    if current_question:
        formatted_lines.append('')
        formatted_lines.append(current_question)
        formatted_lines.extend(current_answer)
    
    # Join lines and clean up multiple blank lines
    formatted_text = '\n'.join(formatted_lines)
    formatted_text = '\n'.join(line for line, _ in itertools.groupby(formatted_text.split('\n')))
    
    return formatted_text

def format_text_with_gemini(text):
    """
    Use Gemini API to format the extracted text into a structured JSON format.
    
    Args:
        text (str): Raw extracted text
        
    Returns:
        dict: Structured JSON format of the text
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    
    headers = {
        'Content-Type': 'application/json'
    }

    prompt = """Extract questions and their answers from the following text and format them into a JSON array. 
    For each question and answer pair:
    1. Print the question number (Q1, Q2, etc.)
    2. Print each subquestion with its label (I A, I B, etc.)
    3. Print the question text
    4. Print the extracted answer
    
    Format the output as follows:
    Q1
    I A. Three meters of wire _______ the core. (surround / surrounds)
    Answer: surrounds

    I B. The Prime Minister, together with his wife, _______ the press cordially. (greet / greets)
    Answer: greets

    Then convert this into a JSON array with the following structure:
    [
      {
        "question": "Q1",
        "subquestions": [
          {
            "label": "I A",
            "question": "Three meters of wire _______ the core. (surround / surrounds)",
            "content": "surrounds"
          },
          {
            "label": "I B",
            "question": "The Prime Minister, together with his wife, _______ the press cordially. (greet / greets)",
            "content": "greets"
          }
        ]
      }
    ]
    
    Important formatting rules:
    1. Use Roman numerals (I, II, III, IV) followed by letters (A, B, C) for labels
    2. Keep the exact question numbers (Q1, Q2, etc.)
    3. Preserve the original question text exactly as given
    4. Extract the answer and put it in the content field
    5. For questions without subquestion labels, use label: null
    6. Print both the question and answer before converting to JSON
    
    Here's the text to format:
    """ + text

    data = {
        "contents": [{
            "parts": [
                {
                    "text": prompt
                }
            ]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            content = result['candidates'][0]['content']
            if 'parts' in content and len(content['parts']) > 0:
                # Extract JSON from the response
                response_text = content['parts'][0]['text']
                
                # Print the questions and answers
                print("\nExtracted Questions and Answers:")
                print("=" * 50)
                
                # Find the JSON array in the response
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                if start_idx != -1 and end_idx != -1:
                    # Print the text before the JSON array
                    print(response_text[:start_idx].strip())
                    
                    # Parse and return the JSON
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
        
        raise ValueError("Invalid response format from Gemini API")
    except Exception as e:
        logger.error(f"Error formatting text with Gemini API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"API Response: {e.response.text}")
        raise

def extract_original_content(pdf_path):
    """
    Extract the original text content from the PDF.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of dictionaries containing original question and answer content
    """
    try:
        doc = fitz.open(pdf_path)
        content = []
        
        # Skip first 2 pages
        for page_num in range(2, len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Split text into questions and answers
            lines = text.split('\n')
            current_question = None
            current_answers = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if 'Question' in line or line.startswith('Q'):
                    if current_question:
                        content.append({
                            'question': current_question,
                            'subquestions': current_answers
                        })
                    current_question = line
                    current_answers = []
                elif any(line.startswith(prefix) for prefix in ['A)', 'B)', 'Ⅲ', 'Ⅳ', 'Ⅱ', 'Ⅰ', ')', '①', '②', '③', '④']):
                    current_answers.append({
                        'label': line[0],
                        'content': line[2:].strip()
                    })
                elif current_answers:
                    current_answers[-1]['content'] += ' ' + line
            
            # Add the last question
            if current_question:
                content.append({
                    'question': current_question,
                    'subquestions': current_answers
                })
        
        return content
    except Exception as e:
        logger.error(f"Error extracting original content: {str(e)}")
        return []

def final_format_with_gemini(extracted_data, original_content):
    """
    Do a final pass with Gemini to compare and format the extracted text with original content.
    
    Args:
        extracted_data (list): List of extracted questions and answers
        original_content (list): List of original questions and answers
        
    Returns:
        list: Formatted and compared data
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    
    # Convert both data structures to strings for comparison
    extracted_str = json.dumps(extracted_data, indent=2)
    original_str = json.dumps(original_content, indent=2)
    
    prompt = f"""Compare the extracted handwritten content with the original PDF content and format the output.
    Ensure that:
    1. Question numbers match exactly with the original PDF
    2. Answer labels (A, B, C, etc.) match the original format
    3. Content is properly formatted and aligned
    4. Any discrepancies are noted
    
    Original PDF Content:
    {original_str}
    
    Extracted Handwritten Content:
    {extracted_str}
    
    Format the output as a JSON array with the following structure:
    [
      {{
        "question": "Q1",
        "original_question": "Q1 from PDF",
        "is_matched": true/false,
        "subquestions": [
          {{
            "label": "A",
            "content": "Extracted answer content",
            "original_content": "Original answer content",
            "is_matched": true/false
          }}
        ]
      }}
    ]
    """

    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        "contents": [{
            "parts": [
                {
                    "text": prompt
                }
            ]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            content = result['candidates'][0]['content']
            if 'parts' in content and len(content['parts']) > 0:
                # Extract JSON from the response
                response_text = content['parts'][0]['text']
                # Find the JSON array in the response
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
        
        raise ValueError("Invalid response format from Gemini API")
    except Exception as e:
        logger.error(f"Error in final formatting with Gemini API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"API Response: {e.response.text}")
        raise

@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'pdf' not in request.files:
        return jsonify({'message': 'No PDF file provided'}), 400
    
    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            pdf_file.save(temp_pdf.name)
            temp_pdf_path = temp_pdf.name

        # Extract original content
        original_content = extract_original_content(temp_pdf_path)
        logger.info("Original Content: %s", json.dumps(original_content, indent=2))

        # Process the PDF for handwritten content
        images = pdf_to_images(temp_pdf_path)
        if not images:
            return jsonify({'message': 'No pages found to process'}), 400

        # Extract text from all images using Gemini
        all_text = ""
        for i, image in enumerate(images):
            text = call_gemini_api(image)
            if text.strip():
                all_text += text + "\n\n"

        if not all_text.strip():
            return jsonify({'message': 'No text extracted from the images'}), 400

        # Format the extracted text
        formatted_text = format_qa_text(all_text)
        logger.info("Formatted Text: %s", formatted_text)
        
        # Convert to structured JSON format
        extracted_data = format_text_with_gemini(formatted_text)
        logger.info("Extracted Data: %s", json.dumps(extracted_data, indent=2))

        # Do final comparison and formatting
        final_data = final_format_with_gemini(extracted_data, original_content)
        logger.info("Final Data: %s", json.dumps(final_data, indent=2))

        # Clean up temporary file
        os.unlink(temp_pdf_path)

        # Return the final formatted data
        return jsonify({
            'extracted_data': extracted_data,
            'formatted_data': final_data
        })

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return jsonify({'message': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
