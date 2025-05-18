import json
import re
import os
import sys
import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import io
import base64

# Set a higher recursion depth limit if needed for complex structures
sys.setrecursionlimit(2000)

# --- Configuration ---
# Replace with your actual API key or set it as an environment variable
# It's recommended to use environment variables for security
# API_KEY = os.getenv("GEMINI_API_KEY") # Use environment variable
API_KEY = ""  # Hardcoded key - Replace with your key or use env var

if not API_KEY or API_KEY == "YOUR_ACTUAL_API_KEY_HERE":
    print("Error: GEMINI_API_KEY is not set or is the placeholder value.")
    print("Please set the GEMINI_API_KEY environment variable or replace API_KEY in the script with your actual key.")
    sys.exit(1)

genai.configure(api_key=API_KEY)

# Choose a Gemini model that supports multimodal input (text + images)
MODEL_NAME = 'gemini-2.0-flash'  # Use pro version for multimodal capabilities


# --- Helper Function to Convert PDF Pages to Images ---
def get_pdf_page_images(pdf_path, start_page=2, dpi=300):
    """
    Converts PDF pages to images starting from the specified page index (0-based).
    Returns a list of PNG images as bytes.
    """
    print(f"Opening PDF file: {pdf_path}")
    image_list = []
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count
        print(f"PDF opened successfully. Total pages: {num_pages}")
        
        # Validate start_page
        if start_page >= num_pages:
            print(f"Warning: start_page ({start_page}) is >= number of pages ({num_pages}). Using page 0.")
            start_page = 0
        
        # Process pages starting from start_page
        for page_num in range(start_page, num_pages):
            print(f"Converting page {page_num + 1} to image...")
            page = doc.load_page(page_num)
            
            # Render page to a pixmap (image)
            pix = page.get_pixmap(dpi=dpi)
            
            # Convert pixmap to PNG bytes
            img_bytes = pix.pil_tobytes(format="png")
            image_list.append(img_bytes)
            
            print(f"Page {page_num + 1} converted successfully.")
        
        doc.close()
        print(f"Converted {len(image_list)} pages to images.")
        return image_list
    
    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
        return []
    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")
        return []


# --- Main Processing Function ---
def process_pdf_with_gemini_vision(pdf_path: str, skip_pages: int = 2) -> list:
    """
    Extracts content from PDF by converting pages to images and using 
    Gemini's vision capabilities to interpret them. Processes one page at a time
    and combines the results.
    """
    # Get images of PDF pages starting from the specified page
    pdf_images = get_pdf_page_images(pdf_path, start_page=skip_pages)
    
    if not pdf_images:
        print("Failed to extract images from the PDF.")
        return []
    
    print(f"\nSuccessfully extracted {len(pdf_images)} page images from the PDF.")
    
    # --- Prepare for Gemini Vision API ---
    try:
        # Initialize the Gemini Pro Vision model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Prepare the prompt for Gemini
        prompt = """
You are an AI assistant designed to extract structured information from an image of a document page, specifically focusing on questions and their corresponding answers.

Analyze the image of the document page provided, identify the main questions and any sub-questions based on patterns like numbering (1, 2, 3...), lettering (A, B, C...), Roman numerals (I, II, III...), or explicit question phrasing. Extract the full text associated with each question or sub-question, which represents the answer written in the document.

This is a single page from a multi-page document. Only extract questions and answers that are visible on this specific page.

Structure the output as a JSON array of objects. Each object in the array should represent a main question and have the following keys:
-   `question`: The label or text of the main question (e.g., "I", "Q2", "Question 3"). If a block of text seems to be an answer but doesn't have a clear preceding main question label, use `null`.
-   `subquestions`: An array of objects, where each object represents a sub-question or a distinct point/answer under the main question. Each sub-question object should have the following keys:
    -   `label`: The label of the sub-question (e.g., "A", "B", "AA", "1", "2"). If a block of text is an answer under a main question but doesn't have a clear sub-question label, use `null`.
    -   `content`: The full text of the answer or content associated with this sub-question or point.

Include only content visible on this specific page. If a question starts on this page but continues to the next, only include the portion visible on this page.

Please provide only the JSON output.
"""
        
        # Initialize the combined results as an empty list
        all_extracted_data = []
        
        # Process each page individually
        for page_num, img_bytes in enumerate(pdf_images):
            # Create PIL Image from bytes
            img = Image.open(io.BytesIO(img_bytes))
            
            print(f"\nProcessing page {page_num + skip_pages + 1}...")
            
            # Create content parts with just the prompt and this single image
            content_parts = [prompt, img]
            
            # Make API call for this single page
            response = model.generate_content(
                content_parts,
                request_options={'timeout': 300}  # Shorter timeout for single page
            )
            
            print(f"Received response from Gemini API for page {page_num + skip_pages + 1}.")
            response_text = response.text.strip()
            
            print(f"\n--- Raw API Response Text (Page {page_num + skip_pages + 1}) ---")
            print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
            print("--- End Raw API Response Text Preview ---")
            
            # Try to extract JSON from the response
            try:
                # Find the start and end of JSON content
                json_start = response_text.find('[')
                json_end = response_text.rfind(']')
                
                if json_start == -1 or json_end == -1 or json_end < json_start:
                    # If array not found, try finding a single JSON object
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}')
                    if json_start == -1 or json_end == -1 or json_end < json_start:
                        print(f"Warning: Could not find valid JSON in API response for page {page_num + skip_pages + 1}.")
                        continue
                    else:
                        json_string = response_text[json_start:json_end + 1]
                        # Convert single object to array for consistency
                        if json_string.strip().startswith('{'):
                            json_string = f"[{json_string}]"
                else:
                    json_string = response_text[json_start:json_end + 1]
                
                # Clean up common issues in AI-generated JSON
                json_string = re.sub(r',\s*}', '}', json_string)
                json_string = re.sub(r',\s*]', ']', json_string)
                
                print(f"\nAttempting to parse JSON string for page {page_num + skip_pages + 1}...")
                page_data = json.loads(json_string)
                print(f"JSON parsed successfully for page {page_num + skip_pages + 1}.")
                
                # Add page information to each item
                for item in page_data:
                    item['page'] = page_num + skip_pages + 1
                
                # Add this page's data to our combined results
                all_extracted_data.extend(page_data)
                
            except Exception as json_err:
                print(f"JSON parsing error for page {page_num + skip_pages + 1}: {json_err}")
                continue
        
        # Return the combined results from all pages
        if all_extracted_data:
            print(f"\nSuccessfully extracted data from {len(all_extracted_data)} question blocks across all pages.")
            return all_extracted_data
        else:
            print("Warning: Could not extract valid JSON data from any page.")
            return []
        
    except Exception as e:
        print(f"An error occurred during the Gemini Vision API processing: {e}")
        return []


# --- Main execution block ---
if __name__ == "__main__":
    # Get the PDF file path from the user
    pdf_file_path = input("Please enter the full path to the PDF file: ")
    
    # Ask how many pages to skip
    try:
        pages_to_skip = int(input("How many initial pages would you like to skip? (default: 2): ") or "2")
    except ValueError:
        print("Invalid input. Using default value of 2.")
        pages_to_skip = 2
    
    # Process the PDF and get the structured data
    structured_output = process_pdf_with_gemini_vision(pdf_file_path, skip_pages=pages_to_skip)
    
    # Print the structured JSON output
    if structured_output:
        print("\n--- Summary of Extracted Data ---")
        print(f"Total question blocks extracted: {len(structured_output)}")
        
        # Group by page number
        pages_with_data = set(item.get('page', 'unknown') for item in structured_output)
        print(f"Pages with extracted data: {sorted(pages_with_data)}")
        
        # Save to file
        output_file = f"{os.path.splitext(pdf_file_path)[0]}_extracted_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structured_output, f, indent=2, ensure_ascii=False)
        print(f"\nExtracted data saved to {output_file}")
        
        # Ask if user wants to see the full JSON output
        show_json = input("\nWould you like to see the full JSON output? (y/n): ").lower()
        if show_json == 'y' or show_json == 'yes':
            print("\n--- Full Extracted Data (JSON) ---")
            print(json.dumps(structured_output, indent=2))
    else:
        print("\nFailed to extract structured data.")