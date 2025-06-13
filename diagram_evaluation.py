# Import necessary libraries
import torch
import numpy as np # Still useful for image processing if needed, though less so without CLIP/Segformer
import os
from PIL import Image
import cv2 # Still useful for basic image loading/manipulation
import warnings
from typing import List, Dict, Tuple

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Install Hugging Face Transformers if not already present
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
except ImportError:
    print("Installing required packages (transformers, accelerate, bitsandbytes)...")
    os.system("pip install transformers accelerate bitsandbytes")
    from transformers import AutoProcessor, AutoModelForImageTextToText

# --- Model Loading (Only LLaVA) ---
def load_llava_model(hf_token=None):
    """Load only the LLaVA model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading LLaVA model...")
    llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", token=hf_token)
    
    if device.type == "cuda":
        try:
            import bitsandbytes as bnb
            print("bitsandbytes found, attempting 4-bit quantization for LLaVA.")
            llava_model = AutoModelForImageTextToText.from_pretrained(
                "llava-hf/llava-1.5-7b-hf", 
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token,
                load_in_4bit=True # Apply 4-bit quantization for speed/memory
            )
        except ImportError:
            print("bitsandbytes not found. Loading LLaVA in float16 without quantization.")
            llava_model = AutoModelForImageTextToText.from_pretrained(
                "llava-hf/llava-1.5-7b-hf", 
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token
            )
    else:
        # For CPU, load in float32 directly
        llava_model = AutoModelForImageTextToText.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", 
            torch_dtype=torch.float32,
            device_map=None, 
            token=hf_token
        )
        llava_model = llava_model.to(device)

    return {
        "llava": {"model": llava_model, "processor": llava_processor},
        "device": device
    }

# --- Utility Functions (Adapted or Kept) ---
def validate_image_paths(student_images: List[str], answer_images: List[str]) -> Tuple[List[str], List[str]]:
    """Validate and filter existing image paths"""
    valid_student = []
    valid_answer = []
    
    for student_path in student_images:
        if os.path.exists(student_path):
            valid_student.append(student_path)
        else:
            print(f"Warning: Student image not found: {student_path}")
    
    for answer_path in answer_images:
        if os.path.exists(answer_path):
            valid_answer.append(answer_path)
        else:
            print(f"Warning: Answer image not found: {answer_path}")
    
    return valid_student, valid_answer

# --- LLaVA-only Evaluation Logic ---
def evaluate_with_llava(question, rubrics, student_img_path, answer_img_path, models):
    """Generate AI-based feedback and score using only the LLaVA model."""
    try:
        llava_model = models["llava"]["model"]
        llava_processor = models["llava"]["processor"]
        device = models["device"]

        # Load images for LLaVA
        student_image = Image.open(student_img_path).convert("RGB")
        answer_image = Image.open(answer_img_path).convert("RGB")

        # --- Student Image Analysis ---
        # Prompt for student image analysis
        student_prompt_text = f"""USER: <image>
You are evaluating a student's diagram based on the following:

QUESTION: {question}
RUBRICS: {rubrics}

Look carefully at the image, which is the student's diagram, and describe its contents. Identify the key elements present, their arrangement, and any labels.
ASSISTANT:"""
        
        student_inputs = llava_processor(text=student_prompt_text, images=student_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            student_output = llava_model.generate(
                **student_inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                pad_token_id=llava_processor.tokenizer.eos_token_id
            )
        
        student_analysis = llava_processor.decode(student_output[0], skip_special_tokens=True)
        student_analysis = student_analysis.split("ASSISTANT:")[-1].strip()

        # --- Answer Key Analysis ---
        # Prompt for answer key analysis
        answer_prompt_text = f"""USER: <image>
Now look at this reference diagram (answer key) for the question:

QUESTION: {question}

Describe what you see in this reference diagram. Identify all the key elements, their ideal arrangement, and any labels that should be present according to the question.
ASSISTANT:"""
        
        answer_inputs = llava_processor(text=answer_prompt_text, images=answer_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            answer_output = llava_model.generate(
                **answer_inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                pad_token_id=llava_processor.tokenizer.eos_token_id
            )
        
        answer_analysis = llava_processor.decode(answer_output[0], skip_special_tokens=True)
        answer_analysis = answer_analysis.split("ASSISTANT:")[-1].strip()

        # --- Comparison and Feedback Generation ---
        # Prompt for comparison
        comparison_prompt_text = f"""USER: Based on your analysis of both diagrams:

Student diagram analysis: {student_analysis}

Reference diagram analysis: {answer_analysis}

Please:
1. Compare the student diagram to the reference diagram in detail, identifying similarities and differences.
2. Assign a score out of 10 based on the rubrics: {rubrics}
3. Format your score exactly like this on its own line: "LLAVA_SCORE: X/10"
4. Provide detailed feedback explaining:
    - Correct parts labeled and covered by the student.
    - Strengths of the student work (e.g., clarity, completeness of certain parts).
    - Areas for improvement (e.g., missing elements, incorrect relationships, poor labeling).
    - Specific missing or incorrect elements compared to the reference.
    - Suggestions for how the student could improve their diagram creation for future tasks.
ASSISTANT:"""
        
        # For the final comparison, no image is directly passed to LLaVA; its "understanding"
        # is already embedded in the analysis texts.
        feedback_inputs = llava_processor(text=comparison_prompt_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            feedback_output = llava_model.generate(
                **feedback_inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                pad_token_id=llava_processor.tokenizer.eos_token_id
            )
        
        feedback_text = llava_processor.decode(feedback_output[0], skip_special_tokens=True)
        feedback_text = feedback_text.split("ASSISTANT:")[-1].strip()

        # Extract LLaVA's score
        llava_score = 0.0 # Default to 0 if not found
        lines = feedback_text.split('\n')
        for line in lines:
            line = line.strip()
            if "LLAVA_SCORE:" in line:
                try:
                    score_text = line.replace("LLAVA_SCORE:", "").replace("/10", "").strip()
                    llava_score = float(score_text)
                    break
                except ValueError:
                    pass

        # Ensure score is within 0-10 range
        llava_score = max(0.0, min(10.0, llava_score))

        return {
            "llava_score": round(llava_score, 2),
            "ai_feedback": feedback_text
        }
    except Exception as e:
        return {
            "error": f"LLaVA evaluation failed: {str(e)}",
            "llava_score": 0.0, # Indicate failure with a low score
            "ai_feedback": f"AI feedback generation failed due to an error: {str(e)}. Please check model loading or input."
        }

# --- Main Evaluation Flow ---
def evaluate_multiple_images(student_images: List[str], answer_images: List[str], 
                            question: str, rubrics: str, hf_token: str = None) -> List[Dict]:
    """
    Evaluate multiple pairs of student and answer images using LLaVA only.
    """
    valid_student, valid_answer = validate_image_paths(student_images, answer_images)
    
    if not valid_student or not valid_answer:
        return [{"error": "No valid images found for comparison. Please check paths."}]
    
    min_length = min(len(valid_student), len(valid_answer))
    if len(valid_student) != len(valid_answer):
        print(f"Warning: Unequal number of images. Processing {min_length} pairs based on matching indices.")
    
    print("Loading LLaVA model (this may take a few minutes)...")
    models = load_llava_model(hf_token) # Changed to load_llava_model
    
    results = []
    
    for i in range(min_length):
        student_path = valid_student[i]
        answer_path = valid_answer[i]
        
        print(f"\nEvaluating pair {i+1}/{min_length}:")
        print(f"Student: {os.path.basename(student_path)}")
        print(f"Answer: {os.path.basename(answer_path)}")
        
        result = evaluate_with_llava(
            question=question,
            rubrics=rubrics,
            student_img_path=student_path,
            answer_img_path=answer_path,
            models=models
        )
        
        result["student_image"] = os.path.basename(student_path)
        result["answer_image"] = os.path.basename(answer_path)
        result["pair_number"] = i + 1
        
        results.append(result)
        
        if "error" not in result:
            print(f"LLaVA Score: {result['llava_score']}/10")
        else:
            print(f"Error for this pair: {result['error']}")
    
    return results

# --- Main Function ---
def main(student_images: List[str], answer_images: List[str], question: str, 
          rubrics: str, hf_token: str = None):
    """
    Main function to evaluate multiple image pairs using only LLaVA.
    """
    results = evaluate_multiple_images(
        student_images=student_images,
        answer_images=answer_images,
        question=question,
        rubrics=rubrics,
        hf_token=hf_token
    )

    return results

# --- Example Usage ---
if __name__ == "__main__":
    # IMPORTANT: Replace these with your actual image paths.
    # If you don't have images, you can create dummy ones for testing:
    # from PIL import Image
    # import numpy as np
    # Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)).save("student_diagram.jpg")
    # Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)).save("answer_key.png")

    student_images = [
        "/kaggle/input/pin-diagram/pin_student.jpg",  # Path to your student's diagram
        # "path/to/another_student_diagram.png", 
    ]
    
    answer_images = [
        "/kaggle/input/pin-diagram/pin_answer.jpg",  # Path to your answer key diagram
        # "path/to/another_answer_key.jpg", 
    ]
    
    hf_token = ""  # Replace with your actual Hugging Face token

    question_text = "Draw a detailed 8085 pin diagram?"
    rubrics_text = "Structure (3 marks), Components (5 marks), Presentation (2 marks)"
    
    print("\nStarting diagram evaluation process using LLaVA only...")
    results = main(
        student_images=student_images,
        answer_images=answer_images,
        question=question_text,
        rubrics=rubrics_text,
        hf_token=hf_token
    )
    
    # Print detailed results
    print(f"\n\n{'='*60}")
    print("             LLaVA-ONLY EVALUATION SUMMARY              ")
    print(f"{'='*60}")
    for i, result in enumerate(results):
        print(f"\n--- Result for Pair {i+1} (Student: {result.get('student_image', 'N/A')}, Answer: {result.get('answer_image', 'N/A')}) ---")
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nAI (LLaVA) Detailed Feedback:\n{result['ai_feedback']}")
        print(f"{'-'*60}")
