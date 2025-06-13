import os
import re
import nltk
import torch
import PyPDF2
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from typing import Dict, List, Tuple, Any
import json
import time
from huggingface_hub import login

# Try to import fitz (PyMuPDF) for faster PDF reading
try:
    import fitz # PyMuPDF
    PDF_READER_LIB = "PyMuPDF"
except ImportError:
    print("PyMuPDF (fitz) not found. Falling back to PyPDF2 for PDF reading (may be slower).")
    PDF_READER_LIB = "PyPDF2"

# Hugging Face Authentication
HF_TOKEN = "" # Your Hugging Face token

class PDFAnswerEvaluator:
    CACHE_FILE = "evaluation_cache.json"

    def __init__(self,
                 embedding_model='all-MiniLM-L6-v2',
                 llm_model_name='mistralai/Mistral-7B-Instruct-v0.3',
                 load_in_4bit: bool = True):
        # NLTK Setup
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Semantic Embedding Model
        self.sbert_model = SentenceTransformer(embedding_model)

        # --- Local LLM Setup ---
        login(token=HF_TOKEN) # Uncomment and set HF_TOKEN if needed
        self.llm_model_name = llm_model_name

        print(f"Loading LLM: {llm_model_name} locally...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        if torch.cuda.is_available() and load_in_4bit:
            print("CUDA is available and load_in_4bit is True. Loading model in 4-bit quantization.")
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 # or torch.float16 if bfloat16 not supported
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                quantization_config=nf4_config,
                device_map="auto" # Distributes model across available GPUs or uses CPU if no GPU
            )
        elif torch.cuda.is_available():
            print("CUDA is available. Loading model in full precision (requires more VRAM).")
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                torch_dtype=torch.float16, # Use float16 for reduced VRAM usage
                device_map="auto"
            )
        else:
            print("CUDA not available. Loading model on CPU (will be very slow and memory-intensive).")
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                device_map="cpu"
            )

        self.llm_model.eval() # Set model to evaluation mode

        # Storage for results
        self.evaluation_results = []

        # --- Persistent Caching ---
        self._pdf_content_cache: Dict[str, str] = {} # Maps PDF path to its raw text content
        self._extracted_question_cache: Dict[str, Dict[int, Dict]] = {} # Maps PDF path to {q_num: extracted_json}
        self._load_cache()

    def _load_cache(self):
        """Loads cached data from a JSON file."""
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    self._pdf_content_cache = cached_data.get('pdf_content', {})
                    self._extracted_question_cache = cached_data.get('extracted_questions', {})
                print(f"Loaded cache from {self.CACHE_FILE}")
            except json.JSONDecodeError as e:
                print(f"Error decoding cache file {self.CACHE_FILE}: {e}. Starting with empty cache.")
                self._pdf_content_cache = {}
                self._extracted_question_cache = {}
            except Exception as e:
                print(f"An unexpected error occurred while loading cache: {e}. Starting with empty cache.")
                self._pdf_content_cache = {}
                self._extracted_question_cache = {}
        else:
            print("No cache file found. Starting with empty cache.")

    def _save_cache(self):
        """Saves current cache data to a JSON file."""
        try:
            with open(self.CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    'pdf_content': self._pdf_content_cache,
                    'extracted_questions': self._extracted_question_cache
                }, f, indent=4)
            print(f"Saved cache to {self.CACHE_FILE}")
        except Exception as e:
            print(f"Error saving cache to {self.CACHE_FILE}: {e}")

    def read_pdf_content(self, file_path: str) -> str:
        """Read PDF content using PyMuPDF if available, else PyPDF2. Uses content cache."""
        if file_path in self._pdf_content_cache:
            print(f"Using cached PDF content for {file_path}")
            return self._pdf_content_cache[file_path]

        text = ""
        if PDF_READER_LIB == "PyMuPDF":
            try:
                doc = fitz.open(file_path)
                for page in doc:
                    text += page.get_text() + "\n"
                doc.close()
            except Exception as e:
                print(f"Error reading file {file_path} with PyMuPDF: {e}. Falling back to PyPDF2.")
                text = self._read_pdf_content_pypdf2(file_path)
        else:
            text = self._read_pdf_content_pypdf2(file_path)

        cleaned_text = self._clean_text(text)
        self._pdf_content_cache[file_path] = cleaned_text # Cache the cleaned text
        return cleaned_text

    def _read_pdf_content_pypdf2(self, file_path: str) -> str:
        """Internal method to read PDF content using PyPDF2."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading file {file_path} with PyPDF2: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """Cleans and normalizes extracted text."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
        text = text.replace('-\n', '')    # Remove hyphenated line breaks
        text = re.sub(r'[^\x00-\x7F]+', '', text) # Remove non-ASCII characters
        return text.strip()

    @torch.no_grad() # Disable gradient calculation for inference
    def generate_llm_response(self, prompt_text: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        """Generate response using local LLM with exponential backoff retries."""
        max_retries = 3
        base_delay = 1 # seconds
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "user", "content": prompt_text}
                ]
                input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(self.llm_model.device)

                output_ids = self.llm_model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id, # Important for generation
                    eos_token_id=self.tokenizer.eos_token_id # Stop generation at EOS token
                )

                response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                return response.strip()

            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA Out of Memory error on attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying after clearing cache and waiting...")
                    torch.cuda.empty_cache()
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    print("Failed after multiple OOM attempts. Consider reducing model size or using more VRAM.")
                    return "{}" # Return empty JSON-like string on final failure
            except Exception as e:
                print(f"Error during local LLM generation on attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"Local LLM generation failed after {max_retries} attempts.")
                    return "{}"

    def _get_extracted_data_from_cache_or_llm(self, question_num: int, full_document_text: str, pdf_path: str, doc_type: str) -> Dict:
        """
        Helper to check cache for extracted data or make LLM call if not found.
        doc_type should be 'answer_key' or 'student_answers'.
        """
        if pdf_path not in self._extracted_question_cache:
            self._extracted_question_cache[pdf_path] = {}

        if question_num in self._extracted_question_cache[pdf_path]:
            print(f"Using cached extracted data for Q{question_num} from {os.path.basename(pdf_path)} ({doc_type}).")
            return self._extracted_question_cache[pdf_path][question_num]

        print(f"Extracting Q{question_num} data from {os.path.basename(pdf_path)} ({doc_type}) using LLM...")

        if doc_type == 'answer_key':
            extraction_prompt = f"""
            From the following document, extract the details for Question {question_num}.
            Identify the exact question text, the total marks, and the "Synoptic Answer" as a list of distinct points.
            Crucially, capture the *full text* of each point, including any multi-line descriptions.
            The `point_marks` should be a direct integer value (e.g., 1, 2, 5).

            DOCUMENT:
            {full_document_text}

            Provide the output in the following EXACT JSON format. If Question {question_num} or its details are not found, return an empty JSON object {{}}.
            {{
                "question_text": "...",
                "total_marks": [integer value, e.g., 6],
                "expert_points": [
                    {{"point_text": "...", "point_marks": [integer]}},
                    {{"point_text": "...", "point_marks": [integer]}}
                    // ... include all points for the question
                ]
            }}
            """
        else: # doc_type == 'student_answers'
            extraction_prompt = f"""
            From the following document, locate Question {question_num} and extract the *entire* student's answer associated with it.
            Capture all text following the question number, up until the next question number or the end of the document.
            Do not truncate or summarize the student's response.

            DOCUMENT:
            {full_document_text}

            Provide the output in the following EXACT JSON format. If Question {question_num} or its answer is not found, return an empty JSON object {{}}.
            {{
                "student_answer_text": "..."
            }}
            """

        # Increased max_tokens slightly for extraction to allow for longer answers
        response_text = self.generate_llm_response(extraction_prompt, max_tokens=1000, temperature=0.0) # Lower temperature for deterministic extraction
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                extracted_data = json.loads(match.group(0))
                # Store in cache
                self._extracted_question_cache[pdf_path][question_num] = extracted_data
                return extracted_data
            print(f"Could not find JSON in LLM response for Q{question_num} extraction ({doc_type}): {response_text[:200]}...")
            return {}
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for extraction Q{question_num} ({doc_type}): {e} - Response: {response_text[:200]}")
            return {}
        except Exception as e:
            print(f"Unexpected error during extraction for Q{question_num} ({doc_type}): {e}")
            return {}

    def _evaluate_answer_pair(self, question_text: str, expert_points: List[Dict[str, Any]],
                            student_answer: str, total_marks: int) -> Dict:
        """Evaluates a student answer against expert points using LLM.
        Returns a dictionary of scores and feedback.
        """
        expert_points_formatted = "\n".join([
            f"- {p['point_text']} ({p['point_marks']} marks)" for p in expert_points
        ])
    
        # MODIFIED PROMPT SECTION STARTS HERE
        evaluation_prompt = f"""
        You are an expert examiner.
        Evaluate the student's answer for the following question against the provided expert answer points.
        
        QUESTION: {question_text}
        
        EXPERT ANSWER POINTS:
        {expert_points_formatted}
        
        STUDENT ANSWER: {student_answer}
        
        TOTAL MARKS FOR THIS QUESTION: {total_marks}
        
        For each expert point, assess how well the student's answer addresses it. When assigning a score from 0 to 1 for each point (where 1 is a perfect match and 0 is no relevant information), consider the following:
            - Direct Coverage: How directly and completely does the student's answer address the specific content of the expert point?
            - Factual Accuracy: Is the information provided by the student related to this point factually correct? If the student attempts to address the point but includes factual errors related to it, the score for this point should be reduced.
            - Relevance & Partial Understanding: If the student's answer doesn't directly cover the point but provides factually correct and relevant information that demonstrates partial understanding or addresses a closely related aspect of the point's core idea, award partial credit.
            - A score of 0 should be given if the student's answer is completely irrelevant to the point, factually incorrect in its attempt to address the point, or if the point is not addressed at all.
        Assign a score from 0 to 1 for each point based on this comprehensive assessment.
        
        Also, provide an overall assessment based on the following criteria for the entire student answer:
        - Overall Semantic Accuracy (0-1): How well does the student answer match the overall meaning of the expert answer? [cite: 26]
        - Overall Factual Correctness (0-1): Are there any factual errors or misconceptions in the student's answer? [cite: 27]
        - Overall Understanding Depth (0-1): Does the student show deep conceptual understanding of the topic? [cite: 28]
        
        Provide constructive feedback (100-150 words) that highlights strengths and areas for improvement, focusing on how the student can improve. [cite: 29]
        
        Format your response EXACTLY as a JSON object with the following keys: [cite: 30]
        {{
            "point_evaluations": [
                {{"expert_point_text": "...", "score": [float 0-1], "feedback_on_point": "..."}},
                // ... for each expert point
            ],
            "overall_semantic_accuracy": [float 0-1],
            "overall_factual_correctness": [float 0-1],
            "overall_understanding_depth": [float 0-1],
            "feedback": "[string]"
        }}
        """
        # MODIFIED PROMPT SECTION ENDS HERE
    
        response_text = self.generate_llm_response(evaluation_prompt, max_tokens=700, temperature=0.1) # Increased max_tokens for detailed feedback
    
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                evaluation_data = json.loads(match.group(0))
                # Ensure scores are floats and within range
                for key in ['overall_semantic_accuracy', 'overall_factual_correctness', 'overall_understanding_depth']:
                    evaluation_data[key] = max(0.0, min(1.0, float(evaluation_data.get(key, 0.0))))
                
                if 'point_evaluations' in evaluation_data and isinstance(evaluation_data['point_evaluations'], list): # [cite: 31]
                    for pe in evaluation_data['point_evaluations']:
                        pe['score'] = max(0.0, min(1.0, float(pe.get('score', 0.0)))) # [cite: 31]
                else:
                    evaluation_data['point_evaluations'] = [] # Ensure it's a list even if LLM fails [cite: 31]
                return evaluation_data
            print(f"Could not find JSON in LLM response for evaluation: {response_text[:200]}...")
            return {}
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for evaluation: {e} - Response: {response_text[:200]}")
            return {}
        except Exception as e:
            print(f"Unexpected error during evaluation: {e}")
            return {}

 


   

    def evaluate_from_pdfs(self, answer_key_pdf_path: str, student_answer_pdf_paths: List[str], max_questions: int = 10):
        """Main evaluation function that processes answer key and multiple student PDFs."""
        print("Reading Answer Key PDF...")
        answer_key_text = self.read_pdf_content(answer_key_pdf_path) # Uses cache

        if not answer_key_text:
            print(f"Error: Could not read answer key PDF from {answer_key_pdf_path}. Aborting evaluation.")
            return None

        print(f"Answer key text length: {len(answer_key_text)} characters")

        self.evaluation_results = [] # Reset results for this evaluation session

        # Process each student PDF
        for student_pdf_path in student_answer_pdf_paths:
            print(f"\n{'#'*80}")
            print(f"EVALUATING STUDENT: {os.path.basename(student_pdf_path)}")
            print(f"{'#'*80}")

            student_text = self.read_pdf_content(student_pdf_path) # Uses cache
            if not student_text:
                print(f"Error: Could not read student answer PDF from {student_pdf_path}. Skipping this student.")
                continue

            print(f"Student text length: {len(student_text)} characters")

            student_total_marks_possible = 0
            student_total_marks_obtained = 0
            student_question_results = []

            for question_num in range(1, max_questions + 1):
                print(f"\n{'='*60}")
                print(f"Processing Question {question_num} for {os.path.basename(student_pdf_path)}...")
                print('='*60)

                # Stage 1: Extract question details and expert answer points from answer key
                expert_details = self._get_extracted_data_from_cache_or_llm(question_num, answer_key_text, answer_key_pdf_path, 'answer_key')
                if not expert_details.get('question_text') or expert_details.get('total_marks') is None:
                    print(f"Question {question_num} details not found in answer key. Stopping evaluation for this student.")
                    break # Stop processing questions for current student if answer key question not found

                # Stage 1: Extract student answer text
                student_ans_details = self._get_extracted_data_from_cache_or_llm(question_num, student_text, student_pdf_path, 'student_answers')
                student_actual_answer = student_ans_details.get('student_answer_text', '')

                if not student_actual_answer:
                    print(f"Student answer for Question {question_num} not found. Assigning 0 marks and feedback.")
                    # Add a placeholder result to keep track
                    question_result = {
                        'student_filename': os.path.basename(student_pdf_path),
                        'question_number': str(question_num),
                        'question': expert_details.get('question_text', 'N/A'),
                        'answer_key_points': expert_details.get('expert_points', []),
                        'student_answer': 'NOT FOUND',
                        'total_marks': expert_details.get('total_marks', 0),
                        'point_evaluations': [],
                        'overall_semantic_accuracy': 0.0,
                        'overall_factual_correctness': 0.0,
                        'overall_understanding_depth': 0.0,
                        'final_score': 0.0,
                        'marks_obtained': 0.0,
                        'feedback': 'Student answer could not be extracted or was empty.'
                    }
                    student_question_results.append(question_result)
                    student_total_marks_possible += expert_details.get('total_marks', 0)
                    continue

                # Stage 2: Evaluate the extracted pair
                evaluation_scores = self._evaluate_answer_pair(
                    question_text=expert_details['question_text'],
                    expert_points=expert_details['expert_points'],
                    student_answer=student_actual_answer,
                    total_marks=expert_details['total_marks']
                )

                # Calculate final score based on point evaluations
                calculated_marks_obtained_for_question = 0.0
                for pe in evaluation_scores.get('point_evaluations', []):
                    # Find the corresponding expert point to get its marks
                    expert_point_match = next((p for p in expert_details['expert_points'] if p['point_text'] == pe['expert_point_text']), None)
                    if expert_point_match:
                        try:
                            # Ensure 'point_marks' is an integer before multiplication
                            calculated_marks_obtained_for_question += pe['score'] * int(expert_point_match['point_marks'])
                        except (ValueError, TypeError) as e:
                            print(f"Error converting 'point_marks' to int for expert point '{expert_point_match.get('point_text', 'N/A')}' in Q{question_num}: {expert_point_match.get('point_marks', 'N/A')}. Error: {e}. Assuming 0 marks for this point.")
                            # Continue without adding to calculated_marks_obtained_for_question
                    else:
                        print(f"Warning: Expert point '{pe['expert_point_text'][:50]}...' from LLM not found in original expert points for Q{question_num}. Check prompt/LLM output.")


                final_score_normalized = (calculated_marks_obtained_for_question / expert_details['total_marks']) if expert_details['total_marks'] > 0 else 0.0
                final_score_normalized = max(0.0, min(1.0, final_score_normalized)) # Ensure 0-1 range

                question_result = {
                    'student_filename': os.path.basename(student_pdf_path),
                    'question_number': str(question_num),
                    'question': expert_details['question_text'],
                    'answer_key_points': expert_details['expert_points'],
                    'student_answer': student_actual_answer,
                    'total_marks': expert_details['total_marks'],
                    'point_evaluations': evaluation_scores.get('point_evaluations', []),
                    'overall_semantic_accuracy': evaluation_scores.get('overall_semantic_accuracy', 0.0),
                    'overall_factual_correctness': evaluation_scores.get('overall_factual_correctness', 0.0),
                    'overall_understanding_depth': evaluation_scores.get('overall_understanding_depth', 0.0),
                    'final_score': final_score_normalized,
                    'marks_obtained': calculated_marks_obtained_for_question,
                    'feedback': evaluation_scores.get('feedback', 'No feedback available.')
                }
                student_question_results.append(question_result)

                student_total_marks_possible += question_result['total_marks']
                student_total_marks_obtained += question_result['marks_obtained']

                # Print individual result
                print(f"Question {question_num}: {question_result['marks_obtained']:.1f}/{question_result['total_marks']} marks")
                print(f"Final Score (Normalized): {question_result['final_score']:.2f}")
                print(f"Question: {question_result['question'][:70]}...") # Truncate for display
                print(f"Student Answer: {question_result['student_answer'][:70]}...") # Truncate for display

            # Add this student's overall summary to the main results list
            overall_percentage = (student_total_marks_obtained / student_total_marks_possible) * 100 if student_total_marks_possible > 0 else 0
            self.evaluation_results.append({
                'student_filename': os.path.basename(student_pdf_path),
                'summary': {
                    'total_questions_evaluated': len(student_question_results),
                    'total_marks_obtained': student_total_marks_obtained,
                    'total_marks_possible': student_total_marks_possible,
                    'overall_percentage': overall_percentage,
                    'letter_grade': self.get_letter_grade(overall_percentage)
                },
                'detailed_results': student_question_results
            })

        self._save_cache() # Save cache after all evaluations are done

        if not self.evaluation_results:
            print("No students were successfully evaluated.")
            return None

        # Print overall summary for all students
        print("\n" + "="*80)
        print("OVERALL EVALUATION SUMMARY FOR ALL STUDENTS")
        print("="*80)
        for student_eval_summary in self.evaluation_results:
            summary = student_eval_summary['summary']
            print(f"\nStudent: {student_eval_summary['student_filename']}")
            print(f"  Questions Evaluated: {summary['total_questions_evaluated']}")
            print(f"  Total Marks Obtained: {summary['total_marks_obtained']:.1f}/{summary['total_marks_possible']}")
            print(f"  Overall Percentage: {summary['overall_percentage']:.1f}%")
            print(f"  Letter Grade: {summary['letter_grade']}")
        print("="*80)

        return self.evaluation_results

    def get_letter_grade(self, percentage):
        """Convert percentage to letter grade"""
        if percentage >= 90:
            return "A+"
        elif percentage >= 85:
            return "A"
        elif percentage >= 80:
            return "A-"
        elif percentage >= 75:
            return "B+"
        elif percentage >= 70:
            return "B"
        elif percentage >= 65:
            return "B-"
        elif percentage >= 60:
            return "C+"
        elif percentage >= 55:
            return "C"
        elif percentage >= 50:
            return "C-"
        elif percentage >= 45:
            return "D"
        else:
            return "F"

    def print_detailed_results(self):
        """Print detailed results for each student and question."""
        if not self.evaluation_results:
            print("No evaluation results available to print.")
            return

        for student_eval_data in self.evaluation_results:
            print(f"\n\n{'='*80}")
            print(f"DETAILED RESULTS FOR STUDENT: {student_eval_data['student_filename']}")
            print(f"{'='*80}")
            for result in student_eval_data['detailed_results']:
                print("\n" + "-"*80)
                print(f"QUESTION {result['question_number']}: {result['question']}")
                print("-"*80)
                print("EXPERT ANSWER POINTS:")
                for point in result['answer_key_points']:
                    print(f"  - {point['point_text']} ({point['point_marks']} marks)")
                print("-"*80)
                print("STUDENT ANSWER:")
                print(result['student_answer'])
                print("-"*80)
                print("POINT-WISE SCORES:")
                if not result['point_evaluations']:
                    print("  No point-wise evaluations available (LLM might have failed to extract or parse them).")
                for pe in result['point_evaluations']:
                    print(f"  • Point: {pe.get('expert_point_text', 'N/A')[:70]}...")
                    print(f"    Score: {pe.get('score', 0.0):.2f}")
                    print(f"    Feedback on Point: {pe.get('feedback_on_point', 'No specific feedback for this point.')}")
                print("-"*80)
                print("OVERALL SCORES:")
                print(f"• Overall Semantic Accuracy:    {result['overall_semantic_accuracy']:.2f}/1.00")
                print(f"• Overall Factual Correctness:  {result['overall_factual_correctness']:.2f}/1.00")
                print(f"• Overall Understanding Depth:  {result['overall_understanding_depth']:.2f}/1.00")
                print(f"• Final Normalized Score:       {result['final_score']:.2f}/1.00")
                print(f"• Marks Obtained:               {result['marks_obtained']:.1f}/{result['total_marks']}")
                print("-"*80)
                print("OVERALL FEEDBACK:")
                print(result['feedback'])
                print("="*80)

    def export_results_to_csv(self, filename="evaluation_results.csv"):
        """Exports results for all students to a CSV file."""
        if not self.evaluation_results:
            print("No evaluation results to export.")
            return

        all_df_data = []
        for student_eval_data in self.evaluation_results:
            student_filename = student_eval_data['student_filename']
            # summary = student_eval_data['summary'] # Not directly used in row, but available
            for result in student_eval_data['detailed_results']:
                # Flatten point evaluations into separate columns or a single string
                point_eval_summary = "; ".join([
                    f"Point: {pe.get('expert_point_text', 'N/A')} | Score: {pe.get('score', 0.0):.2f} | Feedback: {pe.get('feedback_on_point', 'N/A')}"
                    for pe in result['point_evaluations']
                ])

                # Prepare expert points for CSV
                expert_points_text = "; ".join([
                    f"{p['point_text']} ({p['point_marks']} marks)" for p in result['answer_key_points']
                ])

                all_df_data.append({
                    'Student_Filename': student_filename,
                    'Question_Number': result['question_number'],
                    'Question_Text': result['question'],
                    'Student_Answer': result['student_answer'],
                    'Answer_Key_Points_Text': expert_points_text,
                    'Total_Question_Marks': result['total_marks'],
                    'Marks_Obtained': result['marks_obtained'],
                    'Final_Normalized_Score': result['final_score'],
                    'Overall_Semantic_Accuracy': result['overall_semantic_accuracy'],
                    'Overall_Factual_Correctness': result['overall_factual_correctness'],
                    'Overall_Understanding_Depth': result['overall_understanding_depth'],
                    'Point_Evaluations_Detailed': point_eval_summary, # Flattened point evaluations
                    'Overall_Feedback': result['feedback']
                })

        df = pd.DataFrame(all_df_data)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")


def main():
    """Example usage"""
    # Initialize the evaluator
    evaluator = PDFAnswerEvaluator(load_in_4bit=True)

    # IMPORTANT: Replace with actual paths to your PDF files
    answer_key_pdf = "/kaggle/input/answers/Answer_Key.pdf" # Your uploaded answer key
    student_answer_pdfs = [
        "/kaggle/input/answers/student_key.pdf", # Your uploaded student key
        # Add more student PDF paths here if you have them, e.g.:
        # "student_key_2.pdf",
        # "student_key_3.pdf"
    ]

    # Check if files exist before proceeding
    if not os.path.exists(answer_key_pdf):
        print(f"Error: Answer key PDF not found at {answer_key_pdf}")
        return
    for student_pdf in student_answer_pdfs:
        if not os.path.exists(student_pdf):
            print(f"Error: Student answer PDF not found at {student_pdf}. Please check path.")
            return

    print(f"\nLooking for files: {answer_key_pdf} and {student_answer_pdfs}")

    # Evaluate answers from PDFs (max 5 questions by default)
    # This will use/populate the cache for PDF content and extracted question details
    results = evaluator.evaluate_from_pdfs(answer_key_pdf, student_answer_pdfs, max_questions=5)

    if results:
        # Print detailed results
        evaluator.print_detailed_results()

        # Export to CSV
        evaluator.export_results_to_csv("comprehensive_evaluation_results.csv")
    else:
        print("No results to display - check file reading or evaluation process.")

if __name__ == "__main__":
    main()
