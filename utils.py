# Import necessary libraries
from huggingface_hub import login
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
)
from dotenv import load_dotenv
import os
import PyPDF2
import google.generativeai as genai 

# Load environment variables from a .env file
load_dotenv()
# Function to handle Hugging Face login
def hf_login():
    """
    Inputs:
        None directly. Reads `HF_TOKEN` from environment variables.
    Uses:
        Logs into Hugging Face Hub using a stored token.
    Outputs:
        None. Sets up a session for Hugging Face-related operations.
    """
    load_dotenv()
    # Retrieve Hugging Face token from environment variables
    token = os.environ['HF_TOKEN']
    login(token=token)
hf_login()  # Ensure Hugging Face login
# Function to handle Generative AI login
def gem_login():
    """
    Inputs:
        None directly. Reads `GEM_API` from environment variables.
    Uses:
        Configures Google Generative AI SDK with the provided API key.
    Outputs:
        None. Sets up a session for Google Generative AI operations.
    """
    # Retrieve Generative AI API key from environment variables
    api = os.environ["GEM_API"]
    genai.configure(api_key=api)

gem_login()  # Ensure Generative AI login
# Function to generate a summary using the BART model
def generate_summary_bart(text):
    """
    Inputs:
        text (str): The input text to be summarized.
    Uses:
        Hugging Face's BART model to generate a summary of the text.
    Outputs:
        summary (str): The generated summary as a string.
    """
    hf_login()  # Ensure Hugging Face login
    try:
        # Load BART model and tokenizer
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        # Tokenize input text
        tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt", max_length=1024)
        
        # Generate summary
        outputs = model.generate(
            **tokens,
            max_length=120,          # Target length for summary
            min_length=50,           # Minimum summary length
            num_beams=8,             # Number of beams for beam search
            length_penalty=3.0,      # Penalty to discourage overly long summaries
            no_repeat_ngram_size=4,  # Avoid repetition of n-grams
            early_stopping=True,     # Stop generation early if conditions are met
            diversity_penalty=1.5,   # Encourage diverse summaries
            temperature=1.1,         # Add randomness to generation
            top_k=50,                # Limit token sampling to top 50 tokens
            top_p=0.92,              # Nucleus sampling for diversity
            num_beam_groups=2        # Group beam search into subgroups
        )
        
        # Decode summary into a readable string
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = summary.replace("<pad>", "").replace("<n>", "").replace("</s>", "").strip()
    except Exception as e:
        print(f"BART Error: {e}")
        return None
    finally:
        # Release resources to free memory
        del model, tokenizer
    return summary

# Function to generate a summary using Pegasus model
def generate_summary_pegasus(text):
    """
    Inputs:
        text (str): The input text to be summarized.
    Uses:
        Hugging Face's Pegasus model for summarization.
    Outputs:
        summary (str): The generated summary as a string.
    """
    try:
        # Load Pegasus model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/bigbird-pegasus-large-pubmed")
        
        # Tokenize input text
        tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
        
        # Generate summary
        outputs = model.generate(**tokens)
        
        # Decode summary into a readable string
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = summary.replace("<pad>", "").replace("<n>", "").replace("</s>", "").strip()
    except Exception as e:
        print(f"Pegasus Error: {e}")
        return None
    return summary

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """
    Inputs:
        pdf_path (str): Path to the PDF file.
    Uses:
        PyPDF2 to read and extract text from each page of the PDF.
    Outputs:
        pdf_text (str): Combined text from all pages in the PDF.
    """
    try:
        pdf_text = ""
        # Open the PDF file in read-binary mode
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            # Iterate through each page and extract text
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
        return pdf_text  # Return the extracted text
    except Exception as e:
        # Handle and report any errors
        print(f"PDF Error: {e}")
        return None

# Function to answer questions based on provided context
def question_answering(text, question):
    """
    Inputs:
        text (str): Context to base the answer on.
        question (str): The question to be answered.
    Uses:
        Google Generative AI to generate an answer strictly from the context.
    Outputs:
        answer (str): The generated answer or an error message.
    """
      # Ensure Generative AI login
    pre_prompt = f"""
                    You are an AI restricted to answering questions strictly based on the provided context. Do not provide any information, assumptions, or responses that are not explicitly stated in the given text. 

                    Context: 
                    {text}

                    For each question, if the information is not present in the text, respond with: 'The text does not provide this information.'
                    So now provide the complete answer to the question.
                    Question: {question}
                    Answer:
                    """
    try:
        # Use Generative AI to generate a response
        model = genai.GenerativeModel(model_name='gemini-1.5-pro-latest')
        response = model.generate_content(pre_prompt)
        answer = response.text
        return answer
    except Exception as e:
        return str(e)

