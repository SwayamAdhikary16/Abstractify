from huggingface_hub import login
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
)
from dotenv import load_dotenv
import os
import PyPDF2

def hf_login():
    load_dotenv()

    # Login to Hugging Face Hub
    token = os.environ['HF_TOKEN']
    login(token=token)

# Function to generate a summary using BART
def generate_summary_bart(text):
    hf_login()
    try:
        # Load model and tokenizer
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        # Tokenize input text
        tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt", max_length=1024)
        outputs = model.generate(
            **tokens,
            max_length=120,          # Slightly shorter for concise abstraction
            min_length=50,           # Ensure sufficient length for abstraction
            num_beams=8,             # Increase beams to improve output quality
            length_penalty=3.0,      # Stronger penalty for overly long summaries
            no_repeat_ngram_size=4,  # Prevents repeating 4-grams to enhance abstraction
            early_stopping=True,
            diversity_penalty=1.5,   # Higher penalty for more novel summaries
            temperature=1.1,         # Adds randomness to the output for more novel phrasing
            top_k=50,                # Limits tokens to the top 50 likely next tokens
            top_p=0.92,              # Use nucleus sampling to reduce repetition and increase diversity
            num_beam_groups=2       # Set num_beam_groups to a value greater than 1
        )
        
        # Decode summary
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = summary.replace("<pad>","").replace("<n>","").replace("</s>","").strip()
    except Exception as e:
        print(f"BART Error: {e}")
        return None
    finally:
        # Release model and tokenizer to free memory
        del model, tokenizer
    return summary

# Function to generate a summary using Pegasus
def generate_summary_pegasus(text):
    hf_login()
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed", token=True)
        model = AutoModelForSeq2SeqLM.from_pretrained("google/bigbird-pegasus-large-pubmed", token=True)
        #tuner007/pegasus_paraphrase
        # Tokenize input text
        tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
        outputs = model.generate(**tokens)
        
        # Decode summary
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = summary.replace("<pad>","").replace("<n>","").replace("</s>","").strip()
    except Exception as e:
        print(f"Pegasus Error: {e}")
        return None
    # finally:
    #     # Ensure cleanup
    #     del model, tokenizer
    return summary

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
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
        # Print any errors that occur during the process
        print(f"PDF Error: {e}")
        return None