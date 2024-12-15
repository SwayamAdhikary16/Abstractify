from huggingface_hub import login
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
)
from dotenv import load_dotenv
import os 

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
        outputs = model.generate(**tokens)
        
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
        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail", token=True)
        model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail", token=True)
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
