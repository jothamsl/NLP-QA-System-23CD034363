import os
import re
import sys
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Provider selection: 'gemini' or 'openai'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# API Keys
GEMINI_API_KEY = os.getenv("LLM_API_KEY") # Keeping legacy name for backward compatibility
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def preprocess_input(text):
    """
    Applies basic preprocessing:
    1. Lowercasing
    2. Tokenization (simple split)
    3. Punctuation removal
    """
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Remove punctuation (keeping spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Tokenization (just for demonstration, we rejoin for the API)
    tokens = text.split()
    
    # Reconstruct processed text
    processed_text = " ".join(tokens)
    return processed_text

def get_llm_response(question):
    """
    Sends the question to the selected LLM API and returns the answer.
    """
    try:
        if LLM_PROVIDER == 'openai':
            if not OPENAI_API_KEY:
                return "Error: OPENAI_API_KEY not found. Please set it in your .env file."
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", # Or gpt-4o if available/preferred
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ]
            )
            return response.choices[0].message.content

        elif LLM_PROVIDER == 'gemini':
            if not GEMINI_API_KEY:
                return "Error: LLM_API_KEY (for Gemini) not found. Please set it in your .env file."
            
            # Configure the API
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Create the model - switching to another available experimental model due to rate limits
            # Fallback list of models to try
            models_to_try = ['gemini-2.0-flash-thinking-exp-01-21', 'gemini-2.0-flash-exp', 'gemini-1.5-flash']
            
            last_error = None
            for model_name in models_to_try:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(question)
                    return response.text
                except Exception as e:
                    last_error = e
                    continue # Try next model
            
            return f"All Gemini models failed. Last error: {last_error}"
        
        else:
            return f"Error: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Please set it to 'gemini' or 'openai'."
            
    except Exception as e:
        return f"API Request Error: {e}"

def main():
    print("="*50)
    print("NLP Question-Answering CLI System")
    print("="*50)
    print("Type 'exit' or 'quit' to stop the application.\n")

    # Check for API Key based on provider
    if LLM_PROVIDER == 'gemini' and not GEMINI_API_KEY:
        print("No Gemini API Key found (LLM_API_KEY).")
    elif LLM_PROVIDER == 'openai' and not OPENAI_API_KEY:
        print("No OpenAI API Key found (OPENAI_API_KEY).")
        
    print(f"Current Provider: {LLM_PROVIDER.upper()}")
    
    while True:
        user_input = input("\nYour Question: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting application. Goodbye!")
            break
        
        if not user_input:
            continue

        print(f"Original Input: {user_input}")
        
        # Preprocessing
        processed_input = preprocess_input(user_input)
        print(f"Processed Input: {processed_input}")
        
        print("Thinking...")
        
        # API Call
        answer = get_llm_response(processed_input)
            
        print("-" * 50)
        print(f"LLM Answer:\n{answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()
