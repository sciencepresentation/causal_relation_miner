import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
import re
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file
    
    Args:
        pdf_file: Uploaded PDF file object from Streamlit
    
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def split_into_sentences(text):
    """
    Split text into sentences using NLTK
    
    Args:
        text: Input text string
    
    Returns:
        list: List of sentences
    """
    # Clean the text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    
    # Split into sentences
    sentences = sent_tokenize(text)
    
    return sentences

def filter_sentences(sentences, min_chars=15, max_chars=100):
    """
    Filter sentences based on character count and basic quality checks
    
    Args:
        sentences: List of sentences
        min_chars: Minimum number of characters
        max_chars: Maximum number of characters
    
    Returns:
        list: Filtered list of sentences
    """
    filtered = []
    
    for sentence in sentences:
        # Clean sentence
        sentence = sentence.strip()
        
        # Skip if too short or too long
        if len(sentence) < min_chars or len(sentence) > max_chars:
            continue
        
        # Skip if sentence doesn't contain alphabetic characters
        if not re.search(r'[a-zA-Z]', sentence):
            continue
        
        # Skip if sentence is mostly numbers or special characters
        alpha_count = sum(c.isalpha() for c in sentence)
        if alpha_count / len(sentence) < 0.5:
            continue
        
        filtered.append(sentence)
    
    return filtered

def clean_text(text):
    """
    Clean and normalize text
    
    Args:
        text: Input text string
    
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-()]', '', text)
    
    return text.strip()
