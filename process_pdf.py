"""
Direct PDF processing script - no Streamlit interface
Processes 'mapping phenomena relevant.pdf' directly
"""

import sys
import json
from transformers import AutoModel, AutoTokenizer
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
import re

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    print(f"Extracting text from {pdf_path}...")
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def split_and_filter_sentences(text, min_chars=15, max_chars=100):
    """Split text into sentences and filter by length"""
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences
    sentences = sent_tokenize(text)
    
    # Filter
    filtered = []
    for sentence in sentences:
        sentence = sentence.strip()
        if min_chars <= len(sentence) <= max_chars:
            if re.search(r'[a-zA-Z]', sentence):
                alpha_count = sum(c.isalpha() for c in sentence)
                if alpha_count / len(sentence) >= 0.5:
                    filtered.append(sentence)
    
    return filtered

def process_pdf(pdf_path):
    """Main processing function"""
    print("="*70)
    print("CAUSAL RELATIONSHIP EXTRACTION")
    print("="*70)
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    print(f"✓ Extracted {len(text)} characters\n")
    
    # Split and filter sentences
    print("Splitting and filtering sentences...")
    sentences = split_and_filter_sentences(text)
    print(f"✓ Found {len(sentences)} sentences (after filtering)\n")
    
    if len(sentences) == 0:
        print("No sentences found after filtering!")
        return
    
    # Show sample sentences
    print("Sample sentences:")
    for i, sent in enumerate(sentences[:5], 1):
        print(f"  {i}. {sent}")
    print()
    
    # Load model
    print("Loading SocioCausaNet model...")
    print("(This may take a while on first run - downloading ~500MB)")
    try:
        repo_id = "rasoultilburg/SocioCausaNet"
        model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        print("✓ Model loaded!\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTrying to bypass authentication...")
        try:
            from huggingface_hub import login
            # Try without token first
            model = AutoModel.from_pretrained(repo_id, trust_remote_code=True, token=False)
            tokenizer = AutoTokenizer.from_pretrained(repo_id, token=False)
            print("✓ Model loaded!\n")
        except Exception as e2:
            print(f"✗ Still failed: {e2}")
            print("\nPlease try:")
            print("1. Check internet connection")
            print("2. Visit: https://huggingface.co/rasoultilburg/SocioCausaNet")
            return
    
    # Process in batches
    print(f"Processing {len(sentences)} sentences in batches...")
    batch_size = 32
    all_results = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(sentences) + batch_size - 1) // batch_size
        
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} sentences)...")
        
        try:
            results = model.predict(
                batch,
                tokenizer=tokenizer,
                rel_mode="neural",
                rel_threshold=0.5,
                cause_decision="cls+span"
            )
            
            # Add source info and filter causal
            for result in results:
                result['source'] = pdf_path
                if result.get('causal', False):
                    all_results.append(result)
        
        except Exception as e:
            print(f"  ✗ Error in batch {batch_num}: {e}")
            continue
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Total sentences processed: {len(sentences)}")
    print(f"Causal sentences found: {len(all_results)}")
    print(f"Total relations: {sum(len(r['relations']) for r in all_results)}\n")
    
    # Show some results
    if all_results:
        print("Sample causal relationships:")
        for i, result in enumerate(all_results[:10], 1):
            print(f"\n{i}. {result['text']}")
            for rel in result['relations']:
                print(f"   🔴 Cause: {rel['cause']}")
                print(f"   🟢 Effect: {rel['effect']}")
    
    # Save to JSON
    output_file = "causal_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*70}")

if __name__ == "__main__":
    pdf_file = "mapping phenomena relevant.pdf"
    process_pdf(pdf_file)
