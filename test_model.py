"""
Simple test script to verify the SocioCausaNet model works correctly
Run this before using the Streamlit app to ensure everything is set up properly
"""

from transformers import AutoModel, AutoTokenizer
import json

def test_model():
    """Test the SocioCausaNet model with sample sentences"""
    print("=" * 60)
    print("Testing SocioCausaNet Model")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model and tokenizer...")
    try:
        repo_id = "rasoultilburg/SocioCausaNet"
        model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    # Test sentences
    print("\n2. Testing with sample sentences...")
    sentences = [
        "Insomnia causes depression and a lack of concentration in children.",
        "Due to the new regulations, the company's profits declined sharply.",
        "The sun rises in the east."  # Non-causal example
    ]
    
    try:
        # Get predictions
        results = model.predict(
            sentences,
            tokenizer=tokenizer,
            rel_mode="neural",
            rel_threshold=0.5,
            cause_decision="cls+span"
        )
        print("✓ Model predictions completed!")
        
        # Print results
        print("\n3. Results:")
        print("-" * 60)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print("-" * 60)
        
        # Verify results
        print("\n4. Verification:")
        causal_count = sum(1 for r in results if r.get('causal', False))
        relation_count = sum(len(r.get('relations', [])) for r in results)
        
        print(f"   - Total sentences: {len(sentences)}")
        print(f"   - Causal sentences: {causal_count}")
        print(f"   - Total relations found: {relation_count}")
        
        if causal_count >= 2 and relation_count >= 3:
            print("\n✓ Model is working correctly!")
            return True
        else:
            print("\n⚠ Unexpected results. Please check the model.")
            return False
            
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: Everything is set up correctly!")
        print("You can now run the Streamlit app with: streamlit run app.py")
    else:
        print("FAILURE: Please check the error messages above.")
    print("=" * 60)
