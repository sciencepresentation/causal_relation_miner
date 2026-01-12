"""
Demo script with sample text data
Use this to test the causal extraction without needing PDF files
"""

from transformers import AutoModel, AutoTokenizer
import json
from utils.vector_search import VectorSearch

# Sample text from different domains
SAMPLE_TEXTS = {
    "health": """
    Smoking causes lung cancer and heart disease. Lack of exercise leads to obesity.
    Depression can result from chronic stress. Insomnia causes reduced cognitive function.
    Poor diet contributes to diabetes. Regular meditation reduces anxiety levels.
    """,
    
    "economics": """
    High inflation reduces purchasing power. Interest rate hikes slow economic growth.
    Trade barriers cause reduced international commerce. Tax cuts stimulate consumer spending.
    Recession leads to increased unemployment. Investment in infrastructure creates jobs.
    """,
    
    "environment": """
    Deforestation causes soil erosion. Carbon emissions lead to global warming.
    Pollution damages marine ecosystems. Renewable energy reduces carbon footprint.
    Overfishing depletes ocean populations. Urbanization destroys natural habitats.
    """,
    
    "technology": """
    Artificial intelligence improves productivity. Automation reduces manual labor costs.
    Cybersecurity breaches compromise data privacy. Cloud computing enables remote work.
    Social media affects mental health. Digital transformation increases efficiency.
    """
}

def run_demo():
    """Run a demonstration of the causal extraction system"""
    print("=" * 70)
    print("CAUSAL RELATIONSHIP EXTRACTION - DEMO")
    print("=" * 70)
    
    # Load model
    print("\n📥 Loading SocioCausaNet model...")
    repo_id = "rasoultilburg/SocioCausaNet"
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    print("✓ Model loaded!\n")
    
    all_results = []
    
    # Process each domain
    for domain, text in SAMPLE_TEXTS.items():
        print(f"\n{'='*70}")
        print(f"DOMAIN: {domain.upper()}")
        print(f"{'='*70}")
        
        # Split into sentences and clean
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
        
        print(f"Processing {len(sentences)} sentences...\n")
        
        # Get predictions
        results = model.predict(
            sentences,
            tokenizer=tokenizer,
            rel_mode="neural",
            rel_threshold=0.5,
            cause_decision="cls+span"
        )
        
        # Add domain info
        for result in results:
            result['domain'] = domain
            if result.get('causal', False):
                all_results.append(result)
        
        # Display results
        causal_count = sum(1 for r in results if r.get('causal', False))
        print(f"📊 Found {causal_count} causal sentences\n")
        
        for i, result in enumerate(results, 1):
            if result.get('causal', False):
                print(f"  {i}. 📝 {result['text']}")
                for rel in result.get('relations', []):
                    print(f"     🔴 Cause: {rel['cause']}")
                    print(f"     🟢 Effect: {rel['effect']}")
                    print()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    total_relations = sum(len(r['relations']) for r in all_results)
    print(f"Total causal sentences: {len(all_results)}")
    print(f"Total cause-effect relations: {total_relations}")
    
    # Save results
    output_file = "demo_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Test vector search
    print(f"\n{'='*70}")
    print("TESTING VECTOR SEARCH")
    print(f"{'='*70}")
    
    print("\n🔍 Building search index...")
    vector_search = VectorSearch()
    vector_search.build_index(all_results)
    print("✓ Index built!\n")
    
    # Sample queries
    queries = [
        ("What are the effects of smoking?", "find effects"),
        ("What causes global warming?", "find causes"),
        ("economic impacts", "general search")
    ]
    
    for query, query_type in queries:
        print(f"\n❓ Query: '{query}' (Type: {query_type})")
        print("-" * 70)
        
        search_results = vector_search.search(query, query_type, top_k=3)
        
        if search_results:
            for i, result in enumerate(search_results, 1):
                print(f"\n  Result {i} (Score: {result['score']:.3f})")
                print(f"  Domain: {result['domain']}")
                print(f"  Text: {result['text']}")
                if result['relations']:
                    for rel in result['relations']:
                        print(f"    • {rel['cause']} → {rel['effect']}")
        else:
            print("  No results found")
    
    print(f"\n{'='*70}")
    print("DEMO COMPLETE!")
    print(f"{'='*70}")
    print(f"\n💡 Next steps:")
    print(f"   1. Check {output_file} for full results")
    print(f"   2. Run: streamlit run app.py")
    print(f"   3. Upload your own PDF files!")
    print()

if __name__ == "__main__":
    run_demo()
