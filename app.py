import streamlit as st
import json
import os
from transformers import AutoModel, AutoTokenizer
from utils.pdf_processor import extract_text_from_pdf, split_into_sentences, filter_sentences
from utils.vector_search import VectorSearch
import torch

# Set page config
st.set_page_config(
    page_title="Causal Relationship Extractor",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'vector_search' not in st.session_state:
    st.session_state.vector_search = None
if 'causal_results' not in st.session_state:
    st.session_state.causal_results = []

@st.cache_resource
def load_model():
    """Load the SocioCausaNet model and tokenizer"""
    repo_id = "rasoultilburg/SocioCausaNet"
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return model, tokenizer

def process_pdf_files(uploaded_files, model, tokenizer, min_chars=15, max_chars=100):
    """Process uploaded PDF files and extract causal relationships"""
    all_results = []
    all_sentences = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate total sentences first for accurate progress
    total_files = len(uploaded_files)
    
    for file_idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"📄 Processing {uploaded_file.name} ({file_idx + 1}/{total_files})...")
        
        # Extract text from PDF
        text = extract_text_from_pdf(uploaded_file)
        
        # Split into sentences
        sentences = split_into_sentences(text)
        
        # Filter sentences based on character count
        filtered_sentences = filter_sentences(sentences, min_chars, max_chars)
        
        status_text.text(f"🔍 Analyzing {len(filtered_sentences)} sentences from {uploaded_file.name}...")
        
        # Process sentences in batches
        batch_size = 32
        num_batches = (len(filtered_sentences) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(filtered_sentences), batch_size):
            batch = filtered_sentences[batch_idx:batch_idx+batch_size]
            current_batch = (batch_idx // batch_size) + 1
            
            # Update progress within current file
            file_progress = file_idx / total_files
            batch_progress = (current_batch / num_batches) / total_files
            total_progress = file_progress + batch_progress
            progress_bar.progress(min(total_progress, 1.0))
            
            status_text.text(f"🔍 {uploaded_file.name} - Batch {current_batch}/{num_batches} ({file_idx + 1}/{total_files} files)")
            
            # Get predictions from the model
            results = model.predict(
                batch,
                tokenizer=tokenizer,
                rel_mode="neural",
                rel_threshold=0.5,
                cause_decision="cls+span"
            )
            
            # Add source information
            for result in results:
                result['source'] = uploaded_file.name
                if result.get('causal', False):
                    all_results.append(result)
            
            all_sentences.extend(batch)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    return all_results, all_sentences

def main():
    st.title("🔍 Causal Relationship Extractor")
    st.markdown("Upload PDF files to extract causal relationships and ask questions about causes and effects.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Filter settings
        st.subheader("Sentence Filters")
        min_chars = st.slider("Minimum characters", 10, 50, 15)
        max_chars = st.slider("Maximum characters", 50, 200, 100)
        
        st.divider()
        
        # Load model button
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading SocioCausaNet model..."):
                st.session_state.model, st.session_state.tokenizer = load_model()
                st.success("Model loaded successfully!")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "❓ Question Answering", "📊 Results"])
    
    with tab1:
        st.header("Upload PDF Files")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to extract causal relationships"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) uploaded")
            
            if st.button("🚀 Process Files", type="primary"):
                if st.session_state.model is None:
                    st.error("⚠️ Please load the model first from the sidebar!")
                else:
                    with st.spinner("Processing PDF files..."):
                        results, sentences = process_pdf_files(
                            uploaded_files,
                            st.session_state.model,
                            st.session_state.tokenizer,
                            min_chars,
                            max_chars
                        )
                        
                        st.session_state.causal_results = results
                        
                        # Initialize vector search
                        st.session_state.vector_search = VectorSearch()
                        st.session_state.vector_search.build_index(results)
                        
                        st.success(f"✅ Found {len(results)} causal relationships!")
                        
                        # Download JSON
                        json_str = json.dumps(results, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📥 Download Results (JSON)",
                            data=json_str,
                            file_name="causal_relationships.json",
                            mime="application/json"
                        )
    
    with tab2:
        st.header("Question Answering")
        
        if st.session_state.vector_search is None or not st.session_state.causal_results:
            st.warning("⚠️ Please process PDF files first in the 'Upload & Extract' tab")
        else:
            st.markdown("Ask questions about causes and effects from your uploaded documents.")
            
            query_type = st.radio(
                "Query type:",
                ["Find Effects", "Find Causes", "General Search"],
                horizontal=True,
                help="Find Effects: What does X cause? | Find Causes: What causes X? | General: Semantic search"
            )
            
            # Example queries based on type
            if query_type == "Find Effects":
                placeholder = "e.g., insomnia, smoking, pollution, stress"
            elif query_type == "Find Causes":
                placeholder = "e.g., depression, cancer, climate change, inflation"
            else:
                placeholder = "e.g., health impacts of sleep deprivation"
            
            query = st.text_input(
                "Enter your query:",
                placeholder=placeholder,
                help="Type a natural language question or just keywords"
            )
            
            col1, col2, col3 = st.columns([3, 1, 2])
            with col1:
                top_k = st.slider("Number of results", 1, 20, 5)
            with col2:
                similarity_threshold = st.slider(
                    "Similarity threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.25,
                    step=0.01,
                    help="Minimum similarity score for a result to be shown. Higher = stricter."
                )
            
            if query:
                if st.button("🔎 Search", type="primary", use_container_width=True):
                    with st.spinner("🔍 Searching through causal relationships..."):
                        # Clean and process query
                        query_clean = query.lower().strip()
                        
                        # Remove common question words for better matching
                        question_words = ['what', 'are', 'the', 'of', 'is', 'does', 'cause', 'effect', 'causes', 'effects', '?']
                        query_terms = [word for word in query_clean.split() if word not in question_words]
                        
                        # Use cleaned query if available, otherwise use original
                        search_query = ' '.join(query_terms) if query_terms else query
                        
                        results = st.session_state.vector_search.search(
                            search_query,
                            query_type.lower(),
                            top_k=top_k,
                            similarity_threshold=similarity_threshold
                        )
                        
                        if results:
                            st.success(f"✅ Found {len(results)} relevant results")
                            st.divider()
                            
                            for i, result in enumerate(results, 1):
                                with st.expander(f"📄 Result {i} - Score: {result['score']:.3f}", expanded=(i <= 3)):
                                    st.markdown(f"**Source:** `{result['source']}`")
                                    st.markdown(f"**Sentence:** *{result['text']}*")
                                    
                                    if result['relations']:
                                        st.markdown("**Causal Relations:**")
                                        for rel in result['relations']:
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.markdown(f"🔴 **Cause:** {rel['cause']}")
                                            with col2:
                                                st.markdown(f"🟢 **Effect:** {rel['effect']}")
                        else:
                            st.info("❌ No results found. Try:\n- Different keywords\n- Different query type\n- More general terms")
    
    with tab3:
        st.header("All Extracted Relationships")
        
        if not st.session_state.causal_results:
            st.info("No results yet. Process some PDF files first!")
        else:
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            total_relations = sum(len(r['relations']) for r in st.session_state.causal_results)
            unique_sources = len(set(r['source'] for r in st.session_state.causal_results))
            
            col1.metric("Total Causal Sentences", len(st.session_state.causal_results))
            col2.metric("Total Relations", total_relations)
            col3.metric("Sources", unique_sources)
            
            st.divider()
            
            # Display results
            for i, result in enumerate(st.session_state.causal_results, 1):
                with st.expander(f"Sentence {i} - {result['source']}"):
                    st.markdown(f"**Text:** {result['text']}")
                    
                    if result['relations']:
                        st.markdown("**Relations:**")
                        for rel in result['relations']:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"🔴 **Cause:** {rel['cause']}")
                            with col2:
                                st.markdown(f"🟢 **Effect:** {rel['effect']}")

if __name__ == "__main__":
    main()
