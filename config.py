# Example configuration file for the Streamlit app

# Model configuration
MODEL_CONFIG = {
    "repo_id": "rasoultilburg/SocioCausaNet",
    "rel_mode": "neural",
    "rel_threshold": 0.5,
    "cause_decision": "cls+span",
    "batch_size": 32  # Number of sentences to process at once
}

# Sentence filtering configuration
FILTER_CONFIG = {
    "min_chars": 15,
    "max_chars": 100,
    "min_alpha_ratio": 0.5  # Minimum ratio of alphabetic characters
}

# Vector search configuration
VECTOR_SEARCH_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",  # Sentence transformer model
    "similarity_threshold": 0.3,
    "default_top_k": 5
}

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "Causal Relationship Extractor",
    "page_icon": "🔍",
    "layout": "wide"
}
