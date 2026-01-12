from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Dict, Any

class VectorSearch:
    """Vector search class for semantic search over causal relationships"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the vector search with a sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.causal_results = []
        self.embeddings = None
    
    def build_index(self, causal_results: List[Dict[str, Any]]):
        """
        Build FAISS index from causal relationships
        
        Args:
            causal_results: List of causal relationship dictionaries
        """
        self.causal_results = causal_results
        
        if not causal_results:
            return
        
        # Create texts for embedding
        texts = []
        for result in causal_results:
            # Combine sentence text with cause-effect pairs for better search
            text = result['text'] + " "
            for rel in result.get('relations', []):
                text += f"Cause: {rel['cause']} Effect: {rel['effect']} "
            texts.append(text)
        
        # Generate embeddings
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
    
    def search(self, query: str, query_type: str = "general search", top_k: int = 5, similarity_threshold: float = 0.25):
        """
        Search for relevant causal relationships
        
        Args:
            query: Search query string
            query_type: Type of query - "find effects", "find causes", or "general search"
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score for a result to be included
        
        Returns:
            list: List of relevant causal relationships with scores
        """
        if self.index is None or not self.causal_results:
            return []
        
        # Clean query
        query = query.strip().lower()
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search with more candidates for filtering
        search_k = min(top_k * 3, len(self.causal_results))
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Process results based on query type
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < similarity_threshold:
                continue
            result = self.causal_results[idx].copy()
            result['score'] = float(score)
            
            # Filter based on query type
            if query_type == "find effects":
                matching_relations = []
                for rel in result.get('relations', []):
                    cause_lower = rel['cause'].lower()
                    if query in cause_lower or cause_lower in query:
                        matching_relations.append(rel)
                    else:
                        cause_similarity = self._text_similarity(query, rel['cause'])
                        if cause_similarity > similarity_threshold:
                            matching_relations.append(rel)
                if matching_relations:
                    result['relations'] = matching_relations
                    results.append(result)
            elif query_type == "find causes":
                matching_relations = []
                for rel in result.get('relations', []):
                    effect_lower = rel['effect'].lower()
                    if query in effect_lower or effect_lower in query:
                        matching_relations.append(rel)
                    else:
                        effect_similarity = self._text_similarity(query, rel['effect'])
                        if effect_similarity > similarity_threshold:
                            matching_relations.append(rel)
                if matching_relations:
                    result['relations'] = matching_relations
                    results.append(result)
            else:  # General search
                if score > similarity_threshold:
                    results.append(result)
            if len(results) >= top_k:
                break
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            float: Similarity score (0-1)
        """
        embeddings = self.model.encode([text1.lower(), text2.lower()])
        faiss.normalize_L2(embeddings)
        
        # Compute cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)
    
    def get_all_causes(self) -> List[str]:
        """Get all unique causes from the causal relationships"""
        causes = set()
        for result in self.causal_results:
            for rel in result.get('relations', []):
                causes.add(rel['cause'])
        return sorted(list(causes))
    
    def get_all_effects(self) -> List[str]:
        """Get all unique effects from the causal relationships"""
        effects = set()
        for result in self.causal_results:
            for rel in result.get('relations', []):
                effects.add(rel['effect'])
        return sorted(list(effects))
    
    def search_by_cause(self, cause: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for all effects of a given cause
        
        Args:
            cause: The cause to search for
            top_k: Number of results to return
        
        Returns:
            list: List of relevant effects
        """
        return self.search(cause, query_type="find effects", top_k=top_k)
    
    def search_by_effect(self, effect: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for all causes of a given effect
        
        Args:
            effect: The effect to search for
            top_k: Number of results to return
        
        Returns:
            list: List of relevant causes
        """
        return self.search(effect, query_type="find causes", top_k=top_k)
