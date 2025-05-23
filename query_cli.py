#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import sys
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv
import os

# Set tokenizers parallelism to False before importing any HuggingFace libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_weaviate_client():
    """Initialize and return a Weaviate client."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get environment variables
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    assert WEAVIATE_URL, "WEAVIATE_URL is not set"
    assert WEAVIATE_API_KEY, "WEAVIATE_API_KEY is not set"
    assert OPENAI_API_KEY, "OPENAI_API_KEY is not set"
    
    # Connect to Weaviate
    client = weaviate.connect_to_weaviate_cloud(
        WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        headers={
            "X-OpenAI-Api-Key": OPENAI_API_KEY
        }
    )
    
    # Check if Weaviate is ready
    if not client.is_ready():
        print("Error: Weaviate client is not ready. Check credentials and endpoint.")
        sys.exit(1)
    
    return client, OPENAI_API_KEY

# Cache for sentence transformer model - load only once
_model_cache = None

# Cache for reranker model - load only once
_reranker_model_cache = None

def get_model():
    """Get or create the sentence transformer model."""
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer("BAAI/bge-m3")
    return _model_cache

def get_reranker_model():
    """Get or create the reranker model."""
    global _reranker_model_cache
    if _reranker_model_cache is None:
        _reranker_model_cache = CrossEncoder("BAAI/bge-reranker-large")
    return _reranker_model_cache

def query_constitution(query_text, weaviate_client, openai_api_key, collection_name="ROC_Constitution_BG3_M3", limit=5):
    """
    Query the ROC Constitution using vector search and generate a response.
    
    Args:
        query_text: The user's query
        weaviate_client: The Weaviate client
        openai_api_key: OpenAI API key for generating a response
        collection_name: Name of the Weaviate collection
        limit: Number of chunks to retrieve
        
    Returns:
        The generated response
    """
    try:
        # Load embedding model (using cache)
        model = get_model()
        reranker = get_reranker_model()
        
        # Embed query using the SAME model as your document vectors
        query_vector = model.encode(query_text, normalize_embeddings=True)
        
        # Get the collection object
        collection = weaviate_client.collections.get(collection_name)
        
        # Run vector search using the current API
        results = collection.query.hybrid(
            query=query_text,
            vector=query_vector.tolist(),
            alpha=0.5,
            limit=limit,
            return_properties=["title", "content", "article", "chapter", "section"]
        )
        
        # Check if we got results
        if not results.objects:
            return "No results found. Please try a different query."
        
        pairs = [(query_text, object.properties.get("content")) for object in results.objects]
        scores = reranker.predict(pairs)
        # Reorder chunks based on reranker scores
        ranked_chunks = sorted(
            zip(results.objects, scores),
            key=lambda x: x[1],
            reverse=True
        )


        # Format the context with metadata
        context_chunks = []
        for obj, _ in ranked_chunks:
            properties = obj.properties
            chunk = f"--- {properties.get('title', 'Unknown')}"
            
            if properties.get('chapter'):
                chunk += f" | {properties.get('chapter')}"
            
            if properties.get('section') and properties.get('section') != properties.get('chapter'):
                chunk += f" | {properties.get('section')}"
                
            if properties.get('article'):
                chunk += f" | Article {properties.get('article')}"
                
            chunk += f" ---\n{properties.get('content', '')}"
            context_chunks.append(chunk)
        
        context = "\n\n".join(context_chunks)
        
        # Generate final answer using OpenAI with error handling
        try:
            openai_client = OpenAI(api_key=openai_api_key)
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant answering questions about the Republic of China (Taiwan) constitution. Answer the question based only on the provided context. If the answer is not in the context, say you don't know."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as api_error:
            if "insufficient_quota" in str(api_error) or "429" in str(api_error):
                return f"OpenAI API quota exceeded. Please check your billing details or try again later.\n\nHere's the raw context that was found:\n\n{context}"
            else:
                raise
            
    except Exception as e:
        print(f"Error in query_constitution: {str(e)}")
        raise

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Query the ROC Constitution')
    parser.add_argument('-q', '--query', help='The query to run')
    parser.add_argument('-l', '--limit', type=int, default=5, help='Number of chunks to retrieve (default: 5)')
    parser.add_argument('--local', action='store_true', help='Use local processing only (no OpenAI API call)')
    args = parser.parse_args()
    
    # Initialize Weaviate client
    weaviate_client, openai_api_key = setup_weaviate_client()
    
    # Get the query from args or prompt the user
    query = args.query
    if not query:
        print("Enter your query about the ROC Constitution (or type 'exit' to quit):")
        print("-" * 60)
    
    # Interactive loop
    while True:
        if not query:
            query = input("> ")
        
        # Check for exit command
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            weaviate_client.close()
            sys.exit(0)
        
        try:
            # Query the constitution
            print("\nQuerying...\n")
            
            if args.local:
                # Local mode - just retrieve the context without OpenAI processing
                model = get_model()
                query_vector = model.encode(query, normalize_embeddings=True)
                collection = weaviate_client.collections.get("ROC_Constitution_BG3_M3")
                results = collection.query.near_vector(
                    near_vector=query_vector.tolist(),
                    limit=args.limit,
                    return_properties=["title", "content", "article", "chapter", "section"]
                )
                
                print("-" * 60)
                for obj in results.objects:
                    properties = obj.properties
                    print(f"--- {properties.get('title', 'Unknown')} | {properties.get('chapter') or ''} | Article {properties.get('article') or ''} ---")
                    print(properties.get('content', ''))
                    print("-" * 30)
                print("-" * 60)
            else:
                # Normal mode with OpenAI processing
                result = query_constitution(query, weaviate_client, openai_api_key, limit=args.limit)
                print("-" * 60)
                print(result)
                print("-" * 60)
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        # Reset query for next iteration
        query = None

if __name__ == "__main__":
    main()