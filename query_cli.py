#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import sys
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv

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
    
    # Validate required environment variables

    
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
    # Load embedding model
    model = SentenceTransformer("BAAI/bge-m3")
    
    # Embed query using the SAME model as your document vectors
    query_vector = model.encode(query_text, normalize_embeddings=True)
    
    # Run vector search manually
    results = weaviate_client.query.get(
        collection_name, 
        ["title", "content", "article", "chapter", "section"]
    ).with_near_vector(
        {"vector": query_vector.tolist()}
    ).with_limit(limit).do()
    
    # Extract top texts for context
    result_data = results.get("data", {}).get("Get", {}).get(collection_name, [])
    
    if not result_data:
        return "No results found. Please try a different query."
    
    # Format the context with metadata
    context_chunks = []
    for doc in result_data:
        chunk = f"--- {doc.get('title', 'Unknown')}"
        
        if doc.get('chapter'):
            chunk += f" | {doc.get('chapter')}"
        
        if doc.get('section') and doc.get('section') != doc.get('chapter'):
            chunk += f" | {doc.get('section')}"
            
        if doc.get('article'):
            chunk += f" | Article {doc.get('article')}"
            
        chunk += f" ---\n{doc.get('content', '')}"
        context_chunks.append(chunk)
    
    context = "\n\n".join(context_chunks)
    
    # Generate final answer using OpenAI
    openai_client = OpenAI(api_key=openai_api_key)
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering questions about the Republic of China (Taiwan) constitution. Answer the question based only on the provided context. If the answer is not in the context, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
        ]
    )
    
    return response.choices[0].message.content

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Query the ROC Constitution')
    parser.add_argument('-q', '--query', help='The query to run')
    parser.add_argument('-l', '--limit', type=int, default=5, help='Number of chunks to retrieve (default: 5)')
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
            break
        
        try:
            # Query the constitution
            print("\nQuerying...\n")
            result = query_constitution(query, weaviate_client, openai_api_key, limit=args.limit)
            
            # Print the result
            print("-" * 60)
            print(result)
            print("-" * 60)
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        # Reset query for next iteration
        query = None

if __name__ == "__main__":
    main()