#!/usr/bin/env python
# Script to check the contents of the ChromaDB knowledge base

import os
import json
import sys
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def main():
    # Path to the Chroma database
    chroma_dir = "./chroma_db"
    
    if not os.path.exists(chroma_dir):
        print(f"Error: ChromaDB directory not found at {chroma_dir}")
        print("The knowledge base may not have been initialized yet.")
        return
        
    # Get OpenAI API key from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        # Try to get from .env file
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        openai_api_key = line.strip().split("=", 1)[1]
                        break
        except:
            pass
            
    if not openai_api_key:
        print("OpenAI API key not found. Please provide it:")
        openai_api_key = input("OpenAI API Key: ")
        
    # Initialize embedding function
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-ada-002", 
        openai_api_key=openai_api_key
    )
    
    # Connect to the existing database
    try:
        db = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embedding_function,
            collection_name="penelope_knowledge"
        )
        
        count = db._collection.count()
        print(f"\n=== Knowledge Base Statistics ===")
        print(f"Total documents: {count}")
        
        if count == 0:
            print("\nKnowledge base is empty. No documents have been added yet.")
            return
            
        # Sample some documents
        print(f"\n=== Sample Documents ({min(5, count)} of {count}) ===")
        results = db.get(limit=min(5, count))
        
        # Count document types
        doc_types = {}
        sources = {}
        
        for i, doc in enumerate(results['documents']):
            print(f"\nDocument {i+1}:")
            print(f"Content (preview): {doc[:200]}...")
            
            metadata = results['metadatas'][i]
            print(f"Metadata: {json.dumps(metadata, indent=2)}")
            
            # Track document types and sources
            doc_type = metadata.get('file_type', metadata.get('type', 'unknown'))
            if doc_type in doc_types:
                doc_types[doc_type] += 1
            else:
                doc_types[doc_type] = 1
                
            source = metadata.get('source', 'unknown')
            if source in sources:
                sources[source] += 1
            else:
                sources[source] = 1
        
        # Print summary
        if doc_types:
            print("\n=== Document Types ===")
            for t, c in doc_types.items():
                print(f"- {t}: {c}")
                
        if sources:
            print("\n=== Document Sources (sample) ===")
            for src, count in list(sources.items())[:10]:
                print(f"- {src}")
            
            if len(sources) > 10:
                print(f"- ... and {len(sources) - 10} more")
                
        # Perform a test query
        query = "bitcoin"
        print(f"\n=== Test Query: '{query}' ===")
        results = db.similarity_search_with_relevance_scores(query, k=2)
        
        for doc, score in results:
            print(f"\nRelevance Score: {score:.4f}")
            print(f"Content (preview): {doc.page_content[:200]}...")
            print(f"Metadata: {json.dumps(doc.metadata, indent=2)}")
            
    except Exception as e:
        print(f"Error accessing knowledge base: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 