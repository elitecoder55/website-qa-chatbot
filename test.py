#!/usr/bin/env python3
"""
Test script for Website QA Chatbot
Tests core functionality without running the full Streamlit app
"""

import sys
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np

def test_imports():
    """Test if all required packages are installed"""
    print("🧪 Testing imports...")
    try:
        import streamlit
        import beautifulsoup4
        import requests
        import sentence_transformers
        import faiss
        import numpy
        import langchain
        import openai
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_web_crawling():
    """Test basic web crawling functionality"""
    print("\n🧪 Testing web crawling...")
    try:
        # Test with Python docs (reliable and fast)
        url = "https://docs.python.org/3/"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        
        if len(text) > 100:
            print(f"✅ Successfully crawled {len(text)} characters from {url}")
            return True
        else:
            print("❌ Insufficient content extracted")
            return False
            
    except Exception as e:
        print(f"❌ Web crawling failed: {e}")
        return False

def test_embeddings():
    """Test embedding generation"""
    print("\n🧪 Testing embeddings...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        test_texts = [
            "This is a test sentence.",
            "Python is a programming language.",
            "Machine learning is fascinating."
        ]
        
        embeddings = model.encode(test_texts)
        
        if embeddings.shape == (3, 384):
            print(f"✅ Embeddings generated successfully: {embeddings.shape}")
            return True
        else:
            print(f"❌ Unexpected embedding shape: {embeddings.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False

def test_text_processing():
    """Test text chunking"""
    print("\n🧪 Testing text processing...")
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text = """
        Python is a high-level, general-purpose programming language.
        Its design philosophy emphasizes code readability with the use of significant indentation.
        Python is dynamically typed and garbage-collected.
        It supports multiple programming paradigms.
        """ * 5  # Repeat to get enough text
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        
        chunks = splitter.split_text(text)
        
        if len(chunks) > 0:
            print(f"✅ Text split into {len(chunks)} chunks")
            print(f"   First chunk length: {len(chunks[0])} chars")
            return True
        else:
            print("❌ No chunks created")
            return False
            
    except Exception as e:
        print(f"❌ Text processing failed: {e}")
        return False

def test_vector_search():
    """Test FAISS vector search"""
    print("\n🧪 Testing vector search...")
    try:
        import faiss
        
        # Create dummy embeddings
        dimension = 384
        n_vectors = 100
        
        embeddings = np.random.random((n_vectors, dimension)).astype('float32')
        
        # Build index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Search
        query = np.random.random((1, dimension)).astype('float32')
        distances, indices = index.search(query, k=3)
        
        if len(indices[0]) == 3:
            print(f"✅ Vector search successful, found {len(indices[0])} results")
            return True
        else:
            print("❌ Unexpected search results")
            return False
            
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Website QA Chatbot - Component Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Web Crawling", test_web_crawling),
        ("Embeddings", test_embeddings),
        ("Text Processing", test_text_processing),
        ("Vector Search", test_vector_search)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The application should work correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
