import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import time

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_PAGES = 10  # Limit crawling to prevent overload
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient
VECTOR_DB_PATH = "vector_store"

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'indexed_url' not in st.session_state:
    st.session_state.indexed_url = None


class WebsiteCrawler:
    """Handles web crawling and content extraction"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.visited_urls = set()
        self.domain = urlparse(base_url).netloc
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to same domain"""
        try:
            parsed = urlparse(url)
            return (
                parsed.netloc == self.domain and
                parsed.scheme in ['http', 'https'] and
                url not in self.visited_urls
            )
        except:
            return False
    
    def clean_text(self, soup: BeautifulSoup) -> str:
        """Extract and clean text from HTML"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                            'aside', 'meta', 'link', 'noscript', 'iframe',
                            'button', 'input', 'form']):
            element.decompose()
        
        # Remove ads and social media elements
        for element in soup.find_all(class_=re.compile(r'(ad|advertisement|social|share|cookie|banner)', re.I)):
            element.decompose()
        
        # Extract text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def crawl(self, max_pages: int = MAX_PAGES) -> List[Dict]:
        """Crawl website and extract content"""
        to_visit = [self.base_url]
        documents = []
        
        while to_visit and len(self.visited_urls) < max_pages:
            url = to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
                
            try:
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; WebCrawler/1.0)'
                })
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract content
                title = soup.find('title')
                title_text = title.get_text().strip() if title else url
                
                content = self.clean_text(soup)
                
                if content and len(content) > 100:  # Skip pages with minimal content
                    documents.append({
                        'url': url,
                        'title': title_text,
                        'content': content
                    })
                
                self.visited_urls.add(url)
                
                # Find new URLs to crawl
                for link in soup.find_all('a', href=True):
                    new_url = urljoin(url, link['href'])
                    if self.is_valid_url(new_url):
                        to_visit.append(new_url)
                
                time.sleep(0.5)  # Be polite to servers
                
            except Exception as e:
                st.warning(f"Error crawling {url}: {str(e)}")
                continue
        
        return documents


class VectorStore:
    """Handles embedding generation and vector storage using FAISS"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        
    def create_chunks(self, documents: List[Dict]) -> List[Dict]:
        """Split documents into semantic chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        for doc in documents:
            splits = text_splitter.split_text(doc['content'])
            for i, split in enumerate(splits):
                chunks.append({
                    'text': split,
                    'url': doc['url'],
                    'title': doc['title'],
                    'chunk_id': f"{doc['url']}_{i}"
                })
        
        return chunks
    
    def build_index(self, chunks: List[Dict]):
        """Generate embeddings and build FAISS index"""
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant chunks"""
        if self.index is None:
            return []
        
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append({
                    **self.chunks[idx],
                    'score': float(distance)
                })
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk"""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        with open(os.path.join(path, 'chunks.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)
    
    def load(self, path: str):
        """Load vector store from disk"""
        self.index = faiss.read_index(os.path.join(path, 'index.faiss'))
        with open(os.path.join(path, 'chunks.pkl'), 'rb') as f:
            self.chunks = pickle.load(f)


class ChatBot:
    """Handles question answering with LLM"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    def generate_answer(self, question: str, context_chunks: List[Dict], 
                       conversation_history: List[Dict]) -> str:
        """Generate answer using retrieved context"""
        
        if not context_chunks:
            return "The answer is not available on the provided website."
        
        # Prepare context
        context = "\n\n".join([
            f"Source: {chunk['title']}\nContent: {chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Prepare conversation history
        history_text = ""
        if conversation_history:
            history_text = "Previous conversation:\n"
            for msg in conversation_history[-3:]:  # Last 3 exchanges
                history_text += f"{msg['role']}: {msg['content']}\n"
        
        # Create prompt
        system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided website content.

STRICT RULES:
1. Give a DIRECT, CONCISE answer to the question - do NOT dump all the context
2. Answer in 2-3 sentences maximum unless asked for more detail
3. Answer ONLY using information from the provided context
4. If the answer is not in the context, respond EXACTLY with: "The answer is not available on the provided website."
5. Do not use any external knowledge or make assumptions
6. Do NOT just copy-paste the entire context - extract the relevant information and answer the specific question"""

        user_prompt = f"""{history_text}

Context from website:
{context}

Question: {question}

Important: Give a DIRECT, SPECIFIC answer to the question above. Do not dump all the context. Answer in 2-3 sentences."""

        try:
            if self.client:
                # Use OpenAI API
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Very low for factual answers
                    max_tokens=200  # Limit to force conciseness
                )
                return response.choices[0].message.content
            else:
                # Fallback: Simple extraction-based answer
                return self._simple_answer(question, context_chunks)
                
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return self._simple_answer(question, context_chunks)
    
    def _simple_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """Simple fallback answer when LLM is not available"""
        if not context_chunks:
            return "The answer is not available on the provided website."
        
        # Return a concise answer from the most relevant chunk
        best_chunk = context_chunks[0]['text']
        
        # Try to extract the most relevant sentence
        sentences = best_chunk.split('. ')
        relevant_part = sentences[0] if sentences else best_chunk
        
        # Limit to reasonable length
        if len(relevant_part) > 300:
            relevant_part = relevant_part[:300] + "..."
        
        answer = f"{relevant_part}\n\nSource: {context_chunks[0]['title']}"
        return answer


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Website QA Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Website-Based QA Chatbot")
    st.markdown("*Ask questions about any website's content using AI-powered embeddings*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "OpenAI API Key (Optional)",
            type="password",
            help="For better answers. Leave empty to use fallback mode."
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.info(f"""
        **Embedding Model:** {EMBEDDING_MODEL}
        - Fast and efficient
        - 384 dimensions
        - Optimized for semantic search
        
        **LLM:** GPT-3.5-turbo (if API key provided)
        - Cost-effective
        - Good balance of speed and quality
        
        **Vector DB:** FAISS
        - In-memory, fast similarity search
        - No external dependencies
        """)
        
        if st.session_state.indexed_url:
            st.success(f"✅ Indexed: {st.session_state.indexed_url}")
            if st.button("Clear Index"):
                st.session_state.vector_store = None
                st.session_state.chunks = []
                st.session_state.indexed_url = None
                st.session_state.conversation_history = []
                st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("📥 Index Website")
        
        url = st.text_input(
            "Enter Website URL",
            placeholder="https://example.com",
            help="Enter the URL of the website you want to analyze"
        )
        
        if st.button("🔍 Crawl & Index", type="primary"):
            if not url:
                st.error("Please enter a valid URL")
                return
            
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            try:
                with st.spinner("🕷️ Crawling website..."):
                    crawler = WebsiteCrawler(url)
                    documents = crawler.crawl(max_pages=MAX_PAGES)
                
                if not documents:
                    st.error("No content found on the website. Please check the URL.")
                    return
                
                st.success(f"✅ Crawled {len(documents)} pages")
                
                with st.spinner("✂️ Creating chunks..."):
                    vector_store = VectorStore()
                    chunks = vector_store.create_chunks(documents)
                
                st.success(f"✅ Created {len(chunks)} text chunks")
                
                with st.spinner("🧮 Generating embeddings..."):
                    vector_store.build_index(chunks)
                
                st.session_state.vector_store = vector_store
                st.session_state.chunks = chunks
                st.session_state.indexed_url = url
                st.session_state.conversation_history = []
                
                st.success("✅ Website indexed successfully!")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        st.header("💬 Ask Questions")
        
        if st.session_state.vector_store is None:
            st.info("👈 Please index a website first to start asking questions")
        else:
            # Display conversation history
            for msg in st.session_state.conversation_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # Chat input
            if question := st.chat_input("Ask a question about the website..."):
                # Display user message
                with st.chat_message("user"):
                    st.write(question)
                
                # Add to history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": question
                })
                
                # Search for relevant chunks
                with st.spinner("🔍 Searching..."):
                    results = st.session_state.vector_store.search(question, k=3)
                
                # Generate answer
                with st.spinner("💭 Generating answer..."):
                    chatbot = ChatBot(api_key=api_key if api_key else None)
                    answer = chatbot.generate_answer(
                        question,
                        results,
                        st.session_state.conversation_history[:-1]
                    )
                
                # Display assistant message
                with st.chat_message("assistant"):
                    st.write(answer)
                    
                    # Show sources
                    if results and answer != "The answer is not available on the provided website.":
                        with st.expander("📚 View Sources"):
                            for i, result in enumerate(results, 1):
                                st.markdown(f"**Source {i}:** [{result['title']}]({result['url']})")
                                st.caption(result['text'][:200] + "...")
                                st.markdown("---")
                
                # Add to history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": answer
                })
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit, Sentence Transformers, FAISS, and LangChain | "
        "Humanli.ai Assignment"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
