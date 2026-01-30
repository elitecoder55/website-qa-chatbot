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
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

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
        """Split documents into semantic chunks using LangChain"""
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


def generate_answer(question: str, context_chunks: List[Dict], groq_api_key: str = None) -> str:
    """Generate answer using LangChain with Llama model via Groq"""
    
    if not context_chunks:
        return "The answer is not available on the provided website."
    
    # Prepare context from chunks
    context = "\n\n".join([
        f"Source: {chunk['title']}\nContent: {chunk['text']}"
        for chunk in context_chunks
    ])
    
    # If Groq API key is provided, use Llama via LangChain
    if groq_api_key:
        try:
            # Initialize Groq ChatLLM with Llama
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile",  # Llama 3.3 70B
                temperature=0.1,
                max_tokens=300
            )
            
            # Create messages
            system_message = SystemMessage(content="""You are a helpful assistant that answers questions based ONLY on the provided website content.

STRICT RULES:
1. Give a DIRECT, CONCISE answer to the question - do NOT dump all the context
2. Answer in 2-3 sentences maximum unless asked for more detail
3. Answer ONLY using information from the provided context
4. If the answer is not in the context, respond EXACTLY with: "The answer is not available on the provided website."
5. Do not use any external knowledge or make assumptions
6. Do NOT just copy-paste the entire context - extract the relevant information and answer the specific question""")
            
            human_message = HumanMessage(content=f"""Context from website:
{context}

Question: {question}

Important: Give a DIRECT, SPECIFIC answer to the question above. Do not dump all the context. Answer in 2-3 sentences.""")
            
            # Get response
            response = llm.invoke([system_message, human_message])
            return response.content
            
        except Exception as e:
            st.error(f"Error with Llama model: {str(e)}")
            return smart_extraction_fallback(question, context_chunks)
    else:
        # Fallback to smart extraction if no API key
        return smart_extraction_fallback(question, context_chunks)


def smart_extraction_fallback(question: str, context_chunks: List[Dict]) -> str:
    """Fallback extraction method when no API key"""
    if not context_chunks:
        return "The answer is not available on the provided website."
    
    # Combine all chunks
    all_text = " ".join([chunk['text'] for chunk in context_chunks[:3]])
    
    # Split into sentences
    sentences = []
    for sent in all_text.replace('!', '.').replace('?', '.').split('.'):
        clean = sent.strip()
        if len(clean) > 30:  # Only keep substantial sentences
            sentences.append(clean)
    
    if not sentences:
        return "The answer is not available on the provided website."
    
    # Extract keywords from question
    question_lower = question.lower()
    stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who', 'does', 'do', 'can', 'could', 'would', 'should', 'tell', 'me', 'about'}
    question_words = [w for w in question_lower.split() if w not in stop_words and len(w) > 2]
    
    # Score each sentence by keyword matches
    scored = []
    for sent in sentences:
        sent_lower = sent.lower()
        score = sum(2 if word in sent_lower else 0 for word in question_words)
        
        # Bonus for sentences at the beginning (often contain definitions)
        if sentences.index(sent) < 3:
            score += 1
            
        if score > 0:
            scored.append((score, sent))
    
    if not scored:
        # No good match - just use first sentences
        answer = ". ".join(sentences[:2]) + "."
    else:
        # Sort by score and take top 2-3 sentences
        scored.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s[1] for s in scored[:3]]
        answer = ". ".join(top_sentences) + "."
    
    # Limit length
    if len(answer) > 500:
        answer = answer[:500] + "..."
    
    # Add source
    source_info = f"\n\n📚 Source: {context_chunks[0]['title']}"
    
    return answer + source_info


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
        
        groq_api_key = st.text_input(
            "Groq API Key (Optional)",
            type="password",
            help="Get free API key from https://console.groq.com - Uses Llama 3.3 70B model"
        )
        
        if groq_api_key:
            st.success("✅ Using Llama 3.3 70B via Groq!")
        else:
            st.info("💡 Add Groq API key for AI-powered answers with Llama")
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.info(f"""
        **Embedding Model:** {EMBEDDING_MODEL}
        - Fast and efficient
        - 384 dimensions
        - Optimized for semantic search
        
        **LLM:** Llama 3.3 70B (via Groq)
        - Open-source model by Meta
        - Uses LangChain framework
        - FREE API with Groq (30 req/min)
        - Fallback: Smart extraction
        
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
                    answer = generate_answer(question, results, groq_api_key if groq_api_key else None)
                
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
