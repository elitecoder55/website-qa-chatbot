# 🤖 Website-Based QA Chatbot

An AI-powered question-answering chatbot that crawls websites, creates embeddings from content, and answers questions based strictly on the website's information.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Setup Instructions](#setup-instructions)


## 🎯 Overview

This project implements an intelligent chatbot system that:
1. Accepts a website URL from users
2. Crawls and extracts meaningful content from the website
3. Processes and chunks the content into semantic segments
4. Generates embeddings and stores them in a vector database
5. Allows users to ask questions and provides accurate answers based solely on website content
6. Maintains short-term conversational context

## 🏗️ Architecture

```
┌─────────────────┐
│   User Input    │
│   (Website URL) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Web Crawler                │
│  - URL Validation               │
│  - Content Extraction           │
│  - HTML Cleaning                │
│  - Duplicate Removal            │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Text Processing               │
│  - Text Normalization           │
│  - Semantic Chunking            │
│  - Metadata Preservation        │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Embedding Generation          │
│  - Sentence Transformers        │
│  - Vector Creation              │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Vector Database (FAISS)       │
│  - Efficient Storage            │
│  - Similarity Search            │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Question Answering            │
│  - Context Retrieval            │
│  - LLM-based Generation         │
│  - Source Attribution           │
└─────────────────────────────────┘
```

## ✨ Features

### Core Functionality
- ✅ **URL Input & Validation**: Handles invalid/unreachable URLs gracefully
- ✅ **Smart Web Crawling**: Extracts meaningful content while removing noise (headers, footers, ads, navigation)
- ✅ **Content Deduplication**: Avoids processing duplicate content
- ✅ **Semantic Chunking**: Splits content intelligently with configurable size and overlap
- ✅ **Metadata Preservation**: Maintains source URL, page title, and chunk IDs
- ✅ **Vector Storage**: Efficient embedding storage and retrieval using FAISS
- ✅ **Persistent Storage**: Embeddings are saved and reusable
- ✅ **Contextual Q&A**: Answers questions based solely on website content
- ✅ **Hallucination Prevention**: Returns specific message when answer is unavailable
- ✅ **Short-term Memory**: Maintains conversation context within session
- ✅ **Source Attribution**: Shows which pages were used to generate answers

### User Interface
- 🎨 **Clean Streamlit Interface**: Modern, intuitive design
- 💬 **Chat-based Interaction**: Natural conversation flow
- 📊 **Progress Indicators**: Clear feedback during processing
- 📚 **Source Citations**: View exact content used for answers
- ⚙️ **Configuration Panel**: Easy settings management

## 🛠️ Technology Stack

### Frameworks & Libraries

#### **LangChain**
- **Why**: Industry-standard framework for building LLM applications
- **Usage**: 
  - `RecursiveCharacterTextSplitter` for intelligent text chunking
  - Provides semantic splitting with configurable overlap
  - Maintains document structure and context

#### **Sentence Transformers (all-MiniLM-L6-v2)**
- **Why**: 
  - Fast and efficient (384 dimensions)
  - Good balance of speed vs. quality
  - No API costs
  - Optimized for semantic search
  - Only 80MB model size
- **Alternative considered**: OpenAI embeddings (more accurate but requires API calls)

#### **FAISS (Facebook AI Similarity Search)**
- **Why**:
  - Lightning-fast similarity search
  - In-memory processing (no external dependencies)
  - Supports millions of vectors
  - No API costs or rate limits
  - Easy to persist and load
- **Alternatives considered**: 
  - Pinecone (requires API, costs money)
  - ChromaDB (more overhead for simple use case)
  - Qdrant (requires separate server)

#### **GPT-3.5-turbo (Optional)**
- **Why**:
  - Cost-effective ($0.50/$1.50 per 1M tokens)
  - Good balance of quality and speed
  - Supports system prompts for strict grounding
- **Fallback**: Simple extraction-based answering when API key not provided
- **Alternatives considered**: GPT-4 (more expensive), local models (slower, less accurate)

### Web Crawling
- **BeautifulSoup4**: Robust HTML parsing
- **Requests**: HTTP handling with timeout and error management

### UI Framework
- **Streamlit**: Rapid development of interactive web apps

## 📦 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) OpenAI API key for enhanced answers

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd website-qa-chatbot
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (optional)
```bash
# Create a .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Running Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Running with Docker (Optional)

```bash
# Build image
docker build -t website-chatbot .

# Run container
docker run -p 8501:8501 website-chatbot
```

## 📖 Usage Guide

### Step 1: Index a Website
1. Enter a website URL in the input field (e.g., `https://docs.python.org`)
2. Click "Crawl & Index"
3. Wait for the system to:
   - Crawl the website (up to 10 pages)
   - Extract and clean content
   - Create text chunks
   - Generate embeddings
   - Build vector index

### Step 2: Ask Questions
1. Type your question in the chat input
2. The system will:
   - Search for relevant content
   - Retrieve top 3 matching chunks
   - Generate an answer using LLM or fallback method
3. View the answer and source citations

### Step 3: Continue Conversation
- The chatbot maintains context from previous questions
- Ask follow-up questions naturally
- Clear the index to start with a new website

## 🎯 Design Decisions

### Embedding Strategy

**Model Selection: all-MiniLM-L6-v2**
- Chosen for optimal balance of:
  - Speed (0.1s per 100 sentences)
  - Quality (strong semantic understanding)
  - Size (80MB vs 1GB+ for larger models)
  - Cost (completely free)

**Chunking Strategy**
- Chunk size: 500 characters (optimal for semantic coherence)
- Overlap: 50 characters (maintains context across boundaries)
- Recursive splitting on: paragraphs → sentences → words
- Preserves semantic meaning better than fixed-length splits

### Vector Database: FAISS

**Why FAISS over alternatives?**

| Feature | FAISS | Pinecone | ChromaDB | Qdrant |
|---------|-------|----------|----------|--------|
| Setup | ✅ Instant | ❌ API Key | 🟡 Medium | ❌ Server |
| Speed | ✅ Fastest | 🟡 Fast | 🟡 Fast | 🟡 Fast |
| Cost | ✅ Free | ❌ Paid | ✅ Free | ✅ Free |
| Persistence | ✅ Simple | ✅ Cloud | ✅ Local | ✅ Local |
| Scalability | ✅ Millions | ✅ Billions | 🟡 Thousands | ✅ Millions |

For this project size and requirements, FAISS is optimal.

### LLM Selection: GPT-3.5-turbo

**Justification:**
1. **Cost-effective**: 93% cheaper than GPT-4
2. **Fast**: ~1-2 second response time
3. **Sufficient quality**: Excellent for factual Q&A with grounding
4. **Fallback option**: Works without API key using extraction

**Prompt Engineering:**
- Strict system prompt prevents hallucination
- Context window limits prevent token overflow
- Temperature 0.3 for consistent, factual responses
- Clear instructions for unavailable answers

### Web Crawling Strategy

**Content Filtering:**
- Removes: navigation, headers, footers, ads, scripts, styles
- Keeps: main content, paragraphs, headings, lists
- Validates: minimum content length (100 chars)
- Respects: same-domain policy, robots.txt compatible

**Performance Optimization:**
- Maximum 10 pages to prevent overload
- 0.5s delay between requests (server-friendly)
- 10s timeout per request
- Graceful error handling

## 🚧 Limitations & Future Improvements

### Current Limitations

1. **Crawling Depth**: Limited to 10 pages to prevent resource exhaustion
   - *Future*: Implement async crawling, user-configurable depth

2. **Content Types**: Only processes HTML text
   - *Future*: Add PDF, DOCX, image OCR support

3. **Session Persistence**: Conversation history lost on refresh
   - *Future*: Implement database-backed session storage

4. **Single Website**: Can only index one website at a time
   - *Future*: Multi-website indexing with namespace separation

5. **No Authentication**: Cannot crawl login-protected pages
   - *Future*: Add authentication support

6. **Language**: Optimized for English content
   - *Future*: Multi-language support with language detection


## 🔑 Environment Variables

Create a `.env` file (optional):

```bash
# OpenAI API Key (optional - for enhanced answers)
OPENAI_API_KEY=sk-...

# Configuration (optional - defaults provided)
CHUNK_SIZE=500
CHUNK_OVERLAP=50
MAX_PAGES=10
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## 🧪 Testing

**Manual Testing Checklist:**
- [x] Valid URL crawling
- [x] Invalid URL handling
- [x] Empty content handling
- [x] Duplicate content removal
- [x] Question answering accuracy
- [x] Unavailable answer detection
- [x] Conversation context maintenance
- [x] Source attribution
- [x] API key optional functionality

**Example Test Queries:**
```
URL: https://docs.python.org/3/tutorial/
Q: "What is Python?"
Q: "How do I create a list?"
Q: "What is quantum computing?" (should return unavailable)
```

## 📊 Performance Metrics

**Crawling:**
- ~2-3 seconds per page
- ~20-30 seconds for 10 pages

**Indexing:**
- ~5 seconds for 100 chunks
- ~30 seconds for 500 chunks

**Query:**
- Embedding: ~0.1 seconds
- Search: ~0.01 seconds
- LLM generation: ~1-2 seconds
- **Total**: ~1.5-2.5 seconds per query


