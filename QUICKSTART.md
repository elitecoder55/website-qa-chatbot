# 🚀 Quick Start Guide

Get up and running in 5 minutes!

## Option 1: Automatic Setup (Recommended)

```bash
# Run the setup script
bash setup.sh

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Run the app
streamlit run app.py
```

## Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Option 3: Docker

```bash
# Build the image
docker build -t website-chatbot .

# Run the container
docker run -p 8501:8501 website-chatbot
```

## 🎯 First Steps

1. **Open the app** at http://localhost:8501
2. **Enter a website URL** (e.g., `https://docs.python.org`)
3. **Click "Crawl & Index"** and wait ~30 seconds
4. **Ask questions** about the website content!

## 💡 Example Websites to Try

- Documentation: `https://docs.python.org/3/tutorial/`
- News: `https://techcrunch.com`
- Blogs: `https://blog.google`
- Company site: `https://www.anthropic.com`

## 🔑 Optional: Add OpenAI API Key

For better answers, add your OpenAI API key:

1. Copy `.env.example` to `.env`
2. Add your key: `OPENAI_API_KEY=sk-...`
3. Restart the app

**Without API key:** The app still works using fallback extraction mode!

## ❓ Troubleshooting

**Port already in use?**
```bash
streamlit run app.py --server.port=8502
```

**Dependencies not installing?**
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

**Module not found error?**
```bash
# Make sure virtual environment is activated
which python  # Should show path to venv
```

## 📚 Need More Help?

- Read the full [README.md](README.md)
- Check the [Architecture Documentation](README.md#architecture)
- Review [Design Decisions](README.md#design-decisions)

Happy chatting! 🤖
