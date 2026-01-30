# 🚀 Deployment Guide

## Option 1: Streamlit Community Cloud (Recommended)

**Free, easy, and perfect for this assignment!**

### Step-by-Step Deployment

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Website QA Chatbot"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Sign in with your GitHub account
   - Click "New app"

3. **Configure Your App**
   - Repository: Select your repository
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy!"

4. **Add Secrets (Optional)**
   If you want to use OpenAI:
   - Go to App settings → Secrets
   - Add:
     ```toml
     OPENAI_API_KEY = "sk-your-key-here"
     ```

5. **Share the Link**
   - Your app will be available at: `https://<your-app-name>.streamlit.app`
   - Copy this link for the submission form

### Requirements for Streamlit Cloud
✅ Already configured in this project:
- `requirements.txt` is present
- Dependencies are compatible
- No local file dependencies

### Troubleshooting Streamlit Cloud

**App fails to deploy?**
- Check the logs in Streamlit Cloud dashboard
- Verify `requirements.txt` is in repository root
- Ensure no syntax errors in `app.py`

**Out of resources?**
- Streamlit Cloud has memory limits
- Reduce `MAX_PAGES` in app.py (line 11)
- Or upgrade to a paid tier

**Slow loading?**
- First run downloads the embedding model (~80MB)
- Subsequent runs will be faster

---

## Option 2: Heroku

1. **Install Heroku CLI**
   ```bash
   # On macOS
   brew tap heroku/brew && brew install heroku
   
   # On Windows
   # Download from: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Heroku App**
   ```bash
   heroku login
   heroku create your-app-name
   ```

3. **Add Buildpacks**
   ```bash
   heroku buildpacks:add heroku/python
   ```

4. **Create Procfile**
   ```bash
   echo "web: sh setup.sh && streamlit run app.py" > Procfile
   ```

5. **Create setup.sh**
   ```bash
   cat > setup.sh << 'EOF'
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   EOF
   ```

6. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

7. **Open Your App**
   ```bash
   heroku open
   ```

---

## Option 3: Railway

1. **Sign up at Railway.app**
   - Visit: https://railway.app
   - Sign in with GitHub

2. **New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure**
   - Railway auto-detects Python
   - Set start command: `streamlit run app.py --server.port=$PORT`

4. **Add Environment Variables** (Optional)
   - Go to Variables tab
   - Add `OPENAI_API_KEY` if needed

5. **Deploy**
   - Railway deploys automatically
   - Get your public URL from the dashboard

---

## Option 4: Google Cloud Run

1. **Build Docker Image**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/website-chatbot
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy website-chatbot \
     --image gcr.io/PROJECT-ID/website-chatbot \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

---

## Option 5: Local Network Access

If you can't deploy publicly, you can run locally and provide access instructions:

1. **Run with ngrok** (temporary public URL)
   ```bash
   # Install ngrok
   brew install ngrok  # macOS
   # or download from: https://ngrok.com/download
   
   # Run your app
   streamlit run app.py
   
   # In another terminal, expose it
   ngrok http 8501
   ```
   
   Share the ngrok URL (valid for ~8 hours)

2. **Run on local network**
   ```bash
   streamlit run app.py --server.address=0.0.0.0
   ```
   
   Access from other devices: `http://YOUR_LOCAL_IP:8501`

---

## Environment Variables for Production

For any deployment, you might want to set:

```bash
# OpenAI API Key (optional)
OPENAI_API_KEY=sk-...

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
```

---

## Testing Your Deployment

After deployment, test with:

1. **Basic functionality**
   - Enter URL: `https://docs.python.org/3/tutorial/`
   - Click "Crawl & Index"
   - Ask: "What is Python?"

2. **Error handling**
   - Try invalid URL: `not-a-url`
   - Should show error gracefully

3. **Unavailable answers**
   - Ask: "What is quantum computing?"
   - Should say: "The answer is not available on the provided website."

4. **Conversation context**
   - Ask: "What is Python?"
   - Then ask: "Tell me more about it"
   - Should maintain context

---

## Performance Optimization for Production

1. **Reduce MAX_PAGES** (in app.py)
   ```python
   MAX_PAGES = 5  # Instead of 10
   ```

2. **Use smaller embedding model** (if needed)
   ```python
   EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"  # Smaller, faster
   ```

3. **Enable caching**
   Already implemented with `st.session_state`

4. **Add rate limiting** (for public deployments)
   - Prevent abuse
   - Limit crawl requests per IP

---

## Recommended: Streamlit Community Cloud

**Why?**
- ✅ Free
- ✅ Easy to use
- ✅ GitHub integration
- ✅ Perfect for demos
- ✅ Public URL
- ✅ No credit card required

**Limitations:**
- Resource limits (1GB RAM)
- Public apps only (unless Pro)
- Limited compute time

For this assignment, Streamlit Community Cloud is the **best choice**!

---

## Support

If you encounter issues:
1. Check the platform's documentation
2. Review deployment logs
3. Test locally first
4. Simplify configuration if needed

**Need help?** Most platforms have excellent documentation:
- Streamlit: https://docs.streamlit.io/streamlit-community-cloud
- Heroku: https://devcenter.heroku.com
- Railway: https://docs.railway.app
