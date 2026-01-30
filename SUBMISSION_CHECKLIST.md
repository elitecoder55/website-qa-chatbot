# 📋 Submission Checklist

Use this checklist before submitting your assignment.

## ✅ Required Deliverables

### 1. GitHub Repository
- [ ] Repository is public or accessible
- [ ] All source code is committed
- [ ] No hardcoded secrets or API keys
- [ ] Clean and modular project structure
- [ ] `.gitignore` is properly configured

### 2. Source Code
- [ ] `app.py` - Main Streamlit application
- [ ] `requirements.txt` - All dependencies listed
- [ ] Code is clean and well-commented
- [ ] No plagiarized code without attribution

### 3. Documentation
- [ ] `README.md` with all required sections:
  - [ ] Project overview
  - [ ] Architecture explanation
  - [ ] Frameworks used (LangChain/LangGraph)
  - [ ] LLM model choice and justification
  - [ ] Vector database choice and justification
  - [ ] Embedding strategy
  - [ ] Setup and run instructions
  - [ ] Assumptions, limitations, and future improvements

### 4. Streamlit Application
- [ ] Application runs without errors
- [ ] Public deployment link OR clear local run instructions
- [ ] UI allows entering website URL
- [ ] UI shows indexing progress
- [ ] Chat interface for questions
- [ ] Clear display of responses

## ✅ Core Requirements

### URL Input
- [ ] Accepts valid website URLs
- [ ] Handles invalid URLs gracefully
- [ ] Handles unreachable URLs gracefully
- [ ] Handles empty/unsupported content gracefully

### Web Crawling
- [ ] Extracts meaningful textual content
- [ ] Removes headers
- [ ] Removes footers
- [ ] Removes navigation menus
- [ ] Removes advertisements
- [ ] Avoids duplicate content
- [ ] Works with HTML pages

### Text Processing
- [ ] Cleans and normalizes text
- [ ] Splits content into semantic chunks
- [ ] Chunk size is configurable
- [ ] Chunk overlap is configurable
- [ ] Maintains source URL metadata
- [ ] Maintains page title metadata

### Embeddings & Vector Storage
- [ ] Generates embeddings from text chunks
- [ ] Uses embedding model (specified in README)
- [ ] Stores in vector database (specified in README)
- [ ] Embeddings are persisted
- [ ] Embeddings are reusable (not recreated on every query)

### AI Framework & LLM
- [ ] Framework used is mentioned (if any)
- [ ] LLM model is clearly mentioned
- [ ] LLM choice is justified in README
- [ ] Prompt ensures answers from website content only
- [ ] No hallucinated responses
- [ ] No external knowledge in responses

### Question Answering
- [ ] Accepts natural language questions
- [ ] Returns accurate answers
- [ ] Answers based only on website content
- [ ] Returns exact message when answer unavailable:
      "The answer is not available on the provided website."

### Short-Term Memory
- [ ] Maintains conversation context
- [ ] Context limited to current session only
- [ ] Previous questions influence follow-ups

## ✅ Code Quality

- [ ] Code is modular and organized
- [ ] Functions have clear purposes
- [ ] Variables have meaningful names
- [ ] Comments explain complex logic
- [ ] Error handling is implemented
- [ ] No hardcoded values where configuration is needed

## ✅ Testing

- [ ] Tested with valid URLs
- [ ] Tested with invalid URLs
- [ ] Tested with unreachable URLs
- [ ] Tested questions with available answers
- [ ] Tested questions with unavailable answers
- [ ] Tested conversation context
- [ ] Tested with different websites

## ✅ Submission

- [ ] Google Form filled: https://forms.gle/KNdYW4WauH3ngvgY7
- [ ] All required details submitted
- [ ] README includes setup instructions
- [ ] Submission before deadline: **30th January 2026**

## 📝 Notes

### Important Reminders
1. **No hardcoded secrets** - Use environment variables
2. **No plagiarized code** - Explain any borrowed code
3. **Code quality matters** - Clarity and reasoning are important
4. **Strictly grounded answers** - Only website content
5. **Test thoroughly** - Ensure everything works

### Before Final Submission
1. Run `python test.py` to verify components
2. Test the full application with real URLs
3. Review README for completeness
4. Check all links work
5. Verify no sensitive data in repository
6. Test deployment (if public link provided)

## 🎯 Quick Pre-Submission Test

Run these commands:

```bash
# 1. Test components
python test.py

# 2. Run the app
streamlit run app.py

# 3. Test with these URLs:
#    - https://docs.python.org/3/tutorial/
#    - https://www.anthropic.com
#    - Any website of your choice

# 4. Ask these questions:
#    - A question that should have an answer
#    - A question that shouldn't have an answer
#    - A follow-up question to test memory
```

## ✅ Final Check

- [ ] All tests pass
- [ ] Application runs without errors
- [ ] Documentation is complete
- [ ] Submission form filled
- [ ] Repository is accessible
- [ ] Confident about the submission

---

**Good luck with your submission! 🚀**
