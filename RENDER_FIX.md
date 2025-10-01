# 🔧 Render Deployment Fix

## ✅ What Was Fixed

### Problem
```
ModuleNotFoundError: No module named 'app1'
```

### Root Cause
The import statement in `app1/services/pinecone_services.py` was using an absolute import:
```python
from app1.services.embeddings import embed_text  # ❌ Wrong
```

But since the start command runs from inside `app1/` directory:
```bash
cd app1 && uvicorn endpoints.chat:app
```

Python couldn't find the `app1` module.

### Solution Applied
Changed to relative import:
```python
from .embeddings import embed_text  # ✅ Correct
```

---

## 🚀 Deploy to Render Now

### Step 1: Commit the Fix
```bash
git add .
git commit -m "Fix: Use relative imports for Render deployment"
git push origin main
```

### Step 2: Configure Render Service

#### Build Command:
```bash
pip install -r requirements.txt
```

#### Start Command:
```bash
cd app1 && uvicorn endpoints.chat:app --host 0.0.0.0 --port $PORT --log-level info
```

### Step 3: Set Environment Variables

In Render Dashboard → Environment tab, add these **6 required variables**:

| Variable | Value | Where to Get It |
|----------|-------|----------------|
| `OPENAI_API_KEY` | `sk-...` | https://platform.openai.com/api-keys |
| `PINECONE_API_KEY` | `pcsk_...` | https://app.pinecone.io/ |
| `PINECONE_ENVIRONMENT` | `us-east-1` | Your Pinecone environment |
| `PINECONE_INDEX_NAME` | `bhagavad-gita` | Your index name |
| `SUPABASE_URL` | `https://xxx.supabase.co` | Supabase → Settings → API |
| `SUPABASE_ANON_KEY` | `eyJ...` | Supabase → Settings → API |

### Step 4: Deploy

1. Push code to GitHub
2. Render will auto-deploy OR click **"Manual Deploy"**
3. Wait 5-10 minutes
4. Check logs for success

---

## ✅ Verify Deployment

### Check Health Endpoint
```bash
curl https://your-app.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "pinecone_connected": true,
  "index_name": "bhagavad-gita",
  "total_vectors": 12345
}
```

### Test Frontend
Visit: `https://your-app.onrender.com/`

### Check Logs
Look for:
- ✅ `"Supabase context service initialized successfully!"`
- ✅ `"Application startup complete"`
- ❌ NO `ModuleNotFoundError`

---

## 🐛 Troubleshooting

### If You Still See Errors:

#### 1. "Supabase disabled"
**Fix**: Add `SUPABASE_URL` and `SUPABASE_ANON_KEY` environment variables

#### 2. "OpenAI API error"
**Fix**: Verify `OPENAI_API_KEY` is correct and has credits

#### 3. "Pinecone connection failed"
**Fix**: Check all Pinecone variables are correct and index exists

#### 4. "Application startup failed"
**Fix**: Check Render logs for specific error message

#### 5. Service shows "Failed" or "Suspended"
**Fix**: 
- Check all environment variables are set
- Verify build command succeeded
- Review full deployment logs

---

## 📋 Quick Checklist

- [x] Fixed import statement in `pinecone_services.py`
- [x] Created `Procfile` with correct start command
- [x] Created `render.yaml` for easy deployment
- [ ] Committed and pushed changes to GitHub
- [ ] Set all 6 environment variables on Render
- [ ] Deployed service on Render
- [ ] Verified health endpoint works
- [ ] Tested chat functionality

---

## 🎯 Your Turn!

1. **Commit this fix:**
   ```bash
   git add app1/services/pinecone_services.py Procfile render.yaml
   git commit -m "Fix: Resolve import error for Render deployment"
   git push origin main
   ```

2. **Go to Render Dashboard**
   - Set environment variables
   - Trigger manual deploy
   - Wait for "Live" status

3. **Test your app!**
   - Visit your URL
   - Try asking a question about Bhagavad Gita
   - Celebrate! 🎉

---

## 💡 Why This Matters

**Absolute imports** (`from app1.services...`) work when:
- Running from project root
- Using `python -m app1.endpoints.chat`

**Relative imports** (`from .embeddings...`) work when:
- Running from inside the module
- Using `cd app1 && uvicorn endpoints.chat:app`

Since Render uses the second approach, we needed relative imports!

---

**Your deployment should now work!** 🚀

If you still face issues, check the Render logs and the error message carefully.

