# Render Deployment Guide for Bhagavad Gita Chat

## 🚀 Quick Deployment Steps

### 1. Push Your Code to GitHub

First, commit all the recent changes:

```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

### 2. Configure Render Service

#### Option A: Using Render Dashboard (Recommended)

1. **Create New Web Service**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click **"New +"** → **"Web Service"**
   - Connect your GitHub repository

2. **Configure Build & Deploy Settings**
   - **Name**: `bhagavad-gita-backend` (or your preferred name)
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your deployment branch)
   - **Root Directory**: Leave empty
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd app1 && uvicorn endpoints.chat:app --host 0.0.0.0 --port $PORT`

3. **Set Environment Variables**
   
   Go to **Environment** tab and add these variables:

   | Key | Value | Where to Get It |
   |-----|-------|----------------|
   | `OPENAI_API_KEY` | `sk-...` | [OpenAI Dashboard](https://platform.openai.com/api-keys) |
   | `PINECONE_API_KEY` | `pcsk_...` | [Pinecone Console](https://app.pinecone.io/) |
   | `PINECONE_ENVIRONMENT` | `us-east-1` | Your Pinecone environment |
   | `PINECONE_INDEX_NAME` | `bhagavad-gita` | Your Pinecone index name |
   | `SUPABASE_URL` | `https://xxx.supabase.co` | [Supabase Dashboard](https://app.supabase.com/) → Settings → API |
   | `SUPABASE_ANON_KEY` | `eyJ...` | Supabase Dashboard → Settings → API → anon public |
   | `PYTHON_VERSION` | `3.11.0` | (Optional) Python version |

4. **Deploy**
   - Click **"Create Web Service"**
   - Render will automatically deploy your app
   - Wait for the deployment to complete (5-10 minutes)

#### Option B: Using render.yaml (Infrastructure as Code)

The `render.yaml` file has been created. Just:
1. Push to GitHub
2. In Render, select **"Blueprint"** → Connect repository
3. Render will read `render.yaml` and configure automatically
4. You'll still need to add environment variable values manually

### 3. Verify Deployment

Once deployed, test these endpoints:

1. **Health Check**: `https://your-app.onrender.com/health`
2. **API Docs**: `https://your-app.onrender.com/docs`
3. **Frontend**: `https://your-app.onrender.com/`

### 4. Monitor Logs

In Render Dashboard:
- Go to your service → **Logs** tab
- Look for: `"Supabase context service initialized successfully!"`
- Check for any error messages

---

## 🐛 Troubleshooting Common Issues

### Issue 1: "Sorry, I'm having trouble connecting to the server"

**Cause**: Backend not running or crashed

**Solution**:
1. Check Render logs for errors
2. Verify all environment variables are set correctly
3. Make sure your Render service shows "Live" status

### Issue 2: Application Crashes on Startup

**Cause**: Missing environment variables or dependencies

**Solution**:
1. Check logs: `Settings → Logs`
2. Verify all required env vars are set
3. Check if `requirements.txt` has all dependencies

### Issue 3: "Supabase disabled. Falling back to memory service"

**Cause**: Missing `SUPABASE_URL` or `SUPABASE_ANON_KEY`

**Solution**:
1. Add both variables in Environment tab
2. Trigger manual redeploy

### Issue 4: OpenAI API Errors

**Cause**: Invalid or missing `OPENAI_API_KEY`

**Solution**:
1. Get valid key from OpenAI dashboard
2. Add to environment variables
3. Ensure you have credits/billing set up

### Issue 5: Pinecone Connection Failures

**Cause**: Wrong Pinecone credentials or index doesn't exist

**Solution**:
1. Verify index exists in Pinecone dashboard
2. Check `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT` match
3. Ensure index name matches `PINECONE_INDEX_NAME`

---

## 📊 Performance Optimization

### Free Tier Considerations

Render's free tier:
- ✅ Spins down after 15 minutes of inactivity
- ⏱️ First request after spin-down takes ~30-60 seconds
- 💰 Upgrade to paid plan for always-on service

### Recommendations

1. **Keep Alive Service** (Optional):
   - Set up a cron job to ping your service every 10 minutes
   - Prevents spin-down during active hours

2. **Upgrade to Paid Plan**:
   - No spin-down
   - Better performance
   - Custom domains

---

## 🔒 Security Checklist

- [ ] All API keys set as environment variables (not in code)
- [ ] `.env` file is in `.gitignore`
- [ ] CORS configured properly in `chat.py`
- [ ] HTTPS enabled (automatic on Render)
- [ ] Supabase Row Level Security (RLS) enabled

---

## 📝 Quick Commands

### View Live Logs
```bash
# In Render Dashboard → Your Service → Logs
```

### Manual Redeploy
```bash
# In Render Dashboard → Your Service → Manual Deploy → Deploy latest commit
```

### Check Service Status
```bash
curl https://your-app.onrender.com/health
```

---

## 🆘 Still Having Issues?

1. **Check Render Logs**: Look for specific error messages
2. **Test Locally First**: Ensure app runs locally with `python app1/run_app.py`
3. **Verify Environment**: All required env vars must be set
4. **Check Dependencies**: Ensure `requirements.txt` is complete

### Common Error Messages & Fixes

| Error Message | Fix |
|--------------|-----|
| `"Failed to initialize Supabase"` | Add `SUPABASE_URL` and `SUPABASE_ANON_KEY` |
| `"OpenAI API key not found"` | Add `OPENAI_API_KEY` |
| `"Pinecone connection failed"` | Verify Pinecone credentials and index |
| `"Module not found"` | Add missing package to `requirements.txt` |
| `"Port already in use"` | Render handles this automatically |

---

## 📞 Support

If you continue to face issues:
1. Check Render's status page: https://status.render.com/
2. Review deployment logs carefully
3. Ensure all services (OpenAI, Pinecone, Supabase) are active

---

**Your app should now be live at**: `https://your-service-name.onrender.com`

🎉 Happy Deploying!

