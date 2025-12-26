# Complete Deployment Guide

This guide walks through deploying DebateLab to Railway (backend) and Vercel (frontend) with a custom domain.

## Prerequisites

- GitHub account
- Railway account ([railway.app](https://railway.app))
- Vercel account ([vercel.com](https://vercel.com))
- OpenAI API key
- Custom domain `debatelab.ai` (if using)

---

## Step 1: Push Code to GitHub

1. Make sure all changes are committed:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. Verify your code is on GitHub

---

## Step 2: Deploy Backend to Railway

### 2.1 Create Railway Project

1. Go to [railway.app](https://railway.app) and sign in
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your repository
5. Railway will auto-detect the Python backend

### 2.2 Configure Environment Variables

In Railway dashboard → Your Project → **Variables** tab:

Add these variables:

```
OPENAI_API_KEY=your_openai_api_key_here
CORS_ORIGINS=*
SCORING_MODEL=gpt-4o
```

**Note:** Set `CORS_ORIGINS=*` initially. You'll update this after frontend deploys.

### 2.3 Deploy

1. Railway automatically detects the `Procfile` and starts deployment
2. Wait for deployment to complete (usually 2-3 minutes)
3. Get your backend URL: `https://your-project-name.up.railway.app`
4. **Save this URL** - you'll need it for the frontend

### 2.4 Verify Backend

1. Visit: `https://your-backend-url.up.railway.app/v1/health`
   - Should return: `{"status":"ok"}`
2. Visit: `https://your-backend-url.up.railway.app/docs`
   - Should show FastAPI documentation

---

## Step 3: Deploy Frontend to Vercel

### 3.1 Create Vercel Project

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click **"Add New Project"**
3. Import your GitHub repository

### 3.2 Configure Project Settings

In the project configuration:

- **Root Directory**: `frontend`
- **Framework Preset**: `Vite`
- **Build Command**: `npm run build` (auto-detected)
- **Output Directory**: `dist` (auto-detected)

### 3.3 Set Environment Variables

In Vercel → Your Project → **Settings** → **Environment Variables**:

Add:
```
VITE_API_BASE_URL=https://your-backend-url.up.railway.app
```

**Important:** Replace with your actual Railway backend URL from Step 2.3

### 3.4 Deploy

1. Click **"Deploy"**
2. Wait for build to complete (usually 1-2 minutes)
3. Get your frontend URL: `https://your-project-name.vercel.app`
4. **Save this URL**

### 3.5 Test Frontend

1. Visit your Vercel URL
2. Try creating a debate
3. Check browser console for any CORS errors

---

## Step 4: Update CORS in Railway

After frontend is deployed:

1. Go back to Railway dashboard
2. Your Project → **Variables** tab
3. Find `CORS_ORIGINS`
4. Update value to:
   ```
   https://your-project-name.vercel.app
   ```
   Or if you'll use custom domain:
   ```
   https://debatelab.ai
   ```
   Or both (comma-separated):
   ```
   https://debatelab.ai,https://your-project-name.vercel.app
   ```
5. Save - Railway will automatically redeploy
6. Wait for redeploy to complete

---

## Step 5: Add Custom Domain (Optional)

If using `debatelab.ai`:

### 5.1 In Vercel

1. Go to your project → **Settings** → **Domains**
2. Click **"Add Domain"**
3. Enter: `debatelab.ai`
4. Follow Vercel's DNS configuration instructions
5. Wait for DNS to propagate (usually 5-60 minutes)

### 5.2 Update CORS Again

Once domain is active:

1. Go to Railway → Variables
2. Update `CORS_ORIGINS` to:
   ```
   https://debatelab.ai
   ```
3. Save (Railway will redeploy)

### 5.3 Test Custom Domain

1. Visit `https://debatelab.ai`
2. Test creating a debate
3. Verify everything works

---

## Final Checklist

- [ ] Backend deployed to Railway
- [ ] Backend health check works (`/v1/health`)
- [ ] Backend API docs accessible (`/docs`)
- [ ] Frontend deployed to Vercel
- [ ] Frontend connects to backend (no CORS errors)
- [ ] `CORS_ORIGINS` updated in Railway
- [ ] Custom domain added (if using)
- [ ] Custom domain CORS updated
- [ ] Tested creating a debate
- [ ] Tested submitting arguments
- [ ] Tested getting scores

---

## Environment Variables Summary

### Railway (Backend)
```
OPENAI_API_KEY=your_key_here          (Required)
CORS_ORIGINS=https://debatelab.ai     (Required after domain setup)
SCORING_MODEL=gpt-4o                  (Optional, defaults to gpt-4o)
PORT=8000                             (Auto-set by Railway)
```

### Vercel (Frontend)
```
VITE_API_BASE_URL=https://your-backend.up.railway.app  (Required)
```

---

## Troubleshooting

### Backend Issues

**Build fails:**
- Check Railway logs for errors
- Verify `requirements.txt` is correct
- Ensure Python version is compatible (3.9+)

**Port errors:**
- Don't set `PORT` manually - Railway auto-sets it
- Verify `Procfile` uses `$PORT`

**API key errors:**
- Double-check `OPENAI_API_KEY` is set correctly
- No quotes needed around the key value

### Frontend Issues

**CORS errors:**
- Verify `CORS_ORIGINS` includes your frontend URL exactly (no trailing slash)
- Check Railway deployment completed after updating CORS
- Try hard refresh in browser (Ctrl+Shift+R)

**API connection fails:**
- Verify `VITE_API_BASE_URL` is correct
- Check backend is running (visit `/v1/health`)
- Check browser console for specific error messages

**Build fails:**
- Check Vercel build logs
- Verify all dependencies are in `package.json`
- Ensure `frontend` is set as root directory

### Domain Issues

**Domain not working:**
- Wait for DNS propagation (can take up to 24 hours)
- Verify DNS records in your domain registrar match Vercel's instructions
- Check domain status in Vercel dashboard

---

## Quick Reference URLs

After deployment, you'll have:

- **Backend API**: `https://your-project.up.railway.app`
- **Backend Docs**: `https://your-project.up.railway.app/docs`
- **Frontend**: `https://your-project.vercel.app` or `https://debatelab.ai`
- **Health Check**: `https://your-project.up.railway.app/v1/health`

---

## Support

If you encounter issues:

1. Check Railway deployment logs
2. Check Vercel build logs
3. Check browser console for errors
4. Verify all environment variables are set correctly
5. Test backend endpoints directly (using `/docs` or curl)
