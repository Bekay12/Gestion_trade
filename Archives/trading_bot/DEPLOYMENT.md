# 🚀 Deployment Guide - Streamlit Cloud

This guide will help you deploy the Trading Bot web interface to Streamlit Cloud for access from any device.

## Prerequisites

- GitHub account
- Code pushed to a GitHub repository
- Free Streamlit Cloud account (sign up at https://streamlit.io)

## Step-by-Step Deployment

### 1. Prepare Your Repository

Ensure these files are present in your `trading_bot/` directory:
- ✅ `streamlit_app.py` (main application)
- ✅ `requirements.txt` (dependencies)
- ✅ `.streamlit/config.toml` (configuration)
- ✅ All source code in `src/`, `config/`, etc.

### 2. Push to GitHub

```bash
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main
```

### 3. Deploy to Streamlit Cloud

1. Go to https://streamlit.io and sign in with your GitHub account

2. Click "New app" button

3. Fill in the deployment form:
   - **Repository**: `Bekay12/Gestion_trade` (or your fork)
   - **Branch**: `main` (or your preferred branch)
   - **Main file path**: `trading_bot/streamlit_app.py`
   - **App URL**: Choose a custom subdomain (optional)

4. Click "Deploy!"

5. Wait for deployment (usually 2-5 minutes)

### 4. Access Your App

Once deployed, you'll get a URL like:
```
https://your-app-name.streamlit.app
```

You can access this URL from:
- ✅ Your phone (iOS/Android)
- ✅ Your tablet
- ✅ Any computer with a web browser
- ✅ Works independently of your PC!

## Configuration Options

### Environment Variables

If you need to set environment variables (API keys, etc.):

1. In Streamlit Cloud, go to your app settings
2. Click "Advanced settings"
3. Add your secrets in TOML format

Example `.streamlit/secrets.toml` format:
```toml
[database]
api_key = "your-secret-key"
```

### App Resources

Free tier limits:
- 1 GB RAM
- Shared CPU
- Suitable for personal use

For higher limits, consider Streamlit Cloud paid plans.

## Updating Your App

To update the deployed app:

```bash
# Make your changes
git add .
git commit -m "Update trading bot features"
git push origin main
```

Streamlit Cloud will automatically redeploy within minutes!

## Troubleshooting

### App Won't Start

Check logs in Streamlit Cloud dashboard for errors:
1. Go to your app in Streamlit Cloud
2. Click "Manage app"
3. View "Logs" tab

Common issues:
- Missing dependencies in `requirements.txt`
- File path issues (use relative paths)
- Missing data files

### Data Files

If your app needs data files from `data/symbols/`:
- These are included in the repository
- Make sure they're committed to Git
- Check file paths are relative

### Performance Issues

If the app is slow:
- Reduce the number of symbols analyzed at once
- Use caching (`@st.cache_data` decorator)
- Consider upgrading to paid tier for more resources

## Alternative Deployment: Railway

If Streamlit Cloud doesn't meet your needs, try Railway:

1. Go to https://railway.app
2. Connect your GitHub repository
3. Railway will auto-detect the Python app
4. Add start command: `streamlit run trading_bot/streamlit_app.py --server.port=$PORT`

## Alternative Deployment: Heroku

For Heroku deployment, see the main README.md

## Mobile Access Tips

### Add to Home Screen

On mobile devices, you can add the app to your home screen:

**iOS (Safari):**
1. Open the app URL
2. Tap the Share button
3. Select "Add to Home Screen"

**Android (Chrome):**
1. Open the app URL
2. Tap the menu (⋮)
3. Select "Add to Home Screen"

This gives you quick access like a native app!

## Security Considerations

⚠️ **Important**: The deployed app is publicly accessible by default.

To restrict access:
1. Use Streamlit Cloud's authentication features
2. Implement basic auth in your app
3. Use a private repository and share the URL only with trusted users

## Cost

- **Streamlit Cloud Free Tier**: Perfect for personal use
- **Private apps**: Available in paid plans
- **No credit card required** for free tier

## Support

For deployment issues:
- Streamlit Community: https://discuss.streamlit.io
- GitHub Issues: Your repository issues page
- Documentation: https://docs.streamlit.io

---

🎉 **You're ready!** Deploy your app and access your trading bot from anywhere!
