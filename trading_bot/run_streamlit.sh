#!/bin/bash
# Script to run the Streamlit web interface locally

echo "🚀 Starting Trading Bot Streamlit Interface..."
echo ""
echo "The app will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
streamlit run streamlit_app.py
