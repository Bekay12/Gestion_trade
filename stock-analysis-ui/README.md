# 📈 Stock Analysis - Web & Desktop Application

A comprehensive stock analysis platform with both desktop (PyQt5) and web (Flask) interfaces. Analyze stocks, generate trading signals, backtest strategies, and manage symbol lists.

## ✨ Features

### 🔍 Analysis
- **Single Symbol Analysis** - Analyze individual stocks with technical indicators
- **Batch Analysis** - Analyze up to 20 symbols simultaneously
- **Popular Signals** - Quick analysis of curated symbol lists
- **Reliability Scoring** - 0-100% confidence metric for each signal

### 📊 Trading Signals
- **BUY Signals** (🟢) - When to buy
- **SELL Signals** (🔴) - When to sell
- **HOLD Signals** (🟡) - When to wait

### 🔬 Backtesting
- Test strategies on historical data
- Customize moving average periods
- View metrics: Win Rate, Total Gain, Trade Count
- Multiple time periods (1M, 3M, 6M, 1Y, 2Y, 5Y)

### 📋 Symbol Management
- **Popular Symbols** - Pre-curated list of major stocks
- **Personal Lists** - Create and manage custom symbol lists
- **Optimization Lists** - Symbols for backtesting

### 🖥️ Interfaces
- **Web Dashboard** - Modern responsive interface at https://stock-analysis-api-8dz1.onrender.com/
- **Desktop App** - PyQt5-based GUI for advanced users
- **REST API** - Complete API for programmatic access

## 🚀 Quick Start

### Web Version (Recommended)
Visit: **https://stock-analysis-api-8dz1.onrender.com/**

No installation needed! Just open the link in your browser.

### Desktop Version
```bash
# Clone the repository
git clone https://github.com/Bekay12/Gestion_trade.git
cd stock-analysis-ui

# Install dependencies
pip install -r requirements.txt

# Run the desktop application
python src/ui/main_window.py
```

### Local Web Server
```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask server
cd src
python api.py

# Open browser and navigate to: http://localhost:5000
```

## 📁 Project Structure

```
stock-analysis-ui/
├── src/
│   ├── qsi.py                      # Core analysis engine
│   ├── api.py                      # Flask web server
│   ├── config.py                   # Configuration management
│   ├── templates/
│   │   └── index.html              # Web interface
│   ├── ui/
│   │   ├── main_window.py          # Desktop GUI
│   │   └── widgets.py              # UI components
│   ├── utils/                      # Utility functions
│   └── data_cache/                 # Symbol data cache
├── data/                           # Historical data
├── cache_data/                     # Cache storage
├── Dockerfile                      # Docker configuration
├── render.yaml                     # Render deployment config
├── requirements.txt                # Python dependencies
├── INTERFACE_GUIDE.md              # User guide
├── CHANGELOG.md                    # Version history
└── README.md                       # This file
```

## 🔧 API Endpoints

### Analysis
- `POST /api/analyze` - Analyze a single symbol
- `POST /api/analyze-popular` - Analyze popular/personal lists
- `POST /api/analyze-batch` - Analyze multiple symbols

### Lists
- `GET /api/lists` - Get all symbol lists
- `POST /api/lists/<type>` - Add/remove symbols from list

### Backtesting
- `POST /api/backtest` - Run strategy backtest

### Data
- `GET /api/signals` - Get recent signals
- `GET /api/stats` - Get statistics

### System
- `GET /health` - Health check
- `GET /` - Web interface

## 📊 Technical Stack

### Backend
- **Python 3.11** - Core language
- **Flask 2.2.5** - Web framework
- **Pandas 2.1.4** - Data manipulation
- **NumPy 1.26.4** - Numerical computing
- **TA-Lib 0.11.0** - Technical analysis
- **YFinance 0.2.36** - Stock data fetching

### Frontend
- **HTML5** - Markup
- **CSS3** - Styling with animations
- **JavaScript** - Client-side logic
- **Responsive Design** - Mobile-friendly

### Deployment
- **Docker** - Containerization
- **Render** - Cloud hosting
- **Git** - Version control

## 🔐 Data & Privacy

- No personal data collection
- All analysis local to your browser/server
- Uses public financial data (Yahoo Finance)
- No account required for web version

## 📖 Documentation

- **[INTERFACE_GUIDE.md](INTERFACE_GUIDE.md)** - Complete user guide with examples
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and new features

## 🧪 Testing

Run the API test suite:
```bash
python test_api.py
```

Tests:
- Health check
- Endpoint validation
- Data structure validation
- Error handling

## 🐛 Troubleshooting

### "Aucun signal fiable trouvé" (No reliable signals found)
- Symbol may not exist or have insufficient data
- Try a longer time period
- Verify correct symbol format (e.g., AAPL not AAL)

### Web interface not loading
- Check internet connection
- Clear browser cache (Ctrl+Shift+Del)
- Try a different browser
- Server may be starting (wait 30 seconds)

### Desktop app crashes
- Update Python: `pip install --upgrade python`
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
- Check for PyQt5 conflicts

## 🛠️ Development

### Setup Development Environment
```bash
# Clone repo
git clone https://github.com/Bekay12/Gestion_trade.git

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
python test_api.py
```

### Building Docker Image
```bash
docker build -t stock-analysis .
docker run -p 5000:5000 stock-analysis
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For issues or questions:
- Check [INTERFACE_GUIDE.md](INTERFACE_GUIDE.md) for common issues
- Review [CHANGELOG.md](CHANGELOG.md) for recent changes
- Open an issue on GitHub

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🌟 Acknowledgments

- Technical analysis indicators from TA-Lib
- Stock data from Yahoo Finance
- Icons and design inspiration from modern web apps

---

**Current Version:** 1.0.0  
**Last Updated:** January 2025  
**Status:** 🟢 Active Development  
**Web URL:** https://stock-analysis-api-8dz1.onrender.com/
