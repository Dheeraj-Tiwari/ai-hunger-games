# Ìøπ AI Hunger Games

An interactive competition between AI models with blind judging and real-time visualization.

## Ì∫Ä Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-hunger-games.git
   cd ai-hunger-games
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API keys**
   
   Copy the secrets template:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   
   Edit `.streamlit/secrets.toml` and add your real API keys:
   ```toml
   GROQ_API_KEY = "your_actual_groq_key"
   GOOGLE_API_KEY = "your_actual_google_key"
   ```

4. **Run the app**
   ```bash
   streamlit run streamlit_app.py
   ```

## Ìºê Deploy to Streamlit Cloud

1. **Push to GitHub** (secrets are automatically ignored)
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Click** "New app"
4. **Connect** your GitHub repository
5. **Add secrets** in Streamlit Cloud dashboard:
   - Go to Settings ‚Üí Secrets
   - Paste your API keys in TOML format
6. **Deploy!**

## Ì¥ë Get Free API Keys

- **Groq** (14,400 requests/day): [console.groq.com](https://console.groq.com)
- **Google Gemini** (60 requests/min): [makersuite.google.com](https://makersuite.google.com/app/apikey)

## ‚ú® Features

- Ì¥ñ Multi-model AI competition
- ‚öñÔ∏è Blind judging (no favoritism)
- Ì±î Chairman synthesizes best answers
- Ì≥ä Live charts and leaderboards
- Ìæ® Beautiful animated UI
- ÌøÜ Fair ranking based on accuracy + insight

## Ì≥∏ Screenshots

[Add screenshots here after deployment]

## Ìª°Ô∏è Security

- ‚úÖ API keys stored in secrets (never in code)
- ‚úÖ Secrets ignored by git
- ‚úÖ Rate limiting implemented
- ‚úÖ Error handling for API failures

## Ì≥Ñ License

MIT License - Free to use and modify!

## Ì¥ù Contributing

Pull requests welcome! Please ensure:
- No API keys in code
- Follow existing code style
- Test before submitting

## Ì∞õ Issues

Found a bug? [Create an issue](https://github.com/YOUR_USERNAME/ai-hunger-games/issues)

---

Made with ‚ù§Ô∏è for AI enthusiasts
