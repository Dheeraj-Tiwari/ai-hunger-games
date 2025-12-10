# 🏹 AI Hunger Games - Streamlit UI

Beautiful, animated web interface for the AI Hunger Games competition!

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install streamlit groq google-generativeai plotly pandas aiohttp
```

### 2. Run the App
```bash
streamlit run streamlit_ui.py
```

### 3. Configure
- Enter your Groq API key (get free at: https://console.groq.com)
- Optionally enter Google API key (get free at: https://makersuite.google.com/app/apikey)
- Choose number of agents (2-20)
- Click "🚀 Initialize Arena"

### 4. Play!
- Enter your question
- Optionally provide correct answer
- Click "🎬 START ROUND"
- Watch the magic happen! ✨

## ✨ Features

### 🎨 Beautiful Design
- **Gradient animations** on the header
- **Glassmorphism effects** for cards
- **Pulse animations** for winners
- **Smooth transitions** everywhere
- **Dark mode optimized**

### 📊 Live Visualizations
- **Interactive bar charts** showing agent scores
- **Model performance comparison** across rounds
- **Real-time leaderboard** updates
- **Animated stat boxes**

### 🎮 Interactive Elements
- **Expandable response cards** to see full answers
- **Side-by-side comparisons** of correct vs eliminated
- **Chairman's synthesis** in a special highlighted box
- **Live metrics** in sidebar

### 🏆 Competition Features
- **Anonymous judging** - no model bias
- **Blind evaluation** - fair scoring
- **Chairman synthesis** - best answer compilation
- **Multi-round tracking** - see performance over time

## 🎯 UI Components

### Header Section
```
🏹 AI HUNGER GAMES 🏹
Animated gradient text with Orbitron font
```

### Sidebar
- API key configuration
- Agent count slider
- Initialize button
- Live round metrics
- Alive/Eliminated counts

### Main Area
- Question input field
- Correct answer field (optional)
- START ROUND button
- Racing indicator
- Response cards (3 columns)
- Results split view
- Chairman's official answer box
- Live charts

### Agent Cards
- **Active**: Purple gradient with hover effect
- **Winner**: Pink gradient with pulse animation
- **Eliminated**: Black/gray gradient, faded

### Chairman's Answer Box
- Peach gradient background
- Large, readable text
- Confidence indicator
- Top contributors list
- Reasoning explanation

## 🎨 Color Scheme

### Gradients Used
- **Main Header**: Red → Teal → Blue → Coral (animated)
- **Agent Cards**: Purple → Dark Purple
- **Winner Cards**: Pink → Red (pulsing)
- **Eliminated**: Black → Dark Gray
- **Chairman Box**: Peach → Coral
- **Buttons**: Purple → Dark Purple

### Fonts
- **Headers**: Orbitron (sci-fi/futuristic)
- **Body**: Default sans-serif

## 📈 Charts

### Leaderboard Chart
- Bar chart showing top 10 agents
- Color-coded by score
- Interactive tooltips
- Viridis color scale

### Model Performance Chart
- Grouped bar chart
- Wins vs Total Score comparison
- Different colors per metric
- Legend included

## 🎬 Animations

### CSS Animations
1. **Gradient Flow** (3s infinite)
   - Background shifts through colors
   - Applied to main header

2. **Pulse** (2s infinite)
   - Scale transform 1.0 → 1.05 → 1.0
   - Applied to winner cards

3. **Blink** (1s infinite)
   - Opacity 1.0 → 0.5 → 1.0
   - Applied during racing

4. **Hover Effects**
   - translateY(-5px) on agent cards
   - scale(1.05) on buttons
   - Shadow increases

## 🔧 Customization

### Change Colors
Edit the CSS gradients in `st.markdown()`:
```python
background: linear-gradient(45deg, #YourColor1, #YourColor2);
```

### Add More Animations
Add new keyframes in the CSS:
```css
@keyframes yourAnimation {
    0% { /* start state */ }
    100% { /* end state */ }
}
```

### Modify Layout
Change column ratios:
```python
col1, col2 = st.columns([3, 1])  # 3:1 ratio
```

## 💡 Tips

1. **Performance**: For 20+ agents, UI updates may be slower
2. **Mobile**: Best viewed on desktop/tablet
3. **Dark Mode**: Automatically optimized
4. **API Limits**: Monitor your API usage in sidebar
5. **Round History**: Scroll down to see all past rounds

## 🐛 Troubleshooting

### "API key invalid"
- Double-check your key
- Ensure no extra spaces
- Try regenerating the key

### "Agents not initializing"
- Verify at least one API key is entered
- Check internet connection
- Look at terminal for error messages

### "Charts not showing"
- Ensure plotly is installed: `pip install plotly`
- Check browser console for errors
- Try refreshing the page

### "Slow performance"
- Reduce number of agents
- Use Groq for faster inference
- Close other browser tabs

## 🎉 Enjoy!

Watch AI models compete in real-time with beautiful animations and fair judging. May the smartest AI win! 🏆