import streamlit as st
import asyncio
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import random
import plotly.graph_objects as go
import plotly.express as px
from groq import Groq
import google.generativeai as genai
import pandas as pd

# Set page config
st.set_page_config(
    page_title="🏹 AI Hunger Games",
    page_icon="🏹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5em;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #FFA07A);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        margin-bottom: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        color: #888;
        font-size: 1.2em;
        margin-top: 0;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .agent-card:hover {
        transform: translateY(-5px);
    }
    
    .winner-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .eliminated-card {
        background: linear-gradient(135deg, #434343 0%, #000000 100%);
        opacity: 0.6;
    }
    
    .stat-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .racing-indicator {
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.5; }
    }
    
    .chairman-box {
        background: linear-gradient(135deg, #0052A2 0%, #00172D 100%);
        padding: 25px;
        border-radius: 20px;
        border: 3px solid #ff9a76;
        margin: 20px 0;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 15px;
        font-size: 1.1em;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

class ModelProvider(Enum):
    GROQ_LLAMA = "groq_llama"
    GROQ_MIXTRAL = "groq_mixtral"
    GROQ_GEMMA = "groq_gemma"
    GOOGLE_GEMINI = "google_gemini"
    GOOGLE_GEMINI_FLASH = "google_gemini_flash"

@dataclass
class Agent:
    id: int
    name: str
    anonymous_id: str
    personality: str
    temperature: float
    model_provider: ModelProvider
    model_name: str
    wins: int = 0
    eliminations: int = 0
    total_time: float = 0.0
    accuracy_score: float = 0.0
    insight_score: float = 0.0
    is_alive: bool = True

@dataclass
class RoundResult:
    agent_id: int
    anonymous_id: str
    answer: str
    time_taken: float
    is_correct: bool
    accuracy_rating: float = 0.0
    insight_rating: float = 0.0
    error: Optional[str] = None

@dataclass
class ChairmanSummary:
    question: str
    final_answer: str
    reasoning: str
    top_contributors: List[str]
    confidence: str

class ArenaUI:
    def __init__(self):
        self.groq_client = None
        self.google_available = False
        self.agents: List[Agent] = []
        self.round_number = 0
        self.history = []
        
        self.code_names = [
            "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", 
            "Golf", "Hotel", "India", "Juliet", "Kilo", "Lima",
            "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo",
            "Sierra", "Tango", "Uniform", "Victor", "Whiskey", "X-ray",
            "Yankee", "Zulu", "Phoenix", "Dragon", "Falcon", "Tiger"
        ]
        
        self.model_configs = [
            (ModelProvider.GROQ_LLAMA, "llama-3.3-70b-versatile", "Llama 3.3 70B", 0.2),
            (ModelProvider.GROQ_LLAMA, "llama-3.1-8b-instant", "Llama 3.1 8B", 0.3),
            (ModelProvider.GROQ_MIXTRAL, "mixtral-8x7b-32768", "Mixtral 8x7B", 0.4),
            (ModelProvider.GROQ_GEMMA, "gemma2-9b-it", "Gemma 2 9B", 0.5),
            (ModelProvider.GOOGLE_GEMINI, "gemini-1.5-pro", "Gemini Pro", 0.3),
            (ModelProvider.GOOGLE_GEMINI_FLASH, "gemini-1.5-flash", "Gemini Flash", 0.2),
            (ModelProvider.GOOGLE_GEMINI_FLASH, "gemini-2.0-flash-exp", "Gemini 2.0", 0.4),
        ]
        
        self.personalities = [
            "Speed focused - minimal words",
            "Analytical and thorough",
            "Risk-taking and bold",
            "Careful and accurate",
            "Creative problem solver",
            "Ultra-concise minimalist",
            "Educational explainer",
            "Intuitive pattern recognizer",
            "Logical step-by-step",
            "Unpredictable wildcard",
        ]
    
    def initialize_clients(self, groq_key, google_key):
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)
        if google_key:
            genai.configure(api_key=google_key)
            self.google_available = True
    
    def get_available_providers(self):
        available = []
        if self.groq_client:
            available.extend([ModelProvider.GROQ_LLAMA, ModelProvider.GROQ_MIXTRAL, ModelProvider.GROQ_GEMMA])
        if self.google_available:
            available.extend([ModelProvider.GOOGLE_GEMINI, ModelProvider.GOOGLE_GEMINI_FLASH])
        return available
    
    def initialize_agents(self, count: int):
        self.agents = []
        available_providers = self.get_available_providers()
        
        if not available_providers:
            return False
        
        available_configs = [c for c in self.model_configs if c[0] in available_providers]
        
        for i in range(count):
            provider, model_name, display_name, base_temp = available_configs[i % len(available_configs)]
            personality = self.personalities[i % len(self.personalities)]
            temp_variation = (i * 0.1) % 0.5
            temperature = min(base_temp + temp_variation, 1.0)
            anonymous_id = self.code_names[i % len(self.code_names)]
            
            agent = Agent(
                id=i, name=f"{display_name} #{i}", anonymous_id=anonymous_id,
                personality=personality, temperature=temperature,
                model_provider=provider, model_name=model_name
            )
            self.agents.append(agent)
        return True
    
    async def ask_agent(self, agent: Agent, question: str) -> RoundResult:
        if not agent.is_alive:
            return RoundResult(agent.id, agent.anonymous_id, "", 0.0, False, error="Eliminated")
        
        start_time = time.time()
        
        try:
            if agent.model_provider in [ModelProvider.GROQ_LLAMA, ModelProvider.GROQ_MIXTRAL, ModelProvider.GROQ_GEMMA]:
                response = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model=agent.model_name,
                    messages=[
                        {"role": "system", "content": agent.personality},
                        {"role": "user", "content": question}
                    ],
                    temperature=agent.temperature,
                    max_tokens=500
                )
                answer = response.choices[0].message.content.strip()
            else:
                model = genai.GenerativeModel(agent.model_name, generation_config={"temperature": agent.temperature, "max_output_tokens": 500})
                response = await asyncio.to_thread(model.generate_content, f"{agent.personality}\n\n{question}")
                answer = response.text.strip()
            
            time_taken = time.time() - start_time
            return RoundResult(agent.id, agent.anonymous_id, answer, time_taken, False)
        
        except Exception as e:
            return RoundResult(agent.id, agent.anonymous_id, "", time.time() - start_time, False, error=str(e))
    
    async def judge_answers(self, question: str, results: List[RoundResult], correct_answer: Optional[str]) -> List[RoundResult]:
        shuffled = results.copy()
        random.shuffle(shuffled)
        
        is_open_ended = not correct_answer or correct_answer.strip() == ""
        
        if is_open_ended:
            judge_instruction = "For this OPEN-ENDED question, be LENIENT. Accept any thoughtful response. Mark as correct: true if reasonable."
        else:
            judge_instruction = f"Correct answer is: {correct_answer}. Mark as correct: true only if it matches."
        
        judge_prompt = f"""Question: "{question}"
{judge_instruction}

Rate each response (0-10 scale):
1. ACCURACY: Factually correct?
2. INSIGHT: Depth and clarity?

Responses:
"""
        for r in shuffled:
            if not r.error and r.answer:
                judge_prompt += f"\n--- {r.anonymous_id} ---\n{r.answer}\n"
        
        judge_prompt += '\n\nRespond ONLY with JSON:\n[{"agent": "Alpha", "accuracy": 8.5, "insight": 7.0, "correct": true}, ...]'
        
        try:
            if self.groq_client:
                response = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Impartial judge. Respond ONLY with valid JSON."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    temperature=0.1, max_tokens=1000
                )
                judgment_text = response.choices[0].message.content.strip()
            elif self.google_available:
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = await asyncio.to_thread(model.generate_content, judge_prompt)
                judgment_text = response.text.strip()
            else:
                return results
            
            if "```json" in judgment_text:
                judgment_text = judgment_text.split("```json")[1].split("```")[0]
            elif "```" in judgment_text:
                judgment_text = judgment_text.split("```")[1].split("```")[0]
            
            ratings = json.loads(judgment_text.strip())
            
            for rating in ratings:
                agent_code = rating.get("agent")
                for result in results:
                    if result.anonymous_id == agent_code:
                        result.accuracy_rating = rating.get("accuracy", 0)
                        result.insight_rating = rating.get("insight", 0)
                        result.is_correct = rating.get("correct", False)
                        break
            return results
        
        except Exception as e:
            st.warning(f"Judge error: {e}. Using fallback.")
            for result in results:
                if result.answer and len(result.answer.strip()) > 20:
                    result.is_correct = True if not correct_answer else (correct_answer.lower() in result.answer.lower())
                    result.accuracy_rating = 7.0
                    result.insight_rating = 7.0
            return results
    
    async def chairman_compile(self, question: str, results: List[RoundResult]) -> ChairmanSummary:
        correct_results = [r for r in results if r.is_correct and not r.error]
        
        if not correct_results:
            return ChairmanSummary(question, "No correct answers.", "All failed.", [], "Low")
        
        sorted_results = sorted(correct_results, key=lambda x: (x.accuracy_rating + x.insight_rating - x.time_taken/10), reverse=True)
        
        chairman_prompt = f'Question: "{question}"\n\nCorrect responses:\n'
        for r in sorted_results[:5]:
            chairman_prompt += f"\n--- {r.anonymous_id} ---\nAcc:{r.accuracy_rating} Ins:{r.insight_rating} Time:{r.time_taken:.3f}s\n{r.answer}\n"
        
        chairman_prompt += '\n\nSynthesize ONE comprehensive answer. JSON format:\n{"final_answer": "...", "reasoning": "...", "top_contributors": [...], "confidence": "High/Medium/Low"}'
        
        try:
            if self.groq_client:
                response = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Chairman. Synthesize best answer. Respond ONLY with JSON."},
                        {"role": "user", "content": chairman_prompt}
                    ],
                    temperature=0.2, max_tokens=1500
                )
                text = response.choices[0].message.content.strip()
            elif self.google_available:
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = await asyncio.to_thread(model.generate_content, chairman_prompt)
                text = response.text.strip()
            else:
                best = sorted_results[0]
                return ChairmanSummary(question, best.answer, "Used top answer", [best.anonymous_id], "Medium")
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text.strip())
            return ChairmanSummary(
                question, data.get("final_answer", ""), data.get("reasoning", ""),
                data.get("top_contributors", []), data.get("confidence", "Medium")
            )
        
        except Exception as e:
            st.warning(f"Chairman error: {e}")
            best = sorted_results[0]
            return ChairmanSummary(question, best.answer, "Fallback to top answer", [best.anonymous_id], "Medium")


def create_leaderboard_chart(agents):
    alive = [a for a in agents if a.is_alive]
    if not alive:
        return None
    
    df = pd.DataFrame([{
        'Agent': a.anonymous_id,
        'Score': a.accuracy_score + a.insight_score,
        'Wins': a.wins,
        'Avg Time': a.total_time / a.wins if a.wins > 0 else 0
    } for a in sorted(alive, key=lambda x: (x.accuracy_score + x.insight_score), reverse=True)[:10]])
    
    fig = px.bar(df, x='Agent', y='Score', color='Score',
                 color_continuous_scale='Viridis',
                 title='Top 10 Agents by Total Score')
    fig.update_layout(showlegend=False, height=400)
    return fig


def create_model_performance_chart(agents):
    model_stats = {}
    for agent in agents:
        model = agent.model_name
        if model not in model_stats:
            model_stats[model] = {'wins': 0, 'score': 0, 'alive': 0}
        model_stats[model]['wins'] += agent.wins
        model_stats[model]['score'] += agent.accuracy_score + agent.insight_score
        if agent.is_alive:
            model_stats[model]['alive'] += 1
    
    df = pd.DataFrame([{'Model': k, 'Wins': v['wins'], 'Score': v['score'], 'Alive': v['alive']} 
                       for k, v in model_stats.items()])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Wins', x=df['Model'], y=df['Wins'], marker_color='lightsalmon'))
    fig.add_trace(go.Bar(name='Total Score', x=df['Model'], y=df['Score'], marker_color='lightblue'))
    fig.update_layout(title='Model Performance Comparison', barmode='group', height=400)
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🏹 AI HUNGER GAMES 🏹</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">May the smartest AI win!</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'arena' not in st.session_state:
        st.session_state.arena = ArenaUI()
        st.session_state.initialized = False
        st.session_state.round_results = []
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        groq_key = st.text_input("🔑 Groq API Key", type="password", help="Get free key at console.groq.com")
        google_key = st.text_input("🔑 Google API Key", type="password", help="Get free key at makersuite.google.com")
        
        num_agents = st.slider("👥 Number of Agents", 2, 20, 10)
        
        if st.button("🚀 Initialize Arena"):
            if groq_key or google_key:
                st.session_state.arena.initialize_clients(groq_key, google_key)
                if st.session_state.arena.initialize_agents(num_agents):
                    st.session_state.initialized = True
                    st.success(f"✅ {num_agents} agents initialized!")
                else:
                    st.error("❌ Failed to initialize agents")
            else:
                st.error("❌ Please provide at least one API key")
        
        st.divider()
        
        if st.session_state.initialized:
            st.metric("🎮 Round", st.session_state.arena.round_number)
            alive_count = sum(1 for a in st.session_state.arena.agents if a.is_alive)
            st.metric("👥 Alive", alive_count)
            st.metric("💀 Eliminated", len(st.session_state.arena.agents) - alive_count)
    
    # Main content
    if not st.session_state.initialized:
        st.info("👈 Configure API keys and initialize the arena to begin!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="stat-box">
                <h3>⚡ Real-Time</h3>
                <p>Watch AI models compete live</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="stat-box">
                <h3>🎯 Fair Judging</h3>
                <p>Blind evaluation, no bias</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="stat-box">
                <h3>🏆 Best Answer</h3>
                <p>Chairman synthesizes results</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # Question input
    st.header("❓ Ask a Question")
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Enter your question:", placeholder="e.g., What is quantum computing?")
    with col2:
        correct_answer = st.text_input("Correct Answer (optional):", placeholder="Leave empty for open-ended")
    
    if st.button("🎬 START ROUND", type="primary"):
        if question:
            async def run_round():
                st.session_state.arena.round_number += 1
                
                # Show agents racing
                with st.spinner("⚔️ Agents are racing..."):
                    alive_agents = [a for a in st.session_state.arena.agents if a.is_alive]
                    tasks = [st.session_state.arena.ask_agent(agent, question) for agent in alive_agents]
                    results = await asyncio.gather(*tasks)
                
                # Show all responses
                st.subheader("📝 All Responses Received")
                cols = st.columns(3)
                for idx, result in enumerate([r for r in results if not r.error and r.answer]):
                    with cols[idx % 3]:
                        with st.expander(f"Agent {result.anonymous_id}"):
                            st.write(result.answer[:200] + ("..." if len(result.answer) > 200 else ""))
                
                # Judging
                with st.spinner("⚖️ Blind judging in progress..."):
                    results = await st.session_state.arena.judge_answers(question, results, correct_answer)
                
                # Results
                correct_results = [r for r in results if r.is_correct]
                incorrect_results = [r for r in results if not r.is_correct]
                
                st.subheader("🏁 Round Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### ✅ Correct Answers")
                    sorted_correct = sorted(correct_results, key=lambda x: (x.accuracy_rating + x.insight_rating), reverse=True)
                    for r in sorted_correct:
                        score = r.accuracy_rating + r.insight_rating
                        st.markdown(f"""
                        <div class="agent-card">
                            <b>Agent {r.anonymous_id}</b><br>
                            Score: {score:.1f} | Acc: {r.accuracy_rating:.1f} | Ins: {r.insight_rating:.1f}<br>
                            Time: {r.time_taken:.3f}s
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### ❌ Eliminated")
                    for r in incorrect_results:
                        agent = st.session_state.arena.agents[r.agent_id]
                        agent.is_alive = False
                        agent.eliminations += 1
                        st.markdown(f"""
                        <div class="agent-card eliminated-card">
                            <b>💀 Agent {r.anonymous_id}</b>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Update scores
                for r in correct_results:
                    agent = st.session_state.arena.agents[r.agent_id]
                    agent.accuracy_score += r.accuracy_rating
                    agent.insight_score += r.insight_rating
                    agent.wins += 1
                    agent.total_time += r.time_taken
                
                # Chairman's answer
                with st.spinner("👔 Chairman compiling final answer..."):
                    chairman_summary = await st.session_state.arena.chairman_compile(question, results)
                
                st.markdown("---")
                st.markdown(f"""
                <div class="chairman-box">
                    <h2>👔 CHAIRMAN'S OFFICIAL ANSWER</h2>
                    <p style="font-size: 1.1em; line-height: 1.6;">{chairman_summary.final_answer}</p>
                    <hr>
                    <p><b>📊 Confidence:</b> {chairman_summary.confidence}</p>
                    <p><b>🏅 Top Contributors:</b> {', '.join(chairman_summary.top_contributors)}</p>
                    <p><b>💭 Reasoning:</b> {chairman_summary.reasoning}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Charts
                st.header("📊 Live Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = create_leaderboard_chart(st.session_state.arena.agents)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = create_model_performance_chart(st.session_state.arena.agents)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            asyncio.run(run_round())
        else:
            st.warning("⚠️ Please enter a question!")
    
    # Show current leaderboard
    if st.session_state.arena.agents:
        st.header("🏆 Current Leaderboard")
        alive = [a for a in st.session_state.arena.agents if a.is_alive]
        if alive:
            cols = st.columns(5)
            for idx, agent in enumerate(sorted(alive, key=lambda x: (x.accuracy_score + x.insight_score), reverse=True)[:5]):
                with cols[idx]:
                    total_score = agent.accuracy_score + agent.insight_score
                    st.markdown(f"""
                    <div class="agent-card winner-card">
                        <h3>#{idx + 1}</h3>
                        <b>Agent {agent.anonymous_id}</b><br>
                        Score: {total_score:.1f}<br>
                        Wins: {agent.wins}
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()