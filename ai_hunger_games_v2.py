import asyncio
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import os
from datetime import datetime
from enum import Enum

# You'll need: pip install groq google-generativeai aiohttp
from groq import Groq
import google.generativeai as genai
import aiohttp

class ModelProvider(Enum):
    """Different LLM providers"""
    GROQ_LLAMA = "groq_llama"
    GROQ_MIXTRAL = "groq_mixtral"
    GROQ_GEMMA = "groq_gemma"
    GOOGLE_GEMINI = "google_gemini"
    GOOGLE_GEMINI_FLASH = "google_gemini_flash"

@dataclass
class Agent:
    """Represents a contestant in the Hunger Games"""
    id: int
    name: str
    personality: str
    temperature: float
    model_provider: ModelProvider
    model_name: str
    wins: int = 0
    eliminations: int = 0
    total_time: float = 0.0
    is_alive: bool = True

@dataclass
class RoundResult:
    """Stores the result of one agent's attempt"""
    agent_id: int
    answer: str
    time_taken: float
    is_correct: bool
    error: Optional[str] = None

class MultiModelArena:
    """The Game Master - orchestrates multi-model competition"""
    
    def __init__(self, groq_api_key: Optional[str] = None, google_api_key: Optional[str] = None):
        # Initialize clients
        self.groq_client = None
        self.google_available = False
        
        if groq_api_key and groq_api_key != "your-groq-api-key-here":
            self.groq_client = Groq(api_key=groq_api_key)
        
        if google_api_key and google_api_key != "your-google-api-key-here":
            genai.configure(api_key=google_api_key)
            self.google_available = True
        
        self.agents: List[Agent] = []
        self.round_number = 0
        self.history: List[Dict] = []
        
        # Model configurations with their providers
        self.model_configs = [
            # Groq models (all free!)
            (ModelProvider.GROQ_LLAMA, "llama-3.3-70b-versatile", "Groq Llama 3.3 70B", 0.2),
            (ModelProvider.GROQ_LLAMA, "llama-3.1-8b-instant", "Groq Llama 3.1 8B", 0.3),
            (ModelProvider.GROQ_MIXTRAL, "mixtral-8x7b-32768", "Groq Mixtral 8x7B", 0.4),
            (ModelProvider.GROQ_GEMMA, "gemma2-9b-it", "Groq Gemma 2 9B", 0.5),
            
            # Google Gemini models (free tier available!)
            (ModelProvider.GOOGLE_GEMINI, "gemini-1.5-pro", "Google Gemini Pro", 0.3),
            (ModelProvider.GOOGLE_GEMINI_FLASH, "gemini-1.5-flash", "Google Gemini Flash", 0.2),
            (ModelProvider.GOOGLE_GEMINI_FLASH, "gemini-2.0-flash-exp", "Google Gemini 2.0 Flash", 0.4),
        ]
        
        # Personality templates
        self.personalities = [
            "You answer as fast as possible with minimal words. Be concise and direct.",
            "You are analytical and thorough. Explain your reasoning step by step.",
            "You take risks and make bold guesses quickly. Trust your intuition.",
            "You double-check everything before answering. Accuracy is paramount.",
            "You think outside the box with creative solutions. Be innovative.",
            "You use the absolute minimum words needed. Ultra-concise responses only.",
            "You like to teach and explain concepts clearly. Be educational.",
            "You rely on intuition and pattern recognition. Go with your gut.",
            "You use pure logic and step-by-step reasoning. Be methodical.",
            "You're unpredictable in your approach. Mix different strategies.",
        ]
    
    def get_available_providers(self) -> List[ModelProvider]:
        """Get list of available model providers"""
        available = []
        if self.groq_client:
            available.extend([
                ModelProvider.GROQ_LLAMA,
                ModelProvider.GROQ_MIXTRAL,
                ModelProvider.GROQ_GEMMA
            ])
        if self.google_available:
            available.extend([
                ModelProvider.GOOGLE_GEMINI,
                ModelProvider.GOOGLE_GEMINI_FLASH
            ])
        return available
    
    def initialize_agents(self, count: int = 10):
        """Spawn diverse tributes from multiple model providers"""
        self.agents = []
        available_providers = self.get_available_providers()
        
        if not available_providers:
            print("❌ No API keys configured! Please set GROQ_API_KEY or GOOGLE_API_KEY")
            return
        
        # Filter model configs to only available providers
        available_configs = [
            config for config in self.model_configs 
            if config[0] in available_providers
        ]
        
        print(f"\n🌟 Spawning {count} agents from multiple AI models...\n")
        
        for i in range(count):
            # Cycle through available models and personalities
            provider, model_name, display_name, base_temp = available_configs[i % len(available_configs)]
            personality = self.personalities[i % len(self.personalities)]
            
            # Add some temperature variation
            temp_variation = (i * 0.1) % 0.5
            temperature = min(base_temp + temp_variation, 1.0)
            
            agent = Agent(
                id=i,
                name=f"{display_name} #{i}",
                personality=personality,
                temperature=temperature,
                model_provider=provider,
                model_name=model_name
            )
            self.agents.append(agent)
            print(f"✨ {agent.name} (temp: {temperature:.1f})")
    
    async def ask_groq_agent(self, agent: Agent, question: str) -> RoundResult:
        """Ask a Groq-based agent"""
        start_time = time.time()
        
        try:
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
            
            time_taken = time.time() - start_time
            answer = response.choices[0].message.content.strip()
            
            return RoundResult(
                agent_id=agent.id,
                answer=answer,
                time_taken=time_taken,
                is_correct=False
            )
            
        except Exception as e:
            time_taken = time.time() - start_time
            return RoundResult(
                agent_id=agent.id,
                answer="",
                time_taken=time_taken,
                is_correct=False,
                error=str(e)
            )
    
    async def ask_google_agent(self, agent: Agent, question: str) -> RoundResult:
        """Ask a Google Gemini agent"""
        start_time = time.time()
        
        try:
            model = genai.GenerativeModel(
                model_name=agent.model_name,
                generation_config={
                    "temperature": agent.temperature,
                    "max_output_tokens": 500,
                }
            )
            
            # Combine personality with question
            full_prompt = f"{agent.personality}\n\nQuestion: {question}"
            
            response = await asyncio.to_thread(
                model.generate_content,
                full_prompt
            )
            
            time_taken = time.time() - start_time
            answer = response.text.strip()
            
            return RoundResult(
                agent_id=agent.id,
                answer=answer,
                time_taken=time_taken,
                is_correct=False
            )
            
        except Exception as e:
            time_taken = time.time() - start_time
            return RoundResult(
                agent_id=agent.id,
                answer="",
                time_taken=time_taken,
                is_correct=False,
                error=str(e)
            )
    
    async def ask_agent(self, agent: Agent, question: str) -> RoundResult:
        """Route to appropriate provider"""
        if not agent.is_alive:
            return RoundResult(agent.id, "", 0.0, False, "Agent eliminated")
        
        # Route based on provider
        if agent.model_provider in [ModelProvider.GROQ_LLAMA, ModelProvider.GROQ_MIXTRAL, ModelProvider.GROQ_GEMMA]:
            return await self.ask_groq_agent(agent, question)
        elif agent.model_provider in [ModelProvider.GOOGLE_GEMINI, ModelProvider.GOOGLE_GEMINI_FLASH]:
            return await self.ask_google_agent(agent, question)
        else:
            return RoundResult(agent.id, "", 0.0, False, "Unknown provider")
    
    async def judge_answer(self, question: str, answer: str, correct_answer: Optional[str] = None) -> bool:
        """Use a judge LLM to determine if answer is correct"""
        try:
            if correct_answer:
                judge_prompt = f"""Question: {question}
Agent's Answer: {answer}
Correct Answer: {correct_answer}

Is the agent's answer correct? Consider it correct if it conveys the same meaning, even with different wording.
Respond with ONLY 'CORRECT' or 'INCORRECT'."""
            else:
                judge_prompt = f"""Question: {question}
Answer: {answer}

Is this answer correct and reasonable? Be fair but strict.
Respond with ONLY 'CORRECT' or 'INCORRECT'."""
            
            # Use Groq for judging (fastest)
            if self.groq_client:
                response = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a fair judge. Only respond with CORRECT or INCORRECT."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                judgment = response.choices[0].message.content.strip().upper()
            elif self.google_available:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = await asyncio.to_thread(model.generate_content, judge_prompt)
                judgment = response.text.strip().upper()
            else:
                return False
            
            return "CORRECT" in judgment
            
        except Exception as e:
            print(f"⚠️ Judge error: {e}")
            return False
    
    async def run_round(self, question: str, correct_answer: Optional[str] = None):
        """Run one round of the Hunger Games"""
        self.round_number += 1
        print(f"\n{'='*70}")
        print(f"🎮 ROUND {self.round_number}: {question}")
        print(f"{'='*70}")
        
        alive_agents = [a for a in self.agents if a.is_alive]
        print(f"👥 Tributes remaining: {len(alive_agents)}")
        
        # Show model distribution
        model_counts = {}
        for agent in alive_agents:
            model_counts[agent.model_name] = model_counts.get(agent.model_name, 0) + 1
        print(f"🤖 Active models: {', '.join([f'{m}: {c}' for m, c in model_counts.items()])}")
        
        # Ask all agents in parallel
        print("\n⚔️ ALL MODELS RACING...\n")
        tasks = [self.ask_agent(agent, question) for agent in alive_agents]
        results = await asyncio.gather(*tasks)
        
        # Judge all answers
        print("⚖️ JUDGING ANSWERS...\n")
        for result in results:
            if result.error:
                result.is_correct = False
            else:
                result.is_correct = await self.judge_answer(question, result.answer, correct_answer)
        
        # Display results
        correct_results = [r for r in results if r.is_correct]
        incorrect_results = [r for r in results if not r.is_correct]
        
        print("✅ CORRECT ANSWERS:")
        for result in sorted(correct_results, key=lambda x: x.time_taken):
            agent = self.agents[result.agent_id]
            print(f"  {agent.name}: {result.time_taken:.3f}s")
            print(f"    └─ {result.answer[:80]}...")
        
        print(f"\n❌ ELIMINATED ({len(incorrect_results)}):")
        for result in incorrect_results:
            agent = self.agents[result.agent_id]
            agent.is_alive = False
            agent.eliminations += 1
            print(f"  💀 {agent.name}")
            if result.answer:
                print(f"    └─ {result.answer[:80]}...")
        
        # Award win to fastest correct agent
        if correct_results:
            fastest = min(correct_results, key=lambda x: x.time_taken)
            winner = self.agents[fastest.agent_id]
            winner.wins += 1
            winner.total_time += fastest.time_taken
            print(f"\n🏆 ROUND WINNER: {winner.name} ({fastest.time_taken:.3f}s)")
            print(f"\n📝 WINNING ANSWER:")
            print(f"{'-'*70}")
            print(f"{fastest.answer}")
            print(f"{'-'*70}")
        
        # Respawn eliminated agents with new personalities
        eliminated_count = len(incorrect_results)
        if eliminated_count > 0:
            print(f"\n♻️ RESPAWNING {eliminated_count} NEW TRIBUTES...")
            self.respawn_agents(eliminated_count)
    
    def respawn_agents(self, count: int):
        """Create new agents to replace eliminated ones"""
        current_max_id = max(a.id for a in self.agents)
        available_providers = self.get_available_providers()
        
        available_configs = [
            config for config in self.model_configs 
            if config[0] in available_providers
        ]
        
        for i in range(count):
            new_id = current_max_id + i + 1
            provider, model_name, display_name, base_temp = available_configs[new_id % len(available_configs)]
            personality = self.personalities[new_id % len(self.personalities)]
            temperature = min(base_temp + (new_id * 0.1) % 0.5, 1.0)
            
            new_agent = Agent(
                id=new_id,
                name=f"{display_name} #{new_id}",
                personality=personality,
                temperature=temperature,
                model_provider=provider,
                model_name=model_name
            )
            self.agents.append(new_agent)
            print(f"  ✨ {new_agent.name}")
    
    def show_leaderboard(self):
        """Display current standings with model breakdown"""
        print(f"\n{'='*70}")
        print("📊 LEADERBOARD")
        print(f"{'='*70}")
        
        alive = [a for a in self.agents if a.is_alive]
        dead = [a for a in self.agents if not a.is_alive]
        
        print("\n🟢 SURVIVORS:")
        for agent in sorted(alive, key=lambda x: (-x.wins, x.total_time)):
            avg_time = agent.total_time / agent.wins if agent.wins > 0 else 0
            print(f"  {agent.name}: {agent.wins} wins, {avg_time:.3f}s avg")
        
        # Model performance stats
        print(f"\n📈 MODEL PERFORMANCE:")
        model_stats = {}
        for agent in self.agents:
            model = agent.model_name
            if model not in model_stats:
                model_stats[model] = {"wins": 0, "alive": 0, "total": 0}
            model_stats[model]["total"] += 1
            model_stats[model]["wins"] += agent.wins
            if agent.is_alive:
                model_stats[model]["alive"] += 1
        
        for model, stats in sorted(model_stats.items(), key=lambda x: -x[1]["wins"]):
            print(f"  {model}: {stats['wins']} wins, {stats['alive']}/{stats['total']} alive")
        
        if dead:
            print("\n💀 FALLEN TRIBUTES (Top 5):")
            top_dead = sorted(dead, key=lambda x: -x.wins)[:5]
            for agent in top_dead:
                print(f"  {agent.name}: {agent.wins} wins before elimination")
    
    def get_champion(self) -> Optional[Agent]:
        """Find the ultimate champion"""
        alive = [a for a in self.agents if a.is_alive]
        if not alive:
            return None
        return max(alive, key=lambda x: (x.wins, -x.total_time))


def get_user_input() -> tuple[str, Optional[str]]:
    """Get question and optional answer from user"""
    print(f"\n{'='*70}")
    print("❓ ENTER YOUR QUESTION")
    print(f"{'='*70}")
    
    question = input("Question: ").strip()
    
    if not question:
        return "", None
    
    print("\n💡 Optional: Provide the correct answer for better judging")
    print("   (Press Enter to skip - judge will evaluate without it)")
    correct_answer = input("Correct Answer (optional): ").strip()
    
    return question, correct_answer if correct_answer else None


async def interactive_mode(arena: MultiModelArena):
    """Run in interactive mode with terminal input"""
    print(f"\n{'='*70}")
    print("🎮 INTERACTIVE MODE - MULTI-MODEL BATTLE")
    print(f"{'='*70}")
    print("Enter questions one by one. Type 'quit' or 'exit' to stop.")
    print("Type 'leaderboard' or 'stats' to see current standings.")
    print("Type 'champion' to declare the winner and exit.")
    
    while True:
        question, correct_answer = get_user_input()
        
        if not question:
            print("⚠️ Please enter a question!")
            continue
        
        # Check for commands
        question_lower = question.lower()
        
        if question_lower in ['quit', 'exit', 'q']:
            print("\n👋 Exiting Hunger Games...")
            break
        
        if question_lower in ['leaderboard', 'stats', 'lb']:
            arena.show_leaderboard()
            continue
        
        if question_lower in ['champion', 'winner', 'end']:
            champion = arena.get_champion()
            if champion:
                print(f"\n{'='*70}")
                print(f"👑 DECLARING CHAMPION: {champion.name}")
                print(f"{'='*70}")
                print(f"Model: {champion.model_name}")
                print(f"Total Wins: {champion.wins}")
                if champion.wins > 0:
                    print(f"Average Speed: {champion.total_time/champion.wins:.3f}s")
                print(f"Personality: {champion.personality}")
            break
        
        # Run the round
        await arena.run_round(question, correct_answer)
        arena.show_leaderboard()
        
        # Check if only one survivor remains
        alive_count = sum(1 for a in arena.agents if a.is_alive)
        if alive_count <= 1:
            print("\n🎉 Only one tribute remains!")
            champion = arena.get_champion()
            if champion:
                print(f"👑 CHAMPION: {champion.name} ({champion.model_name})")
            break


async def demo_mode(arena: MultiModelArena):
    """Run with pre-set questions for demo"""
    questions = [
        ("What is 15 x 17?", "255"),
        ("What is the capital of France?", "Paris"),
        ("How many planets are in our solar system?", "8"),
        ("What is 2 to the power of 8?", "256"),
        ("What year did World War 2 end?", "1945"),
    ]
    
    print(f"\n{'='*70}")
    print("🎬 DEMO MODE - Multi-Model Battle with preset questions")
    print(f"{'='*70}")
    
    for question, answer in questions:
        await arena.run_round(question, answer)
        arena.show_leaderboard()
        
        alive_count = sum(1 for a in arena.agents if a.is_alive)
        if alive_count <= 1:
            print("\n🎉 We have a final survivor!")
            break
        
        await asyncio.sleep(2)
    
    champion = arena.get_champion()
    if champion:
        print(f"\n{'='*70}")
        print(f"👑 THE ULTIMATE CHAMPION: {champion.name}")
        print(f"{'='*70}")
        print(f"Model: {champion.model_name}")
        print(f"Total Wins: {champion.wins}")
        if champion.wins > 0:
            print(f"Average Speed: {champion.total_time/champion.wins:.3f}s")
        print(f"Personality: {champion.personality}")


async def main():
    """Run the Multi-Model Hunger Games"""
    
    # Get API keys from environment
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-google-api-key-here")
    
    # Check if at least one API key is configured
    has_groq = GROQ_API_KEY != "your-groq-api-key-here"
    has_google = GOOGLE_API_KEY != "your-google-api-key-here"
    
    if not has_groq and not has_google:
        print("❌ No API keys configured!")
        print("\n🔑 You need at least one API key:")
        print("\n1. GROQ (Recommended - Fast & Free)")
        print("   Get key at: https://console.groq.com")
        print("   Set: export GROQ_API_KEY='your_key'")
        print("\n2. GOOGLE GEMINI (Free tier available)")
        print("   Get key at: https://makersuite.google.com/app/apikey")
        print("   Set: export GOOGLE_API_KEY='your_key'")
        print("\n💡 TIP: Use BOTH for maximum model diversity!")
        return
    
    # Welcome banner
    print(f"\n{'='*70}")
    print("🏹 AI AGENT HUNGER GAMES - MULTI-MODEL BATTLE 🏹")
    print(f"{'='*70}")
    
    # Show available providers
    print("\n🤖 Available AI Models:")
    if has_groq:
        print("  ✅ Groq (Llama 3.3, Llama 3.1, Mixtral, Gemma)")
    else:
        print("  ❌ Groq (no API key)")
    
    if has_google:
        print("  ✅ Google (Gemini Pro, Gemini Flash)")
    else:
        print("  ❌ Google (no API key)")
    
    print("\nDifferent models will compete head-to-head!")
    
    # Initialize the arena
    arena = MultiModelArena(GROQ_API_KEY, GOOGLE_API_KEY)
    
    # Ask for number of agents
    print("\nHow many agents should compete? (default: 10)")
    try:
        num_agents = input("Number of agents (press Enter for 10): ").strip()
        num_agents = int(num_agents) if num_agents else 10
        num_agents = max(2, min(num_agents, 50))
    except ValueError:
        num_agents = 10
    
    arena.initialize_agents(num_agents)
    
    if not arena.agents:
        print("❌ Could not initialize agents. Check your API keys.")
        return
    
    # Choose mode
    print(f"\n{'='*70}")
    print("CHOOSE MODE:")
    print("1. Interactive Mode - Enter your own questions")
    print("2. Demo Mode - Watch preset questions")
    print(f"{'='*70}")
    
    choice = input("Enter choice (1 or 2, default: 1): ").strip()
    
    if choice == "2":
        await demo_mode(arena)
    else:
        await interactive_mode(arena)
    
    print("\n🎮 Game Over! Thanks for playing!")
    print(f"🏆 May the best AI model win!")


if __name__ == "__main__":
    asyncio.run(main())