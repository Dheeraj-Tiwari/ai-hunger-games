import asyncio
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import os
from datetime import datetime

# You'll need: pip install groq aiohttp
from groq import Groq
import aiohttp

@dataclass
class Agent:
    """Represents a contestant in the Hunger Games"""
    id: int
    name: str
    personality: str
    temperature: float
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

class HungerGamesArena:
    """The Game Master - orchestrates the entire competition"""
    
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.agents: List[Agent] = []
        self.round_number = 0
        self.history: List[Dict] = []
        
        # Agent personality templates for diversity
        self.personalities = [
            ("Speed Demon", "You answer as fast as possible with minimal words. Be concise.", 0.1),
            ("The Professor", "You are analytical and thorough. Explain your reasoning.", 0.3),
            ("The Gambler", "You take risks and make bold guesses quickly.", 0.8),
            ("The Careful One", "You double-check everything before answering.", 0.2),
            ("The Creative", "You think outside the box with creative solutions.", 0.9),
            ("The Minimalist", "You use the absolute minimum words needed.", 0.1),
            ("The Explainer", "You like to teach and explain concepts clearly.", 0.4),
            ("The Intuitive", "You rely on intuition and pattern recognition.", 0.6),
            ("The Logical", "You use pure logic and step-by-step reasoning.", 0.2),
            ("The Wildcard", "You're unpredictable in your approach.", 1.0),
        ]
    
    def initialize_agents(self, count: int = 10):
        """Spawn the initial tributes"""
        self.agents = []
        for i in range(count):
            name, personality, temp = self.personalities[i % len(self.personalities)]
            agent = Agent(
                id=i,
                name=f"{name} #{i}",
                personality=personality,
                temperature=temp
            )
            self.agents.append(agent)
            print(f"✨ Spawned: {agent.name}")
    
    async def ask_agent(self, agent: Agent, question: str) -> RoundResult:
        """Send question to one agent and time their response"""
        if not agent.is_alive:
            return RoundResult(agent.id, "", 0.0, False, "Agent eliminated")
        
        start_time = time.time()
        
        try:
            # Using Groq's ultra-fast Llama model (FREE!)
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="llama-3.3-70b-versatile",  # Fast & free on Groq
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
                is_correct=False  # Will be judged later
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
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a fair judge. Only respond with CORRECT or INCORRECT."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            judgment = response.choices[0].message.content.strip().upper()
            return "CORRECT" in judgment
            
        except Exception as e:
            print(f"⚠️ Judge error: {e}")
            return False
    
    async def run_round(self, question: str, correct_answer: Optional[str] = None):
        """Run one round of the Hunger Games"""
        self.round_number += 1
        print(f"\n{'='*60}")
        print(f"🎮 ROUND {self.round_number}: {question}")
        print(f"{'='*60}")
        
        alive_agents = [a for a in self.agents if a.is_alive]
        print(f"👥 Tributes remaining: {len(alive_agents)}")
        
        # Ask all agents in parallel
        print("\n⚔️ AGENTS RACING...\n")
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
        
        # Respawn eliminated agents with new personalities
        eliminated_count = len(incorrect_results)
        if eliminated_count > 0:
            print(f"\n♻️ RESPAWNING {eliminated_count} NEW TRIBUTES...")
            self.respawn_agents(eliminated_count)
    
    def respawn_agents(self, count: int):
        """Create new agents to replace eliminated ones"""
        current_max_id = max(a.id for a in self.agents)
        
        for i in range(count):
            new_id = current_max_id + i + 1
            name, personality, temp = self.personalities[new_id % len(self.personalities)]
            new_agent = Agent(
                id=new_id,
                name=f"{name} #{new_id}",
                personality=personality,
                temperature=temp
            )
            self.agents.append(new_agent)
            print(f"  ✨ {new_agent.name}")
    
    def show_leaderboard(self):
        """Display current standings"""
        print(f"\n{'='*60}")
        print("📊 LEADERBOARD")
        print(f"{'='*60}")
        
        alive = [a for a in self.agents if a.is_alive]
        dead = [a for a in self.agents if not a.is_alive]
        
        print("\n🟢 SURVIVORS:")
        for agent in sorted(alive, key=lambda x: (-x.wins, x.total_time)):
            avg_time = agent.total_time / agent.wins if agent.wins > 0 else 0
            print(f"  {agent.name}: {agent.wins} wins, {avg_time:.3f}s avg")
        
        if dead:
            print("\n💀 FALLEN TRIBUTES:")
            top_dead = sorted(dead, key=lambda x: -x.wins)[:5]
            for agent in top_dead:
                print(f"  {agent.name}: {agent.wins} wins before elimination")
    
    def get_champion(self) -> Optional[Agent]:
        """Find the ultimate champion"""
        alive = [a for a in self.agents if a.is_alive]
        if not alive:
            return None
        return max(alive, key=lambda x: (x.wins, -x.total_time))


async def main():
    """Run the Hunger Games simulation"""
    
    # ⚠️ SET YOUR FREE GROQ API KEY HERE
    # Get it from: https://console.groq.com
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
    
    if GROQ_API_KEY == "your-groq-api-key-here":
        print("❌ Please set your GROQ_API_KEY!")
        print("Get free key at: https://console.groq.com")
        return
    
    # Initialize the arena
    arena = HungerGamesArena(GROQ_API_KEY)
    arena.initialize_agents(20)
    
    # Sample questions (add your own!)
    questions = [
        ("What is 15 x 17?", "255"),
        ("What is the capital of France?", "Paris"),
        ("How many planets are in our solar system?", "8"),
        ("What is 2 to the power of 8?", "256"),
        ("What year did World War 2 end?", "1945"),
    ]
    
    # Run multiple rounds
    for question, answer in questions:
        await arena.run_round(question, answer)
        arena.show_leaderboard()
        
        # Check if we should continue
        alive_count = sum(1 for a in arena.agents if a.is_alive)
        if alive_count <= 1:
            print("\n🎉 We have a final survivor!")
            break
        
        await asyncio.sleep(2)  # Pause between rounds
    
    # Declare champion
    champion = arena.get_champion()
    if champion:
        print(f"\n{'='*60}")
        print(f"👑 THE ULTIMATE CHAMPION: {champion.name}")
        print(f"{'='*60}")
        print(f"Total Wins: {champion.wins}")
        print(f"Average Speed: {champion.total_time/champion.wins:.3f}s")
        print(f"Personality: {champion.personality}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
