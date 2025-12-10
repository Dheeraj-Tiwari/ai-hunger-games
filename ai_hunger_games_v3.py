import asyncio
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import os
from datetime import datetime
from enum import Enum
import random
import hashlib

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
    anonymous_id: str  # Hidden identity for judging
    personality: str
    temperature: float
    model_provider: ModelProvider
    model_name: str
    wins: int = 0
    eliminations: int = 0
    total_time: float = 0.0
    accuracy_score: float = 0.0  # Track accuracy over time
    insight_score: float = 0.0   # Track insight quality
    is_alive: bool = True

@dataclass
class RoundResult:
    """Stores the result of one agent's attempt"""
    agent_id: int
    anonymous_id: str
    answer: str
    time_taken: float
    is_correct: bool
    accuracy_rating: float = 0.0  # 0-10 scale
    insight_rating: float = 0.0   # 0-10 scale
    error: Optional[str] = None

@dataclass
class ChairmanSummary:
    """The Chairman's compiled final answer"""
    question: str
    final_answer: str
    reasoning: str
    top_contributors: List[str]  # Anonymous IDs
    confidence: str

class MultiModelArena:
    """The Game Master - orchestrates multi-model competition with anonymization"""
    
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
        
        # Anonymous code names for judging (no model info leaked)
        self.code_names = [
            "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", 
            "Golf", "Hotel", "India", "Juliet", "Kilo", "Lima",
            "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo",
            "Sierra", "Tango", "Uniform", "Victor", "Whiskey", "X-ray",
            "Yankee", "Zulu", "Phoenix", "Dragon", "Falcon", "Tiger"
        ]
        
        # Model configurations
        self.model_configs = [
            (ModelProvider.GROQ_LLAMA, "llama-3.3-70b-versatile", "Groq Llama 3.3 70B", 0.2),
            (ModelProvider.GROQ_LLAMA, "llama-3.1-8b-instant", "Groq Llama 3.1 8B", 0.3),
            (ModelProvider.GROQ_MIXTRAL, "mixtral-8x7b-32768", "Groq Mixtral 8x7B", 0.4),
            (ModelProvider.GROQ_GEMMA, "gemma2-9b-it", "Groq Gemma 2 9B", 0.5),
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
    
    def generate_anonymous_id(self, agent_id: int, model_name: str) -> str:
        """Generate anonymous ID that doesn't reveal model info"""
        # Use code names that hide the model identity
        return self.code_names[agent_id % len(self.code_names)]
    
    def initialize_agents(self, count: int = 10):
        """Spawn diverse tributes from multiple model providers"""
        self.agents = []
        available_providers = self.get_available_providers()
        
        if not available_providers:
            print("❌ No API keys configured! Please set GROQ_API_KEY or GOOGLE_API_KEY")
            return
        
        available_configs = [
            config for config in self.model_configs 
            if config[0] in available_providers
        ]
        
        print(f"\n🌟 Spawning {count} anonymous agents...\n")
        
        for i in range(count):
            provider, model_name, display_name, base_temp = available_configs[i % len(available_configs)]
            personality = self.personalities[i % len(self.personalities)]
            temp_variation = (i * 0.1) % 0.5
            temperature = min(base_temp + temp_variation, 1.0)
            
            anonymous_id = self.generate_anonymous_id(i, model_name)
            
            agent = Agent(
                id=i,
                name=f"{display_name} #{i}",
                anonymous_id=anonymous_id,
                personality=personality,
                temperature=temperature,
                model_provider=provider,
                model_name=model_name
            )
            self.agents.append(agent)
            # Only show anonymous ID during initialization
            print(f"✨ Agent {anonymous_id} initialized")
    
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
                anonymous_id=agent.anonymous_id,
                answer=answer,
                time_taken=time_taken,
                is_correct=False
            )
            
        except Exception as e:
            time_taken = time.time() - start_time
            return RoundResult(
                agent_id=agent.id,
                anonymous_id=agent.anonymous_id,
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
            
            full_prompt = f"{agent.personality}\n\nQuestion: {question}"
            response = await asyncio.to_thread(model.generate_content, full_prompt)
            
            time_taken = time.time() - start_time
            answer = response.text.strip()
            
            return RoundResult(
                agent_id=agent.id,
                anonymous_id=agent.anonymous_id,
                answer=answer,
                time_taken=time_taken,
                is_correct=False
            )
            
        except Exception as e:
            time_taken = time.time() - start_time
            return RoundResult(
                agent_id=agent.id,
                anonymous_id=agent.anonymous_id,
                answer="",
                time_taken=time_taken,
                is_correct=False,
                error=str(e)
            )
    
    async def ask_agent(self, agent: Agent, question: str) -> RoundResult:
        """Route to appropriate provider"""
        if not agent.is_alive:
            return RoundResult(agent.id, agent.anonymous_id, "", 0.0, False, error="Agent eliminated")
        
        if agent.model_provider in [ModelProvider.GROQ_LLAMA, ModelProvider.GROQ_MIXTRAL, ModelProvider.GROQ_GEMMA]:
            return await self.ask_groq_agent(agent, question)
        elif agent.model_provider in [ModelProvider.GOOGLE_GEMINI, ModelProvider.GOOGLE_GEMINI_FLASH]:
            return await self.ask_google_agent(agent, question)
        else:
            return RoundResult(agent.id, agent.anonymous_id, "", 0.0, False, error="Unknown provider")
    
    async def blind_judge_answers(self, question: str, results: List[RoundResult], 
                                  correct_answer: Optional[str] = None) -> List[RoundResult]:
        """Judge answers blindly without knowing which model produced them"""
        
        # Shuffle results to prevent order bias
        shuffled_results = results.copy()
        random.shuffle(shuffled_results)
        
        judge_prompt = f"""You are judging responses to this question: "{question}"
{f'The correct answer is: {correct_answer}' if correct_answer else ''}

You will evaluate each response BLINDLY (you don't know which AI produced it).
Rate each response on TWO criteria (0-10 scale):

1. ACCURACY: Is the answer factually correct?
2. INSIGHT: Does it show depth, clarity, and useful explanation?

Here are the responses from different anonymous agents:

"""
        
        for i, result in enumerate(shuffled_results):
            if not result.error and result.answer:
                judge_prompt += f"\n--- Response {result.anonymous_id} ---\n{result.answer}\n"
        
        judge_prompt += """\n
Respond ONLY with a JSON array of ratings:
[
  {"agent": "Alpha", "accuracy": 8.5, "insight": 7.0, "correct": true},
  {"agent": "Bravo", "accuracy": 9.0, "insight": 8.5, "correct": true},
  ...
]

Be fair and unbiased. Judge only the content, not the style or source."""
        
        try:
            # Use the strongest model as judge
            if self.groq_client:
                response = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are an impartial judge. Respond ONLY with valid JSON."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                judgment_text = response.choices[0].message.content.strip()
            elif self.google_available:
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = await asyncio.to_thread(model.generate_content, judge_prompt)
                judgment_text = response.text.strip()
            else:
                return results
            
            # Parse JSON ratings
            # Remove markdown code blocks if present
            if "```json" in judgment_text:
                judgment_text = judgment_text.split("```json")[1].split("```")[0]
            elif "```" in judgment_text:
                judgment_text = judgment_text.split("```")[1].split("```")[0]
            
            ratings = json.loads(judgment_text.strip())
            
            # Apply ratings back to results
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
            print(f"⚠️ Judge error: {e}")
            # Fallback: simple correctness check
            for result in results:
                if correct_answer and result.answer:
                    result.is_correct = correct_answer.lower() in result.answer.lower()
                result.accuracy_rating = 5.0
                result.insight_rating = 5.0
            return results
    
    async def chairman_compile_answer(self, question: str, results: List[RoundResult]) -> ChairmanSummary:
        """The Chairman synthesizes all correct answers into one final response"""
        
        correct_results = [r for r in results if r.is_correct and not r.error]
        
        if not correct_results:
            return ChairmanSummary(
                question=question,
                final_answer="No correct answers were provided.",
                reasoning="All agents failed to answer correctly.",
                top_contributors=[],
                confidence="Low"
            )
        
        # Sort by combined score (accuracy + insight + speed bonus)
        sorted_results = sorted(
            correct_results, 
            key=lambda x: (x.accuracy_rating + x.insight_rating - x.time_taken/10), 
            reverse=True
        )
        
        chairman_prompt = f"""You are the CHAIRMAN of the AI Hunger Games.

Question asked: "{question}"

You have received these CORRECT responses from different anonymous agents:

"""
        
        for result in sorted_results[:5]:  # Top 5 answers
            chairman_prompt += f"""
--- Agent {result.anonymous_id} ---
Accuracy: {result.accuracy_rating}/10
Insight: {result.insight_rating}/10
Time: {result.time_taken:.3f}s
Answer: {result.answer}

"""
        
        chairman_prompt += """
Your job: Synthesize these responses into ONE comprehensive final answer.

Requirements:
1. Combine the best insights from multiple agents
2. Ensure accuracy by cross-referencing answers
3. Provide clear, well-structured response
4. Note which anonymous agents contributed most
5. Rate your confidence (High/Medium/Low)

Respond in this JSON format:
{
  "final_answer": "Your comprehensive synthesized answer here",
  "reasoning": "Explain how you combined the responses",
  "top_contributors": ["Agent1", "Agent2"],
  "confidence": "High/Medium/Low"
}
"""
        
        try:
            if self.groq_client:
                response = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are the Chairman. Synthesize the best answer. Respond ONLY with valid JSON."},
                        {"role": "user", "content": chairman_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )
                chairman_text = response.choices[0].message.content.strip()
            elif self.google_available:
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = await asyncio.to_thread(model.generate_content, chairman_prompt)
                chairman_text = response.text.strip()
            else:
                # Fallback: use best answer
                best = sorted_results[0]
                return ChairmanSummary(
                    question=question,
                    final_answer=best.answer,
                    reasoning="Using highest-rated answer",
                    top_contributors=[best.anonymous_id],
                    confidence="Medium"
                )
            
            # Parse chairman response
            if "```json" in chairman_text:
                chairman_text = chairman_text.split("```json")[1].split("```")[0]
            elif "```" in chairman_text:
                chairman_text = chairman_text.split("```")[1].split("```")[0]
            
            chairman_data = json.loads(chairman_text.strip())
            
            return ChairmanSummary(
                question=question,
                final_answer=chairman_data.get("final_answer", ""),
                reasoning=chairman_data.get("reasoning", ""),
                top_contributors=chairman_data.get("top_contributors", []),
                confidence=chairman_data.get("confidence", "Medium")
            )
            
        except Exception as e:
            print(f"⚠️ Chairman error: {e}")
            # Fallback
            best = sorted_results[0]
            return ChairmanSummary(
                question=question,
                final_answer=best.answer,
                reasoning="Used highest-rated response due to compilation error",
                top_contributors=[best.anonymous_id],
                confidence="Medium"
            )
    
    async def run_round(self, question: str, correct_answer: Optional[str] = None):
        """Run one round with blind judging and chairman compilation"""
        self.round_number += 1
        print(f"\n{'='*70}")
        print(f"🎮 ROUND {self.round_number}: {question}")
        print(f"{'='*70}")
        
        alive_agents = [a for a in self.agents if a.is_alive]
        print(f"👥 Anonymous agents competing: {len(alive_agents)}")
        
        # Ask all agents in parallel
        print("\n⚔️ ALL AGENTS RACING (identities hidden)...\n")
        tasks = [self.ask_agent(agent, question) for agent in alive_agents]
        results = await asyncio.gather(*tasks)
        
        # Blind judging (no model info revealed to judge)
        print("⚖️ BLIND JUDGING IN PROGRESS...\n")
        results = await self.blind_judge_answers(question, results, correct_answer)
        
        # Display anonymized results
        correct_results = [r for r in results if r.is_correct]
        incorrect_results = [r for r in results if not r.is_correct]
        
        print("✅ CORRECT ANSWERS (by combined score):")
        sorted_correct = sorted(
            correct_results, 
            key=lambda x: (x.accuracy_rating + x.insight_rating - x.time_taken/10),
            reverse=True
        )
        
        for result in sorted_correct:
            combined_score = result.accuracy_rating + result.insight_rating
            print(f"  Agent {result.anonymous_id}: Score={combined_score:.1f} "
                  f"(Acc:{result.accuracy_rating:.1f}, Ins:{result.insight_rating:.1f}, "
                  f"Time:{result.time_taken:.3f}s)")
        
        print(f"\n❌ ELIMINATED ({len(incorrect_results)}):")
        for result in incorrect_results:
            agent = self.agents[result.agent_id]
            agent.is_alive = False
            agent.eliminations += 1
            print(f"  💀 Agent {result.anonymous_id}")
        
        # Update agent scores
        for result in correct_results:
            agent = self.agents[result.agent_id]
            agent.accuracy_score += result.accuracy_rating
            agent.insight_score += result.insight_rating
            agent.wins += 1
            agent.total_time += result.time_taken
        
        # Chairman compiles final answer
        print(f"\n👔 CHAIRMAN COMPILING FINAL ANSWER...\n")
        chairman_summary = await self.chairman_compile_answer(question, results)
        
        # Display Chairman's final answer
        print(f"{'='*70}")
        print("📜 CHAIRMAN'S OFFICIAL ANSWER")
        print(f"{'='*70}")
        print(f"\n{chairman_summary.final_answer}\n")
        print(f"{'-'*70}")
        print(f"📊 Confidence: {chairman_summary.confidence}")
        print(f"🏅 Top Contributors: {', '.join(chairman_summary.top_contributors)}")
        print(f"💭 Reasoning: {chairman_summary.reasoning}")
        print(f"{'='*70}")
        
        # Respawn eliminated agents
        eliminated_count = len(incorrect_results)
        if eliminated_count > 0:
            print(f"\n♻️ RESPAWNING {eliminated_count} NEW AGENTS...")
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
            anonymous_id = self.generate_anonymous_id(new_id, model_name)
            
            new_agent = Agent(
                id=new_id,
                name=f"{display_name} #{new_id}",
                anonymous_id=anonymous_id,
                personality=personality,
                temperature=temperature,
                model_provider=provider,
                model_name=model_name
            )
            self.agents.append(new_agent)
            print(f"  ✨ Agent {anonymous_id} spawned")
    
    def show_leaderboard(self, reveal_identities: bool = False):
        """Display current standings"""
        print(f"\n{'='*70}")
        print("📊 LEADERBOARD")
        print(f"{'='*70}")
        
        alive = [a for a in self.agents if a.is_alive]
        
        print("\n🟢 TOP SURVIVORS (by accuracy + insight):")
        sorted_alive = sorted(alive, key=lambda x: (x.accuracy_score + x.insight_score), reverse=True)
        
        for i, agent in enumerate(sorted_alive[:10], 1):
            total_score = agent.accuracy_score + agent.insight_score
            avg_time = agent.total_time / agent.wins if agent.wins > 0 else 0
            
            if reveal_identities:
                print(f"  {i}. {agent.name} (Agent {agent.anonymous_id})")
            else:
                print(f"  {i}. Agent {agent.anonymous_id}")
            
            print(f"      Score: {total_score:.1f} | Wins: {agent.wins} | Avg Time: {avg_time:.3f}s")
        
        # Model performance (only if revealing)
        if reveal_identities:
            print(f"\n📈 MODEL PERFORMANCE:")
            model_stats = {}
            for agent in self.agents:
                model = agent.model_name
                if model not in model_stats:
                    model_stats[model] = {"wins": 0, "score": 0, "alive": 0, "total": 0}
                model_stats[model]["total"] += 1
                model_stats[model]["wins"] += agent.wins
                model_stats[model]["score"] += agent.accuracy_score + agent.insight_score
                if agent.is_alive:
                    model_stats[model]["alive"] += 1
            
            for model, stats in sorted(model_stats.items(), key=lambda x: -x[1]["score"]):
                print(f"  {model}: Score={stats['score']:.1f}, "
                      f"Wins={stats['wins']}, Alive={stats['alive']}/{stats['total']}")
    
    def get_champion(self) -> Optional[Agent]:
        """Find the ultimate champion"""
        alive = [a for a in self.agents if a.is_alive]
        if not alive:
            return None
        return max(alive, key=lambda x: (x.accuracy_score + x.insight_score, -x.total_time))


def get_user_input() -> tuple[str, Optional[str]]:
    """Get question and optional answer from user"""
    print(f"\n{'='*70}")
    print("❓ ENTER YOUR QUESTION")
    print(f"{'='*70}")
    
    question = input("Question: ").strip()
    
    if not question:
        return "", None
    
    print("\n💡 Optional: Provide the correct answer for better judging")
    print("   (Press Enter to skip)")
    correct_answer = input("Correct Answer (optional): ").strip()
    
    return question, correct_answer if correct_answer else None


async def interactive_mode(arena: MultiModelArena):
    """Run in interactive mode with blind judging"""
    print(f"\n{'='*70}")
    print("🎮 INTERACTIVE MODE - BLIND COMPETITION")
    print(f"{'='*70}")
    print("All agents are anonymous. Judging is blind and fair.")
    print("\nCommands:")
    print("  'leaderboard' or 'stats' - See standings (anonymous)")
    print("  'reveal' - Show leaderboard with model identities")
    print("  'champion' - Declare winner and exit")
    print("  'quit' or 'exit' - Stop the games")
    
    while True:
        question, correct_answer = get_user_input()
        
        if not question:
            print("⚠️ Please enter a question!")
            continue
        
        question_lower = question.lower()
        
        if question_lower in ['quit', 'exit', 'q']:
            print("\n👋 Exiting Hunger Games...")
            break
        
        if question_lower in ['leaderboard', 'stats', 'lb']:
            arena.show_leaderboard(reveal_identities=False)
            continue
        
        if question_lower in ['reveal', 'show']:
            arena.show_leaderboard(reveal_identities=True)
            continue
        
        if question_lower in ['champion', 'winner', 'end']:
            champion = arena.get_champion()
            if champion:
                print(f"\n{'='*70}")
                print(f"👑 THE ULTIMATE CHAMPION")
                print(f"{'='*70}")
                print(f"Agent: {champion.anonymous_id}")
                print(f"Revealed Identity: {champion.name} ({champion.model_name})")
                print(f"Total Score: {champion.accuracy_score + champion.insight_score:.1f}")
                print(f"Wins: {champion.wins}")
                if champion.wins > 0:
                    print(f"Average Speed: {champion.total_time/champion.wins:.3f}s")
            break
        
        # Run the round
        await arena.run_round(question, correct_answer)
        arena.show_leaderboard(reveal_identities=False)
        
        alive_count = sum(1 for a in arena.agents if a.is_alive)
        if alive_count <= 1:
            print("\n🎉 Only one agent remains!")
            champion = arena.get_champion()
            if champion:
                print(f"👑 CHAMPION: Agent {champion.anonymous_id}")
                print(f"   Revealed: {champion.name}")
            break


async def main():
    """Run the Multi-Model Hunger Games with blind judging"""
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-google-api-key-here")
    
    has_groq = GROQ_API_KEY != "your-groq-api-key-here"
    has_google = GOOGLE_API_KEY != "your-google-api-key-here"
    
    if not has_groq and not has_google:
        print("❌ No API keys configured!")
        print("\nGet free keys:")
        print("  Groq: https://console.groq.com")
        print("  Google: https://makersuite.google.com/app/apikey")
        return
    
    print(f"\n{'='*70}")
    print("🏹 AI HUNGER GAMES - BLIND COMPETITION 🏹")
    print(f"{'='*70}")
    print("Fair judging. No favoritism. Pure performance.")
    
    arena = MultiModelArena(GROQ_API_KEY, GOOGLE_API_KEY)
    
    print("\nHow many agents should compete? (default: 10)")
    try:
        num_agents = input("Number of agents: ").strip()
        num_agents = int(num_agents) if num_agents else 10
        num_agents = max(2, min(num_agents, 30))
    except ValueError:
        num_agents = 10
    
    arena.initialize_agents(num_agents)
    
    if not arena.agents:
        print("❌ Could not initialize agents.")
        return
    
    await interactive_mode(arena)
    
    print("\n🎮 Game Over! Thanks for playing!")


if __name__ == "__main__":
    asyncio.run(main())