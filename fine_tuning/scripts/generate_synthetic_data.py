
import json
import os
import asyncio
from typing import List, Dict
import sys

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Reuse config from agents/graph.py (hardcoded here to avoid import issues if not in path)
CUSTOM_API_BASE = "http://49.51.37.239:3006/v1"
CUSTOM_API_KEY = "sk-VrwOEEFLgjJwSOjH5pHRTDorgf0SmJVQrjK2D1uyjxZcfsrn"

def create_llm():
    return ChatOpenAI(
        model="gpt-4o",  # Use a smart model for generation
        temperature=0.8,
        openai_api_key=CUSTOM_API_KEY,
        openai_api_base=CUSTOM_API_BASE,
    )

AGENTS = {
    "concept_mentor": {
        "topics": [
            "Reynolds Number significance in low speed flow",
            "Bernoulli's principle vs Newton's laws for lift",
            "Boundary layer separation and stall",
            "Differences between laminar and turbulent flow",
            "Meaning of NACA 4-digit series parameters",
            "Induced drag and aspect ratio",
            "Ground effect aerodynamics",
            "Magnus effect basics",
            "Pressure coefficient distribution interpretation",
            "Center of pressure vs Aerodynamic center"
        ],
        "system_prompt": "You are an expert aerodynamics professor. Generate diverse student questions and detailed, Socratic-style answers."
    },
    "iteration_engineer": {
        "topics": [
            "Reducing drag at high angles of attack",
            "Increasing maximum lift coefficient (Cl_max)",
            "Smoothing pressure recovery to prevent separation",
            "Adjusting camber for better L/D ratio",
            "Thickness distribution effects on stall characteristics",
            "Modifying leading edge radius",
            "Trailing edge angle adjustments",
            "Trade-offs between thickness and drag",
            "Optimizing for cruise vs takeoff",
            "XFOIL convergence issues troubleshooting"
        ],
        "system_prompt": "You are an airfoil design engineer. Generate realistic design problems where a user asks for improvement, and provide technical, actionable optimization advice including JSON parameter adjustments."
    },
    "strategy_analyst": {
        "topics": [
            "Analyzing a sequence of 5 design iterations",
            "Identifying why a design change failed",
            "Summarizing performance trends from test data",
            "Comparing high-lift vs low-drag design strategies",
            "Reviewing historic airfoil evolution"
        ],
        "system_prompt": "You are a data analyst reviewing airfoil design history. Generate scenarios where a user asks for a review of their design process, and provide a high-level strategic analysis."
    }
}

async def generate_samples(agent_role: str, count: int = 5):
    llm = create_llm()
    config = AGENTS[agent_role]
    samples = []
    
    print(f"Generating {count} samples for {agent_role}...")
    
    for i in range(count):
        topic = config["topics"][i % len(config["topics"])]
        
        prompt = f"""
        Generate a high-quality training sample for the '{agent_role}'.
        Topic: {topic}
        
        Format the output purely as a JSON object with these keys:
        - "instruction": The user's question or request (be diverse in tone).
        - "input": Optional context (or empty string).
        - "output": The ideal agent response (following the persona: {config['system_prompt']}).
        
        Ensure the JSON is valid. Do not include markdown formatting.
        """
        
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            samples.append(data)
            print(f"  - Generated sample {i+1}/{count} on '{topic}'")
        except Exception as e:
            print(f"  - Error generating sample {i+1}: {e}")
            
    return samples

async def main():
    all_data = {}
    
    # Generate 15 samples for concept, 15 for iteration, 10 for strategy
    tasks = [
        generate_samples("concept_mentor", 6), # Small batch for demo speed
        generate_samples("iteration_engineer", 6),
        generate_samples("strategy_analyst", 4)
    ]
    
    results = await asyncio.gather(*tasks)
    
    output_dir = "fine_tuning/data/synthetic"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual agent files
    roles = ["concept_mentor", "iteration_engineer", "strategy_analyst"]
    for role, data in zip(roles, results):
        filename = f"{output_dir}/{role}_synthetic.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} samples to {filename}")

if __name__ == "__main__":
    asyncio.run(main())
