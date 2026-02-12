# -*- coding: utf-8 -*-
"""
Agent API Endpoint
Provides the /agent/chat endpoint for the multi-agent system.
"""

import os
import sys

# Add agents directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio

# Import from agents package
from agents.graph import run_agent

router = APIRouter(prefix="/agent", tags=["Agent"])


class AgentChatRequest(BaseModel):
    """Request body for agent chat."""
    message: str
    user_id: str = "guest"
    context: Optional[dict] = None
    preferred_agent: Optional[str] = None  # 用户选择的模块 (覆盖自动路由)
    # preferred_agent 可选值: "concept_mentor", "iteration_engineer", "strategy_analyst"
    # 如果为 None 或 "auto"，则使用自动路由
    # Context should include:
    # - geometry: {camber, thickness, maxCamberPos, maxThicknessPos, nacaCode}
    # - environment: {alpha, re, velocity, ...}
    # - kpi: {cl, cd, ld, alphaOpt, ldMax}
    # - history: [{naca_code, cl, cd, ld, ...}, ...]


class AgentChatResponse(BaseModel):
    """Response body for agent chat."""
    status: str
    agent: str
    agent_display_name: str
    response: str
    message: Optional[str] = None


# Agent display names mapping
AGENT_DISPLAY_NAMES = {
    "concept_mentor": "📚 概念学习导师",
    "iteration_engineer": "🔧 迭代工程师",
    "strategy_analyst": "📊 策略分析师",
}


@router.post("/chat", response_model=AgentChatResponse)
async def chat_with_agent(request: AgentChatRequest):
    """
    Main endpoint for chatting with the multi-agent system.
    
    Routing modes:
    - Auto: If preferred_agent is None or "auto", automatically routes based on intent
    - Manual: If preferred_agent is set, uses that agent directly (user override)
    """
    try:
        # Determine if user has a preference
        user_preference = None
        if request.preferred_agent and request.preferred_agent != "auto":
            # Map frontend module names to agent names
            module_to_agent = {
                "Concept Learning": "concept_mentor",
                "Model Iteration": "iteration_engineer",
                "Strategy Review": "strategy_analyst",
                "concept_mentor": "concept_mentor",
                "iteration_engineer": "iteration_engineer",
                "strategy_analyst": "strategy_analyst",
            }
            user_preference = module_to_agent.get(request.preferred_agent)
        
        result = await run_agent(
            user_message=request.message,
            user_context=request.context,
            preferred_agent=user_preference,  # Pass user preference
        )
        
        agent_name = result.get("agent", "unknown")
        
        return AgentChatResponse(
            status="success",
            agent=agent_name,
            agent_display_name=AGENT_DISPLAY_NAMES.get(agent_name, agent_name),
            response=result.get("response", ""),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )


@router.get("/status")
async def agent_status():
    """Check if the agent system is available."""
    try:
        # Simple health check
        return {
            "status": "ok",
            "agents": list(AGENT_DISPLAY_NAMES.keys()),
            "message": "Multi-agent system is ready",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }
