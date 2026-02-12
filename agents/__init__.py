# -*- coding: utf-8 -*-
"""
Agents package for Airfoil Lab Multi-Agent System.

Modules:
    - config: Agent configurations and system prompts
    - graph: LangGraph-based orchestration
"""

from .config import (
    AgentConfig,
    AgentRole,
    CONCEPT_MENTOR,
    ITERATION_ENGINEER,
    STRATEGY_ANALYST,
    ALL_AGENTS,
    get_agent_config,
)

from .graph import (
    AgentState,
    agent_graph,
    run_agent,
)

__all__ = [
    "AgentConfig",
    "AgentRole",
    "CONCEPT_MENTOR",
    "ITERATION_ENGINEER", 
    "STRATEGY_ANALYST",
    "ALL_AGENTS",
    "get_agent_config",
    "AgentState",
    "agent_graph",
    "run_agent",
]
