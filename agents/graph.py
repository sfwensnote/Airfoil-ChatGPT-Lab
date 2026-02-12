# -*- coding: utf-8 -*-
"""
Multi-Agent Orchestration using LangGraph
This module defines the state machine and agent workflow for the Airfoil Lab.
"""

from typing import TypedDict, Literal, Annotated, Sequence
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import operator
import os

from config import (
    ALL_AGENTS,
    ROUTER_PROMPT,
    get_agent_config,
    AgentRole,
)


# ============= State Definition =============
class AgentState(TypedDict):
    """The state that flows through the graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: AgentRole | None
    user_context: dict  # Contains geometry, environment, history etc.
    final_response: str | None


# ============= Helper Functions =============

# 自定义 API 配置
CUSTOM_API_BASE = "http://49.51.37.239:3006/v1"
CUSTOM_API_KEY = "sk-VrwOEEFLgjJwSOjH5pHRTDorgf0SmJVQrjK2D1uyjxZcfsrn"


def create_llm(model: str = "gpt-4o-mini", temperature: float = 0.7) -> ChatOpenAI:
    """Create a ChatOpenAI instance with custom API endpoint."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=CUSTOM_API_KEY,
        openai_api_base=CUSTOM_API_BASE,
    )


from langchain_huggingface import HuggingFaceEmbeddings

def get_retriever():
    """Get the vector store retriever."""
    # Use local embeddings as API doesn't support it
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Ensure items exist before loading to avoid errors if DB empty
    if not os.path.exists("knowledge_base"):
        return None
        
    vectorstore = Chroma(
        persist_directory="knowledge_base",
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})


def route_query(state: AgentState) -> AgentRole:
    """
    Route the user query to the appropriate agent.
    Uses a dedicated LLM call to classify the intent.
    """
    llm = create_llm(temperature=0)
    
    # Get the last user message
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_message = msg.content
            break
    
    if not last_message:
        return "concept_mentor"  # Default fallback
    
    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=last_message),
    ])
    
    route = response.content.strip().lower()
    
    # Validate the route
    if route in ALL_AGENTS:
        return route
    else:
        # Fallback to concept_mentor for ambiguous queries
        return "concept_mentor"


# ============= Agent Nodes =============
def concept_mentor_node(state: AgentState) -> dict:
    """Node for the Concept Mentor agent."""
    config = get_agent_config("concept_mentor")
    llm = create_llm(config.model, config.temperature)
    
    # Retrieve knowledge
    messages_list = list(state["messages"])
    last_human_msg = next((m for m in reversed(messages_list) if isinstance(m, HumanMessage)), None)
    
    knowledge_context = ""
    if last_human_msg:
        try:
            retriever = get_retriever()
            if retriever:
                docs = retriever.invoke(last_human_msg.content)
                if docs:
                    knowledge_context = "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Retrieval error: {e}")

    system_prompt = config.system_prompt
    if knowledge_context:
        system_prompt += f"\n\n## Retrieved Knowledge Base\nUse this information to answer if relevant:\n{knowledge_context}"

    messages = [SystemMessage(content=system_prompt)] + messages_list
    response = llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=response.content, name=config.display_name)],
        "current_agent": "concept_mentor",
        "final_response": response.content,
    }


def iteration_engineer_node(state: AgentState) -> dict:
    """Node for the Iteration Engineer agent."""
    config = get_agent_config("iteration_engineer")
    llm = create_llm(config.model, config.temperature)
    
    # Inject current context into the prompt
    context_str = ""
    if state.get("user_context"):
        ctx = state["user_context"]
        if "geometry" in ctx:
            g = ctx["geometry"]
            context_str += f"\n当前翼型: NACA {g.get('nacaCode', 'N/A')}, 弯度={g.get('camber', 0)}%, 厚度={g.get('thickness', 0)}%"
        if "environment" in ctx:
            e = ctx["environment"]
            context_str += f"\n环境: Re≈{e.get('re', 0):.0f}, α={e.get('alpha', 0)}°"
        if "kpi" in ctx:
            k = ctx["kpi"]
            context_str += f"\n当前性能: Cl={k.get('cl', 0):.3f}, Cd={k.get('cd', 0):.4f}, L/D={k.get('ld', 0):.1f}"
    
    enhanced_prompt = config.system_prompt
    if context_str:
        enhanced_prompt += f"\n\n## 用户当前状态{context_str}"
    
    messages = [SystemMessage(content=enhanced_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=response.content, name=config.display_name)],
        "current_agent": "iteration_engineer",
        "final_response": response.content,
    }


def strategy_analyst_node(state: AgentState) -> dict:
    """Node for the Strategy Analyst agent."""
    config = get_agent_config("strategy_analyst")
    llm = create_llm(config.model, config.temperature)
    
    # Inject history into the prompt
    history_str = ""
    if state.get("user_context") and "history" in state["user_context"]:
        history = state["user_context"]["history"]
        if history:
            history_str = "\n\n## 用户设计历史\n"
            for i, h in enumerate(history[-10:], 1):  # Last 10 entries
                history_str += f"{i}. NACA {h.get('naca_code', 'N/A')}, L/D={h.get('ld', 0):.1f}, Cl={h.get('cl', 0):.3f}\n"
    
    enhanced_prompt = config.system_prompt + history_str
    
    messages = [SystemMessage(content=enhanced_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=response.content, name=config.display_name)],
        "current_agent": "strategy_analyst",
        "final_response": response.content,
    }


# ============= Router Node =============
def router_node(state: AgentState) -> dict:
    """Determine which agent should handle the query."""
    selected_agent = route_query(state)
    return {"current_agent": selected_agent}


def route_to_agent(state: AgentState) -> str:
    """Conditional edge function: route to the selected agent."""
    return state.get("current_agent", "concept_mentor")


# ============= Build Graph =============
def build_agent_graph() -> StateGraph:
    """
    Build and compile the multi-agent graph.
    
    Graph Structure:
        START --> router --> [concept_mentor | iteration_engineer | strategy_analyst] --> END
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("concept_mentor", concept_mentor_node)
    workflow.add_node("iteration_engineer", iteration_engineer_node)
    workflow.add_node("strategy_analyst", strategy_analyst_node)
    
    # Add edges
    workflow.add_edge(START, "router")
    
    # Conditional routing from router to agents
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "concept_mentor": "concept_mentor",
            "iteration_engineer": "iteration_engineer",
            "strategy_analyst": "strategy_analyst",
        }
    )
    
    # All agents lead to END
    workflow.add_edge("concept_mentor", END)
    workflow.add_edge("iteration_engineer", END)
    workflow.add_edge("strategy_analyst", END)
    
    return workflow.compile()


# ============= Main Interface =============
# Compile the graph at module load
agent_graph = build_agent_graph()


async def run_agent(
    user_message: str,
    user_context: dict | None = None,
    conversation_history: list[BaseMessage] | None = None,
    preferred_agent: str | None = None,  # 用户指定的智能体 (覆盖自动路由)
) -> dict:
    """
    Main entry point for running the multi-agent system.
    
    Args:
        user_message: The user's query
        user_context: Optional context dict with geometry, environment, history
        conversation_history: Optional previous messages
        preferred_agent: If set, bypasses router and uses this agent directly
    
    Returns:
        dict with 'response', 'agent', 'messages'
    """
    messages = conversation_history or []
    messages.append(HumanMessage(content=user_message))
    
    # If user has specified an agent, invoke that agent directly (bypass router)
    if preferred_agent and preferred_agent in ALL_AGENTS:
        # Direct agent invocation - skip the router
        config = get_agent_config(preferred_agent)
        llm = create_llm(config.model, config.temperature)
        
        # Build context-enhanced prompt
        enhanced_prompt = config.system_prompt
        if user_context:
            ctx = user_context
            context_str = ""
            if "geometry" in ctx:
                g = ctx["geometry"]
                context_str += f"\n当前翼型: NACA {g.get('nacaCode', 'N/A')}, 弯度={g.get('camber', 0)}%, 厚度={g.get('thickness', 0)}%"
            if "environment" in ctx:
                e = ctx["environment"]
                context_str += f"\n环境: Re≈{e.get('re', 0):.0f}, α={e.get('alpha', 0)}°"
            if "kpi" in ctx:
                k = ctx["kpi"]
                context_str += f"\n当前性能: Cl={k.get('cl', 0):.3f}, Cd={k.get('cd', 0):.4f}, L/D={k.get('ld', 0):.1f}"
            if context_str:
                enhanced_prompt += f"\n\n## 用户当前状态{context_str}"
        
        llm_messages = [SystemMessage(content=enhanced_prompt)] + list(messages)
        response = await llm.ainvoke(llm_messages)
        
        return {
            "response": response.content,
            "agent": preferred_agent,
            "messages": messages + [AIMessage(content=response.content)],
        }
    
    # Otherwise, use the full graph with automatic routing
    initial_state: AgentState = {
        "messages": messages,
        "current_agent": None,
        "user_context": user_context or {},
        "final_response": None,
    }
    
    # Invoke the graph
    final_state = await agent_graph.ainvoke(initial_state)
    
    return {
        "response": final_state.get("final_response", ""),
        "agent": final_state.get("current_agent", "unknown"),
        "messages": final_state.get("messages", []),
    }


# ============= CLI Testing =============
if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Test concept question
        result = await run_agent("什么是雷诺数？为什么它对翼型设计很重要？")
        print(f"Agent: {result['agent']}")
        print(f"Response: {result['response'][:500]}...")
        print("---")
        
        # Test iteration question
        result = await run_agent(
            "帮我优化这个翼型，我想要更高的升阻比",
            user_context={
                "geometry": {"nacaCode": "2412", "camber": 2, "thickness": 12},
                "kpi": {"cl": 0.8, "cd": 0.015, "ld": 53.3}
            }
        )
        print(f"Agent: {result['agent']}")
        print(f"Response: {result['response'][:500]}...")
    
    asyncio.run(test())
