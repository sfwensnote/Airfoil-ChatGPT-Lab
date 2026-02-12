# Airfoil Lab 多智能体系统完整设计文档

> **文档版本**: 1.0  
> **创建时间**: 2026-02-05  
> **系统状态**: ✅ 已部署并运行中

---

## 目录
1. [系统概述](#1-系统概述)
2. [架构设计](#2-架构设计)
3. [智能体角色设计](#3-智能体角色设计)
4. [LangGraph 工作流](#4-langgraph-工作流)
5. [训练数据准备](#5-训练数据准备)
6. [系统集成](#6-系统集成)
7. [使用指南](#7-使用指南)

---

## 1. 系统概述

### 1.1 设计目标
为 Airfoil Lab 翼型设计实验室构建一个智能对话系统，能够：
- **自动理解用户意图**，将问题路由到合适的专家
- **提供专业化的回答**，每个智能体专注于特定领域
- **保持上下文感知**，利用当前仿真数据增强回答

### 1.2 技术栈
| 组件 | 技术 |
|------|------|
| 智能体编排 | LangGraph (状态机) |
| LLM 框架 | LangChain |
| 后端 API | FastAPI |
| 前端 | Next.js + React |
| LLM 服务 | 自定义 API (http://49.51.37.239:3006/v1) |

### 1.3 系统现状

```
┌─────────────────────────────────────────────────────────┐
│  ✅ 多智能体系统已激活                                    │
│  • 概念学习导师 (concept_mentor)      ✓                  │
│  • 迭代工程师   (iteration_engineer)  ✓                  │
│  • 策略分析师   (strategy_analyst)    ✓                  │
│  • 训练样本: 30 条                                        │
│  • API 端点: /agent/chat                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 架构设计

### 2.1 混合路由架构 (Hybrid Routing)

系统支持两种路由模式：
- **自动路由**: 根据用户消息意图自动分发到合适的智能体
- **手动覆盖**: 用户在前端选择特定模块时，直接使用该智能体

```
                    ┌─────────────┐
                    │   用户输入   │
                    └──────┬──────┘
                           │
                           ▼
               ┌───────────────────────┐
               │   preferred_agent?    │
               │   检查用户是否指定模块  │
               └───────────┬───────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
    ┌─────────────┐               ┌─────────────┐
    │ 用户已选择   │               │ 自动路由     │
    │ 直接调用智能体│               │ Router 分类  │
    └──────┬──────┘               └──────┬──────┘
           │                               │
           └───────────────┬───────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  📚 概念学习导师  │ │  🔧 迭代工程师   │ │  📊 策略分析师   │
│  Concept Mentor │ │  Iteration Eng  │ │Strategy Analyst │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 2.2 路由决策逻辑

```python
# 在 graph.py 中的 run_agent 函数
if preferred_agent and preferred_agent in ALL_AGENTS:
    # 用户指定了智能体 → 直接调用，跳过 Router
    return direct_invoke(preferred_agent, message, context)
else:
    # 未指定 → 使用 LangGraph 自动路由
    return agent_graph.ainvoke(state)
```

### 2.3 前端模块映射

| 前端 Tab | 后端智能体 |
|----------|------------|
| Concept Learning | `concept_mentor` |
| Model Iteration | `iteration_engineer` |
| Strategy Review | `strategy_analyst` |

---

## 3. 智能体角色设计

### 3.1 角色与网站模块映射

| 网站模块 | 智能体 | 职责 |
|----------|--------|------|
| Concept Learning | 📚 概念学习导师 | 空气动力学理论教学 |
| Model Iteration | 🔧 迭代工程师 | 翼型参数优化建议 |
| Strategy Review | 📊 策略分析师 | 设计历史回顾分析 |

### 3.2 概念学习导师 (Concept Mentor)

**定位**: 空气动力学教授

**教学风格**:
- 苏格拉底式问答法
- 生动类比
- 循循善诱

**知识领域**:
- 伯努利原理、边界层理论
- NACA 翼型命名系统
- 升力/阻力/失速机理
- 雷诺数效应

**典型问题**:
```
用户: 什么是雷诺数？
智能体: 想象一只蚊子和一架波音747在空气中飞行...
        (使用类比解释，以问题结尾)
```

### 3.3 迭代工程师 (Iteration Engineer)

**定位**: 翼型设计工程师

**工作方式**:
1. 分析当前状态
2. 明确优化目标
3. 提出具体建议
4. 输出 JSON 参数

**输出格式**:
```
<thought>
分析用户当前翼型 NACA 2412...
弯度偏低，可以增加到 3%...
</thought>

建议调整：
1. 增加弯度到 3%
2. 减小厚度到 10%

```json
{
  "camber": 3.0,
  "thickness": 10.0,
  "maxCamberPos": 45.0
}
```
```

### 3.4 策略分析师 (Strategy Analyst)

**定位**: 数据分析师

**分析方法**:
1. 纵览全局设计历史
2. 识别关键转折点
3. 归纳有效/无效调整
4. 提供改进洞见

**输出格式**:
```markdown
## 迭代历史表格
| 迭代 | 翼型 | L/D | 变化 |
|------|------|-----|------|
| 1    | 2412 | 45  | -    |
| 2    | 3412 | 52  | +7   |

## 核心发现
- 增加弯度是正确方向
- 厚度增加导致性能下降
```

---

## 4. LangGraph 工作流

### 4.1 状态定义

```python
class AgentState(TypedDict):
    """流经状态图的状态"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: AgentRole | None
    user_context: dict  # 几何、环境、历史等
    final_response: str | None
```

### 4.2 状态图结构

```
                 START
                   │
                   ▼
            ┌──────────────┐
            │   router     │
            │  (意图分类)   │
            └──────┬───────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌────────┐   ┌────────┐   ┌────────┐
│concept │   │iteration│   │strategy│
│_mentor │   │_engineer│   │_analyst│
└────┬───┘   └────┬───┘   └────┬───┘
     │            │            │
     └────────────┼────────────┘
                  │
                  ▼
                 END
```

### 4.3 关键代码

**路由函数**:
```python
def route_query(state: AgentState) -> AgentRole:
    """使用 LLM 进行意图分类"""
    llm = create_llm(temperature=0)
    
    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=last_message),
    ])
    
    route = response.content.strip().lower()
    return route if route in ALL_AGENTS else "concept_mentor"
```

**图构建**:
```python
def build_agent_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("router", router_node)
    workflow.add_node("concept_mentor", concept_mentor_node)
    workflow.add_node("iteration_engineer", iteration_engineer_node)
    workflow.add_node("strategy_analyst", strategy_analyst_node)
    
    # 添加边
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges("router", route_to_agent, {...})
    workflow.add_edge("concept_mentor", END)
    workflow.add_edge("iteration_engineer", END)
    workflow.add_edge("strategy_analyst", END)
    
    return workflow.compile()
```

---

## 5. 训练数据准备

### 5.1 数据统计

| 角色 | 样本数量 | 格式 |
|------|----------|------|
| 概念学习导师 | 14 条 | Alpaca |
| 迭代工程师 | 11 条 | Alpaca |
| 策略分析师 | 5 条 | Alpaca |
| **总计** | **30 条** | - |

### 5.2 数据格式

**Alpaca 格式** (用于本地微调):
```json
{
  "instruction": "解释翼型的升力产生原理",
  "input": "为什么机翼能产生升力？",
  "output": "想象一下你把手伸出车窗..."
}
```

**OpenAI 格式** (用于 API 微调):
```json
{
  "messages": [
    {"role": "system", "content": "你是概念学习导师..."},
    {"role": "user", "content": "什么是雷诺数？"},
    {"role": "assistant", "content": "雷诺数是一个无量纲数..."}
  ]
}
```

### 5.3 数据生成流程

```
1. 定义主题模板 (15+ 概念主题, 8 优化场景, 3 分析模式)
           │
           ▼
2. 调用 LLM 生成样本 (generate_data.py)
           │
           ▼
3. 格式转换 (convert_data.py)
    ├── all_openai.jsonl (OpenAI 微调)
    ├── all_sharegpt.json (LLaMA-Factory)
    └── all_alpaca.json (Unsloth)
```

---

## 6. 系统集成

### 6.1 后端集成

**文件**: `agent_api.py`

```python
@router.post("/chat", response_model=AgentChatResponse)
async def chat_with_agent(request: AgentChatRequest):
    result = await run_agent(
        user_message=request.message,
        user_context=request.context,
    )
    return AgentChatResponse(
        status="success",
        agent=result["agent"],
        agent_display_name=AGENT_DISPLAY_NAMES[result["agent"]],
        response=result["response"],
    )
```

**文件**: `backend.py`

```python
from agent_api import router as agent_router
app.include_router(agent_router)
print("✅ Multi-Agent System loaded successfully")
```

### 6.2 前端集成

**文件**: `ChatWindow.tsx`

```typescript
const USE_MULTI_AGENT = true;  // 启用多智能体

// 调用多智能体 API
const agentResponse = await chatWithAgent({
    message: input.trim(),
    userId: userId,
    context: {
        geometry: {...},
        environment: {...},
        kpi: {...},
    },
});
```

### 6.3 API 配置

**文件**: `agents/graph.py`

```python
CUSTOM_API_BASE = "http://49.51.37.239:3006/v1"
CUSTOM_API_KEY = "sk-VrwOEEFLgjJwSOjH5pHRTDorgf0SmJVQrjK2D1uyjxZcfsrn"

def create_llm(model: str = "gpt-4o-mini", temperature: float = 0.7):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=CUSTOM_API_KEY,
        openai_api_base=CUSTOM_API_BASE,
    )
```

---

## 7. 使用指南

### 7.1 启动系统

```bash
# 启动后端
cd /Users/wensifan/bot-remote-windows
uvicorn backend:app --reload --host 0.0.0.0 --port 8000

# 启动前端
cd airfoil-lab-react
npm run dev
```

### 7.2 测试 API

```bash
# 检查状态
curl http://localhost:8000/agent/status

# 发送测试消息
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "什么是雷诺数？", "user_id": "test"}'
```

### 7.3 扩展训练数据

```bash
# 生成更多样本
python3 fine_tuning/scripts/generate_data.py --num 20

# 转换格式
python3 fine_tuning/scripts/convert_data.py
```

---

## 附录: 文件结构

```
bot-remote-windows/
├── agents/
│   ├── __init__.py           # 包初始化
│   ├── config.py             # 系统提示词配置
│   ├── graph.py              # LangGraph 状态图
│   └── requirements.txt      # Python 依赖
├── training_data/
│   ├── concept_mentor/samples.json
│   ├── iteration_engineer/samples.json
│   └── strategy_analyst/samples.json
├── fine_tuning/
│   ├── data/                 # 转换后的训练数据
│   └── scripts/              # 微调脚本
├── agent_api.py              # FastAPI 路由
└── backend.py                # 主后端入口
```

---

> **注意**: 当前系统使用的是 **Prompt Engineering** 方式实现专业化智能体，而非模型微调。训练数据已准备好，可随时进行微调升级。
