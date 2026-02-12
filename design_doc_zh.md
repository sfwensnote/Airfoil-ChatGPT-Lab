# Airfoil Lab 设计文档 (Design Document)

## 1. 系统架构 (System Architecture)

本项目采用了典型的 **前后端分离 (Client-Server)** 架构，旨在提供高性能、交互性强的翼型仿真与设计体验。

- **Frontend (前端)**: Next.js (React) + TypeScript + Tailwind CSS
- **Backend (后端)**: FastAPI (Python) + SQLite + SQLAlchemy
- **Simulation (仿真核心)**: XFOIL (C/Fortran, via subprocess) + Thin Airfoil Theory (Fallback estimation)

### 架构图 (Architecture Diagram)

```mermaid
graph TD
    User[用户 (User)] -->|HTTP/WebSocket| NextJS[前端 (Next.js)]
    NextJS -->|REST API| FastAPI[后端 (FastAPI)]
    
    subgraph "Backend Services"
        FastAPI -->|CRUD| DB[(SQLite Database)]
        FastAPI -->|Subprocess| XFOIL[XFOIL Solver]
        FastAPI -->|Calc| Theory[Thin Airfoil Theory]
    end
    
    subgraph "Frontend Components"
        NextJS --> Store[Zustand Store]
        NextJS --> Preview[AirfoilPreview (SVG)]
        NextJS --> Charts[PolarCharts (Recharts)]
    end
```

## 2. 技术栈 (Technology Stack)

### 前端 (Frontend)
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS, Shadcn UI (Radix Primitives)
- **State Management**: Zustand
- **Visualization**: Recharts (图表), SVG (翼型绘制)
- **Icons**: Lucide React

### 后端 (Backend)
- **Framework**: FastAPI
- **Database**: SQLite (通过 SQLAlchemy ORM)
- **Data Processing**: Pandas, NumPy
- **Server**: Uvicorn

## 3. 核心模块 (Key Modules)

### 3.1 翼型生成器 (Airfoil Generator)
- **位置**: `src/lib/geometry.ts` (前端) / `backend.py` (后端)
- **算法**: NACA 4位数翼型生成算法 (NACA 4-digit Series Generator)
- **输入**:
  - `m` (Max Camber): 最大弯度
  - `p` (Max Camber Position): 最大弯度位置
  - `t` (Thickness): 最大厚度
  - `tpos` (Max Thickness Position): 最大厚度位置 (非标准NACA，但在UI中可调)
- **输出**: 翼型上下表面坐标点 (X, Y)

### 3.2 仿真引擎 (Simulation Engine)
- **位置**: `backend.py` -> `run_xfoil_polar`
- **逻辑**:
  1. 接收前端传来的翼型参数和工况 (Re, Mach, Alpha)。
  2. 生成 `.dat` 坐标文件。
  3. 调用本地 `xfoil.exe` 进行计算。
  4. 解析 `polar.out` 获取 CL, CD, CM, CP。
  5. **Fallback**: 如果 XFOIL 失败（如不收敛或环境不支持），使用薄翼理论 (Thin Airfoil Theory) 进行估算。

### 3.3 实时交互 (Real-time Interaction)
- **机制**:
  1. 用户拖动滑块 -> 更新 `useSimulationStore` -> 触发 `useEffect`。
  2. 前端根据几何参数，**即时**使用简化公式估算 CL/CD 和 CP 分布。
  3. `AirfoilPreview` 组件根据 CP 数据实时渲染压力热力图 (Heatmap)。

### 3.4 数据存储 (Data Persistence)
- **表结构**:
  - `users`: 用户认证信息
  - `airfoil_history`: 仿真记录 (包含 几何参数 + 气动数据)
  - `conversations`: AI 辅导对话记录
- **流程**: 每次点击 "Run Accurate Simulation" (运行精确仿真)，后端会自动将结果存入 `airfoil_history`。

## 4. 关键文件说明 (Key Files)

- **`src/app/page.tsx`**: 主页面，负责布局、状态统筹和 API 调用。
- **`src/components/airfoil/AirfoilPreview.tsx`**: 核心可视化组件，负责绘制翼型、网格、压力向量和热力图。
- **`backend.py`**: 后端入口，包含所有 API 路由、数据库模型和 XFOIL 调用逻辑。
- **`src/components/history/HistoryPanel.tsx`**: 历史记录面板，展示过往设计并支持回溯。

## 5. 待优化项 (Future Improvements)
- [ ] 增加多点优化 (Multi-point Optimization) 功能。

## 6. 大模型与智能体 (Large Model & Agents)

本项目集成了一个基于 **LangGraph** 的多智能体系统，用于辅助用户进行翼型设计与学习。

### 6.1 智能体架构 (Agent Architecture)

- **框架**: LangChain + LangGraph (StateGraph)
- **模型**: Qwen 2.5 / GPT-4o-mini (通过 OpenAI 兼容接口调用)
- **RAG (检索增强生成)**: 
  - 向量数据库: Chroma
  - Embedding: HuggingFace (`all-MiniLM-L6-v2`)

#### 智能体角色 (Roles)
1.  **Router (路由)**: 分析用户意图，分发给特定智能体。
2.  **Concept Mentor (概念导师)**: 解答空气动力学概念问题 (RAG 增强)。
3.  **Iteration Engineer (迭代工程师)**: 根据当前翼型几何与性能数据，提供优化建议。
4.  **Strategy Analyst (策略分析师)**: 分析用户的设计历史轨迹，总结规律。

### 6.2 模型微调 (Fine-tuning)

为了提升模型在特定领域的表现，本项目提供了本地微调方案。

- **库**: **Unsloth** (加速训练) + **TRL** (SFTTrainer) + **PEFT**
- **方法**: **QLoRA** (4-bit 量化 + LoRA 适配器)
- **基座模型**: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- **训练数据**: JSON 格式 (Alpaca 或 ShareGPT 格式)
- **关键参数**:
  - `lora_r`: 16
  - `max_seq_length`: 2048
  - `load_in_4bit`: True

可通过 `fine_tuning/scripts/local_lora_train.py` 进行本地训练。
