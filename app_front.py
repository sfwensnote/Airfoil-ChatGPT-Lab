# -*- coding: utf-8 -*-
# File: app.py
# Description: Multi-user Airfoil Design Assistant (Windows + Admin Panel)

import streamlit as st
import streamlit.components.v1 as components
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain.memory import ConversationBufferMemory

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil, subprocess, tempfile, os, requests
import re, html
import json






# ===== Config =====
BACKEND_URL = os.getenv("BACKEND_URL", "http://139.196.12.84:8000")
ADMIN_PASS = os.getenv("ADMIN_PASS", "ecnustju")

import os
os.environ["OPENAI_API_KEY"] = "sk-你的key"

# =========================
# —— XFOIL Wrapper (Windows) ——
# =========================
try:
    from xfoil import XFoil
    from xfoil.model import Airfoil
    XFOIL_PY_OK = True
except Exception:
    XFOIL_PY_OK = False


# =========================
# —— XFOIL Wrapper (Windows 专用) ——
# =========================

def _which_xfoil():
    """只在当前目录下查找 xfoil.exe"""
    exe_path = os.path.join(os.getcwd(), "xfoil.exe")
    if os.path.exists(exe_path):
        return exe_path
    raise FileNotFoundError("❌ 没有找到 xfoil.exe，请确认它在 bot-remote-windows 目录下")

def run_xfoil_cli_polar(naca_code: str, Re: float, Mach: float, Ncrit: float,
                        alpha_start: float, alpha_end: float, alpha_step: float) -> pd.DataFrame:
    exe = os.path.abspath("xfoil.exe")  # ✅ 强制使用当前目录下的 xfoil.exe
    if not os.path.exists(exe):
        print("❌ xfoil.exe not found in", exe)
        return pd.DataFrame(columns=["alpha", "CL", "CD", "CM"])

    with tempfile.TemporaryDirectory() as td:
        pol_path = os.path.join(td, "polar.out")

        # ✅ 输入脚本（严格按照 Windows 版 XFOIL 要求）
        script = f"""
NACA {naca_code}
PANE
OPER
VISC {Re:.3e}
MACH {Mach:.4f}
VPAR
N {int(Ncrit)}

PACC
{pol_path}

ASEQ {alpha_start:.1f} {alpha_end:.1f} {alpha_step:.1f}
PACC

QUIT
"""

        # ✅ 在当前目录运行（避免 exe 找不到）
        result = subprocess.run(
            [exe],
            input=script,
            text=True,
            capture_output=True,
            cwd=os.getcwd(),
            timeout=60
        )

        # 调试输出（只看前 400 字符）
        print("=== XFOIL STDOUT ===\n", (result.stdout or "")[:400])
        print("=== XFOIL STDERR ===\n", (result.stderr or "")[:400])

        if not os.path.exists(pol_path):
            print("❌ polar.out not generated")
            return pd.DataFrame(columns=["alpha", "CL", "CD", "CM"])

        # ✅ 解析 polar.out
        rows = []
        with open(pol_path, "r") as f:
            for line in f:
                if ("alpha" in line and "CL" in line and "CD" in line) or set(line.strip()) <= set("-= "):
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        a = float(parts[0])
                        cl = float(parts[1])
                        cd = float(parts[2])
                        cm = float(parts[4])
                        rows.append((a, cl, cd, cm))
                    except Exception:
                        continue

        if not rows:
            print("⚠️ polar.out parsed but empty")
            return pd.DataFrame(columns=["alpha", "CL", "CD", "CM"])

        return pd.DataFrame(rows, columns=["alpha", "CL", "CD", "CM"]).sort_values("alpha").reset_index(drop=True)

@st.cache_data(show_spinner=True)
def run_xfoil_polar(naca_code: str, Re: float, Mach: float, Ncrit: float,
                    alpha_start: float, alpha_end: float, alpha_step: float) -> pd.DataFrame:
    if XFOIL_PY_OK:
        try:
            af = Airfoil.NACA(naca_code)
            xf = XFoil()
            xf.airfoil = af
            xf.Re = max(Re, 1e4)
            xf.M = max(Mach, 0.0)
            xf.n_crit = Ncrit
            try:
                a, cl, cd, cm, _ = xf.aseq(alpha_start, alpha_end, alpha_step)
            except Exception:
                A = np.arange(alpha_start, alpha_end + 1e-9, alpha_step)
                alist, clist, cdlist, cmlist = [], [], [], []
                for a0 in A:
                    try:
                        cl0, cd0, cm0, _cp = xf.a(a0)
                        alist.append(a0); clist.append(cl0); cdlist.append(cd0); cmlist.append(cm0)
                    except Exception:
                        continue
                a = np.array(alist); cl = np.array(clist); cd = np.array(cdlist); cm = np.array(cmlist)
            mask = np.isfinite(a) & np.isfinite(cl) & np.isfinite(cd) & np.isfinite(cm)
            df = pd.DataFrame({"alpha": a[mask], "CL": cl[mask], "CD": cd[mask], "CM": cm[mask]})
            df = df.sort_values("alpha").reset_index(drop=True)
            if not df.empty:
                return df
        except Exception:
            pass
    return run_xfoil_cli_polar(naca_code, Re, Mach, Ncrit, alpha_start, alpha_end, alpha_step)


def fallback_fake_polar(alpha_start, alpha_end, alpha_step):
    alphas = np.arange(alpha_start, alpha_end + 1e-9, alpha_step)
    CL = 0.1 * alphas
    CD = 0.01 + 0.002 * alphas**2
    CM = -0.05 * np.ones_like(alphas)
    return pd.DataFrame({"alpha": alphas, "CL": CL, "CD": CD, "CM": CM})


# =========================
# —— LangGraph + ChatGPT（修复版） ——
# =========================
import os
from langchain_openai import ChatOpenAI

# typing 兼容导入
try:
    from typing import TypedDict, Dict, Any
except ImportError:            # < Py3.8 才会走到这里
    from typing_extensions import TypedDict
    from typing import Dict, Any

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    api_key="sk-VrwOEEFLgjJwSOjH5pHRTDorgf0SmJVQrjK2D1uyjxZcfsrn",   # 写在这里
    base_url="http://49.51.37.239:3006/v1"  # 中转地址
)
memory = ConversationBufferMemory(memory_key="history")

roles = {
    "Concept Learning": (
        "You are an AI tutor specialized in **conceptual learning**. "
        "Your role is to guide students in breaking down abstract fluid mechanics concepts "
        "into smaller, teachable parts. "
        "You should always respond in English with a **guiding and questioning tone**, "
        "encouraging the student to reflect. "
        "Never give the final answer directly. "
        "Ask questions such as: 'Which factors do you think are important, and why?' "
        "or 'How would changing the camber affect the lift coefficient?'"
    ),
    "Model Iteration": (
        "You are an AI assistant specialized in **model iteration and experiment design**. "
        "Your role is to help students analyze simulation data, propose adjustments, "
        "and suggest possible experiment setups. "
        "Always respond in English with a **coaching style**, focusing on reasoning and exploration, "
        "rather than giving direct solutions. "
        "Encourage the student to test ideas and compare results, for example: "
        "'What would happen if you increased the Reynolds number?' "
        "or 'How could you verify whether thickness has a stronger impact than camber?'"
    ),
    "Strategy Review": (
        "You are an AI mentor specialized in **strategy review and reflection**. "
        "Your role is to evaluate the student’s current approach and give constructive feedback "
        "on their design strategies. "
        "Always respond in English with a **critical but supportive tone**, "
        "pointing out strengths and possible improvements. "
        "Do not just state the answer — instead, highlight gaps and suggest next steps. "
        "For example: 'Your strategy improves L/D ratio, but it may ignore stall effects. "
        "What adjustment could account for that?'"
    )
}

role_guides = {
    "Concept Learning": [
        "Break down the key sub-concepts of 'Lift Coefficient' into a 3-level teachable structure.",
        "Based on C-K theory, propose 3 concept-to-concept expansion paths, each with an engineering context."
    ],
    "Model Iteration": [
        "Design a reproducible experiment to compare two airfoils at Re=3e5 (include variables and indicators).",
        "Provide suggestions for XFOIL polar scan parameter settings and explain the rationale for Ncrit values."
    ],
    "Strategy Review": [
        "Check the logic of this argument about lift-to-drag ratio, and point out gaps in claim–evidence–warrant chain.",
        "Give 3 rewriting suggestions to make the argument more academic and evidence-complete."
    ]
}

role_descriptions = {
    "Concept Learning": "Focus on clarifying core concepts.",
    "Model Iteration": "Assist with experiments and parameter tuning.",
    "Strategy Review": "Give feedback on strategies and reasoning."
}

def get_prompt(role):
    return ChatPromptTemplate.from_messages([
        ("system", roles[role] + "\nHistorical Dialogue: {history}"),
        ("user", "{question}")
    ])

# ---- State 类型（dict 形式，LangGraph 推荐）----
class GraphState(TypedDict, total=False):
    role: str
    history: str
    question: str
    content: str   # 由节点写回

# ---- 将每个角色封装为返回 dict 的节点 ----
def create_node(role: str):
    prompt = get_prompt(role)
    chain = prompt | llm

    def _run(state: GraphState) -> Dict[str, Any]:
        res = chain.invoke({
            "history": state.get("history", ""),
            "question": state.get("question", "")
        })
        text = getattr(res, "content", str(res))
        return {"content": text}   # ✅ 写回到状态
    return _run

graph = StateGraph(GraphState)
for r in roles:
    graph.add_node(r, create_node(r))

# ---- 条件路由：函数只返回“下一节点名”，不要当节点本身返回 ----
def route_role(state: GraphState) -> str:
    return state["role"]

# 🚩 router 节点必须返回 dict，这里用空操作占位
graph.add_node("router", lambda state: {})   # ✅ 不要返回字符串！
graph.set_entry_point("router")
graph.add_conditional_edges("router", route_role, {r: r for r in roles})

for r in roles:
    graph.add_edge(r, END)

app = graph.compile()
# >>> PATCH: LLM retry helper (place right after `app = graph.compile()`)
import time as _t

def call_llm_with_retries(state: dict, retries: int = 3, base_delay: float = 2.0) -> str:
    """
    Call the LangGraph app with simple exponential backoff.
    Returns the LLM text or "" if all attempts failed or empty.
    """
    last_err = None
    for k in range(retries):
        try:
            resp_state = app.invoke(state)
            txt = (resp_state or {}).get("content", "")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
        except Exception as e:
            last_err = e
        # backoff: 2s, 4s, 8s ...
        _t.sleep(base_delay * (2 ** k))
    # all failed or got empty content
    if last_err:
        # 让调用方决定是否展示错误/落盘
        # 不抛出异常，返回空串以走统一兜底提示
        pass
    return ""



# =========================
# —— Geometry Utils ——
# =========================
@st.cache_data(show_spinner=False)
def gen_naca4(m: float, p: float, t: float, n_pts: int = 200):
    x = np.linspace(0, 1, n_pts)
    yt = 5 * t * (
        0.2969 * np.sqrt(np.clip(x, 1e-12, 1.0))
        - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    )
    p_eps = max(p, 1e-12); q_eps = max(1-p, 1e-12)
    yc = np.where(x < p,
        (m/(p_eps**2))*(2*p*x - x**2),
        (m/(q_eps**2))*((1-2*p) + 2*p*x - x**2)
    )
    dyc_dx = np.where(x < p,
        (2*m/(p_eps**2))*(p - x),
        (2*m/(q_eps**2))*(p - x)
    )
    theta = np.arctan(dyc_dx)
    xu = x - yt*np.sin(theta); yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta); yl = yc - yt*np.cos(theta)
    xs = np.concatenate([xl[::-1], xu[1:]])
    ys = np.concatenate([yl[::-1], yu[1:]])
    return xs, ys


def naca_code_from_mpt(m: float, p: float, t: float) -> str:
    m_pct = int(round(m*100)); p_tenths = int(round(p*10)); t_pct = int(round(t*100))
    return f"{m_pct}{p_tenths}{t_pct:02d}"


def estimate_Re(rho: float, V: float, chord: float, mu: float) -> float:
    return (rho*V*chord)/max(mu, 1e-9)


# =====================
# —— Streamlit UI ——
# =====================
st.set_page_config(page_title="Fluid Mechanics AI Assistant", layout="wide")

# --- 合并且加固的样式（只保留这一个 <style>） ---
st.markdown("""
<style>
/* 1) 让主容器顶部有充足留白，避免被固定头部覆盖 */
.block-container {
    padding-top: 2.0rem !important;   /* 建议 ≥1.6rem；若仍被遮挡，调到 2.4rem */
    padding-bottom: 0.8rem !important;
    overflow: visible !important;      /* 防止被裁切 */
}

/* 2) 保证标题本身不被挤；行高充足、外边距合理 */
h1 {
    margin: 0 0 .6rem 0 !important;
    line-height: 1.18 !important;
    word-break: break-word;
}

/* 3) 有些项目里会给 header/容器设置 overflow:hidden，这里强制解除 */
header, [data-testid="stHeader"] {
    overflow: visible !important;
}

/* 4) 次级标题更紧凑（可选） */
h3, h4, h5 { margin: .2rem 0 .2rem 0 !important; }

/* 5) 不要用会变动的情绪类名，以下仅对常用控件轻量收紧（可选） */
div[data-testid="stSlider"], div[data-testid="stNumberInput"] { margin-bottom: .35rem; }

/* 6) 右侧输入行微调（可选） */
.input-row { margin-top: .4rem; }

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Top-anchored, no vertical blank area for matplotlib images/canvas */
.fixed-plot {
  height: 320px;                /* 你也可以改成 300/340；固定高度，避免跳动 */
  position: relative;
  overflow: hidden;
}

/* 让图片/画布铺满容器，高度优先，保持等比，不再居中 */
.fixed-plot img, .fixed-plot canvas {
  position: absolute;
  inset: 0;                     /* top:0 right:0 bottom:0 left:0 */
  width: 100%;
  height: 100%;
  object-fit: contain;          /* 保持等比 */
  object-position: top left;    /* 顶对齐（也可用 'top center'）*/
  display: block;               /* 去掉 inline 元素造成的垂直对齐影响 */
}
</style>
""", unsafe_allow_html=True)


st.title("AI-Enhanced Airfoil Design & Learning Lab")
st.caption("An AI-powered airfoil design assistant for NACA geometry preview, XFOIL polar analysis, and role-based learning guidance.")

# ==== Sidebar: User / Admin Login ====
st.sidebar.title("Login")
user_id = st.sidebar.text_input("Enter your User ID", value="guest")
st.sidebar.markdown("---")
st.sidebar.subheader("Admin Access")
admin_password = st.sidebar.text_input("Enter admin password", type="password")
is_admin = (admin_password == ADMIN_PASS)

# ==== Session init ====
if "param_history" not in st.session_state:
    st.session_state.param_history = []
if "prev_role" not in st.session_state:
    st.session_state.prev_role = None

# ✅ 每个用户独立的聊天记录
chat_key = f"chat_history_{user_id}"
if chat_key not in st.session_state:
    st.session_state[chat_key] = []

# ==== Layout ====
col_chat, col_main = st.columns([1, 1], gap="large")

import time

# ===== Left: Dialogue =====
with col_chat:
    # （不要标题了）
    # === 样式定义 ===
    st.markdown(
        """
        <style>
        .chat-wrapper {
            background-color: #f5f5f5;
            padding: 12px;
            border-radius: 10px;
            height: 55vh;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }
        .bubble {
            max-width: 75%;
            padding: 8px 12px;
            margin: 6px;
            border-radius: 8px;
            word-wrap: break-word;
            font-size: 14px;
            line-height: 1.4;
        }
        .user-bubble {
            background-color: #95ec69;
            margin-left: auto;
            text-align: right;
        }
        .ai-bubble {
            margin-right: auto;
            text-align: left;
        }
        .concept { background-color: #d0e6ff; }
        .model { background-color: #ffe4b3; }
        .strategy { background-color: #e6ccff; }
        .system {
            color: #555;
            text-align: center;
            font-size: 13px;
            font-style: italic;
            margin: 4px 0;
        }
        /* 按钮统一样式 */
        div[data-testid="stButton"] button {
            background-color: #4da6ff;
            color: white;
            border-radius: 6px;
            padding: 0.3em 0.6em;
            font-size: 14px;
            border: none;
        }
        div[data-testid="stButton"] button:hover {
            background-color: #1a8cff;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



    st.markdown(
        """
        <style>
        /* —— 紧凑：components.html/搜索/按钮 —— */
        .ctl-label { font-size: 0.82rem; color:#444; margin:0 0 4px 2px; }
        div[data-testid="stMultiSelect"] > div { margin-bottom: 0.2rem; }
        div[data-baseweb="select"] { min-height: 34px; }
        div[data-baseweb="select"] > div { min-height: 34px; padding-top: 2px; padding-bottom: 2px; }
        div[data-baseweb="tag"] { margin: 2px 4px; transform: scale(0.92); }
        div[data-testid="stTextInput"] input { height: 34px; padding: 2px 8px; font-size: 0.90rem; }
        div[data-testid="stTextInput"] label { margin-bottom: 4px; font-size: 0.82rem; }
        .small-btn button { height: 34px; padding: 0 10px; font-size: 0.90rem; border-radius: 8px; }
        .compact-row { margin-bottom: 0.25rem; }
        
        </style>
        """,
        unsafe_allow_html=True,
    )

    # === 恢复历史按钮 ===
    if st.button("📜 Restore historical dialogue", use_container_width=True):
        try:
            resp = requests.get(f"{BACKEND_URL}/export_conversations/{user_id}", timeout=10)
            if resp.status_code == 200:
                history_data = resp.json()
                history_data = sorted(history_data, key=lambda x: x["timestamp"])
                past_msgs = []
                if history_data:
                    past_msgs.append({"role": "system", "content": "—— 🕒 Historical Dialogue ——"})
                    for h in history_data:
                        if h.get("student_question"):
                            past_msgs.append({"role": "user", "content": h["student_question"]})
                        if h.get("ai_response"):
                            past_msgs.append({
                                "role": "ai",
                                "content": h["ai_response"],
                                "module": h.get("role", "AI")
                            })
                st.session_state[chat_key] = past_msgs  # ✅ 覆盖
                st.success("✅ History has been rewritten")
                st.session_state["force_scroll_bottom"] = True  # ✅ 恢复后强制到底
            else:
                st.info("ℹ️ No historical records were found")
        except Exception as e:
            st.error(f"⚠️ Error fetching history: {e}")

    # === 每个用户独立的聊天 key ===
    chat_key = f"chat_history_{user_id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # === 滚动与搜索的 session 状态 ===
    if "force_scroll_bottom" not in st.session_state:
        st.session_state["force_scroll_bottom"] = False
    if "scroll_to_msg_id" not in st.session_state:
        st.session_state["scroll_to_msg_id"] = ""
    if "chat_search_q" not in st.session_state:
        st.session_state["chat_search_q"] = ""
    if "search_hits" not in st.session_state:
        st.session_state["search_hits"] = []
    if "search_pos" not in st.session_state:
        st.session_state["search_pos"] = 0
    if "chat_filter_roles" not in st.session_state:
        st.session_state["chat_filter_roles"] = list(roles.keys())  # 默认全选

    # === 筛选 + 搜索控件（双列布局：左筛选 / 右搜索+按钮）===
    st.markdown("""
    <style>
      /* 右侧按钮：保证横向显示并不折行 */
      .small-btn button {
        height: 28px;          /* 减小按钮高度 */
        padding: 0 10px;       /* 减小内边距 */
        font-size: 0.85rem;    /* 稍微减小字体 */
        border-radius: 6px;
        white-space: nowrap;
        min-width: 80px;
      }
      .ctrl-group { 
        margin-top: 0.1rem;    /* 减小垂直间距 */
        margin-bottom: 0.1rem;
      }
      /* 标签更紧凑 */
      .ctl-label { 
        font-size: 0.82rem; 
        color:#bbb; 
        margin: 0 0 3px 2px;   /* 减小标签底部间距 */
      }
      /* 输入框更矮一些 */
      div[data-testid="stTextInput"] input { 
        height: 32px;          /* 减小搜索框高度 */
        padding: 3px 8px; 
        font-size: 0.9rem; 
      }
      div[data-testid="stMultiSelect"] > div { margin-bottom: 0.2rem; }
      div[data-baseweb="select"] { min-height: 32px; }  /* 匹配搜索框高度 */
      div[data-baseweb="select"] > div { 
        min-height: 32px; 
        padding-top: 1px; 
        padding-bottom: 1px; 
      }
      div[data-baseweb="tag"] { 
        margin: 1px 3px; 
        transform: scale(0.9); /* 稍微缩小标签 */
      }
    </style>
    """, unsafe_allow_html=True)

    ctl_left, ctl_right = st.columns([1, 1], gap="small")

    # 左侧：模块筛选（保持不变）
    with ctl_left:
        st.markdown('<div class="ctl-label">Filter modules</div>', unsafe_allow_html=True)
        st.session_state["chat_filter_roles"] = st.multiselect(
            label="Filter modules",
            options=list(roles.keys()),
            default=st.session_state["chat_filter_roles"],
            help="筛选显示不同 AI 模块的消息（用户消息始终显示）",
            label_visibility="collapsed"
        )

    # 右侧：搜索 + 按钮（优化布局）
    with ctl_right:
        st.markdown('<div class="ctl-label">Search history</div>', unsafe_allow_html=True)
        q = st.text_input(
            "Search history",
            value=st.session_state["chat_search_q"],
            placeholder="Enter keywords; Use Prev/Next to jump",
            key="chat_search_q",
            label_visibility="collapsed",
        ).strip()

        # Prev / Next / Latest 按钮放在同一行
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1.2], gap="small")

        with col_btn1:
            st.markdown('<div class="small-btn ctrl-group">', unsafe_allow_html=True)
            btn_prev = st.button("Prev", use_container_width=True, key="btn_prev")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_btn2:
            st.markdown('<div class="small-btn ctrl-group">', unsafe_allow_html=True)
            btn_next = st.button("Next", use_container_width=True, key="btn_next")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_btn3:
            st.markdown('<div class="small-btn ctrl-group">', unsafe_allow_html=True)
            btn_jump_latest = st.button("⬇️ Latest", use_container_width=True, key="btn_latest")
            st.markdown('</div>', unsafe_allow_html=True)

    if btn_jump_latest:
        st.session_state["force_scroll_bottom"] = True

    # === 构建“显示列表”的索引（按模块过滤，只显示“AI(选中模块)+它前面的学生问”） ===
    messages = st.session_state[chat_key]
    allowed = set(st.session_state["chat_filter_roles"])

    display_indices = []
    n = len(messages)


    def include_pair(ai_idx: int):
        """把 ai_idx 及其前面的最近一条 user 加入显示"""
        if 0 <= ai_idx < n and ai_idx not in display_indices:
            display_indices.append(ai_idx)
        # 寻找 ai 之前最近的一条 user
        j = ai_idx - 1
        while j >= 0 and messages[j]["role"] == "system":
            j -= 1
        if j >= 0 and messages[j]["role"] == "user" and j not in display_indices:
            display_indices.append(j)


    # 1) 收集符合模块的 AI 回复与其前面的 user
    for i, msg in enumerate(messages):
        if msg["role"] == "ai":
            module = msg.get("module", "AI")
            if module in allowed:
                include_pair(i)

    # 2) 如果最后一条是“用户刚发的消息”（可能还没收到 AI），也显示它
    if n > 0 and messages[-1]["role"] == "user":
        if (n - 1) not in display_indices:
            display_indices.append(n - 1)

    # 3) 排序保持时间顺序
    display_indices.sort()

    # === 计算搜索命中（仅在“显示列表”内搜索） ===
    hits = []
    if q:
        q_lower = q.lower()
        for i in display_indices:
            content = str(messages[i].get("content", ""))
            if q_lower in content.lower():
                hits.append(i)

    # === 处理 Next/Prev 导航 ===
    if hits:
        if btn_next:
            st.session_state["search_pos"] = (st.session_state.get("search_pos", 0) + 1) % len(hits)
            st.session_state["scroll_to_msg_id"] = f"msg-{hits[st.session_state['search_pos']]}"
        if btn_prev:
            st.session_state["search_pos"] = (st.session_state.get("search_pos", 0) - 1) % len(hits)
            st.session_state["scroll_to_msg_id"] = f"msg-{hits[st.session_state['search_pos']]}"

    # 命中计数提示
    if q:
        if hits:
            pos = (st.session_state.get('search_pos', 0) % len(hits)) + 1 if len(hits) else 0
            st.caption(f"🔎 Found {len(hits)} hit(s) — {pos}/{len(hits)}")
        else:
            st.caption("🔎 No hits")


    # ===== Markdown rendering (server-side) =====
    def render_markdown_to_html(md_text: str) -> str:
        """Render Markdown to safe HTML. Falls back to a minimal converter if libs are missing."""
        try:
            import markdown as md
            html_out = md.markdown(md_text, extensions=["fenced_code", "tables", "toc", "sane_lists"])
            try:
                import bleach
                allowed = bleach.sanitizer.ALLOWED_TAGS.union(
                    {"p", "pre", "code", "table", "thead", "tbody", "tr", "th", "td", "hr",
                     "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "blockquote"}
                )
                return bleach.clean(html_out, tags=allowed, strip=True)
            except Exception:
                return html_out
        except Exception:
            # minimal fallback: keep bold / code / line breaks
            s = html.escape(md_text, quote=False)
            s = s.replace("**", "<b>").replace("`", "<code>")
            return s.replace("\n", "<br>")


    def highlight_after_rendered(html_text: str, q: str) -> str:
        if not q:
            return html_text
        # highlight matched query segments
        return re.sub(re.escape(q), lambda m: f"<mark>{html.escape(m.group(0))}</mark>",
                      html_text, flags=re.IGNORECASE)


    # === 渲染聊天（组件版：iframe 内完全可控） ===
    inner_html = ""
    for i, msg in enumerate(messages):
        if i not in display_indices:
            continue
        raw = str(msg.get("content", ""))
        rendered = render_markdown_to_html(raw)  # Markdown → safe HTML
        content_html = highlight_after_rendered(rendered, q)  # 可选高亮
        if msg["role"] == "user":
            inner_html += f"<div class='bubble user-bubble' id='msg-{i}'>👤 You ({user_id})<br>{content_html}</div>"
        elif msg["role"] == "ai":
            module = msg.get("module", "AI")
            css_class = {
                "Concept Learning": "concept",
                "Model Iteration": "model",
                "Strategy Review": "strategy"
            }.get(module, "")
            inner_html += f"<div class='bubble ai-bubble {css_class}' id='msg-{i}'>🤖 {module}<br>{content_html}</div>"
        elif msg["role"] == "system":
            inner_html += f"<div class='system' id='msg-{i}'>{content_html}</div>"

    _force = bool(st.session_state.get("force_scroll_bottom", False))
    target_id = st.session_state.get("scroll_to_msg_id", "")

    # 用完即清（避免下次重复触发）
    st.session_state["force_scroll_bottom"] = False
    st.session_state["scroll_to_msg_id"] = ""

    CHAT_HEIGHT = 500

    components.html(f"""
    <!DOCTYPE html>
    <html><head><meta charset="utf-8" />
    <style>
      html, body {{
        margin: 0; padding: 0; height: 100%; overflow: hidden;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      }}
      .chat-wrapper {{
          background-color: #f5f5f5;
          padding: 12px;
          border-radius: 10px;
          height: {CHAT_HEIGHT}px;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          scroll-behavior: smooth;
          overscroll-behavior: contain;
          scrollbar-gutter: stable;
      }}
      .bubble {{
          max-width: 75%;
          padding: 8px 12px;
          margin: 6px;
          border-radius: 8px;
          word-wrap: break-word;
          font-size: 14px;
          line-height: 1.4;
      }}
      .user-bubble {{ background-color: #95ec69; margin-left: auto; text-align: right; }}
      .ai-bubble {{ margin-right: auto; text-align: left; }}
      .concept {{ background-color: #d0e6ff; }}
      .model {{ background-color: #ffe4b3; }}
      .strategy {{ background-color: #e6ccff; }}
      .system {{
          color: #555;
          text-align: center;
          font-size: 13px;
          font-style: italic;
          margin: 4px 0;
      }}
      .hit-focus {{
          outline: 2px solid #ffb703;
          transition: outline 1.2s ease-in-out;
      }}
      mark {{ padding: 0 2px; }}
      /* code/table styling for rendered Markdown */
  pre, code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12.5px;
  }}
  pre {{
    background: #f7f7f9;
    padding: 8px 10px;
    border-radius: 6px;
    overflow: auto;
  }}
  table {{ border-collapse: collapse; max-width: 100%; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 4px 8px; font-size: 13px; }}
  blockquote {{ border-left: 3px solid #e5e7eb; margin: 6px 0; padding: 4px 8px; color:#444; }}
 
    </style>
    </head>
    <body>
      <div class="chat-wrapper" id="chat-box">{inner_html}</div>

  <script>
(function() {{
  const chatBox = document.getElementById('chat-box');
  if (!chatBox) return;

  // Python → JS 注入（保持不变）
  const FORCE  = {str(_force).lower()};
  const TARGET = {json.dumps(target_id)};

  // 使用 user_id 构造本地存储键；避免使用模板字符串以防 f-string 解析
  const STORAGE_KEY = "autoFollow:" + {json.dumps(user_id)};
  const SCROLL_KEY  = "scrollTop:"  + {json.dumps(user_id)};

  // --- auto-follow state (persisted via localStorage) ---
  function getAutoFollow() {{
    const v = localStorage.getItem(STORAGE_KEY);
    return (v === null) ? true : v === "1";
  }}
  function setAutoFollow(on) {{
    localStorage.setItem(STORAGE_KEY, on ? "1" : "0");
  }}

  // --- scrollTop persistence (sessionStorage) ---
  function saveScroll() {{
    try {{
      sessionStorage.setItem(SCROLL_KEY, String(chatBox.scrollTop));
    }} catch (e) {{}}
  }}
  function restoreScroll() {{
    try {{
      const v = sessionStorage.getItem(SCROLL_KEY);
      if (v !== null) {{
        const n = parseInt(v, 10);
        if (!Number.isNaN(n)) chatBox.scrollTop = n;
      }}
    }} catch (e) {{}}
  }}

  function isAtBottom() {{
    return chatBox.scrollTop + chatBox.clientHeight >= chatBox.scrollHeight - 40;
  }}
  function scrollToBottom(force) {{
    if (force || (getAutoFollow() && isAtBottom())) {{
      chatBox.scrollTop = chatBox.scrollHeight;
    }}
  }}
  function scrollToTarget(id) {{
    if (!id) return false;
    const el = document.getElementById(id);
    if (el) {{
      el.scrollIntoView({{ behavior: 'smooth', block: 'end' }});
      setAutoFollow(false);
      el.classList.add("hit-focus");
      setTimeout(() => el.classList.remove("hit-focus"), 1200);
      return true;
    }}
    return false;
  }}

  // Initial mount: prefer TARGET; else restore; apply FORCE if requested
  requestAnimationFrame(() => setTimeout(() => {{
    if (!scrollToTarget(TARGET)) {{
      if (!FORCE) restoreScroll();
      if (FORCE) setAutoFollow(true);
      scrollToBottom(FORCE);
    }}
  }}, 0));

  // Auto-follow when new nodes added (unless user scrolled up)
  const mo = new MutationObserver(() => {{
    if (getAutoFollow()) {{
      chatBox.scrollTop = chatBox.scrollHeight;
    }} else if (isAtBottom()) {{
      setAutoFollow(true);
    }}
  }});
  mo.observe(chatBox, {{ childList: true, subtree: true }});

  // User scroll behavior: scroll up → disable follow; bottom → enable follow
  let lastTop = chatBox.scrollTop;
  chatBox.addEventListener('scroll', () => {{
    saveScroll();
    const nowTop = chatBox.scrollTop;
    const delta  = nowTop - lastTop;
    lastTop = nowTop;
    if (delta < -10) {{
      setAutoFollow(false);
    }} else if (isAtBottom()) {{
      setAutoFollow(true);
    }}
  }});

  // Persist before unload (tab close / refresh)
  window.addEventListener('beforeunload', saveScroll);
}})();
</script>

    </body></html>
    """, height=CHAT_HEIGHT + 6, scrolling=False)

    # === 角色选择 ===
    role_help = {
        "Concept Learning": "Guide conceptual understanding.",
        "Model Iteration": "Assist with simulation and iteration.",
        "Strategy Review": "Review and improve strategies."
    }
    # --- Concise Mode toggle (EN) ---
    concise = st.toggle(
        "✂️ Concise mode (≤6 lines + ≤2 questions)",
        value=True,
        help="Limit length and number of probing questions"
    )


    def get_prompt(role):
        style_guard = (
            "When replying, ALWAYS follow this structure:\n"
            "1) Key points (≤4 bullet points)\n"
            "2) Evidence/Reasoning (≤2 sentences)\n"
            "3) Next step (≤1 actionable step)\n"
            "Ask at most 2 short questions.\n"
        )
        length_guard = "Hard cap: ≤6 lines total. Avoid meta talk." if concise else "No hard cap. Provide details when useful."
        sys = f"{roles[role]}\n{style_guard}{length_guard}\nHistorical Dialogue: {{history}}"
        return ChatPromptTemplate.from_messages([
            ("system", sys),
            ("user", "{question}")
        ])


    selected_role = st.selectbox(
        "🎭 Choose AI Module",
        list(roles.keys()),
        index=0,
        key="role_select"
    )

    # ✅ 不再只显示所选模块简介；改为展示所有模块的一行简介
    st.markdown('<div class="ctl-label">Modules overview</div>', unsafe_allow_html=True)
    for r in roles:
        desc = role_descriptions.get(r) or role_help.get(r, "")
        # 一行、简洁：模块名 + 简短说明
        st.caption(f"• **{r}** — {desc}")

    # 下面保持原有的切换逻辑（切换角色时滚动到底部）
    if "prev_role" not in st.session_state:
        st.session_state["prev_role"] = selected_role
    if st.session_state["prev_role"] != selected_role:
        st.session_state["prev_role"] = selected_role
        st.session_state["force_scroll_bottom"] = True  # ✅ 切换角色后滚到最新

    # === 输入框 + 发送 & 清除按钮（同一行）===
    st.markdown(
        """
        <style>
        .input-row {
            display: flex;
            align-items: center; /* ✅ 垂直居中 */
            gap: 6px; /* ✅ 输入框与按钮间距 */
        }
        .text-input { flex: 1; } /* ✅ 自适应填充 */
        .send-btn button, .clear-btn button {
            height: 45px; /* ✅ 高度和输入框匹配 */
            width: 45px;  /* ✅ 正方形按钮 */
            border-radius: 8px;
            background-color: #4da6ff;
            color: white;
            font-size: 18px;
            border: none;
        }
        .send-btn button:hover, .clear-btn button:hover {
            background-color: #1a8cff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # =========================
    # 方案A：动态 key + epoch（并清理旧键）
    # =========================
    if "question_buf" not in st.session_state:
        st.session_state["question_buf"] = ""
    if "input_epoch" not in st.session_state:
        st.session_state["input_epoch"] = 0

    # 当前活跃的输入框 key
    _active_key = f"text_area_input_{st.session_state['input_epoch']}"

    # 清理历史 widget state，防止长期运行堆积
    for k in list(st.session_state.keys()):
        if k.startswith("text_area_input_") and k != _active_key:
            try:
                del st.session_state[k]
            except Exception:
                pass

    st.markdown('<div class="input-row">', unsafe_allow_html=True)
    question = st.text_area(
        "💬",
        value=st.session_state.get("question_buf", ""),  # ✅ 仍然使用 value 作为初始化/后端参数源
        key=_active_key,  # ✅ 动态 key：每次重置创建新组件，使 value 生效
        height=45,
        placeholder="Type a message...",
        label_visibility="collapsed"  # ✅ 隐藏 label
    )

    # 发送按钮
    submit = st.button("↩️", key="send_btn", help="Send message", use_container_width=False)

    st.markdown('</div>', unsafe_allow_html=True)


    # === 插入仿真数据按钮（风格统一）===
    insert_sim = st.button("📥 Insert Simulation Data", use_container_width=True)

    # === 插入仿真数据逻辑 ===
    if insert_sim:
        m = st.session_state.get("camber_pct", 2.0) / 100
        p = st.session_state.get("p_pct", 40.0) / 100
        t = st.session_state.get("thickness_pct", 12.0) / 100
        alpha = st.session_state.get("alpha_deg", 5.0)
        rho = st.session_state.get("rho", 1.225)
        V = st.session_state.get("V", 10.0)
        chord = st.session_state.get("chord", 1.0)
        mu = st.session_state.get("mu", 1.8e-5)
        M = st.session_state.get("Mach", 0.0)
        Ncrit = st.session_state.get("Ncrit", 7.0)
        a_min, a_max = st.session_state.get("alpha_range", (0.0, 10.0))
        a_step = st.session_state.get("alpha_step", 1.0)

        naca_code = naca_code_from_mpt(m, p, t)
        Re = estimate_Re(rho, V, chord, mu)

        sim_text = (
            f"My current simulation results:\n"
            f"- NACA code: {naca_code}\n"
            f"- Re ≈ {Re:.0f}, Mach={M}, Ncrit={Ncrit}\n"
            f"- α range: {a_min}° to {a_max}°, step={a_step}°\n"
            f"- Current α={alpha}°\n"
            f"- Parameters: camber={m:.2f}, thickness={t:.2f}, max_camber_pos={p:.2f}\n"
            f"- Fluid: ρ={rho}, V={V}, chord={chord}, μ={mu}\n"
            f"➡️ How can I improve my design strategy based on this?"
        )

        # 覆盖 value 源并切换到新 key → 下轮渲染文本框显示仿真内容
        st.session_state["question_buf"] = sim_text
        st.session_state["input_epoch"] += 1
        st.rerun()

    # === 发送消息逻辑 ===
    if submit and question.strip():
        student_question = question.strip()  # 当前输入框真实内容

        # 1) 前端先显示用户消息
        st.session_state[chat_key].append({"role": "user", "content": student_question})

        # 2) 先落盘用户消息（即便还没有 AI 回复）
        try:
            requests.post(
                f"{BACKEND_URL}/save_conversation/",
                json={
                    "user_id": user_id,
                    "role": selected_role,
                    "student_question": student_question,
                    "ai_response": "",  # 空也写入
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                timeout=10
            )
        except Exception as e:
            st.warning(f"⚠️ Failed to save user message: {str(e)}")

        # 3) 调用 LLM（LangGraph）
        # 3) Call LLM with retries (LangGraph)
        answer = ""
        try:
            history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state[chat_key]])
            state = {
                "role": selected_role,
                "history": history,
                "question": student_question,
            }
            # use retry helper
            answer = call_llm_with_retries(state, retries=3, base_delay=2.0)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            # 4) also log the error to backend for troubleshooting
            try:
                requests.post(
                    f"{BACKEND_URL}/save_conversation/",
                    json={
                        "user_id": user_id,
                        "role": selected_role,
                        "student_question": student_question,
                        "ai_response": f"[ERROR] {e}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    },
                    timeout=10
                )
            except Exception:
                pass

        # 可选的用户提示：重试后仍无有效文本
        if not answer:
            st.info(
                "The AI returned no valid content after several attempts. Please retry in a moment or simplify your question.")

        # 5) ✅ 把 AI 回复追加到前端会话，并写回后端（你当前缺的就是这一块）
        if answer:
            st.session_state[chat_key].append({
                "role": "ai",
                "content": answer,
                "module": selected_role
            })
            try:
                requests.post(
                    f"{BACKEND_URL}/save_conversation/",
                    json={
                        "user_id": user_id,
                        "role": selected_role,
                        "student_question": student_question,
                        "ai_response": answer,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    },
                    timeout=10
                )
            except Exception as e:
                st.warning(f"⚠️ Failed to save AI reply: {e}")

        # 6) 清空输入并刷新
        st.session_state["question_buf"] = ""
        st.session_state["input_epoch"] += 1
        st.session_state["force_scroll_bottom"] = True  # ✅ 一定要在 rerun 之前！
        st.rerun()

# ===== Right: Tabs =====
with col_main:
    tab_geo, tab_perf, tab_hist, tab_help, tab_admin = st.tabs([
        "🧩 Geometry & Parameters", "📈 Performance & Polars", "🗂️ History", "❓ Help", "🔑 Admin"
    ])

    # === Geometry Tab ===
    with tab_geo:
        # ===== 上半：预览 =====
        st.markdown("### Airfoil Preview", help=None)
        # 读取参数（保留你的原值来源）
        m, p, t, alpha, rho, V, chord, mu, M, Ncrit, a_min, a_max, a_step, max_t_pos = \
            (st.session_state.get("camber_pct", 2.0) / 100,
             st.session_state.get("p_pct", 40.0) / 100,
             st.session_state.get("thickness_pct", 12.0) / 100,
             st.session_state.get("alpha_deg", 5.0),
             st.session_state.get("rho", 1.225),
             st.session_state.get("V", 10.0),
             st.session_state.get("chord", 1.0),
             st.session_state.get("mu", 1.8e-5),
             st.session_state.get("Mach", 0.0),
             st.session_state.get("Ncrit", 7.0),
             st.session_state.get("alpha_range", (0.0, 10.0))[0],
             st.session_state.get("alpha_range", (0.0, 10.0))[1],
             st.session_state.get("alpha_step", 1.0),
             st.session_state.get("tpos_pct", 30.0) / 100)

        xs, ys = gen_naca4(m, p, t)
        # >>> PATCH: ultra-safe, tighter Airfoil Preview (no tight/constrained/subplots_adjust)
        fig, ax = plt.subplots(figsize=(9.0, 4.8))  # 稍大画布

        # 主曲线
        ax.plot(xs, ys, linewidth=2)

        # 安全的竖线（只在 [0,1] 内才画）
        try:
            if 0.0 <= float(p) <= 1.0:
                ax.axvline(x=float(p), linestyle="--", linewidth=1.2, label="Max camber pos")
            if 0.0 <= float(max_t_pos) <= 1.0:
                ax.axvline(x=float(max_t_pos), linestyle=":", linewidth=1.2, label="Max thickness pos")
            ax.legend(loc="upper right", frameon=False, fontsize=10, borderaxespad=0.3, handlelength=2.4)
        except Exception:
            pass  # 图例出错直接忽略，不影响主图

        # 轴比例与标签
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x/c", labelpad=2)
        ax.set_ylabel("y/c", labelpad=2)
        ax.set_title(f"NACA {naca_code_from_mpt(m, p, t)}", pad=4, fontsize=14)

        # —— 只做“安全边界”计算，全面 NaN/空数组保护 —— #
        mask = np.isfinite(xs) & np.isfinite(ys)
        if np.count_nonzero(mask) >= 2:
            xlo = float(np.nanmin(xs[mask]));
            xhi = float(np.nanmax(xs[mask]))
            ylo = float(np.nanmin(ys[mask]));
            yhi = float(np.nanmax(ys[mask]))
            xr = max(xhi - xlo, 1e-6)
            yr = max(yhi - ylo, 1e-6)
            # 横向几乎贴边；纵向保留 ~6%
            ax.set_xlim(xlo - 0.01 * xr, xhi + 0.01 * xr)
            ax.set_ylim(ylo - 0.06 * yr, yhi + 0.06 * yr)
        else:
            # 极端情况下给个保底边界
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.2, 0.2)

        # 去掉多余脊线，减少视觉留白
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 关键：用固定轴矩形位置“手动紧凑”，避免与任何自动布局冲突
        # [left, bottom, width, height] in figure coordinates (0~1)
        # 可以再把 bottom/left 稍微变小让图更“满”，但不要 <0.05 以免文字被裁
        ax.set_position([0.08, 0.16, 0.90, 0.78])

        # 输出（若你有 fixed-plot 容器就保留；没有也可以直接 st.pyplot）
        with st.container():
            st.markdown('<div class="fixed-plot">', unsafe_allow_html=True)
            st.pyplot(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        Re = estimate_Re(rho, V, chord, mu)
        st.caption(f"Re ≈ {Re:,.0f} · α={alpha:.1f}° · Ncrit={Ncrit:g} · M={M:g}")

        st.markdown("---")

        # ===== 下半：紧凑参数区（3×4 的栅格，尽量压缩竖向空间） =====
        st.markdown("### Parameters", help=None)
        # >>> PATCH: apply pending alpha (must run BEFORE creating the 'alpha_deg' slider anywhere)
        if "pending_alpha_deg" in st.session_state:
            try:
                st.session_state["alpha_deg"] = float(st.session_state.pop("pending_alpha_deg"))
            except Exception:
                st.session_state.pop("pending_alpha_deg", None)
        # >>> END PATCH

        # --- 行1：翼型几何（用 4 列）
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.slider("Camber (%)", 0.0, 10.0, st.session_state.get("camber_pct", 2.0), 0.1, key="camber_pct")
        with g2:
            st.slider("Max camber pos (%)", 0.0, 100.0, st.session_state.get("p_pct", 40.0), 1.0, key="p_pct")
        with g3:
            st.slider("Thickness (%)", 5.0, 20.0, st.session_state.get("thickness_pct", 12.0), 0.1, key="thickness_pct")
        with g4:
            st.slider("Max thickness pos (%)", 0.0, 100.0, st.session_state.get("tpos_pct", 30.0), 1.0, key="tpos_pct")

        # --- 行2：流场参数（4 列）
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.number_input("ρ (kg/m³)", value=float(st.session_state.get("rho", 1.225)), key="rho")
        with f2:
            st.number_input("V (m/s)", value=float(st.session_state.get("V", 10.0)), key="V")
        with f3:
            st.number_input("Chord c (m)", value=float(st.session_state.get("chord", 1.0)),
                            min_value=0.05, step=0.05, key="chord")
        with f4:
            st.number_input("μ (Pa·s)", value=float(st.session_state.get("mu", 1.8e-5)),
                            format="%.6e", key="mu")

        # --- 行3：求解设置（4 列）
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.number_input("Mach", value=float(st.session_state.get("Mach", 0.0)),
                            min_value=0.0, max_value=0.3, step=0.01, key="Mach")
        with s2:
            st.number_input("Ncrit", value=float(st.session_state.get("Ncrit", 7.0)),
                            min_value=1.0, max_value=12.0, step=0.5, key="Ncrit")
        with s3:
            st.slider("Scan range α (°)", 0.0, 15.0, st.session_state.get("alpha_range", (0.0, 10.0)),
                      0.5, key="alpha_range")
        with s4:
            st.number_input("Δα (°)", value=float(st.session_state.get("alpha_step", 1.0)),
                            min_value=0.1, max_value=2.0, step=0.1, key="alpha_step")

        # --- 行4：当前攻角（独占或与其他按钮同行）
        h1, h2 = st.columns([2, 2])
        with h1:
            st.slider("Current α (°)", 0.0, 15.0, st.session_state.get("alpha_deg", 5.0), 0.5, key="alpha_deg")
        with h2:
            # 这里可预留按钮位，比如“一键插入仿真参数”/“保存当前几何”等
            pass

    # === Performance Tab ===
    with tab_perf:
        st.subheader("Performance & Polars")
        m = st.session_state.get("camber_pct", 2.0) / 100
        p = st.session_state.get("p_pct", 40.0) / 100
        t = st.session_state.get("thickness_pct", 12.0) / 100
        alpha = st.session_state.get("alpha_deg", 5.0)
        rho = st.session_state.get("rho", 1.225)
        V = st.session_state.get("V", 10.0)
        chord = st.session_state.get("chord", 1.0)
        mu = st.session_state.get("mu", 1.8e-5)
        M = st.session_state.get("Mach", 0.0)
        Ncrit = st.session_state.get("Ncrit", 7.0)
        a_min, a_max = st.session_state.get("alpha_range", (0.0, 10.0))
        a_step = st.session_state.get("alpha_step", 1.0)
        naca_code = naca_code_from_mpt(m, p, t)
        Re = estimate_Re(rho, V, chord, mu)
        df_polar = run_xfoil_polar(naca_code, Re, M, Ncrit, float(a_min), float(a_max), float(a_step))

        # >>> PATCH: handle XFOIL failure (place right after df_polar = ...)
        if df_polar.empty:
            st.error(
                "XFOIL produced no valid polar. Check the α scan range/step, the Reynolds number magnitude, and Ncrit.")
            if st.button("Apply recommended scan: α = 0–10°, Δα = 0.5°, Re ≈ 3e5, Ncrit ≈ 7",
                         key="btn_apply_recommended_scan", use_container_width=True):
                st.session_state["alpha_range"] = (0.0, 10.0)
                st.session_state["alpha_step"] = 0.5
                st.rerun()

            st.info("A simulated placeholder curve is shown below (non-physical). Adjust inputs and recompute.")
            df_polar = fallback_fake_polar(float(a_min), float(a_max), float(a_step))
        # >>> END PATCH

        idx_current = int(np.argmin(np.abs(df_polar["alpha"].values - alpha)))
        CL = float(df_polar.loc[idx_current, "CL"])
        CD = float(df_polar.loc[idx_current, "CD"])
        LD = CL / CD if CD > 1e-12 else np.nan

        df_valid = df_polar[df_polar["CD"] > 1e-12].copy()
        df_valid["L/D"] = df_valid["CL"] / df_valid["CD"]
        idx_opt = int(df_valid["L/D"].idxmax())
        alpha_opt = float(df_valid.loc[idx_opt, "alpha"])
        ld_max = float(df_valid.loc[idx_opt, "L/D"])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("CL", f"{CL:.3f}")
        k2.metric("CD", f"{CD:.4f}")
        k3.metric("L/D", f"{LD:.1f}" if np.isfinite(LD) else "—")
        k4.metric("α* (best)", f"{alpha_opt:.1f}°")

        st.markdown(f"**Summary:** NACA {naca_code} · Re={Re:,.0f} · L/D_max={ld_max:.1f}")
        # >>> PATCH: current-α explanation + snap button (place right after Summary, BEFORE plotting)
        st.caption(
            "Current α is used to read/mark on charts (it does not change the solved polar). To include it, adjust the scan range.")
        if not df_polar.empty and "alpha" in df_polar:
            if st.button("Snap current α to nearest grid point", key="btn_snap_alpha", use_container_width=True):
                alpha_grid = df_polar["alpha"].to_numpy()
                snapped = float(alpha_grid[np.argmin(np.abs(alpha_grid - alpha))])
                # write to pending key; let tab_geo apply it BEFORE the slider is created
                st.session_state["pending_alpha_deg"] = snapped
                st.rerun()
        # >>> END PATCH

        tab1, tab2, tab3 = st.tabs(["CL vs α", "CD vs α", "L/D vs α"])


        def _vline(ax, a_cur):
            ax.axvline(a_cur, linestyle="--", linewidth=1.2)


        with tab1:
            fig_cl, ax_cl = plt.subplots(figsize=(8.0, 4.2))
            ax_cl.plot(df_polar["alpha"], df_polar["CL"], linewidth=2)
            _vline(ax_cl, alpha)
            ax_cl.set_xlabel("α (deg)");
            ax_cl.set_ylabel("CL");
            ax_cl.set_title("CL vs α")
            st.pyplot(fig_cl, use_container_width=True)

        with tab2:
            fig_cd, ax_cd = plt.subplots(figsize=(8.0, 4.2))
            ax_cd.plot(df_polar["alpha"], df_polar["CD"], linewidth=2)
            _vline(ax_cd, alpha)
            ax_cd.set_xlabel("α (deg)");
            ax_cd.set_ylabel("CD");
            ax_cd.set_title("CD vs α")
            st.pyplot(fig_cd, use_container_width=True)

        with tab3:
            fig2, ax2 = plt.subplots(figsize=(8.0, 4.2))
            ax2.plot(df_valid["alpha"], df_valid["L/D"], linewidth=2)
            _vline(ax2, alpha);
            _vline(ax2, alpha_opt)
            ax2.set_xlabel("α (deg)");
            ax2.set_ylabel("L/D");
            ax2.set_title("L/D vs α (with α* marker)")
            st.pyplot(fig2, use_container_width=True)

        # === Save Button ===
        if st.button("💾 Save this result", use_container_width=True):
            try:
                payload = {
                    "user_id": user_id,
                    "naca_code": naca_code,
                    "camber": m,
                    "thickness": t,
                    "max_camber_pos": p,
                    "alpha": alpha,
                    "rho": rho,
                    "velocity": V,
                    "chord": chord,
                    "mu": mu,
                    "re": Re,
                    "ncrit": Ncrit,
                    "mach": M,
                    "cl": CL,
                    "cd": CD,
                    "ld": LD if np.isfinite(LD) else 0.0,
                    "alpha_opt": alpha_opt,
                    "ld_max": ld_max,
                }
                r = requests.post(f"{BACKEND_URL}/save_airfoil/", json=payload, timeout=10)
                if r.status_code == 200:
                    st.success("✅ Airfoil data saved to backend")
                else:
                    st.error(f"❌ Save failed: {r.text}")
            except Exception as e:
                st.error(f"⚠️ Error when saving: {e}")

    # === History Tab ===
    with tab_hist:
        st.subheader("📜 My History")

        # 刷新按钮
        if st.button("🔄 Refresh History", use_container_width=True):
            st.session_state["refresh_history"] = True

        # 默认第一次进入就刷新
        if "refresh_history" not in st.session_state:
            st.session_state["refresh_history"] = True

        df_hist = None
        if st.session_state["refresh_history"]:
            try:
                resp = requests.get(f"{BACKEND_URL}/export_airfoils/{user_id}", timeout=10).json()
                if resp:
                    df_hist = pd.DataFrame(resp)
                    st.dataframe(
                        df_hist[[
                            "id", "user_id", "naca_code", "camber", "thickness",
                            "max_camber_pos", "alpha", "rho", "velocity", "chord",
                            "mu", "re", "ncrit", "mach", "cl", "cd", "ld",
                            "alpha_opt", "ld_max", "timestamp"
                        ]],
                        use_container_width=True, height=400
                    )
                else:
                    st.info("No saved records yet.")
            except Exception as e:
                st.warning(f"⚠️ Backend fetch failed: {e}")

            # 自动刷新完成后关闭标志，下次只有点刷新按钮才会再请求
            st.session_state["refresh_history"] = False

        # ✅ 如果有数据，提供导出按钮（即便没刷新过也能用）
        if df_hist is not None and not df_hist.empty:
            csv = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download My History (CSV)",
                data=csv,
                file_name=f"{user_id}_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        # === Help Tab ===
        with tab_help:
            st.markdown("## Help Center")
            st.caption("AI-Enhanced Airfoil Design & Learning Lab — Quick reference and troubleshooting")

            st.markdown("### Overview")
            st.markdown("""
        This web app is an **AI-assisted airfoil design and learning environment** combining:  
        1) **Geometry preview & parameter controls** (NACA 4-digit),  
        2) **XFOIL polar analysis** (CL/CD/CM and L/D), and  
        3) **Role-based tutoring** for concept learning, model iteration, and strategy review.
        """)

            st.markdown("### Quick Start")
            st.markdown("""
        1. **Set parameters** in **Geometry & Parameters** (camber `m%`, max camber position `p%`, thickness `t%`, flow & solver settings).  
        2. Switch to **Performance & Polars** to compute polars and view **CL**, **CD**, **L/D**, and **α\***.  
        3. In the **Dialogue** (left), pick a role in **🎭 Choose AI Module**, ask a question, or click **📥 Insert Simulation Data** to auto-paste your current settings, then discuss improvements with the AI.  
        4. Use **History** to view or export saved runs and **Restore historical dialogue** to reload past chats (per user).
        """)

            st.markdown("### Page Layout")
            st.markdown("""
        - **Left column**: Dialogue (role selection, search, history restore, input & send).  
        - **Right column**: Tabs for **Geometry & Parameters**, **Performance & Polars**, **History**, **Help**, and **Admin** (admin only).
        """)

            with st.expander("AI Roles (when to use which)"):
                st.markdown("""
        - **Concept Learning** — Clarify core concepts (e.g., lift coefficient, stall, Reynolds number). Use a **guiding** tone, ask probing questions.  
        - **Model Iteration** — Plan experiments, scan parameters (α range/step, Re, Ncrit) and compare outcomes.  
        - **Strategy Review** — Critically evaluate your approach (claim–evidence–warrant), surface gaps and next steps.
        """)

            with st.expander("Geometry & Parameters (what each control means)"):
                st.markdown("""
        - **NACA 4-digit** (`mpt` → `m%`, `p/10`, `t%`): `m` camber, `p` chordwise location of max camber, `t` thickness.  
        - **Flow**: ρ (density), **V** (velocity), **c** (chord), **μ** (dynamic viscosity). Reynolds number `Re = ρVc/μ`.  
        - **Solver**: **Mach** (typically ≤ 0.3 for incompressible assumptions), **Ncrit** (transition criterion, e.g., 5–9 for typical wind-tunnel atmospheres).  
        - **Scan**: α range `[α_min, α_max]` and step `Δα` for **ASEQ** scanning; use a smaller `Δα` for finer L/D peaks.
        """)

            with st.expander("Performance & Polars (how to read the plots)"):
                st.markdown("""
        - **CL(α)**, **CD(α)**, **CM(α)** are computed from XFOIL polars.  
        - **L/D** helps locate efficient angles; the app highlights **α\*** where L/D is maximal in the scanned range.  
        - If XFOIL returns no valid data, the app will show a **simulated fallback** curve (for UI continuity). Prefer fixing inputs to get physical results.
        """)

            with st.expander("Dialogue & Data (good practices)"):
                st.markdown("""
        - Use **📥 Insert Simulation Data** to paste the current setup into the chat, then ask the AI to critique or suggest iterations.  
        - Phrase prompts for **reasoning**, e.g., *“If I increase `t%` while keeping `Re` fixed, what trade-offs appear in stall and L/D?”*  
        - **History** tab: inspect, refresh, and export your saved runs (CSV). **Admin** can export all users’ data.
        """)

            st.markdown("### Methodological Guidance")
            st.markdown("""
        - Treat the AI as a **Socratic partner**: ask “why/how” questions, test hypotheses, and compare runs under controlled changes.  
        - For argumentation, structure your notes as **Claim–Evidence–Warrant** (and add Qualifiers/Rebuttals when applicable).  
        - Keep **Re**, **α range**, **Δα**, **Ncrit** explicit in your lab notes to ensure **reproducibility**.
        """)

            st.markdown("### FAQ")
            st.markdown("""
        **Q1. My input remains in the box after sending.**  
        The app clears after submission; if you still see text, refresh the page to resync the widget state.

        **Q2. XFOIL returns no polar or `polar.out` is empty.**  
        Check α range/step and `Re` magnitude; try moderate **Ncrit** (e.g., 7) and ensure `xfoil.exe` exists in the app root on Windows. If geometry is extreme (very high camber/thickness or tiny `Re`), start with gentler values.

        **Q3. L/D looks strange or NaN.**  
        This occurs if **CD ≈ 0** or data is sparse. Reduce `Δα`, widen the scan, or adjust `Re`/`Ncrit` for a stable polar.

        **Q4. “Address already in use” on port 8000/8501.**  
        Stop previous processes using those ports (Windows: Task Manager or `netstat` + `taskkill`; Linux/macOS: `lsof -i :PORT` then `kill -9 PID`).
        """)

            st.markdown("### Troubleshooting Checklist")
            st.markdown("""
        - ✅ **Executable**: On Windows, ensure **`xfoil.exe`** is in the project root.  
        - ✅ **Ranges**: Use reasonable **α** ranges (e.g., 0°–10°) and `Δα` (0.5°–1°) to start.  
        - ✅ **Reynolds**: Verify `Re = ρVc/μ` is not pathologically small/large for your case.  
        - ✅ **Ncrit**: Start near 7; move ±2 if convergence is poor.  
        - ✅ **Fallback**: If you see a fallback curve, it means no valid XFOIL data—adjust inputs and recompute.
        """)

            st.markdown("### Notes on Ethics & Data")
            st.markdown("""
        - Use the system for **learning and research**; cite results appropriately in coursework or publications.  
        - Dialogue logs and run data may be stored for analysis; anonymize when required by your IRB/ethics policy.
        """)

            st.info("This Help is adapted from your uploaded quick-guide and refined for clarity and research use. "
                    "For classroom deployment, you may extend with course-specific rubrics and examples. "
                    "Source: internal guide. "
                    ":contentReference[oaicite:0]{index=0}")

    # === Admin Panel ===
    with tab_admin:
        if is_admin:
            st.success("✅ Logged in as Admin")
            st.markdown("### Export All Data")

            # Conversations
            if st.button("📥 Download All Conversations (CSV)", use_container_width=True):
                try:
                    resp = requests.get(f"{BACKEND_URL}/admin/export_all_conversations", timeout=20)
                    if resp.status_code == 200:
                        csv = resp.content
                        st.download_button(
                            label="⬇️ Save Conversations",
                            data=csv,
                            file_name="all_conversations.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.error(f"❌ Failed: {resp.text}")
                except Exception as e:
                    st.error(f"⚠️ Error fetching conversations: {e}")

            # Airfoils
            if st.button("📥 Download All Airfoils (CSV)", use_container_width=True):
                try:
                    resp = requests.get(f"{BACKEND_URL}/admin/export_all_airfoils", timeout=20)
                    if resp.status_code == 200:
                        csv = resp.content
                        st.download_button(
                            label="⬇️ Click here to save Airfoils",
                            data=csv,
                            file_name="all_airfoils.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.error(f"❌ Failed: {resp.text}")
                except Exception as e:
                    st.error(f"⚠️ Error fetching airfoils: {e}")
        else:
            st.warning("Enter the correct admin password to access this panel.")
