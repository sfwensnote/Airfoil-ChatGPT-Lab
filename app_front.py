# -*- coding: utf-8 -*-
# File: app.py
# Description: Multi-user Airfoil Design Assistant (Windows + Admin Panel)

import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain.memory import ConversationBufferMemory

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil, subprocess, tempfile, os, requests

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
# —— LangGraph + ChatGPT ——
# =========================
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
memory = ConversationBufferMemory(memory_key="history")

roles = {
    "Concept Learning": "Assist with conceptual learning, guide clarification of complex tasks, and identify key variables.",
    "Model Iteration": "Assist with model iteration, analyze data and guide reasoning about parameter relationships, leading to conclusions and concepts.",
    "Strategy Review": "Assist with reviewing strategies, provide task-solving paths and parameter-tuning strategies."
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


def get_prompt(role):
    return ChatPromptTemplate.from_messages([
        ("system", roles[role] + "\nHistorical Dialogue: {history}"),
        ("user", "{question}")
    ])


class GraphState:
    role: str
    history: str
    question: str


def create_node(role: str) -> Runnable:
    prompt = get_prompt(role)
    return prompt | llm


graph = StateGraph(GraphState)
for r in roles:
    graph.add_node(r, create_node(r))


async def route_role(state: GraphState):
    return state.role


graph.add_node("router", route_role)
graph.set_entry_point("router")
graph.add_conditional_edges("router", route_role, {r: r for r in roles})
for r in roles:
    graph.add_edge(r, END)
app = graph.compile()

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
st.title("Airfoil Design Assistant")

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

# ==== Layout ====
col_chat, col_main = st.columns([1.05, 1.95], gap="large")

# ===== Left: Dialogue =====
with col_chat:
    st.subheader("AI Dialogue")
    selected_role = st.selectbox("Select AI Module", list(roles.keys()), index=0, key="role_select")
    if st.session_state.prev_role != selected_role:
        st.session_state.prev_role = selected_role
        st.info(f"**{selected_role} Module**\n\n{roles[selected_role]}")
        for i, g in enumerate(role_guides[selected_role], 1):
            if st.button(f"Example {i}: {g}", key=f"guide_{selected_role}_{i}"):
                st.session_state.setdefault("question_buf", "")
                st.session_state.question_buf = g
    question = st.text_area("📝 Enter your question", key="question_buf", height=140)
    submit = st.button("Submit", use_container_width=True)
    if submit and question.strip():
        history = memory.load_memory_variables({}).get("history", "")
        state = GraphState(role=selected_role, history=history, question=question.strip())
        response = app.invoke(state)
        st.markdown(f"### 🤖 AI Response ({selected_role})")
        st.write(getattr(response, "content", response))
        memory.save_context({"input": question}, {"output": getattr(response, "content", str(response))})
        try:
            requests.post(f"{BACKEND_URL}/save_conversation/", json={
                "user_id": user_id,
                "role": selected_role,
                "student_question": question,
                "ai_response": getattr(response, "content", str(response))
            })
        except Exception as e:
            st.warning(f"⚠️ Failed to save conversation: {e}")

# ===== Right: Tabs =====
with col_main:
    tab_geo, tab_perf, tab_hist, tab_admin = st.tabs([
        "🧩 Geometry & Parameters", "📈 Performance & Polars", "🗂️ History", "🔑 Admin"
    ])

    # === Geometry Tab ===
    with tab_geo:
        cg, pg = st.columns([1.2, 1.8], gap="large")
        with cg:
            st.subheader("Airfoil Parameters")
            st.slider("Camber (%)", 0.0, 10.0, 2.0, 0.1, key="camber_pct")
            st.slider("Max camber position (%)", 0.0, 100.0, 40.0, 1.0, key="p_pct")
            st.slider("Thickness (%)", 5.0, 20.0, 12.0, 0.1, key="thickness_pct")
            st.slider("Max thickness position (%)", 0.0, 100.0, 30.0, 1.0, key="tpos_pct")
            st.divider(); st.subheader("Fluid Parameters (Re)")
            st.number_input("Air density ρ (kg/m³)", value=1.225, key="rho")
            st.number_input("Flow velocity V (m/s)", value=10.0, key="V")
            st.number_input("Chord length c (m)", value=1.0, min_value=0.05, step=0.05, key="chord")
            st.number_input("Dynamic viscosity μ (Pa·s)", value=1.8e-5, format="%.6e", key="mu")
            st.divider(); st.subheader("Solver Settings (XFOIL)")
            st.number_input("Mach number M", value=0.0, min_value=0.0, max_value=0.3, step=0.01, key="Mach")
            st.number_input("Ncrit", value=7.0, min_value=1.0, max_value=12.0, step=0.5, key="Ncrit")
            st.slider("Polar scan range (°)", 0.0, 15.0, (0.0, 10.0), 0.5, key="alpha_range")
            st.number_input("Scan step Δα (°)", value=1.0, min_value=0.1, max_value=2.0, step=0.1, key="alpha_step")
            st.slider("Current angle of attack α (°)", 0.0, 15.0, 5.0, 0.5, key="alpha_deg")
        with pg:
            st.subheader("Airfoil Preview")
            m, p, t, alpha, rho, V, chord, mu, M, Ncrit, a_min, a_max, a_step, max_t_pos = \
                (st.session_state.get("camber_pct",2.0)/100,
                 st.session_state.get("p_pct",40.0)/100,
                 st.session_state.get("thickness_pct",12.0)/100,
                 st.session_state.get("alpha_deg",5.0),
                 st.session_state.get("rho",1.225),
                 st.session_state.get("V",10.0),
                 st.session_state.get("chord",1.0),
                 st.session_state.get("mu",1.8e-5),
                 st.session_state.get("Mach",0.0),
                 st.session_state.get("Ncrit",7.0),
                 st.session_state.get("alpha_range",(0.0,10.0))[0],
                 st.session_state.get("alpha_range",(0.0,10.0))[1],
                 st.session_state.get("alpha_step",1.0),
                 st.session_state.get("tpos_pct",30.0)/100)
            xs, ys = gen_naca4(m, p, t)
            fig, ax = plt.subplots(figsize=(7.2,4.6))
            ax.plot(xs, ys, linewidth=2)
            ax.axvline(x=p, linestyle='--', label='Max camber pos')
            ax.axvline(x=max_t_pos, linestyle=':', label='Max thickness pos')
            ax.set_aspect('equal','box')
            ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
            ax.set_title(f"NACA {naca_code_from_mpt(m,p,t)}")
            ax.legend(); st.pyplot(fig, use_container_width=True)
            Re = estimate_Re(rho, V, chord, mu)
            st.caption(f"Re ≈ {Re:,.0f} · α={alpha:.1f}° · Ncrit={Ncrit:g} · M={M:g}")

    # === Performance Tab ===
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

        if df_polar.empty:
            st.warning("⚠️ No valid polar data. Showing simulated fallback data instead.")
            df_polar = fallback_fake_polar(float(a_min), float(a_max), float(a_step))

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

        fig2, ax2 = plt.subplots(figsize=(8.0, 4.5))
        ax2.plot(df_valid["alpha"], df_valid["L/D"], linewidth=2)
        ax2.axvline(alpha_opt, linestyle="--")
        ax2.set_xlabel("α (deg)")
        ax2.set_ylabel("L/D")
        ax2.set_title("L/D vs α")
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

    # === Admin Panel ===
    with tab_admin:
        if is_admin:
            st.success("✅ Logged in as Admin")
            st.markdown("### Export All Data")
            st.markdown(f"[📥 Download All Conversations (CSV)]({BACKEND_URL}/admin/export_all_conversations)")
            st.markdown(f"[📥 Download All Airfoils (CSV)]({BACKEND_URL}/admin/export_all_airfoils)")
        else:
            st.warning("Enter the correct admin password to access this panel.")
