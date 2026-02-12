# -*- coding: utf-8 -*-
# File: backend.py
# Description: Airfoil Assistant Backend (Windows + per-user history + admin export)

from fastapi import FastAPI, Response
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd
import io
import os
from datetime import timezone, timedelta

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TZ_E8 = ZoneInfo("Asia/Shanghai")
except Exception:
    TZ_E8 = timezone(timedelta(hours=8))  # 兜底：+08:00


def as_e8(dt):
    """将数据库里的时间（通常为UTC且naive）转换为东八区；返回带时区信息的datetime。"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.astimezone(TZ_E8)


# ===== Database Setup =====
DB_PATH = os.getenv("DB_PATH", "aero_data.db")   # 默认 SQLite 本地文件
DATABASE_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


# ===== Table Definitions =====
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100))
    role = Column(String(50))
    student_question = Column(Text)
    ai_response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)


class AirfoilHistory(Base):
    __tablename__ = "airfoil_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100))
    naca_code = Column(String(10))
    camber = Column(Float)
    thickness = Column(Float)
    max_camber_pos = Column(Float)

    # ✅ NEW: max thickness position (normalized 0..1) + optional raw percent (0..100)
    max_thickness_pos = Column(Float, default=None)  # 0..1
    tpos_pct = Column(Float, default=None)           # 0..100 (optional)

    alpha = Column(Float)
    rho = Column(Float)
    velocity = Column(Float)
    chord = Column(Float)
    mu = Column(Float)
    re = Column(Float)
    ncrit = Column(Float)
    mach = Column(Float)
    cl = Column(Float)
    cd = Column(Float)
    ld = Column(Float)
    alpha_opt = Column(Float)
    ld_max = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True)
    password = Column(String(100))  # Plain text as requested for visibility
    role = Column(String(20), default="user")  # "admin" or "user"


# ===== Minimal SQLite Migration =====
def _sqlite_add_column_if_missing(table: str, col: str, col_type_sql: str):
    """
    For SQLite: check PRAGMA table_info and ALTER TABLE to add missing columns.
    Safe to run repeatedly.
    """
    with engine.begin() as conn:
        res = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        existing_cols = {r[1] for r in res}  # r[1] is column name
        if col not in existing_cols:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type_sql}"))


def migrate_if_needed():
    # Only needed for existing DBs where table already created without new columns
    try:
        _sqlite_add_column_if_missing("airfoil_history", "max_thickness_pos", "FLOAT")
        _sqlite_add_column_if_missing("airfoil_history", "tpos_pct", "FLOAT")
    except Exception:
        # If migration fails, app still starts; but new fields may not persist.
        pass


Base.metadata.create_all(engine)
migrate_if_needed()

# ===== Initialize Default Admin =====
def init_admin():
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            print("initializing default admin user...")
            new_admin = User(username="admin", password="ecnusjtu", role="admin")
            db.add(new_admin)
            db.commit()
            print("✅ Default admin created: admin / ecnusjtu")
    except Exception as e:
        print(f"Error initializing admin: {e}")
    finally:
        db.close()

init_admin()

# ===== FastAPI App =====
app = FastAPI(title="Airfoil Assistant Backend", version="1.4")

# ===== CORS (allow React frontend) =====
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Multi-Agent System =====
try:
    from agent_api import router as agent_router
    app.include_router(agent_router)
    print("✅ Multi-Agent System loaded successfully")
except ImportError as e:
    print(f"⚠️ Multi-Agent System not available: {e}")
    print("   Install with: pip install -r agents/requirements.txt")

# ===== XFOIL Simulation =====
import tempfile
import subprocess
import numpy as np

def gen_naca4_coords(m, p, t, tpos=0.30, n_pts=200):
    x = np.linspace(0, 1, n_pts)
    x0, xt = 0.30, max(1e-6, min(1-1e-6, tpos))
    x_mapped = np.where(x <= x0, x * (xt / x0), xt + (x - x0) * ((1.0 - xt) / (1.0 - x0))) if abs(xt - x0) > 1e-8 else x
    yt = 5.0 * t * (0.2969 * np.sqrt(np.clip(x_mapped, 1e-12, 1)) - 0.1260 * x_mapped - 0.3516 * x_mapped**2 + 0.2843 * x_mapped**3 - 0.1015 * x_mapped**4)
    p_safe = max(1e-6, min(1-1e-6, p))
    yc = np.where(x < p_safe, (m / (p_safe**2)) * (2*p_safe*x - x**2), (m / ((1-p_safe)**2)) * ((1-2*p_safe) + 2*p_safe*x - x**2))
    dyc = np.where(x < p_safe, (2*m / (p_safe**2)) * (p_safe - x), (2*m / ((1-p_safe)**2)) * (p_safe - x))
    theta = np.arctan(dyc)
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)
    return np.concatenate([xl[::-1], xu[1:]]), np.concatenate([yl[::-1], yu[1:]])

def run_xfoil_polar(xs, ys, Re, Mach, Ncrit, a_start, a_end, a_step, name="AIRFOIL"):
    exe = os.path.abspath("xfoil.exe")
    if not os.path.exists(exe):
        return None, "xfoil.exe not found"
    with tempfile.TemporaryDirectory() as td:
        dat, pol = os.path.join(td, "airfoil.dat"), os.path.join(td, "polar.out")
        with open(dat, "w") as f:
            f.write(f"{name}\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys)))
        script = f"LOAD {dat}\nPANE\nOPER\nVISC {Re:.6e}\nMACH {Mach:.4f}\nVPAR\nN {int(Ncrit)}\n\nPACC\n{pol}\n\nASEQ {a_start:.3f} {a_end:.3f} {a_step:.3f}\nPACC\n\nQUIT\n"
        try:
            subprocess.run([exe], input=script, text=True, capture_output=True, timeout=90)
        except Exception as e:
            return None, str(e)
        if not os.path.exists(pol):
            return None, "XFOIL did not produce output"
        rows = []
        with open(pol, "r") as f:
            for line in f:
                s = line.strip()
                if not s or "alpha" in s.lower() or set(s) <= set("-= "):
                    continue
                parts = s.split()
                if len(parts) >= 5:
                    try:
                        rows.append({"alpha": float(parts[0]), "CL": float(parts[1]), "CD": float(parts[2]), "CM": float(parts[4])})
                    except:
                        continue
        return rows if rows else None, None if rows else "No valid data"


def estimate_cp_distribution(camber, thickness, max_camber_pos, alpha_deg, n=40):
    """Estimate Cp distribution using thin-airfoil theory (always available fallback)."""
    import math
    alpha_rad = math.radians(alpha_deg)
    cp_data = []
    p_safe = max(max_camber_pos, 0.01)
    for i in range(n + 1):
        theta = math.pi * i / n
        x = (1 - math.cos(theta)) / 2
        # Thickness effect on local velocity
        t_effect = thickness * (1 - 2 * x * x)
        # Camber effect
        c_effect = camber * (1 - 2 * x) / p_safe if camber > 0 else 0.0
        # Upper surface: accelerated
        x_safe = max(x, 0.005)
        v_upper = 1 + alpha_rad * (1 - x) / x_safe * 0.05 + t_effect + c_effect * 0.5
        cp_upper = max(-6, min(1, 1 - v_upper * v_upper))
        cp_data.append({"segment": "upper", "x": round(x, 4), "cp": round(cp_upper, 4)})
        # Lower surface: decelerated
        v_lower = 1 - alpha_rad * (1 - x) / x_safe * 0.03 - t_effect * 0.5 - c_effect * 0.3
        cp_lower = max(-6, min(1, 1 - v_lower * v_lower))
        cp_data.append({"segment": "lower", "x": round(x, 4), "cp": round(cp_lower, 4)})
    return cp_data


class SimulateRequest(BaseModel):
    camber: float
    thickness: float
    max_camber_pos: float
    max_thickness_pos: float
    alpha: float
    rho: float
    velocity: float
    chord: float
    mu: float
    ncrit: float
    mach: float
    alpha_start: float
    alpha_end: float
    alpha_step: float
    user_id: str = "guest"


@app.post("/simulate")
def simulate(req: SimulateRequest):
    xs, ys = gen_naca4_coords(req.camber, req.max_camber_pos, req.thickness, req.max_thickness_pos)
    Re = (req.rho * req.velocity * req.chord) / max(req.mu, 1e-12)
    naca_code = f"{int(req.camber*100)}{int(req.max_camber_pos*10)}{int(req.thickness*100):02d}"
    
    # Try XFOIL (will fail on macOS with Windows exe)
    polar, error = run_xfoil_polar(xs, ys, Re, req.mach, req.ncrit, req.alpha_start, req.alpha_end, req.alpha_step, f"NACA{naca_code}")
    
    if polar is None:
        # Fallback to estimation
        polar = [{"alpha": float(a), "CL": 0.11*a + 0.2*req.camber*10, "CD": 0.008 + 0.0001*a**2, "CM": -0.05} for a in np.arange(req.alpha_start, req.alpha_end + 0.001, req.alpha_step)]
        error = error or "Fallback estimation (XFOIL not available)"
    
    current = min(polar, key=lambda p: abs(p["alpha"] - req.alpha))
    cl, cd = current["CL"], max(current["CD"], 1e-12)
    best = max(polar, key=lambda p: p["CL"] / max(p["CD"], 1e-12))
    ld = cl / cd
    alpha_opt = best["alpha"]
    ld_max = best["CL"] / max(best["CD"], 1e-12)
    
    # Save to database
    try:
        db = SessionLocal()
        record = AirfoilHistory(
            user_id=req.user_id,
            naca_code=naca_code,
            camber=req.camber,
            thickness=req.thickness,
            max_camber_pos=req.max_camber_pos,
            max_thickness_pos=req.max_thickness_pos,
            tpos_pct=req.max_thickness_pos * 100 if req.max_thickness_pos else None,
            alpha=req.alpha,
            rho=req.rho,
            velocity=req.velocity,
            chord=req.chord,
            mu=req.mu,
            re=Re,
            ncrit=req.ncrit,
            mach=req.mach,
            cl=cl,
            cd=cd,
            ld=ld,
            alpha_opt=alpha_opt,
            ld_max=ld_max,
        )
        db.add(record)
        db.commit()
        db.close()
    except Exception as e:
        print(f"Failed to save to database: {e}")
    
    # Compute Cp distribution
    cp_data = estimate_cp_distribution(req.camber, req.thickness, req.max_camber_pos, req.alpha)
    
    return {
        "status": "success", 
        "data": {
            "polar": polar, 
            "kpi": {"cl": cl, "cd": cd, "ld": ld, "alphaOpt": alpha_opt, "ldMax": ld_max}, 
            "geometry": {"x": xs.tolist(), "y": ys.tolist(), "nacaCode": naca_code}, 
            "cpData": cp_data,
            "re": Re
        }, 
        "warning": error
    }


# ===== Pydantic Models =====
class ConversationIn(BaseModel):
    user_id: str
    role: str
    student_question: str
    ai_response: str


class AirfoilHistoryIn(BaseModel):
    user_id: str
    naca_code: str
    camber: float
    thickness: float
    max_camber_pos: float

    # ✅ NEW fields (front-end should send these)
    max_thickness_pos: float | None = None  # 0..1
    tpos_pct: float | None = None           # 0..100 (optional)

    alpha: float
    rho: float
    velocity: float
    chord: float
    mu: float
    re: float
    ncrit: float
    mach: float
    cl: float
    cd: float
    ld: float
    alpha_opt: float
    alpha_opt: float
    ld_max: float


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


# ===== Routes =====
@app.post("/save_conversation/")
def save_conversation(data: ConversationIn):
    db = SessionLocal()
    record = Conversation(
        user_id=data.user_id,
        role=data.role,
        student_question=data.student_question,
        ai_response=data.ai_response or "",
        timestamp=datetime.utcnow()
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    db.close()
    return {"status": "success", "message": "Conversation saved.", "id": record.id}


@app.post("/save_airfoil/")
def save_airfoil(data: AirfoilHistoryIn):
    db = SessionLocal()
    record = AirfoilHistory(**data.dict())
    db.add(record)
    db.commit()
    db.close()
    return {"status": "success", "message": "Airfoil data saved."}


@app.delete("/delete_history/{user_id}")
def delete_history(user_id: str):
    db = SessionLocal()
    try:
        # Delete conversations for this user
        num_deleted = db.query(Conversation).filter(Conversation.user_id == user_id).delete()
        db.commit()
        return {"status": "success", "message": f"Deleted {num_deleted} conversations.", "count": num_deleted}
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


# ===== Auth & Admin Routes =====
@app.post("/auth/login")
def login(creds: LoginRequest):
    db = SessionLocal()
    user = db.query(User).filter(User.username == creds.username).first()
    db.close()
    
    if not user or user.password != creds.password:
        return {"status": "error", "message": "Invalid credentials"}
    
    return {
        "status": "success",
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role
        }
    }


@app.get("/admin/users")
def list_users():
    db = SessionLocal()
    users = db.query(User).all()
    db.close()
    return [{
        "id": u.id,
        "username": u.username,
        "password": u.password, # Show password as requested
        "role": u.role
    } for u in users]


@app.post("/admin/users")
def create_user(req: CreateUserRequest):
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == req.username).first()
        if existing:
            return {"status": "error", "message": "Username already exists"}
        
        new_user = User(username=req.username, password=req.password, role=req.role)
        db.add(new_user)
        db.commit()
        return {"status": "success", "message": "User created"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


@app.delete("/admin/users/{user_id}")
def delete_user(user_id: int):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
             return {"status": "error", "message": "User not found"}
        if user.username == "admin":
             return {"status": "error", "message": "Cannot delete default admin"}
             
        db.delete(user)
        db.commit()
        return {"status": "success", "message": "User deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

# ===== Export (per-user only) =====
@app.get("/export_conversations/{user_id}")
def export_conversations(user_id: str):
    db = SessionLocal()
    records = db.query(Conversation).filter(Conversation.user_id == user_id).all()
    db.close()
    return [{
        "id": r.id, "user_id": r.user_id, "role": r.role,
        "student_question": r.student_question, "ai_response": r.ai_response,
        "timestamp": as_e8(r.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    } for r in records]


@app.get("/export_airfoils/{user_id}")
def export_airfoils(user_id: str):
    db = SessionLocal()
    records = db.query(AirfoilHistory).filter(AirfoilHistory.user_id == user_id).all()
    db.close()
    return [{
        "id": r.id, "user_id": r.user_id, "naca_code": r.naca_code,
        "camber": r.camber, "thickness": r.thickness,
        "max_camber_pos": r.max_camber_pos,

        # ✅ NEW in per-user export
        "max_thickness_pos": r.max_thickness_pos,
        "tpos_pct": r.tpos_pct,

        "alpha": r.alpha, "rho": r.rho, "velocity": r.velocity, "chord": r.chord,
        "mu": r.mu, "re": r.re, "ncrit": r.ncrit, "mach": r.mach,
        "cl": r.cl, "cd": r.cd, "ld": r.ld,
        "alpha_opt": r.alpha_opt, "ld_max": r.ld_max,
        "timestamp": as_e8(r.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    } for r in records]


# ===== Admin Export (all users) =====
@app.get("/admin/export_all_conversations")
def export_all_conversations():
    db = SessionLocal()
    records = db.query(Conversation).all()
    db.close()

    df = pd.DataFrame([{
        "id": r.id,
        "user_id": r.user_id,
        "ai_module": r.role,
        "student_question": r.student_question,
        "ai_response": r.ai_response,
        "timestamp": r.timestamp
    } for r in records])

    if not df.empty:
        s = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        s = s.dt.tz_convert("Asia/Shanghai")
        df["timestamp"] = s.dt.strftime("%Y-%m-%d %H:%M:%S")
        df = df[["id", "user_id", "ai_module", "student_question", "ai_response", "timestamp"]]

    stream = io.StringIO()
    df.to_csv(stream, index=False, encoding="utf-8-sig")
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=all_conversations.csv"}
    )


@app.get("/admin/export_all_airfoils")
def export_all_airfoils():
    db = SessionLocal()
    records = db.query(AirfoilHistory).all()
    db.close()

    df = pd.DataFrame([{
        "user_id": r.user_id, "naca_code": r.naca_code,
        "camber": r.camber, "thickness": r.thickness,
        "max_camber_pos": r.max_camber_pos,

        # ✅ NEW in admin export
        "max_thickness_pos": r.max_thickness_pos,
        "tpos_pct": r.tpos_pct,

        "alpha": r.alpha, "rho": r.rho, "velocity": r.velocity, "chord": r.chord, "mu": r.mu,
        "re": r.re, "ncrit": r.ncrit, "mach": r.mach,
        "cl": r.cl, "cd": r.cd, "ld": r.ld,
        "alpha_opt": r.alpha_opt, "ld_max": r.ld_max,
        "timestamp": r.timestamp
    } for r in records])

    if not df.empty:
        s = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        s = s.dt.tz_convert("Asia/Shanghai")
        df["timestamp"] = s.dt.strftime("%Y-%m-%d %H:%M:%S")

    stream = io.StringIO()
    df.to_csv(stream, index=False, encoding="utf-8-sig")
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=all_airfoils.csv"}
    )
