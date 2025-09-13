# -*- coding: utf-8 -*-
# File: backend.py
# Description: Airfoil Assistant Backend (Windows + per-user history + admin export)

from fastapi import FastAPI, Response
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime
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
    # 若是naive，按UTC解释；若已有tz，先转UTC再转上海
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

Base.metadata.create_all(engine)

# ===== FastAPI App =====
app = FastAPI(title="Airfoil Assistant Backend", version="1.2")

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
    ld_max: float

# ===== Routes =====
@app.post("/save_conversation/")
def save_conversation(data: ConversationIn):
    db = SessionLocal()
    record = Conversation(
        user_id=data.user_id,
        role=data.role,
        student_question=data.student_question,
        ai_response=data.ai_response or "",  # 即便为空也写入
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
        "camber": r.camber, "thickness": r.thickness, "max_camber_pos": r.max_camber_pos,
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

    # 👇 在这里将 role 映射为更直观的 ai_module，同时保留 id 方便追溯
    df = pd.DataFrame([{
        "id": r.id,
        "user_id": r.user_id,
        "ai_module": r.role,                 # ⬅️ 关键：把当时的 AI 模块导出
        "student_question": r.student_question,
        "ai_response": r.ai_response,
        "timestamp": r.timestamp             # 先塞原值，随后统一转时区/格式
    } for r in records])

    if not df.empty:
        # 统一按 UTC 解释，再转上海时区
        s = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        s = s.dt.tz_convert("Asia/Shanghai")
        df["timestamp"] = s.dt.strftime("%Y-%m-%d %H:%M:%S")

        # 可选：调整列顺序更友好
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
        "camber": r.camber, "thickness": r.thickness, "max_camber_pos": r.max_camber_pos,
        "alpha": r.alpha, "rho": r.rho, "velocity": r.velocity, "chord": r.chord, "mu": r.mu,
        "re": r.re, "ncrit": r.ncrit, "mach": r.mach,
        "cl": r.cl, "cd": r.cd, "ld": r.ld,
        "alpha_opt": r.alpha_opt, "ld_max": r.ld_max,
        "timestamp": r.timestamp  # 先塞原值
    } for r in records])

    if not df.empty:
        s = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        s = s.dt.tz_convert("Asia/Shanghai")
        df["timestamp"] = s.dt.strftime("%Y-%m-%d %H:%M:%S")

    stream = io.StringIO()
    df.to_csv(stream, index=False, encoding="utf-8-sig")
    return Response(content=stream.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=all_airfoils.csv"})
