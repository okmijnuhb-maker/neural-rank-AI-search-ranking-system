# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, TIMESTAMP, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

# Use SQLite for Render, MySQL for local
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./neuralrank.db")

# Fix for SQLite
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    email = Column(String(100), unique=True)
    password_hash = Column(String(255))
    role = Column(String(10), default='user')
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)

class QueryLog(Base):
    __tablename__ = "queries_log"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    query_text = Column(String(500))
    model_used = Column(String(20))
    response_time_ms = Column(Integer)
    results_count = Column(Integer)
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)

class ModelMetric(Base):
    __tablename__ = "model_metrics"
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50))
    ndcg_at_10 = Column(Float)
    mrr = Column(Float)
    precision_at_10 = Column(Float)
    recorded_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)

# Create all tables automatically
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
