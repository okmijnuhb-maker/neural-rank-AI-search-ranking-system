# main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import get_db, QueryLog, ModelMetric, User
from ranker import search
import time
import datetime

app = FastAPI(title="NeuralRank API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class SearchRequest(BaseModel):
    query: str
    user_id: int = None

class LoginRequest(BaseModel):
    username: str
    password: str

@app.get("/")
def root():
    return {"message": "NeuralRank API is running!"}

@app.post("/search")
def search_endpoint(req: SearchRequest, db: Session = Depends(get_db)):
    start = time.time()
    results = search(req.query)
    elapsed = int((time.time() - start) * 1000)

    log = QueryLog(
        user_id=req.user_id,
        query_text=req.query,
        model_used="both",
        response_time_ms=elapsed,
        results_count=len(results['lambdamart'])
    )
    db.add(log)
    db.commit()

    return {"query": req.query, "response_time_ms": elapsed, "results": results}

@app.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    metrics = db.query(ModelMetric).all()
    return [{"model": m.model_name, "ndcg": m.ndcg_at_10, "mrr": m.mrr, "precision": m.precision_at_10} for m in metrics]

@app.get("/admin/queries")
def get_queries(db: Session = Depends(get_db)):
    logs = db.query(QueryLog).order_by(QueryLog.created_at.desc()).limit(100).all()
    return [{"id": l.id, "query": l.query_text, "time_ms": l.response_time_ms, "created_at": str(l.created_at)} for l in logs]

@app.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()
    if not user or user.password_hash != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "role": user.role, "username": user.username}

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

@app.post("/signup")
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(
        (User.username == req.username) | (User.email == req.email)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    
    user = User(username=req.username, email=req.email,
                password_hash=req.password, role='user')
    db.add(user)
    db.commit()
    return {"message": "Account created successfully"}

@app.get("/admin/users")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at.asc()).all()
    return [{"id": u.id, "username": u.username, "email": u.email,
             "role": u.role, "created_at": str(u.created_at)} for u in users]