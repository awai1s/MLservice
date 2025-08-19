from fastapi import FastAPI
from .routers import health, price, duplicate, recommend

app = FastAPI(title="Campus Exchange ML Service", version="1.0.0")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Campus Exchange ML Service!"}

# Health
app.include_router(health.router)

# EXACT paths required by backend (no prefix):
app.include_router(price.router)
app.include_router(duplicate.router)
app.include_router(recommend.router)

# --- add these two endpoints so Railway healthcheck passes ---
@app.get("/healthz", include_in_schema=False)
def _healthz_root():
    return {"status": "ok"}

@app.get("/api/healthz", include_in_schema=False)
def _healthz_api():
    # Railway pings this path
    return {"status": "ok"}
# ------------------------------------------------------------
