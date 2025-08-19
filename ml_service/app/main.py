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

# (Optional) also expose versioned paths if you want:
# app.include_router(price.router,  prefix="/api/v1")
# app.include_router(duplicate.router,  prefix="/api/v1")
# app.include_router(recommend.router, prefix="/api/v1")
