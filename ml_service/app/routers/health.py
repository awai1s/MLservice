from fastapi import APIRouter

router = APIRouter(tags=["Health"])

@router.get("/healthz")
def healthz():
    return {"ok": True}
