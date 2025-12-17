from fastapi import FastAPI
from dacribagents.infrastructure.http.sms_ingest import router as sms_router

app = FastAPI()
app.include_router(sms_router)
