from fastapi import FastAPI
import uvicorn
from routers.api import router as api_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os

load_dotenv()
os.chdir(os.path.dirname(os.path.abspath(__file__)))


app = FastAPI(dependencies=[])

# change the port if you want (react app)
origins = [
    "https://ttds18-67d62zc6ua-ew.a.run.app/",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )
