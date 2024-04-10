from fastapi import FastAPI, HTTPException
import uvicorn
from routers.api import router as api_router
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
from starlette.exceptions import HTTPException as StarletteHTTPException

load_dotenv()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI(dependencies=[])
app.include_router(api_router)
class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except (HTTPException, StarletteHTTPException) as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            else:
                raise ex
            
app.mount("/", SPAStaticFiles(directory="react", html=True), name="static")
