from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import ORJSONResponse
from os.path import basename
from typing import Optional, Annotated
from pydantic import BaseModel, Field
from os import path
from starlette.responses import FileResponse
router = APIRouter(
    prefix=f"/{basename(__file__).replace('.py', '')}",
    tags=[basename(__file__).replace('.py', '')],
    dependencies=[],
    responses={404: {"description": "Not found"}}
)

class FileBody(BaseModel):
    file: UploadFile = Field(..., description="File to upload")
@router.post("/upload")
async def upload(body: FileBody):
    r'''
    Uploading the file to the backend/files directory.
    ```
        - file: file to upload
    ```
    '''
    with open(f"files/{body.file.filename}", "wb") as f:
        f.write(await body.file.read())
    
    return ORJSONResponse(content={"message": "File uploaded successfully"})

@router.get("/download")
class QueryParam(BaseModel):
    file: str = Field(..., description="File to download")
@router.get("/download")
async def download(params: QueryParam = Depends()):
    r'''
    Downloading the file from the backend/files directory.
    ```
        - file: file to download
    ```
    '''
    file_path = f"files/{params.file}"
    if not path.exists(file_path):
        return ORJSONResponse(content={"message": "File not found"}, status_code=404)
    
    return FileResponse(file_path, media_type="application/octet-stream", filename=params.file)
    
    