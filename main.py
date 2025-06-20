import asyncio
import aiofiles
import requests
import json
import tempfile
import os
from urllib.parse import urlparse
from azure.storage.filedatalake.aio import DataLakeDirectoryClient, FileSystemClient
from azure.storage.filedatalake import ContentSettings
import mimetypes
import logging
from pprint import pprint
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
API_SUBSCRIPTION_KEY = os.getenv("API_SUBSCRIPTION_KEY")

app = FastAPI(title="Speech-to-Text API", version="1.0.0")

origins = [
    "https://audioanalyserui.vercel.app/",
    "https://audioanalyserui.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SarvamClient:
    def __init__(self, url: str):
        self.account_url, self.file_system_name, self.directory_name, self.sas_token = (
            self._extract_url_components(url)
        )
        self.lock = asyncio.Lock()
        logger.info(f"Initialized SarvamClient with directory: {self.directory_name}")

    def update_url(self, url: str):
        self.account_url, self.file_system_name, self.directory_name, self.sas_token = (
            self._extract_url_components(url)
        )
        logger.info(f"Updated URL to directory: {self.directory_name}")

    def _extract_url_components(self, url: str):
        parsed_url = urlparse(url)
        account_url = f"{parsed_url.scheme}://{parsed_url.netloc}".replace(
            ".blob.", ".dfs."
        )
        path_components = parsed_url.path.strip("/").split("/")
        file_system_name = path_components[0]
        directory_name = "/".join(path_components[1:])
        sas_token = parsed_url.query
        return account_url, file_system_name, directory_name, sas_token

    async def upload_files(self, local_file_paths, overwrite=True):
        logger.info(f"Starting upload of {len(local_file_paths)} files")
        async with DataLakeDirectoryClient(
                account_url=f"{self.account_url}?{self.sas_token}",
                file_system_name=self.file_system_name,
                directory_name=self.directory_name,
                credential=None,
        ) as directory_client:
            tasks = []
            for path in local_file_paths:
                file_name = path.split("/")[-1]
                tasks.append(
                    self._upload_file(directory_client, path, file_name, overwrite)
                )
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful_uploads = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(f"Upload completed for {successful_uploads} files")
            return successful_uploads

    async def _upload_file(
            self, directory_client, local_file_path, file_name, overwrite=True
    ):
        try:
            async with aiofiles.open(local_file_path, mode="rb") as file_data:
                mime_type = mimetypes.guess_type(local_file_path)[0] or "audio/wav"
                file_client = directory_client.get_file_client(file_name)
                data = await file_data.read()
                await file_client.upload_data(
                    data,
                    overwrite=overwrite,
                    content_settings=ContentSettings(content_type=mime_type),
                )
                logger.info(f"File uploaded successfully: {file_name} (Type: {mime_type})")
                return True
        except Exception as e:
            logger.error(f"Upload failed for {file_name}: {str(e)}")
            return False

    async def list_files(self):
        logger.info("Listing files in directory...")
        file_names = []
        async with FileSystemClient(
                account_url=f"{self.account_url}?{self.sas_token}",
                file_system_name=self.file_system_name,
                credential=None,
        ) as file_system_client:
            async for path in file_system_client.get_paths(self.directory_name):
                file_name = path.name.split("/")[-1]
                async with self.lock:
                    file_names.append(file_name)
        logger.info(f"Found {len(file_names)} files")
        return file_names

    async def download_file_content(self, file_name) -> bytes:
        """Download file content and return as bytes instead of saving to disk"""
        try:
            async with DataLakeDirectoryClient(
                    account_url=f"{self.account_url}?{self.sas_token}",
                    file_system_name=self.file_system_name,
                    directory_name=self.directory_name,
                    credential=None,
            ) as directory_client:
                file_client = directory_client.get_file_client(file_name)
                stream = await file_client.download_file()
                data = await stream.readall()
                logger.info(f"Downloaded file content: {file_name}")
                return data
        except Exception as e:
            logger.error(f"Download failed for {file_name}: {str(e)}")
            raise


async def initialize_job():
    logger.info("Initializing job...")
    url = "https://api.sarvam.ai/speech-to-text-translate/job/init"
    headers = {"API-Subscription-Key": API_SUBSCRIPTION_KEY}
    response = requests.post(url, headers=headers)

    logger.info(f"Initialize Job - Status Code: {response.status_code}")

    if response.status_code == 202:
        return response.json()
    return None


async def check_job_status(job_id):
    logger.info(f"Checking status for job: {job_id}")
    url = f"https://api.sarvam.ai/speech-to-text-translate/job/{job_id}/status"
    headers = {"API-Subscription-Key": API_SUBSCRIPTION_KEY}
    response = requests.get(url, headers=headers)

    logger.info(f"Job Status - Status Code: {response.status_code}")

    if response.status_code == 200:
        return response.json()
    return None


async def start_job(job_id):
    logger.info(f"Starting job: {job_id}")
    url = "https://api.sarvam.ai/speech-to-text-translate/job"
    headers = {
        "API-Subscription-Key": API_SUBSCRIPTION_KEY,
        "Content-Type": "application/json",
    }
    data = {"job_id": job_id, "job_parameters": {"with_diarization": True}}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    logger.info(f"Start Job - Status Code: {response.status_code}")

    if response.status_code == 200:
        return response.json()
    return None


@app.post("/transcribe", response_model=Dict[str, Any])
async def transcribe_audio(files: List[UploadFile] = File(...)):
    """
    Main endpoint for speech-to-text processing
    Returns transcription results as HTTP response instead of saving files
    """
    try:
        logger.info(f"Starting transcription for {len(files)} files")

        # Step 1: Initialize the job
        job_info = await initialize_job()
        if not job_info:
            raise HTTPException(status_code=500, detail="Job initialization failed")

        job_id = job_info["job_id"]
        input_storage_path = job_info["input_storage_path"]
        output_storage_path = job_info["output_storage_path"]

        # Step 2: Save uploaded files temporarily and upload to storage
        temp_files = []
        try:
            for file in files:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}")
                content = await file.read()
                temp_file.write(content)
                temp_file.close()
                temp_files.append(temp_file.name)

            # Upload to Azure storage
            client = SarvamClient(input_storage_path)
            uploaded_count = await client.upload_files(temp_files)

            if uploaded_count == 0:
                raise HTTPException(status_code=500, detail="Failed to upload files")

            # Step 3: Start the job
            job_start_response = await start_job(job_id)
            if not job_start_response:
                raise HTTPException(status_code=500, detail="Failed to start job")

            # Step 4: Monitor job status
            logger.info("Monitoring job status...")
            max_attempts = 60  # 10 minutes with 10-second intervals
            attempt = 1

            while attempt <= max_attempts:
                job_status = await check_job_status(job_id)
                if not job_status:
                    raise HTTPException(status_code=500, detail="Failed to get job status")

                status = job_status["job_state"]
                if status == "Completed":
                    logger.info("Job completed successfully!")
                    break
                elif status == "Failed":
                    raise HTTPException(status_code=500, detail="Job processing failed")
                else:
                    logger.info(f"Current status: {status} (attempt {attempt})")
                    await asyncio.sleep(10)
                attempt += 1

            if attempt > max_attempts:
                raise HTTPException(status_code=408, detail="Job processing timeout")

            # Step 5: Download and return results
            logger.info("Downloading transcription results...")
            client.update_url(output_storage_path)

            result_files = await client.list_files()
            if not result_files:
                raise HTTPException(status_code=404, detail="No transcription results found")

            # Get job details for file mapping
            final_job_status = await check_job_status(job_id)
            file_mapping = {
                detail["file_id"]: detail["file_name"]
                for detail in final_job_status.get("job_details", [])
            }

            # Download transcription results and return as JSON
            transcription_results = []
            for result_file in result_files:
                try:
                    file_content = await client.download_file_content(result_file)
                    transcription_data = json.loads(file_content.decode('utf-8'))

                    # Map back to original filename
                    file_id = result_file.split(".")[0]
                    original_filename = file_mapping.get(file_id, f"file_{file_id}")

                    transcription_results.append({
                        "original_filename": original_filename,
                        "transcription": transcription_data
                    })
                except Exception as e:
                    logger.error(f"Error processing result file {result_file}: {e}")
                    continue

            return {
                "job_id": job_id,
                "status": "completed",
                "results": transcription_results,
                "files_processed": len(transcription_results)
            }

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/transcribe/urls", response_model=Dict[str, Any])
async def transcribe_audio_from_urls(payload: Dict[str, List[str]] = Body(...)):
    """
    Accepts list of audio file URLs instead of direct uploads.
    Downloads, uploads to Azure, triggers job, returns transcription.
    """
    file_urls = payload.get("file_urls", [])
    if not file_urls:
        raise HTTPException(status_code=400, detail="No file URLs provided")

    logger.info(f"Starting transcription from {len(file_urls)} URLs")

    # Step 1: Initialize the job
    job_info = await initialize_job()
    if not job_info:
        raise HTTPException(status_code=500, detail="Job initialization failed")

    job_id = job_info["job_id"]
    input_storage_path = job_info["input_storage_path"]
    output_storage_path = job_info["output_storage_path"]

    # Step 2: Download files to temp and upload
    temp_files = []
    try:
        for url in file_urls:
            try:
                logger.info(f"Downloading file from: {url}")
                resp = requests.get(url, stream=True)
                if resp.status_code != 200:
                    raise Exception(f"Failed to download: {url}")

                suffix = os.path.splitext(urlparse(url).path)[-1] or ".wav"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                for chunk in resp.iter_content(8192):
                    temp_file.write(chunk)
                temp_file.close()
                temp_files.append(temp_file.name)
            except Exception as e:
                logger.error(f"Download failed for {url}: {e}")

        if not temp_files:
            raise HTTPException(status_code=500, detail="Failed to download any files")

        # Upload to Azure
        client = SarvamClient(input_storage_path)
        uploaded_count = await client.upload_files(temp_files)

        if uploaded_count == 0:
            raise HTTPException(status_code=500, detail="File upload failed")

        # Start the job
        job_start_response = await start_job(job_id)
        if not job_start_response:
            raise HTTPException(status_code=500, detail="Failed to start job")

        # Poll for status (same as before)
        logger.info("Monitoring job status...")
        attempt = 1
        max_attempts = 60
        while attempt <= max_attempts:
            job_status = await check_job_status(job_id)
            if not job_status:
                raise HTTPException(status_code=500, detail="Failed to get job status")

            status = job_status["job_state"]
            if status == "Completed":
                break
            elif status == "Failed":
                raise HTTPException(status_code=500, detail="Job failed")
            else:
                logger.info(f"Waiting... {status} (attempt {attempt})")
                await asyncio.sleep(10)
                attempt += 1

        if attempt > max_attempts:
            raise HTTPException(status_code=408, detail="Job timeout")

        # Step 5: Download transcription results
        client.update_url(output_storage_path)
        result_files = await client.list_files()
        if not result_files:
            raise HTTPException(status_code=404, detail="No results found")

        final_job_status = await check_job_status(job_id)
        file_mapping = {
            detail["file_id"]: detail["file_name"]
            for detail in final_job_status.get("job_details", [])
        }

        transcription_results = []
        for result_file in result_files:
            try:
                file_content = await client.download_file_content(result_file)
                transcription_data = json.loads(file_content.decode('utf-8'))
                file_id = result_file.split(".")[0]
                original_filename = file_mapping.get(file_id, f"file_{file_id}")

                transcription_results.append({
                    "original_filename": original_filename,
                    "transcription": transcription_data
                })
            except Exception as e:
                logger.error(f"Failed to process result {result_file}: {e}")

        return {
            "job_id": job_id,
            "status": "completed",
            "results": transcription_results,
            "files_processed": len(transcription_results)
        }

    finally:
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "speech-to-text-api"}


@app.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Get status of a specific job"""
    try:
        job_status = await check_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        return job_status
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )