"""Client wrapper for MinerU document parsing service."""
from __future__ import annotations

import json
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import requests

from utils.logger import get_logger

LOGGER = get_logger(__name__)


class MineruError(Exception):
    """Raised when MinerU API calls fail."""


class MineruClient:
    def __init__(self, api_key: str, base_url: str = "https://mineru.net/api/v4") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            LOGGER.warning("MinerU API key is not configured; PDF parsing will be disabled.")

    # API endpoints
    TASK_ENDPOINT = "/extract/task"
    TASK_RESULT_ENDPOINT = "/extract/task/{task_id}"
    BATCH_UPLOAD_ENDPOINT = "/file-urls/batch"
    BATCH_RESULT_ENDPOINT = "/extract-results/batch/{batch_id}"

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def create_task(
        self,
        url: str,
        is_ocr: bool = True,
        enable_formula: bool = False,
        enable_table: bool = True,
        language: str = "en",
    ) -> str:
        if not self.api_key:
            raise MineruError("MinerU API key is required to create parsing tasks.")
        payload = {
            "url": url,
            "is_ocr": is_ocr,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
            "language": language,
        }
        response = requests.post(
            self.base_url + self.TASK_ENDPOINT,
            headers=self._headers(),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            raise MineruError(f"Failed to create task: {data}")
        task_id = data.get("data", {}).get("task_id")
        if not task_id:
            raise MineruError("Task ID missing in MinerU response")
        return task_id

    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise MineruError("MinerU API key is required to fetch parsing results.")
        response = requests.get(
            self.base_url + self.TASK_RESULT_ENDPOINT.format(task_id=task_id),
            headers=self._headers(),
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_completion(self, task_id: str, timeout: int = 600, poll_interval: int = 5) -> Dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self.get_task_result(task_id)
            data = result.get("data", {})
            state = data.get("state")
            if state == "done":
                return data
            if state == "failed":
                raise MineruError(data.get("err_msg", "MinerU task failed"))
            time.sleep(poll_interval)
        raise MineruError("MinerU task timed out")

    def download_result(self, zip_url: str, destination: Optional[Path] = None) -> Path:
        if destination is None:
            destination = Path(tempfile.mkdtemp()) / "mineru_result.zip"
        response = requests.get(zip_url, timeout=120)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return destination

    @staticmethod
    def extract_assets_from_zip(
        self, zip_path: Path, output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        if output_dir is None:
            output_dir = zip_path.parent
        raw_json: Optional[Any] = None
        markdown: Optional[str] = None
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                archive.extract(member, output_dir)
                target_path = output_dir / member
                lower_name = member.lower()
                if lower_name.endswith(".json"):
                    with target_path.open("r", encoding="utf-8") as fp:
                        try:
                            raw_json = json.load(fp)
                        except json.JSONDecodeError as exc:
                            raise MineruError(
                                f"Failed to decode MinerU JSON payload: {exc}"
                            ) from exc
                elif lower_name.endswith(".md"):
                    markdown = target_path.read_text(encoding="utf-8")
        if raw_json is None:
            raise MineruError("JSON file not found in MinerU zip archive")
        return {"raw": raw_json, "markdown": markdown}

    def parse_local_json(self, file_path: Path) -> Dict[str, Any]:
        with file_path.open("r", encoding="utf-8") as fp:
            return {"raw": json.load(fp), "markdown": None}

    def parse(self, file_path: Path, **options: Any) -> Dict[str, Any]:
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            LOGGER.info("Using provided JSON file for parsing.")
            return self.parse_local_json(file_path)
        if suffix == ".pdf":
            return self.parse_pdf(file_path, **options)
        raise MineruError(f"Unsupported file type: {file_path.suffix}")

    def parse_pdf(
        self,
        file_path: Path,
        is_ocr: bool = True,
        enable_formula: bool = False,
        enable_table: bool = True,
        language: str = "en",
        extra_formats: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise MineruError("MinerU API key is required for PDF parsing.")
        payload: Dict[str, Any] = {
            "enable_formula": enable_formula,
            "enable_table": enable_table,
            "language": language,
            "files": [
                {
                    "name": file_path.name,
                    "is_ocr": is_ocr,
                }
            ],
        }
        # According to MinerU documentation, markdown and json artefacts are included by default.
        # Only pass the "extra_formats" field if explicitly requested in order to avoid
        # triggering validation errors on deployments that do not recognise it.
        if extra_formats:
            payload["extra_formats"] = list(extra_formats)
        response = requests.post(
            self.base_url + self.BATCH_UPLOAD_ENDPOINT,
            headers=self._headers(),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            raise MineruError(f"Failed to request upload URL: {data}")
        batch_id = data.get("data", {}).get("batch_id")
        file_urls = data.get("data", {}).get("file_urls", [])
        if not batch_id or not file_urls:
            raise MineruError("MinerU batch upload response missing data")
        upload_url = file_urls[0]
        with file_path.open("rb") as fp:
            upload_response = requests.put(upload_url, data=fp.read(), timeout=300)
            upload_response.raise_for_status()

        LOGGER.info("Uploaded file to MinerU, waiting for extraction (%s)", batch_id)
        result = self._wait_batch_completion(batch_id, file_path.name)
        zip_url = result.get("full_zip_url")
        if not zip_url:
            raise MineruError("MinerU did not return a zip result URL")
        zip_path = self.download_result(zip_url)
        return self.extract_assets_from_zip(zip_path)

    def _wait_batch_completion(self, batch_id: str, file_name: str, timeout: int = 900, poll_interval: int = 5) -> Dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            response = requests.get(
                self.base_url + self.BATCH_RESULT_ENDPOINT.format(batch_id=batch_id),
                headers=self._headers(),
                timeout=60,
            )
            response.raise_for_status()
            data = response.json().get("data", {})
            results = data.get("extract_result", [])
            for item in results:
                if item.get("file_name") == file_name:
                    state = item.get("state")
                    if state == "done":
                        LOGGER.debug("MinerU batch %s completed", batch_id)
                        return item
                    if state == "failed":
                        raise MineruError(item.get("err_msg", "MinerU batch task failed"))
                    LOGGER.debug(
                        "MinerU batch %s in state '%s' (extracted %s/%s)",
                        batch_id,
                        state,
                        item.get("extract_progress", {}).get("extracted_pages"),
                        item.get("extract_progress", {}).get("total_pages"),
                    )
            time.sleep(poll_interval)
        raise MineruError("MinerU batch task timed out")
