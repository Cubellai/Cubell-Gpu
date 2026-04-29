import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from worker.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedInput:
    local_path: Path
    result_path: Path
    result_reference: str
    uses_r2: bool


class JobStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prepare_input(
        self,
        *,
        original_reference: str,
        target_language: str,
        job_work_dir: Path,
    ) -> PreparedInput:
        original_path = Path(original_reference)
        if original_path.is_file():
            return self._local_input(original_path, target_language)

        if original_path.is_absolute():
            raise FileNotFoundError(f"Original video path does not exist: {original_path}")

        if self._looks_like_r2_key(original_reference):
            local_path = self._download_r2_input(original_reference, job_work_dir)
            return self._r2_input(original_reference, local_path, target_language)

        relative_path = Path.cwd() / original_path
        if relative_path.is_file():
            return self._local_input(relative_path, target_language)

        if self.is_r2_configured:
            local_path = self._download_r2_input(original_reference, job_work_dir)
            return self._r2_input(original_reference, local_path, target_language)

        raise FileNotFoundError(
            "Original video reference is not a local file and R2 is not configured: "
            f"{original_reference}"
        )

    def publish_result(self, prepared_input: PreparedInput) -> str:
        if not prepared_input.uses_r2:
            return prepared_input.result_reference

        self._upload_r2_result(prepared_input.result_path, prepared_input.result_reference)
        return prepared_input.result_reference

    @property
    def is_r2_configured(self) -> bool:
        return all(
            (
                self.settings.r2_bucket_name,
                self.settings.r2_access_key_id,
                self.settings.r2_secret_access_key,
                self.settings.r2_endpoint_url,
            )
        )

    def _local_input(self, local_path: Path, target_language: str) -> PreparedInput:
        result_path = self.settings.result_dir / self._result_filename(local_path.stem, target_language)
        return PreparedInput(
            local_path=local_path,
            result_path=result_path,
            result_reference=str(result_path),
            uses_r2=False,
        )

    def _r2_input(
        self,
        original_key: str,
        local_path: Path,
        target_language: str,
    ) -> PreparedInput:
        original_stem = PurePosixPath(original_key).stem
        result_key = f"results/{self._result_filename(original_stem, target_language)}"
        result_path = self.settings.result_dir / PurePosixPath(result_key).name
        return PreparedInput(
            local_path=local_path,
            result_path=result_path,
            result_reference=result_key,
            uses_r2=True,
        )

    def _download_r2_input(self, object_key: str, job_work_dir: Path) -> Path:
        self._require_r2_config()
        local_path = job_work_dir / "input" / PurePosixPath(object_key).name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading R2 input %s to %s", object_key, local_path)
        self._r2_client().download_file(
            self.settings.r2_bucket_name,
            object_key,
            str(local_path),
        )
        return local_path

    def _upload_r2_result(self, local_path: Path, object_key: str) -> None:
        if not local_path.is_file():
            raise FileNotFoundError(f"Cannot upload missing result file to R2: {local_path}")

        self._require_r2_config()
        logger.info("Uploading dubbed result %s to R2 key %s", local_path, object_key)
        self._r2_client().upload_file(
            str(local_path),
            self.settings.r2_bucket_name,
            object_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )

    def _require_r2_config(self) -> None:
        missing = [
            env_name
            for env_name, value in (
                ("R2_BUCKET_NAME", self.settings.r2_bucket_name),
                ("R2_ACCESS_KEY_ID", self.settings.r2_access_key_id),
                ("R2_SECRET_ACCESS_KEY", self.settings.r2_secret_access_key),
                ("R2_ENDPOINT_URL", self.settings.r2_endpoint_url),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"R2 storage is missing required settings: {', '.join(missing)}")

    def _r2_client(self):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required when processing R2-backed jobs.") from exc

        return boto3.client(
            "s3",
            endpoint_url=self.settings.r2_endpoint_url,
            aws_access_key_id=self.settings.r2_access_key_id,
            aws_secret_access_key=self.settings.r2_secret_access_key,
        )

    @staticmethod
    def _looks_like_r2_key(reference: str) -> bool:
        return reference.startswith(("uploads/", "results/"))

    @staticmethod
    def _result_filename(original_stem: str, target_language: str) -> str:
        language_slug = target_language.lower().replace(" ", "-")
        return f"{original_stem}-{language_slug}-dubbed.mp4"
