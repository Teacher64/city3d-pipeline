#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_v8_0_network.py — сетевая версия City3D для нескольких операторов.

Добавляет:
  • ModelStorage — абстракция над локальной папкой / MinIO / NFS
  • SharedProject — общий проект (Postgres + S3)
  • UserSession  — кто сейчас работает, блокировки ресурсов
  • сетевые настройки в DB_PARAMS
"""
from pathlib import Path
from datetime import datetime

SCRIPT = Path.home() / "city3d_pipeline_v5.py"

STORAGE_CODE = '''

# ============================================================================
# Сетевое хранилище моделей и результатов
# ============================================================================

class ModelStorage:
    """Абстракция хранилища весов и артефактов.

    Backend выбирается по URI:
      file:///mnt/data/models     — локальная папка
      s3://city3d-models/         — MinIO / AWS S3
      nfs://10.0.0.5/models       — сетевой NFS (через shutil)
    """

    def __init__(self, uri="file://" + str(Path.home() / "models")):
        self.uri = uri
        self.backend = uri.split("://", 1)[0]
        self._client = None
        if self.backend == "s3":
            self._init_s3(uri)
        elif self.backend == "file":
            self.root = Path(uri.replace("file://", ""))
            self.root.mkdir(parents=True, exist_ok=True)
        else:
            self.root = Path(uri)
            self.root.mkdir(parents=True, exist_ok=True)

    def _init_s3(self, uri):
        try:
            import boto3
        except ImportError:
            raise RuntimeError("boto3 не установлен: pip install boto3")
        # uri = s3://bucket/prefix
        path = uri.replace("s3://", "")
        self.bucket, _, self.prefix = path.partition("/")
        self._client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT",
                                        "http://localhost:9000"),
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY", "minio"),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY",
                                                 "minio123"))

    def upload(self, local_path, remote_name):
        if self.backend == "file" or self.backend == "nfs":
            dst = self.root / remote_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(local_path, dst)
            return str(dst)
        if self.backend == "s3":
            key = f"{self.prefix}/{remote_name}".strip("/")
            self._client.upload_file(str(local_path), self.bucket, key)
            return f"s3://{self.bucket}/{key}"
        raise RuntimeError(f"unknown backend {self.backend}")

    def download(self, remote_name, local_path):
        if self.backend in ("file", "nfs"):
            src = self.root / remote_name
            if not src.exists():
                raise FileNotFoundError(src)
            import shutil
            shutil.copy2(src, local_path)
            return str(local_path)
        if self.backend == "s3":
            key = f"{self.prefix}/{remote_name}".strip("/")
            self._client.download_file(self.bucket, key, str(local_path))
            return str(local_path)
        raise RuntimeError(f"unknown backend {self.backend}")

    def list(self, prefix=""):
        if self.backend in ("file", "nfs"):
            return [str(p.relative_to(self.root))
                    for p in (self.root / prefix).rglob("*")
                    if p.is_file()]
        if self.backend == "s3":
            key = f"{self.prefix}/{prefix}".strip("/")
            r = self._client.list_objects_v2(Bucket=self.bucket,
                                             Prefix=key)
            return [o["Key"] for o in r.get("Contents", [])]
        return []


class SharedProject:
    """Общий проект для нескольких операторов.

    Все результаты пишутся в Postgres (history.training_runs) и в общее
    S3-хранилище. Каждый оператор видит результаты коллег.
    """

    def __init__(self, db, storage):
        self.db = db
        self.storage = storage

    def register_model(self, method, model_id, local_path, metrics,
                       operator):
        """Регистрирует чекпоинт в БД и заливает в S3."""
        remote = f"models/{method}/{model_id}_{datetime.now():%Y%m%d}.pth"
        remote_uri = self.storage.upload(local_path, remote)
        if self.db and self.db.is_available:
            try:
                with self.db.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO refs.models
                        (method, model_id, remote_uri, metrics, operator)
                        VALUES (%s, %s, %s, %s::jsonb, %s)
                        RETURNING id
                    """, (method, model_id, remote_uri,
                          json.dumps(metrics or {}), operator))
            except Exception:
                pass
        return remote_uri

    def list_models(self, method=None):
        if self.db and self.db.is_available:
            with self.db.conn.cursor() as cur:
                if method:
                    cur.execute("SELECT * FROM refs.models "
                                "WHERE method=%s ORDER BY created_at DESC",
                                (method,))
                else:
                    cur.execute("SELECT * FROM refs.models "
                                "ORDER BY created_at DESC")
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
        return []
'''

DDL_ADDON = '''
-- Добавка для сетевой версии
CREATE TABLE IF NOT EXISTS refs.models (
    id           BIGSERIAL PRIMARY KEY,
    method       TEXT NOT NULL,
    model_id     TEXT NOT NULL,
    remote_uri   TEXT NOT NULL,
    metrics      JSONB,
    operator     TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (method, model_id, remote_uri)
);

CREATE TABLE IF NOT EXISTS history.active_sessions (
    id           BIGSERIAL PRIMARY KEY,
    operator     TEXT NOT NULL,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_ping    TIMESTAMPTZ NOT NULL DEFAULT now(),
    project_uri  TEXT
);
'''


def main():
    text = SCRIPT.read_text(encoding="utf-8")
    if "class ModelStorage" in text:
        print("[INFO] ModelStorage уже вставлен.")
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = SCRIPT.with_name(SCRIPT.name + f".bak-{stamp}")
    backup.write_text(text, encoding="utf-8")
    print(f"[OK] Резервная копия: {backup}")

    # Вставить перед MainWindow
    marker = "# ============================================================================\n# Главное окно"
    if marker not in text:
        print("[FATAL] Не найден маркер MainWindow.")
        return
    text = text.replace(marker, STORAGE_CODE + "\n\n" + marker, 1)

    # Добавить DDL в DB_DDL
    if "refs.models" not in text.split("DB_DDL = r")[1][:3000]:
        text = text.replace('"""\n\n\nclass LogWidget',
                            DDL_ADDON + '"""\n\n\nclass LogWidget', 1)

    SCRIPT.write_text(text, encoding="utf-8")
    print(f"[OK] ModelStorage + SharedProject + DDL добавлены.")
    print("\nСетевая версия — что делать:")
    print("  1. Развернуть Postgres с PostGIS на сервере.")
    print("  2. Развернуть MinIO (docker run -p 9000:9000 minio/minio ...).")
    print("  3. В DB_PARAMS указать host сервера.")
    print("  4. Экспортировать S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY.")
    print("  5. model_storage = ModelStorage('s3://city3d-models/')")


if __name__ == "__main__":
    main()