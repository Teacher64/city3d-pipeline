#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
City3D PointCloud Pipeline v7.3.1 — структура v4 + M1-M6 + PostgreSQL/PostGIS.

v7.3.1 (fix):
    • CRSPickerDialog: рабочий выбор EPSG из локального crslist.json.
    • EPSGCatalog: загрузка справочника EPSG из JSON (crslist.json).
    • CRSPickerDialog: автодополнение, всплывающие подсказки, поиск по
      коду / названию / территории / проекции.
    • main(): QTWEBENGINE_CHROMIUM_FLAGS → --disable-gpu --no-sandbox —
      устраняет EGL_BAD_CONTEXT на Wayland.
    • Mesh3DViewer: вырожденные треугольники отбрасываются до GLMeshItem.
    • main(): qRegisterMetaType('QTextCursor') — тишина в логах Qt.
"""

import sys
import os
import json
import time
import hashlib
import subprocess
import shutil
import random
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

# PATCHED_PARALLEL_V1 + FIX_M5_V2_APPLIED
import multiprocessing as _mp
import os as _os
from concurrent.futures import (
    ThreadPoolExecutor as _TPE,
    as_completed as _as_completed,
)

_N_CPU = max(1, _mp.cpu_count())
# Пользователь может ограничить число воркеров через env:
#   export CITY3D_WORKERS=4   # тихий режим
try:
    _N_THREADS = max(1, int(_os.environ.get(
        "CITY3D_WORKERS", str(max(1, _N_CPU - 1)))))
except Exception:
    _N_THREADS = max(1, _N_CPU - 1)
print(f"[PARALLEL v2] CPU: {_N_CPU}, "
      f"воркеров: {_N_THREADS} "
      f"(переопределить: CITY3D_WORKERS=N)")

try:
    import open3d as o3d
except ImportError:
    o3d = None

# Unified resolver: importing the GUI does not create user directories.
import json as _json_top
from city3d_backend.settings import read_settings, resolve_paths, save_settings
_APP_SETTINGS = read_settings()
_PATHS_CFG = _APP_SETTINGS.get("paths", {}) or {}
_RESOLVED_PATHS = resolve_paths(_APP_SETTINGS, Path(__file__).parent)
PROJECT_ROOT = _RESOLVED_PATHS["project_root"]
WORKSPACE = _RESOLVED_PATHS["workspace"]
PROJECTS_DIR = _RESOLVED_PATHS["projects_dir"]
DATASETS_DIR = _RESOLVED_PATHS["datasets_dir"]
TRAINING_OUTPUT_DIR = _RESOLVED_PATHS["training_output_dir"]
REFERENCE_DB_PATH = _RESOLVED_PATHS["reference_db"]
EPSG_CATALOG_PATH = _RESOLVED_PATHS["epsg_catalog"]
SLICES_DIR = _RESOLVED_PATHS["slices_dir"]
os.environ.setdefault("HF_HOME", str(_RESOLVED_PATHS["hf_cache"]))

MODELS = {
    "Cloud2BIM": PROJECTS_DIR / "Cloud2BIM",
    "RoomFormer": PROJECTS_DIR / "RoomFormer",
    "SpatialLM": PROJECTS_DIR / "SpatialLM",
    "CADReasoner": PROJECTS_DIR / "CADReasoner",
    "3d_cad_pointcloud": PROJECTS_DIR / "3d_cad_pointcloud",
}

# PATCHED_SETTINGS_TAB_V2: DB_PARAMS from settings.json
_DB_CFG = _APP_SETTINGS.get("database", {}) or {}

DB_PARAMS = {
    "dbname":   _DB_CFG.get("dbname",   "city3d"),
    "host":     _DB_CFG.get("host",     "localhost"),
    "port":     str(_DB_CFG.get("port", "5432")),
    "user":     _DB_CFG.get("user",     "postgres"),
    "password": _DB_CFG.get("password", ""),
    "schema_refs":    _DB_CFG.get("schema_refs",    "refs"),
    "schema_history": _DB_CFG.get("schema_history", "history"),
}

_DB_EXTERNAL = _APP_SETTINGS.get("database_config_file") or os.environ.get("CITY3D_DB_CONFIG")
if not _DB_EXTERNAL:
    _candidate=Path.home()/"Documents"/"подключение к базе данных.json"
    if _candidate.is_file():_DB_EXTERNAL=str(_candidate)
if _DB_EXTERNAL:
    from city3d_backend.database import load_connection_file
    try:
        _external_params=load_connection_file(_DB_EXTERNAL)
        for _key in ("host","port","user","password","service","sslmode","dbname"):
            if _key in _external_params and _key != "dbname":DB_PARAMS[_key]=_external_params[_key]
        if "password" not in _external_params and "password" not in _DB_CFG:DB_PARAMS.pop("password",None)
    except (ValueError,OSError,KeyError):
        print("[DB] Конфигурация подключения требует проверки формата; значения не изменены")

SUPPORTED_INPUT_FORMATS = [
    ".las", ".laz", ".ply", ".pcd", ".xyz", ".e57", ".bin", ".txt"
]

PREVIEW_MESH_EXTS = (".obj", ".stl", ".usda", ".usd", ".usdc")
PREVIEW_CLOUD_EXTS = (".ply", ".pcd", ".las", ".laz", ".xyz")
PREVIEW_MAP_EXTS = (".geojson",)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar, QGroupBox,
    QFormLayout, QComboBox, QCheckBox, QFileDialog, QMessageBox,
    QDoubleSpinBox, QSpinBox, QSplitter, QListWidget, QListWidgetItem,
    QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QStackedWidget, QDialog, QDialogButtonBox, QCompleter, QTableView,
    QAbstractItemView, QStyledItemDelegate, QStyle,
    QSizePolicy
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QStringListModel,
    QSortFilterProxyModel, QModelIndex
)

# --- fix: qRegisterMetaType отсутствует в системном PyQt5 Debian/Ubuntu ---
try:
    from PyQt5.QtCore import qRegisterMetaType  # type: ignore
except ImportError:
    def qRegisterMetaType(*args, **kwargs):
        """No-op fallback.

        Нужен только для регистрации кастомных метатипов в очереди
        сигналов между потоками. В данном приложении все сигналы —
        pyqtSignal(str, str), поэтому регистрация не требуется.
        """
        return None
# --- /fix ---
from PyQt5.QtGui import QFont, QTextCursor, QStandardItemModel, QStandardItem

try:
    if os.environ.get("CITY3D_DISABLE_WEBENGINE") == "1":
        raise ImportError("WebEngine disabled by environment")
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    QWebEngineView = None
    HAS_WEBENGINE = False

import pyqtgraph.opengl as gl
import pyqtgraph as pg

try:
    import psycopg
    HAS_PSYCOPG = True
except ImportError:
    psycopg = None
    HAS_PSYCOPG = False


DB_DDL = r"""
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS refs;
CREATE SCHEMA IF NOT EXISTS history;

CREATE TABLE IF NOT EXISTS refs.scanning_types (
    code        TEXT PRIMARY KEY,
    name_ru     TEXT NOT NULL,
    description TEXT
);

INSERT INTO refs.scanning_types (code, name_ru, description) VALUES
 ('ALS_manned', 'Воздушное (пилотируемое судно)', 'Плотность 4–16 т/м², охват 500×500 м'),
 ('ALS_uav',    'Воздушное (БПЛА)',               'Плотность 50–500 т/м², охват 200×200 м'),
 ('MLS',        'Мобильное (авто, SLAM)',         'Плотность 500–2000 т/м², полоса 50×200 м'),
 ('TLS',        'Наземное (штатив)',              'Плотность 1000–50000 т/м², объект 20×20 м'),
 ('handheld',   'Переносное (ручное)',            'Плотность 5000–100000 т/м², объект 5×5 м'),
 ('stationary', 'Стационарное (промышленное)',    'Плотность 10000+ т/м², сцена 2×2 м'),
 ('unknown',    'Неизвестно / не указано',        'Параметры подбираются вручную')
ON CONFLICT (code) DO NOTHING;

CREATE TABLE IF NOT EXISTS refs.epsg_codes (
    code        INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT,
    area        TEXT,
    country     TEXT,
    zone        INTEGER,
    is_active   BOOLEAN DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS idx_epsg_name
    ON refs.epsg_codes USING gin (to_tsvector('russian', name));

CREATE TABLE IF NOT EXISTS refs.region_crs_history (
    id             BIGSERIAL PRIMARY KEY,
    directory      TEXT NOT NULL,
    crs_epsg       INTEGER,
    n_observations INTEGER DEFAULT 1,
    last_seen      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (directory, crs_epsg)
);

CREATE TABLE IF NOT EXISTS refs.references_clouds (
    id            BIGSERIAL PRIMARY KEY,
    name          TEXT NOT NULL,
    path          TEXT NOT NULL,
    format        TEXT,
    n_points      BIGINT,
    scanning_type TEXT,
    crs_epsg      INTEGER,
    bbox          box3d,
    bbox_4326     geometry(Polygon, 4326),
    classes       JSONB,
    description   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_refs_clouds_bbox
    ON refs.references_clouds USING GIST (bbox_4326);

CREATE TABLE IF NOT EXISTS refs.references_masks (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    layer_type  TEXT,
    color_code  INTEGER,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS refs.references_dem (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    format      TEXT,
    resolution  REAL,
    crs_epsg    INTEGER,
    footprint   geometry(Polygon, 4326),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS refs.references_gis (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    format      TEXT,
    n_features  INTEGER,
    crs_epsg    INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS history.training_runs (
    id              BIGSERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    duration_sec    REAL,
    method          TEXT NOT NULL,
    scanning_type   TEXT,
    crs_epsg        INTEGER,
    n_points        BIGINT,
    bbox            box3d,
    bbox_4326       geometry(Polygon, 4326),
    source_path     TEXT,
    dataset_root    TEXT,
    hyperparams     JSONB,
    metrics         JSONB,
    output_dir      TEXT,
    status          TEXT NOT NULL DEFAULT 'running'
                    CHECK (status IN ('running','success','failed','partial')),
    error_message   TEXT,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_method_type
    ON history.training_runs (method, scanning_type);
CREATE INDEX IF NOT EXISTS idx_runs_started
    ON history.training_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_bbox_gist
    ON history.training_runs USING GIST (bbox_4326);
CREATE INDEX IF NOT EXISTS idx_runs_metrics_gin
    ON history.training_runs USING GIN (metrics);

CREATE TABLE IF NOT EXISTS history.training_artifacts (
    id          BIGSERIAL PRIMARY KEY,
    run_id      BIGINT NOT NULL REFERENCES history.training_runs(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    path        TEXT NOT NULL,
    size_bytes  BIGINT,
    sha256      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON history.training_artifacts (run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_sha ON history.training_artifacts (sha256);

CREATE TABLE IF NOT EXISTS history.scanning_presets (
    id              BIGSERIAL PRIMARY KEY,
    method          TEXT NOT NULL,
    scanning_type   TEXT,
    crs_epsg        INTEGER,
    hyperparams     JSONB NOT NULL,
    source_run_id   BIGINT,
    quality_score   REAL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (method, scanning_type, crs_epsg)
);

CREATE OR REPLACE VIEW history.v_best_presets AS
SELECT DISTINCT ON (method, scanning_type)
    method, scanning_type, hyperparams, source_run_id,
    quality_score, updated_at
FROM history.scanning_presets
WHERE quality_score IS NOT NULL
ORDER BY method, scanning_type, quality_score DESC, updated_at DESC;

CREATE OR REPLACE VIEW history.v_quality_summary AS
SELECT
    scanning_type, method,
    COUNT(*) AS n_runs,
    AVG((metrics->>'quality_score')::real) AS avg_quality,
    MAX((metrics->>'quality_score')::real) AS max_quality,
    AVG(duration_sec)::int AS avg_duration_sec,
    MAX(started_at) AS last_run
FROM history.training_runs
WHERE status = 'success'
GROUP BY scanning_type, method
ORDER BY scanning_type, method;

CREATE OR REPLACE FUNCTION history.find_overlapping(
    p_geom geometry, p_method TEXT DEFAULT NULL,
    p_type TEXT DEFAULT NULL, p_limit INTEGER DEFAULT 20)
RETURNS TABLE (
    id BIGINT, method TEXT, scanning_type TEXT,
    started_at TIMESTAMPTZ, overlap_area_m2 REAL)
LANGUAGE sql STABLE AS $$
    SELECT r.id, r.method, r.scanning_type, r.started_at,
           ST_Area(ST_Intersection(r.bbox_4326, p_geom)::geography)::real AS overlap_area_m2
    FROM history.training_runs r
    WHERE r.bbox_4326 IS NOT NULL
      AND r.bbox_4326 && p_geom
      AND (p_method IS NULL OR r.method = p_method)
      AND (p_type   IS NULL OR r.scanning_type = p_type)
    ORDER BY overlap_area_m2 DESC
    LIMIT p_limit;
$$;
"""


class LogWidget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Monospace", 9))
        self.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4; "
            "border: 1px solid #3c3c3c; border-radius: 4px;")

    def log(self, message, level="INFO"):
        colors = {"INFO": "#d4d4d4", "SUCCESS": "#4ec9b0",
                  "WARNING": "#dcdcaa", "ERROR": "#f44747"}
        color = colors.get(level, "#d4d4d4")
        ts = datetime.now().strftime("%H:%M:%S")
        self.append(
            f'<span style="color:#808080;">[{ts}]</span> '
            f'<span style="color:{color};">[{level}]</span> '
            f'<span style="color:#d4d4d4;">{message}</span>')
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum())


class EmitLogger:
    def __init__(self, emit_fn):
        self._emit = emit_fn

    def log(self, message, level="INFO"):
        try:
            self._emit(str(message), level)
        except Exception:
            pass


_ACTIVE_WORKERS = []


class FrozenControl:
    def __init__(self, value=None, text="", index=0, checked=False):
        self._value, self._text, self._index, self._checked = value, text, index, checked
    def value(self): return self._value
    def text(self): return self._text
    def currentText(self): return self._text
    def currentIndex(self): return self._index
    def isChecked(self): return self._checked
    @classmethod
    def capture(cls, widget):
        return cls(widget.value() if hasattr(widget, "value") else None,
                   widget.text() if hasattr(widget, "text") else (widget.currentText() if hasattr(widget, "currentText") else ""),
                   widget.currentIndex() if hasattr(widget, "currentIndex") else 0,
                   widget.isChecked() if hasattr(widget, "isChecked") else False)


class WorkerThread(QThread):
    """Фоновый воркер с прогрессом, ETA и текстовым статусом.

    Сигналы:
      progress(int)              — 0..100
      detail_signal(str)         — человекочитаемый статус с ETA
      log_signal(str, str)       — (сообщение, уровень)
      task_done(bool)            — успех/провал
    """

    progress = pyqtSignal(int)
    detail_signal = pyqtSignal(str)
    task_done = pyqtSignal(bool)
    log_signal = pyqtSignal(str, str)

    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        self._t0 = None
        self._last_pct = 0
        _ACTIVE_WORKERS.append(self)

    # --- вспомогательный progress-callback с ETA ----------------------
    def make_progress_cb(self):
        import time as _time

        def cb(pct, note=""):
            pct = max(0, min(100, int(pct)))
            if self._t0 is None:
                self._t0 = _time.time()
            elapsed = _time.time() - self._t0
            eta_str = ""
            if pct > 0:
                total_est = elapsed * 100.0 / max(pct, 1)
                remain = max(0.0, total_est - elapsed)
                m, s = divmod(int(remain), 60)
                h, m = divmod(m, 60)
                if h:
                    eta_str = f"ETA ~{h}ч {m:02d}м {s:02d}с"
                elif m:
                    eta_str = f"ETA ~{m}м {s:02d}с"
                else:
                    eta_str = f"ETA ~{s}с"
            msg = (f"{pct:3d}%  прошло {int(elapsed)}с  {eta_str}"
                   + (f"  |  {note}" if note else ""))
            self.detail_signal.emit(msg)
            self.progress.emit(pct)
            self._last_pct = pct
        return cb

    def run(self):
        try:
            progress_cb = self.make_progress_cb()
            result = self.task_func(
                self.log_signal.emit, progress_cb,
                *self.args, **self.kwargs)
            self.task_done.emit(bool(result))
        except Exception as e:
            self.log_signal.emit(f"Критическая ошибка: {e}", "ERROR")
            self.task_done.emit(False)
        finally:
            try:
                _ACTIVE_WORKERS.remove(self)
            except ValueError:
                pass



# PATCHED_CRS_UI: вертикальные системы высот (для России)
VERTICAL_CRS_LIST = [
    ("none",             "Без вертикальной СК (эллипсоид)"),
    ("EPSG:5705",        "Балтийская 1977 (BSC77)"),
    ("EPSG:5704",        "Пулково 1942 (бали. Балт.)"),
    ("EPSG:5773",        "EGM96 geoid"),
    ("EPSG:3855",        "EGM2008 geoid"),
    ("EPSG:4979",        "WGS84 3D (эллипс.)"),
    ("EPSG:5774",        "Балт. 1977 (высоты над уровнем моря)"),
    ("local",            "Локальная (условная)"),
]


# ============================================================================
# Справочник EPSG (загрузка из crslist.json)
# ============================================================================

class EPSGCatalog:
    """Справочник EPSG, загружаемый из локального JSON-файла.

    Ожидается файл crslist.json в формате PROJ-CRS-Explorer:
      https://github.com/OSGeo/PROJ-CRS-Explorer
    Каждая запись — словарь с ключами:
      auth_name, code, name, type, deprecated,
      area_of_use (список, последний элемент — текст),
      projection_method_name
    """

    def __init__(self, catalog_path=None, log_widget=None):
        self.catalog_path = (Path(catalog_path)
                             if catalog_path else EPSG_CATALOG_PATH)
        self.log = log_widget
        self.entries = []      # список словарей
        self._by_code = {}     # code -> entry
        self._load()

    def _load(self):
        if not self.catalog_path.exists():
            if self.log:
                self.log.log(
                    f"Файл справочника EPSG не найден: "
                    f"{self.catalog_path}", "WARNING")
            return
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for item in raw:
                try:
                    code = int(item.get("code", 0))
                except Exception:
                    continue
                # area_of_use может быть списком; последний элемент — текст
                area = ""
                aou = item.get("area_of_use")
                if isinstance(aou, list) and len(aou) >= 5:
                    area = aou[4]
                elif isinstance(aou, str):
                    area = aou

                entry = {
                    "code": code,
                    "auth_name": item.get("auth_name", "EPSG"),
                    "name": item.get("name", ""),
                    "type": item.get("type", ""),
                    "deprecated": bool(item.get("deprecated", False)),
                    "area": area,
                    "projection": item.get("projection_method_name", "") or "",
                }
                self.entries.append(entry)
                self._by_code[code] = entry
            if self.log:
                self.log.log(
                    f"Справочник EPSG загружен: {len(self.entries)} "
                    f"записей из {self.catalog_path.name}", "SUCCESS")
        except Exception as e:
            if self.log:
                self.log.log(
                    f"Ошибка загрузки справочника EPSG: {e}", "ERROR")

    def search(self, query, limit=500):
        """Поиск по коду, имени, территории или проекции."""
        if not query:
            return self.entries[:limit]
        q = query.strip().lower()
        result = []
        for e in self.entries:
            if (q in str(e["code"]) or
                    q in e["name"].lower() or
                    q in e["area"].lower() or
                    q in e["projection"].lower()):
                result.append(e)
            if len(result) >= limit:
                break
        return result

    def get_by_code(self, code):
        return self._by_code.get(int(code))


# ============================================================================
# Диалог выбора CRS
# ============================================================================

class CRSPickerDialog(QDialog):
    """Диалог выбора системы координат с автодополнением и подсказками."""

    def __init__(self, catalog: EPSGCatalog, parent=None,
                 title="Выбор системы координат",
                 suggested_text=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(950, 650)
        self.catalog = catalog
        self.selected_epsg = None
        self.suggested_text = suggested_text
        self._build_ui()
        self._load(initial=suggested_text)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Поле поиска с автодополнением
        top = QHBoxLayout()
        top.addWidget(QLabel("Поиск (код, название, территория):"))
        self.search = QLineEdit()
        self.search.setPlaceholderText(
            "Например: 32636, UTM zone 36N, WGS 84, Pulkovo")
        self.search.textChanged.connect(self._on_search_text_changed)
        self.search.returnPressed.connect(self._on_search)
        top.addWidget(self.search, 1)

        b_find = QPushButton("Найти")
        b_find.clicked.connect(self._on_search)
        top.addWidget(b_find)
        layout.addLayout(top)

        # Модель для автодополнения
        self.completer_model = QStringListModel()
        self.completer = QCompleter(self.completer_model, self)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.search.setCompleter(self.completer)

        # Таблица результатов
        self.table = QTableView()
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(
            ["Код", "Название", "Тип", "Территория", "Проекция"])
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.table.doubleClicked.connect(self._on_accept)
        layout.addWidget(self.table)

        # Всплывающая подсказка
        self.info_label = QLabel(
            "Наведите на строку, чтобы увидеть подробную информацию.")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "color: #dcdcaa; padding: 4px; background: #2d2d30; "
            "border-radius: 4px;")
        layout.addWidget(self.info_label)

        # Кнопки
        bb = QDialogButtonBox()
        b_apply = bb.addButton("Применить",
                               QDialogButtonBox.AcceptRole)
        b_keep = bb.addButton("Оставить EPSG:4326",
                              QDialogButtonBox.ResetRole)
        b_cancel = bb.addButton("Отмена",
                                QDialogButtonBox.RejectRole)
        b_apply.clicked.connect(self._on_accept)
        b_keep.clicked.connect(self._on_keep_default)
        b_cancel.clicked.connect(self.reject)
        layout.addWidget(bb)

    def _load(self, initial=""):
        if not initial:
            initial = self.suggested_text or "WGS 84"
        self._populate(self.catalog.search(initial, limit=500))

    def _populate(self, entries):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(
            ["Код", "Название", "Тип", "Территория", "Проекция"])
        for e in entries:
            row = [
                QStandardItem(str(e["code"])),
                QStandardItem(e["name"]),
                QStandardItem(e["type"]),
                QStandardItem(e["area"]),
                QStandardItem(e["projection"]),
            ]
            row[0].setData(e, Qt.UserRole)
            tip = (f"EPSG:{e['code']}\n{e['name']}\n"
                   f"Тип: {e['type']}\n"
                   f"Территория: {e['area']}\n"
                   f"Проекция: {e['projection']}")
            if e.get("deprecated"):
                tip += "\n⚠ УСТАРЕВШАЯ (deprecated)"
            for item in row:
                item.setToolTip(tip)
            self.model.appendRow(row)
        self._update_completer(entries)

    def _update_completer(self, entries):
        items = []
        for e in entries[:1000]:
            items.append(f"{e['code']} — {e['name']}")
        self.completer_model.setStringList(items)

    def _on_search_text_changed(self, text):
        # Автодополнение обновляем через _on_search при Enter,
        # здесь можно добавить live-фильтр (для простоты оставлено пусто)
        pass

    def _on_search(self):
        q = self.search.text().strip()
        if not q:
            self._load()
            return
        # Если введён только код
        if q.isdigit():
            entry = self.catalog.get_by_code(int(q))
            if entry:
                self._populate([entry])
                return
        self._populate(self.catalog.search(q, limit=500))

    def _current_entry(self):
        idx = self.table.currentIndex()
        if not idx.isValid():
            return None
        return self.model.item(idx.row(), 0).data(Qt.UserRole)

    def _on_accept(self):
        entry = self._current_entry()
        if not entry:
            QMessageBox.warning(self, "Ошибка", "Выберите строку.")
            return
        self.selected_epsg = entry["code"]
        self.accept()

    def _on_keep_default(self):
        self.selected_epsg = 4326
        self.accept()


# ============================================================================
# Загрузчики / детекторы / тренер / экспортёр / подготовка датасета
# ============================================================================

class PointCloudLoader:
    @staticmethod
    def load(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        ext = path.suffix.lower()
        if ext in (".ply", ".pcd"):
            return PointCloudLoader._load_open3d(path)
        elif ext in (".las", ".laz"):
            return PointCloudLoader._load_las(path)
        elif ext in (".xyz", ".txt"):
            return PointCloudLoader._load_xyz(path)
        elif ext == ".e57":
            return PointCloudLoader._load_e57(path)
        elif ext == ".bin":
            return PointCloudLoader._load_bin(path)
        else:
            raise ValueError(f"Неподдерживаемый формат: {ext}")

    @staticmethod
    def _load_open3d(path):
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        return {"points": points, "colors": colors, "intensity": None,
                "classification": None, "source": str(path),
                "format": path.suffix.lower()}

    @staticmethod
    def _load_las(path):
        import laspy
        las = laspy.read(path)
        points = np.vstack((las.x, las.y, las.z)).T
        colors = None
        if hasattr(las, "red"):
            try:
                colors = np.vstack(
                    (las.red, las.green, las.blue)).T / 65535.0
            except Exception:
                pass
        intensity = None
        if hasattr(las, "intensity"):
            intensity = np.array(las.intensity)
        classification = None
        if hasattr(las, "classification"):
            classification = np.array(las.classification)
        return {"points": points, "colors": colors, "intensity": intensity,
                "classification": classification, "source": str(path),
                "format": path.suffix.lower(), "las_data": las, "point_ids": np.arange(len(las.points))}

    @staticmethod
    def _load_xyz(path):
        data = np.loadtxt(path, delimiter=None)
        if data.shape[1] < 3:
            raise ValueError("XYZ должен содержать минимум 3 колонки")
        points = data[:, :3]
        colors = None
        if data.shape[1] >= 6:
            colors = data[:, 3:6]
            if colors.max() > 1.0:
                colors = colors / 255.0
        return {"points": points, "colors": colors, "intensity": None,
                "classification": None, "source": str(path),
                "format": ".xyz"}

    @staticmethod
    def _load_e57(path):
        try:
            import pye57
            e57 = pye57.E57(str(path))
            data = e57.read_scan(0, ignore_missing_fields=True)
            points = np.vstack(
                (data["cartesianX"], data["cartesianY"],
                 data["cartesianZ"])).T
            return {"points": points, "colors": None, "intensity": None,
                    "classification": None, "source": str(path),
                    "format": ".e57"}
        except ImportError:
            result = subprocess.run(
                ["pdal", "translate", str(path), "--stdout"],
                capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("pye57/PDAL не сработали")
            lines = result.stdout.strip().split("\n")
            data = np.array([[float(x) for x in l.split()[:3]]
                             for l in lines if l.strip()])
            return {"points": data, "colors": None, "intensity": None,
                    "classification": None, "source": str(path),
                    "format": ".e57"}

    @staticmethod
    def _load_bin(path):
        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
        return {"points": points[:, :3], "colors": None,
                "intensity": points[:, 3], "classification": None,
                "source": str(path), "format": ".bin"}

    @staticmethod
    def save(data, output_path, format=None):
        output_path = Path(output_path)
        fmt = (format or output_path.suffix).lower()
        points = data["points"]
        colors = data.get("colors")
        if fmt in (".ply", ".pcd"):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None and len(colors) == len(points):
                pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(str(output_path), pcd)
        elif fmt in (".las", ".laz"):
            import laspy
            if data.get("las_data") is not None:
                import copy
                original = data["las_data"]
                ids = np.asarray(data.get("point_ids", np.arange(len(original.points))), dtype=np.int64)
                if len(ids) != len(points):
                    raise ValueError("После отбора LAS нужны исходные point_ids; отказ от потери атрибутов")
                las = laspy.LasData(copy.deepcopy(original.header), points=original.points[ids].copy())
                las.evlrs = copy.deepcopy(original.evlrs)
                las.x, las.y, las.z = np.asarray(points).T
                if data.get("classification") is not None:
                    las.classification = data["classification"]
                las.write(str(output_path))
                return
            header = laspy.LasHeader(point_format=3, version="1.4")
            header.offsets = points.min(axis=0)
            header.scales = np.array([0.001, 0.001, 0.001])
            las = laspy.LasData(header)
            las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
            if colors is not None and len(colors) == len(points):
                las.red = (colors[:, 0] * 65535).astype(np.uint16)
                las.green = (colors[:, 1] * 65535).astype(np.uint16)
                las.blue = (colors[:, 2] * 65535).astype(np.uint16)
            if data.get("intensity") is not None:
                las.intensity = data["intensity"]
            if data.get("classification") is not None:
                las.classification = data["classification"]
            las.write(str(output_path))
        elif fmt == ".xyz":
            combined = (np.hstack([points, colors * 255])
                        if colors is not None else points)
            np.savetxt(str(output_path), combined, fmt="%.6f")
        elif fmt == ".npz":
            np.savez_compressed(str(output_path), points=points,
                                colors=(colors if colors is not None
                                        else np.array([])))
        elif fmt in (".h5", ".hdf5"):
            import h5py
            with h5py.File(str(output_path), "w") as f:
                f.create_dataset("points", data=points, compression="gzip")
                if colors is not None:
                    f.create_dataset("colors", data=colors,
                                     compression="gzip")
        else:
            raise ValueError(f"Неподдерживаемый формат: {fmt}")
        return output_path


class CRSDetector:
    def __init__(self, log_widget=None, history_db=None):
        self.log = log_widget
        self.history_db = history_db
        self.crs = None
        self.crs_source = None
        self.crs_description = None

    def detect(self, file_path):
        file_path = Path(file_path)

        if self.history_db and self.history_db.is_available:
            remembered = self.history_db.get_region_crs(
                str(file_path.parent))
            if remembered:
                from pyproj import CRS as _CRS
                try:
                    crs = _CRS.from_epsg(remembered)
                    self.crs = remembered
                    self.crs_source = "история по папке"
                    self.crs_description = crs.name
                    if self.log:
                        self.log.log(
                            f"CRS: EPSG:{remembered} "
                            f"(история по папке) — {crs.name}", "SUCCESS")
                    return {"epsg": remembered, "description": crs.name}
                except Exception:
                    pass

        methods = [
            ("VLR-запись LAS/LAZ", self._from_las_vlr),
            ("Sidecar .prj", self._from_prj),
            ("Sidecar .wkt", self._from_wkt),
            ("Sidecar .json", self._from_json_sidecar),
            ("Эвристика по диапазону", self._from_heuristic),
        ]
        for name, method in methods:
            try:
                result = method(file_path)
                if result:
                    self.crs = result["epsg"]
                    self.crs_source = name
                    self.crs_description = result["description"]
                    if self.log:
                        self.log.log(
                            f"CRS: EPSG:{self.crs} ({name}) — "
                            f"{self.crs_description}", "SUCCESS")
                    return result
            except Exception as e:
                if self.log:
                    self.log.log(f"Метод '{name}': {e}", "WARNING")

        self.crs = None
        self.crs_source = "не определён"
        self.crs_description = ""
        if self.log:
            self.log.log(
                "CRS не определён автоматически. Требуется выбрать вручную.",
                "WARNING")
        return {"epsg": None, "description": ""}

    def _from_las_vlr(self, file_path):
        if file_path.suffix.lower() not in (".las", ".laz"):
            return None
        import laspy
        from pyproj import CRS
        with laspy.open(file_path) as fh:
            header = fh.header
            for vlr in header.vlrs:
                if vlr.record_id == 2112:
                    try:
                        wkt = vlr.record_data.decode("utf-8",
                                                     errors="ignore")
                        crs = CRS.from_wkt(wkt)
                        return {"epsg": crs.to_epsg(),
                                "description": crs.name}
                    except Exception:
                        continue
            for vlr in header.vlrs:
                if vlr.record_id == 34735:
                    try:
                        data = np.frombuffer(vlr.record_data,
                                             dtype=np.uint16)
                        for i in range(4, len(data) - 3, 4):
                            if data[i] in (3072, 2048):
                                epsg = int(data[i + 3])
                                if epsg > 0:
                                    crs = CRS.from_epsg(epsg)
                                    return {"epsg": epsg,
                                            "description": crs.name}
                    except Exception:
                        continue
            if -180 <= header.x_min <= 180 and -90 <= header.y_min <= 90:
                return {"epsg": 4326,
                        "description": "WGS84 (по диапазону)"}
        return None

    def _from_prj(self, file_path):
        prj = file_path.with_suffix(".prj")
        if not prj.exists():
            return None
        from pyproj import CRS
        crs = CRS.from_wkt(prj.read_text(errors="ignore"))
        return {"epsg": crs.to_epsg(), "description": crs.name}

    def _from_wkt(self, file_path):
        wkt_path = file_path.with_suffix(".wkt")
        if not wkt_path.exists():
            return None
        from pyproj import CRS
        crs = CRS.from_wkt(wkt_path.read_text(errors="ignore"))
        return {"epsg": crs.to_epsg(), "description": crs.name}

    def _from_json_sidecar(self, file_path):
        json_path = file_path.with_suffix(".json")
        if not json_path.exists():
            return None
        from pyproj import CRS
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            return None
        for key in ("crs", "srs", "spatialReference", "coordinateSystem"):
            if key in data:
                val = data[key]
                if isinstance(val, dict):
                    if "epsg" in val:
                        crs = CRS.from_epsg(val["epsg"])
                        return {"epsg": val["epsg"],
                                "description": crs.name}
                    if "wkt" in val:
                        crs = CRS.from_wkt(val["wkt"])
                        return {"epsg": crs.to_epsg(),
                                "description": crs.name}
        return None

    def _from_heuristic(self, file_path):
        if file_path.suffix.lower() in (".las", ".laz"):
            return None
        from pyproj import CRS
        try:
            pcd = o3d.io.read_point_cloud(str(file_path))
            if len(pcd.points) == 0:
                return None
            pts = np.asarray(pcd.points)
            x_min = pts[:, 0].min()
            x_range = pts[:, 0].max() - x_min
            if -180 <= x_min <= 180 and x_range < 10:
                return {"epsg": 4326, "description": "WGS84 (эвристика)"}
            if 100000 < abs(x_min) < 10000000:
                zone = int((x_min + 180) / 6) + 1
                if 1 <= zone <= 60:
                    epsg = 32600 + zone
                    crs = CRS.from_epsg(epsg)
                    return {"epsg": epsg,
                            "description": f"{crs.name} (эвристика)"}
            return None
        except Exception:
            return None

    def transform_to(self, points, target_epsg=4326):
        from pyproj import Transformer
        if not self.crs or self.crs == target_epsg:
            return points
        t = Transformer.from_crs(self.crs, target_epsg, always_xy=True)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        x_new, y_new = t.transform(x, y)
        return np.column_stack([x_new, y_new, z])


class RealTrainer:
    def __init__(self, log_widget):
        self.log = log_widget

    def train_m1(self, pointcloud_path, output_dir,
                 ransac_threshold=0.05, primitive="Плоскость",
                 classifier_classes=None, emit_progress=None):
        from city3d_backend.core import VALID_CLASSES
        from city3d_backend.runner import PipelineRunner, Task
        from city3d_backend.database import LocalJournal, Repository
        options = dict(getattr(self, "_m1_options", {}))
        options.setdefault("scanning_type", getattr(self, "_scanning_type", "unknown"))
        if classifier_classes:
            options["allowed_classes"] = tuple(c["code"] for c in classifier_classes if c["code"] in VALID_CLASSES)
        try:
            task = Task("M1", "m1.cpu", inputs={"asset": str(pointcloud_path)}, params={"config": options})
            journal = LocalJournal(Path(output_dir) / "outbox.sqlite")
            db_settings = {"database": dict(DB_PARAMS)}
            runner = PipelineRunner(journal=journal, repository=Repository(db_settings))
            result = runner.run([task], output_dir,
                cancel=lambda: QThread.currentThread().isInterruptionRequested(),
                progress=emit_progress, spatial=getattr(self, "_spatial_snapshot", {}))
            self._cancelled = result["status"] == "cancelled"
            self._last_quality = None
            if result["status"] != "success":
                self.log.log("M1: " + json.dumps(result["steps"], ensure_ascii=False), "ERROR")
                return False
            self.log.log("M1: LAS сохранён, статус регистрации БД: " + result["db_registration"], "SUCCESS")
            return True
        except Exception as e:
            self.log.log(f"M1: {e}", "ERROR")
            return False

    def _segment_planes(self, pts, nrm, mask, threshold, max_planes=40):
        """Жадная сегментация точек mask в плоскости через RANSAC.

        Возвращает список {indices, centroid, normal, area, n_inliers}.
        """
        idx = np.where(mask)[0]
        if len(idx) < 500:
            return []
        sub_pts = pts[idx]
        sub_nrm = nrm[idx]
        planes = []
        used = np.zeros(len(idx), dtype=bool)
        for _ in range(max_planes):
            free = np.where(~used)[0]
            if len(free) < 500:
                break
            try:
                model, inliers_local = self._ransac_plane(
                    sub_pts[free], threshold,
                    num_iter=300)
            except Exception:
                break
            if len(inliers_local) < 300:
                break
            a, b, c, d = model
            nn = np.array([a, b, c])
            nn = nn / (np.linalg.norm(nn) + 1e-12)
            if nn[2] < 0:
                nn = -nn
            sel = free[inliers_local]
            used[sel] = True
            plane_pts = sub_pts[sel]
            centroid = plane_pts.mean(axis=0)
            # площадь = площадь выпуклой оболочки в плоскости XY
            try:
                from scipy.spatial import ConvexHull
                _, _, basis = np.linalg.svd(plane_pts - centroid, full_matrices=False)
                uv = (plane_pts - centroid) @ basis[:2].T
                area = (ConvexHull(uv).volume
                        if len(uv) >= 3 else 0.0)
            except Exception:
                area = 0.0
            planes.append({
                "indices": idx[sel],
                "centroid": [float(x) for x in centroid],
                "normal": [float(x) for x in nn],
                "area_m2": float(area),
                "n_inliers": int(len(sel)),
            })
        return planes

    @staticmethod
    def _ransac_plane(pts, threshold, num_iter=300):
        """Минимальный локальный RANSAC для плоскости (без open3d overhead)."""
        n = len(pts)
        if n < 3:
            raise ValueError("too few points")
        rng = np.random.default_rng(42)
        best_count = 0
        best_model = None
        best_inl = None
        for _ in range(num_iter):
            i = rng.choice(n, 3, replace=False)
            p0, p1, p2 = pts[i]
            v1 = p1 - p0
            v2 = p2 - p0
            nn = np.cross(v1, v2)
            norm = np.linalg.norm(nn)
            if norm < 1e-9:
                continue
            nn = nn / norm
            d = -float(nn @ p0)
            dist = np.abs(pts @ nn + d)
            inl = np.where(dist < threshold)[0]
            if len(inl) > best_count:
                best_count = len(inl)
                best_model = (nn[0], nn[1], nn[2], d)
                best_inl = inl
        if best_model is None:
            raise ValueError("RANSAC failed")
        center = pts[best_inl].mean(axis=0)
        _, _, basis = np.linalg.svd(pts[best_inl] - center, full_matrices=False)
        normal = basis[-1]
        best_model = (*normal, -float(normal @ center))
        best_inl = np.flatnonzero(np.abs((pts-center) @ normal) < threshold)
        return best_model, best_inl

    def train_m2(self, raster_path, output_dir):
        from city3d_backend.runner import PipelineRunner, Task
        options=dict(getattr(self,"_m2_snapshot",{}));model_id=options.get("model_id","yolov8n")
        executor={"yolov8n":"yolo.detect","yoloworld_s":"yolo.world"}.get(model_id,model_id)
        try:
            result=PipelineRunner().run([Task("M2",executor,inputs={"asset":str(raster_path)},params=options)],output_dir,
                cancel=lambda:QThread.currentThread().isInterruptionRequested())
            if result["status"]!="success":self.log.log(str(result["steps"]),"ERROR")
            return result["status"]=="success"
        except Exception as e:
            self.log.log(str(e),"ERROR")
            return False

    def train_m3(self, masks_dir, output_dir, n_classes=11, epochs=10):
        from city3d_backend.runner import PipelineRunner, Task
        try:
            params={"epochs":epochs}
            if getattr(self,"_m3_parent",None):params["parent"]=self._m3_parent
            result=PipelineRunner().run([Task("M3","m3.train","train",inputs={"dataset":str(masks_dir)},params=params)],output_dir,
                cancel=lambda:QThread.currentThread().isInterruptionRequested(),progress=getattr(self,"_m3_progress",None))
            self._cancelled=result["status"]=="cancelled"
            if result["status"]!="success":self.log.log(str(result["steps"]),"ERROR")
            return result["status"]=="success"
        except Exception as e:
            self.log.log(str(e),"ERROR")
            return False

    def train_m4(self, dem_path, output_dir):
        from city3d_backend.runner import PipelineRunner, Task
        try:
            result = PipelineRunner().run([Task("M4", "m4.features", inputs={"asset":str(dem_path)},
                 params={"units":"metres"})], output_dir,
                 cancel=lambda: QThread.currentThread().isInterruptionRequested())
            if result["status"] != "success":
                self.log.log(str(result["steps"]), "ERROR")
            return result["status"] == "success"
        except Exception as e:
            self.log.log(str(e), "ERROR")
            return False

    def train_m5(self, classified_las, output_dir):
        from city3d_backend.runner import PipelineRunner, Task
        try:
            result = PipelineRunner().run([Task("M5", "m5.qa", mode="evaluate",
                inputs={"asset":str(classified_las)})], output_dir,
                cancel=lambda: QThread.currentThread().isInterruptionRequested())
            return result["status"] == "success"
        except Exception as e:
            self.log.log(str(e), "ERROR")
            return False

    def train_m6(self, gis_path, output_dir):
        from city3d_backend.runner import PipelineRunner, Task
        try:
            result = PipelineRunner().run([Task("M6", "m6.vector", inputs={"asset":str(gis_path)})], output_dir,
                cancel=lambda: QThread.currentThread().isInterruptionRequested())
            if result["status"] != "success":self.log.log(str(result["steps"]), "ERROR")
            return result["status"] == "success"
        except Exception as e:
            self.log.log(str(e), "ERROR")
            return False

    def train_combined(self, methods, weights, paths, output_dir):
        self.log.log("Совместное semantic fusion не реализовано. Выполните методы отдельно; JSON весов не является обучением.", "ERROR")
        return False


class SemanticModelAdapter:
    """Адаптер семантических 3D-моделей (RandLA / KPConv / PTv3).

    Загружает веса через torch-pointcloud (единый API для трёх моделей),
    выполняет инференс с разбиением на блоки, возвращает метки классов.
    """

    MODELS = {
        "randla_kitti": {
            "hf_id": "randlanet.semantickitti.tsung-han-wu",
            "weights": "models/M1/randlanet_semantickitti.pth",
            "classes": ["car","bicycle","motorcycle","truck","other-vehicle",
                        "person","bicyclist","motorcyclist","road","parking",
                        "sidewalk","other-ground","building","fence",
                        "vegetation","trunk","terrain","pole","traffic-sign"],
            "input_channels": 3,
        },
        "kpconv_s3dis": {
            "hf_id": "kpfcnn-base-sm.s3dis.hugues-thomas",
            "weights": "models/M1/kpconv_s3dis.pth",
            "classes": ["ceiling","floor","wall","beam","column","window",
                        "door","table","chair","sofa","bookcase","board",
                        "clutter"],
            "input_channels": 5,
        },
        "ptv3_scannet": {
            "hf_id": "ptv3-base.scannet200.pointcept",
            "weights": "models/M1/ptv3_scannet200.pth",
            "classes": [f"class_{i}" for i in range(200)],
            "input_channels": 6,
        },
    }

    def __init__(self, model_id, log_widget=None):
        self.model_id = model_id
        self.cfg = self.MODELS.get(model_id)
        if not self.cfg:
            raise ValueError(f"Неизвестная модель: {model_id}")
        self.log = log_widget
        self.model = None
        self.info = None

    def is_available(self):
        try:
            import torch  # noqa
            import torch_pointcloud  # noqa
            return True, "torch + torch-pointcloud установлены"
        except ImportError as e:
            return False, f"pip install torch torch-pointcloud: {e}"

    def load(self, weights_path=None):
        """Загружает модель через torch-pointcloud."""
        ok, msg = self.is_available()
        if not ok:
            if self.log:
                self.log.log(f"Модель {self.model_id}: {msg}", "ERROR")
            return False
        import torch
        import torch_pointcloud as tp
        try:
            if self.log:
                self.log.log(
                    f"Загрузка {self.model_id} ({self.cfg['hf_id']})…",
                    "INFO")
            self.model, self.info = tp.create_model(
                self.cfg["hf_id"],
                task="segmentation",
                pretrained=True,
                return_info=True)
            # Если указан локальный путь — перезагружаем веса оттуда
            if weights_path and Path(weights_path).exists():
                state = torch.load(weights_path, map_location="cpu")
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                self.model.load_state_dict(state)
                if self.log:
                    self.log.log(
                        f"Веса загружены из {weights_path}", "SUCCESS")
            self.model = self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                if self.log:
                    self.log.log("Модель на GPU", "SUCCESS")
            return True
        except Exception as e:
            if self.log:
                self.log.log(f"Ошибка загрузки {self.model_id}: {e}",
                             "ERROR")
            return False

    def infer(self, points, colors=None, block_size=50.0,
              overlap=5.0, emit_progress=None):
        """Инференс с разбиением на блоки.

        points: (N, 3) float32
        colors: (N, 3) float32 в [0, 1] или None
        Возвращает: labels (N,) int, confidence (N,) float
        """
        raise RuntimeError("Neural backend требует проверенного outdoor checkpoint/config и inverse mapping. Используйте CPU M1; универсальный непроверенный adapter отключён.")


class SemanticTrainer:
    """Обёртка над SemanticModelAdapter для RealTrainer."""

    @staticmethod
    def run(model_id, pointcloud_path, output_dir, weights_path=None,
            log_widget=None, emit_progress=None):
        """Запускает семантическую модель, сохраняет результаты."""
        import numpy as np
        from pathlib import Path as _P

        output_dir = _P(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log = log_widget
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(pointcloud_path))
            points = np.asarray(pcd.points, dtype=np.float64)
            colors = (np.asarray(pcd.colors).astype(np.float32)
                      if pcd.has_colors() else None)
            n_total = len(points)
            if log:
                log.log(f"Загружено точек: {n_total:,}", "INFO")

            adapter = SemanticModelAdapter(model_id, log_widget=log)
            if not adapter.load(weights_path):
                return False

            labels, conf = adapter.infer(
                points, colors, emit_progress=emit_progress)
            if labels is None:
                return False

            # Сохранение меток
            np.save(output_dir / "semantic_labels.npy", labels)
            np.save(output_dir / "semantic_confidence.npy", conf)

            # Цветное облако
            from collections import Counter
            unique, counts = np.unique(labels, return_counts=True)
            class_names = adapter.cfg["classes"]
            stats = {}
            for c, n in zip(unique, counts):
                name = (class_names[c] if 0 <= c < len(class_names)
                        else f"class_{c}")
                stats[name] = int(n)

            colors_out = np.zeros((n_total, 3), dtype=np.float64)
            for c in unique:
                if c < 0:
                    continue
                color = SEMANTIC_COLORS.get(
                    int(c) % 10, (0.5, 0.5, 0.5))
                colors_out[labels == c] = color
            out_pcd = o3d.geometry.PointCloud()
            out_pcd.points = o3d.utility.Vector3dVector(points)
            out_pcd.colors = o3d.utility.Vector3dVector(colors_out)
            classified = output_dir / "semantic_classified.ply"
            o3d.io.write_point_cloud(str(classified), out_pcd)

            # Отчёт
            lines = [
                "=" * 70,
                f"ОТЧЁТ {model_id}",
                "=" * 70,
                f"Источник: {pointcloud_path}",
                f"Всего точек: {n_total:,}",
                f"Классов: {len(unique)}",
                "",
                "Распределение:",
            ]
            for name, n in sorted(stats.items(),
                                  key=lambda kv: -kv[1]):
                pct = n / n_total * 100
                lines.append(f"  {name:20s} {n:>12,}  ({pct:5.1f}%)")
            lines += [
                "",
                "ФАЙЛЫ:",
                f"  • {classified}",
                f"  • {output_dir / 'semantic_labels.npy'}",
                f"  • {output_dir / 'semantic_confidence.npy'}",
            ]
            report = output_dir / "semantic_report.txt"
            report.write_text("\n".join(lines), encoding="utf-8")
            if log:
                for ln in lines:
                    log.log(ln, "INFO")
            return True
        except Exception as e:
            import traceback
            if log:
                log.log(f"SemanticTrainer: {e}", "ERROR")
                log.log(traceback.format_exc(), "ERROR")
            return False



class FormatExporter:
    def __init__(self, workspace):
        self.workspace = workspace

    @staticmethod
    def _check_tool(name):
        return shutil.which(name) is not None

    @staticmethod
    def _ok(path, min_bytes=64):
        p = Path(path)
        try:
            return p.exists() and p.stat().st_size >= min_bytes
        except Exception:
            return False

    def _result(self, ok, path, err=""):
        return {"ok": bool(ok), "path": str(path), "error": err}

    def export_to_citygml(self, input_path, output_path, bbox=None):
        return self._result(False, output_path, "CityGML требует подтверждённых объектных геометрий/LOD и CRS; LAS-классификация не является восстановленными зданиями")

    def export_to_usd(self, input_path, output_path):
        try:
            from city3d_backend.runner import verify_snapshot_member
            verify_snapshot_member(input_path)
        except Exception as e:
            return self._result(False, output_path, str(e))
        try:
            from pxr import Usd, UsdGeom, Gf
        except ImportError as e:
            return self._result(False, output_path,
                f"Библиотека pxr (OpenUSD) не установлена: {e}")
        try:
            loaded = PointCloudLoader.load(input_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(loaded["points"])
            if loaded.get("colors") is not None:
                pcd.colors = o3d.utility.Vector3dVector(loaded["colors"])
            points = np.asarray(pcd.points)
            if len(points) == 0:
                return self._result(False, output_path,
                    "В исходном облаке нет точек.")
            stage = Usd.Stage.CreateNew(str(output_path))
            xform = UsdGeom.Xform.Define(stage, '/World')
            stage.SetDefaultPrim(xform.GetPrim())
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            origin = points.min(axis=0)
            xform.AddTranslateOp().Set(Gf.Vec3d(*origin))
            pp = UsdGeom.Points.Define(stage, '/World/PointCloud')
            pp.GetPointsAttr().Set([Gf.Vec3f(*p) for p in (points-origin)])
            stage.Save()
            if not self._ok(output_path):
                return self._result(False, output_path,
                    "Файл USD не создан.")
            return self._result(True, output_path)
        except Exception as e:
            return self._result(False, output_path, str(e))

    def export_to_obj(self, input_path, output_path):
        try:
            from city3d_backend.runner import verify_snapshot_member
            verify_snapshot_member(input_path)
        except Exception as e:
            return self._result(False, output_path, str(e))
        try:
            loaded = PointCloudLoader.load(input_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(loaded["points"])
            if loaded.get("colors") is not None:
                pcd.colors = o3d.utility.Vector3dVector(loaded["colors"])
            if len(pcd.points) < 4:
                return self._result(False, output_path,
                    "Точек меньше 4 — нельзя построить меш.")
            mesh, _ = pcd.compute_convex_hull()
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(output_path), mesh)
            if not self._ok(output_path):
                return self._result(False, output_path,
                    "Файл OBJ не создан.")
            return self._result(True, output_path)
        except Exception as e:
            return self._result(False, output_path, str(e))

    def export_to_geojson(self, input_path, output_path, bbox=None):
        try:
            from city3d_backend.runner import verify_snapshot_member
            verify_snapshot_member(input_path)
        except Exception as e:
            return self._result(False, output_path, str(e))
        if not self._check_tool("ogr2ogr"):
            return self._result(False, output_path,
                "Утилита 'ogr2ogr' (GDAL) не найдена в PATH.")
        cmd = ["ogr2ogr", "-f", "GeoJSON", str(output_path),
               str(input_path), "-nln", "pointcloud"]
        if bbox:
            cmd.extend(["-spat", str(bbox[0]), str(bbox[1]),
                        str(bbox[3]), str(bbox[4])])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return self._result(False, output_path,
                f"ogr2ogr: {r.stderr.strip()[:300]}")
        if not self._ok(output_path):
            return self._result(False, output_path,
                "ogr2ogr отработал, но GeoJSON не создан.")
        return self._result(True, output_path)

    def export_to_postgis(self, input_path, db_params, table_name):
        try:
            from city3d_backend.runner import verify_snapshot_member
            verify_snapshot_member(input_path)
        except Exception as e:
            return self._result(False, "PostGIS", str(e))
        if not self._check_tool("ogr2ogr"):
            return self._result(False, table_name,
                "Утилита 'ogr2ogr' (GDAL) не найдена в PATH.")
        pg = (f"PG:dbname={db_params['dbname']} "
              f"host={db_params['host']} port={db_params['port']} "
              f"user={db_params['user']} password={db_params['password']}")
        cmd = ["ogr2ogr", "-f", "PostgreSQL", pg, str(input_path),
               "-nln", table_name, "-nlt", "PROMOTE_TO_MULTI"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return self._result(False, table_name,
                f"ogr2ogr: {r.stderr.strip()[:300]}")
        return self._result(True, table_name)

    def export_to_dxf(self, input_path, output_path):
        try:
            from city3d_backend.runner import verify_snapshot_member
            verify_snapshot_member(input_path)
        except Exception as e:
            return self._result(False, output_path, str(e))
        p = MODELS["3d_cad_pointcloud"]
        if not p.exists():
            return self._result(False, output_path,
                f"Проект не найден: {p}")
        main_py = p / "main.py"
        if not main_py.exists():
            return self._result(False, output_path,
                f"Не найден {main_py}")
        cmd = [sys.executable, str(main_py),
               "--input", str(input_path), "--output", str(output_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return self._result(False, output_path,
                f"3d_cad_pointcloud: {r.stderr.strip()[:300]}")
        if not self._ok(output_path):
            return self._result(False, output_path,
                "Модуль отработал, но DXF не создан.")
        return self._result(True, output_path)

    def export_to_ifc(self, input_path, output_path):
        try:
            from city3d_backend.runner import verify_snapshot_member
            verify_snapshot_member(input_path)
        except Exception as e:
            return self._result(False, output_path, str(e))
        p = MODELS["Cloud2BIM"]
        if not p.exists():
            return self._result(False, output_path,
                f"Проект не найден: {p}")
        main_py = p / "main.py"
        if not main_py.exists():
            return self._result(False, output_path,
                f"Не найден {main_py}")
        cmd = [sys.executable, str(main_py),
               "--input", str(input_path), "--output", str(output_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return self._result(False, output_path,
                f"Cloud2BIM: {r.stderr.strip()[:300]}")
        if not self._ok(output_path):
            return self._result(False, output_path,
                "Модуль отработал, но IFC не создан.")
        return self._result(True, output_path)

    # Оставляем совместимость со старым API: если код где-то ещё
    # ожидает subprocess.CompletedProcess — вернём ok/err.
    def _as_proc(self, res):
        return type('R', (), {
            'returncode': 0 if res['ok'] else 1,
            'stderr': res.get('error', ''),
        })()


class DatasetPreparer:
    def __init__(self, dataset_root, log_widget=None):
        self.root = Path(dataset_root)
        self.log = log_widget
        self.metadata = {
            "created": datetime.now().isoformat(),
            "source_files": [], "n_scenes": 0, "n_points_total": 0,
            "classes": [],
            "splits": {"train": 0, "val": 0, "test": 0},
            "formats": [], "crs": None, "scanning_type": "unknown"}

    def create_structure(self):
        for d in [self.root, self.root / "raw",
                  self.root / "processed",
                  self.root / "processed" / "train",
                  self.root / "processed" / "val",
                  self.root / "processed" / "test",
                  self.root / "patches", self.root / "annotations",
                  self.root / "augmented", self.root / "checkpoints",
                  self.root / "logs"]:
            d.mkdir(parents=True, exist_ok=True)
        if self.log:
            self.log.log(f"Структура: {self.root}", "SUCCESS")
        return self.root

    def add_scene(self, file_path, copy_raw=True):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        if copy_raw:
            raw_dst = self.root / "raw" / file_path.name
            if not raw_dst.exists():
                shutil.copy2(file_path, raw_dst)
        data = PointCloudLoader.load(file_path)
        self.metadata["source_files"].append(str(file_path))
        self.metadata["n_scenes"] += 1
        self.metadata["n_points_total"] += len(data["points"])
        fmt = data["format"]
        if fmt not in self.metadata["formats"]:
            self.metadata["formats"].append(fmt)
        if data.get("classification") is not None:
            for c in np.unique(data["classification"]):
                if int(c) not in self.metadata["classes"]:
                    self.metadata["classes"].append(int(c))
        return data

    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, seed=42):
        raw_dir = self.root / "raw"
        files = sorted([f for f in raw_dir.iterdir()
                        if f.suffix.lower() in SUPPORTED_INPUT_FORMATS])
        if not files:
            if self.log:
                self.log.log("Нет файлов в raw/.", "ERROR")
            return {}
        random.seed(seed)
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits = {"train": files[:n_train],
                  "val": files[n_train:n_train + n_val],
                  "test": files[n_train + n_val:]}
        names_ru = {"train": "обучающая",
                    "val": "валидационная",
                    "test": "тестовая"}
        for split_name, split_files in splits.items():
            dst_dir = self.root / "processed" / split_name
            for f in split_files:
                data = PointCloudLoader.load(f)
                if f.suffix.lower() in (".las", ".laz"):
                    shutil.copy2(f, dst_dir / f.name)
                else:
                    out = dst_dir / (f.stem + ".ply")
                    PointCloudLoader.save(data, out, ".ply")
            self.metadata["splits"][split_name] = len(split_files)
            if self.log:
                self.log.log(
                    f"{names_ru[split_name].capitalize()}: "
                    f"{len(split_files)} сцен", "SUCCESS")
        self._save_metadata()
        return splits

    def augment(self, rotations=8, noise_std=0.005,
                dropout_ratio=0.05, seed=42):
        rng = np.random.RandomState(seed)
        train_dir = self.root / "processed" / "train"
        aug_dir = self.root / "augmented"
        aug_dir.mkdir(exist_ok=True)
        files = sorted(list(train_dir.glob("*.las")) + list(train_dir.glob("*.laz")))
        count = 0
        for f in files:
            data = PointCloudLoader.load(f)
            points = data["points"]
            center = points.mean(axis=0)
            for i in range(rotations):
                angle = 2 * np.pi * i / rotations
                rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
                rotated = (points-center) @ rot.T
                if noise_std > 0:
                    rotated += rng.normal(0, noise_std, rotated.shape)
                mask = rng.rand(len(rotated)) >= dropout_ratio
                payload = {"points": rotated[mask], "origin": center,
                           "classification": data["classification"][mask], "rotation_matrix": rot, "angle_radians": angle, "seed": seed, "noise_std": noise_std, "dropout_ratio": dropout_ratio,
                           "point_ids": np.flatnonzero(mask), "source": str(f)}
                for name in ("colors", "intensity"):
                    if data.get(name) is not None:
                        payload[name] = data[name][mask]
                np.savez_compressed(aug_dir / f"{f.stem}_rot{i:02d}.npz", **payload)
                count += 1
        self.metadata["augmented_count"] = count
        self._save_metadata()
        if self.log:
            self.log.log(
                f"Расширение выборки: получено {count} новых сцен",
                "SUCCESS")
        return count

    def _save_metadata(self):
        (self.root / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False))

    def summary(self):
        return {"root": str(self.root),
                "n_scenes": self.metadata["n_scenes"],
                "n_points_total": self.metadata["n_points_total"],
                "splits": self.metadata["splits"],
                "formats": self.metadata["formats"],
                "augmented": self.metadata.get("augmented_count", 0)}


class ReferenceDatabase:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.conn = None
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path))
            self._create_tables()
        except Exception as e:
            print(f"Предупреждение: SQLite-БД эталонов недоступна: {e}")

    def _create_tables(self):
        if self.conn is None:
            return
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS references_clouds (
                id INTEGER PRIMARY KEY, name TEXT, path TEXT,
                format TEXT, n_points INTEGER, classes TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        """)
        self.conn.commit()

    def add_reference_cloud(self, name, path, description=""):
        if self.conn is None:
            raise RuntimeError("SQLite-БД эталонов недоступна")
        data = PointCloudLoader.load(path)
        classes = (list(np.unique(data["classification"]))
                   if data.get("classification") is not None else [])
        self.conn.execute("""
            INSERT INTO references_clouds
            (name, path, format, n_points, classes, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, str(path), data["format"], len(data["points"]),
              json.dumps(classes), description))
        self.conn.commit()

    def list_all(self):
        result = {}
        if self.conn is None:
            return {"references_clouds": []}
        try:
            rows = self.conn.execute(
                "SELECT * FROM references_clouds").fetchall()
            result["references_clouds"] = rows
        except Exception:
            result["references_clouds"] = []
        return result
    # ============================================================================
# PostgreSQL: история обучений
# ============================================================================

class TrainingHistoryDB:
    def __init__(self, db_params, log_widget=None):
        self.params = db_params
        self.log = log_widget
        self.conn = None
        self.schema_h = db_params.get("schema_history", "history")
        self.schema_r = db_params.get("schema_refs", "refs")
        self._connect()

    def _connect(self):
        from city3d_backend.database import identifier
        identifier(self.schema_h); identifier(self.schema_r)
        if not HAS_PSYCOPG:
            self._log("psycopg не установлен", "WARNING")
            return
        try:
            from psycopg import sql
            params={k:v for k,v in self.params.items() if k in ("dbname","host","port","user","password","service","sslmode")}
            self.conn=psycopg.connect(**params,autocommit=True,connect_timeout=3)
            with self.conn.cursor() as cur:
                cur.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema_h)))
            self._log("Подключение к существующей PostGIS БД установлено; миграции выполняются отдельно", "SUCCESS")
        except Exception as e:
            self.conn=None
            self._log("БД недоступна для этой сессии: " + type(e).__name__, "WARNING")

    def _log(self, msg, level="INFO"):
        if self.log:
            try:
                self.log.log(msg, level)
            except Exception:
                pass

    @property
    def is_available(self):
        return self.conn is not None

    @staticmethod
    def _bbox_to_box3d(bbox):
        if bbox is None:
            return None
        mnx, mny, mnz, mxx, mxy, mxz = bbox
        return f"BOX3D({mnx} {mny} {mnz}, {mxx} {mxy} {mxz})"

    @staticmethod
    def _footprint_wkt_4326(bbox_4326):
        if bbox_4326 is None:
            return None
        mnx, mny, mxx, mxy = bbox_4326
        return (f"POLYGON(({mnx} {mny}, {mxx} {mny}, "
                f"{mxx} {mxy}, {mnx} {mxy}, {mnx} {mny}))")

    @staticmethod
    def _sha256_file(path, chunk=1 << 20):
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while True:
                    b = f.read(chunk)
                    if not b:
                        break
                    h.update(b)
            return h.hexdigest()
        except Exception:
            return None

    def start_run(self, method, scanning_type, crs_epsg,
                  source_path=None, dataset_root=None,
                  hyperparams=None, n_points=None,
                  bbox=None, bbox_4326=None):
        if not self.is_available:
            return None
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.schema_h}.training_runs
                        (method, scanning_type, crs_epsg, source_path,
                         dataset_root, hyperparams, n_points,
                         bbox, bbox_4326, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s,
                            %s::box3d,
                            CASE WHEN %s::text IS NULL THEN NULL
                                 ELSE ST_GeomFromText(%s, 4326) END,
                            'running')
                    RETURNING id
                """, (
                    method, scanning_type, crs_epsg, source_path,
                    dataset_root, json.dumps(hyperparams or {}),
                    n_points,
                    self._bbox_to_box3d(bbox),
                    self._footprint_wkt_4326(bbox_4326),
                    self._footprint_wkt_4326(bbox_4326),
                ))
                run_id = cur.fetchone()[0]
            self._log(f"История: run_id={run_id} "
                      f"({method}/{scanning_type})", "INFO")
            return run_id
        except Exception as e:
            self._log(f"Ошибка start_run: {e}", "WARNING")
            return None

    def add_artifact(self, run_id, kind, path):
        if not self.is_available or run_id is None:
            return
        p = Path(path)
        if not p.exists() or not p.is_file():
            return
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.schema_h}.training_artifacts
                        (run_id, kind, path, size_bytes, sha256)
                    VALUES (%s, %s, %s, %s, %s)
                """, (run_id, kind, str(p), p.stat().st_size,
                      self._sha256_file(p)))
        except Exception as e:
            self._log(f"Ошибка add_artifact: {e}", "WARNING")

    def finish_run(self, run_id, status, metrics=None,
                   duration_sec=None, error_message=None):
        if not self.is_available or run_id is None:
            return
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {self.schema_h}.training_runs
                    SET status=%s, finished_at=now(),
                        duration_sec=%s, metrics=%s, error_message=%s
                    WHERE id=%s
                """, (status, duration_sec,
                      json.dumps(metrics or {}),
                      error_message, run_id))
        except Exception as e:
            self._log(f"Ошибка finish_run: {e}", "WARNING")

    def save_preset(self, method, scanning_type, crs_epsg,
                    hyperparams, quality_score, source_run_id):
        if not self.is_available:
            return
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.schema_h}.scanning_presets
                        (method, scanning_type, crs_epsg, hyperparams,
                         source_run_id, quality_score)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (method, scanning_type, crs_epsg)
                    DO UPDATE SET
                        hyperparams=EXCLUDED.hyperparams,
                        source_run_id=EXCLUDED.source_run_id,
                        quality_score=EXCLUDED.quality_score,
                        updated_at=now()
                    WHERE
                        {self.schema_h}.scanning_presets.quality_score IS NULL
                        OR EXCLUDED.quality_score >
                           {self.schema_h}.scanning_presets.quality_score
                """, (method, scanning_type, crs_epsg,
                      json.dumps(hyperparams),
                      source_run_id, quality_score))
        except Exception as e:
            self._log(f"Ошибка save_preset: {e}", "WARNING")

    def save_region_crs(self, directory, crs_epsg):
        if not self.is_available:
            return
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.schema_r}.region_crs_history
                        (directory, crs_epsg, n_observations)
                    VALUES (%s, %s, 1)
                    ON CONFLICT (directory, crs_epsg)
                    DO UPDATE SET
                        n_observations = {self.schema_r}.region_crs_history.n_observations + 1,
                        last_seen = now()
                """, (directory, crs_epsg))
        except Exception as e:
            self._log(f"Ошибка save_region_crs: {e}", "WARNING")

    def find_similar(self, method, scanning_type, crs_epsg=None, limit=5):
        if not self.is_available:
            return []
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT id, started_at, hyperparams, metrics,
                           crs_epsg, duration_sec
                    FROM {self.schema_h}.training_runs
                    WHERE method=%s AND scanning_type=%s
                      AND status='success'
                      AND (%s IS NULL OR crs_epsg=%s)
                    ORDER BY started_at DESC
                    LIMIT %s
                """, (method, scanning_type, crs_epsg, crs_epsg, limit))
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception as e:
            self._log(f"Ошибка find_similar: {e}", "WARNING")
            return []

    def get_preset(self, method, scanning_type, crs_epsg=None):
        if not self.is_available:
            return None
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT hyperparams, quality_score, source_run_id
                    FROM {self.schema_h}.v_best_presets
                    WHERE method=%s AND scanning_type=%s
                    LIMIT 1
                """, (method, scanning_type))
                row = cur.fetchone()
                if row:
                    return {"hyperparams": row[0],
                            "quality_score": row[1],
                            "source_run_id": row[2]}
        except Exception as e:
            self._log(f"Ошибка get_preset: {e}", "WARNING")
        return None

    def get_region_crs(self, directory):
        if not self.is_available:
            return None
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT crs_epsg
                    FROM {self.schema_r}.region_crs_history
                    WHERE directory=%s
                    ORDER BY n_observations DESC, last_seen DESC
                    LIMIT 1
                """, (directory,))
                row = cur.fetchone()
                return row[0] if row else None
        except Exception:
            return None

    def list_runs(self, limit=200, method=None, scanning_type=None,
                  status=None):
        if not self.is_available:
            return []
        try:
            sql = (f"SELECT id, started_at, finished_at, method, "
                   f"scanning_type, crs_epsg, n_points, duration_sec, "
                   f"status, output_dir, metrics "
                   f"FROM {self.schema_h}.training_runs WHERE 1=1")
            args = []
            if method:
                sql += " AND method=%s"
                args.append(method)
            if scanning_type:
                sql += " AND scanning_type=%s"
                args.append(scanning_type)
            if status:
                sql += " AND status=%s"
                args.append(status)
            sql += " ORDER BY started_at DESC LIMIT %s"
            args.append(limit)
            with self.conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception as e:
            self._log(f"Ошибка list_runs: {e}", "WARNING")
            return []

    def search_epsg(self, query, limit=200):
        """Резервный поиск по таблице refs.epsg_codes в Postgres.

        Основной справочник — EPSGCatalog (JSON). Этот метод нужен
        только для случаев, когда JSON недоступен.
        """
        if not self.is_available:
            return []
        try:
            q = f"%{query.strip()}%"
            with self.conn.cursor() as cur:
                if query.strip().isdigit():
                    cur.execute(f"""
                        SELECT code, name, kind, area
                        FROM {self.schema_r}.epsg_codes
                        WHERE code = %s
                        LIMIT %s
                    """, (int(query.strip()), limit))
                else:
                    cur.execute(f"""
                        SELECT code, name, kind, area
                        FROM {self.schema_r}.epsg_codes
                        WHERE name ILIKE %s
                        ORDER BY code
                        LIMIT %s
                    """, (q, limit))
                return cur.fetchall()
        except Exception as e:
            self._log(f"Ошибка search_epsg: {e}", "WARNING")
            return []

    def load_epsg_catalog(self):
        if not self.is_available:
            return 0
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.schema_r}.epsg_codes")
                count = cur.fetchone()[0]
            if count > 1000:
                self._log(f"EPSG-справочник уже заполнен ({count} записей)",
                          "INFO")
                return count

            from pyproj.database import query_crs_info
            infos = query_crs_info(auth_name="EPSG",
                                   allow_deprecated=False)
            inserted = 0
            with self.conn.cursor() as cur:
                for info in infos:
                    try:
                        code = int(info.code)
                    except Exception:
                        continue
                    area = (info.area_of_use.name
                            if info.area_of_use else None)
                    cur.execute(f"""
                        INSERT INTO {self.schema_r}.epsg_codes
                            (code, name, kind, area, is_active)
                        VALUES (%s, %s, %s, %s, TRUE)
                        ON CONFLICT (code) DO UPDATE
                        SET name = EXCLUDED.name, area = EXCLUDED.area
                    """, (code, info.name,
                          (info.type_name or "").lower() or None,
                          area))
                    inserted += 1
            self._log(f"EPSG-справочник загружен: {inserted} записей",
                      "SUCCESS")
            return inserted
        except Exception as e:
            self._log(f"Ошибка load_epsg_catalog: {e}", "WARNING")
            return 0


class ScanningPresets:
    PRESETS = {
        "ALS_manned": {"sor_k": 6,  "sor_std": 1.0, "voxel": 0.25,
                       "rotations": 8, "noise": 0.005, "dropout": 0.02,
                       "ransac_threshold": 0.15},
        "ALS_uav":    {"sor_k": 10, "sor_std": 1.5, "voxel": 0.15,
                       "rotations": 8, "noise": 0.005, "dropout": 0.03,
                       "ransac_threshold": 0.10},
        "MLS":        {"sor_k": 20, "sor_std": 2.0, "voxel": 0.05,
                       "rotations": 8, "noise": 0.005, "dropout": 0.05,
                       "ransac_threshold": 0.05},
        "TLS":        {"sor_k": 20, "sor_std": 1.5, "voxel": 0.01,
                       "rotations": 2, "noise": 0.002, "dropout": 0.02,
                       "ransac_threshold": 0.02},
        "handheld":   {"sor_k": 20, "sor_std": 1.5, "voxel": 0.005,
                       "rotations": 2, "noise": 0.002, "dropout": 0.02,
                       "ransac_threshold": 0.01},
        "stationary": {"sor_k": 15, "sor_std": 1.2, "voxel": 0.002,
                       "rotations": 1, "noise": 0.001, "dropout": 0.01,
                       "ransac_threshold": 0.005},
        "unknown":    {"sor_k": 20, "sor_std": 2.0, "voxel": 0.05,
                       "rotations": 8, "noise": 0.005, "dropout": 0.05,
                       "ransac_threshold": 0.05},
    }

    @classmethod
    def get(cls, scanning_type):
        return dict(cls.PRESETS.get(scanning_type,
                                    cls.PRESETS["unknown"]))


# ============================================================================
# 3D-просмотрщик облака точек
# ============================================================================

class PointCloudViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pcd = None
        self.scatter = None
        self.bbox_item = None
        self.center = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.info_label = QLabel("Нет данных")
        self.info_label.setStyleSheet("color: #4ec9b0; padding: 4px;")
        toolbar.addWidget(self.info_label)
        toolbar.addStretch()
        for t, s in [("Сброс", self.reset), ("Сверху", self.top),
                     ("Спереди", self.front), ("Сбоку", self.side)]:
            b = QPushButton(t)
            b.clicked.connect(s)
            toolbar.addWidget(b)
        layout.addLayout(toolbar)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((30, 30, 30, 255))
        self.view.setCameraPosition(distance=100, elevation=30, azimuth=45)
        layout.addWidget(self.view)
        axis = gl.GLAxisItem()
        axis.setSize(50, 50, 50)
        self.view.addItem(axis)

    def set_pointcloud(self, pcd):
        self.pcd = pcd
        if pcd is None or len(pcd.points) == 0:
            self.info_label.setText("Облако пустое")
            return
        if self.scatter is not None:
            self.view.removeItem(self.scatter)
        pts = np.asarray(pcd.points)
        n = len(pts)
        if n > 500_000:
            idx = np.random.choice(n, 500_000, replace=False)
            pts = pts[idx]
        self.center = pts.mean(axis=0)
        disp = pts - self.center
        z = disp[:, 2]
        zn = (z - z.min()) / (z.max() - z.min() + 1e-9)
        colors = np.column_stack([
            zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn, np.ones(len(disp))])
        self.scatter = gl.GLScatterPlotItem(
            pos=disp.astype(np.float32),
            color=colors.astype(np.float32),
            size=2.0, pxMode=True)
        self.view.addItem(self.scatter)
        self.info_label.setText(f"Точек: {n:,}")
        diag = float(np.linalg.norm(
            disp.max(axis=0) - disp.min(axis=0)))
        self.view.setCameraPosition(distance=max(diag * 1.5, 1.0))

    def show_bbox(self, bbox):
        if self.pcd is None or self.center is None:
            return
        if self.bbox_item is not None:
            self.view.removeItem(self.bbox_item)
        mnx, mny, mnz, mxx, mxy, mxz = bbox
        corners = np.array([
            [mnx, mny, mnz], [mxx, mny, mnz],
            [mxx, mxy, mnz], [mnx, mxy, mnz],
            [mnx, mny, mxz], [mxx, mny, mxz],
            [mxx, mxy, mxz], [mnx, mxy, mxz]]) - self.center
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        segs = []
        for i, j in edges:
            segs.append(corners[i])
            segs.append(corners[j])
        self.bbox_item = gl.GLLinePlotItem(
            pos=np.array(segs, dtype=np.float32),
            color=(1.0, 0.4, 0.0, 1.0), width=2.0, mode="lines")
        self.view.addItem(self.bbox_item)

    def reset(self):
        self.view.setCameraPosition(distance=100, elevation=30, azimuth=45)
        self.view.update()

    def top(self):
        self.view.setCameraPosition(elevation=90, azimuth=0)
        self.view.update()

    def front(self):
        self.view.setCameraPosition(elevation=0, azimuth=0)
        self.view.update()

    def side(self):
        self.view.setCameraPosition(elevation=0, azimuth=90)
        self.view.update()


class Mesh3DViewer(QWidget):
    """3D-просмотр меша/облака (с фильтрацией вырожденных треугольников)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        self.info_label = QLabel("Нет данных")
        self.info_label.setStyleSheet("color: #4ec9b0; padding: 4px;")
        toolbar.addWidget(self.info_label)
        toolbar.addStretch()
        for label, slot in [("Сброс", self._reset_view),
                            ("Сверху", self._top_view),
                            ("Спереди", self._front_view)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            toolbar.addWidget(b)
        layout.addLayout(toolbar)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((25, 25, 28, 255))
        self.view.setCameraPosition(distance=100, elevation=30, azimuth=45)
        layout.addWidget(self.view)

        self.axis = gl.GLAxisItem()
        self.axis.setSize(10, 10, 10)
        self.view.addItem(self.axis)

        self._mesh_item = None

    def clear(self):
        """Полная очистка: убирает меш, сбрасывает камеру и ось.
        PATCHED_EXPORT_UI"""
        if self._mesh_item is not None:
            try:
                self.view.removeItem(self._mesh_item)
            except Exception:
                pass
            self._mesh_item = None
        # Сбрасываем вид
        try:
            self.view.setCameraPosition(distance=100,
                                        elevation=30, azimuth=45)
            self.view.update()
        except Exception:
            pass
        # Сбрасываем подписи
        try:
            self.info_label.setText("Нет данных")
        except Exception:
            pass

    def _fit_axes(self, span):
        try:
            s = float(max(span)) if span is not None else 10.0
            s = max(s, 1e-3)
        except Exception:
            s = 10.0
        self.axis.setSize(s, s, s)

    @staticmethod
    def _filter_degenerate(verts, faces, tol=1e-9):
        if len(faces) == 0:
            return faces
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1)
        keep = areas > tol
        return faces[keep]

    def show_mesh(self, path):
        self.clear()
        path = str(path)
        try:
            mesh = o3d.io.read_triangle_mesh(path)
            if len(mesh.triangles) > 0:
                if not mesh.has_vertex_normals():
                    mesh.compute_vertex_normals()
                verts = np.asarray(mesh.vertices).astype(np.float32)
                faces = np.asarray(mesh.triangles).astype(np.uint32)

                centroid = verts.mean(axis=0)
                verts = verts - centroid

                n_before = len(faces)
                faces = self._filter_degenerate(verts, faces)
                n_removed = n_before - len(faces)

                span = verts.max(axis=0) - verts.min(axis=0)
                diag = float(np.linalg.norm(span)) or 1.0
                self._fit_axes(span)
                self.view.setCameraPosition(
                    distance=max(diag * 1.8, 1.0),
                    elevation=30, azimuth=45)

                try:
                    self._mesh_item = gl.GLMeshItem(
                        vertexes=verts, faces=faces,
                        smooth=True, drawEdges=True,
                        edgeColor=(0.15, 0.6, 0.45, 1.0),
                        color=(0.5, 0.7, 0.9, 1.0),
                        shader='shaded')
                except Exception:
                    self._mesh_item = gl.GLMeshItem(
                        vertexes=verts, faces=faces,
                        smooth=False, drawEdges=False,
                        color=(0.5, 0.7, 0.9, 1.0),
                        shader=None)
                self.view.addItem(self._mesh_item)

                note = ""
                if n_removed > 0:
                    note = f", удалено вырожденных: {n_removed:,}"
                self.info_label.setText(
                    f"Меш: {len(verts):,} вершин, "
                    f"{len(faces):,} треугольников{note}")
                self.view.update()
                return True

            pcd = o3d.io.read_point_cloud(path)
            pts = None
            if len(pcd.points) > 0:
                pts = np.asarray(pcd.points).astype(np.float32)
            else:
                try:
                    import laspy
                    las = laspy.read(path)
                    pts = np.vstack(
                        (las.x, las.y, las.z)).T.astype(np.float32)
                except Exception:
                    pts = None
            if pts is None or len(pts) == 0:
                self.info_label.setText(
                    "Файл не содержит ни меша, ни точек")
                return False

            n0 = len(pts)
            if n0 > 400_000:
                idx = np.random.choice(n0, 400_000, replace=False)
                pts = pts[idx]

            centroid = pts.mean(axis=0)
            pts = pts - centroid
            span = pts.max(axis=0) - pts.min(axis=0)
            diag = float(np.linalg.norm(span)) or 1.0
            self._fit_axes(span)
            self.view.setCameraPosition(
                distance=max(diag * 1.8, 1.0),
                elevation=30, azimuth=45)

            self._mesh_item = gl.GLScatterPlotItem(
                pos=pts, color=(0.35, 0.85, 0.65, 1.0),
                size=2.0, pxMode=True)
            self.view.addItem(self._mesh_item)
            self.info_label.setText(
                f"Облако точек: {n0:,} (показано {len(pts):,})")
            self.view.update()
            return True
        except Exception as e:
            self.info_label.setText(f"Ошибка чтения: {e}")
            return False

    def _reset_view(self):
        self.view.setCameraPosition(elevation=30, azimuth=45)
        self.view.update()

    def _top_view(self):
        self.view.setCameraPosition(elevation=90, azimuth=0)
        self.view.update()

    def _front_view(self):
        self.view.setCameraPosition(elevation=0, azimuth=0)
        self.view.update()


from city3d_backend.gui_layers import OfflineMapViewer as WebMapViewer


class ExportPreviewViewer(QWidget):
    TEXT_PREVIEW_LIMIT = 200_000

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.header = QLabel("Предпросмотр экспортированных данных")
        self.header.setStyleSheet(
            "color: #4ec9b0; font-weight: bold; padding: 6px; "
            "background-color: #252526; border-radius: 4px;")
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.stack = QStackedWidget()
        self.mesh_view = Mesh3DViewer()

        if True:
            self.web_view = WebMapViewer()
        else:
            self.web_view = None

        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setFont(QFont("Monospace", 9))
        self.text_view.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4; "
            "border: 1px solid #3c3c3c;")

        self.stack.addWidget(self.mesh_view)
        if self.web_view is not None:
            self.stack.addWidget(self.web_view)
        self.stack.addWidget(self.text_view)
        layout.addWidget(self.stack)

        self._current_path = None

    def show_file(self, path):
        if not path:
            self.header.setText("Нет файла для предпросмотра")
            return
        p = Path(path)
        if not p.exists():
            self.header.setText(f"⚠ Файл не найден: {path}")
            self._show_text_placeholder(f"Файл не найден: {path}")
            return
        self._current_path = p
        ext = p.suffix.lower()
        size_str = self._human_size(p)

        if ext in PREVIEW_MESH_EXTS or ext in PREVIEW_CLOUD_EXTS:
            kind = "Меш" if ext in PREVIEW_MESH_EXTS else "Облако"
            self.header.setText(f"{kind}: {p.name}  ({size_str})")
            if self.mesh_view.show_mesh(p):
                self.stack.setCurrentWidget(self.mesh_view)
            else:
                self._show_text(p, note="open3d не смог прочитать файл")
            return

        if ext == ".geojson":
            if self.web_view is not None:
                self.header.setText(
                    f"Карта (Leaflet + OpenStreetMap): "
                    f"{p.name}  ({size_str})")
                if self.web_view.show_geojson(p):
                    self.stack.setCurrentWidget(self.web_view)
                else:
                    self._show_text(p, note="не удалось открыть карту")
            else:
                self._show_text(
                    p, note="PyQtWebEngine не установлен — "
                            "показан текст.")
            return

        self._show_text(p)

    def _show_text(self, p, note=""):
        try:
            with open(p, "rb") as f:
                raw = f.read(self.TEXT_PREVIEW_LIMIT)
            text = raw.decode("utf-8", errors="replace")
            if p.suffix.lower() == ".json":
                try:
                    obj = json.loads(text)
                    text = json.dumps(obj, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            size_str = self._human_size(p)
            header = f"{p.name}  ({size_str})\n"
            if note:
                header += f"⚠  {note}\n"
            header += "─" * 60 + "\n\n"
            self.text_view.setPlainText(header + text)
            self.header.setText(f"Текстовый предпросмотр: {p.name}")
            self.stack.setCurrentWidget(self.text_view)
        except Exception as e:
            self._show_text_placeholder(f"Ошибка чтения {p}:\n{e}")

    def _show_text_placeholder(self, msg):
        self.text_view.setPlainText(msg)
        self.stack.setCurrentWidget(self.text_view)

    @staticmethod
    def _human_size(p):
        try:
            sz = p.stat().st_size
        except Exception:
            return "?"
        for unit in ("Б", "КБ", "МБ", "ГБ"):
            if sz < 1024:
                return f"{sz:.1f} {unit}"
            sz /= 1024
        return f"{sz:.1f} ТБ"


# ============================================================================
# Вкладка 1 — Загрузка
# ============================================================================

class LoadDataTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.crs_detector = CRSDetector(
            history_db=getattr(main_window, "history_db", None))
        # Справочник EPSG берём у MainWindow, он уже загружен из JSON
        self.epsg_catalog = main_window.epsg_catalog
        self._pending_info = None
        self._pending_crs = None
        self._pending_crs_src = None
        self._pending_need_dialog = False
        self._pending_file_path = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Vertical)
        top = QWidget()
        tl = QVBoxLayout(top)

        fg = QGroupBox("Исходное облако точек")
        fl = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText(
            "Форматы: .las, .laz, .ply, .pcd, .xyz, .e57, .bin")
        fl.addWidget(self.file_path)
        b = QPushButton("Обзор...")
        b.clicked.connect(self.browse)
        fl.addWidget(b)
        fg.setLayout(fl)
        tl.addWidget(fg)

        stg = QGroupBox("Вид лазерного сканирования")
        stg.setToolTip(
            "Тип съёмки задаёт рекомендованные параметры предобработки,\n"
            "аугментации и методов M1-M6. Сохраняется в истории обучений.")
        stl = QFormLayout()
        self.scanning_type = QComboBox()
        self.scanning_type.addItems([
            "unknown    — не указано",
            "ALS_manned — воздушное (пилотируемое)",
            "ALS_uav    — воздушное (БПЛА)",
            "MLS        — мобильное (авто, SLAM)",
            "TLS        — наземное (штатив)",
            "handheld   — переносное (ручное)",
            "stationary — стационарное",
        ])
        self.scanning_type.currentIndexChanged.connect(
            self._on_scanning_type_changed)
        stl.addRow("Тип:", self.scanning_type)
        stg.setLayout(stl)
        tl.addWidget(stg)

        # PATCHED_CRS_UI: расширенный блок CRS + вертикальная СК
        cg = QGroupBox("Система координат и высот")
        cl = QFormLayout()
        self.crs_display = QLineEdit()
        self.crs_display.setReadOnly(True)
        cl.addRow("Горизонтальный CRS:", self.crs_display)
        self.crs_source = QLineEdit()
        self.crs_source.setReadOnly(True)
        cl.addRow("Источник:", self.crs_source)

        self.vertical_crs_combo = QComboBox()
        for code, name in VERTICAL_CRS_LIST:
            self.vertical_crs_combo.addItem(name, code)
        self.vertical_crs_combo.setToolTip(
            "Вертикальная система высот (Балтийская, EGM96, эллипсоид)")
        cl.addRow("Вертикальная СК:", self.vertical_crs_combo)

        self.height_offset_spin = QDoubleSpinBox()
        self.height_offset_spin.setRange(-10000.0, 10000.0)
        self.height_offset_spin.setDecimals(3)
        self.height_offset_spin.setValue(0.0)
        self.height_offset_spin.setToolTip(
            "Смещение высот (м), если нужно подправить Z")
        cl.addRow("Смещение высот (м):", self.height_offset_spin)

        b_pick_crs = QPushButton("Выбрать горизонтальный CRS…")
        b_pick_crs.setToolTip(
            "Открыть справочник EPSG с поиском")
        b_pick_crs.clicked.connect(self.open_crs_dialog)
        cl.addRow("", b_pick_crs)
        cg.setLayout(cl)
        tl.addWidget(cg)

        ig = QGroupBox("Информация")
        il = QVBoxLayout()
        self.info_label = QLabel("Файл не загружен")
        self.info_label.setWordWrap(True)
        il.addWidget(self.info_label)
        ig.setLayout(il)
        tl.addWidget(ig)

        bl = QHBoxLayout()
        self.btn_load = QPushButton("Загрузить (любой формат)")
        self.btn_load.clicked.connect(self.load_pcd)
        bl.addWidget(self.btn_load)
        tl.addLayout(bl)
        splitter.addWidget(top)
        self.viewer = PointCloudViewer()
        splitter.addWidget(self.viewer)
        splitter.setSizes([420, 500])
        layout.addWidget(splitter)
        self.setLayout(layout)

    def _on_scanning_type_changed(self):
        st = self.current_scanning_type()
        preset = ScanningPresets.get(st)
        hdb = getattr(self.main_window, "history_db", None)
        if hdb and hdb.is_available:
            db_preset = hdb.get_preset("preprocess", st)
            if db_preset and isinstance(db_preset.get("hyperparams"), dict):
                preset.update(db_preset["hyperparams"])
        try:
            pp = self.main_window.tabs.widget(1)
            pp.sor_k.setValue(int(preset["sor_k"]))
            pp.sor_std.setValue(float(preset["sor_std"]))
            pp.voxel.setValue(float(preset["voxel"]))
        except Exception:
            pass
        try:
            dt = self.main_window.tabs.widget(3)
            dt.rotations.setValue(int(preset["rotations"]))
            dt.noise.setValue(float(preset["noise"]))
            dt.dropout.setValue(float(preset["dropout"]))
        except Exception:
            pass
        try:
            tt = self.main_window.tabs.widget(4)
            tt.m1_ransac_threshold.setValue(
                float(preset["ransac_threshold"]))
        except Exception:
            pass

    def current_scanning_type(self):
        txt = self.scanning_type.currentText()
        return txt.split()[0] if txt else "unknown"

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите облако", str(WORKSPACE),
            "Point Cloud (*.las *.laz *.ply *.pcd *.xyz *.e57 *.bin "
            "*.txt);;All Files (*)")
        if path:
            self.file_path.setText(path)

    def load_pcd(self):
        path = self.file_path.text()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Ошибка", "Файл не найден.")
            return
        self._pending_info = None
        self._pending_crs = None
        self._pending_crs_src = None
        self._pending_need_dialog = False
        self._pending_file_path = path
        self.main_window.scanning_type = self.current_scanning_type()

        worker = WorkerThread(self._load_task, path)
        worker.log_signal.connect(self.main_window.log_widget.log)
        worker.task_done.connect(self._on_loaded)
        worker.start()
        self._worker = worker

    def _load_task(self, emit_log, emit_progress, path):
        emit_log(f"Загрузка: {path}", "INFO")
        try:
            self.crs_detector.log = EmitLogger(emit_log)
            result = self.crs_detector.detect(path)
            data = PointCloudLoader.load(path)
            emit_progress(50)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data["points"])
            if data.get("colors") is not None:
                pcd.colors = o3d.utility.Vector3dVector(data["colors"])
            self.main_window.pcd = pcd
            self.main_window.pcd_path = path
            self.main_window.pcd_data = data
            self.main_window.crs_detector = self.crs_detector

            self._pending_info = (
                f"Точек: {len(data['points']):,}\n"
                f"Формат: {data['format']}\n"
                f"Вид сканирования: {self.current_scanning_type()}\n"
                f"Цвета: "
                f"{'да' if data.get('colors') is not None else 'нет'}\n"
                f"Классификация: "
                f"{'да' if data.get('classification') is not None else 'нет'}")

            if result and result.get("epsg"):
                self._pending_crs = (
                    f"EPSG:{self.crs_detector.crs} — "
                    f"{self.crs_detector.crs_description}")
                self._pending_crs_src = self.crs_detector.crs_source
            else:
                self._pending_crs = "не определён"
                self._pending_crs_src = "нужно выбрать вручную"
                self._pending_need_dialog = True

            emit_log(f"Загружено {len(data['points']):,} точек.",
                     "SUCCESS")
            emit_progress(100)
            return True
        except Exception as e:
            emit_log(f"Ошибка загрузки: {e}", "ERROR")
            return False

    def _on_loaded(self, ok):
        if not ok or self.main_window.pcd is None:
            return
        if self._pending_info:
            self.info_label.setText(self._pending_info)
        if self._pending_crs:
            self.crs_display.setText(self._pending_crs)
        if self._pending_crs_src:
            self.crs_source.setText(self._pending_crs_src)
        self.viewer.set_pointcloud(self.main_window.pcd)

        if self._pending_need_dialog:
            self._ask_crs_dialog()

    def _ask_crs_dialog(self):
        hdb = getattr(self.main_window, "history_db", None)
        catalog = self.epsg_catalog

        if not catalog or not catalog.entries:
            QMessageBox.information(
                self, "CRS не определён",
                "Автоматически CRS не определён, а справочник EPSG "
                "недоступен (файл crslist.json не найден или пуст).\n\n"
                "Данные будут обработаны в исходной СК, но экспорт в "
                "GeoJSON/CityGML/PostGIS может оказаться некорректным.")
            return

        pts = np.asarray(self.main_window.pcd.points)
        suggested = ""
        if len(pts) > 0:
            x_min = float(pts[:, 0].min())
            if 100000 < abs(x_min) < 10000000:
                zone = int((x_min + 180) / 6) + 1
                if 1 <= zone <= 60:
                    suggested = f"UTM zone {zone}N"
        dlg = CRSPickerDialog(
            catalog, self,
            title="CRS не определён — выберите из справочника",
            suggested_text=suggested)
        dlg.exec_()
        epsg = dlg.selected_epsg
        if epsg:
            from pyproj import CRS
            try:
                crs = CRS.from_epsg(epsg)
                self.crs_detector.crs = epsg
                self.crs_detector.crs_source = "выбрано вручную"
                self.crs_detector.crs_description = crs.name
                self.crs_display.setText(f"EPSG:{epsg} — {crs.name}")
                self.crs_source.setText("выбрано вручную")
                if self._pending_file_path and hdb and hdb.is_available:
                    hdb.save_region_crs(
                        str(Path(self._pending_file_path).parent), epsg)
                self.main_window.log_widget.log(
                    f"CRS установлен вручную: EPSG:{epsg} — {crs.name}",
                    "SUCCESS")
            except Exception as e:
                self.main_window.log_widget.log(
                    f"Не удалось установить CRS: {e}", "ERROR")

    def open_crs_dialog(self):
        catalog = self.epsg_catalog
        if not catalog or not catalog.entries:
            QMessageBox.warning(
                self, "Справочник недоступен",
                f"Для выбора CRS нужен файл crslist.json рядом с "
                f"программой.\nОжидаемый путь: {EPSG_CATALOG_PATH}")
            return
        dlg = CRSPickerDialog(catalog, self, title="Выбор CRS")
        dlg.exec_()
        epsg = dlg.selected_epsg
        if epsg:
            from pyproj import CRS
            try:
                crs = CRS.from_epsg(epsg)
                self.crs_detector.crs = epsg
                self.crs_detector.crs_source = "выбрано вручную"
                self.crs_detector.crs_description = crs.name
                self.crs_display.setText(f"EPSG:{epsg} — {crs.name}")
                self.crs_source.setText("выбрано вручную")
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", str(e))


# ============================================================================
# Вкладка 2 — Предобработка
# ============================================================================

class PreprocessTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        g = QGroupBox("Параметры")
        f = QFormLayout()
        self.sor_k = QSpinBox()
        self.sor_k.setRange(5, 100)
        self.sor_k.setValue(20)
        self.sor_k.setToolTip(
            "SOR: число ближайших соседей для оценки выбросов.\n"
            "ALS: 6, MLS: 20, TLS: 20.")
        f.addRow("SOR соседей:", self.sor_k)
        self.sor_std = QDoubleSpinBox()
        self.sor_std.setRange(0.5, 5.0)
        self.sor_std.setValue(2.0)
        self.sor_std.setSingleStep(0.1)
        self.sor_std.setToolTip(
            "SOR: множитель стандартного отклонения.\n"
            "ALS: 1.0, MLS: 2.0, TLS: 1.5.")
        f.addRow("SOR std ratio:", self.sor_std)
        self.voxel = QDoubleSpinBox()
        self.voxel.setRange(0.001, 1.0)
        self.voxel.setValue(0.05)
        self.voxel.setDecimals(3)
        self.voxel.setToolTip(
            "Размер вокселя (м): ALS 0.25, MLS 0.05, TLS 0.01.")
        f.addRow("Воксел (м):", self.voxel)
        g.setLayout(f)
        layout.addWidget(g)
        bl = QHBoxLayout()
        for text, slot in [("Очистить (SOR)", self.clean),
                           ("Даунсэмплинг", self.downsample),
                           ("Сохранить", self.save_clean)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            bl.addWidget(b)
        layout.addLayout(bl)
        self.log = LogWidget()
        layout.addWidget(self.log)
        self.setLayout(layout)

    def clean(self):
        if self.main_window.pcd is None:
            self.log.log("Загрузите облако.", "ERROR")
            return
        worker = WorkerThread(self._clean_task)
        worker.log_signal.connect(self.log.log)
        worker.start()
        self._worker = worker

    def _clean_task(self, emit_log, emit_progress):
        pcd = self.main_window.pcd
        pcd_c, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.sor_k.value(),
            std_ratio=self.sor_std.value())
        emit_progress(50)
        emit_log(f"После очистки: {len(pcd_c.points):,}", "SUCCESS")
        self.main_window.pcd = pcd_c
        emit_progress(100)
        return True

    def downsample(self):
        if self.main_window.pcd is None:
            self.log.log("Загрузите облако.", "ERROR")
            return
        worker = WorkerThread(self._downsample_task)
        worker.log_signal.connect(self.log.log)
        worker.start()
        self._worker = worker

    def _downsample_task(self, emit_log, emit_progress):
        pcd = self.main_window.pcd
        pcd_d = pcd.voxel_down_sample(voxel_size=self.voxel.value())
        emit_log(f"После даунсэмплинга: {len(pcd_d.points):,}", "SUCCESS")
        self.main_window.pcd = pcd_d
        emit_progress(100)
        return True

    def save_clean(self):
        if self.main_window.pcd is None:
            return
        path = WORKSPACE / "pointcloud_clean.ply"
        o3d.io.write_point_cloud(str(path), self.main_window.pcd)
        self.main_window.clean_pcd_path = str(path)
        self.log.log(f"Сохранено: {path}", "SUCCESS")


# ============================================================================
# Вкладка 3 — BBOX
# ============================================================================

class BoundingBoxTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Выбор области (BBOX)"))
        g = QGroupBox("Границы")
        f = QFormLayout()
        self.spins = {}
        for key, label in [("min_x", "Мин. X"), ("max_x", "Макс. X"),
                           ("min_y", "Мин. Y"), ("max_y", "Макс. Y"),
                           ("min_z", "Мин. Z"), ("max_z", "Макс. Z")]:
            s = QDoubleSpinBox()
            s.setRange(-1e9, 1e9)
            s.setDecimals(3)
            s.valueChanged.connect(self._on_change)
            f.addRow(f"{label}:", s)
            self.spins[key] = s
        g.setLayout(f)
        ll.addWidget(g)
        for text, slot in [("Автоопределение", self.auto_bbox),
                           ("Обрезать", self.crop)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            ll.addWidget(b)
        self.log = LogWidget()
        ll.addWidget(self.log)
        splitter.addWidget(left)
        self.viewer = PointCloudViewer()
        splitter.addWidget(self.viewer)
        splitter.setSizes([380, 720])
        layout.addWidget(splitter)
        self.setLayout(layout)

    def _on_change(self):
        if self.main_window.pcd is None:
            return
        self.viewer.set_pointcloud(self.main_window.pcd)
        self.viewer.show_bbox(self.get_bbox())

    def auto_bbox(self):
        if self.main_window.pcd is None:
            return
        pcd = self.main_window.pcd
        mb = pcd.get_min_bound()
        xb = pcd.get_max_bound()
        for s in self.spins.values():
            s.blockSignals(True)
        self.spins["min_x"].setValue(mb[0])
        self.spins["max_x"].setValue(xb[0])
        self.spins["min_y"].setValue(mb[1])
        self.spins["max_y"].setValue(xb[1])
        self.spins["min_z"].setValue(mb[2])
        self.spins["max_z"].setValue(xb[2])
        for s in self.spins.values():
            s.blockSignals(False)
        self._on_change()
        self.log.log("Границы определены.", "SUCCESS")

    def get_bbox(self):
        return (self.spins["min_x"].value(), self.spins["min_y"].value(),
                self.spins["min_z"].value(), self.spins["max_x"].value(),
                self.spins["max_y"].value(), self.spins["max_z"].value())

    def crop(self):
        if self.main_window.pcd is None:
            return
        worker = WorkerThread(self._crop_task)
        worker.log_signal.connect(self.log.log)
        worker.task_done.connect(self._on_cropped)
        worker.start()
        self._worker = worker

    def _crop_task(self, emit_log, emit_progress):
        pcd = self.main_window.pcd
        bbox = self.get_bbox()
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([bbox[0], bbox[1], bbox[2]]),
            max_bound=np.array([bbox[3], bbox[4], bbox[5]]))
        cropped = pcd.crop(aabb)
        if len(cropped.points) == 0:
            emit_log("В области нет точек.", "WARNING")
            return False
        self.main_window.cropped_pcd = cropped
        self.main_window.cropped_pcd_path = str(
            WORKSPACE / "pointcloud_cropped.ply")
        o3d.io.write_point_cloud(
            self.main_window.cropped_pcd_path, cropped)
        emit_log(f"Обрезано: {len(cropped.points):,}", "SUCCESS")
        emit_progress(100)
        return True

    def _on_cropped(self, ok):
        if ok:
            self.viewer.set_pointcloud(self.main_window.cropped_pcd)


# ============================================================================
# Вкладка 4 — Датасет
# ============================================================================

class DatasetPrepareTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left.setMaximumWidth(500)
        ll = QVBoxLayout(left)
        info = QLabel(
            "Подготовка датасета: путь → добавление файлов → "
            "разбиение → расширение выборки")
        info.setWordWrap(True)
        ll.addWidget(info)

        pg = QGroupBox("1. Путь и структура")
        pl = QHBoxLayout()
        self.root_path = QLineEdit(str(DATASETS_DIR / "spb_city"))
        pl.addWidget(self.root_path)
        b = QPushButton("...")
        b.clicked.connect(self.browse_root)
        pl.addWidget(b)
        pg.setLayout(pl)
        ll.addWidget(pg)

        self.btn_create = QPushButton("Создать структуру")
        self.btn_create.clicked.connect(self.create_structure)
        ll.addWidget(self.btn_create)

        ag = QGroupBox("2. Добавление файлов")
        al = QVBoxLayout()
        self.btn_add_files = QPushButton("Добавить файлы")
        self.btn_add_files.clicked.connect(self.add_files)
        al.addWidget(self.btn_add_files)
        self.scene_list = QListWidget()
        self.scene_list.setMaximumHeight(150)
        al.addWidget(self.scene_list)
        ag.setLayout(al)
        ll.addWidget(ag)

        sg = QGroupBox("3. Разбиение выборки на части")
        sf = QFormLayout()
        self.train_ratio = QDoubleSpinBox()
        self.train_ratio.setRange(0.1, 0.95)
        self.train_ratio.setValue(0.7)
        self.train_ratio.setSingleStep(0.05)
        sf.addRow("Обучающая часть:", self.train_ratio)
        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setRange(0.05, 0.5)
        self.val_ratio.setValue(0.15)
        self.val_ratio.setSingleStep(0.05)
        sf.addRow("Валидационная часть:", self.val_ratio)
        sg.setLayout(sf)
        ll.addWidget(sg)

        self.btn_split = QPushButton(
            "Разбить на обучающую / валидационную / тестовую")
        self.btn_split.clicked.connect(self.split_dataset)
        ll.addWidget(self.btn_split)

        aug = QGroupBox("4. Расширение выборки")
        al2 = QFormLayout()
        self.rotations = QSpinBox()
        self.rotations.setRange(1, 72)
        self.rotations.setValue(8)
        al2.addRow("Число поворотов:", self.rotations)
        self.noise = QDoubleSpinBox()
        self.noise.setRange(0, 0.1)
        self.noise.setValue(0.005)
        self.noise.setDecimals(4)
        al2.addRow("Шум (м):", self.noise)
        self.dropout = QDoubleSpinBox()
        self.dropout.setRange(0, 0.5)
        self.dropout.setValue(0.05)
        self.dropout.setDecimals(3)
        al2.addRow("Доля удаляемых точек:", self.dropout)
        aug.setLayout(al2)
        ll.addWidget(aug)

        self.btn_augment = QPushButton("Расширить выборку")
        self.btn_augment.clicked.connect(self.run_augment)
        ll.addWidget(self.btn_augment)

        self.log = LogWidget()
        ll.addWidget(self.log)

        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        splitter.addWidget(scroll)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("Сводка"))
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(
            ["Параметр", "Значение"])
        self.summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        rl.addWidget(self.summary_table)
        splitter.addWidget(right)
        splitter.setSizes([550, 550])
        layout.addWidget(splitter)
        self.setLayout(layout)

    def browse_root(self):
        d = QFileDialog.getExistingDirectory(
            self, "Папка", str(DATASETS_DIR))
        if d:
            self.root_path.setText(d)

    def create_structure(self):
        root = Path(self.root_path.text())
        preparer = DatasetPreparer(root, self.log)
        preparer.create_structure()
        self.main_window.dataset_preparer = preparer
        self.main_window.dataset_root = str(root)
        self._update_summary()

    def add_files(self):
        if not hasattr(self.main_window, "dataset_preparer"):
            QMessageBox.warning(self, "Ошибка", "Создайте структуру.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "Облака", "",
            "Point Cloud (*.las *.laz *.ply *.pcd *.xyz *.e57 *.bin);;"
            "All Files (*)")
        if not files:
            return
        worker = WorkerThread(self._add_files_task, files)
        worker.log_signal.connect(self.log.log)
        worker.task_done.connect(self._on_files_added)
        worker.start()
        self._worker = worker

    def _add_files_task(self, emit_log, emit_progress, files):
        preparer = self.main_window.dataset_preparer
        preparer.log = EmitLogger(emit_log)
        for i, f in enumerate(files):
            try:
                preparer.add_scene(f)
                emit_log(f"Добавлено: {Path(f).name}", "SUCCESS")
            except Exception as e:
                emit_log(f"Ошибка {Path(f).name}: {e}", "ERROR")
            emit_progress(int(100 * (i + 1) / len(files)))
        return True

    def _on_files_added(self, ok):
        if ok:
            self._update_scene_list()
            self._update_summary()

    def _update_scene_list(self):
        self.scene_list.clear()
        if hasattr(self.main_window, "dataset_preparer"):
            for f in self.main_window.dataset_preparer.metadata[
                    "source_files"]:
                self.scene_list.addItem(Path(f).name)

    def split_dataset(self):
        if not hasattr(self.main_window, "dataset_preparer"):
            return
        train_r = self.train_ratio.value()
        val_r = self.val_ratio.value()
        if train_r + val_r >= 1.0:
            QMessageBox.warning(
                self, "Ошибка",
                "Сумма обучающей и валидационной частей должна быть "
                "меньше 1.0.\n\n"
                "Пример корректного разбиения:\n"
                "  обучающая      0.70\n"
                "  валидационная  0.15\n"
                "  тестовая       0.15  (остаток)")
            return
        worker = WorkerThread(self._split_task, train_r, val_r)
        worker.log_signal.connect(self.log.log)
        worker.task_done.connect(lambda ok: self._update_summary())
        worker.start()
        self._worker = worker

    def _split_task(self, emit_log, emit_progress, train_r, val_r):
        preparer = self.main_window.dataset_preparer
        preparer.log = EmitLogger(emit_log)
        emit_log("Разбиение выборки...", "INFO")
        preparer.split_dataset(train_r, val_r)
        emit_progress(100)
        return True

    def run_augment(self):
        if not hasattr(self.main_window, "dataset_preparer"):
            return
        worker = WorkerThread(self._augment_task)
        worker.log_signal.connect(self.log.log)
        worker.task_done.connect(lambda ok: self._update_summary())
        worker.start()
        self._worker = worker

    def _augment_task(self, emit_log, emit_progress):
        preparer = self.main_window.dataset_preparer
        preparer.log = EmitLogger(emit_log)
        preparer.augment(rotations=self.rotations.value(),
                         noise_std=self.noise.value(),
                         dropout_ratio=self.dropout.value())
        emit_progress(100)
        return True

    def _update_summary(self):
        if not hasattr(self.main_window, "dataset_preparer"):
            return
        s = self.main_window.dataset_preparer.summary()
        items = [("Корень датасета", s["root"]),
                 ("Сцен", str(s["n_scenes"])),
                 ("Всего точек", f"{s['n_points_total']:,}"),
                 ("Обучающих сцен", str(s["splits"].get("train", 0))),
                 ("Валидационных сцен", str(s["splits"].get("val", 0))),
                 ("Тестовых сцен", str(s["splits"].get("test", 0))),
                 ("Сцен после расширения", str(s["augmented"]))]
        self.summary_table.setRowCount(len(items))
        for i, (k, v) in enumerate(items):
            self.summary_table.setItem(i, 0, QTableWidgetItem(k))
            self.summary_table.setItem(i, 1, QTableWidgetItem(v))
            


# ============================================================================
# Константы: реестр моделей и пресеты конвейеров
# ============================================================================

MODELS_TEMPLATE = {
    "version": "1.0",
    "comment": "Реестр локальных AI-моделей City3D. Путь к весам — "
               "относительный или абсолютный.",
    "models": [
        {"id": "ransac", "method": "M1", "name": "CPU SMRF/CSF + ExtraTrees",
         "role": "geometry", "task": "Плоскости", "kind": "classic",
         "requires": ["open3d"], "weights": None,
         "trained_on": "—", "notes": "Без обучения. Быстрый baseline."},
        {"id": "randla_kitti", "method": "M1",
         "name": "RandLA-Net (SemanticKITTI)",
         "role": "semantic", "task": "3D-сегментация", "kind": "dl3d",
         "requires": ["torch", "open3d"],
         "weights": "models/M1/randlanet_semantickitti.pth",
         "trained_on": "SemanticKITTI (19 классов)",
         "notes": "Adapter/config/checkpoint не проверены; качество на проекте неизвестно."},
        {"id": "kpconv_s3dis", "method": "M1", "name": "KPConv (S3DIS)",
         "role": "semantic", "task": "3D-сегментация", "kind": "dl3d",
         "requires": ["torch"], "weights": "models/M1/kpconv_s3dis.pth",
         "trained_on": "S3DIS (13 классов)",
         "notes": "Для интерьеров. Городские сцены — RandLA."},
        {"id": "yolov8n", "method": "M2", "name": "YOLOv8n (Ultralytics)",
         "role": "detect", "task": "2D-детекция", "kind": "dl2d",
         "requires": ["ultralytics"], "weights": "yolov8n.pt",
         "trained_on": "COCO (80 классов)",
         "notes": "Скачивает веса при первом запуске."},
        {"id": "yoloworld_s", "method": "M2", "name": "YOLO-World-S",
         "role": "detect_open", "task": "Открытая детекция",
         "kind": "dl2d",
         "requires": ["ultralytics"],
         "weights": "yolov8s-worldv2.pt",
         "trained_on": "COCO + LVIS, zero-shot",
         "notes": "Использует classifier.txt как промпты."},
        {"id": "unet_buildings", "method": "M3", "name": "U-Net (smp)",
         "role": "semantic2d", "task": "Сегментация 2D", "kind": "dl2d",
         "requires": ["torch", "segmentation_models_pytorch"],
         "weights": "models/M3/unet_resnet34_buildings.pth",
         "trained_on": "WHU Building Dataset",
         "notes": "Требует дообучения на эталонах."},
        {"id": "geoai_buildings", "method": "M3",
         "name": "GeoAI Building Seg",
         "role": "semantic2d", "task": "Здания из GeoTIFF",
         "kind": "geoai",
         "requires": ["geoai", "rasterio", "torch"],
         "weights": "models/M3/geoai_buildings_v1.pth",
         "trained_on": "WHU Building Dataset",
         "notes": "opengeos/geoai + orthogonalize()."},
        {"id": "sam_vit_h", "method": "M3", "name": "SAM ViT-H (Meta)",
         "role": "interactive", "task": "Интерактивная разметка",
         "kind": "foundation",
         "requires": ["segment_anything", "torch"],
         "weights": "models/M3/sam_vit_h_4b8939.pth",
         "trained_on": "SA-1B", "notes": "Для ручной разметки."},
        {"id": "terrain_rf", "method": "M4", "name": "RandomForest DEM",
         "role": "feature", "task": "Классификация рельефа", "kind": "ml",
         "requires": ["sklearn", "rasterio"],
         "weights": "models/M4/rf_terrain.pkl",
         "trained_on": "—",
         "notes": "Признаки: slope, aspect, TPI, curvature."},
        {"id": "asprs_stat", "method": "M5", "name": "ASPRS-статистика",
         "role": "reference", "task": "Валидация меток",
         "kind": "classic",
         "requires": ["laspy", "numpy"], "weights": None,
         "trained_on": "ASPRS LAS 1.4",
         "notes": "Не ИИ. Использует classification из LAS."},
        {"id": "gis_vector", "method": "M6", "name": "GIS-вектор",
         "role": "reference", "task": "Кросс-валидация",
         "kind": "classic",
         "requires": ["geopandas"], "weights": None,
         "trained_on": "—", "notes": "SXF/SHP как эталон."},
        {"id": "lgbm_fuse", "method": "fuse", "name": "LightGBM fusion",
         "role": "fuse", "task": "Стеккинг M1–M6", "kind": "ml",
         "requires": ["lightgbm", "sklearn"],
         "weights": "models/ensemble/lgbm_v1.txt",
         "trained_on": "—", "notes": "Вход: признаки всех методов."},
    ],
}


PIPELINE_PRESETS = {
    "cpu_las": {"name":"CPU M1 → проверенный LAS", "steps":["M1:ransac", "export:LAS"], "hint":"Ground/CPU inference, без обещания LOD-реконструкции."},
    "qa_cpu": {"name":"QA меток → CPU M1 → LAS", "steps":["M5:asprs_stat","M1:ransac","export:LAS"], "hint":"Распределение исходных меток не является accuracy."},
    "terrain": {"name":"DEM: метрические признаки", "steps":["M4:terrain_rf"], "hint":"Slope/aspect/TPI/curvature; это признаки, не обученный RF."},
    "vectors": {"name":"GIS: нормализация GeoJSON", "steps":["M6:gis_vector"], "hint":"Сохранение объектов и provenance; SXF требует Panorama SDK."},
}


# ============================================================================
# Реестр локальных AI-моделей
# ============================================================================

class ModelRegistryWidget(QWidget):
    """Подвкладка «Реестр моделей»: таблица + карточка выбранной модели.

    Показывает:
      • список моделей из models.json
      • статус доступности (проверка модулей и файлов весов)
      • карточку с деталями и рекомендациями по роли в конвейере
    """

    def __init__(self, registry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.info_by_model = {}
        self._build_ui()
        self.reload()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel(
            "Локальные модели ИИ, доступные для методов M1–M6 и fusion:"))
        top.addStretch()
        b_reload = QPushButton("Проверить доступность")
        b_reload.clicked.connect(self.reload)
        top.addWidget(b_reload)
        b_open = QPushButton("Открыть models.json")
        b_open.clicked.connect(self._open_json)
        top.addWidget(b_open)
        layout.addLayout(top)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Метод", "ID", "Название", "Роль", "Задача",
            "Тип", "Веса", "Статус"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self._show_detail)
        layout.addWidget(self.table)

        gb = QGroupBox("Подробно")
        gl = QVBoxLayout(gb)
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setFont(QFont("Monospace", 9))
        self.detail.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; border:1px solid #3c3c3c;")
        gl.addWidget(self.detail)
        layout.addWidget(gb)

    def reload(self):
        report = self.registry.check_availability()
        self.info_by_model = {}
        self.table.setRowCount(len(report))
        for i, r in enumerate(report):
            m = r["model"]
            self.info_by_model[m["id"]] = r
            status = "OK готов" if r["ok"] else "! " + "; ".join(r["issues"])
            row = [
                m.get("method", ""), m.get("id", ""),
                m.get("name", ""), m.get("role", ""),
                m.get("task", ""), m.get("kind", ""),
                m.get("weights") or "—", status,
            ]
            for j, val in enumerate(row):
                it = QTableWidgetItem(str(val))
                if not r["ok"] and j == len(row) - 1:
                    it.setForeground(Qt.yellow)
                self.table.setItem(i, j, it)
        if report:
            self.table.selectRow(0)

    def _current_id(self):
        rows = self.table.selectionModel().selectedRows() \
            if self.table.selectionModel() else []
        if not rows:
            return None
        return self.table.item(rows[0].row(), 1).text()

    def _show_detail(self):
        mid = self._current_id()
        if not mid or mid not in self.info_by_model:
            return
        r = self.info_by_model[mid]
        m = r["model"]
        lines = [
            f"ID          : {m.get('id')}",
            f"Название    : {m.get('name')}",
            f"Метод       : {m.get('method')}",
            f"Роль в пайпе: {m.get('role')}",
            f"Задача      : {m.get('task')}",
            f"Тип         : {m.get('kind')}",
            f"Веса        : {m.get('weights') or '—'}",
            f"Обучена на  : {m.get('trained_on', '—')}",
            "",
            f"Зависимости : {', '.join(m.get('requires', []) or ['—'])}",
        ]
        for req, ok in r["requires_state"]:
            lines.append(f"   {'[+]' if ok else '[-]'} {req}")
        lines += [
            "",
            "Статус      : "
            + ("OK готов" if r["ok"]
               else "! " + "; ".join(r["issues"])),
            "",
            "Комментарий:",
            m.get("notes", "—"),
            "",
            "Рекомендации по порядку:",
            self._hint_for_role(m.get("role")),
        ]
        self.detail.setPlainText("\n".join(lines))

    @staticmethod
    def _hint_for_role(role):
        H = {
            "preprocess": "Всегда 1-м. Готовит данные для остальных.",
            "feature":    "До semantic/geometry — даёт признаки.",
            "geometry":   "После preprocess, до semantic.",
            "semantic":   "После geometry. Один semantic на конвейер.",
            "semantic2d": "После preprocess, желательно после M1.",
            "detect":     "Параллельно semantic, до fuse.",
            "detect_open": "Как detect, но требует classifier.txt.",
            "interactive": "Только для ручной разметки оператором.",
            "reference":  "Параллельно, для валидации/кросс-чека.",
            "fuse":       "После всех вычислительных шагов.",
            "export":     "Всегда последним.",
        }
        return H.get(role, "—")

    def _open_json(self):
        p = self.registry.path
        if not p.exists():
            QMessageBox.warning(self, "Ошибка", f"Файл не найден: {p}")
            return
        try:
            subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            QMessageBox.information(self, "Путь", f"{p}\n\n{e}")



class AIModelRegistry:
    """Загружает реестр моделей из models.json рядом с программой.

    Проверяет:
      • наличие python-модулей из 'requires'
      • наличие файла весов (если путь абсолютный или содержит '/')
    """

    def __init__(self, path=None, log_widget=None):
        self.path = Path(path) if path else (
            Path(_APP_SETTINGS.get("model_registry", str(Path(__file__).parent / "models.json"))))
        self.log = log_widget
        self.models = []
        self._load()
        self.models.append({"id":"city3d_unet_cpu","method":"M3","name":"City3D CPU U-Net baseline","role":"semantic2d","task":"Raster segmentation","kind":"ml","requires":["torch","rasterio"],"weights":"M3/city3d_tiny_unet.npz","trained_on":"Собственный reviewed dataset","notes":"Новый train/infer adapter; старые чужие U-Net weights не совместимы."})

    def _load(self):
        if not self.path.exists():
            self.models = list(MODELS_TEMPLATE["models"])
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.models = data.get("models", [])
            if self.log:
                self.log.log(
                    f"Реестр моделей: {len(self.models)} записей "
                    f"({self.path.name})", "SUCCESS")
        except Exception as e:
            if self.log:
                self.log.log(f"Ошибка models.json: {e}", "ERROR")

    def _write_template(self):
        try:
            self.path.write_text(
                json.dumps(MODELS_TEMPLATE, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as e:
            if self.log:
                self.log.log(f"Не удалось создать models.json: {e}",
                             "WARNING")

    def by_method(self, method):
        return [m for m in self.models if m.get("method") == method]

    def by_id(self, mid):
        for m in self.models:
            if m.get("id") == mid:
                return m
        return None

    @staticmethod
    def _check_module(name):
        import importlib.util
        try:
            ok=importlib.util.find_spec(name) is not None
            return ok, "" if ok else "нет модуля: " + name
        except Exception as e:
            return False, "ошибка проверки зависимости: " + type(e).__name__

    def _check_weights(self, w):
        if not w:return True, ""
        root=Path(_APP_SETTINGS.get("models_root", str(PROJECTS_DIR / "city3d_models")))
        p=Path(w).expanduser()
        if not p.is_absolute():p=root/str(p).removeprefix("models/")
        return p.is_file(), "" if p.is_file() else "нет локальных весов: " + str(p)

    def check_availability(self):
        from city3d_backend.registry import inventory
        root=_APP_SETTINGS.get("models_root",str(PROJECTS_DIR / "city3d_models"))
        states={r["id"]:r for r in inventory(root)}
        report=[]
        for m in self.models:
            state=states.get(m.get("id"),{"status":"adapter_missing","reason":"Нет проверенного adapter", "infer_enabled":False,"missing_dependencies":[]})
            issues=[state["status"],state.get("reason","")]
            if state.get("checkpoint") and not state.get("checkpoint_exists"):issues.append("Нет локального checkpoint")
            report.append({"model":m,"ok":state["infer_enabled"],"issues":issues,"requires_state":[(r,r not in state["missing_dependencies"]) for r in m.get("requires",[])],"capability":state})
        return report



class PipelineContext:
    """Единое состояние конвейера."""

    def __init__(self, out_dir, log_fn=None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = log_fn or (lambda m, l="INFO": None)

        self.pointcloud_path = None
        self.m5_las_refs = []
        self.rasters = []
        self.dem_path = None
        self.gis_path = None

        self.step_results = []
        self.classification = None
        self.n_points = 0

        self.crs_epsg = None
        self.vertical_crs = None
        self.height_offset = 0.0

        self.best_classified_las = None
        self.best_classified_ply = None
        self.best_report = None

        self.metrics_agg = {
            "coverage_total": 0.0,
            "n_methods_ok": 0,
            "n_methods_total": 0,
            "quality_by_method": {},
        }

    def register(self, idx, role, mid, ok, step_dir, extra=None):
        rec = {"idx": idx, "role": role, "id": mid,
               "ok": bool(ok), "out_dir": str(step_dir)}
        if extra:
            rec.update(extra)
        self.step_results.append(rec)
        self.metrics_agg["n_methods_total"] += 1
        if ok:
            self.metrics_agg["n_methods_ok"] += 1

    def feed_classified(self, las_path, ply_path=None,
                        report_path=None, quality=None):
        if las_path and Path(las_path).exists():
            self.best_classified_las = str(las_path)
        if ply_path and Path(ply_path).exists():
            self.best_classified_ply = str(ply_path)
        if report_path and Path(report_path).exists():
            self.best_report = str(report_path)
        if quality is not None:
            self.metrics_agg["coverage_total"] = max(
                self.metrics_agg["coverage_total"], quality)

    def export_input(self):
        return (self.best_classified_las
                or self.best_classified_ply
                or self.pointcloud_path)

    def save_summary(self):
        import json as _j
        summary = {
            "timestamp": datetime.now().isoformat(),
            "out_dir": str(self.out_dir),
            "crs_epsg": self.crs_epsg,
            "vertical_crs": self.vertical_crs,
            "height_offset": self.height_offset,
            "sources": {
                "pointcloud": self.pointcloud_path,
                "m5_las": self.m5_las_refs,
                "rasters": self.rasters,
                "dem": self.dem_path,
                "gis": self.gis_path,
            },
            "steps": self.step_results,
            "metrics": self.metrics_agg,
            "best_classified_las": self.best_classified_las,
            "export_input": self.export_input(),
        }
        (self.out_dir / "pipeline_summary.json").write_text(
            _j.dumps(summary, ensure_ascii=False, indent=2,
                     default=str),
            encoding="utf-8")
        lines = [
            "=" * 70,
            "ОТЧЁТ КОНВЕЙЕРА",
            "=" * 70,
            f"Папка: {self.out_dir}",
            f"CRS: EPSG:{self.crs_epsg}  Vertical: {self.vertical_crs}",
            f"Δ Z: {self.height_offset:.3f} м",
            "",
            f"Шагов: {self.metrics_agg['n_methods_total']}",
            f"Успешно: {self.metrics_agg['n_methods_ok']}",
            "",
            "ПОШАГОВО:",
        ]
        for r in self.step_results:
            mark = "OK" if r["ok"] else "FAIL"
            lines.append(
                f"  [{mark}] {r['idx']:2d}. {r['role']:12s} "
                f"({r.get('id', '')}) -> {r['out_dir']}")
        (self.out_dir / "pipeline_report.txt").write_text(
            "\n".join(lines), encoding="utf-8")
        return summary


class PipelineComposer(QWidget):
    """Подвкладка «Конвейер»: сборка и запуск цепочки моделей."""

    ROLES_ORDER = ["preprocess", "feature", "geometry", "semantic",
                   "semantic2d", "detect", "detect_open",
                   "reference", "fuse", "export"]

    def __init__(self, registry, training_tab, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.training_tab = training_tab
        self._build_ui()
        self._load_presets()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- пресеты -----------------------------------------------------
        pgb = QGroupBox("Пресеты конвейеров")
        pgl = QHBoxLayout(pgb)
        self.preset_combo = QComboBox()
        for k, v in PIPELINE_PRESETS.items():
            self.preset_combo.addItem(v["name"], k)
        pgl.addWidget(self.preset_combo, 1)
        b_load = QPushButton("Загрузить пресет")
        b_load.clicked.connect(self._load_selected_preset)
        pgl.addWidget(b_load)
        layout.addWidget(pgb)

        # --- доступные модели -------------------------------------------
        avail_gb = QGroupBox("Доступные модели (двойной клик — добавить)")
        avail_l = QVBoxLayout(avail_gb)
        self.avail_list = QListWidget()
        self.avail_list.itemDoubleClicked.connect(
            lambda it: self._add_step_by_id(it.data(Qt.UserRole)))
        avail_l.addWidget(self.avail_list)
        layout.addWidget(avail_gb)

        # --- текущий конвейер -------------------------------------------
        mid = QHBoxLayout()

        pipe_gb = QGroupBox("Порядок выполнения (сверху вниз)")
        pipe_l = QVBoxLayout(pipe_gb)
        self.pipe_list = QListWidget()
        self.pipe_list.itemSelectionChanged.connect(self._update_hint)
        pipe_l.addWidget(self.pipe_list)

        ctrl = QHBoxLayout()
        for label, slot in [("▲", self._move_up), ("▼", self._move_down),
                            ("✕", self._remove_step),
                            ("+", self._add_blank)]:
            b = QPushButton(label)
            b.setFixedWidth(36)
            b.clicked.connect(slot)
            ctrl.addWidget(b)
        pipe_l.addLayout(ctrl)
        mid.addWidget(pipe_gb, 2)

        hint_gb = QGroupBox("Подсказка оператору")
        hint_l = QVBoxLayout(hint_gb)
        self.hint_text = QTextEdit()
        self.hint_text.setReadOnly(True)
        self.hint_text.setFont(QFont("Monospace", 9))
        self.hint_text.setStyleSheet(
            "background:#1e1e1e; color:#dcdcaa; border:1px solid #3c3c3c;")
        hint_l.addWidget(self.hint_text)
        mid.addWidget(hint_gb, 3)

        layout.addLayout(mid)

        # --- кнопки запуска ----------------------------------------------
        bl = QHBoxLayout()
        b_run = QPushButton("▶ Запустить конвейер")
        b_run.clicked.connect(self._run_pipeline)
        bl.addWidget(b_run)
        b_save = QPushButton("Сохранить конвейер…")
        b_save.clicked.connect(self._save_pipeline)
        bl.addWidget(b_save)
        b_loadf = QPushButton("Загрузить конвейер…")
        b_loadf.clicked.connect(self._load_pipeline_file)
        bl.addWidget(b_loadf)
        bl.addStretch()
        layout.addLayout(bl)

        self.log = LogWidget()
        layout.addWidget(self.log)

    # --- заполнение списка доступных ------------------------------------
    def _load_presets(self):
        report = self.registry.check_availability()
        self.avail_list.clear()
        for r in report:
            m = r["model"]
            mark = "✔" if r["ok"] else "⚠"
            label = (f"{mark} [{m.get('method')}] {m.get('name')}  "
                     f"— {m.get('role')}")
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, m.get("id"))
            if not r["ok"]:
                it.setForeground(Qt.yellow)
                it.setToolTip("; ".join(r["issues"]))
            self.avail_list.addItem(it)

    # --- операции над конвейером ----------------------------------------
    def _add_step_by_id(self, mid):
        m = self.registry.by_id(mid)
        if not m:
            return
        label = f"[{m.get('method')}] {m.get('name')}  ({m.get('role')})"
        it = QListWidgetItem(label)
        it.setData(Qt.UserRole, {"kind": "model",
                                 "method": m.get("method"),
                                 "id": m.get("id"),
                                 "role": m.get("role")})
        self.pipe_list.addItem(it)
        self._update_hint()

    def _add_blank(self):
        items = ["preprocess", "export"]
        for k in items:
            it = QListWidgetItem(k)
            it.setData(Qt.UserRole, {"kind": "builtin", "id": k,
                                     "role": "preprocess" if k == "preprocess"
                                             else "export"})
            self.pipe_list.addItem(it)
        self._update_hint()

    def _move_up(self):
        r = self.pipe_list.currentRow()
        if r > 0:
            it = self.pipe_list.takeItem(r)
            self.pipe_list.insertItem(r - 1, it)
            self.pipe_list.setCurrentRow(r - 1)
            self._update_hint()

    def _move_down(self):
        r = self.pipe_list.currentRow()
        if 0 <= r < self.pipe_list.count() - 1:
            it = self.pipe_list.takeItem(r)
            self.pipe_list.insertItem(r + 1, it)
            self.pipe_list.setCurrentRow(r + 1)
            self._update_hint()

    def _remove_step(self):
        r = self.pipe_list.currentRow()
        if r >= 0:
            self.pipe_list.takeItem(r)
            self._update_hint()

    # --- подсказка ------------------------------------------------------
    def _update_hint(self):
        lines = []
        prev_roles = []
        for i in range(self.pipe_list.count()):
            it = self.pipe_list.item(i)
            d = it.data(Qt.UserRole) or {}
            role = d.get("role", "?")
            lines.append(f"{i+1:2d}. {role:12s}  {it.text()}")
            if prev_roles and role == "semantic" and \
                    "semantic" in prev_roles:
                lines.append("    ⚠ Два semantic-метода подряд — "
                             "возможен конфликт классов. Вставь fuse "
                             "или оставь один.")
            if role == "export" and i < self.pipe_list.count() - 1:
                lines.append("    ⚠ export не последний — "
                             "переставь его в конец.")
            prev_roles.append(role)

        if self.pipe_list.count() == 0:
            lines = ["Конвейер пуст. Двойной клик по модели слева — "
                     "добавить шаг.",
                     "",
                     "Типовой минимум: preprocess → M1 → export"]

        lines.append("")
        lines.append("Порядок ролей (рекомендация):")
        lines.append("  " + " → ".join(self.ROLES_ORDER))
        self.hint_text.setPlainText("\n".join(lines))

    # --- пресеты --------------------------------------------------------
    def _load_selected_preset(self):
        key = self.preset_combo.currentData()
        preset = PIPELINE_PRESETS.get(key)
        if not preset:
            return
        self.pipe_list.clear()
        for step in preset["steps"]:
            if ":" in step:
                method, mid = step.split(":", 1)
                m = self.registry.by_id(mid)
                if m:
                    self._add_step_by_id(mid)
                    continue
            # builtin
            it = QListWidgetItem(step)
            it.setData(Qt.UserRole, {
                "kind": "builtin", "id": step,
                "role": "export" if step.startswith("export")
                        else "fuse" if step == "fuse"
                        else "preprocess" if step == "preprocess"
                        else "?"})
            self.pipe_list.addItem(it)
        self._update_hint()
        self.log.log(f"Загружен пресет: {preset['name']} — "
                     f"{preset['hint']}", "SUCCESS")

    # --- сохранение/загрузка --------------------------------------------
    def _save_pipeline(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить конвейер",
            str(Path.home() / "pipeline.json"), "JSON (*.json)")
        if not path:
            return
        steps = []
        for i in range(self.pipe_list.count()):
            it = self.pipe_list.item(i)
            steps.append({"text": it.text(),
                          "data": it.data(Qt.UserRole)})
        Path(path).write_text(json.dumps(
            {"steps": steps, "timestamp": datetime.now().isoformat()},
            ensure_ascii=False, indent=2), encoding="utf-8")
        self.log.log(f"Сохранено: {path}", "SUCCESS")

    def _load_pipeline_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить конвейер", str(Path.home()),
            "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.pipe_list.clear()
            for s in data.get("steps", []):
                it = QListWidgetItem(s.get("text", "?"))
                it.setData(Qt.UserRole, s.get("data", {}))
                self.pipe_list.addItem(it)
            self._update_hint()
            self.log.log(f"Загружено: {path}", "SUCCESS")
        except Exception as e:
            self.log.log(f"Ошибка загрузки: {e}", "ERROR")

    # --- запуск ---------------------------------------------------------
    def _run_pipeline(self):
        if self.pipe_list.count() == 0:
            QMessageBox.warning(self, "Ошибка", "Конвейер пуст.")
            return

        steps = []
        for i in range(self.pipe_list.count()):
            it = self.pipe_list.item(i)
            d = it.data(Qt.UserRole) or {}
            steps.append({"index": i + 1, "label": it.text(),
                          "data": d})

        warns = []
        for i, s in enumerate(steps):
            role = s["data"].get("role")
            if role == "export" and i < len(steps) - 1:
                warns.append(f"export на шаге {s['index']} не последний")
            if (i > 0 and role == "semantic"
                    and steps[i-1]["data"].get("role") == "semantic"):
                warns.append(
                    f"два semantic подряд: "
                    f"{steps[i-1]['index']}, {s['index']}")
        if warns:
            ans = QMessageBox.question(
                self, "Порядок вызывает вопросы",
                "Найдены проблемы:\n• " + "\n• ".join(warns)
                + "\n\nПродолжить?",
                QMessageBox.Yes | QMessageBox.No)
            if ans != QMessageBox.Yes:
                return

        out_dir = TRAINING_OUTPUT_DIR / (
            f"pipeline_{datetime.now():%Y%m%d-%H%M%S}")
        out_dir.mkdir(parents=True, exist_ok=True)

        self.log.log("=" * 60, "INFO")
        self.log.log(f"ЗАПУСК КОНВЕЙЕРА: {len(steps)} шагов", "INFO")
        self.log.log(f"Выходная папка: {out_dir}", "INFO")
        self.log.log("=" * 60, "INFO")

        tt = self.training_tab
        cd = getattr(tt.main_window, "crs_detector", None)
        snapshot = {
            "pointcloud_path": str(getattr(tt.main_window, "pcd_path", "") or ""),
            "m5_las_refs": [tt.m5_las_list.item(i).text() for i in range(tt.m5_las_list.count())],
            "rasters": [tt.m2_raster_path.text()] if tt.m2_raster_path.text() else [],
            "dem_path": tt.m4_dem_path.text(), "gis_path": tt.m6_gis_path.text(),
            "crs_epsg": getattr(cd, "crs", None), "vertical_crs": None, "height_offset": 0,
            "m1_options": dict(_APP_SETTINGS.get("m1", {}),
                ground_backend=tt.m1_ground.currentText(), model=tt.m1_cpu_model.text().strip(),
                preserve_classes=tt.m1_preserve.isChecked()),
        }
        if any(x["data"].get("role") in ("geometry","feature") for x in steps):
            if QMessageBox.question(self, "Единицы координат", "Подтвердите единицы XY/Z: метры.",
                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
            snapshot["m1_options"]["units"] = "metres"
        lt=tt.main_window.tabs.widget(0)
        snapshot["vertical_crs"]=lt.vertical_crs_combo.currentData()
        snapshot["height_offset"]=lt.height_offset_spin.value()
        radius,neighbors=[(0.6,16),(1.0,32),(1.5,64)][tt.m1_profile.currentIndex()]
        snapshot["m1_options"].update(scanning_type=getattr(tt.main_window,"scanning_type","unknown"),radius=radius,neighbors=neighbors)
        from city3d_backend.core import sha256, VALID_CLASSES
        classes=tt._load_classifier()
        if classes:snapshot["m1_options"]["allowed_classes"]=tuple(c["code"] for c in classes if c["code"] in VALID_CLASSES)
        cf=Path(__file__).parent/"classifier.txt"
        snapshot["classifier_sha256"]=sha256(cf) if cf.exists() else None
        snapshot["m3_image"]=tt.m3_image.text().strip()
        snapshot["m3_checkpoint"]=tt.m3_checkpoint.text().strip()
        snapshot["models_root"]=_APP_SETTINGS.get("models_root",str(PROJECTS_DIR/"city3d_models"))
        snapshot["database"]=dict(DB_PARAMS)
        worker = WorkerThread(self._pipeline_task_real, steps,
                              str(out_dir), snapshot)
        worker.log_signal.connect(self.log.log)
        worker.detail_signal.connect(
            lambda s: self.log.log(s, "INFO"))
        worker.progress.connect(
            lambda p: self.log.log(f"progress {p}%", "INFO"))
        worker.task_done.connect(
            lambda ok: self.log.log(
                "КОНВЕЙЕР ЗАВЕРШЁН" if ok else "КОНВЕЙЕР ПРОВАЛЕН",
                "SUCCESS" if ok else "ERROR"))
        worker.start()
        self._worker = worker

    def _pipeline_task_real(self, emit_log, emit_progress, steps, out_dir, snapshot=None):
        from city3d_backend.runner import PipelineRunner, Task
        from city3d_backend.registry import DEFINITIONS
        from city3d_backend.database import LocalJournal, Repository
        snapshot=snapshot or {}
        tasks=[];previous=None;classified=None
        for i,step in enumerate(steps):
            d=step["data"];mid=d.get("id","");task_id=f"step_{i+1}"
            dependencies=(previous,) if previous else ()
            adapter=DEFINITIONS.get(mid,(None,))[0]
            inputs={};params={};mode="infer"
            if adapter=="m1.cpu":
                inputs={"asset":snapshot.get("pointcloud_path","")};params={"config":snapshot.get("m1_options",{})};classified=task_id
            elif adapter=="m4.features":
                inputs={"asset":snapshot.get("dem_path","")};params={"units":"metres"}
            elif adapter=="m5.qa":
                refs=snapshot.get("m5_las_refs",[])
                inputs={"asset":refs[0] if refs else snapshot.get("pointcloud_path","")};mode="evaluate"
            elif adapter=="m6.vector":inputs={"asset":snapshot.get("gis_path","")}
            elif adapter=="m3.raster":
                inputs={"asset":snapshot.get("m3_image","")};params={"checkpoint":snapshot.get("m3_checkpoint","")}
            elif adapter=="yolo.detect":
                rasters=snapshot.get("rasters",[]);inputs={"asset":rasters[0] if rasters else ""}
                model=self.registry.by_id(mid) or {};weight=Path(model.get("weights", ""))
                if not weight.is_absolute():weight=Path(snapshot["models_root"])/str(weight).removeprefix("models/")
                params={"checkpoint":str(weight),"model_id":mid}
            elif mid=="export:LAS" or (d.get("role")=="export" and mid in ("export","LAS")):
                adapter="export.las";mode="export"
                inputs={"asset":{"task":classified,"output":"classified"}} if classified else {}
                dependencies=tuple(dict.fromkeys(dependencies+((classified,) if classified else ())))
            # Unimplemented methods remain explicit in preflight; no role/string guessing.
            tasks.append(Task(task_id,adapter or mid,mode,dependencies,inputs,params));previous=task_id
        journal=LocalJournal(Path(out_dir)/"outbox.sqlite")
        runner=PipelineRunner(journal=journal,repository=Repository({"database":snapshot.get("database",{})}))
        result=runner.run(tasks,out_dir,
            cancel=lambda:QThread.currentThread().isInterruptionRequested(),progress=emit_progress,
            spatial={"vertical_crs":snapshot.get("vertical_crs"),"height_offset":snapshot.get("height_offset"),
                     "classifier_sha256":snapshot.get("classifier_sha256"),"ontology_version":"city3d:1"})
        emit_log("Конвейер: "+result["status"]+"; регистрация БД: "+result["db_registration"],"SUCCESS" if result["status"]=="success" else "ERROR")
        for key,state in result["steps"].items():emit_log(key+": "+json.dumps(state,ensure_ascii=False),"INFO")
        return result["status"]=="success"


class TrainingTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.sub_tabs = QTabWidget()

        # --- подвкладка 1: одиночная модель (существующая логика) -----
        # PATCHED_LAYOUT_FIX: оборачиваем в QScrollArea, чтобы при
        # нехватке вертикального места виджеты не накладывались
        single_container = QWidget()
        single_container_layout = QVBoxLayout(single_container)
        single_container_layout.setContentsMargins(0, 0, 0, 0)

        single_scroll = QScrollArea()
        single_scroll.setWidgetResizable(True)
        single_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        single = QWidget()
        layout = QVBoxLayout(single)
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Метод:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "M1 — Векторизация облака",
            "M2 — Обучение по растрам",
            "M3 — Обучение по маскам",
            "M4 — Обучение по ЦМР",
            "M5 — Классифицированные облака",
            "M6 — SXF/Shape",
            "Комбинирование методов"
        ])
        self.method_combo.currentIndexChanged.connect(
            self._switch_method)
        method_layout.addWidget(self.method_combo, 1)
        layout.addLayout(method_layout)

        self.method_stack = QStackedWidget()
        self.method_stack.addWidget(self._create_m1_page())
        self.method_stack.addWidget(self._create_m2_page())
        self.method_stack.addWidget(self._create_m3_page())
        self.method_stack.addWidget(self._create_m4_page())
        self.method_stack.addWidget(self._create_m5_page())
        self.method_stack.addWidget(self._create_m6_page())
        self.method_stack.addWidget(self._create_combined_page())
        layout.addWidget(self.method_stack)
        layout.addWidget(self._create_params_widget())

        # PATCHED_LAYOUT_FIX: завершаем обёртку scroll area
        single_scroll.setWidget(single)
        single_container_layout.addWidget(single_scroll)
        self.sub_tabs.addTab(single_container, "Одиночная модель")

        # --- подвкладка 2: реестр моделей ------------------------------
        try:
            self.model_registry = AIModelRegistry(
                log_widget=getattr(self.main_window, "log_widget", None))
            self.registry_widget = ModelRegistryWidget(self.model_registry)
            self.sub_tabs.addTab(self.registry_widget, "Реестр моделей")
        except Exception as e:
            import traceback
            print("[WARN] Реестр моделей не создан:", e, file=sys.stderr)
            traceback.print_exc()
            self.model_registry = None
            self.registry_widget = None

        # --- подвкладка 3: конвейер ------------------------------------
        if self.model_registry is not None:
            self.pipeline_widget = PipelineComposer(
                self.model_registry, self)
            self.sub_tabs.addTab(self.pipeline_widget, "Конвейер")

        outer.addWidget(self.sub_tabs)

    def _switch_method(self, idx):
        self.method_stack.setCurrentIndex(idx)

    def _create_m1_page(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel(
            "M1: SMRF/CSF → HAG → геометрические признаки → CPU ExtraTrees. Без модели распознаётся земля; остальные объекты требуют разметки."))
        g = QGroupBox("Параметры M1")
        f = QFormLayout()
        self.m1_primitive = QComboBox()
        self.m1_primitive.addItems([
            "Плоскость", "Ребро", "Узел", "Линия",
            "Куб", "Сфера", "Полусфера", "Труба",
            "Столб", "Дорожный знак", "Другое"])
        f.addRow("Примитив:", self.m1_primitive)
        self.m1_ransac_threshold = QDoubleSpinBox()
        self.m1_ransac_threshold.setRange(0.001, 1.0)
        self.m1_ransac_threshold.setValue(0.05)
        self.m1_ransac_threshold.setDecimals(3)
        f.addRow("RANSAC threshold:", self.m1_ransac_threshold)
        self.m1_primitive.setEnabled(False)
        self.m1_ransac_threshold.setEnabled(False)
        self.m1_ground = QComboBox()
        self.m1_ground.addItems(["smrf", "csf", "trusted", "legacy"])
        self.m1_ground.setCurrentText(_APP_SETTINGS.get("m1", {}).get("ground_backend", "smrf"))
        f.addRow("Ground backend:", self.m1_ground)
        self.m1_cpu_model = QLineEdit(_APP_SETTINGS.get("m1", {}).get("model", ""))
        self.m1_cpu_model.setPlaceholderText("Локальная доверенная модель .joblib; обучение через city3d-cli train")
        f.addRow("CPU модель:", self.m1_cpu_model)
        self.m1_profile = QComboBox()
        self.m1_profile.addItems(["Быстро", "Сбалансированно", "Точно"])
        self.m1_profile.setCurrentIndex(1)
        f.addRow("Признаки:", self.m1_profile)
        self.m1_preserve = QCheckBox("Сохранять существующие классы кроме 0/1")
        self.m1_preserve.setChecked(True)
        f.addRow(self.m1_preserve)
        f.addRow(QLabel("Словарь классов: class_schema.json; коды 8/12 зарезервированы. Score не является accuracy."))

        g.setLayout(f)
        layout.addWidget(g)
        btn = QPushButton("Запустить M1")
        btn.clicked.connect(lambda: self._run_method("M1"))
        layout.addWidget(btn)
        layout.addStretch()
        return w

    def _create_m2_page(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("M2: анализ растра / inference; обучение не реализовано."))
        g = QGroupBox("Растр")
        gl = QHBoxLayout()
        self.m2_raster_path = QLineEdit()
        self.m2_raster_path.setPlaceholderText("GeoTiff / JPEG / PNG; detections имеют pixel coordinates")
        gl.addWidget(self.m2_raster_path)
        b = QPushButton("Обзор...")
        b.clicked.connect(self._browse_m2)
        gl.addWidget(b)
        g.setLayout(gl)
        layout.addWidget(g)
        m2_form=QFormLayout()
        self.m2_model_id=QComboBox();self.m2_model_id.addItems(["yolov8n","yoloworld_s"])
        m2_form.addRow("Model ID:",self.m2_model_id)
        self.m2_checkpoint=QLineEdit();self.m2_checkpoint.setPlaceholderText("Существующий локальный .pt; автоматической загрузки нет")
        m2_form.addRow("Checkpoint:",self.m2_checkpoint)
        layout.addLayout(m2_form)
        self.m2_model_id.setToolTip("YOLO-World требует проверенного offline text encoder; пока adapter blocked, подмена на YOLOv8 запрещена")
        btn = QPushButton("Запустить M2")
        btn.clicked.connect(lambda: self._run_method("M2"))
        layout.addWidget(btn)
        layout.addStretch()
        return w

    def _create_m3_page(self):
        w=QWidget();layout=QVBoxLayout(w)
        label=QLabel("M3: реальное обучение CPU U-Net по GeoTIFF image/mask. Классы и bands задаёт manifest; стартовые веса не скачиваются.")
        label.setWordWrap(True);layout.addWidget(label)
        form=QFormLayout();self.m3_manifest=QLineEdit();self.m3_manifest.setPlaceholderText("training.json: gold labels, независимые train/val территории")
        form.addRow("Training manifest:",self.m3_manifest)
        self.m3_checkpoint=QLineEdit();self.m3_checkpoint.setPlaceholderText("Локальная проверенная модель .npz и .json")
        form.addRow("Checkpoint:",self.m3_checkpoint)
        self.m3_finetune=QCheckBox("Дообучить выбранный checkpoint в НОВУЮ версию");form.addRow(self.m3_finetune)
        self.m3_image=QLineEdit();self.m3_image.setPlaceholderText("GeoTIFF с совместимыми bands");form.addRow("Raster inference:",self.m3_image)
        self.m3_layer=QComboBox();self.m3_layer.addItem("Классы из manifest/checkpoint");self.m3_layer.setEnabled(False);form.addRow("Class schema:",self.m3_layer)
        layout.addLayout(form)
        b=QPushButton("Обучить / дообучить M3");b.clicked.connect(lambda:self._run_method("M3"));layout.addWidget(b)
        b=QPushButton("Применить M3 к растру");b.clicked.connect(self._run_m3_infer);layout.addWidget(b)
        layout.addWidget(QLabel("Формат manifest и пример: resources/raster_training_example.json. Выход: model.npz + training report либо georeferenced labels.tif."))
        layout.addStretch();return w

    def _create_m4_page(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("M4: анализ ЦМР."))
        g = QGroupBox("DEM")
        gl = QHBoxLayout()
        self.m4_dem_path = QLineEdit()
        self.m4_dem_path.setPlaceholderText("GeoTiff DEM / DSM")
        gl.addWidget(self.m4_dem_path)
        b = QPushButton("Обзор...")
        b.clicked.connect(self._browse_m4)
        gl.addWidget(b)
        g.setLayout(gl)
        layout.addWidget(g)
        btn = QPushButton("Запустить M4")
        btn.clicked.connect(lambda: self._run_method("M4"))
        layout.addWidget(btn)
        layout.addStretch()
        return w

    def _create_m5_page(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel(
            "M5: анализ ASPRS-классов. Можно загрузить несколько LAS.\n"
            "Каждый файл автоматически регистрируется в базе эталонов "
            "(вкладка 7)."))
        g = QGroupBox("LAS-файлы (M5_LAS_LIST)")
        gl = QVBoxLayout()
        self.m5_las_list = QListWidget()
        self.m5_las_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection)
        self.m5_las_list.setToolTip(
            "Список загруженных LAS. Можно добавлять несколько.")
        gl.addWidget(self.m5_las_list)
        row = QHBoxLayout()
        b_add = QPushButton("Добавить LAS...")
        b_add.clicked.connect(self._browse_m5)
        row.addWidget(b_add)
        b_remove = QPushButton("Удалить выделенные")
        b_remove.clicked.connect(self._remove_m5_las)
        row.addWidget(b_remove)
        b_clear = QPushButton("Очистить")
        b_clear.clicked.connect(self._clear_m5_las)
        row.addWidget(b_clear)
        row.addStretch()
        gl.addLayout(row)
        g.setLayout(gl)
        layout.addWidget(g)
        btn = QPushButton("Запустить M5")
        btn.clicked.connect(lambda: self._run_method("M5"))
        layout.addWidget(btn)
        layout.addStretch()
        return w

    def _remove_m5_las(self):
        for item in self.m5_las_list.selectedItems():
            self.m5_las_list.takeItem(self.m5_las_list.row(item))

    def _clear_m5_las(self):
        self.m5_las_list.clear()

    def _create_m6_page(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("M6: SXF/Shape."))
        g = QGroupBox("ГИС")
        gl = QHBoxLayout()
        self.m6_gis_path = QLineEdit()
        self.m6_gis_path.setPlaceholderText("SXF / SHP / GeoPackage")
        gl.addWidget(self.m6_gis_path)
        b = QPushButton("Обзор...")
        b.clicked.connect(self._browse_m6)
        gl.addWidget(b)
        g.setLayout(gl)
        layout.addWidget(g)
        btn = QPushButton("Запустить M6")
        btn.clicked.connect(lambda: self._run_method("M6"))
        layout.addWidget(btn)
        layout.addStretch()
        return w

    def _create_combined_page(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Комбинирование M1-M6 с весами."))
        g = QGroupBox("Методы и веса")
        gl = QVBoxLayout()
        self.combo_checks = {}
        self.combo_weights = {}
        for m in ["M1", "M2", "M3", "M4", "M5", "M6"]:
            row = QHBoxLayout()
            cb = QCheckBox(m)
            row.addWidget(cb)
            row.addWidget(QLabel("вес:"))
            sp = QDoubleSpinBox()
            sp.setRange(0, 10)
            sp.setValue(1.0)
            sp.setSingleStep(0.1)
            row.addWidget(sp)
            row.addStretch()
            gl.addLayout(row)
            self.combo_checks[m] = cb
            self.combo_weights[m] = sp
        g.setLayout(gl)
        layout.addWidget(g)
        btn = QPushButton("Запустить комбинирование")
        btn.clicked.connect(self._run_combined)
        layout.addWidget(btn)
        layout.addStretch()
        return w

    def _create_params_widget(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        mg = QGroupBox("Параметры")
        ml = QFormLayout()
        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Путь к датасету")
        # PATCHED_LAYOUT_FIX: Обзор в одну строку с полем Датасет,
        # иначе кнопка налезает на Эпох/Классов при сжатии
        _ds_row = QHBoxLayout()
        _ds_row.setContentsMargins(0, 0, 0, 0)
        _ds_row.addWidget(self.dataset_path, 1)
        _browse_btn = QPushButton("Обзор...")
        _browse_btn.setMaximumWidth(90)
        _browse_btn.clicked.connect(self.browse_dataset)
        _ds_row.addWidget(_browse_btn)
        ml.addRow("Датасет:", _ds_row)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(50)
        ml.addRow("Эпох:", self.epochs)
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 100)
        self.num_classes.setEnabled(False)
        self.num_classes.setToolTip("Число классов задаёт версионный manifest/checkpoint; старый числовой override отключён")
        self.num_classes.setValue(8)
        ml.addRow("Классов:", self.num_classes)
        mg.setLayout(ml)
        layout.addWidget(mg)
        self.progress = QProgressBar()
        self.progress.setFormat("%p%  —  ожидание")
        layout.addWidget(self.progress)
        self.eta_label = QLabel("ETA: —")
        self.eta_label.setStyleSheet(
            "color: #dcdcaa; font-family: monospace; padding: 2px;")
        layout.addWidget(self.eta_label)
        # ── ПАНЕЛЬ МОНИТОРИНГА ────────────────────────────────
        mon_gb = QGroupBox("Мониторинг качества (live)")
        mon_l = QVBoxLayout(mon_gb)
        self.monitor_label = QLabel("Ожидание запуска...")
        self.monitor_label.setWordWrap(True)
        self.monitor_label.setFont(QFont("Monospace", 9))
        self.monitor_label.setStyleSheet(
            "color: #dcdcaa; padding: 6px; background: #1a1a1a; "
            "border-radius: 4px;")
        mon_l.addWidget(self.monitor_label)

        btn_row = QHBoxLayout()
        self.btn_stop_training = QPushButton("⛔ Остановить обучение")
        self.btn_stop_training.setStyleSheet(
            "background-color: #a03030; color: white; font-weight: bold;")
        self.btn_stop_training.setEnabled(False)
        self.btn_stop_training.clicked.connect(self._stop_training)
        btn_row.addWidget(self.btn_stop_training)
        btn_row.addStretch()
        mon_l.addLayout(btn_row)
        layout.addWidget(mon_gb)

        self.result_label = QLabel("Результат последнего прогона: —")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            "color: #4ec9b0; padding: 6px; background: #252526; "
            "border-radius: 4px;")
        layout.addWidget(self.result_label)
        self.log = LogWidget()
        layout.addWidget(self.log)
        return w

    def _browse_m2(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Растр", "",
            "Raster (*.tif *.tiff *.jpg *.png *.pdf);;All Files (*)")
        if p:
            self.m2_raster_path.setText(p)

    def _browse_m4(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "DEM", "", "GeoTiff (*.tif *.tiff);;All Files (*)")
        if p:
            self.m4_dem_path.setText(p)

    def _browse_m5(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "LAS (можно несколько)", "",
            "LAS (*.las *.laz);;All Files (*)")
        for p in paths:
            # не дублируем
            existing = [self.m5_las_list.item(i).text()
                        for i in range(self.m5_las_list.count())]
            if p not in existing:
                self.m5_las_list.addItem(p)

    def _browse_m6(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "ГИС", "",
            "GIS (*.sxf *.shp *.gpkg *.geojson);;All Files (*)")
        if p:
            self.m6_gis_path.setText(p)

    def browse_dataset(self):
        d = QFileDialog.getExistingDirectory(
            self, "Датасет", str(DATASETS_DIR))
        if d:
            self.dataset_path.setText(d)

    def _make_run_dir(self, method):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        d = TRAINING_OUTPUT_DIR / f"{stamp}_{method}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _check_references(self, method_name):
        """Проверяет базу эталонов перед запуском обучения.

        Возвращает dict с информацией или None. Диалог не блокирует —
        только информирует оператора.
        """
        hdb = getattr(self.main_window, "history_db", None)
        if not hdb or not hdb.is_available:
            return None
        crs = None
        bbox = None
        if getattr(self.main_window, "crs_detector", None):
            crs = self.main_window.crs_detector.crs
        if self.main_window.pcd is not None:
            pts = np.asarray(self.main_window.pcd.points)
            bbox = (float(pts[:, 0].min()), float(pts[:, 1].min()),
                    float(pts[:, 2].min()), float(pts[:, 0].max()),
                    float(pts[:, 1].max()), float(pts[:, 2].max()))
        try:
            with hdb.conn.cursor() as cur:
                if crs:
                    cur.execute(f"""
                        SELECT id, name, path, scanning_type, crs_epsg
                        FROM {hdb.schema_r}.references_clouds
                        WHERE crs_epsg = %s
                        ORDER BY created_at DESC LIMIT 10
                    """, (crs,))
                else:
                    cur.execute(f"""
                        SELECT id, name, path, scanning_type, crs_epsg
                        FROM {hdb.schema_r}.references_clouds
                        ORDER BY created_at DESC LIMIT 10
                    """)
                rows = cur.fetchall()
            if not rows:
                self.log.log(
                    f"{method_name}: в базе эталонов нет записей для "
                    f"CRS EPSG:{crs}. Добавьте эталоны на вкладке "
                    f"«7. База эталонов» для повышения качества "
                    f"обучения.", "WARNING")
                return None
            self.log.log(
                f"{method_name}: найдено эталонов: {len(rows)} "
                f"(используются для валидации и переноса обучения).",
                "SUCCESS")
            for r in rows:
                self.log.log(
                    f"    ref#{r[0]}  {r[1]}  [{r[2]}]  "
                    f"type={r[3]}  crs=EPSG:{r[4]}", "INFO")
            return {"n_refs": len(rows), "crs": crs}
        except Exception as e:
            self.log.log(f"{method_name}: проверка эталонов: {e}",
                         "WARNING")
            return None

    def _load_classifier(self):
        """Читает classifier.txt рядом с программой.

        Формат: одна строка = один класс.
           <код>  <имя>   [<мин_точек>]   [# комментарий]
        Пример:
           2  ground          5000
           6  building        2000
           3  vegetation_low  1000
        """
        path = Path(__file__).parent / "classifier.txt"
        classes = []
        if not path.exists():
            return classes
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                code = int(parts[0])
            except ValueError:
                continue
            name = parts[1]
            min_pts = int(parts[2]) if len(parts) >= 3 else 0
            classes.append({"code": code, "name": name,
                            "min_points": min_pts})
        return classes


    def _stop_training(self):
        """Запрос на остановку текущего обучения."""
        w = getattr(self, "_worker", None)
        if w is None or not w.isRunning():
            return
        try:
            w.requestInterruption()
            self.log.log("⛔ Запрошена остановка обучения. "
                         "Ожидание завершения текущего шага...",
                         "WARNING")
            self.monitor_label.setText(
                "⛔ ОСТАНОВКА... дождитесь завершения текущего шага")
            self.btn_stop_training.setEnabled(False)
        except Exception as e:
            self.log.log(f"Ошибка остановки: {e}", "ERROR")

    def _on_worker_detail(self, msg):
        """Обработчик detail_signal от WorkerThread — обновляет монитор."""
        if not hasattr(self, "monitor_label"):
            return
        self.monitor_label.setText(msg)
        # Эвристика: если в сообщении — "quality" или "кач", подсвечиваем
        low = msg.lower()
        if "quality_score" in low or "качество" in low:
            self.monitor_label.setStyleSheet(
                "color: #4ec9b0; padding: 6px; background: #1a1a1a; "
                "border-radius: 4px;")

    def _run_method(self, method_name):
        # F. Проверка эталонов до запуска
        try:
            self._check_references(method_name)
        except Exception as e:
            self.log.log(f"Проверка эталонов: {e}", "WARNING")

        if method_name == "M1":
            src = getattr(self.main_window, "pcd_path", None) or ""
        elif method_name == "M2":
            src = self.m2_raster_path.text()
        elif method_name == "M3":
            src = self.m3_manifest.text().strip()
        elif method_name == "M4":
            src = self.m4_dem_path.text()
        elif method_name == "M5":
            files = [self.m5_las_list.item(i).text()
                     for i in range(self.m5_las_list.count())]
            if not files:
                QMessageBox.warning(self, "Ошибка",
                                    "Добавьте хотя бы один LAS")
                return
            # Регистрируем каждый файл в БД (вкладка 7)
            hdb0 = getattr(self.main_window, "history_db", None)
            if hdb0 and hdb0.is_available:
                for p in files:
                    try:
                        self._register_las_in_refs(p)
                    except Exception as e:
                        self.log.log(
                            f"Регистрация {Path(p).name}: {e}",
                            "WARNING")
            # Запускаем обработку по каждому файлу последовательно
            output_dir = self._make_run_dir("M5")
            st = getattr(self.main_window, "scanning_type", "unknown")
            hyper = {"ransac_threshold": self.m1_ransac_threshold.value()}
            worker = WorkerThread(self._train_multi_m5_task,
                                  files, str(output_dir), hyper)
            worker.log_signal.connect(self.log.log)
            worker.detail_signal.connect(self.eta_label.setText)
            worker.progress.connect(self.progress.setValue)
            worker.task_done.connect(self._on_training_done)
            worker.start()
            self._worker = worker
            return
        elif method_name == "M6":
            src = self.m6_gis_path.text()
        else:
            src = ""
        if not src or not Path(src).exists():
            QMessageBox.warning(self, "Ошибка",
                                f"Источник {method_name} не найден: {src}")
            return

        if method_name == "M4":
            if QMessageBox.question(self,"Единицы DEM","Подтвердите: XY и Z DEM заданы в метрах.",QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes:
                return
        output_dir = self._make_run_dir(method_name)
        st = getattr(self.main_window, "scanning_type", "unknown")
        hyper = {
            "ransac_threshold": self.m1_ransac_threshold.value(),
            "epochs": self.epochs.value(),
            "n_classes": self.num_classes.value(),
        }

        hdb = getattr(self.main_window, "history_db", None)
        if hdb and hdb.is_available:
            preset = hdb.get_preset(method_name, st)
            if preset and isinstance(preset.get("hyperparams"), dict):
                hp = preset["hyperparams"]
                run_ref = preset.get("source_run_id")
                q = preset.get("quality_score")
                if QMessageBox.question(
                        self, "Использовать пресет?",
                        f"Найден пресет из прогона #{run_ref} "
                        f"(качество ≈ {q}).\n"
                        f"Применить его гиперпараметры?\n\n"
                        f"{json.dumps(hp, indent=2, ensure_ascii=False)}",
                        QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                    try:
                        if "ransac_threshold" in hp:
                            self.m1_ransac_threshold.setValue(
                                float(hp["ransac_threshold"]))
                            hyper["ransac_threshold"] = float(
                                hp["ransac_threshold"])
                        if "epochs" in hp:
                            self.epochs.setValue(int(hp["epochs"]))
                            hyper["epochs"] = int(hp["epochs"])
                        if "n_classes" in hp:
                            self.num_classes.setValue(int(hp["n_classes"]))
                            hyper["n_classes"] = int(hp["n_classes"])
                    except Exception:
                        pass

        if method_name=="M2":
            hyper["m2_options"]={"model_id":self.m2_model_id.currentText(),"checkpoint":self.m2_checkpoint.text().strip()}
        if method_name=="M3" and self.m3_finetune.isChecked():
            hyper["m3_parent"]=self.m3_checkpoint.text().strip()
        # Загружаем список классов из classifier.txt (если есть)
        classes = self._load_classifier()
        if classes:
            self.log.log(
                f"Загружен classifier.txt: {len(classes)} классов "
                f"({', '.join(c['name'] for c in classes[:6])}"
                f"{'…' if len(classes) > 6 else ''})", "INFO")
        hyper["scanning_type"] = getattr(self.main_window, "scanning_type", "unknown")
        cd = getattr(self.main_window, "crs_detector", None)
        hyper["crs_epsg"] = getattr(cd, "crs", None)
        hyper["dataset_root"] = str(getattr(self.main_window, "dataset_root", "") or "")
        if method_name == "M1":
            if Path(src).suffix.lower() not in (".las", ".laz"):
                QMessageBox.warning(self, "M1", "Для сохранения исходных измерений выберите оригинальный LAS/LAZ")
                return
            if QMessageBox.question(self, "Единицы координат",
                    "Подтвердите: исходные XY и Z заданы в метрах. Географические координаты нужно перепроецировать отдельно.",
                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
            options = dict(_APP_SETTINGS.get("m1", {}))
            profiles = [(0.6, 16), (1.0, 32), (1.5, 64)]
            radius, neighbors = profiles[self.m1_profile.currentIndex()]
            options.update(units="metres", scanning_type=hyper["scanning_type"],
                           ground_backend=self.m1_ground.currentText(), model=self.m1_cpu_model.text().strip(),
                           preserve_classes=self.m1_preserve.isChecked(), radius=radius, neighbors=neighbors)
            lt=self.main_window.tabs.widget(0)
            hyper["spatial"]={"vertical_crs":lt.vertical_crs_combo.currentData(),"height_offset":lt.height_offset_spin.value()}
            hyper["m1_options"] = options
        worker = WorkerThread(self._train_task, method_name, src,
                              str(output_dir), hyper, classes)
        worker.log_signal.connect(self.log.log)
        worker.detail_signal.connect(self._on_worker_detail)
        worker.progress.connect(self.progress.setValue)
        worker.task_done.connect(self._on_training_done)
        worker.start()
        self._worker = worker
        if hasattr(self, "btn_stop_training"):
            self.btn_stop_training.setEnabled(True)
        if hasattr(self, "monitor_label"):
            self.monitor_label.setText(
                f"▶ {method_name}: запуск...")


    # ── M5 helpers ────────────────────────────────────────────────
    def _register_las_in_refs(self, path):
        """Регистрирует LAS в refs.references_clouds, если его там нет."""
        import numpy as np
        from pathlib import Path as _P
        hdb = getattr(self.main_window, "history_db", None)
        if not hdb or not hdb.is_available:
            return
        p = _P(path)
        with hdb.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id FROM {hdb.schema_r}.references_clouds
                WHERE path=%s LIMIT 1
            """, (str(p),))
            if cur.fetchone():
                return  # уже есть
        # Читаем header LAS без полной загрузки
        try:
            import laspy
            with laspy.open(p) as fh:
                h = fh.header
                n_points = h.point_count
                xmin, ymin = h.x_min, h.y_min
                xmax, ymax = h.x_max, h.y_max
                zmin, zmax = h.z_min, h.z_max
        except Exception as e:
            self.log.log(f"Не удалось прочитать {p.name}: {e}", "WARNING")
            return
        # CRS: определяем из VLR или из main_window.crs_detector
        crs = None
        if getattr(self.main_window, "crs_detector", None):
            crs = self.main_window.crs_detector.crs
        bbox_line = hdb._bbox_to_box3d((xmin, ymin, zmin, xmax, ymax, zmax))
        with hdb.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {hdb.schema_r}.references_clouds
                    (name, path, format, n_points, scanning_type,
                     crs_epsg, bbox, description)
                VALUES (%s, %s, 'LAS', %s, %s, %s, %s::box3d, %s)
            """, (p.stem, str(p), int(n_points),
                  getattr(self.main_window, "scanning_type", "unknown"),
                  crs, bbox_line, "Авторегистрация из M5"))
        self.log.log(f"Зарегистрировано в БД: {p.name} "
                     f"({n_points:,} точек)", "SUCCESS")

    def _train_multi_m5_task(self, emit_log, emit_progress,
                             files, out_dir, hyper):
        """Обрабатывает список LAS для M5 последовательно."""
        from pathlib import Path as _P
        import time as _time
        mw = self.main_window
        hdb = getattr(mw, "history_db", None)
        run_id = None
        t0 = _time.time()
        if hdb and hdb.is_available:
            run_id = hdb.start_run(
                method="M5",
                scanning_type=(hyper or {}).get("scanning_type", "unknown"),
                crs_epsg=(getattr(mw.crs_detector, "crs", None)
                          if mw.crs_detector else None),
                source_path=";".join(files),
                hyperparams=hyper or {},
                n_points=None, bbox=None, bbox_4326=None)
        trainer = RealTrainer(EmitLogger(emit_log))
        n = len(files)
        all_ok = True
        total_classes = {}
        for i, f in enumerate(files):
            emit_log(f"─── [{i+1}/{n}] {_P(f).name} ───", "INFO")
            sub_dir = _P(out_dir) / f"{i+1:02d}_{_P(f).stem}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            ok = trainer.train_m5(f, str(sub_dir))
            if not ok:
                all_ok = False
            # Собираем статистику
            stats_file = sub_dir / "m5_qa.json"
            if stats_file.exists():
                try:
                    import json as _j
                    d = _j.loads(stats_file.read_text())
                    for k, v in d.get("classes", {}).items():
                        total_classes[k] = total_classes.get(k, 0) + v
                except Exception:
                    pass
            emit_progress(int(100 * (i + 1) / n))
        # Сводный отчёт
        try:
            import json as _j
            summary = {
                "method": "M5_multi",
                "n_files": n,
                "files": files,
                "aggregated_classes": total_classes,
                "duration_sec": _time.time() - t0,
            }
            (_P(out_dir) / "m5_multi_summary.json").write_text(
                _j.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8")
            emit_log(
                f"M5: сводка по {n} файлам → m5_multi_summary.json",
                "SUCCESS")
        except Exception as e:
            emit_log(f"M5: сводка не создана: {e}", "WARNING")
        if hdb and hdb.is_available and run_id:
            hdb.finish_run(run_id,
                           "success" if all_ok else "partial",
                           metrics={"n_files": n,
                                    "classes": total_classes},
                           duration_sec=_time.time() - t0)
        return all_ok

    def _run_combined(self):
        selected = [m for m, cb in self.combo_checks.items()
                    if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Ошибка", "Выберите метод.")
            return
        weights = {m: self.combo_weights[m].value() for m in selected}
        paths = {
            "M1": str(WORKSPACE / "pointcloud_clean.ply"),
            "M2": self.m2_raster_path.text(),
            "M3": self.dataset_path.text(),
            "M4": self.m4_dem_path.text(),
            "M5": self.m5_las_path.text(),
            "M6": self.m6_gis_path.text()}
        output_dir = self._make_run_dir("combined")
        worker = WorkerThread(self._combined_train_task,
                              selected, weights, paths, str(output_dir))
        worker.log_signal.connect(self.log.log)
        worker.progress.connect(self.progress.setValue)
        worker.task_done.connect(self._on_training_done)
        worker.start()
        self._worker = worker

    def _on_training_done(self, ok):
        self.progress.setValue(100 if ok else 0)
        if hasattr(self, "btn_stop_training"):
            self.btn_stop_training.setEnabled(False)
        if hasattr(self, "monitor_label"):
            state = "✅ завершено" if ok else "❌ ошибка/остановка"
            self.monitor_label.setText(
                f"{self.monitor_label.text()}\n── {state} ──")
        try:
            hist = self.main_window.tabs.widget(7)
            if hasattr(hist, "reload"):
                hist.reload()
        except Exception:
            pass

        # G. Собрать информацию о последнем прогоне из training_output
        try:
            latest = max(
                (d for d in TRAINING_OUTPUT_DIR.iterdir()
                 if d.is_dir()),
                key=lambda p: p.stat().st_mtime,
                default=None)
            if latest is None:
                self.result_label.setText("Результат: папок нет.")
                return
            files = sorted(f.name for f in latest.iterdir()
                           if f.is_file())
            report = latest / "m1_report.txt"
            if report.exists():
                head = report.read_text(encoding="utf-8")
                short = "\n".join(head.splitlines()[:14])
                self.result_label.setText(
                    f"<b>Папка прогона:</b> {latest}\n"
                    f"<pre style='color:#d4d4d4'>{short}</pre>\n"
                    f"<b>Всего файлов:</b> {len(files)} — "
                    f"{', '.join(files)}")
            else:
                self.result_label.setText(
                    f"<b>Папка прогона:</b> {latest}\n"
                    f"<b>Файлов:</b> {len(files)} — "
                    f"{', '.join(files) if files else '(пусто)'}\n"
                    f"Проверьте лог выше: если экспорт не удался, "
                    f"файлы могут отсутствовать.")
        except Exception as e:
            self.result_label.setText(f"Результат: ошибка сбора — {e}")

    def _train_task(self, emit_log, emit_progress, method, src, out_dir,
                    hyper=None, classes=None):
        emit_log(f"Запуск {method} | {src}", "INFO")
        emit_log(f"Выходная папка: {out_dir}", "INFO")
        mw = self.main_window
        hdb = getattr(mw, "history_db", None)
        run_id = None
        t0 = time.time()

        if hdb and hdb.is_available:
            bbox = None
            bbox_4326 = None
            n_points = None
            crs = (hyper or {}).get("crs_epsg")
            if Path(src).suffix.lower() in (".las", ".laz"):
                import laspy
                with laspy.open(src) as reader:
                    h = reader.header
                    bbox = tuple(h.mins) + tuple(h.maxs)
                    n_points = h.point_count
            run_id = hdb.start_run(
                method=method,
                scanning_type=(hyper or {}).get("scanning_type", "unknown"),
                crs_epsg=crs,
                source_path=src,
                dataset_root=(hyper or {}).get("dataset_root"),
                hyperparams=hyper or {},
                n_points=n_points,
                bbox=bbox,
                bbox_4326=bbox_4326)

        trainer = RealTrainer(EmitLogger(emit_log))
        trainer._scanning_type = (hyper or {}).get("scanning_type", "unknown")
        trainer._m1_options = (hyper or {}).get("m1_options", {})
        trainer._spatial_snapshot=(hyper or {}).get("spatial",{})
        trainer._m3_parent=(hyper or {}).get("m3_parent")
        trainer._m3_progress=emit_progress
        trainer._m2_snapshot=(hyper or {}).get("m2_options",{})
        emit_progress(0)
        if method == "M1":
            result = trainer.train_m1(
                src, out_dir,
                ransac_threshold=float(
                    (hyper or {}).get("ransac_threshold",
                                      0.05)),
                classifier_classes=classes,
                emit_progress=emit_progress)
        elif method == "M2":
            result = trainer.train_m2(src, out_dir)
        elif method == "M3":
            result = trainer.train_m3(
                src, out_dir,
                n_classes=int((hyper or {}).get(
                    "n_classes", 11)),
                epochs=int((hyper or {}).get(
                    "epochs", 10)))
        elif method == "M4":
            result = trainer.train_m4(src, out_dir)
        elif method == "M5":
            result = trainer.train_m5(src, out_dir)
        elif method == "M6":
            result = trainer.train_m6(src, out_dir)
        else:
            result = False
        if result:
            emit_progress(100)
        duration = time.time() - t0

        metrics = {}
        try:
            for p in Path(out_dir).glob("*_stats.json"):
                try:
                    d = json.loads(p.read_text())
                    if isinstance(d, dict):
                        metrics.update(d)
                except Exception:
                    pass
            m1j = Path(out_dir) / "metrics.json"
            if m1j.exists():
                d = json.loads(m1j.read_text())
                if "quality_score" in d:
                    metrics["quality_score"] = d["quality_score"]
        except Exception:
            pass

        if hdb and hdb.is_available and run_id:
            hdb.finish_run(run_id,
                           "success" if result else ("cancelled" if getattr(trainer, "_cancelled", False) else "failed"),
                           metrics=metrics,
                           duration_sec=duration)
            for f in Path(out_dir).iterdir():
                if f.is_file():
                    hdb.add_artifact(run_id, f.suffix.lstrip("."),
                                     str(f))
            if result and metrics.get("quality_score") is not None:
                q = metrics["quality_score"]
                hdb.save_preset(
                    method=method,
                    scanning_type=(hyper or {}).get("scanning_type", "unknown"),
                    crs_epsg=(hyper or {}).get("crs_epsg"),
                    hyperparams=hyper or {},
                    quality_score=float(q),
                    source_run_id=run_id)

        # Живой отчёт по метрикам
        q = metrics.get("quality_score")
        if q is not None:
            if q >= 0.7:
                verdict = "✅ КАЧЕСТВО ДОСТАТОЧНОЕ"
            elif q >= 0.4:
                verdict = "⚠ УДОВЛЕТВОРИТЕЛЬНО — можно улучшить"
            else:
                verdict = "❌ НИЗКОЕ — рекомендуем сменить эталоны"
            emit_log(f"{method}: quality={q:.3f} — {verdict}", "SUCCESS")

        if result:
            emit_log(f"{method}: успешно ({duration:.1f} с)", "SUCCESS")
        else:
            emit_log(f"{method}: не удалось", "ERROR")
        return result

    def _combined_train_task(self, emit_log, emit_progress,
                             methods, weights, paths, out_dir):
        emit_log(f"Комбинирование: {methods}", "INFO")
        trainer = RealTrainer(EmitLogger(emit_log))
        result = trainer.train_combined(methods, weights, paths, out_dir)
        emit_progress(100)
        return result

    def _run_m3_infer(self):
        image=self.m3_image.text().strip();checkpoint=self.m3_checkpoint.text().strip()
        if not Path(image).is_file() or not Path(checkpoint).is_file():
            QMessageBox.warning(self,"M3","Укажите существующие GeoTIFF и локальный checkpoint .npz")
            return
        out=self._make_run_dir("M3_infer")
        worker=WorkerThread(self._m3_infer_task,image,checkpoint,str(out))
        worker.log_signal.connect(self.log.log);worker.progress.connect(self.progress.setValue)
        worker.detail_signal.connect(self._on_worker_detail);worker.task_done.connect(self._on_training_done)
        worker.start();self._worker=worker;self.btn_stop_training.setEnabled(True)

    def _m3_infer_task(self,emit_log,emit_progress,image,checkpoint,out_dir):
        from city3d_backend.runner import PipelineRunner,Task
        result=PipelineRunner().run([Task("M3","m3.raster",inputs={"asset":image},params={"checkpoint":checkpoint})],out_dir,
            cancel=lambda:QThread.currentThread().isInterruptionRequested(),progress=emit_progress)
        emit_log("M3 inference: "+result["status"],"SUCCESS" if result["status"]=="success" else "ERROR")
        if result["status"]!="success":emit_log(str(result["steps"]),"ERROR")
        return result["status"]=="success"


# ============================================================================
# Вкладка 6 — Экспорт
# ============================================================================


# PATCHED_SLICE_V8 — расширенный экспорт срезов
# PATCHED_SLICE_V8: helpers — константы и менеджер настроек
# (вставляется в city3d_pipeline_v5.py перед SliceExporterWidget)

ASPRS_CLASSES = {
    0:  "Never classified",
    1:  "Unclassified",
    2:  "Земля",
    3:  "Низкая растительность",
    4:  "Средняя растительность",
    5:  "Высокая растительность",
    6:  "Здания",
    7:  "Шум",
    8:  "Ключевые точки модели",
    9:  "Вода",
    10: "Рельсы",
    11: "Дорожное покрытие",
    12: "Перекрытие",
    13: "Провода (защита)",
    14: "Провода (проводник)",
    15: "Опоры ЛЭП",
    16: "Изоляторы",
    17: "Мосты",
    18: "Высокий шум",
    19: "Высокая растительность (авто)",
    20: "Здания (авто)",
    21: "Забор/стена",
    22: "Дорожное покрытие (авто)",
    23: "Автомобили",
}


class SliceSettingsManager:
    """Сохраняет/загружает настройки сессии срезов в JSON."""

    VERSION = "v8.1"

    def __init__(self, path=None):
        if path is None:
            try:
                cfg_dir = Path.home() / ".config" / "city3d"
                cfg_dir.mkdir(parents=True, exist_ok=True)
                path = cfg_dir / "slice_settings.json"
            except Exception:
                path = None
        self.path = path
        self.data = self._default()
        self.load()

    @staticmethod
    def _default():
        base_slice = {
            "enabled": False,
            "hag_min": 0.06, "hag_max": 1.00,
            "mode": 0,                 # 0=all, 1=classes, 2=sections
            "classes": [2, 6],
            "step": 0.5,
            "range_lo": 0.25, "range_hi": 0.25,
            "range_mode": "strict",    # strict | range
            "format": 0,               # 0=GeoTIFF, 1=LAS
            "shape": "square",         # square | round
            "px_x": 0.10, "px_y": 0.10,
            "name": "slice_1",
            "sections_single_file": False,
        }
        s1 = dict(base_slice); s1["enabled"] = True
        s2 = dict(base_slice); s2.update(
            {"hag_min": 1.00, "hag_max": 2.00, "name": "slice_2"})
        s3 = dict(base_slice); s3.update(
            {"hag_min": 2.00, "hag_max": 999.0, "name": "slice_3"})
        return {
            "version": SliceSettingsManager.VERSION,
            "source_paths": [],
            "source_mode": "single",   # single | files | dir
            "in_epsg": "",
            "out_epsg": "",
            "boundary": {
                "enable": False,
                "holes": True,
                "min_hole_area": 100.0,
                "concavity": 50.0,
                "min_area": 0.0,
                "format": 0,           # 0=GeoJSON, 1=WKT, 2=DXF
            },
            "slices": [s1, s2, s3],
        }

    def load(self):
        if not self.path or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            default = self._default()
            for k, v in default.items():
                if k not in raw:
                    raw[k] = v
            if not isinstance(raw.get("slices"), list):
                raw["slices"] = default["slices"]
            while len(raw["slices"]) < 3:
                raw["slices"].append(default["slices"][len(raw["slices"])])
            self.data = raw
        except Exception:
            self.data = self._default()

    def save(self):
        if not self.path:
            return
        try:
            self.data["version"] = self.VERSION
            self.path.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception:
            pass

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value

    def slice_cfg(self, idx):
        try:
            return self.data["slices"][idx]
        except Exception:
            return self._default()["slices"][idx]

    def update_slice(self, idx, **kwargs):
        try:
            self.data["slices"][idx].update(kwargs)
        except Exception:
            pass


class SliceExporterWidget(QWidget):
    """Экспорт срезов облака точек по HAG (PATCHED_SLICE_V8).

    Возможности:
      • Источник: 1 файл / несколько / папка (рекурсивно)
      • Перепроецирование EPSG вход→выход
      • Векторные границы (PDAL/alphashape/ConvexHull) с дырами
      • 3 режима среза: все / по классам ASPRS / сечения рельефа
      • Сохранение настроек сессии в ~/.config/city3d/slice_settings.json
    """

    MAX_SLICES = 3
    MAX_RASTER_PIXELS = 8_000_000_000

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._slices = []
        self._settings = SliceSettingsManager()
        self._worker = None
        self._build_ui()
        self._load_from_settings()

    # ── UI ──────────────────────────────────────────────────
    def _build_ui(self):

        # Источник
        src_gb = QGroupBox("Источник LAS")
        src_l = QVBoxLayout(src_gb)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Режим:"))
        self.src_mode_combo = QComboBox()
        self.src_mode_combo.addItems(
            ["Один файл", "Несколько файлов", "Папка (рекурсивно)"])
        mode_row.addWidget(self.src_mode_combo, 1)
        src_l.addLayout(mode_row)
        self.src_path = QLineEdit()
        self.src_path.setPlaceholderText("Выберите файл(ы) или папку с LAS")
        src_l.addWidget(self.src_path)
        row = QHBoxLayout()
        b1 = QPushButton("Файл...")
        b1.clicked.connect(self._browse_src_file)
        row.addWidget(b1)
        b2 = QPushButton("Файлы...")
        b2.clicked.connect(self._browse_src_files)
        row.addWidget(b2)
        b3 = QPushButton("Папка...")
        b3.clicked.connect(self._browse_src_dir)
        row.addWidget(b3)
        b4 = QPushButton("M1")
        b4.setToolTip("Последний m1_classified.las")
        b4.clicked.connect(self._use_last_m1)
        row.addWidget(b4)
        row.addStretch()
        src_l.addLayout(row)
        self.src_list = QListWidget()
        self.src_list.setMaximumHeight(90)
        self.src_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        src_l.addWidget(self.src_list)
        list_row = QHBoxLayout()
        ba = QPushButton("+ Файлы")
        ba.clicked.connect(self._add_files_to_list)
        list_row.addWidget(ba)
        br = QPushButton("− Удалить")
        br.clicked.connect(self._remove_from_list)
        list_row.addWidget(br)
        bc = QPushButton("Очистить")
        bc.clicked.connect(lambda: self.src_list.clear())
        list_row.addWidget(bc)
        list_row.addStretch()
        src_l.addLayout(list_row)
        # PATCHED_SINGLE_SCROLL_V3: единый layout без scroll
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(src_gb)

        # Перепроецирование
        crs_gb = QGroupBox("Перепроецирование (опционально)")
        crs_l = QFormLayout(crs_gb)
        self.in_epsg = QLineEdit()
        self.in_epsg.setPlaceholderText("пусто = из VLR LAS")
        crs_l.addRow("Входной EPSG:", self.in_epsg)
        self.out_epsg = QLineEdit()
        self.out_epsg.setPlaceholderText("пусто = не менять")
        crs_l.addRow("Выходной EPSG:", self.out_epsg)
        layout.addWidget(crs_gb)

        # Векторные границы
        bg = QGroupBox("Векторные границы LAS (boundary)")
        bl = QFormLayout(bg)
        self.boundary_enable = QCheckBox("Создать полигон границы")
        bl.addRow(self.boundary_enable)
        self.boundary_holes = QCheckBox("Включить отверстия (дыры)")
        self.boundary_holes.setChecked(True)
        bl.addRow(self.boundary_holes)
        self.boundary_min_hole = QDoubleSpinBox()
        self.boundary_min_hole.setRange(0.0, 1_000_000.0)
        self.boundary_min_hole.setValue(100.0)
        self.boundary_min_hole.setSuffix(" м²")
        bl.addRow("Мин. площадь дыры:", self.boundary_min_hole)
        self.boundary_concavity = QDoubleSpinBox()
        self.boundary_concavity.setRange(1.0, 5000.0)
        self.boundary_concavity.setValue(50.0)
        self.boundary_concavity.setSuffix(" м")
        bl.addRow("Вогнутость (concavity):", self.boundary_concavity)
        self.boundary_min_area = QDoubleSpinBox()
        self.boundary_min_area.setRange(0.0, 1_000_000.0)
        self.boundary_min_area.setValue(0.0)
        self.boundary_min_area.setSuffix(" м²")
        bl.addRow("Мин. площадь полигона:", self.boundary_min_area)
        self.boundary_format = QComboBox()
        self.boundary_format.addItems(["GeoJSON", "WKT", "DXF"])
        bl.addRow("Формат границы:", self.boundary_format)
        layout.addWidget(bg)

        # Срезы
        for i in range(self.MAX_SLICES):
            self._add_slice_group(i, layout)

        # Кнопки
        btn_row = QHBoxLayout()
        b_stats = QPushButton("HAG статистика")
        b_stats.clicked.connect(self._show_hag_stats)
        btn_row.addWidget(b_stats)
        b_save = QPushButton("💾 Сохранить настройки")
        b_save.clicked.connect(self._save_settings_now)
        btn_row.addWidget(b_save)
        b_load = QPushButton("📂 Загрузить настройки")
        b_load.clicked.connect(self._load_settings_now)
        btn_row.addWidget(b_load)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        exp_row = QHBoxLayout()
        b_exp = QPushButton("ЭКСПОРТИРОВАТЬ АКТИВНЫЕ СРЕЗЫ")
        b_exp.setStyleSheet(
            "background-color:#0e639c; color:white; "
            "font-weight:bold; padding:10px;")
        b_exp.setMinimumHeight(40)
        b_exp.clicked.connect(self._export_slices)
        exp_row.addWidget(b_exp, 1)
        b_open = QPushButton("📁 Папка срезов")
        b_open.clicked.connect(self._open_slices_dir)
        exp_row.addWidget(b_open)
        layout.addLayout(exp_row)

        self.log = LogWidget()
        self.log.setMaximumHeight(160)
        layout.addWidget(self.log)

    def _add_slice_group(self, idx, parent_layout):
        gb = QGroupBox(f"Срез {idx + 1}")
        gb.setCheckable(True)
        gb.setChecked(idx == 0)
        gl = QFormLayout(gb)

        # HAG
        hag_row = QHBoxLayout()
        min_spin = QDoubleSpinBox()
        min_spin.setRange(-100.0, 999.0)
        min_spin.setValue(0.06)
        min_spin.setDecimals(2)
        min_spin.setSuffix(" м")
        min_spin.setMaximumWidth(90)
        hag_row.addWidget(QLabel("От:"))
        hag_row.addWidget(min_spin)
        max_spin = QDoubleSpinBox()
        max_spin.setRange(-100.0, 999.0)
        max_spin.setValue(1.0)
        max_spin.setDecimals(2)
        max_spin.setSuffix(" м")
        max_spin.setMaximumWidth(90)
        hag_row.addWidget(QLabel("До:"))
        hag_row.addWidget(max_spin)
        hag_row.addStretch()
        gl.addRow("HAG:", hag_row)

        # Mode
        mode_combo = QComboBox()
        mode_combo.addItems([
            "0 — Все точки над землёй",
            "1 — По выбранным классам ASPRS",
            "2 — Сечения рельефа (по земле)",
        ])
        gl.addRow("Режим:", mode_combo)

        # Classes
        classes_widget = QWidget()
        cw = QVBoxLayout(classes_widget)
        cw.setContentsMargins(0, 0, 0, 0)
        cr = QHBoxLayout()
        cr.addWidget(QLabel("Классы:"))
        classes_edit = QLineEdit("2, 6")
        classes_edit.setPlaceholderText("напр. 2, 6, 11")
        cr.addWidget(classes_edit, 1)
        cw.addLayout(cr)
        ch = QLabel("2=земля 6=здания 3-5=растит. 9=вода 11=дороги")
        ch.setStyleSheet("color:#808080; font-size:9px;")
        cw.addWidget(ch)
        gl.addRow(classes_widget)
        classes_widget.setVisible(False)

        # Sections
        sec_widget = QWidget()
        sw = QVBoxLayout(sec_widget)
        sw.setContentsMargins(0, 0, 0, 0)
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Шаг:"))
        step_spin = QDoubleSpinBox()
        step_spin.setRange(0.05, 100.0)
        step_spin.setValue(0.5)
        step_spin.setSuffix(" м")
        step_spin.setDecimals(2)
        step_row.addWidget(step_spin)
        sw.addLayout(step_row)
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Режим:"))
        range_combo = QComboBox()
        range_combo.addItems([
            "Строго на уровне (±step/2)",
            "Диапазон −/+",
        ])
        range_row.addWidget(range_combo, 1)
        sw.addLayout(range_row)
        range_lo_row = QHBoxLayout()
        range_lo_row.addWidget(QLabel("−:"))
        range_lo = QDoubleSpinBox()
        range_lo.setRange(0.0, 50.0)
        range_lo.setValue(0.25)
        range_lo.setSuffix(" м")
        range_lo.setDecimals(2)
        range_lo_row.addWidget(range_lo)
        range_lo_row.addWidget(QLabel("+:"))
        range_hi = QDoubleSpinBox()
        range_hi.setRange(0.0, 50.0)
        range_hi.setValue(0.25)
        range_hi.setSuffix(" м")
        range_hi.setDecimals(2)
        range_lo_row.addWidget(range_hi)
        range_lo_row.addStretch()
        sw.addLayout(range_lo_row)
        single_file_cb = QCheckBox("Все сечения в один многостраничный TIFF")
        sw.addWidget(single_file_cb)
        gl.addRow(sec_widget)
        sec_widget.setVisible(False)

        def _upd_vis(midx):
            classes_widget.setVisible(midx == 1)
            sec_widget.setVisible(midx == 2)
        mode_combo.currentIndexChanged.connect(_upd_vis)

        # Format
        fmt_combo = QComboBox()
        fmt_combo.addItems([
            "GeoTIFF (1-бит B/W прозрачный)",
            "LAS (копия источника)",
        ])
        gl.addRow("Формат:", fmt_combo)

        # Shape + pixel
        shape_widget = QWidget()
        shl = QFormLayout(shape_widget)
        shl.setContentsMargins(0, 0, 0, 0)
        shape_combo = QComboBox()
        shape_combo.addItems(["Квадратный", "Круглый"])
        shl.addRow("Форма пикс.:", shape_combo)
        px_row = QHBoxLayout()
        px_x = QDoubleSpinBox()
        px_x.setRange(0.01, 5.0)
        px_x.setValue(0.10)
        px_x.setDecimals(2)
        px_x.setSuffix(" м")
        px_x.setMaximumWidth(80)
        px_row.addWidget(QLabel("X:"))
        px_row.addWidget(px_x)
        px_y = QDoubleSpinBox()
        px_y.setRange(0.01, 5.0)
        px_y.setValue(0.10)
        px_y.setDecimals(2)
        px_y.setSuffix(" м")
        px_y.setMaximumWidth(80)
        px_row.addWidget(QLabel("Y:"))
        px_row.addWidget(px_y)
        px_row.addStretch()
        shl.addRow("Пиксель:", px_row)
        gl.addRow(shape_widget)

        def _upd_fmt(fidx):
            shape_widget.setVisible(fidx == 0)
        fmt_combo.currentIndexChanged.connect(_upd_fmt)

        # Name
        name_edit = QLineEdit(f"slice_{idx + 1}")
        gl.addRow("Имя:", name_edit)

        self._slices.append({
            "group": gb, "min": min_spin, "max": max_spin,
            "mode": mode_combo, "classes_edit": classes_edit,
            "step": step_spin, "range_mode": range_combo,
            "range_lo": range_lo, "range_hi": range_hi,
            "sections_single": single_file_cb,
            "format": fmt_combo, "shape": shape_combo,
            "px_x": px_x, "px_y": px_y, "name": name_edit,
        })
        parent_layout.addWidget(gb)

    # ── Настройки: load/save ────────────────────────────────
    def _load_from_settings(self):
        s = self._settings
        paths = s.get("source_paths", [])
        for p in paths:
            self.src_list.addItem(p)
        if paths:
            self.src_path.setText(paths[0])
            if len(paths) > 1:
                self.src_mode_combo.setCurrentIndex(1)
        mode = s.get("source_mode", "single")
        if mode == "dir":
            self.src_mode_combo.setCurrentIndex(2)
        elif mode == "files":
            self.src_mode_combo.setCurrentIndex(1)
        self.in_epsg.setText(s.get("in_epsg", ""))
        self.out_epsg.setText(s.get("out_epsg", ""))
        b = s.get("boundary", {})
        self.boundary_enable.setChecked(bool(b.get("enable", False)))
        self.boundary_holes.setChecked(bool(b.get("holes", True)))
        self.boundary_min_hole.setValue(float(b.get("min_hole_area", 100.0)))
        self.boundary_concavity.setValue(float(b.get("concavity", 50.0)))
        self.boundary_min_area.setValue(float(b.get("min_area", 0.0)))
        self.boundary_format.setCurrentIndex(int(b.get("format", 0)))
        for i in range(self.MAX_SLICES):
            self._apply_slice_cfg(i, s.slice_cfg(i))

    def _apply_slice_cfg(self, idx, cfg):
        if idx >= len(self._slices):
            return
        sl = self._slices[idx]
        try:
            sl["group"].setChecked(bool(cfg.get("enabled", idx == 0)))
            sl["min"].setValue(float(cfg.get("hag_min", 0.06)))
            sl["max"].setValue(float(cfg.get("hag_max", 1.0)))
            sl["mode"].setCurrentIndex(int(cfg.get("mode", 0)))
            cls = cfg.get("classes", [2, 6])
            sl["classes_edit"].setText(", ".join(str(c) for c in cls))
            sl["step"].setValue(float(cfg.get("step", 0.5)))
            sl["range_lo"].setValue(float(cfg.get("range_lo", 0.25)))
            sl["range_hi"].setValue(float(cfg.get("range_hi", 0.25)))
            rm = cfg.get("range_mode", "strict")
            sl["range_mode"].setCurrentIndex(1 if rm == "range" else 0)
            sl["sections_single"].setChecked(
                bool(cfg.get("sections_single_file", False)))
            sl["format"].setCurrentIndex(int(cfg.get("format", 0)))
            sh = cfg.get("shape", "square")
            sl["shape"].setCurrentIndex(1 if sh == "round" else 0)
            sl["px_x"].setValue(float(cfg.get("px_x", 0.10)))
            sl["px_y"].setValue(float(cfg.get("px_y", 0.10)))
            sl["name"].setText(str(cfg.get("name", f"slice_{idx+1}")))
        except Exception:
            pass

    def _collect_settings(self):
        paths = [self.src_list.item(i).text()
                 for i in range(self.src_list.count())]
        if not paths and self.src_path.text().strip():
            paths = [self.src_path.text().strip()]
        self._settings.set("source_paths", paths)
        mi = self.src_mode_combo.currentIndex()
        self._settings.set("source_mode",
                           ["single", "files", "dir"][mi])
        self._settings.set("in_epsg", self.in_epsg.text().strip())
        self._settings.set("out_epsg", self.out_epsg.text().strip())
        self._settings.set("boundary", {
            "enable": self.boundary_enable.isChecked(),
            "holes": self.boundary_holes.isChecked(),
            "min_hole_area": self.boundary_min_hole.value(),
            "concavity": self.boundary_concavity.value(),
            "min_area": self.boundary_min_area.value(),
            "format": self.boundary_format.currentIndex(),
        })
        for i, sl in enumerate(self._slices):
            classes = self._parse_classes(sl["classes_edit"].text())
            self._settings.update_slice(
                i,
                enabled=sl["group"].isChecked(),
                hag_min=sl["min"].value(),
                hag_max=sl["max"].value(),
                mode=sl["mode"].currentIndex(),
                classes=classes,
                step=sl["step"].value(),
                range_lo=sl["range_lo"].value(),
                range_hi=sl["range_hi"].value(),
                range_mode=("range"
                            if sl["range_mode"].currentIndex() == 1
                            else "strict"),
                format=sl["format"].currentIndex(),
                shape=("round" if sl["shape"].currentIndex() == 1
                       else "square"),
                px_x=sl["px_x"].value(),
                px_y=sl["px_y"].value(),
                name=sl["name"].text().strip() or f"slice_{i+1}",
                sections_single_file=sl["sections_single"].isChecked(),
            )

    def _save_settings_now(self):
        self._collect_settings()
        self._settings.save()
        self.log.log(f"Настройки: {self._settings.path}", "SUCCESS")

    def _load_settings_now(self):
        self._settings.load()
        self._load_from_settings()
        self.log.log("Настройки загружены", "SUCCESS")

    # ── Источники ───────────────────────────────────────────
    def _browse_src_file(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "LAS", str(WORKSPACE),
            "LAS/LAZ (*.las *.laz);;All Files (*)")
        if p:
            self.src_path.setText(p)
            self.src_list.clear()
            self.src_list.addItem(p)
            self.src_mode_combo.setCurrentIndex(0)

    def _browse_src_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "LAS (можно несколько)", str(WORKSPACE),
            "LAS/LAZ (*.las *.laz);;All Files (*)")
        if files:
            self.src_mode_combo.setCurrentIndex(1)
            self.src_path.setText(f"{len(files)} файл(ов)")
            for f in files:
                existing = [self.src_list.item(i).text()
                            for i in range(self.src_list.count())]
                if f not in existing:
                    self.src_list.addItem(f)

    def _browse_src_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Папка с LAS", str(WORKSPACE))
        if d:
            self.src_mode_combo.setCurrentIndex(2)
            self.src_path.setText(d)
            try:
                found = sorted(
                    list(Path(d).rglob("*.las"))
                    + list(Path(d).rglob("*.laz")))
                self.src_list.clear()
                for f in found:
                    self.src_list.addItem(str(f))
                self.log.log(
                    f"Найдено {len(found)} LAS в {d}", "SUCCESS")
            except Exception as e:
                self.log.log(f"Ошибка сканирования: {e}", "ERROR")

    def _add_files_to_list(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Добавить LAS", str(WORKSPACE),
            "LAS/LAZ (*.las *.laz);;All Files (*)")
        for f in files:
            existing = [self.src_list.item(i).text()
                        for i in range(self.src_list.count())]
            if f not in existing:
                self.src_list.addItem(f)
        if files:
            self.src_mode_combo.setCurrentIndex(1)

    def _remove_from_list(self):
        for it in self.src_list.selectedItems():
            self.src_list.takeItem(self.src_list.row(it))

    def _use_last_m1(self):
        try:
            cands = sorted(
                TRAINING_OUTPUT_DIR.glob("*/m1_classified.las"),
                key=lambda p: p.stat().st_mtime, reverse=True)
            if not cands:
                cands = sorted(
                    TRAINING_OUTPUT_DIR.glob(
                        "pipeline_*/**/m1_classified.las"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
            if cands:
                self.src_path.setText(str(cands[0]))
                self.src_list.clear()
                self.src_list.addItem(str(cands[0]))
                self.src_mode_combo.setCurrentIndex(0)
                self.log.log(f"M1: {cands[0].name}", "SUCCESS")
            else:
                self.log.log("m1_classified.las не найден", "WARNING")
        except Exception as e:
            self.log.log(f"Ошибка поиска M1: {e}", "ERROR")

    def _open_slices_dir(self):
        d = SLICES_DIR
        d.mkdir(exist_ok=True)
        try:
            subprocess.Popen(["xdg-open", str(d)])
        except Exception as e:
            self.log.log(f"Не открыть: {e}", "ERROR")

    def _collect_source_files(self):
        files = []
        mode = self.src_mode_combo.currentIndex()
        if mode == 0:
            p = self.src_path.text().strip()
            if p and Path(p).exists():
                files = [p]
        elif mode == 1:
            files = [self.src_list.item(i).text()
                     for i in range(self.src_list.count())]
        elif mode == 2:
            d = self.src_path.text().strip()
            if d and Path(d).is_dir():
                files = sorted(
                    [str(f) for f in Path(d).rglob("*.las")]
                    + [str(f) for f in Path(d).rglob("*.laz")])
        for i in range(self.src_list.count()):
            t = self.src_list.item(i).text()
            if t not in files:
                files.append(t)
        return [f for f in files if Path(f).exists()]

    @staticmethod
    def _parse_classes(txt):
        out = []
        for p in (txt or "").replace(";", ",").split(","):
            p = p.strip()
            if p.isdigit():
                c = int(p)
                if 0 <= c <= 255:
                    out.append(c)
        return out

    # ── HAG ─────────────────────────────────────────────────
    @staticmethod
    def _compute_hag(x, y, z, ground_mask):
        from city3d_backend.core import idw
        xyz = np.column_stack([x, y, z]).astype(np.float64)
        origin = xyz.min(axis=0)
        xyz -= origin
        ground_z, valid = idw(xyz[:, :2], xyz[ground_mask, :2], xyz[ground_mask, 2], 5.0)
        return xyz[:, 2] - ground_z, ground_z + origin[2]

    def _show_hag_stats(self):
        files = self._collect_source_files()
        if not files:
            QMessageBox.warning(self, "Ошибка", "Укажите LAS")
            return
        self.log.log(f"Анализ: {Path(files[0]).name}", "INFO")
        try:
            import laspy
            las = laspy.read(files[0])
            x = np.asarray(las.x)
            y = np.asarray(las.y)
            z = np.asarray(las.z)
            cls = np.asarray(las.classification)
            n = len(x)
            uniq, cnt = np.unique(cls, return_counts=True)
            self.log.log("Классы:", "INFO")
            for cc, nn in zip(uniq, cnt):
                nm = ASPRS_CLASSES.get(int(cc), f"cls_{cc}")
                self.log.log(
                    f"   {int(cc):3d} {nm:35s} {int(nn):>12,}", "INFO")
            gm = cls == 2
            if gm.sum() < 100:
                self.log.log(f"Мало класса 2: {int(gm.sum())}", "ERROR")
                return
            self.log.log("HAG (IDW, 4 соседа)...", "INFO")
            hag, _ = self._compute_hag(x, y, z, gm)
            self.log.log(
                f"HAG: min={hag.min():.2f} p5="
                f"{np.percentile(hag, 5):.2f} med="
                f"{np.median(hag):.2f} p95="
                f"{np.percentile(hag, 95):.2f} max={hag.max():.2f}",
                "SUCCESS")
            edges = [-100, -10, -1, 0, 0.06, 0.5, 1, 2, 3, 5,
                     8, 12, 20, 30, 50, 100, 999]
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                m = ((hag >= lo) & (hag < hi)).sum()
                pct = 100 * m / n
                bar = "#" * int(pct / 2)
                self.log.log(
                    f"   {lo:7.2f}-{hi:6.2f}: {int(m):>12,} "
                    f"({pct:5.1f}%) {bar}", "INFO")
        except Exception as e:
            import traceback
            self.log.log(f"Ошибка: {e}", "ERROR")
            self.log.log(traceback.format_exc(), "ERROR")

    # ── Загрузка всех источников ────────────────────────────
    def _load_all_sources(self, emit_log, emit_progress, files):
        import laspy
        from pyproj import CRS, Transformer

        in_epsg_txt = self._slice_snapshot["in_epsg"].text().strip()
        out_epsg_txt = self._slice_snapshot["out_epsg"].text().strip()
        in_epsg = None
        if in_epsg_txt:
            dig = "".join(c for c in in_epsg_txt if c.isdigit())
            in_epsg = int(dig) if dig else None
        out_epsg = None
        if out_epsg_txt:
            dig = "".join(c for c in out_epsg_txt if c.isdigit())
            out_epsg = int(dig) if dig else None

        all_x, all_y, all_z, all_c, all_hag = [], [], [], [], []
        self._slice_sources = []
        source_crs = None
        offset = 0
        for fi, fp in enumerate(files):
            las = laspy.read(fp)
            crs = las.header.parse_crs()
            if crs is None and in_epsg:
                crs = CRS.from_epsg(in_epsg)
            if crs is None or not crs.is_projected:
                raise ValueError("Для срезов нужен подтверждённый проекционный CRS; EPSG не угадывается")
            if source_crs is not None and source_crs != crs:
                raise ValueError("Сначала приведите источники срезов к одному CRS")
            source_crs = crs
            if out_epsg and crs.to_epsg() != out_epsg:
                raise ValueError("Перепроецирование срезов выполняйте отдельно с контролем атрибутов")
            n = len(las.points)
            self._slice_sources.append((fp, offset, offset+n))
            offset += n
            all_x.append(np.asarray(las.x)); all_y.append(np.asarray(las.y)); all_z.append(np.asarray(las.z))
            all_c.append(np.asarray(las.classification))
            all_hag.append(np.asarray(las.c3_hag) if "c3_hag" in las.point_format.dimension_names else None)
            emit_progress(int(25*(fi+1)/len(files)))
        return {"x": np.concatenate(all_x), "y": np.concatenate(all_y), "z": np.concatenate(all_z),
                "cls": np.concatenate(all_c), "epsg": source_crs.to_epsg(),
                "hag": np.concatenate(all_hag) if all(h is not None for h in all_hag) else None}

    def _write_preserved_slice(self, emit_log, out, selection):
        import laspy, copy
        from city3d_backend.core import check
        for fi, (source, start, end) in enumerate(self._slice_sources):
            check(lambda: QThread.currentThread().isInterruptionRequested())
            selected = selection[start:end]
            if not selected.any():
                continue
            target = Path(out)
            if len(self._slice_sources) > 1:
                target = target.with_name(target.stem + f"_src{fi+1}" + target.suffix)
            original = laspy.read(source)
            las = laspy.LasData(copy.deepcopy(original.header), points=original.points[selected].copy())
            las.evlrs = copy.deepcopy(original.evlrs)
            tmp = target.with_suffix(".tmp.las")
            las.write(tmp)
            os.replace(tmp, target)
            emit_log(f"LAS: {target.name}, исходные измерения сохранены", "SUCCESS")

    # ── Границы ─────────────────────────────────────────────
    def _create_boundary(self, emit_log, pts_xyz, epsg, out_dir):
        if not self._slice_snapshot["boundary_enable"].isChecked():
            return None
        try:
            from shapely.geometry import Polygon
            from shapely import wkt as shp_wkt
        except ImportError:
            emit_log("Нужен shapely", "ERROR")
            return None

        conc = self._slice_snapshot["boundary_concavity"].value()
        holes_on = self._slice_snapshot["boundary_holes"].isChecked()
        min_area = self._slice_snapshot["boundary_min_area"].value()
        emit_log(f"Boundary: concavity={conc} м, holes="
                 f"{'да' if holes_on else 'нет'}", "INFO")

        geom = None
        # PDAL
        try:
            import subprocess as _sp
            import laspy
            tmp = out_dir / "_boundary_in.las"
            header = laspy.LasHeader(point_format=3, version="1.4")
            header.offsets = pts_xyz.min(axis=0)
            header.scales = np.array([0.001, 0.001, 0.001])
            las = laspy.LasData(header)
            las.x = pts_xyz[:, 0]
            las.y = pts_xyz[:, 1]
            las.z = pts_xyz[:, 2]
            las.write(str(tmp))
            pipeline = {"pipeline": [
                {"type": "readers.las", "filename": str(tmp)},
                {"type": "filters.hexbin",
                 "edge_size": float(conc),
                 "holes": bool(holes_on)},
                {"type": "writers.text",
                 "filename": str(out_dir / "_boundary.wkt"),
                 "format": "wkt", "order": "X,Y"}
            ]}
            pf = out_dir / "_boundary_pipeline.json"
            pf.write_text(json.dumps(pipeline, indent=2))
            r = _sp.run(["pdal", "pipeline", str(pf)],
                        capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and (out_dir / "_boundary.wkt").exists():
                txt = (out_dir / "_boundary.wkt").read_text()
                if "POLYGON" in txt:
                    geom = shp_wkt.loads(txt)
                    emit_log("PDAL hexbin: OK", "SUCCESS")
            for f in (tmp, pf):
                try: f.unlink()
                except Exception: pass
        except FileNotFoundError:
            emit_log("PDAL не найден — fallback", "WARNING")
        except Exception as e:
            emit_log(f"PDAL: {e}", "WARNING")

        if geom is None:
            try:
                import alphashape
                n = len(pts_xyz)
                sample = (pts_xyz[np.random.choice(n, 50_000,
                                                   replace=False), :2]
                          if n > 50_000 else pts_xyz[:, :2])
                geom = alphashape.alphashape(sample, 1.0 / max(conc, 1.0))
                emit_log("alphashape: OK", "SUCCESS")
            except ImportError:
                from scipy.spatial import ConvexHull
                sample = pts_xyz[:, :2]
                if len(sample) > 100_000:
                    sample = sample[np.random.choice(
                        len(sample), 100_000, replace=False)]
                hull = ConvexHull(sample)
                geom = Polygon(sample[hull.vertices])
                emit_log("ConvexHull (без дыр)", "WARNING")

        if geom is None or geom.is_empty:
            emit_log("Граница не построена", "ERROR")
            return None
        if min_area > 0 and geom.area < min_area:
            emit_log(f"Площадь {geom.area:.0f} < {min_area} м²", "WARNING")
            return None

        fmt = self._slice_snapshot["boundary_format"].currentText()
        name = self._slices[0]["name"].text().strip() or "boundary"
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y%m%d-%H%M%S")
        base = out_dir / f"{ts}_boundary"

        if fmt == "GeoJSON":
            feat = {
                "type": "Feature",
                "properties": {
                    "area_m2": float(geom.area),
                    "n_holes": (len(geom.interiors) if holes_on else 0),
                    "epsg": int(epsg),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(geom.exterior.coords)]
                    + ([list(h.coords) for h in geom.interiors]
                       if holes_on else []),
                },
            }
            out = base.with_suffix(".geojson")
            out.write_text(json.dumps(
                {"type": "FeatureCollection", "features": [feat]},
                ensure_ascii=False, indent=2), encoding="utf-8")
        elif fmt == "WKT":
            out = base.with_suffix(".wkt")
            out.write_text(geom.wkt, encoding="utf-8")
        else:
            try:
                import ezdxf
                doc = ezdxf.new("R2010")
                msp = doc.modelspace()
                msp.add_lwpolyline(list(geom.exterior.coords))
                if holes_on:
                    for interior in geom.interiors:
                        msp.add_lwpolyline(list(interior.coords))
                out = base.with_suffix(".dxf")
                doc.saveas(str(out))
            except ImportError:
                emit_log("Нужен ezdxf для DXF", "WARNING")
                return None

        emit_log(f"Boundary → {out.name} "
                 f"(площадь {geom.area:.0f} м², дыр "
                 f"{len(geom.interiors) if holes_on else 0})",
                 "SUCCESS")
        return out

    # ── Экспорт срезов ──────────────────────────────────────
    def _export_slices(self):
        files = self._collect_source_files()
        if not files:
            QMessageBox.warning(self, "Ошибка", "Укажите LAS-файл(ы)")
            return
        active = []
        for i, sl in enumerate(self._slices):
            if not sl["group"].isChecked():
                continue
            if sl["min"].value() >= sl["max"].value():
                QMessageBox.warning(
                    self, "Ошибка", f"Срез {i+1}: От >= До")
                return
            active.append((i, {key: FrozenControl.capture(value) if isinstance(value, QWidget) else value for key, value in sl.items()}))
        if not active:
            QMessageBox.warning(
                self, "Ошибка", "Активируйте хотя бы один срез")
            return
        self._collect_settings()
        self._settings.save()
        self._slice_snapshot = {name: FrozenControl.capture(getattr(self, name)) for name in
            ("in_epsg", "out_epsg", "boundary_enable", "boundary_concavity", "boundary_holes", "boundary_min_area", "boundary_format")
            if hasattr(self, name)}
        self._worker = WorkerThread(
            self._export_slices_task, files, active)
        self._worker.log_signal.connect(self.log.log)
        self._worker.task_done.connect(self._on_slices_done)
        self._worker.start()

    def _on_slices_done(self, ok):
        self.log.log("Срезы готовы" if ok else "Ошибка срезов",
                     "SUCCESS" if ok else "ERROR")
        try:
            exp = self.main_window.tabs.widget(5)
            if hasattr(exp, "refresh_preview_list"):
                exp.refresh_preview_list()
        except Exception:
            pass

    def _export_slices_task(self, emit_log, emit_progress, files, active):
        try:
            from datetime import datetime as _dt
            out_dir = SLICES_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            emit_log(f"Файлов: {len(files)}, срезов: {len(active)}",
                     "INFO")

            data = self._load_all_sources(emit_log, emit_progress, files)
            x, y, z, cls = data["x"], data["y"], data["z"], data["cls"]
            epsg = data["epsg"]
            emit_log(f"Точек: {len(x):,}, EPSG:{epsg}", "SUCCESS")

            gm = cls == 2
            if data.get("hag") is None and not gm.any():
                emit_log(f"Мало класса 2: {int(gm.sum())}", "ERROR")
                return False
            emit_log("HAG...", "INFO")
            emit_progress(30)
            hag = data.get("hag")
            if hag is None:
                hag, _ = self._compute_hag(x, y, z, gm)
            emit_log(f"HAG [{hag.min():.2f}, {hag.max():.2f}]", "SUCCESS")
            emit_progress(45)

            if self._slice_snapshot["boundary_enable"].isChecked():
                try:
                    self._create_boundary(
                        emit_log, np.column_stack([x, y, z]),
                        epsg, out_dir)
                except Exception as e:
                    emit_log(f"Boundary: {e}", "WARNING")
            emit_progress(55)

            ts = _dt.now().strftime("%Y%m%d-%H%M%S")
            n_ok = 0
            n_steps = len(active)
            for si, (s_idx, sl) in enumerate(active):
                emit_progress(55 + int(40 * si / max(n_steps, 1)))
                nm = sl["name"].text().strip() or f"slice_{s_idx+1}"
                mode = sl["mode"].currentIndex()
                emit_log("─" * 50, "INFO")
                emit_log(f"Срез {s_idx+1} [{nm}] mode={mode}", "INFO")
                try:
                    if mode == 2:
                        cnt = self._export_relief_sections(
                            emit_log, out_dir, ts, nm,
                            x, y, z, hag, cls, epsg, sl)
                    else:
                        cnt = self._export_regular_slice(
                            emit_log, out_dir, ts, nm,
                            x, y, z, cls, hag, epsg, sl)
                    if cnt > 0:
                        n_ok += 1
                except Exception as e:
                    import traceback
                    emit_log(f"  Ошибка среза {s_idx+1}: {e}", "ERROR")
                    emit_log(traceback.format_exc(), "ERROR")

            emit_progress(100)
            emit_log("─" * 50, "INFO")
            emit_log(f"ИТОГО: {n_ok}/{n_steps} → {out_dir}", "SUCCESS")
            return n_ok > 0
        except Exception as e:
            import traceback
            emit_log(f"Критическая: {e}", "ERROR")
            emit_log(traceback.format_exc(), "ERROR")
            return False

    def _export_regular_slice(self, emit_log, out_dir, ts, name,
                              x, y, z, cls, hag, epsg, sl):
        min_h = sl["min"].value()
        max_h = sl["max"].value()
        mode = sl["mode"].currentIndex()
        fmt_idx = sl["format"].currentIndex()
        sel = (hag >= min_h) & (hag < max_h)
        class_info = ""
        if mode == 1:
            classes = self._parse_classes(sl["classes_edit"].text())
            if not classes:
                emit_log("  Классы не заданы", "WARNING")
                return 0
            sel = sel & np.isin(cls, list(classes))
            class_info = f" (классы {sorted(classes)})"
        n_sel = int(sel.sum())
        emit_log(f"  {name}: точек {n_sel:,}{class_info}", "INFO")
        if n_sel < 10:
            emit_log("  ПРОПУСК (< 10)", "WARNING")
            return 0
        sx, sy, sz = x[sel], y[sel], z[sel]
        if fmt_idx == 1:
            out = out_dir / f"{ts}_{name}.las"
            self._write_preserved_slice(emit_log, out, sel)
        else:
            out = out_dir / f"{ts}_{name}.tif"
            self._write_slice_geotiff(
                emit_log, out, sx, sy, epsg,
                sl["px_x"].value(), sl["px_y"].value(),
                sl["shape"].currentIndex() == 1)
        return 1

    def _export_relief_sections(self, emit_log, out_dir, ts, name,
                                x, y, z, hag, cls, epsg, sl):
        step = sl["step"].value()
        rng_mode = sl["range_mode"].currentIndex()
        rng_lo = sl["range_lo"].value()
        rng_hi = sl["range_hi"].value()
        fmt_idx = sl["format"].currentIndex()
        z_min = float(np.nanmin(hag))
        z_max = float(np.nanmax(hag))
        if z_max <= z_min:
            emit_log(f"  {name}: плоский рельеф", "WARNING")
            return 0
        levels = np.arange(
            np.floor(z_min / step) * step,
            np.ceil(z_max / step) * step + step,
            step)
        emit_log(f"  {name}: {len(levels)} сечений (шаг {step} м, "
                 f"HAG [{z_min:.2f}, {z_max:.2f}])", "INFO")
        px_x = sl["px_x"].value()
        px_y = sl["px_y"].value()
        shape_round = sl["shape"].currentIndex() == 1
        total = 0
        for lev in levels:
            if rng_mode == 0:
                half = step / 2.0
                m = (hag >= lev - half) & (hag < lev + half)
            else:
                m = (hag >= lev - rng_lo) & (hag <= lev + rng_hi)
            n_m = int(m.sum())
            if n_m < 10:
                continue
            sx, sy, sz = x[m], y[m], z[m]
            lv = f"{name}_z{lev:+.2f}".replace(
                "+", "p").replace("-", "m").replace(".", "_")
            if fmt_idx == 1:
                out = out_dir / f"{ts}_{lv}.las"
                self._write_preserved_slice(emit_log, out, m)
            else:
                out = out_dir / f"{ts}_{lv}.tif"
                self._write_slice_geotiff(
                    emit_log, out, sx, sy, epsg,
                    px_x, px_y, shape_round)
            total += 1
        emit_log(f"  {name}: {total} сечений сохранено", "SUCCESS")
        return total

    # ── Писатели ────────────────────────────────────────────
    @staticmethod
    def _write_slice_las(emit_log, out_path, sx, sy, sz, epsg):
        import laspy
        from pyproj import CRS
        n = len(sx)
        header = laspy.LasHeader(point_format=3, version="1.4")
        try:
            header.point_count = int(n)
        except Exception:
            pass
        pts = np.column_stack([sx, sy, sz])
        header.offsets = pts.min(axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])
        try:
            crs = CRS.from_epsg(int(epsg))
            try: header.add_crs(crs)
            except Exception: pass
        except Exception:
            pass
        las = laspy.LasData(header)
        las.x = sx; las.y = sy; las.z = sz
        las.classification = np.zeros(n, dtype=np.uint8)
        las.write(str(out_path))
        try:
            mb = out_path.stat().st_size / 1024 / 1024
            emit_log(f"  LAS: {out_path.name} ({mb:.1f} МБ)", "SUCCESS")
        except Exception:
            pass

    @staticmethod
    def _write_slice_geotiff(emit_log, out_path, x, y, epsg,
                             px_x, px_y, shape_round):
        import rasterio
        from rasterio.transform import from_origin
        from pyproj import CRS
        if len(x) == 0:
            raise ValueError("Нет точек")
        x_min = float(x.min()); x_max = float(x.max())
        y_min = float(y.min()); y_max = float(y.max())
        cols = max(int(np.ceil((x_max - x_min) / px_x)), 1)
        rows = max(int(np.ceil((y_max - y_min) / px_y)), 1)
        if rows * cols > SliceExporterWidget.MAX_RASTER_PIXELS:
            import math
            scale = math.sqrt(
                (rows * cols) /
                SliceExporterWidget.MAX_RASTER_PIXELS)
            px_x *= scale; px_y *= scale
            cols = max(int(np.ceil((x_max - x_min) / px_x)), 1)
            rows = max(int(np.ceil((y_max - y_min) / px_y)), 1)
        px = ((x - x_min) / px_x).astype(np.int32)
        py = ((y_max - y) / px_y).astype(np.int32)
        valid = (px >= 0) & (px < cols) & (py >= 0) & (py < rows)
        px = px[valid]; py = py[valid]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        mask[py, px] = 1
        if shape_round:
            try:
                from scipy import ndimage
                struct = np.array(
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
                mask = ndimage.binary_dilation(
                    mask.astype(bool), structure=struct).astype(np.uint8)
            except ImportError:
                m = mask.copy()
                m[1:, :] |= mask[:-1, :]
                m[:-1, :] |= mask[1:, :]
                m[:, 1:] |= mask[:, :-1]
                m[:, :-1] |= mask[:, 1:]
                mask = m
        gray = np.where(mask == 1, 0, 255).astype(np.uint8)
        alpha = np.where(mask == 1, 255, 0).astype(np.uint8)
        transform = from_origin(x_min, y_max, px_x, px_y)
        try:
            crs = CRS.from_epsg(int(epsg))
        except Exception:
            crs = None
        meta = {
            'driver': 'GTiff', 'dtype': 'uint8', 'count': 2,
            'width': cols, 'height': rows,
            'transform': transform,
            'compress': 'DEFLATE', 'zlevel': 9,
            'tiled': True, 'blockxsize': 256, 'blockysize': 256,
            'BIGTIFF': 'IF_SAFER',
            'photometric': 'MINISBLACK',
            'interleave': 'band',
            'PREDICTOR': 2,
            'ALPHA': 'YES',
            'SPARSE_OK': 'TRUE',
        }
        if crs:
            meta['crs'] = crs
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(gray, 1)
            dst.write(alpha, 2)
        if crs:
            out_path.with_suffix(".prj").write_text(
                crs.to_wkt(), encoding="utf-8")
        tfw = (f"{px_x:.10f}\n0.0000000000\n0.0000000000\n"
               f"{-px_y:.10f}\n"
               f"{x_min + px_x / 2:.10f}\n"
               f"{y_max - px_y / 2:.10f}\n")
        out_path.with_suffix(".tfw").write_text(tfw, encoding="utf-8")
        try:
            mb = out_path.stat().st_size / 1024 / 1024
            bpp = out_path.stat().st_size * 8 / max(rows * cols, 1)
            emit_log(
                f"  GeoTIFF: {out_path.name} ({mb:.2f} МБ, "
                f"{rows}×{cols}, {bpp:.2f} бит/пикс., "
                f"пикс. {px_x:.3f} м)", "SUCCESS")
        except Exception:
            pass


class ExportTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.exporter = FormatExporter(WORKSPACE)
        self._exported_files = []
        self._last_output = None
        self._last_format = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Horizontal)

        # PATCHED_LAYOUT_FIX: оборачиваем левую колонку в QScrollArea,
        # иначе панель срезов не помещается и виджеты налезают
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(650)  # PATCHED_SINGLE_SCROLL_V3
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left = QWidget()
        ll = QVBoxLayout(left)

        g1 = QGroupBox("Формат")
        g1l = QVBoxLayout()
        self.combo = QComboBox()
        self.combo.addItems([
            "USD (OpenUSD)", "OBJ (меш)",
            "GeoJSON (веб)", "PostGIS (БД)",
            "DXF (CAD)", "IFC (BIM)", "CityGML (3dfier)"])
        g1l.addWidget(self.combo)
        g1.setLayout(g1l)
        ll.addWidget(g1)

        g2 = QGroupBox("Параметры")
        g2l = QFormLayout()
        self.name = QLineEdit("city_model")
        g2l.addRow("Имя:", self.name)
        self.use_cropped = QCheckBox("Использовать обрезанное")
        g2l.addRow("", self.use_cropped)

        # PATCHED_CRS_UI: целевой CRS
        self.target_crs_combo = QComboBox()
        self.target_crs_combo.addItems([
            "Не менять (CRS источника)",
            "EPSG:4326 — WGS 84 (географ.)",
            "EPSG:32639 — WGS 84 / UTM 39N",
            "EPSG:32638 — WGS 84 / UTM 38N",
            "EPSG:3857 — Web Mercator",
            "EPSG:28410 — Пулково 1942 / Gauss-Kruger CM 57E",
            "EPSG:5555 — WGS 84 / UTM 39N",
        ])
        self.target_crs_combo.setToolTip(
            "Во что перевести CRS при экспорте")
        g2l.addRow("Целевой CRS:", self.target_crs_combo)

        # Целевая вертикальная СК
        self.target_vert_combo = QComboBox()
        for code, name in VERTICAL_CRS_LIST:
            self.target_vert_combo.addItem(name, code)
        self.target_vert_combo.setToolTip(
            "Целевая система высот")
        g2l.addRow("Высоты:", self.target_vert_combo)

        # Смещение высот
        self.export_height_offset = QDoubleSpinBox()
        self.export_height_offset.setRange(-10000.0, 10000.0)
        self.export_height_offset.setDecimals(3)
        self.export_height_offset.setValue(0.0)
        self.export_height_offset.setToolTip(
            "Смещение Z при экспорте (м)")
        g2l.addRow("Δ высот (м):", self.export_height_offset)

        # Показать CRS источника
        self.export_src_crs = QLabel("—")
        self.export_src_crs.setStyleSheet(
            "color: #dcdcaa; font-family: monospace; padding: 2px;")
        g2l.addRow("CRS источника:", self.export_src_crs)

        g2.setLayout(g2l)
        ll.addWidget(g2)

        b = QPushButton("Экспортировать")
        b.clicked.connect(self.export)
        ll.addWidget(b)
        b_all = QPushButton("Экспортировать во все форматы")
        b_all.clicked.connect(self.export_all)
        ll.addWidget(b_all)

        pg = QGroupBox("Предпросмотр файлов")
        pl = QVBoxLayout()
        pl.addWidget(QLabel("Выберите файл:"))
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(
            self._on_preview_file_selected)
        pl.addWidget(self.file_combo)

        bb = QHBoxLayout()
        b_refresh = QPushButton("Обновить список")
        b_refresh.clicked.connect(self.refresh_preview_list)
        bb.addWidget(b_refresh)
        b_open = QPushButton("Открыть файл…")
        b_open.clicked.connect(self._open_file_dialog)
        bb.addWidget(b_open)
        pl.addLayout(bb)

        bb2 = QHBoxLayout()
        b_out = QPushButton("Открыть training_output")
        b_out.clicked.connect(self._open_training_output)
        bb2.addWidget(b_out)
        b_clear = QPushButton("Очистить список")
        b_clear.clicked.connect(self._clear_preview_list)
        bb2.addWidget(b_clear)
        # PATCHED_EXPORT_UI: удаление старых файлов
        b_purge = QPushButton("🗑 Удалить старые (>7 дней)")
        b_purge.setToolTip("Удаляет файлы экспорта старше 7 дней")
        b_purge.clicked.connect(self._purge_old_exports)
        bb2.addWidget(b_purge)
        pl.addLayout(bb2)

        pg.setLayout(pl)
        ll.addWidget(pg)

        self.log = LogWidget()
        ll.addWidget(self.log)

        # PATCHED_SLICE_EXPORTER: панель срезов по HAG
        try:
            slice_gb = QGroupBox(
                "Срезы облака по высоте (HAG)")
            slice_l = QVBoxLayout(slice_gb)
            self.slice_widget = SliceExporterWidget(
                self.main_window)
            slice_l.addWidget(self.slice_widget)
            ll.addWidget(slice_gb)
        except Exception as _e:
            import traceback
            print("[WARN] SliceExporterWidget:", _e,
                  file=sys.stderr)
            traceback.print_exc()
            self.slice_widget = None

        # PATCHED_LAYOUT_FIX: завершаем обёртку scroll area
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)
        self.preview = ExportPreviewViewer()
        splitter.addWidget(self.preview)
        splitter.setSizes([720, 850])  # PATCHED_SINGLE_SCROLL_V3
        layout.addWidget(splitter)
        self.setLayout(layout)

        self.refresh_preview_list()
        # PATCHED_CRS_UI: обновляем отображение CRS источника
        try:
            self._refresh_src_crs_label()
        except Exception:
            pass

    def _refresh_src_crs_label(self):
        """Показывает CRS источника в ExportTab. PATCHED_CRS_UI"""
        cd = getattr(self.main_window, "crs_detector", None)
        if cd and cd.crs:
            self.export_src_crs.setText(
                f"EPSG:{cd.crs} — {cd.crs_description or ''}")
        else:
            self.export_src_crs.setText("не определён")

    def refresh_preview_list(self):
        """Обновляет список. По умолчанию показывает ТОЛЬКО свежие
        файлы (младше 7 дней) чтобы старые ошибочные результаты не
        мешали. PATCHED_EXPORT_UI"""
        import time as _t
        current_data = self.file_combo.currentData()
        self.file_combo.blockSignals(True)
        self.file_combo.clear()

        # Фильтр по дате: только свежие (настраивается)
        max_age_days = 7
        cutoff = _t.time() - max_age_days * 86400

        search_dirs = [WORKSPACE, TRAINING_OUTPUT_DIR]
        for sub in sorted(TRAINING_OUTPUT_DIR.glob("*")):
            if sub.is_dir():
                search_dirs.append(sub)

        preparer = getattr(self.main_window, "dataset_preparer", None)
        if preparer is not None:
            for split in ("train", "val", "test"):
                search_dirs.append(preparer.root / "processed" / split)
            search_dirs.append(preparer.root / "augmented")
            search_dirs.append(preparer.root / "raw")

        exts = (".gml", ".usda", ".usd", ".obj", ".geojson",
                ".dxf", ".ifc", ".ply", ".stl",
                ".las", ".laz", ".pcd", ".xyz",
                ".json", ".txt")

        found = []
        seen = set()
        for d in search_dirs:
            if not d.exists():
                continue
            for ext in exts:
                for p in sorted(d.glob(f"*{ext}")):
                    key = str(p.resolve())
                    if key in seen:
                        continue
                    # PATCHED_EXPORT_UI: отсекаем старые и пустые
                    try:
                        if p.stat().st_mtime < cutoff:
                            continue
                        if p.stat().st_size < 100:
                            continue
                    except Exception:
                        continue
                    # Отсекаем .obj с < 500 треугольников
                    # (это заглушка от convex_hull)
                    if ext == ".obj":
                        try:
                            with open(p, "r") as f:
                                n_faces = sum(
                                    1 for line in f
                                    if line.startswith("f "))
                            if n_faces < 100:
                                continue
                        except Exception:
                            pass
                    seen.add(key)
                    found.append(p)
        for p_str in self._exported_files:
            pp = Path(p_str)
            if pp.exists():
                key = str(pp.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(pp)

        def key_func(p):
            try:
                mtime = -p.stat().st_mtime  # свежие — первыми
            except Exception:
                mtime = 0
            return (mtime, str(p))
        found.sort(key=key_func)  # PATCHED_EXPORT_UI

        for p in found:
            label = f"{p.parent.name}/{p.name}  " \
                    f"({ExportPreviewViewer._human_size(p)})"
            self.file_combo.addItem(label, str(p))

        self.file_combo.blockSignals(False)

        if current_data:
            idx = self.file_combo.findData(current_data)
            if idx >= 0:
                self.file_combo.setCurrentIndex(idx)
                return
        if self.file_combo.count() > 0:
            self.file_combo.setCurrentIndex(0)
            self._on_preview_file_selected(0)
        else:
            self.preview.show_file(None)
            self.preview.header.setText(
                "Файлов для предпросмотра не найдено.\n"
                "Экспортируйте что-нибудь или нажмите «Открыть файл…».")


    def _purge_old_exports(self):
        """Удаляет файлы старше 7 дней в WORKSPACE."""
        import time as _t
        cutoff = _t.time() - 7 * 86400
        exts = (".obj", ".usda", ".usd", ".gml", ".geojson",
                ".dxf", ".ifc", ".ply")
        removed = 0
        sizes = 0
        for p in WORKSPACE.glob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                try:
                    if p.stat().st_mtime < cutoff:
                        sz = p.stat().st_size
                        p.unlink()
                        removed += 1
                        sizes += sz
                except Exception:
                    pass
        self.log.log(
            f"Удалено старых экспортов: {removed} "
            f"({sizes/1024/1024:.1f} МБ)", "SUCCESS")
        self._exported_files.clear()
        self.refresh_preview_list()

    def _clear_preview_list(self):
        self._exported_files.clear()
        self.refresh_preview_list()

    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть файл для предпросмотра",
            str(WORKSPACE),
            "Все поддерживаемые (*.obj *.stl *.ply *.pcd *.las *.laz "
            "*.xyz *.usda *.usd *.geojson *.gml *.dxf *.ifc *.json "
            "*.txt);;Все файлы (*)")
        if not path:
            return
        p = Path(path)
        if str(p) not in self._exported_files:
            self._exported_files.append(str(p))
        self.refresh_preview_list()
        idx = self.file_combo.findData(str(p))
        if idx >= 0:
            self.file_combo.setCurrentIndex(idx)
        else:
            self.preview.show_file(str(p))

    def _open_training_output(self):
        if not TRAINING_OUTPUT_DIR.exists():
            self.log.log("Папка training_output ещё не создана.",
                         "WARNING")
            return
        files = sorted(
            [f for f in TRAINING_OUTPUT_DIR.rglob("*") if f.is_file()])
        if not files:
            self.log.log("В training_output нет файлов.", "WARNING")
            return
        for f in files:
            if f.suffix.lower() == ".json":
                self._exported_files.append(str(f))
                self.refresh_preview_list()
                idx = self.file_combo.findData(str(f))
                if idx >= 0:
                    self.file_combo.setCurrentIndex(idx)
                else:
                    self.preview.show_file(str(f))
                return
        f = files[0]
        self._exported_files.append(str(f))
        self.refresh_preview_list()
        idx = self.file_combo.findData(str(f))
        if idx >= 0:
            self.file_combo.setCurrentIndex(idx)
        else:
            self.preview.show_file(str(f))

    def _on_preview_file_selected(self, idx):
        if idx < 0:
            return
        path = self.file_combo.itemData(idx)
        if path:
            self.preview.show_file(path)

    def _input_path(self):
        cropped_mem = getattr(self.main_window, "cropped_pcd_path", None)
        if (self.use_cropped.isChecked() and cropped_mem
                and Path(cropped_mem).exists()):
            return cropped_mem
        cropped_file = WORKSPACE / "pointcloud_cropped.ply"
        if self.use_cropped.isChecked() and cropped_file.exists():
            return str(cropped_file)
        clean_mem = getattr(self.main_window, "clean_pcd_path", None)
        if clean_mem and Path(clean_mem).exists():
            return clean_mem
        clean_file = WORKSPACE / "pointcloud_clean.ply"
        if clean_file.exists():
            return str(clean_file)
        if self.main_window.pcd is not None:
            p = str(WORKSPACE / "pointcloud_current.ply")
            o3d.io.write_point_cloud(p, self.main_window.pcd)
            return p
        preparer = getattr(self.main_window, "dataset_preparer", None)
        if preparer is not None:
            for split in ("train", "val", "test"):
                folder = preparer.root / "processed" / split
                if folder.exists():
                    files = sorted(folder.glob("*.ply"))
                    if files:
                        return str(files[0])
        if preparer is not None:
            raw = preparer.root / "raw"
            if raw.exists():
                for ext in SUPPORTED_INPUT_FORMATS:
                    files = sorted(raw.glob(f"*{ext}"))
                    if files:
                        return str(files[0])
        return None

    def export(self):
        input_path = self._input_path()
        if not input_path:
            self.log.log("Нет данных.", "ERROR")
            return
        self.log.log(f"Источник: {input_path}", "INFO")
        worker = WorkerThread(self._task)
        worker.log_signal.connect(self.log.log)
        worker.task_done.connect(self._on_done)
        worker.start()
        self._worker = worker

    def _task(self, emit_log, emit_progress):
        input_path = self._input_path()
        if not input_path:
            emit_log("Нет данных.", "ERROR")
            return False
        fmt = self.combo.currentText()
        name = self.name.text()
        self._last_output = None
        self._last_format = fmt

        def _finish(res, label, default_path=None):
            if res.get("ok"):
                emit_log(f"{label}: {res['path']}", "SUCCESS")
                self._last_output = res["path"]
                return True
            msg = res.get("error", "неизвестная ошибка")
            emit_log(f"{label}: ОШИБКА — {msg}", "ERROR")
            if default_path:
                emit_log(f"        ожидаемый путь: {default_path}",
                         "WARNING")
            return False

        try:
            if "CityGML" in fmt:
                out = WORKSPACE / f"{name}.gml"
                res = self.exporter.export_to_citygml(input_path, out)
                _finish(res, "CityGML", out)
            elif "USD" in fmt:
                out = WORKSPACE / f"{name}.usda"
                res = self.exporter.export_to_usd(input_path, out)
                _finish(res, "USD", out)
            elif "OBJ" in fmt:
                out = WORKSPACE / f"{name}.obj"
                res = self.exporter.export_to_obj(input_path, out)
                _finish(res, "OBJ", out)
            elif "GeoJSON" in fmt:
                out = WORKSPACE / f"{name}.geojson"
                res = self.exporter.export_to_geojson(input_path, out)
                _finish(res, "GeoJSON", out)
            elif "PostGIS" in fmt:
                res = self.exporter.export_to_postgis(
                    input_path, DB_PARAMS, name)
                _finish(res, "PostGIS")
            elif "DXF" in fmt:
                out = WORKSPACE / f"{name}.dxf"
                res = self.exporter.export_to_dxf(input_path, out)
                _finish(res, "DXF", out)
            elif "IFC" in fmt:
                out = WORKSPACE / f"{name}.ifc"
                res = self.exporter.export_to_ifc(input_path, out)
                _finish(res, "IFC", out)
            emit_progress(100)
            # Итоговый отчёт оператору
            if self._last_output:
                p = Path(self._last_output)
                sz = ExportPreviewViewer._human_size(p)
                emit_log(
                    f"ГОТОВО: {fmt} — {p.name} ({sz}). "
                    f"Откройте файл на вкладке «6. Экспорт» → "
                    f"Предпросмотр файлов.", "SUCCESS")
            else:
                emit_log(
                    f"ЭКСПОРТ {fmt} НЕ ВЫПОЛНЕН. Проверьте, "
                    f"установлена ли соответствующая утилита "
                    f"(3dfier / GDAL / pxr / Cloud2BIM / 3d_cad_pointcloud) "
                    f"и корректны ли исходные данные.", "ERROR")
            return True
        except Exception as e:
            import traceback
            emit_log(f"Ошибка экспорта: {e}", "ERROR")
            emit_log(traceback.format_exc(), "ERROR")
            return False

    def _on_done(self, ok):
        """После экспорта — принудительно обновляем список. 
        PATCHED_EXPORT_UI"""
        if not ok:
            return
        if self._last_output:
            p = Path(self._last_output)
            if p.exists():
                if self._last_output not in self._exported_files:
                    self._exported_files.append(self._last_output)
                # Дописываем метку вручную, чтобы не фильтровалось по
                # возрасту (файл только что создан, но всё равно)
        # Принудительно перечитываем список
        self.refresh_preview_list()
        # Выбираем последний экспортированный файл
        if self._last_output:
            idx = self.file_combo.findData(self._last_output)
            if idx >= 0:
                self.file_combo.setCurrentIndex(idx)
                self._on_preview_file_selected(idx)
                self.log.log(
                    f"Предпросмотр открыт: "
                    f"{Path(self._last_output).name}", "SUCCESS")
            else:
                # Если фильтр отсеял — открываем напрямую
                self.preview.show_file(self._last_output)

    def export_all(self):
        if self._input_path() is None:
            self.log.log("Нет данных.", "ERROR")
            return
        for fmt in ["USD (OpenUSD)", "OBJ (меш)", "GeoJSON (веб)",
                    "DXF (CAD)", "IFC (BIM)", "CityGML (3dfier)"]:
            self.combo.setCurrentText(fmt)
            self.export()


# ============================================================================
# Вкладка 7 — База эталонов
# ============================================================================

class ReferenceDBTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.db = ReferenceDatabase(REFERENCE_DB_PATH)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        info = QLabel(
            f"Эталоны: PostgreSQL (если доступен) или "
            f"SQLite-фолбэк: {REFERENCE_DB_PATH}")
        info.setWordWrap(True)
        layout.addWidget(info)
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Категория", "Название", "Путь", "Дата"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        layout.addWidget(self.table)
        # ── Скачивание OSM / SRTM ─────────────────────────────
        dl_gb = QGroupBox("Скачивание эталонов OSM/SRTM")
        dl_l = QVBoxLayout(dl_gb)
        dl_l.addWidget(QLabel(
            "Источник: файл LAS на вкладке 1 (bbox определяется "
            "автоматически)."))

        dl_row1 = QHBoxLayout()
        self.btn_download_osm = QPushButton("🌐 Скачать OSM")
        self.btn_download_osm.setToolTip(
            "Buildings / roads / waterways через Overpass API")
        self.btn_download_osm.clicked.connect(
            lambda: self._download_external("osm"))
        dl_row1.addWidget(self.btn_download_osm)

        self.btn_download_srtm = QPushButton("⛰ Скачать SRTM 30m")
        self.btn_download_srtm.setToolTip(
            "Скачивает тайл с AWS S3 (в обход blocked-серверов)")
        self.btn_download_srtm.clicked.connect(
            lambda: self._download_external("srtm"))
        dl_row1.addWidget(self.btn_download_srtm)

        self.btn_download_all = QPushButton("⬇ Скачать всё")
        self.btn_download_all.clicked.connect(
            lambda: self._download_external("all"))
        dl_row1.addWidget(self.btn_download_all)
        dl_row1.addStretch()
        dl_l.addLayout(dl_row1)

        self.download_log = QTextEdit()
        self.download_log.setReadOnly(True)
        self.download_log.setMaximumHeight(100)
        self.download_log.setFont(QFont("Monospace", 8))
        self.download_log.setStyleSheet(
            "background:#1a1a1a; color:#d4d4d4;")
        dl_l.addWidget(self.download_log)
        layout.addWidget(dl_gb)

        bl = QHBoxLayout()
        b1 = QPushButton("Добавить эталон")
        b1.clicked.connect(self.add_cloud)
        bl.addWidget(b1)
        b2 = QPushButton("Обновить")
        b2.clicked.connect(self.refresh)
        bl.addWidget(b2)
        bl.addStretch()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #808080;")
        bl.addWidget(self.status_label)
        layout.addLayout(bl)
        self.log = LogWidget()
        layout.addWidget(self.log)
        self.setLayout(layout)
        self.refresh()


    # ── Скачивание OSM/SRTM ─────────────────────────────────────
    def _download_external(self, kind):
        """kind ∈ {"osm", "srtm", "all"}."""
        las_path = getattr(self.main_window, "pcd_path", None)
        if not las_path or not Path(las_path).exists():
            QMessageBox.warning(
                self, "Нет исходного файла",
                "Сначала загрузите LAS на вкладке 1.")
            return

        self.download_log.clear()
        self._dl("Старт: " + kind)
        self._dl(f"Источник: {las_path}")

        worker = WorkerThread(self._download_task, las_path, kind)
        worker.log_signal.connect(lambda m, lv="INFO":
                                  self._dl(f"[{lv}] {m}"))
        worker.task_done.connect(self._on_download_done)
        worker.start()
        self._worker = worker

    def _dl(self, msg):
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%H:%M:%S")
        self.download_log.append(f"[{ts}] {msg}")
        self.download_log.verticalScrollBar().setValue(
            self.download_log.verticalScrollBar().maximum())

    def _download_task(self, emit_log, emit_progress, las_path, kind):
        """Запускается в фоновом потоке."""
        import subprocess
        from pathlib import Path as _P

        # Используем уже готовый скрипт setup_references_v5.py
        script = _P.home() / "tools" / "setup_references_v5.py"
        if not script.exists():
            emit_log(f"Скрипт не найден: {script}", "ERROR")
            return False

        emit_log("Запускаю setup_references_v5.py...")
        try:
            r = subprocess.run(
                ["python", str(script), las_path, "--epsg", "32639"],
                capture_output=True, text=True, timeout=900)
            for line in r.stdout.splitlines():
                emit_log(line)
            if r.returncode != 0:
                emit_log(f"Код выхода: {r.returncode}", "ERROR")
                return False
            emit_log("Готово", "SUCCESS")
            emit_progress(100)
            return True
        except Exception as e:
            emit_log(f"Ошибка: {e}", "ERROR")
            return False

    def _on_download_done(self, ok):
        if ok:
            self._dl("── УСПЕХ ──")
            self.refresh()  # обновим таблицу
        else:
            self._dl("── ОШИБКА ──")

    def add_cloud(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Эталон", str(WORKSPACE),
            "Point Cloud (*.las *.laz *.ply *.pcd);;All Files (*)")
        if not path:
            return
        hdb = getattr(self.main_window, "history_db", None)
        try:
            if hdb and hdb.is_available:
                data = PointCloudLoader.load(path)
                classes = (list(np.unique(data["classification"]))
                           if data.get("classification") is not None
                           else [])
                scanning_type = getattr(
                    self.main_window, "scanning_type", "unknown")
                crs = (getattr(self.main_window.crs_detector, "crs", None)
                       if self.main_window.crs_detector else None)
                pts = data["points"]
                bbox = (float(pts[:, 0].min()), float(pts[:, 1].min()),
                        float(pts[:, 2].min()), float(pts[:, 0].max()),
                        float(pts[:, 1].max()), float(pts[:, 2].max()))
                bbox_line = TrainingHistoryDB._bbox_to_box3d(bbox)
                with hdb.conn.cursor() as cur:
                    cur.execute(f"""
                        INSERT INTO {hdb.schema_r}.references_clouds
                            (name, path, format, n_points, scanning_type,
                             crs_epsg, bbox, classes, description)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::box3d,
                                %s::jsonb, %s)
                    """, (Path(path).stem, str(path), data["format"],
                          len(data["points"]), scanning_type, crs,
                          bbox_line, json.dumps(classes),
                          "Добавлено вручную"))
                self.log.log(f"Добавлено в Postgres: {Path(path).name}",
                             "SUCCESS")
            else:
                self.db.add_reference_cloud(
                    Path(path).stem, path, "Добавлено вручную")
                self.log.log(f"Добавлено в SQLite: {Path(path).name}",
                             "SUCCESS")
            self.refresh()
        except Exception as e:
            self.log.log(f"Ошибка: {e}", "ERROR")

    def refresh(self):
        rows = []
        hdb = getattr(self.main_window, "history_db", None)
        if hdb and hdb.is_available:
            try:
                with hdb.conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT id, name, path, created_at
                        FROM {hdb.schema_r}.references_clouds
                        ORDER BY created_at DESC
                    """)
                    for r in cur.fetchall():
                        rows.append(("clouds", r[1], r[2], r[3]))
                    cur.execute(f"""
                        SELECT id, name, path, created_at
                        FROM {hdb.schema_r}.references_gis
                        ORDER BY created_at DESC
                    """)
                    for r in cur.fetchall():
                        rows.append(("gis", r[1], r[2], r[3]))
                    cur.execute(f"""
                        SELECT id, name, path, created_at
                        FROM {hdb.schema_r}.references_dem
                        ORDER BY created_at DESC
                    """)
                    for r in cur.fetchall():
                        rows.append(("dem", r[1], r[2], r[3]))
                    cur.execute(f"""
                        SELECT id, name, path, created_at
                        FROM {hdb.schema_r}.references_masks
                        ORDER BY created_at DESC
                    """)
                    for r in cur.fetchall():
                        rows.append(("masks", r[1], r[2], r[3]))
                self.status_label.setText(
                    f"Источник: PostgreSQL ({len(rows)} записей)")
            except Exception as e:
                self.log.log(f"Ошибка чтения Postgres: {e}", "WARNING")
        if not rows:
            try:
                all_refs = self.db.list_all()
                for table_name, records in all_refs.items():
                    for r in records:
                        rows.append((table_name, r[1], r[2], r[-1]))
                self.status_label.setText("Источник: SQLite (фолбэк)")
            except Exception:
                pass
        self.table.setRowCount(len(rows))
        for i, (cat, name, path, dt) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(cat))
            self.table.setItem(i, 1, QTableWidgetItem(str(name)))
            self.table.setItem(i, 2, QTableWidgetItem(str(path)))
            self.table.setItem(i, 3, QTableWidgetItem(str(dt)))


# ============================================================================
# Вкладка 8 — История обучений
# ============================================================================

class HistoryTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(QLabel("Метод:"))
        self.f_method = QComboBox()
        self.f_method.addItems(["—", "M1", "M2", "M3", "M4", "M5", "M6",
                                "combined"])
        top.addWidget(self.f_method)
        top.addWidget(QLabel("Тип сканирования:"))
        self.f_type = QComboBox()
        self.f_type.addItems(["—", "ALS_manned", "ALS_uav", "MLS",
                              "TLS", "handheld", "stationary", "unknown"])
        top.addWidget(self.f_type)
        top.addWidget(QLabel("Статус:"))
        self.f_status = QComboBox()
        self.f_status.addItems(["—", "success", "failed", "running",
                                "partial"])
        top.addWidget(self.f_status)
        b_apply = QPushButton("Применить")
        b_apply.clicked.connect(self.reload)
        top.addWidget(b_apply)
        b_export = QPushButton("Экспорт в CSV")
        b_export.clicked.connect(self._export_csv)
        top.addWidget(b_export)
        top.addStretch()
        layout.addLayout(top)

        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "#", "Начало", "Метод", "Тип сканирования", "CRS",
            "Точек", "Длит., с", "Статус", "Папка"
        ])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(self.table)

        bl = QHBoxLayout()
        b_open = QPushButton("Открыть папку прогона")
        b_open.clicked.connect(self._open_run_dir)
        bl.addWidget(b_open)
        b_metrics = QPushButton("Показать метрики (JSON)")
        b_metrics.clicked.connect(self._show_metrics)
        bl.addWidget(b_metrics)
        b_repeat = QPushButton("Повторить прогон")
        b_repeat.clicked.connect(self._repeat_run)
        bl.addWidget(b_repeat)
        bl.addStretch()
        layout.addLayout(bl)

        self.info = QLabel("")
        self.info.setStyleSheet("color: #dcdcaa; padding: 4px;")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        # ── Сводка по качеству + рекомендации ────────────────────
        summary_gb = QGroupBox("history_summary_panel — Сводка качества "
                                "и рекомендации")
        summary_l = QVBoxLayout(summary_gb)
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(6)
        self.summary_table.setHorizontalHeaderLabels([
            "Метод", "Тип", "Прогонов", "Средний quality",
            "Лучший quality", "Рекомендация"])
        self.summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        summary_l.addWidget(self.summary_table)
        layout.addWidget(summary_gb)

        self.setLayout(layout)
        self.reload()


    def _refresh_summary(self):
        """Сводка по качеству обучений + рекомендации."""
        hdb = getattr(self.main_window, "history_db", None)
        if not hdb or not hdb.is_available:
            self.summary_table.setRowCount(0)
            return
        try:
            with hdb.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT method, scanning_type, COUNT(*) AS n,
                           AVG((metrics->>'quality_score')::real) AS avg_q,
                           MAX((metrics->>'quality_score')::real) AS max_q,
                           MAX(started_at) AS last_run
                    FROM {hdb.schema_h}.training_runs
                    WHERE status='success'
                    GROUP BY method, scanning_type
                    ORDER BY method, scanning_type
                """)
                rows = cur.fetchall()
        except Exception as e:
            self.info.setText(f"Ошибка сводки: {e}")
            return

        self.summary_table.setRowCount(len(rows))
        for i, (m, t, n, avg_q, max_q, last) in enumerate(rows):
            avg_q = avg_q or 0.0
            max_q = max_q or 0.0
            if max_q >= 0.7:
                rec = "✅ ДОСТАТОЧНО — можно двигаться дальше"
            elif max_q >= 0.4:
                rec = "⚠ ДОБАВЬТЕ эталонов с другой геометрией"
            else:
                rec = "❌ ПЛОХО — проверьте эталоны и параметры"
            cells = [str(m), str(t or "—"), str(n),
                     f"{avg_q:.3f}", f"{max_q:.3f}", rec]
            for j, v in enumerate(cells):
                self.summary_table.setItem(i, j, QTableWidgetItem(v))
        self.info.setText(
            f"История: {len(rows)} комбинаций метод/тип. "
            f"Обновлено: {datetime.now().strftime('%H:%M:%S')}")

    def _current_filters(self):
        m = self.f_method.currentText()
        t = self.f_type.currentText()
        s = self.f_status.currentText()
        return (None if m == "—" else m,
                None if t == "—" else t,
                None if s == "—" else s)

    def reload(self):
        hdb = getattr(self.main_window, "history_db", None)
        if not hdb or not hdb.is_available:
            self.info.setText(
                "PostgreSQL недоступен — история обучений не ведётся. "
                "Установите psycopg и поднимите БД city3d.")
            self.table.setRowCount(0)
            return
        m, t, s = self._current_filters()
        runs = hdb.list_runs(limit=500, method=m, scanning_type=t,
                             status=s)
        self.table.setRowCount(len(runs))
        for i, r in enumerate(runs):
            self.table.setItem(i, 0, QTableWidgetItem(str(r["id"])))
            self.table.setItem(i, 1, QTableWidgetItem(str(r["started_at"])))
            self.table.setItem(i, 2, QTableWidgetItem(str(r["method"])))
            self.table.setItem(i, 3,
                               QTableWidgetItem(str(r["scanning_type"])))
            self.table.setItem(i, 4,
                               QTableWidgetItem(str(r["crs_epsg"])))
            self.table.setItem(i, 5,
                               QTableWidgetItem(str(r["n_points"])))
            dur = r["duration_sec"]
            self.table.setItem(i, 6,
                               QTableWidgetItem(
                                   f"{dur:.1f}" if dur else "—"))
            self.table.setItem(i, 7, QTableWidgetItem(str(r["status"])))
            self.table.setItem(i, 8,
                               QTableWidgetItem(str(r["output_dir"])))
        self.info.setText(f"Найдено прогонов: {len(runs)}")
        # Обновляем сводку
        try:
            self._refresh_summary()
        except Exception as e:
            self.info.setText(f"{self.info.text()} | сводка: {e}")

    def _selected_run(self):
        rows = self.table.selectionModel().selectedRows() \
            if self.table.selectionModel() else []
        if not rows:
            return None
        row = rows[0].row()
        try:
            run_id = int(self.table.item(row, 0).text())
        except Exception:
            return None
        hdb = getattr(self.main_window, "history_db", None)
        if not hdb or not hdb.is_available:
            return None
        try:
            with hdb.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT id, started_at, method, scanning_type,
                           crs_epsg, n_points, duration_sec, status,
                           output_dir, hyperparams, metrics,
                           source_path, dataset_root
                    FROM {hdb.schema_h}.training_runs WHERE id=%s
                """, (run_id,))
                cols = [d.name for d in cur.description]
                r = cur.fetchone()
                if not r:
                    return None
                return dict(zip(cols, r))
        except Exception:
            return None

    def _open_run_dir(self):
        r = self._selected_run()
        if not r or not r.get("output_dir"):
            QMessageBox.warning(self, "Нет данных", "Выберите прогон.")
            return
        p = Path(r["output_dir"])
        if not p.exists():
            QMessageBox.warning(self, "Нет папки", f"Папка не найдена: {p}")
            return
        try:
            subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            QMessageBox.information(self, "Путь", f"{p}\n\n{e}")

    def _show_metrics(self):
        r = self._selected_run()
        if not r:
            return
        txt = json.dumps({
            "run_id": r["id"],
            "method": r["method"],
            "scanning_type": r["scanning_type"],
            "crs_epsg": r["crs_epsg"],
            "hyperparams": r["hyperparams"],
            "metrics": r["metrics"],
            "duration_sec": r["duration_sec"],
            "status": r["status"],
        }, indent=2, ensure_ascii=False, default=str)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Прогон #{r['id']}")
        dlg.resize(720, 480)
        v = QVBoxLayout(dlg)
        view = QTextEdit()
        view.setReadOnly(True)
        view.setFont(QFont("Monospace", 9))
        view.setPlainText(txt)
        v.addWidget(view)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        dlg.exec_()

    def _repeat_run(self):
        r = self._selected_run()
        if not r:
            return
        QMessageBox.information(
            self, "Повторить прогон",
            f"Прогон #{r['id']} ({r['method']}, {r['scanning_type']}).\n\n"
            "Гиперпараметры из этого прогона будут предложены при "
            "следующем запуске соответствующего метода.")

    def _export_csv(self):
        hdb = getattr(self.main_window, "history_db", None)
        if not hdb or not hdb.is_available:
            QMessageBox.warning(self, "Ошибка", "Postgres недоступен.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить CSV",
            str(WORKSPACE / "training_history.csv"),
            "CSV (*.csv)")
        if not path:
            return
        runs = hdb.list_runs(limit=5000)
        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                if runs:
                    keys = list(runs[0].keys())
                    wr.writerow(keys)
                    for r in runs:
                        wr.writerow([r.get(k) for k in keys])
            QMessageBox.information(self, "Готово", f"CSV: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", str(e))


# ============================================================================
# Главное окно
# ============================================================================



# ═══════════════════════════════════════════════════════════════════
# PATCHED_SETTINGS_TAB_V2 — вкладка «9. Настройки»
# ═══════════════════════════════════════════════════════════════════

_NL = chr(10)


class SettingsTab(QWidget):
    """Настройки: директории, БД, Git, объём."""

    SETTINGS_FILE = Path.home() / ".config" / "city3d" / "settings.json"

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._dir_rows = {}
        self._build_ui()
        self.refresh_all()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 4, 4, 4)

        # ═════ БЛОК 1: Директории ═════
        dir_gb = QGroupBox("📁 Директории и папки программы")
        dir_l = QVBoxLayout(dir_gb)
        hint = QLabel(
            "Изменения сохраняются в " + str(self.SETTINGS_FILE)
            + ". Пути применяются при следующем запуске. "
            "БД — можно менять сразу, без перезапуска.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#dcdcaa; padding:4px; "
                           "background:#2d2d30; border-radius:3px;")
        dir_l.addWidget(hint)
        self.dir_table = QTableWidget()
        self.dir_table.setColumnCount(4)
        self.dir_table.setHorizontalHeaderLabels(
            ["Назначение", "Путь", "Существует?", "Действия"])
        hh = self.dir_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        dir_l.addWidget(self.dir_table)
        self._dir_defs = [
            ("project_root",   "Проект (root)",
             lambda: str(PROJECT_ROOT), True),
            ("workspace",      "Рабочая папка",
             lambda: str(WORKSPACE), True),
            ("datasets_dir",   "Датасеты",
             lambda: str(DATASETS_DIR), True),
            ("training_output_dir", "Результаты обучения",
             lambda: str(TRAINING_OUTPUT_DIR), True),
            ("slices_dir",     "Срезы (HAG)",
             lambda: str(SLICES_DIR), True),
            ("projects_dir",   "AI-проекты",
             lambda: str(PROJECTS_DIR), True),
            ("reference_db",   "SQLite-эталоны",
             lambda: str(REFERENCE_DB_PATH), False),
            ("epsg_catalog",   "Справочник EPSG",
             lambda: str(EPSG_CATALOG_PATH), False),
            ("settings_dir",   "Настройки (app)",
             lambda: str(Path.home() / ".config" / "city3d"), True),
            ("settings_file",  "settings.json",
             lambda: str(self.SETTINGS_FILE), False),
            ("hf_cache",       "HuggingFace кэш",
             lambda: str(_RESOLVED_PATHS["hf_cache"]), True),
        ]
        self._render_dir_table()
        dir_btns = QHBoxLayout()
        b_save = QPushButton("💾 Сохранить пути (перезапуск)")
        b_save.setStyleSheet(
            "background:#0e639c; color:white; font-weight:bold;")
        b_save.clicked.connect(self._save_paths)
        dir_btns.addWidget(b_save)
        b_reset = QPushButton("↺ Сбросить к дефолтам")
        b_reset.clicked.connect(self._reset_paths)
        dir_btns.addWidget(b_reset)
        b_open_cfg = QPushButton("📂 Папка настроек")
        b_open_cfg.clicked.connect(self._open_cfg_dir)
        dir_btns.addWidget(b_open_cfg)
        dir_btns.addStretch()
        dir_l.addLayout(dir_btns)
        layout.addWidget(dir_gb)

        # ═════ БЛОК 2: БД ═════
        db_gb = QGroupBox("🗄 База данных PostgreSQL")
        db_l = QFormLayout(db_gb)
        self.db_status = QLabel("—")
        self.db_status.setStyleSheet(
            "padding:6px; font-weight:bold; border-radius:3px; "
            "background:#252526;")
        db_l.addRow("Статус:", self.db_status)
        self.db_name = QLineEdit()
        self.db_host = QLineEdit()
        self.db_port = QLineEdit()
        self.db_user = QLineEdit()
        self.db_pass = QLineEdit()
        self.db_pass.setEchoMode(QLineEdit.Password)
        self.db_schema_refs = QLineEdit()
        self.db_schema_hist = QLineEdit()
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("БД:"))
        row1.addWidget(self.db_name, 1)
        row1.addWidget(QLabel("Хост:"))
        row1.addWidget(self.db_host, 1)
        row1.addWidget(QLabel("Порт:"))
        self.db_port.setMaximumWidth(80)
        row1.addWidget(self.db_port)
        db_l.addRow("Подключение:", row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Польз.:"))
        row2.addWidget(self.db_user, 1)
        row2.addWidget(QLabel("Пароль:"))
        row2.addWidget(self.db_pass, 1)
        db_l.addRow("", row2)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Schema refs:"))
        row3.addWidget(self.db_schema_refs, 1)
        row3.addWidget(QLabel("Schema history:"))
        row3.addWidget(self.db_schema_hist, 1)
        db_l.addRow("", row3)
        db_btns = QHBoxLayout()
        b_test = QPushButton("🧪 Тест подключения")
        b_test.clicked.connect(self._test_db)
        db_btns.addWidget(b_test)
        b_apply = QPushButton("↻ Применить сейчас")
        b_apply.setStyleSheet(
            "background:#0e639c; color:white; font-weight:bold;")
        b_apply.clicked.connect(self._apply_db_now)
        db_btns.addWidget(b_apply)
        b_save_db = QPushButton("💾 Сохранить")
        b_save_db.clicked.connect(self._save_db)
        db_btns.addWidget(b_save_db)
        db_btns.addStretch()
        db_l.addRow("", db_btns)
        self.db_log = LogWidget()
        self.db_log.setMaximumHeight(100)
        db_l.addRow(self.db_log)
        layout.addWidget(db_gb)

        # ═════ БЛОК 3: Git ═════
        git_gb = QGroupBox("📦 Git репозиторий")
        git_l = QFormLayout(git_gb)
        self.git_dir = QLineEdit()
        self.git_dir.setText(str(Path(__file__).parent.resolve()))
        git_l.addRow("Папка репо:", self.git_dir)
        self.git_info = QTextEdit()
        self.git_info.setReadOnly(True)
        self.git_info.setFont(QFont("Monospace", 9))
        self.git_info.setMaximumHeight(200)
        self.git_info.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; border:1px solid #3c3c3c;")
        git_l.addRow(self.git_info)
        git_btns = QHBoxLayout()
        b_rf = QPushButton("🔄 Обновить")
        b_rf.clicked.connect(self._refresh_git)
        git_btns.addWidget(b_rf)
        b_log = QPushButton("📜 git log (30)")
        b_log.clicked.connect(self._show_git_log)
        git_btns.addWidget(b_log)
        b_status = QPushButton("📊 git status")
        b_status.clicked.connect(self._show_git_status)
        git_btns.addWidget(b_status)
        git_btns.addStretch()
        git_l.addRow("", git_btns)
        layout.addWidget(git_gb)

        # ═════ БЛОК 4: Объём ═════
        vol_gb = QGroupBox("💾 Объём ресурсов")
        vol_l = QVBoxLayout(vol_gb)
        self.vol_table = QTableWidget()
        self.vol_table.setColumnCount(3)
        self.vol_table.setHorizontalHeaderLabels(
            ["Ресурс", "Размер", "Свободно на диске"])
        self.vol_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        vol_l.addWidget(self.vol_table)
        vol_btns = QHBoxLayout()
        b_rec = QPushButton("🔄 Пересчитать")
        b_rec.clicked.connect(self._refresh_volumes)
        vol_btns.addWidget(b_rec)
        b_disk = QPushButton("📊 По дискам")
        b_disk.clicked.connect(self._show_disks)
        vol_btns.addWidget(b_disk)
        vol_btns.addStretch()
        vol_l.addLayout(vol_btns)
        layout.addWidget(vol_gb)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    # ── Директории ────────────────────────────────────────────
    def _render_dir_table(self):
        rows = self._dir_defs
        self.dir_table.setRowCount(len(rows))
        self._dir_rows = {}
        for i, (key, label, path_fn, is_dir) in enumerate(rows):
            path_str = path_fn()
            p = Path(path_str)
            exists = p.exists()
            it_label = QTableWidgetItem(label)
            it_path = QTableWidgetItem(path_str)
            it_exists = QTableWidgetItem("✅ да" if exists else "❌ нет")
            if not exists:
                it_exists.setForeground(Qt.yellow)
            self.dir_table.setItem(i, 0, it_label)
            self.dir_table.setItem(i, 1, it_path)
            self.dir_table.setItem(i, 2, it_exists)
            cell = QWidget()
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)
            if exists:
                b_open = QPushButton("📂")
                b_open.setFixedWidth(30)
                b_open.clicked.connect(
                    lambda _, pp=path_str: self._open_path(pp))
                cl.addWidget(b_open)
            if key in ("project_root", "workspace", "datasets_dir",
                       "training_output_dir", "projects_dir",
                       "reference_db", "epsg_catalog"):
                b_edit = QPushButton("✎")
                b_edit.setFixedWidth(30)
                b_edit.clicked.connect(
                    lambda _, k=key, d=is_dir: self._edit_path(k, d))
                cl.addWidget(b_edit)
            cl.addStretch()
            self.dir_table.setCellWidget(i, 3, cell)
            self._dir_rows[key] = (i, path_fn)

    def _open_path(self, path_str):
        p = Path(path_str)
        if p.is_file():
            p = p.parent
        if not p.exists():
            QMessageBox.warning(self, "Ошибка", "Не найдено: " + str(p))
            return
        try:
            subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            QMessageBox.information(
                self, "Путь", str(p) + _NL + _NL + str(e))

    def _edit_path(self, key, is_dir):
        if key in ("settings_dir", "settings_file"):
            QMessageBox.information(self, "Настройки", "Расположение настроек фиксировано: ~/.config/city3d/settings.json")
            return
        i, path_fn = self._dir_rows[key]
        cur = path_fn()
        if is_dir:
            new = QFileDialog.getExistingDirectory(
                self, "Папка для " + key,
                str(Path(cur).parent if Path(cur).exists()
                    else Path.home()))
        else:
            new, _ = QFileDialog.getSaveFileName(
                self, "Файл для " + key, cur, "All Files (*)")
        if not new:
            return
        self.dir_table.setItem(i, 1, QTableWidgetItem(new))
        self._save_path_to_file(key, new)
        self.dir_table.item(i, 2).setText(
            "✅ да" if Path(new).exists() else "⚠ нет")

    def _save_path_to_file(self, key, value):
        try:
            self.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if self.SETTINGS_FILE.exists():
                try:
                    data = _json_top.loads(
                        self.SETTINGS_FILE.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            data.setdefault("paths", {})[key] = value
            data.setdefault("version", "v2")
            save_settings(data, self.SETTINGS_FILE)
            print("[SETTINGS] " + key + " = " + value)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка записи", str(e))

    def _save_paths(self):
        QMessageBox.information(
            self, "Готово",
            "Пути сохранены: " + _NL + str(self.SETTINGS_FILE) + _NL
            + "Изменения применятся при следующем запуске.")
        self.refresh_all()

    def _reset_paths(self):
        ans = QMessageBox.question(
            self, "Сбросить?",
            "Удалить сохранённые пути?",
            QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        try:
            if self.SETTINGS_FILE.exists():
                data = _json_top.loads(
                    self.SETTINGS_FILE.read_text(encoding="utf-8"))
                data.pop("paths", None)
                save_settings(data, self.SETTINGS_FILE)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", str(e))
        self._render_dir_table()

    def _open_cfg_dir(self):
        d = Path.home() / ".config" / "city3d"
        d.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(["xdg-open", str(d)])
        except Exception as e:
            QMessageBox.information(
                self, "Путь", str(d) + _NL + _NL + str(e))

    # ── БД ────────────────────────────────────────────────────
    def _load_db_to_form(self):
        p = DB_PARAMS
        self.db_name.setText(p.get("dbname", "city3d"))
        self.db_host.setText(p.get("host", "localhost"))
        self.db_port.setText(str(p.get("port", "5432")))
        self.db_user.setText(p.get("user", "postgres"))
        self.db_pass.setText(p.get("password", ""))
        self.db_schema_refs.setText(p.get("schema_refs", "refs"))
        self.db_schema_hist.setText(p.get("schema_history", "history"))

    def _form_to_db(self):
        return {
            "dbname": self.db_name.text().strip(),
            "host": self.db_host.text().strip(),
            "port": self.db_port.text().strip() or "5432",
            "user": self.db_user.text().strip(),
            "password": self.db_pass.text(),
            "schema_refs": self.db_schema_refs.text().strip() or "refs",
            "schema_history": self.db_schema_hist.text().strip()
                              or "history",
        }

    def _test_db(self):
        params = self._form_to_db()
        try:
            import psycopg
        except ImportError:
            self.db_status.setText("❌ psycopg не установлен")
            self.db_status.setStyleSheet(
                "padding:6px; font-weight:bold; border-radius:3px; "
                "background:#a03030; color:white;")
            return
        dsn = ("dbname=" + params["dbname"] + " host=" + params["host"]
               + " port=" + params["port"] + " user=" + params["user"]
               + " password=" + params.get("password", ""))
        try:
            conn = psycopg.connect(dsn, connect_timeout=5)
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                ver = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema IN (%s, %s)",
                    (params["schema_refs"], params["schema_history"]))
                n_tab = cur.fetchone()[0]
            conn.close()
            self.db_status.setText(
                "✅ " + params["dbname"] + " @ "
                + params["host"] + ":" + params["port"]
                + " (" + str(n_tab) + " таблиц)")
            self.db_status.setStyleSheet(
                "padding:6px; font-weight:bold; border-radius:3px; "
                "background:#2d5c2d; color:white;")
            self.db_log.log("Тест: OK", "SUCCESS")
        except Exception as e:
            self.db_status.setText("❌ " + str(e))
            self.db_status.setStyleSheet(
                "padding:6px; font-weight:bold; border-radius:3px; "
                "background:#a03030; color:white;")
            self.db_log.log("Тест: " + str(e), "ERROR")

    def _save_db(self):
        params = self._form_to_db()
        try:
            self.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if self.SETTINGS_FILE.exists():
                try:
                    data = _json_top.loads(
                        self.SETTINGS_FILE.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            data["database"] = params
            data.setdefault("version", "v2")
            save_settings(data, self.SETTINGS_FILE)
            self.db_log.log(
                "Сохранено: " + str(self.SETTINGS_FILE), "SUCCESS")
        except Exception as e:
            self.db_log.log("Ошибка: " + str(e), "ERROR")

    def _apply_db_now(self):
        params = self._form_to_db()
        try:
            DB_PARAMS.update(params)
        except Exception:
            pass
        try:
            hdb = getattr(self.main_window, "history_db", None)
            if hdb and hdb.conn:
                try:
                    hdb.conn.close()
                except Exception:
                    pass
            new_hdb = TrainingHistoryDB(
                params, log_widget=self.main_window.log_widget)
            self.main_window.history_db = new_hdb
            if new_hdb.is_available:
                new_hdb.load_epsg_catalog()
                self.db_log.log("Переподключено", "SUCCESS")
            else:
                self.db_log.log("Не удалось", "WARNING")
        except Exception as e:
            self.db_log.log("Ошибка: " + str(e), "ERROR")
        self._test_db()

    # ── Git ───────────────────────────────────────────────────
    def _refresh_git(self):
        repo = self.git_dir.text().strip()
        if not repo or not Path(repo).exists():
            self.git_info.setPlainText("❌ Папка не найдена")
            return
        lines = ["Папка: " + repo]
        def _run_git(*args):
            try:
                r = subprocess.run(
                    ["git", "-C", repo, *args],
                    capture_output=True, text=True, timeout=10)
                return (r.stdout.strip() if r.returncode == 0
                        else "[err] " + r.stderr.strip())
            except FileNotFoundError:
                return "[err] git не найден"
            except Exception as e:
                return "[err] " + str(e)
        is_repo = _run_git("rev-parse", "--is-inside-work-tree")
        if "true" not in is_repo:
            self.git_info.setPlainText(
                "❌ Не Git-репозиторий: " + repo)
            return
        branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
        head_sha = _run_git("rev-parse", "--short", "HEAD")
        head_msg = _run_git("log", "-1", "--pretty=%s")
        head_when = _run_git("log", "-1", "--pretty=%ar")
        head_author = _run_git("log", "-1", "--pretty=%an")
        n_commits = _run_git("rev-list", "--count", "HEAD")
        n_mod = _run_git("status", "--porcelain", "--untracked-files=no")
        n_mod_cnt = len([l for l in n_mod.split(_NL) if l.strip()])
        n_unt = _run_git("status", "--porcelain", "--untracked-files=all")
        n_unt_cnt = len(
            [l for l in n_unt.split(_NL) if l.startswith("??")])
        remote = _run_git("remote", "-v")
        remotes_lines = []
        for ln in remote.split(_NL):
            if ln.strip():
                remotes_lines.append("  " + ln.strip())
        lines += [
            "Ветка:         " + branch,
            "HEAD:          " + head_sha,
            "Последний:     " + head_msg + " (" + head_when + ")",
            "Автор:         " + head_author,
            "Всего коммитов: " + n_commits,
            "Изменено:      " + str(n_mod_cnt),
            "Untracked:     " + str(n_unt_cnt),
            "",
            "Remotes:",
        ] + (remotes_lines or ["  (нет)"])
        self.git_info.setPlainText(_NL.join(lines))

    def _show_git_log(self):
        repo = self.git_dir.text().strip()
        try:
            r = subprocess.run(
                ["git", "-C", repo, "log", "-30",
                 "--pretty=%h %ad %an : %s", "--date=short"],
                capture_output=True, text=True, timeout=10)
            txt = r.stdout if r.returncode == 0 else r.stderr
        except Exception as e:
            txt = str(e)
        dlg = QDialog(self)
        dlg.setWindowTitle("git log -30")
        dlg.resize(900, 600)
        v = QVBoxLayout(dlg)
        view = QTextEdit()
        view.setReadOnly(True)
        view.setFont(QFont("Monospace", 9))
        view.setPlainText(txt)
        v.addWidget(view)
        dlg.exec_()

    def _show_git_status(self):
        repo = self.git_dir.text().strip()
        try:
            r = subprocess.run(
                ["git", "-C", repo, "status"],
                capture_output=True, text=True, timeout=10)
            txt = r.stdout if r.returncode == 0 else r.stderr
        except Exception as e:
            txt = str(e)
        dlg = QDialog(self)
        dlg.setWindowTitle("git status")
        dlg.resize(900, 600)
        v = QVBoxLayout(dlg)
        view = QTextEdit()
        view.setReadOnly(True)
        view.setFont(QFont("Monospace", 9))
        view.setPlainText(txt)
        v.addWidget(view)
        dlg.exec_()

    # ── Объём ─────────────────────────────────────────────────
    @staticmethod
    def _human_size(nbytes):
        for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
            if nbytes < 1024:
                return str(round(nbytes, 1)) + " " + unit
            nbytes /= 1024
        return str(round(nbytes, 1)) + " ПБ"

    @staticmethod
    def _dir_size_du(path):
        try:
            r = subprocess.run(
                ["du", "-sb", str(path)],
                capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                return int(r.stdout.split()[0])
        except Exception:
            pass
        total = 0
        try:
            for f in Path(path).rglob("*"):
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except Exception:
                        pass
        except Exception:
            pass
        return total

    def _refresh_volumes(self):
        items = [
            ("Проект (root)", PROJECT_ROOT),
            ("Workspace", WORKSPACE),
            ("  training_output", TRAINING_OUTPUT_DIR),
            ("  datasets", DATASETS_DIR),
            ("  slices", SLICES_DIR),
            ("AI-проекты", PROJECTS_DIR),
            ("~/.config/city3d", Path.home() / ".config" / "city3d"),
            ("HF кэш", Path.home() / ".cache" / "huggingface"),
        ]
        self.vol_table.setRowCount(len(items))
        for i, (label, path) in enumerate(items):
            p = Path(path)
            if p.exists():
                sz = self._dir_size_du(p)
                sz_txt = self._human_size(sz)
            else:
                sz_txt = "— (нет)"
            free_txt = "—"
            try:
                target = p if p.exists() else p.parent
                if target.exists():
                    du = shutil.disk_usage(str(target))
                    free_txt = (self._human_size(du.free) + " из "
                                + self._human_size(du.total))
            except Exception:
                pass
            self.vol_table.setItem(i, 0, QTableWidgetItem(label))
            self.vol_table.setItem(i, 1, QTableWidgetItem(sz_txt))
            self.vol_table.setItem(i, 2, QTableWidgetItem(free_txt))

    def _show_disks(self):
        try:
            r = subprocess.run(["df", "-h"], capture_output=True,
                               text=True, timeout=10)
            txt = r.stdout
        except Exception as e:
            txt = str(e)
        dlg = QDialog(self)
        dlg.setWindowTitle("df -h")
        dlg.resize(900, 400)
        v = QVBoxLayout(dlg)
        view = QTextEdit()
        view.setReadOnly(True)
        view.setFont(QFont("Monospace", 9))
        view.setPlainText(txt)
        v.addWidget(view)
        dlg.exec_()

    def refresh_all(self):
        for fn in (self._load_db_to_form, self._test_db,
                   self._refresh_git, self._refresh_volumes):
            try:
                fn()
            except Exception:
                pass



# CITY3D_TAB1_PATCH_1_0_0
from city3d_tab1.gui import upgrade_load_tab as _upgrade_city3d_tab1
LoadDataTab = _upgrade_city3d_tab1(LoadDataTab, globals())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pcd = None
        self.pcd_path = None
        self.clean_pcd_path = None
        self.cropped_pcd = None
        self.cropped_pcd_path = None
        self.ifc_path = None
        self.crs_detector = None
        self.dataset_preparer = None
        self.dataset_root = None
        self.pcd_data = None
        self.scanning_type = "unknown"

        # === ИСПРАВЛЕНИЕ: справочник EPSG из crslist.json ===
        self.epsg_catalog = EPSGCatalog(
            catalog_path=EPSG_CATALOG_PATH, log_widget=None)

        # PostgreSQL — история обучений (не обязательна для CRS-диалога)
        self.history_db = TrainingHistoryDB(DB_PARAMS, log_widget=None)
        if self.history_db.is_available:
            self.history_db.load_epsg_catalog()

        self.setWindowTitle(
            "City3D PointCloud Pipeline v7.5.0 — CPU M1 + M2-M6 + PG")
        self.setMinimumSize(1700, 1000)
        self.init_ui()

        # Привязываем журналы после создания UI
        self.history_db.log = self.log_widget
        self.epsg_catalog.log = self.log_widget

        n = len(self.epsg_catalog.entries)
        if n > 0:
            self.log_widget.log(
                f"Справочник EPSG готов к работе: {n} записей "
                f"(источник: {EPSG_CATALOG_PATH.name})", "SUCCESS")
        else:
            self.log_widget.log(
                f"Справочник EPSG пуст. Поместите crslist.json рядом "
                f"с программой (ожидается: {EPSG_CATALOG_PATH}).",
                "WARNING")

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #d4d4d4; }
            QTabWidget::pane { border: 1px solid #3c3c3c;
                background-color: #252526; border-radius: 4px; }
            QTabBar::tab { background-color: #2d2d30; color: #808080;
                padding: 8px 14px; margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: #252526;
                color: #4ec9b0; border-bottom: 2px solid #4ec9b0; }
            QGroupBox { color: #4ec9b0; border: 1px solid #3c3c3c;
                border-radius: 4px; margin-top: 12px;
                padding-top: 16px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px;
                padding: 0 5px; }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; border-radius: 3px;
                padding: 4px 8px; min-height: 22px; }
            QPushButton { background-color: #0e639c; color: #ffffff;
                border: none; border-radius: 4px; padding: 8px 16px;
                font-weight: bold; min-height: 28px; }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:pressed { background-color: #0a4a75; }
            QCheckBox { color: #d4d4d4; spacing: 8px; }
            QLabel { color: #d4d4d4; }
            QTableWidget { background-color: #252526; color: #d4d4d4;
                gridline-color: #3c3c3c; }
            QHeaderView::section { background-color: #2d2d30;
                color: #4ec9b0; padding: 4px;
                border: 1px solid #3c3c3c; }
            QListWidget { background-color: #252526; color: #d4d4d4;
                border: 1px solid #3c3c3c; }
        """)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        title = QLabel(
            "City3D PointCloud Pipeline v7.5.0 — CPU M1 + M2-M6 + PG")
        title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #4ec9b0; "
            "padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        main_splitter = QSplitter(Qt.Horizontal)
        self.tabs = QTabWidget()
        self.tabs.addTab(LoadDataTab(self), "1. Загрузка")
        self.tabs.addTab(PreprocessTab(self), "2. Предобработка")
        self.tabs.addTab(BoundingBoxTab(self), "3. BBOX")
        self.tabs.addTab(DatasetPrepareTab(self), "4. Датасет")
        self.tabs.addTab(TrainingTab(self), "5. Обучение M1-M6")
        self.tabs.addTab(ExportTab(self), "6. Экспорт")
        self.tabs.addTab(ReferenceDBTab(self), "7. База эталонов")
        self.tabs.addTab(HistoryTab(self), "8. История обучений")
        # PATCHED_SETTINGS_TAB_V2
        self.tabs.addTab(SettingsTab(self), "9. Настройки")
        from city3d_backend.gui_layers import LayerTab
        self.layer_tab=LayerTab(self)
        self.tabs.addTab(self.layer_tab,"10. Карта и слои")
        main_splitter.addWidget(self.tabs)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("Журнал выполнения")
        lbl.setStyleSheet(
            "color: #4ec9b0; font-weight: bold; padding: 4px;")
        rl.addWidget(lbl)
        self.log_widget = LogWidget()
        rl.addWidget(self.log_widget)
        main_splitter.addWidget(right)
        main_splitter.setSizes([1300, 400])
        layout.addWidget(main_splitter)
        self.statusBar().showMessage(f"Рабочая директория: {WORKSPACE}")
        from PyQt5.QtWidgets import QDockWidget, QShortcut
        from PyQt5.QtGui import QKeySequence
        self.help_registry=json.loads((Path(__file__).parent/"resources/help_ru.json").read_text())
        self.help_panel=QTextEdit();self.help_panel.setReadOnly(True)
        dock=QDockWidget("Справка оператора — F1",self);dock.setWidget(self.help_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea,dock)
        self.help_shortcut=QShortcut(QKeySequence("F1"),self)
        self.help_shortcut.activated.connect(self._show_context_help)
        self.tabs.currentChanged.connect(self._show_context_help)
        for index in range(self.tabs.count()):
            help_entry=self.help_registry.get(str(index),{})
            self.tabs.widget(index).setToolTip(help_entry.get("need",""))
        self._show_context_help()

    def closeEvent(self, event):
        layer_worker=getattr(getattr(self,"layer_tab",None),"worker",None)
        if hasattr(self,"layer_tab"):self.layer_tab.pending.clear()
        workers=list(_ACTIVE_WORKERS)
        if layer_worker is not None and layer_worker.isRunning():workers.append(layer_worker)
        for worker in workers:worker.requestInterruption()
        for w in workers:
            try:
                if not w.wait(3000):
                    self.statusBar().showMessage("Отмена: ожидание завершения текущей операции")
                    event.ignore()
                    return
            except Exception:
                pass
        try:
            if self.history_db and self.history_db.conn:
                self.history_db.conn.close()
        except Exception:
            pass
        event.accept()

    def _show_context_help(self, *args):
        entry=self.help_registry.get(str(self.tabs.currentIndex()),{})
        text=[entry.get("title","")]
        for key,title in (("need","Что нужно"),("action","Что произойдёт"),("result","Что получится"),("limits","Ограничения"),("next","Следующий шаг")):
            text.append(title+": "+entry.get(key,""))
        self.help_panel.setPlainText("\n\n".join(text))


# ============================================================================
# Точка входа
# ============================================================================





# BEGIN CITY3D_FIXSET_20261016
from city3d_fixes.integration import install as _install_city3d_fixes
_install_city3d_fixes(globals(), level=3)
# END CITY3D_FIXSET_20261016

# CITY3D_LOAD_LAYOUT_REPAIR_20261016
from city3d_load_layout_fix import install as _install_load_layout_fix
_install_load_layout_fix(globals())

# CITY3D_GLOBE_REPAIR_20261016
from city3d_globe_fix import install as _install_globe_fix
_install_globe_fix(globals())

def main():
    # 1) Отключаем шум KDE/KF5/Wayland и GPU у Chromium
    os.environ.setdefault(
        "QT_LOGGING_RULES",
        "kf.*=false;qt.qpa.wayland=false")
    os.environ.setdefault(
        "QTWEBENGINE_CHROMIUM_FLAGS",
        "--disable-gpu --disable-software-rasterizer --no-sandbox")

    # 2) Регистрируем метатип QTextCursor
    try:
        qRegisterMetaType("QTextCursor")
    except Exception:
        pass

    # 3) Общий контекст OpenGL
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("City3D PointCloud Pipeline v7.3.1")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
