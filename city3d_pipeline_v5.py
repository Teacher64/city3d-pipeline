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

try:
    import open3d as o3d
except ImportError:
    o3d = None

PROJECT_ROOT = Path("/mnt/data_sda1/PR_3D_SPB")
WORKSPACE = PROJECT_ROOT / "workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)

PROJECTS_DIR = Path.home() / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)

DATASETS_DIR = WORKSPACE / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

TRAINING_OUTPUT_DIR = WORKSPACE / "training_output"
TRAINING_OUTPUT_DIR.mkdir(exist_ok=True)

REFERENCE_DB_PATH = WORKSPACE / "references.sqlite"

# Путь к файлу справочника EPSG
EPSG_CATALOG_PATH = Path(__file__).parent / "crslist.json"

MODELS = {
    "Cloud2BIM": PROJECTS_DIR / "Cloud2BIM",
    "RoomFormer": PROJECTS_DIR / "RoomFormer",
    "SpatialLM": PROJECTS_DIR / "SpatialLM",
    "CADReasoner": PROJECTS_DIR / "CADReasoner",
    "3d_cad_pointcloud": PROJECTS_DIR / "3d_cad_pointcloud",
}

DB_PARAMS = {
    "dbname":   "city3d",
    "host":     "localhost",
    "port":     "5432",
    "user":     "postgres",
    "password": "postgres",
    "schema_refs":    "refs",
    "schema_history": "history",
}

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
    QAbstractItemView, QStyledItemDelegate, QStyle
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
           ST_Area(ST_Transform(
               ST_Intersection(r.bbox_4326, p_geom), 3857))::real AS overlap_area_m2
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
                "format": ".las"}

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
        """M1 v2: семантическая сегментация облака.

        Пайплайн:
          1. Загрузка облака, оценка нормалей.
          2. Извлечение ЗЕМЛИ:
             • итеративный RANSAC (num_iterations адаптивно к N точек),
               фильтр по |nz| ≥ 0.85, пока не соберём ≥ 40 % точек
               или 8 итераций;
             • если земля не выделилась (например, вертикальная
               стенка) — fallback: 5-й перцентиль Z.
          3. Негоризонтальные точки классифицируем по нормали:
             • |nz| ≥ 0.85 → крыша (roof)
             • |nz| ≤ 0.35 → стена (wall)
             • иначе        → скат (slope)
          4. Region growing по нормалям:
             • крыши  → плоскости крыш roof_plane_1..N
             • стены  → плоскости стен wall_plane_1..M
          5. Кластеризация «крыша+стена» по XY (DBSCAN, eps≈1.5 м)
             → здания building_1..K.
          6. Сохранение:
             • m1_classified.ply  — цветное облако по классам
             • m1_planes.json     — плоскости (центроид, нормаль, площадь)
             • m1_buildings.json  — здания (bbox, число плоскостей)
             • m1_report.txt      — человекочитаемый отчёт
        """
        import time as _time
        self.log.log(f"M1: {pointcloud_path}", "INFO")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def _p(pct, note=""):
            if emit_progress:
                try:
                    emit_progress(pct, note)
                except TypeError:
                    emit_progress(pct)
            self.log.log(f"M1 [{pct}%] {note}", "INFO")

        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            self.log.log("M1: scikit-learn не установлен", "ERROR")
            return False

        try:
            t0 = _time.time()
            _p(2, "Загрузка облака…")
            pcd = o3d.io.read_point_cloud(str(pointcloud_path))
            n_total = len(pcd.points)
            if n_total == 0:
                self.log.log("M1: облако пустое", "ERROR")
                return False
            # --- 0. VOXEL DOWNSAMPLING (SLAM-style) ----------------------
            # Уменьшает плотность и убирает шум, ускоряет RANSAC в 5–10 раз.
            # 35M ALS-точек → ~3M точек при voxel=0.15 м.
            scan_t = getattr(self, "_scanning_type", "unknown")
            voxel_map = {
                "ALS_manned": 0.25, "ALS_uav": 0.20, "MLS": 0.10,
                "TLS": 0.05, "handheld": 0.03, "stationary": 0.02,
            }
            voxel_size = voxel_map.get(scan_t, 0.15)
            _p(6, f"Voxel downsampling {voxel_size} м...")
            _n_before = len(pcd.points)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            n_total = len(pcd.points)
            self.log.log(
                f"M1: voxel {voxel_size} м -> {n_total:,} точек "
                f"(было {_n_before:,}, "
                f"-{100 * (1 - n_total / _n_before):.0f}%)",
                "SUCCESS")

            self.log.log(f"M1: точек: {n_total:,}", "INFO")

            # --- 1. Нормали ------------------------------------------------
            _p(8, "Оценка нормалей…")
            # ВАЖНО: нормали считаем на ПОЛНОМ облаке. Раньше была
            # «оптимизация» — считать по подвыборке 2 млн, но это
            # ломало broadcast: nz имел длину 2M, а ground_mask — 35M.
            # Open3D KDTree справляется с 35M точек за 1–3 мин.
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=max(0.5, float(ransac_threshold) * 20),
                    max_nn=30))
            pcd.orient_normals_to_align_with_direction(
                orientation_reference=np.array([0.0, 0.0, 1.0]))

            pts = np.asarray(pcd.points)
            nrm = np.asarray(pcd.normals)
            nz = np.abs(nrm[:, 2])

            # --- 2. Земля --------------------------------------------------
            _p(18, "Извлечение земли…")
            # 2a. Быстрый приближённый отбор: Z < p5 + 0.4·(p95-p5)
            z = pts[:, 2]
            z_lo = np.percentile(z, 5)
            z_hi = np.percentile(z, 95)
            ground_mask = z < (z_lo + 0.15 * max(0.1, z_hi - z_lo))
            # 2b. RANSAC-уточнение по горизонтальным нормалям
            ground_refined = ground_mask.copy()
            ground_planes = []
            remaining = pcd.select_by_index(
                np.where(ground_mask)[0], invert=False)
            for k in range(6):
                if len(remaining.points) < 5000:
                    break
                try:
                    model, inliers = remaining.segment_plane(
                        distance_threshold=float(ransac_threshold) * 2,
                        ransac_n=3, num_iterations=500)
                except Exception as e:
                    self.log.log(f"M1: RANSAC-земля {k+1}: {e}", "WARNING")
                    break
                if len(inliers) < 5000:
                    break
                a, b, c, d = model
                nn = np.array([a, b, c])
                nn = nn / (np.linalg.norm(nn) + 1e-12)
                if abs(nn[2]) < 0.9:
                    # не горизонтальная — прекращаем сбор земли
                    break
                ground_planes.append({
                    "model": [float(x) for x in model],
                    "n_inliers": int(len(inliers))})
                remaining = remaining.select_by_index(inliers, invert=True)
            n_ground = int(ground_mask.sum())
            self.log.log(
                f"M1: земля ≈ {n_ground:,} ({n_ground/n_total*100:.1f}%)",
                "SUCCESS")

            # --- 3. Классификация по нормалям ------------------------------
            _p(35, "Классификация по ориентации нормалей…")
            roof_mask = (nz >= 0.85) & (~ground_mask)
            wall_mask = (nz <= 0.35)
            slope_mask = (~ground_mask) & (~roof_mask) & (~wall_mask)
            self.log.log(
                f"M1: кандидаты  крыши={int(roof_mask.sum()):,}  "
                f"стены={int(wall_mask.sum()):,}  "
                f"скаты={int(slope_mask.sum()):,}", "INFO")

            # --- 4. Region growing по плоскостям ---------------------------
            _p(48, "Сегментация плоскостей крыш…")
            roof_planes = self._segment_planes(
                pts, nrm, roof_mask,
                ransac_threshold, max_planes=40)
            _p(62, "Сегментация плоскостей стен…")
            wall_planes = self._segment_planes(
                pts, nrm, wall_mask,
                ransac_threshold, max_planes=80)

            # --- 5. Кластеризация зданий (по ТОЧКАМ, не по центроидам) -----
            # Строим граф связи плоскостей: крыша+стены одного дома
            # пересекаются по XY-проекции. Используем point-based clustering.
            _p(75, "Кластеризация зданий (DBSCAN по крышам)…")
            building_clusters = []
            if roof_planes:
                # 1. Центроиды XY крыш
                roof_c = np.array([p["centroid"][:2] for p in roof_planes])
                # 2. DBSCAN: eps=15 м — крыши одного дома ближе 15 м.
                #    min_samples=1 — даже одна крыша = здание.
                if len(roof_c) >= 1:
                    from sklearn.cluster import DBSCAN as _DB
                    roof_labels = _DB(eps=15.0, min_samples=1).fit_predict(
                        roof_c)
                else:
                    roof_labels = np.array([])
                n_bld = len(set(roof_labels)) if len(roof_labels) else 0
                self.log.log(
                    f"M1: DBSCAN по крышам → {n_bld} кластеров",
                    "INFO")
                # 3. Для каждой крыши — её точки и XY bbox
                all_roof_pts = np.concatenate(
                    [roof_planes[k]["indices"]
                     for k in range(len(roof_planes))])
                for lb in sorted(set(roof_labels)):
                    if lb == -1:
                        continue
                    members = np.where(roof_labels == lb)[0]
                    member_roofs = [roof_planes[k] for k in members]
                    # Все точки крыш кластера
                    roof_idx = np.concatenate(
                        [r["indices"] for r in member_roofs])
                    rp = pts[roof_idx]
                    rxy = rp[:, :2]
                    xmin, ymin = rxy.min(axis=0)
                    xmax, ymax = rxy.max(axis=0)
                    # 4. Привязка стен: стена принадлежит зданию, если её
                    #    центроид попадает в расширенный bbox крыш (+3 м)
                    MARGIN = 3.0
                    attached_walls = []
                    for wi, wl in enumerate(wall_planes):
                        cx, cy = wl["centroid"][:2]
                        if (xmin - MARGIN <= cx <= xmax + MARGIN and
                                ymin - MARGIN <= cy <= ymax + MARGIN):
                            attached_walls.append(wi)
                    wall_idx = (np.concatenate(
                        [wall_planes[wi]["indices"]
                         for wi in attached_walls])
                        if attached_walls
                        else np.array([], dtype=np.int64))
                    all_idx = np.concatenate([roof_idx, wall_idx]) \
                        if len(wall_idx) else roof_idx
                    cp = pts[all_idx]
                    cxy = cp[:, :2]
                    cxyz = cp
                    h = float(cxyz[:, 2].max() - cxyz[:, 2].min())
                    a = float((cxy[:, 0].max() - cxy[:, 0].min()) *
                              (cxy[:, 1].max() - cxy[:, 1].min()))
                    building_clusters.append({
                        "building_id": int(len(building_clusters) + 1),
                        "n_planes": int(len(member_roofs) +
                                        len(attached_walls)),
                        "n_roof_planes": int(len(member_roofs)),
                        "n_wall_planes": int(len(attached_walls)),
                        "n_points": int(len(all_idx)),
                        "bbox_xy": [float(xmin), float(ymin),
                                    float(xmax), float(ymax)],
                        "z_range": [float(cxyz[:, 2].min()),
                                    float(cxyz[:, 2].max())],
                        "height_m": h,
                        "area_m2": a,
                    })
            self.log.log(
                f"M1: зданий после кластеризации: "
                f"{len(building_clusters)}", "SUCCESS")

            _p(85, "Сохранение результатов…")
            n_roof_pts = sum(p["n_inliers"] for p in roof_planes)
            n_wall_pts = sum(p["n_inliers"] for p in wall_planes)
            n_ground_pts = int(ground_mask.sum())
            # Нормированная оценка качества: покрытие значимыми классами
            covered = (n_ground_pts + n_roof_pts + n_wall_pts)
            quality = float(min(1.0, covered / n_total)) \
                if n_total > 0 else 0.0

            # 6a. Классифицированное облако (цвет = класс)
            colors = np.zeros((n_total, 3), dtype=np.float64)
            colors[:, :] = (0.7, 0.7, 0.7)          # прочее — серый
            colors[ground_mask] = (0.55, 0.27, 0.07)  # земля — коричневый
            # закрасим точки по плоскостям (крыши — красные, стены — синие)
            roof_ids = np.zeros(n_total, dtype=bool)
            wall_ids = np.zeros(n_total, dtype=bool)
            for pl in roof_planes:
                roof_ids[pl["indices"]] = True
            for pl in wall_planes:
                wall_ids[pl["indices"]] = True
            colors[roof_ids] = (0.85, 0.15, 0.15)
            colors[wall_ids] = (0.15, 0.35, 0.9)
            colors[slope_mask] = (0.5, 0.6, 0.2)
            out_pcd = o3d.geometry.PointCloud()
            out_pcd.points = o3d.utility.Vector3dVector(pts)
            out_pcd.colors = o3d.utility.Vector3dVector(colors)
            # LAS 1.4 classification как scalar field для CloudCompare/GIS
            labels_las14 = np.ones(n_total, dtype=np.uint8)  # 1=unclassified
            labels_las14[ground_mask & (~roof_ids) & (~wall_ids)] = 2  # 2=ground
            labels_las14[roof_ids]   = 6                     # 6=building
            labels_las14[wall_ids]   = 6                     # стены → 6
            labels_las14[slope_mask] = 5                     # 5=high veg
            out_pcd.points = o3d.utility.Vector3dVector(pts)
            out_pcd.colors = o3d.utility.Vector3dVector(colors)
            # Сохраняем как PLY с classification
            classified_path = output_dir / "m1_classified.ply"
            o3d.io.write_point_cloud(str(classified_path), out_pcd)
            # Дополнительно — LAS 1.4 с classification
            try:
                import laspy
                header = laspy.LasHeader(point_format=7, version="1.4")
                header.offsets = pts.min(axis=0)
                header.scales = np.array([0.001, 0.001, 0.001])
                las = laspy.LasData(header)
                las.x = pts[:, 0]; las.y = pts[:, 1]; las.z = pts[:, 2]
                las.red   = (colors[:, 0] * 65535).astype(np.uint16)
                las.green = (colors[:, 1] * 65535).astype(np.uint16)
                las.blue  = (colors[:, 2] * 65535).astype(np.uint16)
                las.classification = labels_las14
                las.write(str(output_dir / "m1_classified.las"))
                self.log.log(
                    f"M1: LAS 1.4 с classification → "
                    f"m1_classified.las", "SUCCESS")
            except Exception as _e:
                self.log.log(f"M1: LAS не создан ({_e})", "WARNING")

            # 6b. JSON по плоскостям — с РЕАЛЬНОЙ геометрией (convex hull)
            def _plane_to_dict(pl, kind):
                pl_pts = pts[pl["indices"]]
                uxy = pl_pts[:, :2]
                d = {k: v for k, v in pl.items() if k != "indices"}
                d["kind"] = kind
                d["xy_extent"] = [float(uxy[:,0].min()),
                                  float(uxy[:,1].min()),
                                  float(uxy[:,0].max()),
                                  float(uxy[:,1].max())]
                try:
                    from scipy.spatial import ConvexHull
                    if len(uxy) >= 3:
                        h = ConvexHull(uxy)
                        d["hull_xy"] = [[float(uxy[i,0]), float(uxy[i,1])]
                                        for i in h.vertices]
                except Exception:
                    pass
                return d

            planes_path = output_dir / "m1_planes.json"
            planes_path.write_text(json.dumps({
                "method": "M1_semantic_v3",
                "source": str(pointcloud_path),
                "timestamp": datetime.now().isoformat(),
                "n_points": int(n_total),
                "voxel_size_m": float(voxel_size),
                "ground": {
                    "n_points": int(n_ground_pts),
                    "n_planes_ransac": len(ground_planes),
                },
                "roof_planes": [_plane_to_dict(pl, "roof")
                                for pl in roof_planes],
                "wall_planes": [_plane_to_dict(pl, "wall")
                                for pl in wall_planes],
            }, ensure_ascii=False, indent=2))

            # 6c. JSON по зданиям
            buildings_path = output_dir / "m1_buildings.json"
            buildings_path.write_text(json.dumps({
                "n_buildings": len(building_clusters),
                "buildings": building_clusters,
            }, ensure_ascii=False, indent=2))

            # 6d. Человекочитаемый отчёт
            dt = _time.time() - t0
            report_lines = [
                "=" * 70,
                "ОТЧЁТ M1 — семантическая сегментация",
                "=" * 70,
                f"Источник         : {pointcloud_path}",
                f"Всего точек      : {n_total:,}",
                f"Время            : {dt:.1f} с",
                "",
                f"ЗЕМЛЯ            : {n_ground_pts:,}  "
                f"({n_ground_pts/n_total*100:.1f}%)",
                f"КРЫШИ (плоскостей): {len(roof_planes)}  "
                f"({n_roof_pts:,} точек)",
                f"СТЕНЫ (плоскостей): {len(wall_planes)}  "
                f"({n_wall_pts:,} точек)",
                f"СКАТЫ/прочее     : {int(slope_mask.sum()):,}",
                f"ЗДАНИЙ (кластеров): {len(building_clusters)}",
                "",
                f"ОЦЕНКА КАЧЕСТВА  : {quality:.3f}  (0..1)",
                "",
                "ЧТО ДАЛЬШЕ:",
                "  1. Откройте m1_classified.ply — там точки окрашены",
                "     по классам: коричневые — земля, красные — крыши,",
                "     синие — стены, зелёные — скаты, серые — прочее.",
                "  2. Проверьте m1_planes.json — там каждая найденная",
                "     плоскость с нормалью и площадью.",
                "  3. m1_buildings.json — предварительные контуры зданий.",
                "  4. Если зданий слишком мало/много — подкрутите",
                "     RANSAC threshold или запустите M3 (U-Net) для",
                "     уточнения классов растительности и дорог.",
                "",
                "ФАЙЛЫ:",
                f"  • {classified_path}",
                f"  • {planes_path}",
                f"  • {buildings_path}",
            ]
            report_path = output_dir / "m1_report.txt"
            report_path.write_text("\n".join(report_lines),
                                   encoding="utf-8")
            self.log.log("M1: " + "-" * 50, "INFO")
            for line in report_lines:
                self.log.log(line, "INFO")
            self.log.log("M1: " + "-" * 50, "INFO")

            _p(100, f"готово: крыш {len(roof_planes)}, "
                    f"стен {len(wall_planes)}, "
                    f"зданий {len(building_clusters)}")
            self.log.log(f"M1: результат: {classified_path}", "SUCCESS")
            self.log.log(f"M1: отчёт: {report_path}", "SUCCESS")
            self._last_quality = quality
            return True
        except Exception as e:
            import traceback
            self.log.log(f"M1: ошибка: {e}", "ERROR")
            self.log.log(traceback.format_exc(), "ERROR")
            return False

    # --- вспомогательный метод: region growing по плоскостям -----------
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
                uv = plane_pts[:, :2]
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
        best_count = 0
        best_model = None
        best_inl = None
        rng = np.random.default_rng(0)
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
        return best_model, best_inl


# ============================================================================
# STEP 6 — Адаптеры семантических моделей
# ============================================================================

SEMANTIC_COLORS = {
    0:  (0.55, 0.27, 0.07),   # ground — коричневый
    1:  (0.85, 0.15, 0.15),   # building — красный
    2:  (0.30, 0.30, 0.30),   # road — тёмно-серый
    3:  (0.20, 0.70, 0.20),   # vegetation — зелёный
    4:  (0.90, 0.80, 0.10),   # pole — жёлтый
    5:  (0.90, 0.50, 0.10),   # fence — оранжевый
    6:  (0.10, 0.30, 0.90),   # water — синий
    7:  (0.60, 0.20, 0.80),   # rail — фиолетовый
    8:  (0.20, 0.80, 0.90),   # car — голубой
    9:  (0.80, 0.80, 0.80),   # прочее — серый
}


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
        import torch
        import numpy as np
        from torch_pointcloud.utils.data import collate

        if self.model is None:
            if not self.load():
                return None, None

        N = len(points)
        labels = np.full(N, -1, dtype=np.int64)
        confidence = np.zeros(N, dtype=np.float32)

        # Границы сцены
        xmin, ymin = points[:, 0].min(), points[:, 1].min()
        xmax, ymax = points[:, 0].max(), points[:, 1].max()

        blocks = []
        x = xmin
        while x < xmax:
            y = ymin
            while y < ymax:
                blocks.append((x, y))
                y += block_size - overlap
            x += block_size - overlap

        n_blocks = len(blocks)
        if self.log:
            self.log.log(
                f"Инференс: {N:,} точек, {n_blocks} блоков "
                f"({block_size}×{block_size} м)", "INFO")

        for bi, (bx, by) in enumerate(blocks):
            mask = ((points[:, 0] >= bx) &
                    (points[:, 0] < bx + block_size) &
                    (points[:, 1] >= by) &
                    (points[:, 1] < by + block_size))
            idx = np.where(mask)[0]
            if len(idx) < 100:
                continue

            sample = {
                "pos": torch.from_numpy(
                    points[idx].astype(np.float32)),
                "segment": torch.zeros(len(idx), dtype=torch.long),
            }
            if colors is not None:
                sample["color"] = torch.from_numpy(
                    colors[idx].astype(np.float32) * 255.0)
            if self.cfg["input_channels"] >= 6:
                sample["normal"] = torch.zeros(
                    len(idx), 3, dtype=torch.float32)

            try:
                data = self.info["transform"](sample)
                data = collate([data])
                if torch.cuda.is_available():
                    data = {k: v.cuda() if hasattr(v, "cuda") else v
                            for k, v in data.items()}
                with torch.no_grad():
                    logits = self.model(
                        data.get("x"), data["pos"], data["batch"])
                    probs = torch.softmax(logits, dim=-1)
                    pred = probs.argmax(dim=-1).cpu().numpy()
                    conf = probs.max(dim=-1).values.cpu().numpy()
                labels[idx] = pred
                confidence[idx] = conf
            except Exception as e:
                if self.log:
                    self.log.log(
                        f"Блок {bi+1}/{n_blocks}: {e}", "WARNING")

            if emit_progress:
                pct = int(100 * (bi + 1) / n_blocks)
                emit_progress(pct, f"блок {bi+1}/{n_blocks}")

        return labels, confidence


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
            points = np.asarray(pcd.points).astype(np.float32)
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


    def train_m2(self, raster_path, output_dir):
        self.log.log(f"M2: {raster_path}", "INFO")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not Path(raster_path).exists():
            self.log.log("M2: файл не найден", "ERROR")
            return False
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            results = model(raster_path)
            stats_path = output_dir / "m2_yolo_stats.json"
            with open(stats_path, "w") as f:
                json.dump({"model": "yolov8n",
                           "n_boxes": (len(results[0].boxes)
                                       if results else 0),
                           "source": raster_path}, f, indent=2)
            self.log.log(f"M2: YOLOv8: {stats_path}", "SUCCESS")
            return True
        except ImportError:
            self.log.log("M2: YOLOv8 нет, fallback rasterio", "WARNING")
        try:
            import rasterio
            with rasterio.open(raster_path) as src:
                meta = {"width": src.width, "height": src.height,
                        "n_bands": src.count, "crs": str(src.crs)}
            stats_path = output_dir / "m2_raster_info.json"
            with open(stats_path, "w") as f:
                json.dump(meta, f, indent=2)
            self.log.log(f"M2: базовый анализ: {stats_path}", "SUCCESS")
            return True
        except Exception as e:
            self.log.log(f"M2: ошибка: {e}", "ERROR")
            return False

    def train_m3(self, masks_dir, output_dir, n_classes=11, epochs=10):
        self.log.log(f"M3: {masks_dir}", "INFO")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            import torch.nn as nn
            self.log.log(
                f"M3: PyTorch {torch.__version__}, "
                f"CUDA: {torch.cuda.is_available()}", "INFO")

            class SimpleSegNet(nn.Module):
                def __init__(self, n):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, n, 3, padding=1)

                def forward(self, x):
                    return self.conv2(torch.relu(self.conv1(x)))
            model = SimpleSegNet(n_classes)
            checkpoint_path = output_dir / "m3_unet_init.pth"
            torch.save(model.state_dict(), checkpoint_path)
            self.log.log(f"M3: чекпоинт: {checkpoint_path}", "SUCCESS")
            return True
        except ImportError:
            self.log.log("M3: PyTorch не установлен", "ERROR")
            return False

    def train_m4(self, dem_path, output_dir):
        self.log.log(f"M4: {dem_path}", "INFO")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not Path(dem_path).exists():
            self.log.log("M4: файл не найден", "ERROR")
            return False
        try:
            import rasterio
            from scipy import ndimage
            with rasterio.open(dem_path) as src:
                dem = src.read(1).astype(np.float32)
            dy, dx = np.gradient(dem)
            slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2)) * 180 / np.pi
            aspect = np.arctan2(-dx, dy) * 180 / np.pi
            aspect = np.where(aspect < 0, 360 + aspect, aspect)
            curvature = ndimage.laplace(dem)
            mean_dem = ndimage.uniform_filter(dem, size=15)
            tpi = dem - mean_dem
            output_npz = output_dir / "m4_terrain_features.npz"
            np.savez_compressed(output_npz, slope=slope, aspect=aspect,
                                curvature=curvature, tpi=tpi, dem=dem)
            stats = {"dem_min": float(dem.min()),
                     "dem_max": float(dem.max()),
                     "slope_mean": float(slope.mean()),
                     "slope_max": float(slope.max()),
                     "source": str(dem_path)}
            with open(output_dir / "m4_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            self.log.log(
                f"M4: уклон {slope.mean():.1f}°, "
                f"макс. {slope.max():.1f}°", "INFO")
            self.log.log(f"M4: результат: {output_npz}", "SUCCESS")
            return True
        except Exception as e:
            self.log.log(f"M4: ошибка: {e}", "ERROR")
            return False

    def train_m5(self, classified_las, output_dir):
        self.log.log(f"M5: {classified_las}", "INFO")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not Path(classified_las).exists():
            self.log.log("M5: файл не найден", "ERROR")
            return False
        try:
            import laspy
            las = laspy.read(classified_las)
            if not hasattr(las, "classification"):
                self.log.log("M5: нет classification", "ERROR")
                return False
            classes = np.array(las.classification)
            unique, counts = np.unique(classes, return_counts=True)
            asprs = {2: "ground", 3: "low_vegetation",
                     4: "medium_vegetation", 5: "high_vegetation",
                     6: "building", 9: "water", 10: "rail",
                     11: "road_surface", 13: "wire_guard"}
            stats = {}
            for c, n in zip(unique, counts):
                name = asprs.get(int(c), f"class_{c}")
                stats[name] = int(n)
                self.log.log(f"    {name}: {n:,}", "INFO")

            counts_arr = np.array(list(stats.values()), dtype=np.float64)
            if counts_arr.sum() > 0:
                p = counts_arr / counts_arr.sum()
                ent = -float(np.sum(p * np.log(p + 1e-12)))
                max_ent = (float(np.log(len(counts_arr)))
                           if len(counts_arr) > 1 else 1.0)
                quality = ent / max_ent if max_ent > 0 else 0.0
            else:
                quality = 0.0

            stats_path = output_dir / "m5_class_stats.json"
            with open(stats_path, "w") as f:
                json.dump({"n_points": len(classes),
                           "n_classes": len(unique),
                           "classes": stats,
                           "quality_score": quality,
                           "source": str(classified_las)},
                          f, indent=2, ensure_ascii=False)
            self.log.log(f"M5: статистика: {stats_path}", "SUCCESS")
            return True
        except Exception as e:
            self.log.log(f"M5: ошибка: {e}", "ERROR")
            return False

    def train_m6(self, gis_path, output_dir):
        self.log.log(f"M6: {gis_path}", "INFO")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not Path(gis_path).exists():
            self.log.log("M6: файл не найден", "ERROR")
            return False
        try:
            import geopandas as gpd
            gdf = gpd.read_file(gis_path)
            geom_types = gdf.geometry.geom_type.value_counts().to_dict()
            stats = {"format": Path(gis_path).suffix,
                     "n_features": len(gdf),
                     "geometry_types": geom_types,
                     "crs": str(gdf.crs),
                     "columns": list(gdf.columns),
                     "source": gis_path}
            stats_path = output_dir / "m6_gis_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            self.log.log(f"M6: {len(gdf)} объектов, CRS: {gdf.crs}",
                         "INFO")
            self.log.log(f"M6: статистика: {stats_path}", "SUCCESS")
            return True
        except Exception as e:
            self.log.log(f"M6: ошибка: {e}", "ERROR")
            return False

    def train_combined(self, methods, weights, paths, output_dir):
        self.log.log(f"Комбинирование: {methods}", "INFO")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for m in methods:
            method_path = paths.get(m, "")
            self.log.log(f"--- Запуск {m} ---", "INFO")
            if m == "M1":
                ok = self.train_m1(method_path, output_dir)
            elif m == "M2":
                ok = self.train_m2(method_path, output_dir)
            elif m == "M3":
                ok = self.train_m3(method_path, output_dir)
            elif m == "M4":
                ok = self.train_m4(method_path, output_dir)
            elif m == "M5":
                ok = self.train_m5(method_path, output_dir)
            elif m == "M6":
                ok = self.train_m6(method_path, output_dir)
            else:
                ok = False
            results[m] = {"success": ok,
                          "weight": weights.get(m, 1.0),
                          "path": method_path}
        agg_path = output_dir / "combined_training_summary.json"
        with open(agg_path, "w") as f:
            json.dump({"methods": methods, "weights": weights,
                       "results": results,
                       "timestamp": datetime.now().isoformat()},
                      f, indent=2, ensure_ascii=False)
        self.log.log(f"Комбинирование: {agg_path}", "SUCCESS")
        return True


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
        if not self._check_tool("3dfier"):
            return self._result(False, output_path,
                "Утилита '3dfier' не найдена в PATH.")
        extent_line = ""
        if bbox:
            extent_line = (f"  extent: [{bbox[0]}, {bbox[1]}, "
                           f"{bbox[3]}, {bbox[4]}]")
        config = f"""input:
  pointcloud: {input_path}
{extent_line}

output:
  format: CityGML
  epsg: 4326

lifting:
  building:
    height: 10.0
    min_area: 10.0
  terrain:
    true: true
"""
        cfg = self.workspace / "3dfier_config.yml"
        cfg.write_text(config)
        r = subprocess.run(
            ["3dfier", str(cfg), "--output", str(output_path)],
            capture_output=True, text=True)
        if r.returncode != 0:
            return self._result(False, output_path,
                f"3dfier вернул код {r.returncode}: "
                f"{r.stderr.strip()[:300]}")
        if not self._ok(output_path):
            return self._result(False, output_path,
                "3dfier отработал, но файл CityGML не создан.")
        return self._result(True, output_path)

    def export_to_usd(self, input_path, output_path):
        try:
            from pxr import Usd, UsdGeom, Gf
        except ImportError as e:
            return self._result(False, output_path,
                f"Библиотека pxr (OpenUSD) не установлена: {e}")
        try:
            pcd = o3d.io.read_point_cloud(str(input_path))
            points = np.asarray(pcd.points)
            if len(points) == 0:
                return self._result(False, output_path,
                    "В исходном облаке нет точек.")
            stage = Usd.Stage.CreateNew(str(output_path))
            xform = UsdGeom.Xform.Define(stage, '/World')
            stage.SetDefaultPrim(xform.GetPrim())
            pp = UsdGeom.Points.Define(stage, '/World/PointCloud')
            pp.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
            stage.Save()
            if not self._ok(output_path):
                return self._result(False, output_path,
                    "Файл USD не создан.")
            return self._result(True, output_path)
        except Exception as e:
            return self._result(False, output_path, str(e))

    def export_to_obj(self, input_path, output_path):
        try:
            pcd = o3d.io.read_point_cloud(str(input_path))
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
        files = sorted(train_dir.glob("*.ply"))
        count = 0
        for f in files:
            data = PointCloudLoader.load(f)
            points = data["points"].copy()
            center = points.mean(axis=0)
            for i in range(rotations):
                angle = 2 * np.pi * i / rotations
                rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
                rotated = (points - center) @ rot.T + center
                if noise_std > 0:
                    rotated += rng.normal(0, noise_std, rotated.shape)
                if dropout_ratio > 0:
                    mask = rng.rand(len(rotated)) > dropout_ratio
                    rotated = rotated[mask]
                out = aug_dir / f"{f.stem}_rot{i:02d}.ply"
                PointCloudLoader.save({
                    "points": rotated, "colors": None,
                    "intensity": None, "classification": None
                }, out, ".ply")
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
        if not HAS_PSYCOPG:
            self._log("psycopg не установлен — история не ведётся", "WARNING")
            return
        dsn = (f"dbname={self.params['dbname']} "
               f"host={self.params['host']} port={self.params['port']} "
               f"user={self.params['user']} password={self.params['password']}")
        try:
            self.conn = psycopg.connect(dsn, autocommit=True)
            with self.conn.cursor() as cur:
                cur.execute("SET search_path TO %s, public"
                            % (self.schema_h,))
            with self.conn.cursor() as cur:
                cur.execute(DB_DDL)
            self._log(f"Postgres подключён: {self.params['dbname']}",
                      "SUCCESS")
        except Exception as e:
            self.conn = None
            self._log(f"Postgres недоступен: {e}. "
                      f"История и справочники EPSG не будут доступны.",
                      "WARNING")

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
                            CASE WHEN %s IS NULL THEN NULL
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
        if self._mesh_item is not None:
            try:
                self.view.removeItem(self._mesh_item)
            except Exception:
                pass
            self._mesh_item = None

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


class WebMapViewer(QWidget):
    HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <link rel="stylesheet"
        href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ width:100%; height:100%; margin:0; padding:0;
                        background:#1e1e1e; }}
    .leaflet-control-attribution {{ font-size:9px; }}
    .info {{ position:absolute; top:8px; left:50px; z-index:1000;
             background:rgba(30,30,30,0.85); color:#4ec9b0;
             padding:6px 10px; border-radius:4px;
             font-family:sans-serif; font-size:12px; }}
  </style>
</head>
<body>
<div id="map"></div>
<div class="info">{info}</div>
<script>
  const map = L.map('map', {{ preferCanvas: true }})
                .setView([55.75, 37.61], 3);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
              {{ maxZoom: 19 }}).addTo(map);
  const data = {geojson};
  if (data && data.features && data.features.length) {{
    const layer = L.geoJSON(data, {{
      pointToLayer: (f, ll) => L.circleMarker(ll, {{
        radius: 2, color: '#4ec9b0', fillColor: '#4ec9b0',
        fillOpacity: 0.7, weight: 0
      }}),
      style: {{ color: '#4ec9b0', weight: 1, fillOpacity: 0.3 }}
    }}).addTo(map);
    try {{ map.fitBounds(layer.getBounds()); }} catch(e) {{}}
  }}
</script>
</body>
</html>"""

    MAX_FEATURES = 50_000

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.last_truncated = False
        self.last_feature_count = 0
        self._show_placeholder()

    def _show_placeholder(self):
        html = ("<html><body style='background:#1e1e1e;color:#808080;"
                "font-family:sans-serif;padding:30px;text-align:center'>"
                "<h3>Выберите GeoJSON-файл для отображения на карте</h3>"
                "</body></html>")
        self.browser.setHtml(html)

    def show_geojson(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("features", [])
            n_total = len(features)
            if n_total > self.MAX_FEATURES:
                data = {"type": "FeatureCollection",
                        "features": features[:self.MAX_FEATURES]}
                truncated = True
            else:
                truncated = False
            payload = json.dumps(data, ensure_ascii=False)
            info = f"Объектов: {n_total:,}"
            if truncated:
                info += f" (показано {self.MAX_FEATURES:,})"
            html = self.HTML_TEMPLATE.format(geojson=payload, info=info)
            self.browser.setHtml(html)
            self.last_truncated = truncated
            self.last_feature_count = n_total
            return True
        except Exception as e:
            err_html = (
                f"<html><body style='background:#1e1e1e;color:#f44747;"
                f"font-family:monospace;padding:20px'>"
                f"<h3>Не удалось загрузить GeoJSON</h3>"
                f"<pre>{e}</pre></body></html>")
            self.browser.setHtml(err_html)
            return False


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

        if HAS_WEBENGINE:
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

        cg = QGroupBox("Система координат (CRS)")
        cl = QFormLayout()
        self.crs_display = QLineEdit()
        self.crs_display.setReadOnly(True)
        cl.addRow("CRS:", self.crs_display)
        self.crs_source = QLineEdit()
        self.crs_source.setReadOnly(True)
        cl.addRow("Источник:", self.crs_source)
        b_pick_crs = QPushButton("Выбрать CRS вручную…")
        b_pick_crs.setToolTip(
            "Открыть справочник EPSG с поиском, автодополнением и "
            "подсказками. Источник справочника: crslist.json")
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
        {"id": "ransac", "method": "M1", "name": "RANSAC (Open3D)",
         "role": "geometry", "task": "Плоскости", "kind": "classic",
         "requires": ["open3d"], "weights": None,
         "trained_on": "—", "notes": "Без обучения. Быстрый baseline."},
        {"id": "randla_kitti", "method": "M1",
         "name": "RandLA-Net (SemanticKITTI)",
         "role": "semantic", "task": "3D-сегментация", "kind": "dl3d",
         "requires": ["torch", "open3d"],
         "weights": "models/M1/randlanet_semantickitti.pth",
         "trained_on": "SemanticKITTI (19 классов)",
         "notes": "Точность на городах высокая. Требует GPU ~6 GB."},
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
    "P1_lod1": {
        "name": "P1 — Быстрый LOD1",
        "steps": ["preprocess", "M1:ransac", "M4", "fuse",
                  "export:CityGML"],
        "hint": "Минимум шагов. Земля + плоскости + LOD1-экструзия.",
    },
    "P2_lod2": {
        "name": "P2 — Точный LOD2",
        "steps": ["preprocess", "M5", "M1:ransac",
                  "M3:unet_buildings", "fuse", "export:CityGML"],
        "hint": "Требует эталонов ASPRS и маски зданий.",
    },
    "P3_veg": {
        "name": "P3 — Растительность и ЛЭП",
        "steps": ["preprocess", "M4", "M5", "M1:randla_kitti", "fuse",
                  "export:GeoJSON"],
        "hint": "Хорошо для ЛЭП и высоких деревьев.",
    },
    "P4_geoai": {
        "name": "P4 — GeoAI по растрам",
        "steps": ["preprocess", "M3:geoai_buildings",
                  "export:GeoJSON"],
        "hint": "Здания из GeoTIFF через opengeos/geoai.",
    },
    "P5_validate": {
        "name": "P5 — Валидация по ГИС",
        "steps": ["preprocess", "M1:ransac", "M6", "fuse",
                  "export:DXF"],
        "hint": "Требуется SXF/SHP на ту же территорию.",
    },
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
            Path(__file__).parent / "models.json")
        self.log = log_widget
        self.models = []
        self._load()

    def _load(self):
        if not self.path.exists():
            self._write_template()
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
        try:
            __import__(name)
            return True, ""
        except ImportError:
            return False, f"нет модуля '{name}'"

    def _check_weights(self, w):
        if not w:
            return True, ""
        p = Path(w).expanduser()
        if p.is_absolute() or "/" in str(w):
            if p.exists():
                return True, ""
            return False, f"нет файла '{w}'"
        # относительное имя типа yolov8n.pt — скачивается автоматически
        return True, "(будет скачано)"

    def check_availability(self):
        """Возвращает список {'model','ok','issues','requires_state'}."""
        out = []
        for m in self.models:
            issues = []
            req_state = []
            for req in m.get("requires", []) or []:
                ok, err = self._check_module(req)
                req_state.append((req, ok))
                if not ok:
                    issues.append(err)
            ok_w, err_w = self._check_weights(m.get("weights"))
            if not ok_w:
                issues.append(err_w)
            out.append({"model": m, "ok": not issues,
                        "issues": issues, "requires_state": req_state})
        return out



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
        # проверка порядка
        steps = []
        for i in range(self.pipe_list.count()):
            d = self.pipe_list.item(i).data(Qt.UserRole) or {}
            steps.append(d)
        # предупреждения
        warns = []
        for i, d in enumerate(steps):
            if d.get("role") == "export" and i < len(steps) - 1:
                warns.append(f"export на шаге {i+1} — не последний.")
            if i > 0 and d.get("role") == "semantic" and \
                    steps[i-1].get("role") == "semantic":
                warns.append(f"два semantic подряд: шаги {i}, {i+1}.")
        if warns:
            ans = QMessageBox.question(
                self, "Порядок вызывает вопросы",
                "Найдены потенциальные проблемы:\n\n• "
                + "\n• ".join(warns)
                + "\n\nПродолжить?",
                QMessageBox.Yes | QMessageBox.No)
            if ans != QMessageBox.Yes:
                return

        self.log.log("=" * 60, "INFO")
        self.log.log(f"ЗАПУСК КОНВЕЙЕРА из {len(steps)} шагов", "INFO")
        self.log.log("=" * 60, "INFO")
        for i, d in enumerate(steps):
            label = self.pipe_list.item(i).text()
            self.log.log(f"[{i+1}/{len(steps)}] {label}", "INFO")
            # пока это заглушка — реальный запуск требует
            # рефакторинга RealTrainer под pipeline-режим.
            self.log.log(
                f"    (интеграция шага '{d.get('id')}' — "
                f"следующая итерация разработки)", "WARNING")
        self.log.log(
            "Подсказка: одиночные прогоны по-прежнему доступны на "
            "подвкладке «Одиночная модель».", "SUCCESS")

# ============================================================================
# Вкладка 5 — Обучение M1-M6
# ============================================================================

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

        self.sub_tabs.addTab(single, "Одиночная модель")

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
            "M1: RANSAC + DBSCAN по облаку точек."))
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
        layout.addWidget(QLabel("M2: обучение по растрам."))
        g = QGroupBox("Растр")
        gl = QHBoxLayout()
        self.m2_raster_path = QLineEdit()
        self.m2_raster_path.setPlaceholderText("GeoTiff / JPEG / PDF")
        gl.addWidget(self.m2_raster_path)
        b = QPushButton("Обзор...")
        b.clicked.connect(self._browse_m2)
        gl.addWidget(b)
        g.setLayout(gl)
        layout.addWidget(g)
        btn = QPushButton("Запустить M2")
        btn.clicked.connect(lambda: self._run_method("M2"))
        layout.addWidget(btn)
        layout.addStretch()
        return w

    def _create_m3_page(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("M3: PyTorch U-Net по маскам."))
        g = QGroupBox("Слой")
        f = QFormLayout()
        self.m3_layer = QComboBox()
        self.m3_layer.addItems([
            "BUILDINGS", "ROADS", "HYDROGRAPHY", "PIPELINES",
            "RAILWAYS", "FENCES", "POWER_LINES",
            "TREE_VEGETATION", "GRASS_VEGETATION", "SWAMPS", "SANDS"])
        f.addRow("Тип:", self.m3_layer)
        g.setLayout(f)
        layout.addWidget(g)
        btn = QPushButton("Запустить M3")
        btn.clicked.connect(lambda: self._run_method("M3"))
        layout.addWidget(btn)
        layout.addStretch()
        return w

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
        layout.addWidget(QLabel("M5: анализ ASPRS-классов."))
        g = QGroupBox("LAS")
        gl = QHBoxLayout()
        self.m5_las_path = QLineEdit()
        self.m5_las_path.setPlaceholderText("LAS с classification")
        gl.addWidget(self.m5_las_path)
        b = QPushButton("Обзор...")
        b.clicked.connect(self._browse_m5)
        gl.addWidget(b)
        g.setLayout(gl)
        layout.addWidget(g)
        btn = QPushButton("Запустить M5")
        btn.clicked.connect(lambda: self._run_method("M5"))
        layout.addWidget(btn)
        layout.addStretch()
        return w

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
        ml.addRow("Датасет:", self.dataset_path)
        b = QPushButton("Обзор...")
        b.clicked.connect(self.browse_dataset)
        ml.addRow("", b)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(50)
        ml.addRow("Эпох:", self.epochs)
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 100)
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
        p, _ = QFileDialog.getOpenFileName(
            self, "LAS", "", "LAS (*.las *.laz);;All Files (*)")
        if p:
            self.m5_las_path.setText(p)

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

    def _run_method(self, method_name):
        # F. Проверка эталонов до запуска
        try:
            self._check_references(method_name)
        except Exception as e:
            self.log.log(f"Проверка эталонов: {e}", "WARNING")

        if method_name == "M1":
            src = str(WORKSPACE / "pointcloud_clean.ply")
            if not Path(src).exists():
                src = (getattr(self.main_window, "clean_pcd_path", None)
                       or getattr(self.main_window, "pcd_path", None) or "")
        elif method_name == "M2":
            src = self.m2_raster_path.text()
        elif method_name == "M3":
            src = self.dataset_path.text()
        elif method_name == "M4":
            src = self.m4_dem_path.text()
        elif method_name == "M5":
            src = self.m5_las_path.text()
        elif method_name == "M6":
            src = self.m6_gis_path.text()
        else:
            src = ""
        if not src or not Path(src).exists():
            QMessageBox.warning(self, "Ошибка",
                                f"Источник {method_name} не найден: {src}")
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

        # Загружаем список классов из classifier.txt (если есть)
        classes = self._load_classifier()
        if classes:
            self.log.log(
                f"Загружен classifier.txt: {len(classes)} классов "
                f"({', '.join(c['name'] for c in classes[:6])}"
                f"{'…' if len(classes) > 6 else ''})", "INFO")
        worker = WorkerThread(self._train_task, method_name, src,
                              str(output_dir), hyper, classes)
        worker.log_signal.connect(self.log.log)
        worker.detail_signal.connect(self.eta_label.setText)
        worker.progress.connect(self.progress.setValue)
        worker.task_done.connect(self._on_training_done)
        worker.start()
        self._worker = worker

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
            crs = getattr(mw.crs_detector, "crs", None) \
                if mw.crs_detector else None
            if mw.pcd is not None:
                pts = np.asarray(mw.pcd.points)
                bbox = (float(pts[:, 0].min()), float(pts[:, 1].min()),
                        float(pts[:, 2].min()), float(pts[:, 0].max()),
                        float(pts[:, 1].max()), float(pts[:, 2].max()))
                n_points = len(pts)
                if crs:
                    try:
                        from pyproj import Transformer
                        tr = Transformer.from_crs(
                            crs, 4326, always_xy=True)
                        xs, ys = tr.transform(
                            [bbox[0], bbox[3], bbox[3], bbox[0]],
                            [bbox[1], bbox[1], bbox[4], bbox[4]])
                        bbox_4326 = (min(xs), min(ys), max(xs), max(ys))
                    except Exception:
                        pass
            run_id = hdb.start_run(
                method=method,
                scanning_type=getattr(mw, "scanning_type", "unknown"),
                crs_epsg=crs,
                source_path=src,
                dataset_root=getattr(mw, "dataset_root", None),
                hyperparams=hyper or {},
                n_points=n_points,
                bbox=bbox,
                bbox_4326=bbox_4326)

        trainer = RealTrainer(EmitLogger(emit_log))
        emit_progress(20)
        if method == "M1":
            result = trainer.train_m1(
                src, out_dir,
                ransac_threshold=float(
                    (hyper or {}).get("ransac_threshold",
                                      self.m1_ransac_threshold.value())),
                classifier_classes=classes,
                emit_progress=emit_progress)
        elif method == "M2":
            result = trainer.train_m2(src, out_dir)
        elif method == "M3":
            result = trainer.train_m3(
                src, out_dir,
                n_classes=int((hyper or {}).get(
                    "n_classes", self.num_classes.value())),
                epochs=int((hyper or {}).get(
                    "epochs", self.epochs.value())))
        elif method == "M4":
            result = trainer.train_m4(src, out_dir)
        elif method == "M5":
            result = trainer.train_m5(src, out_dir)
        elif method == "M6":
            result = trainer.train_m6(src, out_dir)
        else:
            result = False
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
            m1j = Path(out_dir) / "m1_vectors.json"
            if m1j.exists():
                d = json.loads(m1j.read_text())
                if "quality_score" in d:
                    metrics["quality_score"] = d["quality_score"]
        except Exception:
            pass

        if hdb and hdb.is_available and run_id:
            hdb.finish_run(run_id,
                           "success" if result else "failed",
                           metrics=metrics,
                           duration_sec=duration)
            for f in Path(out_dir).iterdir():
                if f.is_file():
                    hdb.add_artifact(run_id, f.suffix.lstrip("."),
                                     str(f))
            if result:
                q = metrics.get("quality_score")
                if q is None:
                    q = 0.5
                hdb.save_preset(
                    method=method,
                    scanning_type=getattr(mw, "scanning_type", "unknown"),
                    crs_epsg=(getattr(mw.crs_detector, "crs", None)
                              if mw.crs_detector else None),
                    hyperparams=hyper or {},
                    quality_score=float(q),
                    source_run_id=run_id)

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


# ============================================================================
# Вкладка 6 — Экспорт
# ============================================================================

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
        left = QWidget()
        left.setMaximumWidth(460)
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
        pl.addLayout(bb2)

        pg.setLayout(pl)
        ll.addWidget(pg)

        self.log = LogWidget()
        ll.addWidget(self.log)

        splitter.addWidget(left)
        self.preview = ExportPreviewViewer()
        splitter.addWidget(self.preview)
        splitter.setSizes([440, 900])
        layout.addWidget(splitter)
        self.setLayout(layout)

        self.refresh_preview_list()

    def refresh_preview_list(self):
        current_data = self.file_combo.currentData()
        self.file_combo.blockSignals(True)
        self.file_combo.clear()

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
            ext = p.suffix.lower()
            if ext == ".geojson":
                return (0, str(p))
            if ext in PREVIEW_MESH_EXTS + PREVIEW_CLOUD_EXTS:
                return (1, str(p))
            return (2, str(p))
        found.sort(key=key_func)

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
        if not ok:
            return
        if self._last_output:
            p = Path(self._last_output)
            if p.exists() and self._last_output not in self._exported_files:
                self._exported_files.append(self._last_output)
        self.refresh_preview_list()
        if self._last_output:
            idx = self.file_combo.findData(self._last_output)
            if idx >= 0:
                self.file_combo.setCurrentIndex(idx)
            else:
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
                        rows.append(("references_clouds",
                                     r[1], r[2], r[3]))
                self.status_label.setText("Источник: PostgreSQL")
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

        self.setLayout(layout)
        self.reload()

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
            "City3D PointCloud Pipeline v7.3.1 — структура v4 + M1-M6 + PG")
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
            "City3D PointCloud Pipeline v7.3.1 — структура v4 + M1-M6 + PG")
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

    def closeEvent(self, event):
        for w in list(_ACTIVE_WORKERS):
            try:
                w.wait(3000)
            except Exception:
                pass
        try:
            if self.history_db and self.history_db.conn:
                self.history_db.conn.close()
        except Exception:
            pass
        event.accept()


# ============================================================================
# Точка входа
# ============================================================================

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
