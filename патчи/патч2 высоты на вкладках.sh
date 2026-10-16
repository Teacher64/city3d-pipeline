python3 << 'ENDPY'
#!/usr/bin/env python3
"""PATCH: селекторы CRS/высот на вкладках 1 (Загрузка) и 6 (Экспорт)."""

import py_compile
from datetime import datetime
from pathlib import Path

SCRIPT = Path("/mnt/data_sda1/PR_3D_SPB/city3d-pipeline/city3d_pipeline_v5.py")
text = SCRIPT.read_text(encoding="utf-8")

if "PATCHED_CRS_UI" in text:
    print("[skip] CRS UI уже добавлен")
    raise SystemExit(0)

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
backup = SCRIPT.with_name(SCRIPT.name + f".bak-crsui-{stamp}")
backup.write_text(text, encoding="utf-8")
print(f"[OK] резервная копия: {backup}")

# ── 1. Константы вертикальных CRS ─────────────────────────────────────
CONSTS = '''
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


'''
MARKER_CONST = "# ============================================================================\n# Справочник EPSG"
if MARKER_CONST in text:
    text = text.replace(MARKER_CONST, CONSTS + MARKER_CONST, 1)
    print("[OK] константы VERTICAL_CRS_LIST добавлены")

# ── 2. В LoadDataTab: расширить блок CRS ──────────────────────────────
OLD_CRS_BLOCK = '''        cg = QGroupBox("Система координат (CRS)")
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
        tl.addWidget(cg)'''

NEW_CRS_BLOCK = '''        # PATCHED_CRS_UI: расширенный блок CRS + вертикальная СК
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
        tl.addWidget(cg)'''

if OLD_CRS_BLOCK not in text:
    print("[WARN] блок CRS на вкладке 1 не найден")
else:
    text = text.replace(OLD_CRS_BLOCK, NEW_CRS_BLOCK, 1)
    print("[OK] вкладка 1: расширенный блок CRS + вертикальная СК")

# ── 3. В ExportTab: добавить CRS/высоты ──────────────────────────────
OLD_EXP_PARAMS = '''        g2 = QGroupBox("Параметры")
        g2l = QFormLayout()
        self.name = QLineEdit("city_model")
        g2l.addRow("Имя:", self.name)
        self.use_cropped = QCheckBox("Использовать обрезанное")
        g2l.addRow("", self.use_cropped)
        g2.setLayout(g2l)
        ll.addWidget(g2)'''

NEW_EXP_PARAMS = '''        g2 = QGroupBox("Параметры")
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
        ll.addWidget(g2)'''

if OLD_EXP_PARAMS not in text:
    print("[WARN] блок параметров на вкладке 6 не найден")
else:
    text = text.replace(OLD_EXP_PARAMS, NEW_EXP_PARAMS, 1)
    print("[OK] вкладка 6: добавлены селекторы CRS/высот")

# ── 4. Update export_src_crs при открытии вкладки ────────────────────
OLD_EXP_INIT = '''        self.refresh_preview_list()

    def refresh_preview_list(self):'''
NEW_EXP_INIT = '''        self.refresh_preview_list()
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

    def refresh_preview_list(self):'''

if OLD_EXP_INIT not in text:
    print("[WARN] __init__ ExportTab не найден")
else:
    text = text.replace(OLD_EXP_INIT, NEW_EXP_INIT, 1)
    print("[OK] ExportTab: автообновление метки CRS источника")

SCRIPT.write_text(text, encoding="utf-8")

try:
    py_compile.compile(str(SCRIPT), doraise=True)
    print("[OK] синтаксис корректен")
except py_compile.PyCompileError as e:
    print(f"[FATAL] {e}\\nоткат...")
    SCRIPT.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    raise SystemExit(1)

print()
print("=" * 70)
print("ПАТЧ CRS/ВЫСОТ ГОТОВ.")
print("Вкладка 1: горизонтальный CRS + вертикальная СК + ΔZ")
print("Вкладка 6: целевой CRS + высоты + ΔZ + CRS источника")
print("=" * 70)
ENDPY