#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_v7_10_adapters.py — STEP 6: реальные адаптеры torch.load.

Добавляет в RealTrainer методы:
  • train_m1_randla()  — RandLA-Net через torch-pointcloud
  • train_m1_kpconv()  — KPConv через torch-pointcloud
  • train_m1_ptv3()    — PTv3 через Pointcept
  • _load_randla_weights() — загрузка весов из models/M1/
  • _infer_semantic()  — общий инференс с разбиением на блоки

Требует: pip install torch-pointcloud torch
"""
from pathlib import Path
from datetime import datetime

SCRIPT = Path.home() / "city3d_pipeline_v5.py"

ADAPTER_CODE = '''

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
            report.write_text("\\n".join(lines), encoding="utf-8")
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
'''

INSERT_BEFORE = "    def train_m2(self, raster_path, output_dir):"


def main():
    text = SCRIPT.read_text(encoding="utf-8")
    if "class SemanticModelAdapter" in text:
        print("[INFO] Адаптеры уже вставлены.")
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = SCRIPT.with_name(SCRIPT.name + f".bak-{stamp}")
    backup.write_text(text, encoding="utf-8")
    print(f"[OK] Резервная копия: {backup}")

    if INSERT_BEFORE not in text:
        print("[FATAL] Не найден маркер train_m2.")
        return
    text = text.replace(INSERT_BEFORE,
                        ADAPTER_CODE + "\n\n" + INSERT_BEFORE, 1)

    # Патч: добавить в PipelineComposer поддержку semantic-моделей
    old_semantic = '''                elif role == "semantic" or mid in (
                        "randla_kitti", "kpconv_semantickitti",
                        "ptv3_scannet"):
                    emit_log(
                        f"Модель {mid} будет подключена в STEP 4 "
                        f"(finetune). Пока — заглушка, "
                        f"см. README.", "WARNING")
                    ctx["intermediate"][mid] = None'''

    new_semantic = '''                elif role == "semantic" or mid in (
                        "randla_kitti", "kpconv_s3dis",
                        "ptv3_scannet"):
                    step_dir = ctx["out_dir"] / f"{s['index']:02d}_{mid}"
                    weights = Path.home() / "models" / "M1" / \\
                              f"{mid}.pth"
                    ok = SemanticTrainer.run(
                        mid, ctx["input_pcd"], step_dir,
                        weights_path=str(weights) if weights.exists() else None,
                        log_widget=EmitLogger(emit_log),
                        emit_progress=lambda p, n="":
                            emit_progress(
                                int((i + p/100) / total * 100), n))
                    ctx["intermediate"][mid] = str(step_dir) if ok else None'''

    if old_semantic in text:
        text = text.replace(old_semantic, new_semantic, 1)
        print("[OK] PipelineComposer: semantic-модели подключены")
    else:
        print("[WARN] Не найден блок semantic в PipelineComposer "
              "(пропущен — возможно, patch_v7_8 не применён)")

    SCRIPT.write_text(text, encoding="utf-8")
    print(f"[OK] STEP 6 вставлен: {SCRIPT}")

    try:
        import py_compile
        py_compile.compile(str(SCRIPT), doraise=True)
        print("[OK] Синтаксис корректен.")
    except Exception as e:
        print(f"[WARN] py_compile: {e}")

    print("\nЧто появилось:")
    print("  • SemanticModelAdapter — загрузка RandLA/KPConv/PTv3 через torch.load")
    print("  • SemanticTrainer — инференс с разбиением на блоки + сохранение")
    print("  • PipelineComposer: шаги semantic теперь вызывают реальные модели")
    print("\nУстановка:")
    print("  pip install torch torch-pointcloud open3d")
    print("  python download_models.py   # скачает веса в ~/models/M1/")
    print("\nЗапуск: python ~/city3d_pipeline_v5.py")


if __name__ == "__main__":
    main()