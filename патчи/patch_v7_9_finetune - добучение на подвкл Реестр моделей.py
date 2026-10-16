#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_v7_9_finetune.py — дообучение локальных моделей на эталонах.

Добавляет:
  • класс FineTuner — обучение U-Net / LightGBM на эталонах из refs
  • кнопку «Дообучить» в ModelRegistryWidget
  • сохранение чекпоинтов в ~/models/<method>/<name>_<date>.pth
"""
from pathlib import Path
from datetime import datetime

SCRIPT = Path.home() / "city3d_pipeline_v5.py"

FINETUNER_CODE = '''

class FineTuner:
    """Дообучение моделей на эталонах проекта.

    Сценарии:
      1. U-Net по маскам зданий → M3
      2. LightGBM по признакам M1–M6 → fuse
      3. RandomForest по DEM-фичам → M4
    """

    def __init__(self, log_widget=None):
        self.log = log_widget
        self.models_dir = Path.home() / "models"
        self.models_dir.mkdir(exist_ok=True)

    def finetune_unet(self, images_dir, masks_dir, output_name="unet_buildings",
                      epochs=20, batch=4, lr=1e-4):
        """Дообучение U-Net на парах (изображение, маска)."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Dataset
            import segmentation_models_pytorch as smp
            from PIL import Image
            import numpy as np
        except ImportError as e:
            return {"ok": False, "error": f"не хватает пакетов: {e}"}

        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)
        images = sorted(images_dir.glob("*.png")) + \\
                 sorted(images_dir.glob("*.tif"))
        if not images:
            return {"ok": False, "error": "нет изображений"}

        class SegDataset(Dataset):
            def __init__(self, imgs, mdir, size=256):
                self.imgs = imgs
                self.mdir = mdir
                self.size = size
            def __len__(self):
                return len(self.imgs)
            def __getitem__(self, i):
                img = Image.open(self.imgs[i]).convert("RGB")
                img = img.resize((self.size, self.size))
                mname = self.imgs[i].stem + "_mask.png"
                m = Image.open(self.mdir / mname).convert("L")
                m = m.resize((self.size, self.size))
                return (torch.from_numpy(np.array(img)/255.0)
                             .permute(2, 0, 1).float(),
                        (torch.from_numpy(np.array(m)) > 127)
                             .long())

        ds = SegDataset(images, masks_dir)
        dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = smp.Unet(encoder_name="resnet34",
                         encoder_weights="imagenet",
                         in_channels=3, classes=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for ep in range(epochs):
            total = 0.0
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
                total += loss.item()
            avg = total / max(1, len(dl))
            if self.log:
                self.log.log(f"U-Net epoch {ep+1}/{epochs}: "
                             f"loss={avg:.4f}", "INFO")

        out_path = (self.models_dir / "M3" /
                    f"{output_name}_{datetime.now():%Y%m%d}.pth")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        if self.log:
            self.log.log(f"U-Net сохранён: {out_path}", "SUCCESS")
        return {"ok": True, "path": str(out_path)}

    def finetune_lgbm(self, features_npz, labels_npy,
                      output_name="lgbm_fuse"):
        """Обучение LightGBM-стеккинга по признакам M1–M6."""
        try:
            import numpy as np
            import lightgbm as lgb
        except ImportError as e:
            return {"ok": False, "error": str(e)}
        data = np.load(features_npz)
        X = data["X"]  # shape (N, F)
        y = np.load(labels_npy)
        ds = lgb.Dataset(X, label=y)
        params = {"objective": "multiclass",
                  "num_class": int(y.max()) + 1,
                  "metric": "multi_logloss",
                  "learning_rate": 0.05,
                  "num_leaves": 63,
                  "verbose": -1}
        bst = lgb.train(params, ds, num_boost_round=300)
        out_path = (self.models_dir / "ensemble" /
                    f"{output_name}_{datetime.now():%Y%m%d}.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bst.save_model(str(out_path))
        if self.log:
            self.log.log(f"LightGBM сохранён: {out_path}", "SUCCESS")
        return {"ok": True, "path": str(out_path)}

    def finetune_rf_dem(self, features_npz, labels_npy,
                        output_name="rf_terrain"):
        try:
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            import joblib
        except ImportError as e:
            return {"ok": False, "error": str(e)}
        data = np.load(features_npz)
        X, y = data["X"], np.load(labels_npy)
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        clf.fit(X, y)
        out_path = (self.models_dir / "M4" /
                    f"{output_name}_{datetime.now():%Y%m%d}.pkl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, out_path)
        if self.log:
            self.log.log(f"RandomForest сохранён: {out_path}", "SUCCESS")
        return {"ok": True, "path": str(out_path)}
'''

# Кнопка «Дообучить» в ModelRegistryWidget
ADD_BUTTON_HOOK = '''        b_open = QPushButton("Открыть models.json")
        b_open.clicked.connect(self._open_json)
        top.addWidget(b_open)'''

NEW_BUTTONS = '''        b_open = QPushButton("Открыть models.json")
        b_open.clicked.connect(self._open_json)
        top.addWidget(b_open)
        b_ft = QPushButton("Дообучить…")
        b_ft.clicked.connect(self._finetune_dialog)
        top.addWidget(b_ft)'''

FINETUNE_DIALOG = '''    def _finetune_dialog(self):
        from PyQt5.QtWidgets import QInputDialog
        mid = self._current_id()
        if not mid:
            QMessageBox.warning(self, "Ошибка", "Выберите модель.")
            return
        mode, ok = QInputDialog.getItem(
            self, "Дообучение",
            "Выберите тип дообучения:",
            ["U-Net (изображения+маски)",
             "LightGBM (features.npz+labels.npy)",
             "RandomForest DEM (features.npz+labels.npy)"],
            0, False)
        if not ok:
            return
        ft = FineTuner(log_widget=None)
        if mode.startswith("U-Net"):
            imgs = QFileDialog.getExistingDirectory(
                self, "Папка с изображениями")
            if not imgs:
                return
            masks = QFileDialog.getExistingDirectory(
                self, "Папка с масками")
            if not masks:
                return
            res = ft.finetune_unet(imgs, masks)
        elif mode.startswith("LightGBM"):
            npz, _ = QFileDialog.getOpenFileName(
                self, "features.npz", "", "NPZ (*.npz)")
            if not npz:
                return
            npy, _ = QFileDialog.getOpenFileName(
                self, "labels.npy", "", "NPY (*.npy)")
            if not npy:
                return
            res = ft.finetune_lgbm(npz, npy)
        else:
            npz, _ = QFileDialog.getOpenFileName(
                self, "features.npz", "", "NPZ (*.npz)")
            if not npz:
                return
            npy, _ = QFileDialog.getOpenFileName(
                self, "labels.npy", "", "NPY (*.npy)")
            if not npy:
                return
            res = ft.finetune_rf_dem(npz, npy)
        if res.get("ok"):
            QMessageBox.information(
                self, "Готово", f"Модель сохранена:\\n{res['path']}")
            self.reload()
        else:
            QMessageBox.warning(self, "Ошибка", res.get("error", "?"))
'''


def main():
    text = SCRIPT.read_text(encoding="utf-8")
    if "class FineTuner" in text:
        print("[INFO] FineTuner уже вставлен.")
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = SCRIPT.with_name(SCRIPT.name + f".bak-{stamp}")
    backup.write_text(text, encoding="utf-8")
    print(f"[OK] Резервная копия: {backup}")

    # Вставить FineTuner перед ModelRegistryWidget
    marker = "class ModelRegistryWidget(QWidget):"
    if marker not in text:
        print("[FATAL] Нет ModelRegistryWidget — примените patch_v7_5.")
        return
    text = text.replace(marker, FINETUNER_CODE + "\n\n" + marker, 1)

    # Добавить кнопку «Дообучить»
    if ADD_BUTTON_HOOK not in text:
        print("[WARN] Не найден хук кнопки — пропуск.")
    else:
        text = text.replace(ADD_BUTTON_HOOK, NEW_BUTTONS, 1)
        # Вставить метод _finetune_dialog после _open_json
        hook2 = "    def _open_json(self):"
        idx = text.find(hook2)
        if idx > 0:
            # Найти конец метода
            end = text.find("\n\nclass ", idx)
            text = text[:end] + "\n" + FINETUNE_DIALOG + text[end:]

    SCRIPT.write_text(text, encoding="utf-8")
    print(f"[OK] FineTuner + кнопка 'Дообучить' в {SCRIPT}")


if __name__ == "__main__":
    main()