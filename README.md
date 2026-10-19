cat > README.md << 'EOF'
# City3D PointCloud Pipeline

Платформа обработки облаков точек для построения Urban Digital Twin.

## Возможности

- **M1 (CPU)**: HAG, ground extraction, ExtraTrees
- **Semantic**: RandLA-Net (DALES/SensatUrban), KPConv, PTv3
- **FeatureBus**: per-point feature matrix, fuse.stack
- **Импортёры**: ЕЭКО/ФФПД (SXF), ЕГРН, ИСОГД, ТИМ (IFC/LandXML)
- **Публикация**: OGC API Features / 3D GeoVolumes / Tiles / MCP
- **Мониторинг**: Prometheus, structlog, bias audit

## Быстрый старт

```bash
# Клонирование
git clone git@github.com:Teacher64/city3d-pipeline.git
cd city3d-pipeline

# Окружение
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Запуск GUI
python city3d_pipeline_v5.py

# CLI
python -m city3d_backend.cli --help
