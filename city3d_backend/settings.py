import json
import os
import shutil
import tempfile
from pathlib import Path

SETTINGS_FILE = Path.home() / '.config/city3d/settings.json'

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def read_settings(path=SETTINGS_FILE):
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('settings.json должен содержать объект JSON')
    return data

def validate_paths(data):
    for key, value in data.get('paths', {}).items():
        if key in ('settings_dir', 'settings_file'):
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Путь должен быть непустой строкой: ' + key)
        target = Path(value).expanduser()
        if not target.is_absolute():
            raise ValueError('Нужен абсолютный путь: ' + key)
        parent = target if target.exists() and target.is_dir() else target.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not os.access(parent, os.W_OK):
            raise PermissionError('Нет права записи для пути: ' + key)

def save_settings(data, path=SETTINGS_FILE):
    validate_paths(data)
    path = Path(path)
    if path.exists():
        shutil.copy2(path, path.with_suffix('.json.bak'))
    data = dict(data, schema_version=3)
    atomic_json(path, data)

def resolve_paths(data, code_dir):
    cfg = data.get('paths', {})
    historical = Path('/mnt/data_sda1/PR_3D_SPB')
    default = historical if historical.is_dir() and os.access(historical, os.W_OK) else Path.home() / 'City3D'
    root = Path(cfg.get('project_root', default)).expanduser()
    workspace = Path(cfg.get('workspace', root / 'workspace')).expanduser()
    defaults = dict(project_root=root, workspace=workspace,
                    datasets_dir=workspace / 'datasets', training_output_dir=workspace / 'training_output',
                    projects_dir=Path.home() / 'projects', reference_db=workspace / 'references.sqlite',
                    epsg_catalog=Path(code_dir) / 'crslist.json', slices_dir=workspace / 'slices',
                    hf_cache=root / 'cache/huggingface')
    # Settings location is fixed; these GUI fields are read-only.
    defaults.update(settings_dir=SETTINGS_FILE.parent, settings_file=SETTINGS_FILE)
    return {k: Path(cfg.get(k, v)).expanduser() if k not in ('settings_dir', 'settings_file') else v for k, v in defaults.items()}
