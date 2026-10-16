"""Operator-maintained CRS catalogue with immutable revision backups."""
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from pyproj import CRS


class CustomCatalog:
    def __init__(self, path=None):
        self.path = Path(path or Path.home() / '.config/city3d/custom_crs.json')

    def entries(self):
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if data.get('schema') != 'city3d:custom-crs:1' or not isinstance(data.get('entries'), list):
            raise ValueError('Неподдерживаемый формат справочника МСК')
        seen = set()
        for item in data['entries']:
            if item['id'] in seen:
                raise ValueError('Повторяющийся ID МСК')
            seen.add(item['id'])
            CRS.from_wkt(item['wkt'])
        return data['entries']

    def write(self, entries):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            archive = self.path.parent / 'custom_crs_history'
            archive.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            shutil.copy2(self.path, archive / f'{stamp}_{uuid.uuid4().hex[:8]}.json')
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix='custom_crs_', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump({'schema': 'city3d:custom-crs:1', 'entries': entries}, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def save(self, name, definition, source, entry_id=None):
        if not name.strip() or not source.strip():
            raise ValueError('Укажите название и источник параметров МСК')
        crs = CRS.from_user_input(definition)
        if not (crs.is_projected or crs.is_geographic or crs.is_compound):
            raise ValueError('Нужна горизонтальная либо составная CRS')
        entries = self.entries()
        old = next((x for x in entries if x['id'] == entry_id), None)
        if entry_id and old is None:
            raise ValueError('Редактируемая МСК не найдена')
        item = dict(id=entry_id or 'USER:' + str(uuid.uuid4()), name=name.strip(),
                    source=source.strip(), wkt=crs.to_wkt(),
                    revision=(old['revision'] + 1 if old else 1),
                    updated_at=datetime.now(timezone.utc).isoformat(), verification='operator_supplied')
        self.write([x for x in entries if x['id'] != item['id']] + [item])
        return item

    def delete(self, entry_id):
        entries = self.entries()
        if not any(x['id'] == entry_id for x in entries):
            raise ValueError('МСК не найдена')
        self.write([x for x in entries if x['id'] != entry_id])
