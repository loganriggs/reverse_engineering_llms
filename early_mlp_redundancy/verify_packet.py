"""Verify all packaged file hashes; no third-party dependencies or model required."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text())
for row in manifest['files']:
    path=root/row['path']
    if not path.is_file(): raise SystemExit('Missing: '+row['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        raise SystemExit('Hash mismatch: '+row['path'])
print(f"Verified {len(manifest['files'])} files")
