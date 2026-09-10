"""CRC-checked extraction into raw; record archive SHA256 and file inventory."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
archive = root / 'downloads/teeth.zip'
destination = (root / 'raw').resolve()
destination.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        target = (destination / info.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError('Unsafe archive path')
        # ZipFile validates CRC while reading each member during extraction.
        z.extract(info, destination)
    inventory = [{'path': i.filename, 'bytes': i.file_size, 'crc32': i.CRC} for i in z.infolist()]
with archive.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
(root / 'downloads/dataset_provenance.json').write_text(json.dumps({
    'source': 'https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images',
    'version': 1, 'doi': '10.34740/KAGGLE/DSV/5884500', 'license': 'CC0-1.0',
    'archive_bytes': archive.stat().st_size, 'sha256': digest, 'inventory': inventory
}, indent=2) + '\n')
print(json.dumps({'extracted_files': len(inventory), 'sha256': digest}), flush=True)
