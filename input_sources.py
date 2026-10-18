"""Discover recorder runs in directories and independently extracted ZIP archives."""
from pathlib import Path
import stat
import zipfile


def discover_runs(root):
    return sorted({p.parent for p in root.rglob('trajectories') if p.is_dir()} |
                  {p.parent.parent for p in root.rglob('statistics/frames.csv')})


def extract_zip(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            name = member.filename
            target = (base/name).resolve()
            if (not target.is_relative_to(base) or '\\' in name or
                    stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f'Unsafe ZIP member path: {name}')
        archive.extractall(base)


def collect_inputs(source, temporary, exclude=None):
    """Return (run, permanent provenance) pairs and per-archive failures.

    Archives with identical internal directory names have separate extraction
    roots. A damaged archive does not hide valid experiments in its siblings.
    """
    source = Path(source).resolve()
    runs, failures = [], []
    if source.is_dir():
        runs = [(r, str(r)) for r in discover_runs(source) if exclude is None or not r.is_relative_to(exclude)]
        archives = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() == '.zip'
                          and (exclude is None or not p.is_relative_to(exclude)))
    elif source.is_file() and source.suffix.lower() == '.zip':
        archives = [source]
    else:
        raise ValueError(f'Expected an experiment directory or ZIP archive: {source}')
    for i, archive in enumerate(archives):
        base = Path(temporary)/f'archive_{i:04d}'
        try:
            extract_zip(archive, base)
            found = discover_runs(base)
            if not found: raise ValueError('No recorder experiments found inside ZIP.')
            runs.extend((r, str(archive)+'::'+str(r.relative_to(base))) for r in found)
        except (ValueError, OSError, zipfile.BadZipFile, RuntimeError) as exc:
            if source.is_file(): raise ValueError(str(exc)) from exc
            failures.append({'source': str(archive), 'run': archive.name, 'error': str(exc)})
    return runs, failures
