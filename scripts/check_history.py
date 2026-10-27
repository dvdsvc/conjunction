"""Verify historical source mappings without importing or executing old modules."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inspect_history():
    index = json.loads((ROOT / 'history/source_index.json').read_text())
    summary = index['summary']
    failures = []
    files = index['files']
    occurrences = index['occurrences']
    excluded = index['excluded']
    hashes = {}

    def check(path, expected):
        resolved = (ROOT / path).resolve()
        if not resolved.is_relative_to(ROOT) or not resolved.is_file() or resolved.is_symlink():
            failures.append('Unsafe/missing historical payload: '+path)
            return
        if path not in hashes:
            hashes[path] = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if hashes[path] != expected:
            failures.append('Historical hash mismatch: '+path)

    for row in files:
        if not row['path'].startswith('history/'):
            failures.append('New historical file outside history/: '+row['path'])
        check(row['path'],row['sha256'])
        if (ROOT/row['path']).is_file() and (ROOT/row['path']).stat().st_size != row['bytes']:
            failures.append('Historical size mismatch: '+row['path'])
    for row in occurrences:
        check(row['payload_path'],row['payload_sha256'])
        if row['treatment']=='byte-identical' and row['source_sha256']!=row['payload_sha256']:
            failures.append('Byte-identity declaration mismatch: '+row['source'])
    sources = [r['source'] for r in occurrences]+[r['source'] for r in excluded]
    if len(set(sources))!=len(sources):
        failures.append('Duplicate logical source occurrence')
    actual = {'selected_occurrences':len(sources),'included_occurrences':len(occurrences),
              'excluded_occurrences':len(excluded),
              'distinct_included_source_contents':len({r['source_sha256'] for r in occurrences}),
              'new_archived_files':len(files),
              'existing_research_contents_reused':len({r['source_sha256'] for r in occurrences if r['location']=='existing-research'}),
              'new_python_files':sum(r['path'].endswith('.py') for r in files),
              'new_shell_files':sum(r['path'].endswith('.sh') for r in files),
              'path_sanitized_files':sum(r['treatment']!='byte-identical' for r in files),
              'git_commits_represented':len(index['git_commits']),
              'included_sources_by_family':dict(Counter(r['source_kind'] for r in occurrences))}
    if any(summary.get(k)!=v for k,v in actual.items()):
        failures.append('Historical summary counts differ from source index')
    allowed = {r['path'] for r in files}|{'history/README.md','history/README.zh-CN.md','history/source_index.json'}
    present = {str(p.relative_to(ROOT)) for p in (ROOT/'history').rglob('*') if p.is_file()}
    if allowed != present:
        failures.append('Historical directory differs from indexed payload')
    if summary['syntax_issues'] or summary['shell_syntax_issues']:
        failures.append('Historical syntax issues require explicit review')
    return {**actual,'payloads_hash_checked':len(hashes),'failures':failures}


def main():
    result = inspect_history()
    print(json.dumps(result,indent=2))
    if result['failures']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
