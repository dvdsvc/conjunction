"""Scan the tracked publication payload; reports never contain matched values."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true', help='Also scan every reachable Git revision')
    args = parser.parse_args()
    tool = shutil.which('gitleaks')
    if not tool:
        raise SystemExit('Install Gitleaks 8.30.1 or newer and add it to PATH; no scan was performed.')
    with tempfile.TemporaryDirectory(prefix='publication-secret-check-') as directory:
        temp = Path(directory)
        files = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT)
        names = sorted({x.decode() for x in files.split(b'\0') if x})
        payload = temp / 'payload'
        payload.mkdir()
        for name in names:
            source = ROOT / name
            if source.is_symlink():
                raise SystemExit('Refusing tracked symlink: ' + name)
            if not source.is_file():
                raise SystemExit('Tracked file missing: ' + name)
            target = payload / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        modes = [('current', 'dir', payload)]
        if args.history:
            modes.append(('history', 'git', ROOT))
        findings = []
        for label, mode, target in modes:
            report = temp / (label + '.json')
            command = [tool, mode, str(target), '--config', str(ROOT / '.gitleaks.toml'),
                '--no-banner', '--no-color', '--redact=100', '--ignore-gitleaks-allow',
                '--max-archive-depth=3', '--max-decode-depth=5',
                '--report-format=json', '--report-path', str(report)]
            if mode == 'git':
                command += ['--log-opts=--all --full-history']
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            if result.returncode not in (0, 1) or not report.exists():
                raise SystemExit('Credential scanner failed; diagnostic output suppressed to protect values.')
            for row in json.loads(report.read_text()):
                path = row['File']
                if mode == 'dir':
                    path = str(Path(path).relative_to(payload)) if Path(path).is_absolute() else path
                findings.append({'scope': label, 'rule': row['RuleID'], 'file': path,
                    'line': row['StartLine'], 'commit': row.get('Commit', '')})
        print(json.dumps({'checked_files': len(names), 'history_checked': args.history,
            'findings': findings, 'passed': not findings}, indent=2))
        if findings:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
