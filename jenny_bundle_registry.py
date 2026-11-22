"""Resolve exact-hash Jenny recordings and emotion labels from an extracted audio bundle."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import wave

from check_sad_audio import rows
from shared_emotion_counts import MODEL, PREFIXES

VOICE = 'en-US-JennyNeural'
STYLES = ('neutral', 'angry', 'happy', 'sad')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for data in iter(lambda: source.read(4*1024**2), b''):
            h.update(data)
    return h.hexdigest()


class Registry:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.labels = {}
        self.by_hash = {}
        self.options = defaultdict(list)
        self.verified = {}
        manifest = self.root/'JENNY_AUDIO_MANIFEST.json'
        mappings = {}
        if manifest.exists():
            mappings = {r['source']: self.root/r['member'] for r in json.loads(manifest.read_text())['files']}
        inventory = self.root/'metadata/audio_inventory.jsonl'
        if not inventory.exists():
            inventory = self.root/'outputs/shared-three-emotions/audio_inventory.jsonl'
        if not inventory.exists():
            raise FileNotFoundError(f'No Jenny audio inventory under {self.root}; extract the audio ZIP first.')
        for row in rows(inventory):
            key = row['audio_sha256']
            paths = [*row.get('paths', []), row.get('audio_file')]
            for name in paths:
                if not name:
                    continue
                choices = [mappings.get(name)]
                path = Path(name)
                if path.is_absolute() and '/outputs/' in name:
                    choices.append(self.root/'outputs'/name.split('/outputs/', 1)[1])
                if not manifest.exists():
                    choices.append(self.root/path if not path.is_absolute() else path)
                chosen = next((p for p in choices if p and p.is_file()), None)
                if chosen:
                    self.by_hash[key] = chosen.resolve()
                    break
        for source in [*sorted(self.root.rglob('results-emotion2vec.jsonl'))]:
            for row in rows(source):
                key = row.get('audio_sha256') or row.get('clip_id')
                if (key in self.by_hash and row.get('model') == MODEL
                        and row.get('processing_mode') == 'whole_recording'
                        and row.get('classification_status') == 'classified'):
                    previous = self.labels.get(key)
                    if previous and previous['emotion'] != row['emotion']:
                        raise ValueError(f'Conflicting saved emotion labels for hash {key}.')
                    self.labels[key] = row
        # Source plans retain gold references and spoken text; labels alone never supply either.
        for source in sorted(self.root.rglob('plan.jsonl')):
            for row in rows(source):
                q = row.get('question_id') or ''
                name = next((n for n,p in PREFIXES.items() if q.startswith(p)), None)
                if (name is None or row.get('voice') != VOICE or row.get('provider') != 'azure'
                        or row.get('style') not in STYLES or not row.get('audio_path') or not row.get('question')):
                    continue
                path = source.parent/row['audio_path']
                if manifest.exists() and not path.resolve().is_relative_to(self.root):
                    continue
                if not path.is_file():
                    continue
                key = self.verify(path)
                if key not in self.by_hash:
                    continue
                with wave.open(str(path), 'rb') as audio:
                    seconds = audio.getnframes()/audio.getframerate()
                answer = '' if row.get('answer') is None else str(row['answer']).rsplit('####', 1)[-1].strip()
                record = dict(question_id=q, voice=VOICE, style=row['style'], audio_file=str(path.resolve()),
                              audio_sha256=key, seconds=seconds, reference_answer=answer)
                item = dict(record=record, question=row['question'], answer=answer, source_plan=str(source))
                options = self.options[name, q, row['style']]
                if not any(x['record']['audio_sha256'] == key for x in options):
                    options.append(item)

    def verify(self, path):
        path = Path(path).resolve()
        if path not in self.verified:
            self.verified[path] = sha(path)
        return self.verified[path]

    def relocate(self, record):
        key = record['audio_sha256']
        path = self.by_hash.get(key)
        if path is None or self.verify(path) != key:
            raise FileNotFoundError(f'Original audio hash absent/changed in ZIP: {record["question_id"]}/{record.get("style")}: {key}')
        return dict(record, audio_file=str(path))

    def choose(self, name, q, style, *, expected=None, question=None, answer=None, agreed=False):
        options = self.options.get((name, q, style), [])
        for item in sorted(options, key=lambda x: x['record']['audio_sha256']):
            record = item['record']
            if expected and record['audio_sha256'] != expected:
                continue
            if question is not None and item['question'] != question:
                continue
            if answer is not None and item['answer'] != answer:
                continue
            if agreed and self.labels.get(record['audio_sha256'], {}).get('emotion') != style:
                continue
            return item
        return None

    def neutral_ids(self, name):
        return sorted({q for (task,q,style) in self.options if task == name and style == 'neutral'})
