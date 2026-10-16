"""Score a leaderboard submission (a GitHub issue) and update the leaderboard.

Reads ISSUE_BODY, ISSUE_NUMBER, ISSUE_USER, ISSUE_CREATED and REPO_OWNER from the
environment, writes the reply to the path given as argument, and prints accepted=true|false
to $GITHUB_OUTPUT."""
import datetime
import io
import json
import os
from pathlib import Path
import re
import sys
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from score import CLS, REG, load_test_labels, metrics

ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS = ROOT / 'submissions.json'
LEADERBOARD = ROOT / 'docs' / 'leaderboard.json'
TZ = ZoneInfo('Europe/Copenhagen')
MAX_BYTES = 5_000_000
ATTACHMENT = re.compile(r'https://github\.com/user-attachments/files/\d+/[\w.\-]+')
TEAM = re.compile(r'^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,39}$')
PRIMARY = {'regression': 'ST-RAE', 'classification': 'MCC'}
SCORE_NAME = {'regression': 'MA-ST-RAE', 'classification': 'MCC'}


class Rejected(Exception):
    pass


def parse_form(body):
    fields = {}
    for block in re.split(r'^### ', body or '', flags=re.M)[1:]:
        title, _, value = block.partition('\n')
        value = value.strip()
        fields[title.strip()] = '' if value == '_No response_' else value
    return fields


def download(text):
    urls = ATTACHMENT.findall(text)
    if len(urls) != 1:
        raise Rejected('Attach exactly one CSV file (drag `test_predictions.csv` into the form).')
    with urllib.request.urlopen(urls[0], timeout=60) as response:
        content = response.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise Rejected('The file is larger than 5 MB.')
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        raise Rejected('The attachment could not be read as a CSV file.')


def check_predictions(frame, labels, targets, task):
    missing_cols = [c for c in ['Molecule_Name'] + targets if c not in frame.columns]
    if missing_cols:
        raise Rejected(f'Missing columns: {", ".join(missing_cols)}. '
                       f'Submit the `test_predictions.csv` written by `finetune.py` for the {task} task.')
    if frame.Molecule_Name.duplicated().any():
        raise Rejected('Some molecules appear more than once.')
    expected = set(labels.Molecule_Name)
    got = set(frame.Molecule_Name)
    if got != expected:
        raise Rejected(f'The molecules do not match the test split: {len(expected - got)} missing and '
                       f'{len(got - expected)} unexpected. Use the split made by `prepare_data.py` '
                       f'with the default seed (42), and the predictions for this task.')
    values = frame.set_index('Molecule_Name').loc[labels.Molecule_Name, targets]
    values = values.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise Rejected('All predictions must be finite numbers.')
    if task == 'classification' and ((values < 0) | (values > 1)).any():
        raise Rejected('Classification predictions must be probabilities between 0 and 1.')
    return values


def best_per_team(records, task):
    primary = PRIMARY[task]
    lower_is_better = task == 'regression'
    best = {}
    for r in records:
        if r['task'] != task:
            continue
        key = r['team'].lower()
        score = r['scores']['macro_' + primary]
        current = best.get(key)
        if (current is None or (score < current['score'] if lower_is_better else score > current['score'])):
            best[key] = {'team': r['team'], 'model': r['model'], 'score': score,
                         'submitted': r['time'], 'issue': r['issue']}
    counts = {}
    for r in records:
        if r['task'] == task:
            counts[r['team'].lower()] = counts.get(r['team'].lower(), 0) + 1
    rows = sorted(best.values(), key=lambda e: e['score'], reverse=not lower_is_better)
    for e in rows:
        e['submissions'] = counts[e['team'].lower()]
    return rows


def summarize(scores, targets, task):
    primary = PRIMARY[task]
    secondary = 'MAE' if task == 'regression' else 'AUROC'
    return {'macro_' + primary: scores['macro_' + primary],
            'per_target': {t: scores[t][primary] for t in targets},
            'secondary': {t: scores[t][secondary] for t in targets}}


def process(env):
    fields = parse_form(env['ISSUE_BODY'])
    team = fields.get('Team name', '').strip()
    task = fields.get('Task', '').strip().lower()
    model = ' '.join(fields.get('Model description', '').split())[:200]
    if not TEAM.match(team):
        raise Rejected('Team name must be 1-40 characters: letters, digits, spaces, `_`, `-` or `.`.')
    if task not in PRIMARY:
        raise Rejected('Task must be `regression` or `classification`.')
    if not model:
        raise Rejected('Add a short model description.')
    user = env['ISSUE_USER']
    created = datetime.datetime.fromisoformat(env['ISSUE_CREATED'].replace('Z', '+00:00'))
    day = created.astimezone(TZ).date().isoformat()
    records = json.loads(SUBMISSIONS.read_text()) if SUBMISSIONS.exists() else []
    if user != env['REPO_OWNER']:
        for r in records:
            same = r['team'].lower() == team.lower() or r['user'] == user
            if same and r['task'] == task and r['day'] == day:
                raise Rejected(f'Only one {task} submission per team per day is allowed '
                               f'(already used today by #{r["issue"]}). Try again tomorrow '
                               f'(days are counted in Copenhagen time).')
    targets = REG if task == 'regression' else CLS
    labels = load_test_labels(task)
    predictions = check_predictions(download(fields.get('Predictions file', '')), labels, targets, task)
    scores = summarize(metrics(labels, predictions, targets, task), targets, task)
    records.append({'issue': int(env['ISSUE_NUMBER']), 'user': user, 'team': team, 'task': task,
                    'model': model, 'time': created.isoformat(), 'day': day, 'scores': scores})
    SUBMISSIONS.write_text(json.dumps(records, indent=2) + '\n')
    board = {'updated': created.isoformat(),
             'regression': best_per_team(records, 'regression'),
             'classification': best_per_team(records, 'classification')}
    LEADERBOARD.write_text(json.dumps(board, indent=2) + '\n')
    rank = next(i for i, e in enumerate(board[task], 1) if e['team'].lower() == team.lower())
    score = scores['macro_' + PRIMARY[task]]
    return (f'**Accepted.** Team **{team}**, {task}: {SCORE_NAME[task]} = **{score:.4f}**\n\n'
            f'Your team is ranked **{rank}** on the {task} leaderboard (best submission per team).')


def main():
    reply_path, accepted = Path(sys.argv[1]), False
    try:
        reply, accepted = process(os.environ), True
    except Rejected as e:
        reply = f'**Rejected:** {e}\n\nThis does not count towards your daily limit. Fix it and open a new submission.'
    reply_path.write_text(reply + '\n')
    with open(os.environ.get('GITHUB_OUTPUT', os.devnull), 'a') as f:
        f.write(f'accepted={str(accepted).lower()}\n')


if __name__ == '__main__':
    main()
