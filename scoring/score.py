"""Test labels and metrics, identical to prepare_data.py and finetune.py in the starter repo
(https://github.com/khelverskovp/dtu-cyp-project)."""
import io
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, mean_absolute_error, r2_score, roc_auc_score, average_precision_score

DATA_REVISION = '146282d2ce1532f1803679a834a9bec9dac1e482'
REG = [f'{c}_pIC50_direct_inhibition' for c in ['CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4']]
CLS = ['CYP2D6_is_TDI', 'CYP3A4_is_TDI']
FILES = ['cyp-challenge-TRAIN_inhibition.csv', 'cyp-challenge-TRAIN_TDI.csv']
TEST_MOLECULES = Path(__file__).with_name('test_molecules.csv')


def load_test_labels():
    """Labels of the starter repo's test split, from the public challenge training data."""
    tables = []
    for name in FILES:
        url = f'https://huggingface.co/datasets/openadmet/cyp-challenge-train-test/resolve/{DATA_REVISION}/{name}'
        with urllib.request.urlopen(url, timeout=120) as response:
            tables.append(pd.read_csv(io.BytesIO(response.read())))
    reg, cls = tables
    frame = reg.merge(cls[['Molecule_Name', 'SMILES'] + CLS], on=['Molecule_Name', 'SMILES'],
                      how='outer', validate='one_to_one')
    mapping = {'true': 1., 'false': 0., '1': 1., '0': 0., '1.0': 1., '0.0': 0.}
    for col in CLS:
        frame[col] = frame[col].astype(str).str.lower().map(mapping)
    names = pd.read_csv(TEST_MOLECULES).Molecule_Name
    test = frame.set_index('Molecule_Name').loc[names].reset_index()
    if len(test) != len(names):
        raise RuntimeError('Test molecules missing from the challenge data')
    return test


def metrics(frame, predictions, targets, task):
    result = {}
    for j, target in enumerate(targets):
        y = frame[target].to_numpy(dtype=float)
        mask = np.isfinite(y)
        y, pred = y[mask], predictions[mask, j]
        if not len(y):
            result[target] = {'n': 0, ('ST-RAE' if task == 'regression' else 'MCC'): None}
            continue
        if task == 'regression':
            low = frame[target + '_conf_low'].to_numpy(dtype=float)[mask]
            high = frame[target + '_conf_high'].to_numpy(dtype=float)[mask]
            if not (np.isfinite(low).all() and np.isfinite(high).all() and (low <= high).all()):
                raise ValueError(f'Invalid credible intervals for {target}')
            distance = lambda p: np.maximum(low - p, 0) + np.maximum(p - high, 0)
            denom = distance(y.mean()).sum()
            result[target] = {'MAE': float(mean_absolute_error(y, pred)),
                              'R2': float(r2_score(y, pred)) if len(y) > 1 else None,
                              'ST-RAE': float(distance(pred).sum() / denom) if denom > 0 else None,
                              'n': len(y)}
        else:
            both = len(np.unique(y)) == 2
            result[target] = {'MCC': float(matthews_corrcoef(y, pred >= .5)),
                              'AUROC': float(roc_auc_score(y, pred)) if both else None,
                              'AP': float(average_precision_score(y, pred)) if both else None,
                              'n': len(y), 'positive_fraction': float(y.mean())}
    primary = 'ST-RAE' if task == 'regression' else 'MCC'
    scores = [result[t][primary] for t in targets]
    # Do not silently omit an endpoint when reporting a macro score.
    result['macro_' + primary] = (float(np.mean(scores))
        if all(s is not None and np.isfinite(s) for s in scores) else None)
    return result
