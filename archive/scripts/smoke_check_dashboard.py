import os
import pandas as pd

LOGS = 'logs'

def load_latest_metrics():
    candidates = []
    if not os.path.exists(LOGS):
        return None, None
    for fname in os.listdir(LOGS):
        if fname.startswith('models_metrics') and fname.endswith('.csv'):
            candidates.append(os.path.join(LOGS, fname))
    if not candidates:
        return None, None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    path = candidates[0]
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = None
    return path, df


def load_feature_map():
    path = os.path.join(LOGS, 'feature_name_map.csv')
    if os.path.exists(path):
        try:
            return path, pd.read_csv(path)
        except Exception:
            return path, None
    return None, None


def load_pareto():
    # prefer pareto_feature_importance_with_base, else pareto_base_top50
    p1 = os.path.join(LOGS, 'pareto_feature_importance_with_base.csv')
    p2 = os.path.join(LOGS, 'pareto_base_top50.csv')
    if os.path.exists(p1):
        try:
            return p1, pd.read_csv(p1)
        except Exception:
            return p1, None
    if os.path.exists(p2):
        try:
            return p2, pd.read_csv(p2)
        except Exception:
            return p2, None
    return None, None


if __name__ == '__main__':
    print('Running dashboard smoke-check...')
    path, metrics = load_latest_metrics()
    if path is None:
        print('ERROR: No metrics CSV found in logs/.')
    else:
        print('Loaded metrics from:', path)
        if metrics is None:
            print('  Could not parse metrics CSV.')
        else:
            print('  Metrics head:')
            with pd.option_context('display.max_rows', 10, 'display.max_columns', 6):
                print(metrics.head())

    fmap_path, fmap = load_feature_map()
    if fmap_path is None:
        print('WARNING: feature_name_map.csv not found in logs/. Dashboard will fallback to engineered names.')
    else:
        print('Loaded feature map from:', fmap_path)
        if fmap is None:
            print('  feature_name_map.csv could not be parsed')
        else:
            print('  feature_name_map sample (engineered -> base):')
            print(fmap.head(10))

    pareto_path, pareto_df = load_pareto()
    if pareto_path is None:
        print('WARNING: No pareto CSV found (with base names).')
    else:
        print('Loaded pareto from:', pareto_path)
        if pareto_df is None:
            print('  Could not parse pareto CSV')
        else:
            print('  Pareto sample:')
            with pd.option_context('display.max_rows', 12, 'display.max_columns', 6):
                print(pareto_df.head(12))

    print('\nSmoke-check complete.')
