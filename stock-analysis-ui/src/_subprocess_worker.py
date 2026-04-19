#!/usr/bin/env python3
"""Subprocess worker for stock analysis.

Runs analysis/download tasks in an isolated process to prevent
curl_cffi heap corruption from affecting the Qt UI process.

Usage: python -u _subprocess_worker.py <args_pickle> <result_pickle>
"""
import sys
import os
import pickle


def main():
    if len(sys.argv) < 3:
        print("Usage: _subprocess_worker.py <args_file> <result_file>", file=sys.stderr)
        sys.exit(1)

    args_file = sys.argv[1]
    result_file = sys.argv[2]

    with open(args_file, 'rb') as f:
        args = pickle.load(f)

    # Setup import paths (same as main_window.py's PROJECT_SRC)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    os.chdir(src_dir)

    # Safety: ensure C acceleration and online consensus are disabled
    os.environ.setdefault('QSI_DISABLE_C_ACCELERATION', '1')
    os.environ.setdefault('QSI_CONSENSUS_OFFLINE', '1')

    task = args.get('task')
    try:
        if task == 'analyse_signaux':
            _run_analysis(args, result_file)
        elif task == 'download':
            _run_download(args, result_file)
        else:
            raise ValueError(f"Unknown task: {task}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            with open(result_file, 'wb') as f:
                pickle.dump({'success': False, 'error': str(e)}, f)
        except Exception:
            pass
        sys.exit(1)


def _run_analysis(args, result_file):
    from qsi import analyse_signaux_populaires

    result = analyse_signaux_populaires(
        args['symbols'],
        args['mes_symbols'],
        period=args.get('period', '12mo'),
        afficher_graphiques=False,
        plot_all=False,
        verbose=True,
        taux_reussite_min=args.get('taux_reussite_min', 30),
        min_holding_days=args.get('min_holding_days', 7),
    )
    with open(result_file, 'wb') as f:
        pickle.dump({'success': True, 'result': result}, f, protocol=pickle.HIGHEST_PROTOCOL)


def _run_download(args, result_file):
    from qsi import download_stock_data

    data = download_stock_data(args['symbols'], args.get('period', '12mo'))
    result = {'data': data}

    with open(result_file, 'wb') as f:
        pickle.dump({'success': True, 'result': result}, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
