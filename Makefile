.PHONY: setup data shift_analysis models conformal eval figures tables all clean test

setup:
	pip install -r requirements.txt
	pip install -e .

data:
	python -c "from src.config import load_config, config_to_dict; from src.datasets import load_all_datasets, save_processed_datasets; c=config_to_dict(load_config()); d=load_all_datasets(c); save_processed_datasets(d, c['processed_data_dir']); print(f'Loaded {len(d)} datasets')"

shift_analysis:
	python -c "from src.config import load_config, config_to_dict; from src.datasets import load_all_datasets; from src.shift_analysis import run_shift_analysis; c=config_to_dict(load_config()); d=load_all_datasets(c); run_shift_analysis(d, 'results/shift_analysis'); print('Shift analysis done')"

models:
	python -c "from src.base_models import LightGBMTuner; print('Base models module OK')"

conformal:
	python -c "from src.conformal import SplitConformal, ShiftSafeCP; print('Conformal module OK')"

eval:
	python -c "from src.metrics import evaluate_all_methods; print('Metrics module OK')"

figures:
	python -c "from src.visualization import generate_all_figures; print('Visualization module OK')"

tables:
	python -c "from src.paper_tables import generate_all_tables; print('Paper tables module OK')"

all: setup
	python run_all.py --config configs/base.yaml --seed 42 --n-seeds 1 --device cuda --n-trials 5

test:
	pytest tests/ -v

clean:
	rm -rf results/logs __pycache__ .pytest_cache *.egg-info
	find . -name "*.pyc" -delete 2>/dev/null || true
