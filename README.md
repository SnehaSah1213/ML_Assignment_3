# ML Assignment 3 — CIFAR10

**Name:** Sneha Sah  
**Roll number:** M26CSA008  
**Seed:** 26008

## Repository status
This repository was reconstructed from the executed Colab notebook uploaded with the submission-preparation request.

- `ML_Assignment3.ipynb`: original executed notebook.
- `ML_Assignment3.py`: code cells extracted in notebook order.
- `logs/recovered_notebook_outputs.txt`: text logs recovered from saved cell outputs.
- `results/results_summary.txt`: final summary recovered from the notebook's wrap-up output.
- `figures/`: image outputs embedded in the notebook (when available).
- `configs/config.json`: experiment settings extracted from the notebook.
- `scripts/recreate_data_artifacts.py`: recreate deterministic split and index files.
- `checkpoints/`: currently empty; original trained weights were not embedded as downloadable files in the uploaded notebook.

## Environment
The notebook output recorded:
- PyTorch 2.14.1+cu130
- torchvision 0.29.1+cu130
- NumPy 2.4.6
- scikit-learn 1.6.1

Other package versions were not recorded in the visible notebook output, so they are not falsely pinned here.

## Reproduction

Use a GPU runtime comparable to Colab Tesla T4. From the repository root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python scripts/recreate_data_artifacts.py
python ML_Assignment3.py
```

For a notebook-based run, open `ML_Assignment3.ipynb` in Colab, select a GPU runtime, and run all cells from top to bottom.

**Important:** The full experiment is compute-intensive. The notebook estimates approximately 1–1.5 hours on a Colab T4. Do not set `SMOKE_TEST=1` for final results.

## Important reproducibility note
The notebook contains saved experiment outputs and final reported numbers. The split/index artifacts can be recreated deterministically from CIFAR10 and seed 26008 using the script above. Model checkpoints and original standalone figure/table files were not included as separate downloadable artifacts in the uploaded notebook; rerunning the full notebook is needed to regenerate them. Verify the recreated artifacts and rerun logs before treating this repository as the final reproducibility package.

## Final commit
Add the GitHub repository URL and final commit hash to page 1 of the report after pushing the completed repository.# ML_Assignment_3
