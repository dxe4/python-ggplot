from pathlib import Path

current_path = Path(__file__).resolve()
project_dir = current_path.parent.parent
data_path = project_dir / "data"
plots_path = project_dir / "plots"
