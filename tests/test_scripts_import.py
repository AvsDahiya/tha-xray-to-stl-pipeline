import importlib.util
import unittest
from pathlib import Path


SCRIPT_NAMES = [
    "01_curate_datasets.py",
    "02_create_pairs.py",
    "03_preprocess.py",
    "04_train_pix2pix.py",
    "05_train_segmenter.py",
    "06_run_inference.py",
    "07_segment_and_reconstruct.py",
    "08_evaluate.py",
    "09_ablation_studies.py",
    "10_run_5fold_cv.py",
    "11_compile_statistics_report.py",
    "12_run_full_experiment.py",
]


class ScriptImportTests(unittest.TestCase):
    def test_script_modules_import(self) -> None:
        scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
        for script_name in SCRIPT_NAMES:
            spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), scripts_dir / script_name)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)


if __name__ == "__main__":
    unittest.main()
