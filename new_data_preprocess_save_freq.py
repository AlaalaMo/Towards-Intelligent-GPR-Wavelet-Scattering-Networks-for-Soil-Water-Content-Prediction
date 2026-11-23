"""Compatibility launcher for the frequency preprocessing pipeline."""


from preprocessing_main import main as run_preprocessing
from preprocessing_main import process_configuration as main


if __name__ == "__main__":
    run_preprocessing()
