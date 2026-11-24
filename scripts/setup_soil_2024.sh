#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required. Initialize it in this shell and rerun setup." >&2
    exit 1
fi

conda env create --file environment.yml

setup_dir="$(mktemp -d "${TMPDIR:-/tmp}/soil-2024-setup.XXXXXXXX")"
trap 'rm -rf -- "$setup_dir"' EXIT
cp -R -- pyproject.toml akela_shared "$setup_dir/"
conda run -n soil_2024 python -m pip install --no-deps --no-build-isolation "$setup_dir"
conda run -n soil_2024 python -m pip check
conda run -n soil_2024 python -B -c 'import preprocessing_main, main_fitting'
(
    cd 'Deep learning'
    conda run -n soil_2024 python -B -c 'import akela_shared.csv_loader, akela_shared.splitting'
)
