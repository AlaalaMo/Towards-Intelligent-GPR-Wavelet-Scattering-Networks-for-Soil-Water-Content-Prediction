# Source this file after activating the job's Python environment.
akela_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
akela_install_dir="$(mktemp -d "${TMPDIR:-/tmp}/akela-shared.XXXXXXXX")" || return 1
trap 'rm -rf -- "$akela_install_dir"' EXIT

# Stage the sources so concurrent jobs build and install in separate directories.
mkdir -- "$akela_install_dir/source" || return 1
cp -R -- "$akela_repo_root/pyproject.toml" "$akela_repo_root/akela_shared" \
    "$akela_install_dir/source/" || return 1

# Fitting jobs may start after the preprocessing or submission shell exits.
if [[ ${1:-} == --persistent ]]; then
    python -m pip install --no-deps --no-build-isolation --force-reinstall \
        "$akela_install_dir/source"
    return $?
fi

python -m pip install --no-deps --no-build-isolation \
    --target "$akela_install_dir/site-packages" "$akela_install_dir/source" || return 1
export PYTHONPATH="$akela_install_dir/site-packages${PYTHONPATH:+:$PYTHONPATH}"
