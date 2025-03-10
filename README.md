## Age and Gender Classification

### Installation and Setup

**Install `pipx`:** To manage `poetry`, this is recommended as it isolates Poetry in its own virtual environment, preventing conflicts with system-wide Python packages.
    
    # macOS
    brew update
    brew install pipx
    pipx ensurepath

    # linux (debian)
    sudo apt update
    sudo apt install pipx
    export PATH="$HOME/.local/bin:$PATH"

    # alt, not recommended
    python -m pip install --user pipx
    python -m pipx ensurepath

**Install `poetry`:** Restart or refresh your shell, install Poetry, and set desired Python version (>=3.11).

    source ~/.bashrc

    pipx install poetry
    poetry env use 3.11.1

**Activate venv:** Set configurations, create a virtual environment, and activate it. Note: use `poetry init` when starting a new project from scratch, use `poetry install` to set up dependencies from an existing lock file.

    poetry config virtualenvs.create true
    poetry config virtualenvs.in-project true

    poetry install
    source .venv/bin/activate

To install or remove packages, respectively, use `poetry add` and `poetry remove`.

    poetry add [--dev] <package_name>
    poetry remove [--dev] <package_name>

**Set environment variables:** Set the `PYTHONPATH` environment variable to make local directories accessible for import in your venv, and define any other vars in a `.env` file; follow `.env.sample`

    export PYTHONPATH=$(pwd):$PYTHONPATH
    set -a; source .env; set +a

You are good to go!

---

### Start the server

    TODO 
    bash entrypoint.sh
