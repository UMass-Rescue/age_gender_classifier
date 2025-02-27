## Age and Gender Classification

### Installation

Install `pipx` to manage `poetry`; this is recommended as it isolates Poetry in its own virtual environment, preventing conflicts with system-wide Python packages.
    
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

Restart or refresh your shell, install Poetry, and confirm.

    source ~/.bashrc
    which pipx

    pipx install poetry
    which poetry

Set configurations, create a virtual environment, and activate it. Note: use `poetry init` when starting a new project from scratch, use `poetry install` to set up dependencies from an existing lock file.

    poetry config virtualenvs.create true
    poetry config virtualenvs.in-project true

    poetry install
    source .venv/bin/activate

To install or remove packages, respectively, use `poetry add` and `poetry remove`.

    poetry add [--dev] <package_name>
    poetry remove [--dev] <package_name>

Set the `PYTHONPATH` environment variable to make local directories accessible for import in your venv. And define environment variables in a `.env` file; follow `.env.sample`

    export PYTHONPATH=$(pwd):$PYTHONPATH

You are good to go!

---

### Start the server

    bash entrypoint.sh
