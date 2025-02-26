## Age and Gender Classification

### Installation

Install `poetry`; using `pipx` on macOS as an example:
    
    brew install pipx
    pipx ensurepath

    pipx install poetry

Refresh your source file, or open a new shell, and confirm installation.

    source ~/.zshrc
    which poetry

Set some configurations, create a virtual environment, and activate it. 

    poetry config virtualenvs.create true
    poetry config virtualenvs.in-project true

    poetry install
    source .venv/bin/activate

To install new packages use `poetry add <package_name>`



### Start the server

    python <TODO>
