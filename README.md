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

**Get the ONNX models**

In order to run survey_models.py you must download the onnx model files from this GoogleDrive [link](https://drive.google.com/drive/folders/1IgG6w6lJ9cd8Qlckd7HwdBUjWCd_-gxN), then copy them in their respective directories.

    cp ~/Downloads/v001_model.onnx src/onnx_models/age_classify_v001/v001_model.onnx 
    cp ~/Downloads/vit_model.onnx src/onnx_models/vit_age_classifier/vit_model.onnx
    cp ~/Downloads/vit_model.onnx src/onnx_models/fareface/fareface_age.onnx

Alternatively, you could run the `convert_to_onnx.py` files in each directory to regenerate the respective ONNX files.

Your models directory structure should then look like this:
```
/onnx_models
--/age_classify_v001
----convert_to_onnx.py
----model.py
----v001_model.onnx
--/vit_age_classifier
----convert_to_onnx.py
----model.py
----vit_model.onnx
--surey_models.py
```


You are good to go!

---

### Start the server
    python <TODO>
    TODO 
    bash entrypoint.sh



