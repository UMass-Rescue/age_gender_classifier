## Age and Gender Classification

### Installation

Ensure pipenv is installed.

    pip install pipenv

Activate the virtual environment.

    pipenv shell

Install the dependencies.

    pipenv install

### Start the server

    python <TODO>



Onnx models setup:

In order to run survey_models.py you must download the onnx model files from this GoogleDrive link

https://drive.google.com/drive/folders/1IgG6w6lJ9cd8Qlckd7HwdBUjWCd_-gxN

v001_model.onnx goes in age_classify_v0001 directory
vit_model.onnx goes in vit_age_classifier directory

You can also run the convert_to_onnx.py files in each directory respectively instead, to regenerate the ONNX files.

Your directory structure should then look like

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