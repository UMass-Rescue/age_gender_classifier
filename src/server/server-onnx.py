from typing import TypedDict
from pathlib import Path
import pandas as pd

from src.onnx_models.survey_models import SurveyModels

from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    DirectoryInput,
    FileResponse,
    InputSchema,
    InputType,
    ResponseBody,
    TaskSchema,
)


def create_transform_case_task_schema() -> TaskSchema:
    """
    Configure UI Elements in RescueBox Desktop
    InputType.DIRECTORY: a single directory path
    """
    input_schema = InputSchema(
        key="input_dataset",
        label="Path to the directory containing images or videos",
        input_type=InputType.DIRECTORY,
    )
    output_schema = InputSchema(
        key="output_file",
        label="Path to the output file",
        input_type=InputType.DIRECTORY,
    )
    return TaskSchema(inputs=[input_schema, output_schema], parameters=[])


class Inputs(TypedDict):
    """Specify the input and output types for the task"""
    input_dataset: DirectoryInput
    output_file: DirectoryInput


class Parameters(TypedDict):
    pass



models = SurveyModels()
server = MLServer(__name__)

server.add_app_metadata(
    name="Age Classifier",
    author="UMass Rescue",
    version="0.2.0",
    info=load_file_as_string("img-app-info.md"),
)


@server.route("/ageclassifier", task_schema_func=create_transform_case_task_schema)
def sentiment_detection(inputs: Inputs, parameters: Parameters) -> ResponseBody:
    """
    In Flask-ML, an inference function takes two arguments: inputs and parameters.
    In Flask-ML, the types of inputs and parameters must be Python TypedDict types.
    """
    input_path = Path(inputs["input_dataset"].path)
    out_path = Path(inputs["output_file"].path)
    files = [str(fpath) for fpath in input_path.iterdir() if fpath.is_file()]

    results = models.predict(files)

    # TODO: should this go to csv or into the DB?
    results_path = out_path / "....csv"
    # df = pd.json_normalize(results)
    # df.to_csv(res_path, index=False)

    # import pdb; pdb.set_trace()
    return ResponseBody(FileResponse(path=str(results_path), file_type="csv"))


if __name__ == "__main__":
    server.run(port=5000)
