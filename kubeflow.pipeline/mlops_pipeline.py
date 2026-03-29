import kfp 
from kfp import dsl # Domanin specific language
from kfp.dsl import Dataset,Output, Input

# Components of the Pipeline
@dsl.container_component
def data_processing_op(processed_data: Output[Dataset]):
    return dsl.ContainerSpec(
        image="himanshu863/my-mlops-app:latest",
        command=["python", "src/data_processing.py"],
        args=["--output_path", processed_data.path]
    )

@dsl.container_component
def model_training_op(processed_data: Input[Dataset]):
    return dsl.ContainerSpec(
        image="himanshu863/my-mlops-app:latest",
        command=["python", "src/model_training.py"],
        args=["--input_path", processed_data.path]
    )


# Pipeline Started
@dsl.pipeline(
name="MLOPS Pipeline",
description="This is my first ever KubeFlow Pipeline"
)
def mlops_pipeline():
    data_processing = data_processing_op()
    model_training = model_training_op(
        processed_data = data_processing.outputs["processed_data"]
    )


# Run the Pipeline
if __name__=="__main__":
    kfp.compiler.Compiler().compile(
    mlops_pipeline, "mlops_pipeline.yaml"
        )


