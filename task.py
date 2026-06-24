from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from dataset import build_dataset
from scorer import image_presence

@task
def image_eval():
    return Task(
        dataset=build_dataset(),
        solver=generate(),
        scorer=image_presence(),
        config=GenerateConfig(modalities=["image"]),
    )

