from inspect_ai import eval
from task import image_eval
from models import TARGET_MODELS

if __name__ == "__main__":
    for model in TARGET_MODELS:
        eval(image_eval(), model=model) #we need to log the runs per models