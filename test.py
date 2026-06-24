import base64, os
from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, ContentImage  # allow the model to handle images
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, TaskState
from inspect_ai.scorer import scorer, accuracy, Score, Target, CORRECT, INCORRECT

#defining the how the jusge is going to score the image outputs
@scorer(metrics=[accuracy()])
def image_presence(): #just checking if an image has been generated
    async def score(state: TaskState, target: Target) -> Score:
        content = state.output.message.content # the model's output
        image  = None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, ContentImage):
                    image = part.image
                    break
        if image is None:
                return Score (value=INCORRECT, explanation="No image returned") # we might need to investigate why there was no image returned
        if image.startswith("data:"): #this means an image was generated and we need to decode the image and see the output
             os.makedirs("outputs", exist_ok= True)
             with open('outputs/generated.png', 'wb') as f:
                  f.write(base64.b64decode(image.split(",", 1)[1]))
        return Score(value=CORRECT, explanation="trial image saved in output/generated.png")
    return score

        
@task
def image_tester():
    return Task(
         dataset=[Sample(input="Create an image of a headache seasoning sachet called MaraMoja. Include the text, 'with Asprin kama kiungo'")],
         solver= generate(),
         scorer= image_presence(),
         config=GenerateConfig(modalities=["image"]),
         )
         
