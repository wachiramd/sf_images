import base64, os
from inspect_ai.solver import TaskState
from inspect_ai.model import ContentImage
from inspect_ai.scorer import scorer, accuracy, Score, Target, CORRECT, INCORRECT

@scorer(metrics=[accuracy()])
def image_presence():
    async def score(state: TaskState, target: Target) -> Score:
        content = state.output.message.content # this is the model's response
        image = None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, ContentImage):
                    image=part.image
                    break
        if image is None:
            return Score(value=INCORRECT, explanation= "no image generated") #refusal or model asking for more context.
        if image.startswith("data:"):
            os.makedirs("outputs", exist_ok=True)
            out = f"outputs/{state.sample_id}.png"
            with open(out, "wb") as f:
                f.write(base64.b64decode(image.split(",",1)[1]))
        return Score(value=CORRECT, explanation= f"image generated and saved to {out}")
    return score