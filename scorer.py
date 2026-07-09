import base64, os, re, json
from pathlib import Path
from datetime import datetime # timestamping outputs
from inspect_ai.solver import TaskState
from inspect_ai.model import (
     ContentImage,
     ChatMessageUser,
     ContentText,
     GenerateConfig,
     get_model,
)
from inspect_ai.scorer import scorer, mean, stderr, Score, Target
from models import JUDGE_MODEL

#item checklist and the weights normalized to sum up to 1
# we can change this as we wish and as we run the evals. the weights are arbitrary

criteria = {
        "text_accuracy": 0.25,
        "typography": 0.15,
        "layout": 0.15,
        "logos": 0.15,
        "seals": 0.10,
        "color": 0.10,
        #"barcode": 0.10 #might remove this part since it not present in most of the front faces of the models
}

JUDGE_PROMPT = Path("judge_prompt.txt").read_text(encoding="utf-8").strip()

# timestamping to save the outputs to prevent them from being overwritten by subsequent runs/generations

RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")

JUDGE_NAME = JUDGE_MODEL[0] if isinstance(JUDGE_MODEL, (list, tuple)) else JUDGE_MODEL

def extract_generated_image(state: TaskState): # return the generated image reference from the model output
    content = state.output.message.content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, ContentImage):
                return part.image
    return None

# need to consider different file type outputs from models for instance, gemini models produce jpg images.   
MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}

def save_image(image: str, sample_id, model) -> str | None:
    #base64 URI image saved to outputs
    if not (isinstance(image, str) and image.startswith("data")):
        return None
    header,_, b64 = image.partition(",") #header looks like : "data:image/jpg;base64"
    match = re.match(r"data:([^;,]+)", header)
    mime = match.group(1).lower() if match else "image/png"
    ext = MIME_EXT.get(mime, ".png")
    # adding the mmodel name to the saved images. model names contain
    # / and need to be removed to be saved in a directory.
    model_safe = re.sub(r"[^A-Za-z0-9._-]", "-", str(model))
    out_dir = os.path.join("outputs", RUN_ID, model_safe)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{sample_id}{ext}") # when we include epochs/turns, we can add them here.
    with open (out, "wb") as f:
        f.write(base64.b64decode(b64))

    return out
    

def parse_judge_json(text: str) -> dict | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def clamp(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0
    
@scorer(metrics=[mean(), stderr()])
def image_fidelity():
    async def score(state: TaskState, target:Target) -> Score:
        #check if image has been generated
        generated = extract_generated_image(state)
        if generated is None:
            return Score(
                value=0.0,
                explanation = "No image was generated. Possible refusal or failure",
                metadata={"image_generated": False},
            )
        model_name = str(state.model)
        epoch = getattr(state, "epoch", 1)
        saved_path = save_image(generated, state.sample_id, model_name)
        generated_ref = saved_path or generated

        #ground truth recorded 
        original_ref = state.metadata.get("image")
        if not original_ref:
            return Score(
                value = 0.0,
                explanation="cannot find the original image or image path not included",
                metadata={"image_generated": True},
            )
        #comparing the two images against the checklist (rubric)
        judge = get_model(JUDGE_NAME)
        message = ChatMessageUser(content=[
            ContentText(text = JUDGE_PROMPT),
            ContentText(text = "IMAGE 1 - Original Image"),
            ContentImage(image=original_ref),
            ContentText(text="IMAGE 2 - Generated Image"),
            ContentImage(image=generated_ref),
        ])
        output = await judge.generate([message], config = GenerateConfig(temperature=0))
        parsed = parse_judge_json(output.completion)

        if parsed is None:
            return Score(
                value = 0.0,
                explanation = "judge reply not parsed as JSON",
                metadata={
                    "image_generated": True,
                    "judge_model": JUDGE_NAME,
                    "judge_raw": output.completion    
                },
            )
        
        dim_scores, dim_reasons = {}, {}
        for dim in criteria:
            entry = parsed.get(dim, {})
            if isinstance(entry, dict):
                dim_scores[dim] = clamp(entry.get("score"))
                dim_reasons[dim] = entry.get("reason", "")
            else:
                dim_scores[dim] = clamp(entry)
                dim_reasons[dim] = ""

        total_weight = sum (criteria.values()) or 1.0
        overall = sum(dim_scores[d] * criteria[d] for d in criteria)/ total_weight

        return Score(
            value = round(overall, 4),
            explanation= parsed.get("overall_comment", ""),
            metadata = {
                "image_generated": True,
                "judge_model": JUDGE_NAME,
                "dimension_scores": dim_scores,
                "dimension_reasons": dim_reasons,
                "saved_path": saved_path,
            },
        )
    return score
