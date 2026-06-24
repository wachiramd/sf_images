from pathlib import Path
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText

PROMPT = Path("prompt.txt").read_text(encoding="utf-8").strip()

IMAGES = [
    {"id": "img01", "image": "images/img01.png"},
    {"id": "img02", "image": "images/img02.png"},
    {"id": "img03", "image": "images/img03.png"},
    {"id": "img04", "image": "images/img04.png"},
    {"id": "img05", "image": "images/img05.png"},
    {"id": "img06", "image": "images/img06.png"}
]
def build_dataset() -> MemoryDataset:
    samples = []
    for e in IMAGES:
        samples.append(
            Sample(
                id=e["id"],
                input=[
                    ChatMessageUser(content=[
                        ContentImage(image=e["image"]),
                        ContentText(text=PROMPT),
                    ])
                ],
                metadata={"image": e["image"], "prompt": PROMPT},
            )
        )
    return MemoryDataset(samples)