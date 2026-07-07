from pathlib import Path
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText

PROMPT = Path("prompt.txt").read_text(encoding="utf-8").strip()

# path to the images directory to allow looping of all the images
IMAGES_DIR = Path("images")
IMAGES_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def build_dataset() -> MemoryDataset:
    if not IMAGES_DIR.exists():
        IMAGES_DIR.mkdir(parents=True)
        raise FileNotFoundError (
            f"images folder {IMAGES_DIR}, did not exist and was created."
            f"Add image files ({','.join(sorted(IMAGES_EXTS))}) and re-run."
        )
    samples = []
    for path in sorted (IMAGES_DIR.iterdir()):
        if path.suffix.lower() not in IMAGES_EXTS:
            continue
        image = path.as_posix()
        samples.append(
            Sample(
                id=path.stem,
                input=[
                    ChatMessageUser(content=[
                        ContentImage(image=image),
                        ContentText(text=PROMPT),
                    ])
                ],
                metadata={"image": image, "prompt":PROMPT},
            )
        )
    if not samples:
        raise ValueError(
            f"No imahe foiund in '{IMAGES_DIR}'."
            f"Add images ({', '.join(sorted(IMAGES_EXTS))}) and re-run"
        )


    return MemoryDataset(samples)