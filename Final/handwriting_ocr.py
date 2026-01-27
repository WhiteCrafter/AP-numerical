import argparse
import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image):
    # Basic cleanup to help OCR on handwriting.
    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray)
    arr = np.array(gray)
    threshold = arr.mean()
    bw = (arr > threshold).astype(np.uint8) * 255
    return Image.fromarray(bw)


def read_handwriting(image, langs, use_gpu):
    try:
        import easyocr
    except ImportError as exc:
        raise ImportError("easyocr is required. Install with: pip install easyocr") from exc

    reader = easyocr.Reader(langs, gpu=use_gpu)
    results = reader.readtext(np.array(image), detail=0, paragraph=True)
    text = " ".join([r.strip() for r in results if r.strip()])
    return text


def read_handwriting_from_path(image_path, lang="en", use_gpu=False, preprocess=True):
    image = Image.open(image_path)
    if preprocess:
        image = preprocess_image(image)
    langs = [s.strip() for s in lang.split(",") if s.strip()]
    return read_handwriting(image, langs, use_gpu)


def main():
    parser = argparse.ArgumentParser(description="Simple handwriting OCR.")
    parser.add_argument("image", help="Path to input image.")
    parser.add_argument("--lang", default="en", help="Language code (default: en).")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip preprocessing step.",
    )
    args = parser.parse_args()

    image = Image.open(args.image)
    if not args.no_preprocess:
        image = preprocess_image(image)

    langs = [s.strip() for s in args.lang.split(",") if s.strip()]
    text = read_handwriting(image, langs, args.gpu)
    print(text)


if __name__ == "__main__":
    main()
