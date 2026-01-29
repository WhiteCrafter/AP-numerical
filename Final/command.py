import argparse

import handwriting_ocr
import drone_swarm_prototipe_squared as swarm


# Tunable settings (edit here instead of CLI flags).
SWARM_SPACING = 2.0
SWARM_FRAMES = 80
SWARM_DELAY = 2.0
SWARM_INTERVAL = 50
SWARM_V_MAX = 8
SWARM_K_V = 15
SWARM_K_D = 4
SWARM_K_REP = 12
SWARM_R_SAFE = 0.8
SWARM_MASS = 0.5
SWARM_ARRIVE_TOL = 0.15
SWARM_MAX_EXTRA = 600
SWARM_HOLD = 0.0
SWARM_HIDE_TARGETS = False


def main():
    parser = argparse.ArgumentParser(description="OCR image and render drone swarm text.")
    parser.add_argument("image", help="Path to input image.")
    parser.add_argument("--lang", default="en", help="Language code (default: en).")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip OCR preprocessing step.",
    )
    args = parser.parse_args()

    text = handwriting_ocr.read_handwriting_from_path(
        args.image,
        lang=args.lang,
        use_gpu=args.gpu,
        preprocess=not args.no_preprocess,
    )
    text = text.strip()
    if not text:
        text = "?"
    print(text)
    swarm.run_texts(
        [text, "Modding"],
        spacing=SWARM_SPACING,
        frames=SWARM_FRAMES,
        delay=SWARM_DELAY,
        interval=SWARM_INTERVAL,
        v_max=SWARM_V_MAX,
        k_v=SWARM_K_V,
        k_d=SWARM_K_D,
        k_rep=SWARM_K_REP,
        r_safe=SWARM_R_SAFE,
        mass=SWARM_MASS,
        arrive_tol=SWARM_ARRIVE_TOL,
        max_extra=SWARM_MAX_EXTRA,
        hold_after=SWARM_HOLD,
        hide_targets=SWARM_HIDE_TARGETS,
    )


if __name__ == "__main__":
    main()
