from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model", default="paraformer-zh")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # FunASR emits version/progress text to stdout. Keep the worker protocol
    # machine-readable by redirecting that chatter away from the JSON result.
    with contextlib.redirect_stdout(io.StringIO()):
        from funasr import AutoModel

        model_kwargs = {
            "model": args.model,
            "device": args.device,
            "disable_update": True,
        }
        # A local checkpoint is intentionally offline: do not trigger implicit
        # ModelScope downloads for optional VAD/punctuation companions.
        if not os.path.isdir(args.model):
            model_kwargs["vad_model"] = os.environ.get("AUTODUB_PARAFORMER_VAD", "fsmn-vad")
            model_kwargs["punc_model"] = os.environ.get("AUTODUB_PARAFORMER_PUNC", "ct-punc")
        model = AutoModel(**model_kwargs)
        result = model.generate(input=args.audio, batch_size_s=300, sentence_timestamp=True)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
