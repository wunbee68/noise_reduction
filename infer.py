import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from denoise.audio import enhance_audio, read_wav, wav_files, write_wav
from denoise.models import DEFAULT_CONFIG, load_enhancer


def run(input_dir, output_dir, config, threads):
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if input_dir == output_dir or input_dir.is_relative_to(output_dir) or output_dir.is_relative_to(input_dir):
        raise ValueError("Input and output directories must be separate.")
    inputs = wav_files(input_dir)
    if not inputs:
        raise FileNotFoundError(f"No WAV files found under {input_dir}")
    if len({source.name.casefold() for source in inputs}) != len(inputs):
        raise ValueError("Input WAV filenames must be unique, including in subdirectories.")
    expected = {output_dir / source.name for source in inputs}
    extra = set(wav_files(output_dir)) - expected
    if extra:
        raise ValueError(f"Output already contains unrelated WAVs. Choose an empty directory: {min(extra)}")
    if any(path.is_symlink() or path.is_dir() for path in expected):
        raise ValueError("An expected output path points to a directory or symlink.")

    enhance = load_enhancer(config, threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    # A failed inference leaves the existing output set untouched.
    with TemporaryDirectory(dir=output_dir) as temp:
        temporary = Path(temp)
        for index, source in enumerate(inputs, 1):
            audio, rate = read_wav(source)
            try:
                result = enhance_audio(audio, rate, enhance)
                write_wav(temporary / source.name, result, rate)
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError(f"Failed to process {source}: {exc}") from exc
            if index % 50 == 0 or index == len(inputs):
                print(f"{index}/{len(inputs)} WAVs processed", flush=True)
        for source in inputs:
            (temporary / source.name).replace(output_dir / source.name)
    if set(wav_files(output_dir)) != expected:
        raise RuntimeError("Output WAV filenames do not match the input WAV filenames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise every WAV and keep its filename and format.")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run(args.input_dir, args.output_dir, args.config, args.threads)
