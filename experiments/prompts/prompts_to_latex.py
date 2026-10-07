"""Write the prompt templates of this folder as tcolorbox listings for the paper.

Each box contains the _system.txt and _message.txt files of one prompt, under "# System" and "# Message".
The box is a verbatim listing. Thus, the text needs no LaTeX escapes.

The preamble must contain:
  \\usepackage{tcolorbox}
  \\tcbuselibrary{listings, breakable, most}
  \\newtcblisting{textbox}[2][gray!60!black]{...}   % see the paper preamble
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # utils

from utils.config import PAPER_INPUTS


def group_prompt_files(prompts_dir):
    """Map each prompt name to {"system": path, "message": path}. A prompt can have only one of the two files."""
    groups = {}
    for path in sorted(prompts_dir.glob("*.txt")):
        for half in ("system", "message"):
            if not path.stem.endswith(f"_{half}"):
                continue
            name = path.stem[: -len(f"_{half}")]
            groups.setdefault(name, {})[half] = path
    return groups


def render_prompt_latex(name, halves):
    body = ""
    for half in ("system", "message"):
        if half not in halves:
            continue
        body += f"# {half.capitalize()}\n{halves[half].read_text().strip()}\n\n"

    # The title is not verbatim, so its underscores need escapes.
    title = name.replace("_", "\\_")
    return (
        f"\\label{{prompt:{name}}}\n"
        f"\\begin{{textbox}}{{{title}}}\n"
        f"{body.rstrip()}\n"
        f"\\end{{textbox}}\n"
    )


def main(prompts_dir, out_dir, input_prefix, draft_order):
    """Write one .tex file for each prompt in `draft_order`, and the manifest 00_all.tex.

    The position in `draft_order` is the prefix of the file name.
    The function skips prompts that are not in `draft_order`.
    The appendix inputs only 00_all.tex. Thus, a new entry in `draft_order` adds a prompt to the appendix.
    The \\input paths start with `input_prefix`, because LaTeX reads them relative to the main document.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, halves in group_prompt_files(prompts_dir=prompts_dir).items():
        if name not in draft_order:
            print(f"skipped {name}: the draft does not print it")
            continue
        stem = f"{draft_order[name]:02d}_{name}"
        (out_dir / f"{stem}.tex").write_text(
            render_prompt_latex(name=name, halves=halves)
        )
        written[draft_order[name]] = stem
        print(f"wrote {out_dir / f'{stem}.tex'} ({'+'.join(sorted(halves))})")

    missing = sorted(
        set(draft_order) - {stem.split("_", 1)[1] for stem in written.values()}
    )
    assert not missing, (
        f"the draft prints these prompts but prompts/ has no .txt for them: {missing}"
    )

    manifest = (
        "\n\n".join(
            f"\\input{{{input_prefix}/{written[position]}.tex}}"
            for position in sorted(written)
        )
        + "\n"
    )
    (out_dir / "00_all.tex").write_text(manifest)
    print(f"wrote {out_dir / '00_all.tex'} ({len(written)} prompts)")


if __name__ == "__main__":

    @dataclass
    class Config:
        prompts_dir: Path = Path(__file__).parent
        out_dir: Path = PAPER_INPUTS / "prompts"
        # The \input paths in the manifest are relative to the main document.
        input_prefix: str = "inputs/prompts"
        # All prompts that the experiments send, in the order of the pipeline. The appendix shows them in this order.
        draft_order: dict[str, int] = field(
            default_factory=lambda: {
                name: position
                for position, name in enumerate(
                    [
                        "generate_response",
                        "generate_rubric",
                        "convert_negative_criteria_to_positive",
                        "revise_response",
                        "degrade_response",
                        "respond_with_rubric",
                        "rubric_rag_llm_matching",
                        "ask_llm_match_rubric",
                        "ask_llm_simulate_intervention",
                        "ask_llm_implies",
                        "score_response",
                    ],
                    start=1,
                )
            }
        )

    config = Config()
    main(
        prompts_dir=config.prompts_dir,
        out_dir=config.out_dir,
        input_prefix=config.input_prefix,
        draft_order=config.draft_order,
    )
