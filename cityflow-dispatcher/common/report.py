# Writes results/<step>/summary.md: command, inputs, outputs, key numbers, flags.
import os
import sys


def table_to_text(table):
    """Markdown-ish table without needing the tabulate package."""
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    divider = "|" + "---|" * len(table.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, divider] + rows)


def write_summary(out_dir, title, inputs, outputs, table=None, flags=None, extra_lines=None):
    lines = [f"# {title}", "", "Command: `" + " ".join(sys.argv) + "`", ""]
    lines += ["Inputs:"] + [f"- {x}" for x in inputs] + [""]
    lines += ["Outputs:"] + [f"- {x}" for x in outputs] + [""]
    if extra_lines:
        lines += extra_lines + [""]
    if table is not None:
        lines += ["Key numbers:", "", table_to_text(table), ""]
    lines += ["Flags:"] + ([f"- {x}" for x in flags] if flags else ["- none"]) + [""]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
