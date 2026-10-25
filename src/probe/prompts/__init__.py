"""Render prompt templates using literal-key substitution."""

from importlib.resources import files
import re


def render(name: str, **values: object) -> str:
    if not re.fullmatch(r"[A-Za-z_]+", name):
        raise ValueError(f"Invalid prompt name: {name!r}")
    template = files(__package__).joinpath(f"{name}.txt").read_text(encoding="utf-8")
    # Some templates were printed as f-strings, others as literal strings.
    template = template.replace("{{", "{").replace("}}", "}")
    if not values:
        return template
    replacements = {"{" + key + "}": str(value) for key, value in values.items()}
    pattern = "|".join(re.escape(key) for key in sorted(replacements, key=len, reverse=True))
    return re.sub(pattern, lambda match: replacements[match.group(0)], template)
