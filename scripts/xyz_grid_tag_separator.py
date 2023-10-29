from enum import Enum
from typing import TYPE_CHECKING

from modules import scripts as scripts
from pathlib import Path
from modules.processing import StableDiffusionProcessing

if TYPE_CHECKING:
    import xyz_grid

xyz_grid_mod: "xyz_grid" = None  # XYZ Grid module
tagsep_class = None  # TagSeparator scripts.Script class


class TagSepArgs(int, Enum):
    enabled = 0
    neg_enabled = 1
    ignore_caps = 2
    tag_sep = 3
    word_sep = 4


def update_script_args(p: StableDiffusionProcessing, value, arg_idx: int):
    global tagsep_class
    s: scripts.Script
    for s in scripts.scripts_txt2img.alwayson_scripts:
        if isinstance(s, tagsep_class):
            args = list(p.script_args)
            args[s.args_from + arg_idx] = value
            p.script_args = tuple(args)
            break


def apply_tag_sep(field, is_bool: bool = False):
    def apply_fn(p, x, xs):
        if is_bool:
            x = True if x.lower() == "true" else False
        update_script_args(p, x, TagSepArgs[field].value)

    return apply_fn


def initialize(script):
    global xyz_grid_mod, tagsep_class

    tagsep_class = script
    script_tuple: scripts.ScriptClassData
    for script_tuple in scripts.scripts_data:
        if Path(script_tuple.path).name.lower() in ["xy_grid.py", "xyz_grid.py"]:
            xyz_grid_mod = script_tuple.module
            tag_sep_enabled = xyz_grid_mod.AxisOption(
                "TagSep Enabled",
                bool,
                apply_tag_sep(TagSepArgs.enabled.name, is_bool=True),
                format_value=xyz_grid_mod.boolean_choice,
            )
            tag_sep_negative = xyz_grid_mod.AxisOption(
                "TagSep Negative",
                bool,
                apply_tag_sep(TagSepArgs.neg_enabled.name, is_bool=True),
                format_value=xyz_grid_mod.boolean_choice,
            )
            tag_sep_ignore_caps = xyz_grid_mod.AxisOption(
                "TagSep Ignore Meta Tags",
                bool,
                apply_tag_sep(TagSepArgs.ignore_caps.name, is_bool=True),
                format_value=xyz_grid_mod.boolean_choice,
            )
            tag_sep_tag_char = xyz_grid_mod.AxisOption(
                "TagSep Tag Separator",
                str,
                apply_tag_sep(TagSepArgs.tag_sep.name),
                choices=lambda: [x.name for x in tagsep_class.SepCharacter],
            )
            tag_sep_word_char = xyz_grid_mod.AxisOption(
                "TagSep Word Separator",
                str,
                apply_tag_sep(TagSepArgs.word_sep.name),
                choices=lambda: [x.name for x in tagsep_class.SepCharacter],
            )

            xyz_grid_mod.axis_options.extend(
                [tag_sep_enabled, tag_sep_negative, tag_sep_ignore_caps, tag_sep_tag_char, tag_sep_word_char]
            )
