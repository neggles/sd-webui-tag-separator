import logging
from enum import Enum
from typing import TYPE_CHECKING
from pathlib import Path

from modules import scripts
from modules.processing import StableDiffusionProcessing

if TYPE_CHECKING:
    import xyz_grid
    from xyz_grid import AxisOption

xyz_grid_mod: "xyz_grid" = None  # XYZ Grid module
tagsep_class: scripts.Script = None  # TagSeparator scripts.Script class

logger = logging.getLogger("tag_sep")
logger.setLevel(logging.INFO)


class TagSepArgs(int, Enum):
    enabled = 0
    neg_enabled = 1
    ignore_meta = 2
    tag_sep = 3
    word_sep = 4


def edit_tuple(x: tuple, idx: int, value):
    as_list = list(x)
    as_list[idx] = value
    return tuple(as_list)


def update_script_args(p: StableDiffusionProcessing, value, arg_idx: int):
    global tagsep_class
    s: scripts.Script
    for s in scripts.scripts_txt2img.alwayson_scripts:
        if isinstance(s, tagsep_class):
            arg_num = s.args_from + arg_idx
            prev_val = p.script_args[arg_num]
            p.script_args = edit_tuple(p.script_args, arg_num, value)
            logger.debug(f"Updated {s.title()} arg #{arg_num} from {prev_val} to {value}")
            break


def apply_tag_sep(field: TagSepArgs, is_bool: bool = False):
    def apply_fn(p: StableDiffusionProcessing, x, xs: int):
        if is_bool:
            x = True if x.lower() == "true" else False
        update_script_args(p, x, field.value)

    return apply_fn


def format_value_tag_sep(p, opt: "AxisOption", x):
    opt_label = opt.label.split("]")[1].strip()
    if opt_label == "Enabled":
        opt_label = "TagSep Enabled"
    elif opt_label == "Negative":
        opt_label = "TagSep Negative"

    return f"{opt_label}: {x}"


def initialize(script_class: scripts.Script):
    global xyz_grid_mod, tagsep_class

    tagsep_class = script_class
    script_tuple: scripts.ScriptClassData
    for script_tuple in scripts.scripts_data:
        if Path(script_tuple.path).name.lower() in ["xy_grid.py", "xyz_grid.py"]:
            # we found the XYZ Grid module
            xyz_grid_mod = script_tuple.module
            # create option list
            tag_sep_enabled = xyz_grid_mod.AxisOption(
                label="[TagSep] Enabled",
                type=str,
                apply=apply_tag_sep(TagSepArgs.enabled, is_bool=True),
                choices=xyz_grid_mod.boolean_choice(reverse=True),
                format_value=format_value_tag_sep,
            )
            tag_sep_negative = xyz_grid_mod.AxisOption(
                label="[TagSep] Negative",
                type=str,
                apply=apply_tag_sep(TagSepArgs.neg_enabled, is_bool=True),
                choices=xyz_grid_mod.boolean_choice(),
                format_value=format_value_tag_sep,
            )
            tag_sep_ignore_meta = xyz_grid_mod.AxisOption(
                label="[TagSep] Ignore Meta",
                type=str,
                apply=apply_tag_sep(TagSepArgs.ignore_meta, is_bool=True),
                choices=xyz_grid_mod.boolean_choice(),
                format_value=format_value_tag_sep,
            )
            tag_sep_tag_char = xyz_grid_mod.AxisOption(
                label="[TagSep] Tag Separator",
                type=str,
                apply=apply_tag_sep(TagSepArgs.tag_sep),
                choices=lambda: tagsep_class.sep_names,
                format_value=format_value_tag_sep,
            )
            tag_sep_word_char = xyz_grid_mod.AxisOption(
                label="[TagSep] Word Separator",
                type=str,
                apply=apply_tag_sep(TagSepArgs.word_sep),
                choices=lambda: tagsep_class.sep_names,
                format_value=format_value_tag_sep,
            )
            # Add options to XYZ Grid module
            xyz_grid_mod.axis_options.extend(
                [tag_sep_enabled, tag_sep_negative, tag_sep_ignore_meta, tag_sep_tag_char, tag_sep_word_char]
            )
            # No need to continue searching
            break
