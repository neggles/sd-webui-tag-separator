import logging
import re
from enum import Enum
from traceback import format_exc
from sys import stderr

import gradio as gr
from modules import scripts, script_callbacks
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img

logger = logging.getLogger("tag_sep")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

extn_name = "Tag Separator"
extn_id = "tag_sep"

try:
    from scripts import xyz_grid_tag_separator
except ImportError as e:
    logger.error(f"Failed to import Tag Separator XYZ grid support module: {e}")
    xyz_grid_tag_separator = None


re_spaces = re.compile(r" {2,}", re.I + re.M)
re_whitespace = re.compile(r"[\t\n\r\f\v]+", re.I + re.M)
re_all_caps = re.compile(r"(\b[\.\-_\']*[A-Z]+[\.\-_\']*[A-Z]*[\.\-_\']*\b)")
re_lora = re.compile(r"((<.*?>))", re.I + re.M)


class SepCharacter(str, Enum):
    Backslash = "\\"
    Comma = ","
    Dash = "-"
    Equals = "="
    Plus = "+"
    Space = " "
    Underscore = "_"
    Empty = ""  # not recommended

    @classmethod
    def names(cls):
        return [x.name for x in cls]

    @classmethod
    def values(cls):
        return [x.value for x in cls]


class TagSeparator(scripts.Script):
    is_txt2img: bool = False

    sep_names = SepCharacter.names()
    sep_values = SepCharacter.values()

    def title(self):
        return extn_name

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        with gr.Accordion(label=extn_name, open=False):
            with gr.Row(elem_id=f"{extn_id}_row"):
                enabled = gr.Checkbox(
                    label="Enabled",
                    value=True,
                    description="Enable prompt processing",
                    elem_id=f"{extn_id}_enabled",
                )
                neg_enabled = gr.Checkbox(
                    label="Negative",
                    value=True,
                    description="Process negative prompt",
                    elem_id=f"{extn_id}_neg_enabled",
                )
                ignore_meta = gr.Checkbox(
                    label="Ignore Meta Tags",
                    value=True,
                    description="Ignore meta tags in allcaps (BREAK, AND, etc.)",
                    elem_id=f"{extn_id}_ignore_meta",
                )
                tag_sep = gr.Dropdown(
                    label=extn_name,
                    value=SepCharacter.Space.name,
                    choices=[x.name for x in SepCharacter],
                    elem_id=f"{extn_id}_tag_sep",
                )
                word_sep = gr.Dropdown(
                    label="Word Separator",
                    value=SepCharacter.Dash.name,
                    choices=[x.name for x in SepCharacter],
                    elem_id=f"{extn_id}_word_sep",
                )

        return [enabled, neg_enabled, ignore_meta, tag_sep, word_sep]

    def process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        neg_enabled: bool,
        ignore_meta: bool,
        tag_sep: str,
        word_sep: str,
    ):
        if enabled is not True:
            return
        # enum back to string
        tag_sep_char = SepCharacter[tag_sep].value
        word_sep_char = SepCharacter[word_sep].value

        def rewrite_prompt(prompt: str):
            # turn newlines, tabs etc. into spaces
            prompt = re_whitespace.sub(" ", prompt)
            # strip multiple sequential spaces
            prompt = re_spaces.sub(" ", prompt)
            # storage for tag lists
            prompt_tags = []

            # split the prompt on commas, as well as LoRA blocks
            prompt_blocks = [x.strip() for x in re_lora.sub(r",\1,", prompt).split(",") if len(x.strip()) > 0]
            if ignore_meta:
                # if ignoring meta tags, we need to split on them (ALLCAPS words) as well
                for block in prompt_blocks:
                    if block.startswith("<") and block.endswith(">"):
                        # this is a LoRA block, leave it as is
                        prompt_tags.append(block)
                    else:
                        # wrap meta tags in commas, then split by comma again
                        prompt_tags.extend(
                            [
                                x.strip()
                                for x in re_all_caps.sub(r",\1,", block).split(",")
                                if len(x.strip()) > 0
                            ]
                        )
            else:
                # not ignoring meta tags, so just use the blocks as-is
                prompt_tags = prompt_blocks

            # replace spaces with the word separator in each tag
            processed_tags = []
            for tag in prompt_tags:
                tag = tag.strip()
                if tag.startswith("<") and tag.endswith(">"):
                    # if LoRA block, ignore
                    processed_tags.append(tag)
                else:
                    processed_tags.append(tag.replace(" ", word_sep_char))

            # join tags with the tag separator
            prompt = tag_sep_char.join(processed_tags)
            # strip multiple sequential spaces again just in case
            prompt = re_spaces.sub(" ", prompt).strip()
            # return the rewritten prompt
            return prompt

        # check if we're doing t2i with HR
        is_t2i = isinstance(p, StableDiffusionProcessingTxt2Img)
        hr_enabled = p.enable_hr if is_t2i else False

        logger.info(f"{extn_name} processing... tag sep: '{tag_sep_char}', word sep: '{word_sep_char}'")
        if word_sep_char == tag_sep_char:
            logger.warning("Using the same character for word and tag separators is not recommended!")

        orig_pos_prompt = p.all_prompts[0]
        orig_neg_prompt = p.all_negative_prompts[0]

        batch_size = p.batch_size
        for b_idx in range(p.n_iter):
            for s_offs in range(batch_size):
                s_idx = b_idx * batch_size + s_offs  # offset of the prompt in all_prompts

                s_prompt = rewrite_prompt(p.all_prompts[s_idx])
                p.all_prompts[s_idx] = s_prompt
                logger.debug(f"[B{b_idx:02d}][I{s_offs:02d}] prompt: {s_prompt}")

                if neg_enabled:
                    s_neg_prompt = rewrite_prompt(p.all_negative_prompts[s_idx])
                    p.all_negative_prompts[s_idx] = s_neg_prompt
                    logger.debug(f"[B{b_idx:02d}][I{s_offs:02d}] neg prompt: {s_neg_prompt}")

                if is_t2i and hr_enabled:
                    s_hr_prompt = rewrite_prompt(p.all_hr_prompts[s_idx])
                    p.all_hr_prompts[s_idx] = s_hr_prompt
                    if s_hr_prompt != s_prompt:
                        logger.debug(f"[B{b_idx:02d}][I{s_offs:02d}] HR prompt: {s_hr_prompt}")

                    if neg_enabled:
                        s_hr_neg_prompt = rewrite_prompt(p.all_hr_negative_prompts[s_idx])
                        p.all_hr_negative_prompts[s_idx] = s_hr_neg_prompt
                        if s_hr_neg_prompt != s_neg_prompt:
                            logger.debug(f"[B{b_idx:02d}][I{s_offs:02d}] HR neg prompt: {s_hr_neg_prompt}")

        # save original prompt (only for image 0)
        p.extra_generation_params["TagSep Prompt"] = orig_pos_prompt
        if neg_enabled:
            p.extra_generation_params["TagSep Negative"] = orig_neg_prompt

        logger.info(f"{extn_name} processing done.")


def before_ui_callback():
    try:
        xyz_grid_tag_separator.initialize(TagSeparator)
    except Exception:
        logger.exception(f"Failed to initialize {extn_name} XYZ extension")


script_callbacks.on_before_ui(before_ui_callback)

# TODO: Implement infotext_pasted_cb to restore the original prompt when loading image meta
# script_callbacks.on_infotext_pasted(infotext_pasted_cb)
