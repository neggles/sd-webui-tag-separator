from enum import Enum
import logging
import re

import gradio as gr
from modules import scripts
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingTxt2Img,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

extn_name = "Tag Separator"
elem_pfx = "fkcommas"

re_spaces = re.compile(r" {2,}", re.I + re.M)
re_whitespace = re.compile(r"[\t\n\r\f\v]+", re.I + re.M)


class SepCharacter(str, Enum):
    Space = " "
    Dash = "-"
    Underscore = "_"
    Comma = ","

    def names(self):
        return [x.name for x in self.__class__]

    def values(self):
        return [x.value for x in self.__class__]


class TagSeparator(scripts.Script):
    is_txt2img: bool = False

    def title(self):
        return extn_name

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        with gr.Accordion(label=extn_name, open=False):
            with gr.Row(elem_id=f"{elem_pfx}_row"):
                enabled = gr.Checkbox(
                    label="Enabled",
                    value=True,
                    description="Enable prompt processing",
                    elem_id=f"{elem_pfx}_enabled",
                )
                neg_enabled = gr.Checkbox(
                    label="Negative",
                    value=True,
                    description="Process negative prompt as well",
                    elem_id=f"{elem_pfx}_neg_enabled",
                )
                tag_sep = gr.Dropdown(
                    choices=[x.name for x in SepCharacter],
                    label=extn_name,
                    value=SepCharacter.Space.name,
                    elem_id=f"{elem_pfx}_tag_sep",
                )
                word_sep = gr.Dropdown(
                    label="Word Separator",
                    choices=[x.name for x in SepCharacter],
                    value=SepCharacter.Dash.name,
                    elem_id=f"{elem_pfx}_word_sep",
                )

        return [enabled, tag_sep, word_sep, neg_enabled]

    def process(
        self, p: StableDiffusionProcessing, enabled: bool, tag_sep: str, word_sep: str, neg_enabled: bool
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
            # build tag list
            prompt_tags = []

            # check for BREAK meta-tag
            if "BREAK" in prompt:
                # split prompt into blocks separated by BREAKs
                prompt_blocks = [x.strip() for x in prompt.split("BREAK")]
                for block in prompt_blocks:
                    # split each block into tags separated by commas and add to the list
                    prompt_tags.extend([x.strip() for x in block.split(",")])
                    # add a BREAK tag after each block
                    prompt_tags.append("BREAK")
            else:
                # only have one block, split it into tags separated by commas
                prompt_tags.extend([x.strip() for x in prompt.split(",")])

            # replace spaces with the word separator in each tag
            prompt_tags = [x.replace(" ", word_sep_char) for x in prompt_tags]
            # join tags with the tag separator
            prompt = tag_sep_char.join(prompt_tags)
            # strip multiple sequential spaces again just in case
            prompt = re_spaces.sub(" ", prompt).strip()
            # return the rewritten prompt
            return prompt

        # check if we're doing t2i with HR
        is_t2i = isinstance(p, StableDiffusionProcessingTxt2Img)
        hr_enabled = p.enable_hr if is_t2i else False

        if word_sep_char == tag_sep_char:
            logger.warning("Using the same character for word and tag separators is not recommended!")

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

        logger.info(f"{extn_name} processing done.")
