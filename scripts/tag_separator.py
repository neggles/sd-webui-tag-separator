import logging
import re
from enum import Enum

import gradio as gr
from modules import scripts
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

extn_name = "Tag Separator"
elem_pfx = "fkcommas"

re_spaces = re.compile(r" {2,}", re.I + re.M)
re_whitespace = re.compile(r"[\t\n\r\f\v]+", re.I + re.M)
re_all_caps = re.compile(r"(\b[\.\-_\']*[A-Z]+[\.\-_\']*[A-Z]*[\.\-_\']*\b)")
re_lora = re.compile(r"((<.*?>))", re.I + re.M)


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
                ignoreCaps = gr.Checkbox(
                    label="Ignore Meta Tags",
                    value=True,
                    description="Ignores tags in all caps, used for special keywords such as BREAK and AND",
                    elem_id=f"{elem_pfx}_ignoreCaps",
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

        return [enabled, tag_sep, word_sep, neg_enabled, ignoreCaps]

    def process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        tag_sep: str,
        word_sep: str,
        neg_enabled: bool,
        ignoreCaps: bool,
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
            processed_tags = []

            prompt_blocks = [x.strip() for x in re_lora.sub(r",\1,", prompt).split(",") if len(x.strip()) > 0]
            if ignoreCaps:
                for block in prompt_blocks:
                    if block.startswith("<") and block.endswith(">"):
                        # if LoRA block, ignore
                        prompt_tags.append(block)
                    else:
                        # otherwise, split by comma (wrapping ALLCAPS words in commas)
                        prompt_tags.extend(
                            [
                                x.strip()
                                for x in re_all_caps.sub(r",\1,", block).split(",")
                                if len(x.strip()) > 0
                            ]
                        )
            else:
                prompt_tags = prompt_blocks

            # replace spaces with the word separator in each tag
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
            p.extra_generation_params["TagSep Negative"] = orig_pos_prompt

        logger.info(f"{extn_name} processing done.")
