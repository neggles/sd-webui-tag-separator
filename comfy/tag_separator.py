import re
from enum import Enum
from typing import Optional

re_spaces = re.compile(r" {2,}", re.I + re.M)
re_whitespace = re.compile(r"[\t\n\r\f\v]+", re.I + re.M)
re_all_caps = re.compile(r"(\b[\.\-_\']*[A-Z]+[\.\-_\']*[A-Z]*[\.\-_\']*\b)")
re_lora = re.compile(r"((<.*?>))", re.I + re.M)


class SepCharacter(str, Enum):
    Backslash = "\\"
    Comma = ","
    CommaSpace = ", "
    Dash = "-"
    Equals = "="
    Plus = "+"
    Slash = "/"
    Space = " "
    Underscore = "_"
    At = "@"
    Hash = "#"
    Percent = "%"
    Ampersand = "&"
    Asterisk = "*"
    Caret = "^"
    Tilde = "~"
    Empty = ""  # not recommended
    BREAK = " BREAK "  # are you insane?
    Unmodified = "U"

    @classmethod
    def names(cls):
        return [x.name for x in cls]

    @classmethod
    def values(cls):
        return [x.value for x in cls]

    @classmethod
    @property
    def tag_only(cls):
        return [
            cls.BREAK.name,
        ]

    @classmethod
    @property
    def word_only(cls):
        return [
            cls.Empty.name,
        ]

    @classmethod
    @property
    def tag_separators(cls):
        return [x for x in cls.names() if x not in cls.word_only]

    @classmethod
    @property
    def word_separators(cls):
        return [x for x in cls.names() if x not in cls.tag_only]


class TagSeparator:
    """
    TagSeparator class for ComfyUI.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pos_prompt": (
                    "STRING",
                    {"multiline": True},
                ),
                "ignore_meta": (["enable", "disable"],),
                "tag_sep": (SepCharacter.tag_separators,),
                "word_sep": (SepCharacter.word_separators,),
            },
            "optional": {
                "neg_prompt": (
                    "STRING",
                    {"multiline": True},
                ),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "pos_out",
        "neg_out",
    )

    FUNCTION = "process_prompt"

    # OUTPUT_NODE = False

    CATEGORY = "conditioning"

    def process_prompt(
        self,
        pos_prompt: str,
        ignore_meta: str,
        tag_sep: str,
        word_sep: str,
        neg_prompt: Optional[str] = None,
    ):
        ignore_meta = True if ignore_meta == "enable" else False
        tag_sep_char = SepCharacter[tag_sep].value if tag_sep != "Unmodified" else ", "
        word_sep_char = SepCharacter[word_sep].value if word_sep != "Unmodified" else " "

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

        pos_out = rewrite_prompt(pos_prompt) if pos_prompt else ""
        neg_out = rewrite_prompt(neg_prompt) if neg_prompt else ""

        return pos_out, neg_out


NODE_CLASS_MAPPINGS = {
    "TagSeparator": TagSeparator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TagSeparator": "Tag Separator",
}
