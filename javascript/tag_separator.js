class TagSeparator {
    static t2iToolsSelector = "#txt2img_tools > div";
    static i2iToolsSelector = "#img2img_tools > div";

    static SepCharacter = {
        Backslash: "\\",
        Comma: ",",
        CommaSpace: ", ",
        Dash: "-",
        Equals: "=",
        Plus: "+",
        Slash: "/",
        Space: " ",
        Underscore: "_",
        At: "@",
        Hash: "#",
        Percent: "%",
        Ampersand: "&",
        Asterisk: "*",
        Caret: "^",
        Tilde: "~",
        Empty: "",
        Unmodified: "U",
    };

    static injectRestoreListener({ onClick }) {
        const restoreButton = gradioApp().getElementById("tag_sep_restore");
        restoreButton.addEventListener("click", onClick);
        return restoreButton;
    }

    static getExtensionInput(id) {
        return gradioApp().getElementById(`tag_sep_${id}`).querySelector("input");
    }

    static getCurrentSettings() {
        return {
            Enabled: this.getExtensionInput("enabled").checked,
            Negative: this.getExtensionInput("neg_enabled").checked,
            IgnoreMeta: this.getExtensionInput("ignore_meta").checked,
            TagSep: this.SepCharacter[this.getExtensionInput("tag_sep").value],
            WordSep: this.SepCharacter[this.getExtensionInput("word_sep").value],
        };
    }

    static reverseFormat(id, tagSepChar, wordSepChar, autoRefresh = false) {
        if (tagSepChar == ", " && wordSepChar == " ") {
            console.info(`No reverse-processing needed for ${id}`);
            return;
        }

        try {
            let textArea = gradioApp().getElementById(id).querySelector("textarea");

            console.info(`TagSep reversing ${id}, wordSep=${wordSepChar}, tagSep=${tagSepChar}`);

            let origText = textArea.value;
            if (origText == "") {
                console.info(`No text in ${id} to reverse`);
                return;
            } else if (origText.includes(", ")) {
                console.info(`Input ${id} contains comma-separated tags already`);
                return;
            }

            console.info(`TagSep Input: ${origText}`);
            let newText = origText.replaceAll(tagSepChar, ", ").replaceAll(wordSepChar, " ");
            console.info(`TagSep Output: ${newText}`);
            textArea.value = newText;

            if (autoRefresh) updateInput(textArea);
        } catch (e) {
            console.error(`TagSep failed to reverse ${id}: ${e}`);
            return;
        }
    }
}

onUiLoaded(async () => {
    const restoreButton = TagSeparator.injectRestoreListener({
        onClick: () => {
            const ids = [
                "txt2img_prompt",
                "txt2img_neg_prompt",
                "img2img_prompt",
                "img2img_neg_prompt",
                "hires_prompt",
                "hires_neg_prompt",
            ];
            let settings = TagSeparator.getCurrentSettings();
            if (settings.Enabled) {
                ids.forEach((id) =>
                    TagSeparator.reverseFormat(id, settings.TagSep, settings.WordSep, true),
                );
            }
        },
    });
    console.info("TagSep Restore button listener injected");
});
