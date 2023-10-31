class TagSeparator {
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
        BREAK: "BREAK",
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

    static restoreFormat(id, tagSepChar, wordSepChar, autoRefresh = false) {
        if (tagSepChar == ", " && wordSepChar == " ") {
            console.info(`TagSep ${id}: no need to restore (default separators), skipping`);
            return;
        }

        try {
            console.info(`TagSep reversing ${id}, wordSep=${wordSepChar}, tagSep=${tagSepChar}`);
            let textArea = gradioApp().getElementById(id).querySelector("textarea"),
                origText = textArea.value,
                newText = "";

            if (origText == "") {
                console.info(`TagSep ${id}: no input text, skipping restore`);
                return;
            } else {
                console.info(`TagSep ${id} Input: ${origText}`);
            }

            if (tagSepChar == ", ") {
                console.warn(`TagSep: tagSepChar is comma-space, skipping restore`);
            } else {
                newText = origText.split(tagSepChar).join(", ");
            }

            if (wordSepChar == " ") {
                console.warn(`TagSep: wordSepChar is already space, skipping`);
            } else {
                newText = newText
                    .split(", ")
                    .map((tag) => tag.replaceAll(wordSepChar, " "))
                    .join(", ");
            }
            console.info(`TagSep ${id} Output: ${newText}`);
            textArea.value = newText;

            if (autoRefresh) updateInput(textArea);
        } catch (e) {
            console.error(`TagSep failed to restore ${id}: ${e}`);
            return;
        }
    }
}

onUiLoaded(async () => {
    const restoreButton = TagSeparator.injectRestoreListener({
        onClick: () => {
            const pos_ids = ["txt2img_prompt", "img2img_prompt", "hires_prompt"];
            const neg_ids = ["txt2img_neg_prompt", "img2img_neg_prompt", "hires_neg_prompt"];
            let settings = TagSeparator.getCurrentSettings();
            if (settings.Enabled)
                pos_ids.forEach((id) =>
                    TagSeparator.restoreFormat(id, settings.TagSep, settings.WordSep, true),
                );
            if (settings.Enabled && settings.Negative)
                neg_ids.forEach((id) =>
                    TagSeparator.restoreFormat(id, settings.TagSep, settings.WordSep, true),
                );
        },
    });
    console.debug("TagSep Restore button listener injected");
});
