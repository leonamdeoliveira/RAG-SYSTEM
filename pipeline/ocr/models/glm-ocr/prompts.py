MARKDOWN_PROMPT = "Extract all text from this document image faithfully and output as clean Markdown, preserving tables as pipe tables, keeping all numbers and values exactly as written, maintaining the original reading order, and including all metadata such as page numbers and footnotes."

HTML_PROMPT = "Extract all text from this document image faithfully."

JSON_PROMPT = "Extract all text from this document image faithfully as JSON."

PROMPTS = {
    "markdown": MARKDOWN_PROMPT,
    "html": HTML_PROMPT,
    "json": JSON_PROMPT,
}
