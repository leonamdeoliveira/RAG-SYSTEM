MARKDOWN_PROMPT = """Extract ALL text from this document image faithfully and output it as clean Markdown.

RULES - STRICTLY FOLLOW:
- Do NOT invent, add, summarize, paraphrase, or modify any content.
- Do NOT translate. Preserve the original language exactly.
- Preserve natural reading order (left-to-right, top-to-bottom).
- Use # for headings, ## for subheadings, etc.
- Use | table | syntax for tables.
- Use - for list items.
- Use ``` for code blocks.
- If any part is unreadable, use [unclear] - do NOT guess or invent.
- Output ONLY the Markdown content, no preamble, no explanations."""

HTML_PROMPT = """Extract ALL text from this document image faithfully and output it as structured HTML.

RULES - STRICTLY FOLLOW:
- Do NOT invent, add, summarize, paraphrase, or modify any content.
- Do NOT translate. Preserve the original language exactly.
- Preserve natural reading order.
- Use ONLY clean semantic HTML: <h1>-<h6> for headings, <p> for paragraphs, <table> for tables, <ul>/<ol> for lists.
- Use <pre><code> for code blocks or formulas if applicable.
- NEVER output HTML attributes (no data-bbox, data-label, style, class, id, or any other attributes).
- Tags only, no attributes whatsoever.
- If any part is unreadable, use [unclear] - do NOT guess or invent.
- Output ONLY the HTML content, no <html>/<head>/<body> wrappers, no preamble, no explanations."""

JSON_PROMPT = """Extract ALL text from this document image faithfully and output it as a JSON structure.

RULES - STRICTLY FOLLOW:
- Do NOT invent, add, summarize, paraphrase, or modify any content.
- Do NOT translate. Preserve the original language exactly.
- Use this exact JSON structure:
{
  "title": "document title if detected",
  "sections": [
    {
      "heading": "section heading or null",
      "content": "paragraph text or list items",
      "type": "paragraph|list|table|code"
    }
  ],
  "tables": [
    ["row1col1", "row1col2"],
    ["row2col1", "row2col2"]
  ]
}
- Preserve natural reading order.
- If any part is unreadable or uncertain, use "[unclear]" as the value.
- Output ONLY the JSON object, no markdown fences (```), no preamble, no explanations."""

PROMPTS = {
    "markdown": MARKDOWN_PROMPT,
    "html": HTML_PROMPT,
    "json": JSON_PROMPT,
}
