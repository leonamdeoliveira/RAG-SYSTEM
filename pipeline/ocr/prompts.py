MARKDOWN_PROMPT = """Extract ALL text from this document image faithfully and output it in clean Markdown format.

Rules:
- Do NOT invent, add, or summarize any content.
- Do NOT translate. Preserve the original language.
- Preserve natural reading order (left-to-right, top-to-bottom).
- Use Markdown headings (# ##) for titles and section headers.
- Use Markdown tables for tabular data. Keep them readable.
- Use bullet lists or numbered lists where appropriate.
- Wrap inline code or formulas in backticks if they appear technical.
- If any part is unreadable or uncertain, mark it with [unclear].
- CRITICAL: Use ONLY Markdown syntax. Do NOT output HTML tags like <div>, <p>, <br/>, <table>, <img>, <span>, <h1>-<h6>, <ul>, <ol>, <li>, or any other HTML tags. Use pure Markdown only.
- Output ONLY the extracted content, no explanations."""

HTML_PROMPT = """Extract ALL text from this document image faithfully and output it as structured HTML.

Rules:
- Do NOT invent, add, or summarize any content.
- Do NOT translate. Preserve the original language.
- Preserve natural reading order.
- Use semantic HTML tags: <h1>-<h6> for headings, <p> for paragraphs, <table> for tables, <ul>/<ol> for lists.
- Use <pre><code> for code blocks or formulas if applicable.
- If any part is unreadable or uncertain, wrap it in <span class="unclear">.
- Output ONLY the HTML content inside <body> tags, no <html>/<head> wrappers, no explanations."""

JSON_PROMPT = """Extract ALL text from this document image faithfully and output it as a JSON structure.

Rules:
- Do NOT invent, add, or summarize any content.
- Do NOT translate. Preserve the original language.
- Use this JSON structure:
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
- Output ONLY the JSON, no explanations or markdown fences."""

PROMPTS = {
    "markdown": MARKDOWN_PROMPT,
    "html": HTML_PROMPT,
    "json": JSON_PROMPT,
}
