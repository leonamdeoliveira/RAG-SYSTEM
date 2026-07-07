PROMPTS = {
    "markdown": (
        "Extract ALL text from this document page and format it as clean Markdown. "
        "Rules:\n"
        "- Preserve the exact text, spelling, numbers, and punctuation.\n"
        "- Use # for main titles, ## for section headings, ### for sub-headings.\n"
        "- Format tables using Markdown table syntax (| col | col |).\n"
        "- Use - for bullet lists, 1. for numbered lists.\n"
        "- Use **bold** and *italic* where the original document uses them.\n"
        "- Preserve the reading order (left to right, top to bottom).\n"
        "- Do NOT add any commentary, explanations, or extra text.\n"
        "- If text is unclear, mark it as [unclear].\n"
        "- Output ONLY the Markdown content, nothing else."
    ),
}
