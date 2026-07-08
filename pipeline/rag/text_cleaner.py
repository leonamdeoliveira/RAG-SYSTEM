"""Text preprocessing para melhorar qualidade de chunks e embeddings.

Remove ruídos comuns em documentos OCR/PDF:
- Referências numéricas soltas [1], [2], etc
- URLs repetidas
- Linhas muito curtas (< 10 chars)
- Espaços em branco excessivos
- Caracteres especiais inválidos

Integrado no pipeline de chunking para melhorar qualidade dos embeddings.
"""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """Aplica todas as limpezas no texto."""
    text = _remove_reference_numbers(text)
    text = _remove_repeated_urls(text)
    text = _remove_url_fragments(text)
    text = _remove_duplicate_list_items(text)
    text = _remove_short_lines(text)
    text = _normalize_whitespace(text)
    text = _remove_invalid_chars(text)
    return text


def _remove_reference_numbers(text: str) -> str:
    """Remove referências numéricas soltas como [1], [2], etc."""
    # Remove [1], [2], [123], etc (mas não [texto])
    text = re.sub(r'\[\d+\]', '', text)
    return text


def _remove_url_fragments(text: str) -> str:
    """Remove fragmentos de URL quebrados como 'ript-to-1/'."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Detectar fragmentos de URL: linhas curtas com hífens, slashes, ou sem espaços
        # que não formam palavras válidas
        if (len(stripped) < 30 and 
            not stripped.startswith('#') and
            not stripped.startswith('-') and
            ('/' in stripped or '-' in stripped) and
            ' ' not in stripped and
            not any(c.isupper() for c in stripped[:3])):
            # Provavelmente é um fragmento de URL
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def _remove_duplicate_list_items(text: str) -> str:
    """Remove itens de lista duplicados (com e sem '- ')."""
    lines = text.split('\n')
    cleaned_lines = []
    seen_items = set()
    seen_urls = set()
    
    for line in lines:
        stripped = line.strip()
        
        # Verificar se é uma URL
        url_match = re.search(r'https?://[^\s<>"]+', stripped)
        if url_match:
            url = url_match.group(0)
            if url in seen_urls:
                # URL já vista, pular linha
                continue
            seen_urls.add(url)
        
        # Normalizar item de lista (remover "- " ou "1. " no início)
        normalized = re.sub(r'^[-*]\s+|^\d+\.\s+', '', stripped)
        
        if normalized and normalized in seen_items:
            # Item já visto, pular
            continue
        
        if normalized:
            seen_items.add(normalized)
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def _remove_repeated_urls(text: str) -> str:
    """Remove URLs que aparecem múltiplas vezes."""
    # Encontrar todas as URLs
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    urls = re.findall(url_pattern, text)
    
    # Contar ocorrências
    url_counts = {}
    for url in urls:
        url_counts[url] = url_counts.get(url, 0) + 1
    
    # Remover URLs que aparecem mais de uma vez (manter apenas a primeira)
    seen_urls = set()
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Encontrar URLs na linha
        line_urls = re.findall(url_pattern, line)
        should_remove_line = False
        
        for url in line_urls:
            if url in seen_urls:
                # URL já vista, remover linha inteira se for apenas URL
                if line.strip() == url or line.strip().startswith(url):
                    should_remove_line = True
                    break
            else:
                seen_urls.add(url)
        
        if not should_remove_line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def _remove_short_lines(text: str) -> str:
    """Remove linhas muito curtas (< 10 chars) que são ruído."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Manter linhas com conteúdo significativo
        # Preservar headings (#), listas (-, *, +, 1.), e código (```)
        if (len(stripped) >= 10 or 
            not stripped or
            stripped.startswith('#') or
            stripped.startswith(('-', '*', '+')) or
            stripped.startswith('```') or
            re.match(r'^\d+[.)]\s', stripped)):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def _normalize_whitespace(text: str) -> str:
    """Normaliza espaços em branco."""
    # Remover espaços múltiplos (exceto newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remover newlines múltiplos (máximo 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remover espaços no início/fim de linhas
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines)


def _remove_invalid_chars(text: str) -> str:
    """Remove caracteres inválidos ou de controle."""
    # Remover caracteres de controle (exceto newline e tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def clean_markdown(text: str) -> str:
    """Limpeza específica para Markdown."""
    # Remover headings vazios
    text = re.sub(r'^#{1,6}\s*\n', '', text, flags=re.MULTILINE)
    # Remover links vazios []() ou com texto mas sem URL
    text = re.sub(r'\[[^\]]*\]\(\s*\)', '', text)
    # Remover imagens vazias ![]()
    text = re.sub(r'!\[[^\]]*\]\(\s*\)', '', text)
    # Remover linhas apenas com !
    text = re.sub(r'^!\s*$', '', text, flags=re.MULTILINE)
    return text
