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
    text = _normalize_line_endings(text)
    text = _remove_reference_numbers(text)
    text = _remove_repeated_urls(text)
    text = _remove_url_fragments(text)
    text = _remove_duplicate_list_items(text)
    text = _remove_keyword_blocks(text)
    text = _detect_implicit_headings(text)
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
    """Remove fragmentos de URL quebrados preservando código."""
    lines = text.split('\n')
    cleaned_lines = []
    in_code_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # Detectar blocos de código
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            cleaned_lines.append(line)
            continue
        
        # Preservar linhas dentro de código
        if in_code_block:
            cleaned_lines.append(line)
            continue
        
        # Detectar fragmentos de URL reais (mais específico)
        if (len(stripped) < 30 and 
            not stripped.startswith('#') and
            not stripped.startswith('-') and
            not stripped.startswith('const ') and
            not stripped.startswith('let ') and
            not stripped.startswith('var ') and
            '=' not in stripped and
            '<' not in stripped and
            '>' not in stripped and
            ('/' in stripped or '-' in stripped) and
            ' ' not in stripped and
            not any(c.isupper() for c in stripped[:3])):
            # Provavelmente é fragmento de URL
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
    """Normaliza espaços em branco preservando estrutura."""
    lines = text.split('\n')
    cleaned = []
    in_code_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # Detectar blocos de código
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            cleaned.append(line)
            continue
        
        # Preservar indentação em código
        if in_code_block:
            cleaned.append(line)
            continue
        
        # Preservar indentação de listas
        if re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+[.)]\s+', line):
            # Normalizar apenas espaços internos, manter indentação inicial
            indent = len(line) - len(line.lstrip())
            content = ' '.join(stripped.split())
            cleaned.append(' ' * indent + content)
            continue
        
        # Normalizar linha normal
        if stripped:
            cleaned.append(' '.join(stripped.split()))
        else:
            cleaned.append('')
    
    # Remover newlines múltiplos (máximo 2)
    text = '\n'.join(cleaned)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def _remove_invalid_chars(text: str) -> str:
    """Remove caracteres inválidos ou de controle."""
    # Remover caracteres de controle (exceto newline e tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def _normalize_line_endings(text: str) -> str:
    """Normaliza \\r\\n para \\n."""
    return text.replace('\r\n', '\n').replace('\r', '\n')


def _remove_keyword_blocks(text: str) -> str:
    """Remove blocos de palavras-chave/keywords."""
    # Detectar seções como "Palavras-chave:" ou "Keywords:" seguidas de lista
    pattern = r'(?i)(palavras?[\-\s]chaves?|keywords?|tags?|etiquetas?):?\s*\n[^\n]+(?:,[^\n]+)*'
    text = re.sub(pattern, '', text)
    return text


def _detect_implicit_headings(text: str) -> str:
    """Detecta headings implícitos (linhas curtas sem #) e adiciona #."""
    lines = text.split('\n')
    result = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Verificar se é um heading implícito:
        # - Linha curta (10-80 chars)
        # - Não começa com #, -, *, +, ou número
        # - Não é URL
        # - Não contém vírgula (não é lista)
        # - Não termina com . (não é frase)
        # - Seguida por linha em branco ou conteúdo
        # - Tem palavras capitalizadas ou é título
        # - Linha anterior está vazia (headings vêm após linha vazia)
        if (10 <= len(stripped) <= 80 and
            not stripped.startswith(('#', '-', '*', '+', '`')) and
            not re.match(r'^\d+[.)]\s', stripped) and
            not re.match(r'^https?://', stripped) and
            ',' not in stripped and
            not stripped.endswith(('.', '!', '?')) and
            not stripped.endswith(':') and
            (i == 0 or lines[i - 1].strip() == '') and
            (i + 1 >= len(lines) or lines[i + 1].strip() == '' or not lines[i + 1].strip().startswith('#'))):
            
            # Verificar se parece um título (palavras capitalizadas ou todas maiúsculas)
            words = stripped.split()
            capitalized = sum(1 for w in words if w[0].isupper()) if words else 0
            if capitalized > len(words) * 0.5 or stripped.isupper():
                # Adicionar # como heading
                result.append(f'# {stripped}')
                continue
        
        result.append(line)
    
    return '\n'.join(result)


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
