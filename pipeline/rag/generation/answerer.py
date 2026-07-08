"""Camada de geracao de resposta — LLM plugavel.

Interface `LLMClient`:
  - complete(system, user, temperature, max_tokens) -> str
  - complete_stream(...) opcional

Implementacoes:
  - OpenAICompatibleClient: default local-first. Aponta para LM Studio / Ollama /
    qualquer endpoint OpenAI-compatible em `base_url` (default
    http://localhost:1234/v1). Usa `openai` SDK quando disponivel, fallback
    para `requests` direto (menos deps).
  - StubLLMClient: devolve texto fixo — usado em testes offline e quando nenhum
    LLM esta disponivel. Nao inventa: apenas repassa o contexto + pergunta,
    deixando explicito que o LLM nao respondeu.

Politica anti-alucinacao: o system prompt (prompts.py) e o guardiao. O
Answerer apenas orquestra e empacota em `Answer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable, Any

from pipeline.rag.generation.prompt_builder import build_prompt, build_sources_block
from pipeline.rag.models import Answer, Evidence
from pipeline.rag.utils.logging import get_logger

log = get_logger("app.answerer")


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
        ...


@dataclass
class OpenAICompatibleClient:
    """Cliente OpenAI-compatible para uso local (LM Studio / Ollama / vLLM)."""

    base_url: str = "http://localhost:1234/v1"
    api_key: str = ""
    model: str = "local"
    timeout: float = 60.0
    _client: Optional[Any] = field(default=None, init=False, repr=False)
    
    def _ensure_client(self) -> Optional[Any]:
        """Cria cliente apenas uma vez (cache)."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=self.timeout
                )
                log.debug("OpenAI client criado: %s", self.base_url)
            except ImportError:
                pass
            except Exception as e:
                log.debug("Erro ao criar OpenAI client: %s", e)
        return self._client

    def complete(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # 1) tentar via openai SDK (mais robusto) - reutiliza cliente
        client = self._ensure_client()
        if client is not None:
            try:
                resp = client.chat.completions.create(**payload)
                return resp.choices[0].message.content or ""
            except Exception as e:
                log.warning("openai SDK falhou (%s); fallback requests", e)

        # 2) fallback requests direto
        import json
        import urllib.request

        url = self.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"] or ""
        except Exception as e:
            raise RuntimeError(f"LLM indisponivel em {url}: {e}") from e


@dataclass
class StubLLMClient:
    """LLM fake para testes offline. Nao inventa — apenas declara insuficiencia."""

    def complete(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
        return (
            "[STUB LLM] Nenhum modelo de linguagem carregado. "
            "Baseado no contexto recuperado, nao posso gerar uma resposta final. "
            "Configure um cliente LLM (OpenAI-compatible local) para uso real."
        )


# ---------------------------------------------------------------- Answerer


@dataclass
class Answerer:
    llm: LLMClient
    default_mode: str = "answer"
    temperature: float = 0.2
    max_tokens: int = 700
    # Confianca considerada "baixa" (proxy: score max de retrieval). Default
    # conservador: cosine normalizado em [0,1]; < 0.3 sugere recuperacao fraca.
    low_confidence_threshold: float = 0.3
    max_chars_per_chunk: int = 1600

    def answer(
        self,
        question: str,
        evidence: list[Evidence],
        mode: Optional[str] = None,
    ) -> Answer:
        m = mode or self.default_mode
        system, user = build_prompt(question, evidence, mode=m, max_chars_per_chunk=self.max_chars_per_chunk)
        try:
            text = self.llm.complete(system, user, temperature=self.temperature, max_tokens=self.max_tokens)
        except Exception as e:
            log.error("LLM falhou: %s", e)
            text = (
                "Based on the documents provided, I could not find an answer. "
                f"(LLM error: {e})"
            )

        confidence = max((e.score for e in evidence), default=0.0)
        insufficient = (not evidence) or (confidence < self.low_confidence_threshold)
        # marcadores heuristicos no texto
        insuf_marker = (
            "could not find an answer" in text.lower()
            or "não encontrei" in text.lower()
            or "nao encontrei" in text.lower()
        )
        insufficient = insufficient or insuf_marker
        conflict = "conflict" in text.lower() or "conflito" in text.lower()

        return Answer(
            text=text,
            mode=m,
            evidence=tuple(evidence),
            confidence=confidence,
            insufficient_context=insufficient,
            conflict=conflict,
        )

    def answer_with_sources(self, question: str, evidence: list[Evidence], mode: Optional[str] = None) -> str:
        """Conveniencia: retorna texto da resposta + bloco de fontes."""
        ans = self.answer(question, evidence, mode=mode)
        return ans.text + "\n\n" + build_sources_block(evidence)
