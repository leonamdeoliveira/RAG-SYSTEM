#!/usr/bin/env python3
"""RAG System — Pipeline completo local-first: OCR + RAG.

Comandos:
    python main.py ingest <input>     OCR + ingestão de documentos
    python main.py query <pergunta>   Consulta aos documentos indexados
    python main.py reindex            Reindexação completa
    python main.py run <input> <q>    Pipeline completo (one-shot)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _setup_logging(level: str = "INFO") -> None:
    """Configura logging globalmente para todos os módulos."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Configura loggers da aplicação
    for logger_name in ["app", "pipeline", "rag", "ocr"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)


def _get_skill_dir() -> Path:
    return Path(__file__).resolve().parent


def _cleanup_ocr_temp(ocr_output_dir: Path) -> None:
    """Remove arquivos temporarios do OCR (.partial, .json, images/) apos copiar o .md."""
    import shutil
    try:
        shutil.rmtree(str(ocr_output_dir), ignore_errors=True)
    except Exception:
        pass


def _close_store(store) -> None:
    """Fecha o ZvecStore para liberar locks e recursos."""
    if store is None:
        return
    try:
        if hasattr(store, 'collection') and store.collection is not None:
            if hasattr(store.collection, 'close'):
                store.collection.close()
    except Exception as e:
        logger = logging.getLogger("rag-system")
        logger.debug("Erro ao fechar store (ignorado): %s", e)


def cmd_ingest(args: argparse.Namespace) -> None:
    from pipeline.ocr.lmstudio_client import LMStudioClient, LMStudioClientError
    from pipeline.ocr.ocr_pipeline import OCRPipeline
    from pipeline.ocr.model_loader import load_model_config
    from pipeline.ocr.native_extractor import extract_text, EXTRACTORS as NATIVE_FORMATS
    from pipeline.ocr.ocr_engine import AIEngine, TesseractEngine, OCRRouter, HybridOCRConfig

    skill_dir = _get_skill_dir()
    markdown_dir = Path(args.markdown_dir or os.environ.get("RAG_MARKDOWN_DIR", str(skill_dir / "markdown")))
    markdown_dir.mkdir(parents=True, exist_ok=True)
    use_dummy = args.provider == "dummy"

    if args.rag_only:
        _run_rag_ingest(args, skill_dir, markdown_dir, use_dummy)
        return

    input_path = Path(args.input) if args.input else None
    if input_path is None:
        logger = logging.getLogger("rag-system")
        logger.error("Nenhum input especificado. Use: python main.py ingest <arquivo>")
        sys.exit(1)

    logger = logging.getLogger("rag-system")
    logger.info("=== OCR Pipeline ===")

    files = _collect_input_files(input_path)
    logger.info("Encontrados %d arquivos para processar", len(files))

    ocr_mode = args.ocr_mode or os.environ.get("RAG_OCR_MODE", "hybrid")
    model_name = args.model or os.environ.get("RAG_OCR_MODEL", "glm-ocr")
    quality_threshold = args.quality_threshold if args.quality_threshold is not None else float(os.environ.get("RAG_OCR_QUALITY_THRESHOLD", "0.70"))
    ocr_langs = args.ocr_langs or os.environ.get("RAG_OCR_LANGS", "por+eng")
    dpi_val = args.dpi if args.dpi is not None else int(os.environ.get("RAG_OCR_DPI", "200"))
    base_url = args.lmstudio_url or os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    lm_model = args.lmstudio_model or os.environ.get("LMSTUDIO_MODEL", "glm-ocr")
    api_key = args.lmstudio_api_key or os.environ.get("LMSTUDIO_API_KEY", "")
    timeout_val = args.timeout if hasattr(args, 'timeout') and args.timeout else 300
    max_retries = getattr(args, 'retries', 3) or 3
    resume = getattr(args, 'resume', False) or False

    model_config = load_model_config(model_name)
    lmstudio_model = args.lmstudio_model or os.environ.get("LMSTUDIO_MODEL") or model_config.get("lmstudio_model", model_name)
    timeout_val = timeout_val if timeout_val != 300 else model_config.get("default_timeout", 300)
    max_tokens = model_config.get("max_tokens", 48000)
    use_multi = model_config.get("use_multi_image", False)
    dpi_final = dpi_val if dpi_val else model_config.get("default_dpi", 200)

    extra_params = {}
    ngram_size = model_config.get("no_repeat_ngram_size")
    ngram_window = model_config.get("ngram_window")
    if ngram_size:
        extra_params["custom_logit_processor"] = (
            f"DeepseekOCRNoRepeatNGram(size={ngram_size},window={ngram_window or 128})"
        )
        extra_params["custom_params"] = {"ngram_size": ngram_size, "window_size": ngram_window or 128}

    client = LMStudioClient(
        base_url=base_url, model=lmstudio_model, api_key=api_key,
        timeout=timeout_val, max_retries=max_retries,
        max_tokens=max_tokens, extra_params=extra_params,
    )

    hybrid_config = None
    router = None
    if ocr_mode != "legacy":
        hybrid_config = HybridOCRConfig(
            mode=ocr_mode, classic_engine="tesseract",
            langs=ocr_langs, quality_threshold_accept=quality_threshold,
            enable_glm_fallback=True, ocr_timeout=timeout_val,
        )
    if ocr_mode in ("hybrid", "classic_only"):
        ai_engine = AIEngine(client=client, model_name=model_name, mode="text-first")
        tesseract = TesseractEngine(langs=ocr_langs, timeout=timeout_val)
        router = OCRRouter(engines={ai_engine.name: ai_engine, tesseract.name: tesseract}, config=hybrid_config)

    total_processed = 0
    for f in files:
        logger.info("Processando: %s", f)
        ext = f.suffix.lower()

        if ext in NATIVE_FORMATS:
            dest = markdown_dir / f"{f.stem}.md"
            if dest.exists() and not resume:
                logger.info("  → pulando (ja existe): %s", dest)
                total_processed += 1
                continue
            logger.info("Extração nativa (%s)", ext)
            text = extract_text(f)
            dest.write_text(text, encoding="utf-8")
            logger.info("  → %s", dest)
            total_processed += 1
            continue

        if ext not in (".pdf", ".png", ".jpg", ".jpeg"):
            logger.warning("Formato não suportado: %s", ext)
            continue

        dest = markdown_dir / f"{f.stem}.md"
        if dest.exists() and not resume:
            logger.info("  → pulando (ja existe): %s", dest)
            total_processed += 1
            continue

        ocr_output_dir = markdown_dir / f"_ocr_{f.stem}"
        ocr_output_dir.mkdir(parents=True, exist_ok=True)

        pipeline = OCRPipeline(
            client=client, output_dir=ocr_output_dir,
            formats=["markdown"], model_name=model_name,
            mode="text-first", dpi=dpi_final, resume=resume,
            basename=f.stem, hybrid_config=hybrid_config,
            router=router, use_layout=True,
            image_output_dir=ocr_output_dir / "images",
        )

        try:
            if ext == ".pdf":
                if use_multi:
                    from pipeline.ocr.pdf_utils import load_pdf as load_pdf_fn
                    pages = load_pdf_fn(path=f, dpi=dpi_final, mode="text-first")
                    bs = model_config.get("batch_size", 0) or len(pages)
                    batches = [pages[i:i+bs] for i in range(0, len(pages), bs)]
                    pipeline._prepare_outputs()
                    for batch_idx, batch in enumerate(batches):
                        try:
                            content = pipeline._process_batch(batch, "markdown")
                            pipeline._append_saida("markdown", content)
                        except LMStudioClientError as e:
                            logger.error("Batch %d failed: %s", batch_idx + 1, e)
                else:
                    pipeline.run(f)
            elif ext in (".png", ".jpg", ".jpeg"):
                from PIL import Image as PILImage
                from pipeline.ocr.pdf_utils import PDFPage
                image = PILImage.open(f).convert("RGB")
                page = PDFPage(page_num=1, image=image)
                content = pipeline.process_page(page, "markdown")
                pipeline._save_partial(1, "markdown", content)
                pipeline._append_saida("markdown", content)
            else:
                continue

            md_path = pipeline._output_path("markdown")
            if md_path.exists() and md_path.stat().st_size > 0:
                dest.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
                logger.info("  → %s", dest)
                total_processed += 1
            _cleanup_ocr_temp(ocr_output_dir)
        except KeyboardInterrupt:
            logger.warning("Processamento interrompido pelo usuário")
            raise
        except Exception as e:
            logger.error("Erro no OCR de %s: %s", f.name, e)
            _cleanup_ocr_temp(ocr_output_dir)

    logger.info("OCR concluído: %d arquivos processados", total_processed)

    if not args.ocr_only and total_processed > 0:
        _run_rag_ingest(args, skill_dir, markdown_dir, use_dummy)


def _collect_input_files(input_path: Path) -> list[Path]:
    OCR_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".docx", ".pptx", ".xlsx",
                ".xlsm", ".epub", ".csv", ".html", ".htm", ".md", ".tex", ".txt"}
    if input_path.is_dir():
        files: list[Path] = []
        for ext in OCR_EXTS:
            files.extend(input_path.rglob(f"*{ext}"))
        return files
    return [input_path]


def _run_rag_ingest(args: argparse.Namespace, skill_dir: Path, markdown_dir: Path, use_dummy: bool = False) -> None:
    from pipeline.rag.config import get_settings
    from pipeline.rag.factory import build_ingest_pipeline

    logger = logging.getLogger("rag-system")
    logger.info("=== RAG Ingest ===")

    index_dir = args.index_dir or os.environ.get("RAG_INDEX_DIR", str(skill_dir / "index"))

    overrides = {
        "data_dir": str(markdown_dir),
        "index_dir": str(index_dir),
        "zvec_path": str(Path(index_dir) / "zvec_collection"),
    }
    if use_dummy:
        overrides["embedding_provider"] = "dummy"
        overrides["embedding_dimension"] = args.dimension or 64
        overrides["zvec_enable_sparse"] = False
    if args.dimension:
        overrides["embedding_dimension"] = int(args.dimension)

    s = get_settings(**overrides)
    pipeline = build_ingest_pipeline(s)
    report = pipeline.run()
    logger.info(str(report))
    logger.info("RAG ingest concluído.")


def cmd_query(args: argparse.Namespace) -> None:
    from pipeline.rag.config import get_settings
    from pipeline.rag.factory import build_query_pipeline

    skill_dir = _get_skill_dir()
    index_dir = args.index_dir or os.environ.get("RAG_INDEX_DIR", str(skill_dir / "index"))
    markdown_dir = args.markdown_dir or os.environ.get("RAG_MARKDOWN_DIR", str(skill_dir / "markdown"))
    use_dummy = args.provider == "dummy"

    overrides = {
        "data_dir": str(markdown_dir),
        "index_dir": str(index_dir),
        "zvec_path": str(Path(index_dir) / "zvec_collection"),
    }
    if use_dummy:
        overrides["embedding_provider"] = "dummy"
        overrides["embedding_dimension"] = args.dimension or 64
        overrides["zvec_enable_sparse"] = False
    if args.dimension:
        overrides["embedding_dimension"] = int(args.dimension)
    if args.mode:
        overrides["answer_mode"] = args.mode
    if args.llm_model:
        overrides["llm_model"] = args.llm_model
    if args.retrieval_mode:
        overrides["retrieval_mode"] = args.retrieval_mode

    s = get_settings(**overrides)
    query_pipeline, retriever, store = build_query_pipeline(s)

    try:
        filters = _parse_filters(args.filter)

        if args.interactive:
            _interactive_loop(query_pipeline, filters)
            return

        question = args.question
        if not question:
            logger = logging.getLogger("rag-system")
            logger.error("Nenhuma pergunta especificada. Use: python main.py query \"sua pergunta\"")
            sys.exit(1)

        result = query_pipeline.run(question, filters=filters)
        _display_answer(result)
    finally:
        _close_store(store)


def cmd_retrieve(args: argparse.Namespace) -> None:
    """Apenas retrieval (sem LLM) — retorna chunks formatados para o chat IA."""
    import time
    from pipeline.rag.config import get_settings
    from pipeline.rag.factory import build_query_pipeline

    skill_dir = _get_skill_dir()
    index_dir = args.index_dir or os.environ.get("RAG_INDEX_DIR", str(skill_dir / "index"))
    markdown_dir = args.markdown_dir or os.environ.get("RAG_MARKDOWN_DIR", str(skill_dir / "markdown"))
    use_dummy = args.provider == "dummy"

    overrides = {
        "data_dir": str(markdown_dir),
        "index_dir": str(index_dir),
        "zvec_path": str(Path(index_dir) / "zvec_collection"),
    }
    if use_dummy:
        overrides["embedding_provider"] = "dummy"
        overrides["embedding_dimension"] = args.dimension or 64
        overrides["zvec_enable_sparse"] = False
    if args.dimension:
        overrides["embedding_dimension"] = int(args.dimension)
    if args.retrieval_mode:
        overrides["retrieval_mode"] = args.retrieval_mode
    if args.top_k:
        overrides["retrieval_top_k"] = int(args.top_k)

    t0 = time.perf_counter()
    s = get_settings(**overrides)
    t1 = time.perf_counter()
    query_pipeline, retriever, store = build_query_pipeline(s)
    t2 = time.perf_counter()

    try:
        filters = _parse_filters(args.filter)

        if args.interactive:
            print("\nRAG Retrieve — Modo Interativo (modelo carregado)")
            print("Digite a pergunta ou :quit para sair, :mode <dense|fts|hybrid|sparse> para trocar\n")
            while True:
                try:
                    q = input(">>> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n")
                    break
                if not q:
                    continue
                if q == ":quit":
                    break
                if q.startswith(":mode "):
                    m = q.split(" ", 1)[1].strip()
                    retriever.config.mode = m
                    print(f"  Modo: {m}")
                    continue
                tq0 = time.perf_counter()
                hits, confidence = retriever.retrieve_with_confidence(q, filters)
                tq1 = time.perf_counter()
                print(f"\nEvidencias ({len(hits)} chunks, conf={confidence:.3f}, {int((tq1 - tq0) * 1000)}ms):\n")
                for i, ev in enumerate(hits, 1):
                    loc_parts = [ev.source_path]
                    if ev.h1:
                        loc_parts.append(ev.h1)
                    if ev.h2:
                        loc_parts.append(ev.h2)
                    loc = " > ".join(loc_parts)
                    snippet = ev.snippet[:200]
                    suffix = "..." if len(ev.snippet) > 200 else ""
                    print(f"[{i}] {loc}  (score={ev.score:.3f})")
                    try:
                        print(f"    {snippet}{suffix}\n")
                    except UnicodeEncodeError:
                        print(f"    {snippet.encode('ascii', errors='replace').decode()}{suffix}\n")
            return

        question = args.question
        if not question:
            logger = logging.getLogger("rag-system")
            logger.error("Nenhuma pergunta especificada. Use: python main.py retrieve \"sua pergunta\" ou --interactive")
            sys.exit(1)

        hits, confidence = retriever.retrieve_with_confidence(question, filters)
        t3 = time.perf_counter()
        max_chars = getattr(args, 'max_chars', None) or s.prompt_max_chars_per_chunk

        logger = logging.getLogger("rag-system")
        logger.info("TIMING: build=%.1fms  retrieval=%.0fms  TOTAL=%.0fms",
                    (t2 - t1) * 1000, (t3 - t2) * 1000, (t3 - t1) * 1000)

        print(f"\nEvidencias recuperadas ({len(hits)} chunks, confianca={confidence:.3f}):\n")
        for i, ev in enumerate(hits, 1):
            loc_parts = [ev.source_path]
            if ev.h1:
                loc_parts.append(ev.h1)
            if ev.h2:
                loc_parts.append(ev.h2)
            loc = " > ".join(loc_parts)
            snippet = ev.snippet[:max_chars]
            suffix = "..." if len(ev.snippet) > max_chars else ""
            print(f"[{i}] {loc}  (score={ev.score:.3f})")
            try:
                print(f"    {snippet}{suffix}")
            except UnicodeEncodeError:
                print(f"    {snippet.encode('ascii', errors='replace').decode()}{suffix}")
            print()
    finally:
        _close_store(store)


def cmd_retrieve_batch(args: argparse.Namespace) -> None:
    import time
    from pipeline.rag.config import get_settings
    from pipeline.rag.factory import build_query_pipeline

    skill_dir = _get_skill_dir()
    index_dir = args.index_dir or os.environ.get("RAG_INDEX_DIR", str(skill_dir / "index"))
    markdown_dir = args.markdown_dir or os.environ.get("RAG_MARKDOWN_DIR", str(skill_dir / "markdown"))
    use_dummy = args.provider == "dummy"

    overrides = {
        "data_dir": str(markdown_dir),
        "index_dir": str(index_dir),
        "zvec_path": str(Path(index_dir) / "zvec_collection"),
    }
    if use_dummy:
        overrides["embedding_provider"] = "dummy"
        overrides["embedding_dimension"] = args.dimension or 64
        overrides["zvec_enable_sparse"] = False
    if args.dimension:
        overrides["embedding_dimension"] = int(args.dimension)
    if args.retrieval_mode:
        overrides["retrieval_mode"] = args.retrieval_mode
    if args.top_k:
        overrides["retrieval_top_k"] = int(args.top_k)

    t0 = time.perf_counter()
    s = get_settings(**overrides)
    query_pipeline, retriever, store = build_query_pipeline(s)

    try:
        filters = _parse_filters(args.filter)
        max_chars = getattr(args, 'max_chars', None) or s.prompt_max_chars_per_chunk
        t1 = time.perf_counter()
        logger = logging.getLogger("rag-system")
        logger.info("Modelo carregado em %.1fs. Processando %d queries...", t1 - t0, len(args.questions))

        all_results = []
        for q in args.questions:
            tq0 = time.perf_counter()
            hits, confidence = retriever.retrieve_with_confidence(q, filters)
            tq1 = time.perf_counter()
            results = []
            for ev in hits:
                results.append({
                    "chunk_id": ev.chunk_id,
                    "source_path": ev.source_path,
                    "score": ev.score,
                    "snippet": ev.snippet[:max_chars],
                    "h1": ev.h1 or "",
                    "h2": ev.h2 or "",
                    "title": ev.title or "",
                })
            all_results.append({
                "query": q,
                "confidence": confidence,
                "count": len(hits),
                "time_ms": int((tq1 - tq0) * 1000),
                "results": results,
            })

        t2 = time.perf_counter()
        output = {
            "total_time_ms": int((t2 - t0) * 1000),
            "model_load_ms": int((t1 - t0) * 1000),
            "query_count": len(args.questions),
            "queries": all_results,
        }
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        _close_store(store)


def cmd_reindex(args: argparse.Namespace) -> None:
    from pipeline.rag.config import get_settings
    from pipeline.rag.factory import build_ingest_pipeline, build_store

    skill_dir = _get_skill_dir()
    markdown_dir = Path(args.markdown_dir or os.environ.get("RAG_MARKDOWN_DIR", str(skill_dir / "markdown")))
    index_dir = Path(args.index_dir or os.environ.get("RAG_INDEX_DIR", str(skill_dir / "index")))
    use_dummy = args.provider == "dummy"

    overrides = {
        "data_dir": str(markdown_dir),
        "index_dir": str(index_dir),
        "zvec_path": str(index_dir / "zvec_collection"),
    }
    if use_dummy:
        overrides["embedding_provider"] = "dummy"
        overrides["embedding_dimension"] = args.dimension or 64
        overrides["zvec_enable_sparse"] = False
    if args.dimension:
        overrides["embedding_dimension"] = int(args.dimension)

    s = get_settings(**overrides)

    if args.purge:
        import shutil
        try:
            store = build_store(s)
            store.collection.destroy()
        except Exception:
            pass
        manifest_path = index_dir / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        zvec_dir = index_dir / "zvec_collection"
        if zvec_dir.exists():
            try:
                shutil.rmtree(str(zvec_dir), ignore_errors=True)
            except Exception:
                pass

    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    s = get_settings(**overrides)
    pipeline = build_ingest_pipeline(s)
    report = pipeline.run()

    logger = logging.getLogger("rag-system")
    logger.info(str(report))


# Modos válidos para validação
VALID_ANSWER_MODES = {"answer", "answer_with_citations", "extractive_summary", "study_mode"}
VALID_RETRIEVAL_MODES = {"dense", "fts", "hybrid", "sparse", "semantic"}


def cmd_run(args: argparse.Namespace) -> None:
    """Pipeline completo: OCR + ingest + query."""
    skill_dir = _get_skill_dir()
    markdown_dir = Path(args.markdown_dir or os.environ.get("RAG_MARKDOWN_DIR", str(skill_dir / "markdown")))
    index_dir = Path(args.index_dir or os.environ.get("RAG_INDEX_DIR", str(skill_dir / "index")))

    logger = logging.getLogger("rag-system")

    # Reutiliza args com overrides mínimos
    ingest_args = argparse.Namespace(**vars(args))
    ingest_args.ocr_only = False
    ingest_args.rag_only = False
    ingest_args.markdown_dir = str(markdown_dir)
    ingest_args.index_dir = str(index_dir)
    
    cmd_ingest(ingest_args)

    # Query também reutiliza
    query_args = argparse.Namespace(**vars(args))
    query_args.interactive = False
    query_args.markdown_dir = str(markdown_dir)
    query_args.index_dir = str(index_dir)
    
    cmd_query(query_args)


def _interactive_loop(query_pipeline, filters: dict) -> None:
    print("\nRAG System — Modo Interativo")
    print(f"Modos disponíveis: {', '.join(VALID_ANSWER_MODES)}")
    print("Comandos: :quit, :mode <modo>, :help\n")
    
    mode = "answer_with_citations"
    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break
        if not q:
            continue
        if q == ":quit":
            break
        if q.startswith(":mode "):
            new_mode = q.split(" ", 1)[1].strip()
            if new_mode in VALID_ANSWER_MODES:
                mode = new_mode
                print(f"  ✓ Modo: {mode}")
            else:
                print(f"  ✗ Modo inválido: '{new_mode}'")
                print(f"     Válidos: {', '.join(VALID_ANSWER_MODES)}")
            continue
        if q == ":help":
            print(f"  Modos: {', '.join(VALID_ANSWER_MODES)}")
            print("  :quit - Sair")
            print("  :mode <modo> - Trocar modo")
            continue
        result = query_pipeline.run(q, filters=filters, mode=mode)
        _display_answer(result)


def _display_answer(answer) -> None:
    if answer.insufficient_context:
        print("\n[!] Contexto insuficiente para responder com base nos documentos.\n")
        return
    if answer.conflict:
        print("\n[!] Conflito detectado entre fontes.\n")
    print(f"\n{answer.text}\n")
    if answer.evidence:
        print("--- Fontes ---")
        for i, ev in enumerate(answer.evidence, 1):
            loc = f"{ev.source_path}"
            if ev.h1:
                loc += f" > {ev.h1}"
            if ev.h2:
                loc += f" > {ev.h2}"
            print(f"  [{i}] {loc}  (score={ev.score:.3f})")
        print()


def _parse_filters(raw: list[str]):
    """Converte lista de 'chave=valor' em FilterBuilder do Zvec.
    
    Valida as chaves permitidas para evitar erros ou SQL injection.
    Chaves permitidas: doc_id, source_path, language, doc_type, file_name
    """
    from pipeline.rag.retrieval.filters import FilterBuilder

    ALLOWED_KEYS = {"doc_id", "source_path", "language", "doc_type", "file_name"}
    
    fb = FilterBuilder()
    logger = logging.getLogger("rag-system")
    
    for f in raw:
        if "=" not in f:
            logger.warning("Filtro inválido (ignorado): '%s'. Use formato chave=valor", f)
            continue
            
        k, v = f.split("=", 1)
        k = k.strip()
        v = v.strip()
        
        if k not in ALLOWED_KEYS:
            logger.warning(
                "Chave de filtro desconhecida (ignorado): '%s'. "
                "Chaves permitidas: %s", k, ", ".join(sorted(ALLOWED_KEYS))
            )
            continue
            
        if not v:
            logger.warning("Valor vazio para filtro '%s' (ignorado)", k)
            continue
            
        fb.eq(k, v)
        
    return fb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG System — Pipeline completo local-first (OCR + RAG)",
    )
    sub = parser.add_subparsers(dest="command", help="Comando")

    # ---- ingest ----
    p_ingest = sub.add_parser("ingest", help="OCR + ingestão de documentos")
    p_ingest.add_argument("input", nargs="?", type=str, help="Arquivo ou diretório de entrada")
    p_ingest.add_argument("--ocr-only", action="store_true", help="Somente OCR (sem ingest RAG)")
    p_ingest.add_argument("--rag-only", action="store_true", help="Somente ingest RAG (Markdown já existe)")
    p_ingest.add_argument("--ocr-mode", type=str, help="hybrid | legacy | classic_only | ai_only")
    p_ingest.add_argument("--model", type=str, help="Modelo OCR: glm-ocr, chandra-ocr-2, granite-docling")
    p_ingest.add_argument("--quality-threshold", type=float, help="Score mínimo OCR (0-1)")
    p_ingest.add_argument("--ocr-langs", type=str, help="Idiomas Tesseract (ex: por+eng)")
    p_ingest.add_argument("--dpi", type=int, help="DPI de renderização")
    p_ingest.add_argument("--data-dir", type=str, help="Diretório de entrada (arquivos brutos)")
    p_ingest.add_argument("--markdown-dir", type=str, help="Diretório de saída .md")
    p_ingest.add_argument("--index-dir", type=str, help="Diretório do índice Zvec")
    p_ingest.add_argument("--provider", type=str, default="local", help="local | dummy")
    p_ingest.add_argument("--dimension", type=int, help="Dimensão dos embeddings")
    p_ingest.add_argument("--lmstudio-url", type=str, help="URL do LM Studio")
    p_ingest.add_argument("--lmstudio-model", type=str, help="Modelo no LM Studio")
    p_ingest.add_argument("--lmstudio-api-key", type=str, help="API key LM Studio")
    p_ingest.add_argument("--timeout", type=int, help="Timeout por requisição (s)")
    p_ingest.add_argument("--retries", type=int, default=3, help="Máximo de tentativas")
    p_ingest.add_argument("--resume", action="store_true", help="Retomar processamento parcial (usa .partial existentes)")

    # ---- query ----
    p_query = sub.add_parser("query", help="Consulta aos documentos indexados (requer LLM local)")
    p_query.add_argument("question", nargs="?", type=str, help="Pergunta")
    p_query.add_argument("--interactive", action="store_true", help="Modo interativo (REPL)")
    p_query.add_argument("--mode", type=str, help="answer | answer_with_citations | extractive_summary | study_mode")
    p_query.add_argument("--filter", action="append", default=[], help="Filtro chave=valor (ex: language=pt)")
    p_query.add_argument("--provider", type=str, default="local", help="local | dummy")
    p_query.add_argument("--dimension", type=int, help="Dimensão dos embeddings")
    p_query.add_argument("--llm-model", type=str, help="Modelo LLM (stub para offline)")
    p_query.add_argument("--retrieval-mode", type=str, help="dense | fts | hybrid | sparse")
    p_query.add_argument("--index-dir", type=str, help="Diretório do índice Zvec")
    p_query.add_argument("--markdown-dir", type=str, help="Diretório dos .md")

    # ---- retrieve (só busca, sem LLM — para uso com IA do chat) ----
    p_retrieve = sub.add_parser("retrieve", help="Apenas retrieval (sem LLM) — para uso com o assistente IA")
    p_retrieve.add_argument("question", nargs="?", type=str, help="Pergunta (opcional no modo interativo)")
    p_retrieve.add_argument("--interactive", action="store_true", help="Modo interativo (modelo carregado uma vez)")
    p_retrieve.add_argument("--filter", action="append", default=[], help="Filtro chave=valor")
    p_retrieve.add_argument("--provider", type=str, default="local", help="local | dummy")
    p_retrieve.add_argument("--dimension", type=int, help="Dimensão dos embeddings")
    p_retrieve.add_argument("--retrieval-mode", type=str, help="dense | fts | hybrid | sparse")
    p_retrieve.add_argument("--index-dir", type=str, help="Diretório do índice Zvec")
    p_retrieve.add_argument("--markdown-dir", type=str, help="Diretório dos .md")
    p_retrieve.add_argument("--top-k", type=int, help="Número de chunks a recuperar")

    # ---- retrieve-batch (multi-query eficiente) ----
    p_rbatch = sub.add_parser("retrieve-batch", help="Multiplas queries em um unico processo (modelo carregado 1x)")
    p_rbatch.add_argument("questions", nargs="+", type=str, help="Lista de perguntas")
    p_rbatch.add_argument("--filter", action="append", default=[], help="Filtro chave=valor")
    p_rbatch.add_argument("--provider", type=str, default="local", help="local | dummy")
    p_rbatch.add_argument("--dimension", type=int, help="Dimensao dos embeddings")
    p_rbatch.add_argument("--retrieval-mode", type=str, help="dense | fts | hybrid | sparse")
    p_rbatch.add_argument("--index-dir", type=str, help="Diretorio do indice Zvec")
    p_rbatch.add_argument("--markdown-dir", type=str, help="Diretorio dos .md")
    p_rbatch.add_argument("--top-k", type=int, help="Numero de chunks a recuperar")
    p_rbatch.add_argument("--max-chars", type=int, help="Limite de chars por snippet (default: config)")

    # ---- reindex ----
    p_reindex = sub.add_parser("reindex", help="Reindexação completa")
    p_reindex.add_argument("--purge", action="store_true", help="Remove collection Zvec antes de reindexar")
    p_reindex.add_argument("--provider", type=str, default="local", help="local | dummy")
    p_reindex.add_argument("--dimension", type=int, help="Dimensão dos embeddings")
    p_reindex.add_argument("--index-dir", type=str, help="Diretório do índice Zvec")
    p_reindex.add_argument("--markdown-dir", type=str, help="Diretório dos .md")

    # ---- run (one-shot) ----
    p_run = sub.add_parser("run", help="Pipeline completo: OCR + ingest + query")
    p_run.add_argument("input", type=str, help="Arquivo de entrada")
    p_run.add_argument("question", type=str, help="Pergunta")
    p_run.add_argument("--mode", type=str, help="answer | answer_with_citations | extractive_summary | study_mode")
    p_run.add_argument("--filter", action="append", default=[], help="Filtro chave=valor")
    p_run.add_argument("--ocr-mode", type=str, help="hybrid | legacy | classic_only | ai_only")
    p_run.add_argument("--model", type=str, help="Modelo OCR")
    p_run.add_argument("--quality-threshold", type=float, help="Score mínimo OCR")
    p_run.add_argument("--ocr-langs", type=str, help="Idiomas Tesseract")
    p_run.add_argument("--dpi", type=int, help="DPI de renderização")
    p_run.add_argument("--provider", type=str, default="local", help="local | dummy")
    p_run.add_argument("--dimension", type=int, help="Dimensão dos embeddings")
    p_run.add_argument("--llm-model", type=str, help="Modelo LLM")
    p_run.add_argument("--retrieval-mode", type=str, help="dense | fts | hybrid | sparse")
    p_run.add_argument("--lmstudio-url", type=str)
    p_run.add_argument("--lmstudio-model", type=str)
    p_run.add_argument("--lmstudio-api-key", type=str)
    p_run.add_argument("--timeout", type=int)
    p_run.add_argument("--retries", type=int, default=3)
    p_run.add_argument("--index-dir", type=str)
    p_run.add_argument("--data-dir", type=str)
    p_run.add_argument("--markdown-dir", type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    log_level = os.environ.get("RAG_LOG_LEVEL", "INFO")
    _setup_logging(log_level)
    logger = logging.getLogger("rag-system")

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "retrieve":
        cmd_retrieve(args)
    elif args.command == "retrieve-batch":
        cmd_retrieve_batch(args)
    elif args.command == "reindex":
        cmd_reindex(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        logger.error("Comando inválido. Use: ingest | query | retrieve | reindex | run")
        sys.exit(1)


if __name__ == "__main__":
    main()
