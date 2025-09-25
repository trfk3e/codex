"""Batch-only article generator using OpenAI's Batch API.

This script reads a keyword specification file, prepares one request per
article, submits them in a single batch job, and saves the generated
articles to disk. The implementation is intentionally single-threaded to
match the requirement of avoiding background worker pools.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from openai import OpenAI

DEFAULT_MODEL = "gpt-5-nano"
BATCH_API_COMPLETION_WINDOW = "24h"
BATCH_API_MAX_FILE_BYTES = 200 * 1024 * 1024
BYTES_IN_MEGABYTE = 1024 * 1024


@dataclass
class KeywordEntry:
    """Single keyword directive from the source file."""

    keyword: str
    quantity: int


@dataclass
class KeywordFileConfig:
    """All directives parsed from the keyword file."""

    language: str = "Русский"
    target_link: Optional[str] = None
    topic: Optional[str] = None
    entries: List[KeywordEntry] = field(default_factory=list)


@dataclass
class ArticleTask:
    """Concrete article generation job that maps 1:1 to a batch request."""

    custom_id: str
    keyword: str
    ordinal: int
    total_for_keyword: int
    language: str
    target_link: Optional[str]
    topic: Optional[str]
    output_path: Path


class KeywordFileParser:
    """Parse keyword specification files with optional directives."""

    KEYWORD_PATTERN = re.compile(r"\s*")

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def parse(self, file_path: Path) -> KeywordFileConfig:
        config = KeywordFileConfig()
        with file_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("!==+"):
                    config.target_link = line[4:].strip()
                    self.logger.info("Указана целевая ссылка: %s", config.target_link)
                    continue
                if line.startswith("!===+"):
                    config.language = line[5:].strip() or config.language
                    self.logger.info("Указан язык генерации: %s", config.language)
                    continue
                if line.startswith("!====+"):
                    config.topic = line[6:].strip() or None
                    self.logger.info("Дополнительный контекст темы: %s", config.topic)
                    continue

                keyword, quantity = self._parse_keyword_line(line)
                config.entries.append(KeywordEntry(keyword=keyword, quantity=quantity))
                self.logger.debug("Добавлено ключевое слово '%s' x%d", keyword, quantity)
        if not config.entries:
            raise ValueError("Файл ключевых слов не содержит задач для генерации")
        return config

    def _parse_keyword_line(self, line: str) -> tuple[str, int]:
        if "\t" in line:
            left, right = line.split("\t", 1)
        else:
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Строка '{line}' должна содержать ключевую фразу и количество через пробел или табуляцию")
            left = " ".join(parts[:-1])
            right = parts[-1]
        keyword = left.strip().strip("\"'")
        if not keyword:
            raise ValueError(f"Не удалось прочитать ключевую фразу из строки: {line}")
        try:
            quantity = int(right.strip())
        except ValueError as exc:
            raise ValueError(f"Количество должно быть целым числом в строке: {line}") from exc
        if quantity <= 0:
            raise ValueError(f"Количество должно быть положительным (строка: {line})")
        return keyword, quantity


def sanitize_filename(candidate: str, max_length: int = 80) -> str:
    candidate = candidate.replace("\n", " ")
    candidate = re.sub(r"[\\/*?:\"<>|]", "", candidate)
    candidate = re.sub(r"\s+", "_", candidate.strip())
    candidate = candidate[:max_length]
    return candidate or "article"


class BatchArticleGenerator:
    """Create and execute a single batch job for article generation."""

    def __init__(
        self,
        client: OpenAI,
        output_dir: Path,
        model: str = DEFAULT_MODEL,
        poll_interval: float = 5.0,
    ) -> None:
        self.client = client
        self.output_dir = output_dir
        self.model = model
        self.poll_interval = poll_interval
        self.logger = logging.getLogger("batch-generator")

    def build_tasks(self, config: KeywordFileConfig) -> List[ArticleTask]:
        tasks: List[ArticleTask] = []
        used_names: Dict[str, int] = {}
        for entry in config.entries:
            base_slug = sanitize_filename(entry.keyword.lower())
            if not base_slug:
                base_slug = "article"
            for idx in range(1, entry.quantity + 1):
                slug = base_slug
                if slug in used_names:
                    used_names[slug] += 1
                    slug = f"{slug}-{used_names[slug]}"
                else:
                    used_names[slug] = 0
                filename = self.output_dir / f"{slug}.html"
                custom_id = f"article-{slug}-{uuid.uuid4().hex}"
                tasks.append(
                    ArticleTask(
                        custom_id=custom_id,
                        keyword=entry.keyword,
                        ordinal=idx,
                        total_for_keyword=entry.quantity,
                        language=config.language,
                        target_link=config.target_link,
                        topic=config.topic,
                        output_path=filename,
                    )
                )
        self.logger.info("Подготовлено %d запросов для Batch API", len(tasks))
        return tasks

    def build_payload_lines(self, tasks: Iterable[ArticleTask]) -> tuple[List[str], int]:
        lines: List[str] = []
        total_bytes = 0
        for task in tasks:
            payload = self._request_payload(task)
            line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            encoded = line.encode("utf-8")
            projected_bytes = total_bytes + len(encoded) + 1  # newline
            if projected_bytes > BATCH_API_MAX_FILE_BYTES:
                raise ValueError(
                    "Размер JSONL превышает лимит 200 МБ Batch API. Уменьшите количество задач или объем инструкции."
                )
            total_bytes = projected_bytes
            lines.append(line)
        self.logger.info(
            "Общий размер JSONL: %.2f МБ (%d байт)", total_bytes / BYTES_IN_MEGABYTE, total_bytes
        )
        return lines, total_bytes

    def _request_payload(self, task: ArticleTask) -> Dict[str, object]:
        system_prompt = (
            "Ты профессиональный SEO-копирайтер. Пиши структурированные статьи в HTML. "
            "Отвечай строго валидным JSON с полями 'title' и 'body_html'."
        )
        requirements = [
            f"Ключевая фраза: '{task.keyword}'",
            f"Язык статьи: {task.language}",
            "Статья должна включать <h1> с заголовком, несколько разделов с <h2>/<h3>, списки и минимум одну таблицу, если это уместно.",
            "В конце добавь краткое заключение.",
        ]
        if task.topic:
            requirements.append(f"Дополнительный контекст или тематика: {task.topic}")
        if task.target_link:
            requirements.append(
                "Вставь одну ссылку <a> на указанную цель с естественным анкорным текстом в первой половине статьи."
            )
            requirements.append(f"URL для вставки: {task.target_link}")
        requirements.append(
            "Ответ должен быть JSON вида {\"title\": str, \"body_html\": str}. В body_html не используй <html>/<body>, только содержимое."  # noqa: E501
        )
        user_prompt = "\n".join(f"- {req}" for req in requirements)
        body: Dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Сгенерируй детальную статью. Дополнительные требования перечислены ниже:\n" + user_prompt
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        metadata = {
            "keyword": task.keyword,
            "ordinal": str(task.ordinal),
            "total_for_keyword": str(task.total_for_keyword),
            "filename": task.output_path.name,
        }
        return {
            "custom_id": task.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
            "metadata": metadata,
        }

    def execute(self, tasks: List[ArticleTask]) -> None:
        lines, _ = self.build_payload_lines(tasks)
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".jsonl") as tmp:
            for line in lines:
                tmp.write(line)
                tmp.write("\n")
            jsonl_path = Path(tmp.name)
        self.logger.info("JSONL файл подготовлен: %s", jsonl_path)

        upload = self.client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
        input_file_id = upload.id
        self.logger.info("Файл загружен: %s", input_file_id)

        batch = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=BATCH_API_COMPLETION_WINDOW,
            metadata={"total_requests": str(len(tasks))},
        )
        batch_id = batch.id
        self.logger.info("Batch задача создана: %s (статус %s)", batch_id, batch.status)

        try:
            final_batch = self._wait_for_completion(batch_id)
            output_file_id = final_batch.output_file_id
            if not output_file_id:
                raise RuntimeError("Batch завершился без output файла")
            output_lines = self._download_jsonl(output_file_id)
            self._process_results(output_lines, tasks)
            if final_batch.error_file_id:
                self._log_error_file(final_batch.error_file_id)
        finally:
            try:
                os.remove(jsonl_path)
            except OSError:
                pass
            try:
                self.client.files.delete(input_file_id)
            except Exception:
                self.logger.debug("Не удалось удалить входной файл %s", input_file_id)

    def _wait_for_completion(self, batch_id: str):
        self.logger.info("Ожидание завершения batch %s", batch_id)
        last_status = None
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            if status != last_status:
                self.logger.info("Статус batch %s: %s", batch_id, status)
                last_status = status
            if status == "completed":
                return batch
            if status in {"failed", "expired", "cancelled"}:
                raise RuntimeError(f"Batch {batch_id} завершился со статусом {status}")
            time.sleep(self.poll_interval)

    def _download_jsonl(self, file_id: str) -> List[Dict[str, object]]:
        self.logger.info("Загрузка результатов %s", file_id)
        response = self.client.files.content(file_id)
        if hasattr(response, "read"):
            raw = response.read()
            if isinstance(raw, (bytes, bytearray)):
                text = raw.decode("utf-8")
            else:
                text = str(raw)
        else:
            text = response.text if hasattr(response, "text") else str(response)
        entries: List[Dict[str, object]] = []
        for idx, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entries.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                self.logger.error("Ошибка разбора JSONL (строка %d): %s", idx, exc)
        self.logger.info("Получено %d ответов", len(entries))
        return entries

    def _process_results(self, entries: List[Dict[str, object]], tasks: List[ArticleTask]) -> None:
        tasks_by_id = {task.custom_id: task for task in tasks}
        success = 0
        for entry in entries:
            custom_id = entry.get("custom_id")
            task = tasks_by_id.get(custom_id)
            if not task:
                self.logger.warning("Получен ответ с неизвестным custom_id: %s", custom_id)
                continue
            response = entry.get("response") or {}
            status_code = response.get("status_code")
            if status_code != 200:
                self.logger.error(
                    "Запрос %s завершился с HTTP %s", custom_id, status_code
                )
                continue
            body = response.get("body") or {}
            choices = body.get("choices") or []
            if not choices:
                self.logger.error("Ответ для %s не содержит вариантов", custom_id)
                continue
            content = choices[0].get("message", {}).get("content")
            if not content:
                self.logger.error("Ответ для %s пуст", custom_id)
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                self.logger.error("Ответ для %s не является JSON: %s", custom_id, exc)
                continue
            title = payload.get("title") or task.keyword
            body_html = payload.get("body_html")
            if not body_html:
                self.logger.error("Ответ для %s не содержит body_html", custom_id)
                continue
            self._save_article(task, title, body_html)
            success += 1
        self.logger.info("Успешно сохранено статей: %d/%d", success, len(tasks))

    def _save_article(self, task: ArticleTask, title: str, body_html: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        html = self._compose_html_document(title, body_html)
        task.output_path.write_text(html, encoding="utf-8")
        self.logger.debug("Файл сохранен: %s", task.output_path)

    def _compose_html_document(self, title: str, body_html: str) -> str:
        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"{lang}\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\">\n"
            "  <title>{title}</title>\n"
            "</head>\n"
            "<body>\n"
            "  <h1>{title}</h1>\n"
            "  {body}\n"
            "</body>\n"
            "</html>\n"
        ).format(lang="ru", title=title, body=body_html)

    def _log_error_file(self, file_id: str) -> None:
        self.logger.warning("Batch содержит ошибки, выгружаем %s", file_id)
        entries = self._download_jsonl(file_id)
        for entry in entries:
            custom_id = entry.get("custom_id")
            error = entry.get("error") or {}
            self.logger.error("Ошибка в запросе %s: %s", custom_id, error)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Генерация статей через OpenAI Batch API")
    parser.add_argument("keywords_file", type=Path, help="Путь к файлу ключевых слов")
    parser.add_argument("output_dir", type=Path, help="Папка для сохранения статей")
    parser.add_argument("--api-key", dest="api_key", default=os.getenv("OPENAI_API_KEY"), help="API ключ OpenAI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Модель для генерации")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Интервал между проверками статуса batch")
    parser.add_argument("--verbose", action="store_true", help="Подробный лог")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    logger = logging.getLogger("main")

    if not args.api_key:
        logger.error("Не указан API ключ. Используйте --api-key или переменную OPENAI_API_KEY")
        return 1

    parser_obj = KeywordFileParser(logger)
    try:
        config = parser_obj.parse(args.keywords_file)
    except Exception as exc:
        logger.error("Ошибка разбора файла ключевых слов: %s", exc)
        return 1

    client = OpenAI(api_key=args.api_key, timeout=30, max_retries=0)
    generator = BatchArticleGenerator(client, args.output_dir, model=args.model, poll_interval=args.poll_interval)
    try:
        tasks = generator.build_tasks(config)
        generator.execute(tasks)
    except Exception as exc:
        logger.error("Ошибка генерации: %s", exc)
        return 1
    finally:
        try:
            client.close()
        except Exception:
            pass

    logger.info("Готово")
    return 0


if __name__ == "__main__":
    sys.exit(main())
