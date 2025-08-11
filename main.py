# check_openai_keys.py
"""Утилита для проверки списка API-ключей OpenAI.

Скрипт читает файл :data:`KEYS_FILE` (по одному ключу в строке),
вызывает лёгкий запрос к выбранной модели и собирает подробные
сведения об ответе. Для каждого ключа в журнал записывается максимум
полезной информации, что помогает понять причину отказа.

Создаются два файла:

* :data:`BAD_KEYS_FILE` — все ключи, по которым вызов завершился
  ошибкой.
* :data:`LOG_FILE` — расширенный лог с деталями по каждому ключу.

Скрипт максимально аккуратно обрабатывает каждую строку в исходном
файле, поддерживает как современный SDK ``openai>=1.0``, так и
наследованные версии ``0.x``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import pathlib
import time
from typing import List, Mapping, Sequence

# ---------------------------------------------------------------------------
#  OpenAI SDK compatibility layer
# ---------------------------------------------------------------------------
try:  # ``openai>=1.0``
    from openai import AsyncOpenAI, OpenAI, OpenAIError  # type: ignore
except ImportError:  # fallback для старого клиента 0.x
    import openai  # type: ignore

    class OpenAIError(openai.error.OpenAIError):  # type: ignore
        """Единый тип исключения для совместимости."""

    class _LegacyClient:  # адаптер под современный интерфейс
        def __init__(self, api_key: str):
            self.api_key = api_key

        def chat(self):  # noqa: D401 - совместимость API
            return self

        def completions(self):  # noqa: D401 - совместимость API
            return self

        def create(self, **kwargs):  # noqa: ANN001 - совместимость API
            return openai.ChatCompletion.create(api_key=self.api_key, **kwargs)

    def OpenAI(api_key: str, max_retries: int = 0):  # type: ignore  # noqa: N802
        return _LegacyClient(api_key)

    AsyncOpenAI = None  # в 0.x асинхронный клиент отсутствует

# ---------------------------------------------------------------------------
#  Конфигурация
# ---------------------------------------------------------------------------
KEYS_FILE = "KEYS.txt"
BAD_KEYS_FILE = "bad-api.txt"
LOG_FILE = "log.txt"
MODEL = "o3-mini"      # какую модель пингуем
TIMEOUT = 20           # секунд на запрос
PAUSE = 1              # пауза между запросами (для старых версий)
CONCURRENCY = 3        # одновременных запросов при поддержке async


# ---------------------------------------------------------------------------
#  Вспомогательные структуры
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class KeyResult:
    """Результат проверки одного API‑ключа."""

    key: str
    ok: bool
    status_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    model: str | None = None
    tokens: int | None = None
    request_id: str | None = None
    rate_limit_remaining: str | None = None
    retry_after: str | None = None
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
#  Утилиты
# ---------------------------------------------------------------------------
def load_keys(path: pathlib.Path) -> List[str]:
    """Читает ключи из файла, игнорируя пустые строки и пробелы."""

    if not path.exists():  # pragma: no cover - runtime guard
        raise FileNotFoundError(f"Не найден файл {path.absolute()}")

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_headers(headers: Mapping[str, str] | None) -> tuple[str | None, str | None, str | None]:
    """Возвращает ``(request_id, remaining, retry_after)`` из заголовков."""

    if not headers:
        return None, None, None

    request_id = headers.get("x-request-id")
    remaining = headers.get("x-ratelimit-remaining-requests") or headers.get("x-ratelimit-remaining")
    retry_after = headers.get("retry-after")
    return request_id, remaining, retry_after


# ---------------------------------------------------------------------------
#  Проверка ключа (sync)
# ---------------------------------------------------------------------------
def _check_key_sync(key: str) -> KeyResult:
    client = OpenAI(api_key=key, max_retries=0)
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Привет"}],
            timeout=TIMEOUT,
        )
        elapsed = time.perf_counter() - start
        headers = getattr(resp, "response", None)
        headers = getattr(headers, "headers", None)
        request_id, remaining, retry_after = _parse_headers(headers)
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "total_tokens", None)
        model = getattr(resp, "model", None)
        return KeyResult(
            key=key,
            ok=True,
            status_code=200,
            model=model,
            tokens=tokens,
            request_id=request_id,
            rate_limit_remaining=remaining,
            retry_after=retry_after,
            elapsed=elapsed,
        )
    except OpenAIError as exc:  # ошибки SDK
        elapsed = time.perf_counter() - start
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        request_id, remaining, retry_after = _parse_headers(headers)
        status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
        return KeyResult(
            key=key,
            ok=False,
            status_code=status,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            request_id=request_id,
            rate_limit_remaining=remaining,
            retry_after=retry_after,
            elapsed=elapsed,
        )
    except Exception as exc:  # прочие сбои (сеть и пр.)
        elapsed = time.perf_counter() - start
        return KeyResult(
            key=key,
            ok=False,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            elapsed=elapsed,
        )


# ---------------------------------------------------------------------------
#  Проверка ключа (async)
# ---------------------------------------------------------------------------
async def _check_key_async(key: str, sem: asyncio.Semaphore) -> KeyResult:
    async with sem:
        client = AsyncOpenAI(api_key=key, max_retries=0)  # type: ignore[arg-type]
        start = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Привет"}],
                timeout=TIMEOUT,
            )
            elapsed = time.perf_counter() - start
            headers = getattr(resp, "response", None)
            headers = getattr(headers, "headers", None)
            request_id, remaining, retry_after = _parse_headers(headers)
            usage = getattr(resp, "usage", None)
            tokens = getattr(usage, "total_tokens", None)
            model = getattr(resp, "model", None)
            return KeyResult(
                key=key,
                ok=True,
                status_code=200,
                model=model,
                tokens=tokens,
                request_id=request_id,
                rate_limit_remaining=remaining,
                retry_after=retry_after,
                elapsed=elapsed,
            )
        except OpenAIError as exc:  # ошибки SDK
            elapsed = time.perf_counter() - start
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", None)
            request_id, remaining, retry_after = _parse_headers(headers)
            status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
            return KeyResult(
                key=key,
                ok=False,
                status_code=status,
                error_type=exc.__class__.__name__,
                error_message=str(exc),
                request_id=request_id,
                rate_limit_remaining=remaining,
                retry_after=retry_after,
                elapsed=elapsed,
            )
        except Exception as exc:  # прочие сбои
            elapsed = time.perf_counter() - start
            return KeyResult(
                key=key,
                ok=False,
                error_type=exc.__class__.__name__,
                error_message=str(exc),
                elapsed=elapsed,
            )


# ---------------------------------------------------------------------------
#  Массовая проверка (async)
# ---------------------------------------------------------------------------
async def _check_all_async(keys: Sequence[str]) -> List[KeyResult]:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def runner(idx: int, key: str) -> tuple[int, KeyResult]:
        return idx, await _check_key_async(key, sem)

    tasks = [asyncio.create_task(runner(i, k)) for i, k in enumerate(keys)]
    total = len(tasks)
    results: list[KeyResult | None] = [None] * total
    for done, fut in enumerate(asyncio.as_completed(tasks), 1):
        idx, res = await fut
        results[idx] = res
        status = "✅" if res.ok else "❌"
        print(f"[{done}/{total}] {status} {res.error_type or 'OK'}")
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
#  Сводка и вывод
# ---------------------------------------------------------------------------
def _format_log(res: KeyResult) -> str:
    if res.ok:
        return (
            f"{res.key} = OK model={res.model} tokens={res.tokens} "
            f"request_id={res.request_id} remaining={res.rate_limit_remaining}"
        )
    return (
        f"{res.key} = {res.error_type} (status {res.status_code}) "
        f"{res.error_message} retry_after={res.retry_after}"
    )


def _write_outputs(results: Sequence[KeyResult]) -> None:
    bad_keys = [r.key for r in results if not r.ok]
    logs = [_format_log(r) for r in results]
    pathlib.Path(BAD_KEYS_FILE).write_text("\n".join(bad_keys), encoding="utf-8")
    pathlib.Path(LOG_FILE).write_text("\n".join(logs), encoding="utf-8")


# ---------------------------------------------------------------------------
#  Главная функция
# ---------------------------------------------------------------------------
def main() -> None:  # pragma: no cover - CLI точка входа
    keys = load_keys(pathlib.Path(KEYS_FILE))
    total = len(keys)
    print(f"Найдено {total} ключей. Проверяем…\n")

    if AsyncOpenAI is not None:  # асинхронная проверка
        results = asyncio.run(_check_all_async(keys))
    else:  # синхронная проверка (старый SDK)
        results = []
        for idx, key in enumerate(keys, 1):
            res = _check_key_sync(key)
            results.append(res)
            status = "✅" if res.ok else "❌"
            print(f"[{idx}/{total}] {status} {res.error_type or 'OK'}")
            time.sleep(PAUSE)

    _write_outputs(results)

    failed = sum(not r.ok for r in results)
    print("\n— Готово —")
    print(f"Не прошло: {failed} из {total} (см. {BAD_KEYS_FILE} и {LOG_FILE})")


if __name__ == "__main__":  # pragma: no cover
    main()

