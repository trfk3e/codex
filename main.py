# bot.py ────────────────────────────────────────────────────────────────────
import json, re, html
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import logging
import threading
try:
    import fcntl  # POSIX locking
except ModuleNotFoundError:  # pragma: no cover - Windows fallback
    fcntl = None
    import msvcrt
from contextlib import contextmanager
from collections import deque
from dataclasses import dataclass, field

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    constants as tg_const,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from telethon import TelegramClient
from telethon.tl.types import Message
from openai import OpenAI, AuthenticationError

# ────────────── КОНФИГ ─────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
def _parse_admin_ids(raw: str) -> set[int]:
    ids: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.add(int(chunk))
        except ValueError:
            logging.warning("Пропускаю некорректный идентификатор в ADMIN_IDS: %s", chunk)
    return ids


ADMIN_IDS = _parse_admin_ids(os.getenv("ADMIN_IDS", ""))
TELEGRAM_BOT_TOKEN   = "7621000604:AAHrWFyNx8JCrPkCtmtC4MWAV2Ri5-EpOQo"
TG_API_ID     = "28511990"
TG_API_HASH = "f51873d6f1402467b6188a37100754ae"


PROCESSED_FILE = Path("processed_ids.json")
API_DB_PATH = Path("openai_keys.sqlite3")
PROCESSED_LIMIT = 1000
CONCURRENCY = 5
MODEL_NAME     = "o3-mini"
ATTEMPT_LIMIT  = 50
ATTEMPT_MSG    = (
    "После 50 попыток все ответы AI были - NO, повторите ещё раз, "
    "поменяйте промпт, или выберите другой канал"
)

# ────────────── ГЛОБАЛЬНЫЕ КЛИЕНТЫ ─────────────────────────────────────────
class OpenAIConfigError(RuntimeError):
    """Raised when OpenAI configuration is missing or invalid."""


class APIKeyManager:
    """Manage OpenAI API keys stored in SQLite with automatic rotation."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS openai_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()
        self._keys: list[tuple[int, str]] = []
        self._index_map: dict[int, int] = {}
        self._clients: dict[int, OpenAI] = {}
        self._current_index = 0
        self._cycle_count = 0
        self._reload_locked()

    def _reload_locked(self) -> None:
        prev_id: int | None = None
        if self._keys and 0 <= self._current_index < len(self._keys):
            prev_id = self._keys[self._current_index][0]

        cur = self._conn.execute(
            "SELECT id, api_key FROM openai_keys ORDER BY id"
        )
        rows = cur.fetchall()
        existing_clients = self._clients
        self._keys = rows
        self._index_map = {row[0]: idx for idx, row in enumerate(rows)}
        self._clients = {
            key_id: existing_clients[key_id]
            for key_id, _ in rows
            if key_id in existing_clients
        }

        if prev_id is not None and prev_id in self._index_map:
            self._current_index = self._index_map[prev_id]
        else:
            self._current_index = 0
        if not self._keys:
            self._current_index = 0
        self._cycle_count = 0

    def _raise_empty(self) -> None:
        raise OpenAIConfigError(
            "Нет сохранённых OpenAI API ключей. Добавьте ключи через меню “Меню API ключей”."
        )

    def _advance_after_failure(self, failed_index: int, key_id: int) -> None:
        if not self._keys:
            self._raise_empty()

        self._clients.pop(key_id, None)
        if not self._keys:
            self._current_index = 0
            return

        next_index = (failed_index + 1) % len(self._keys)
        wrapped = next_index <= failed_index if self._keys else False
        self._current_index = next_index
        if wrapped:
            self._cycle_count += 1
            if self._cycle_count >= 2:
                raise OpenAIConfigError(
                    "Все OpenAI API ключи дважды отклонены. Очистите список и добавьте новые ключи через меню “Меню API ключей”."
                )

    def has_keys(self) -> bool:
        with self._lock:
            return bool(self._keys)

    def count(self) -> int:
        with self._lock:
            return len(self._keys)

    def masked_keys(self) -> list[str]:
        with self._lock:
            masked: list[str] = []
            for _, key in self._keys:
                if len(key) <= 12:
                    masked.append(key)
                else:
                    masked.append(f"{key[:8]}…{key[-4:]}")
            return masked

    def add_keys(self, keys: list[str]) -> list[str]:
        cleaned = [k.strip() for k in keys if k.strip()]
        if not cleaned:
            return []
        added: list[str] = []
        with self._lock:
            for key in cleaned:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO openai_keys(api_key) VALUES (?)",
                    (key,),
                )
                if cur.rowcount:
                    added.append(key)
            self._conn.commit()
            if added:
                self._reload_locked()
        return added

    def clear_keys(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM openai_keys")
            self._conn.commit()
            self._clients.clear()
            self._keys = []
            self._index_map = {}
            self._current_index = 0
            self._cycle_count = 0

    def acquire_client(self) -> tuple[int, int, OpenAI]:
        with self._lock:
            if not self._keys:
                self._raise_empty()

            attempts = 0
            total = len(self._keys)
            while attempts < total:
                idx = self._current_index
                key_id, key_value = self._keys[idx]
                client = self._clients.get(key_id)
                if client is None:
                    try:
                        client = OpenAI(api_key=key_value)
                    except AuthenticationError:
                        self._advance_after_failure(idx, key_id)
                        attempts += 1
                        continue
                    self._clients[key_id] = client
                return idx, key_id, client
            raise OpenAIConfigError(
                "OpenAI API ключи недействительны. Обновите список ключей через меню “Меню API ключей”."
            )

    def report_failure(self, index: int) -> None:
        with self._lock:
            if not self._keys:
                self._raise_empty()
            if index >= len(self._keys):
                if self._keys:
                    self._current_index = self._current_index % len(self._keys)
                return
            key_id, _ = self._keys[index]
            self._advance_after_failure(index, key_id)

    def report_success(self, index: int) -> None:
        with self._lock:
            if not self._keys:
                self._current_index = 0
                self._cycle_count = 0
                return
            if index >= len(self._keys):
                index = 0
            self._current_index = index
            self._cycle_count = 0


key_manager = APIKeyManager(API_DB_PATH)

if OPENAI_API_KEY:
    added_env = key_manager.add_keys([OPENAI_API_KEY])
    if added_env:
        logging.info(
            "Добавлен OpenAI API ключ из переменной окружения. Управляйте ключами через меню “Меню API ключей”."
        )

if not key_manager.has_keys():
    logging.warning(
        "Не найдены OpenAI API ключи. Добавьте их через меню “Меню API ключей” или переменную окружения OPENAI_API_KEY."
    )
tg_client     = TelegramClient(
    "seo_news_session", TG_API_ID, TG_API_HASH, timeout=10
)

async def openai_call(factory, *, timeout=60):
    """Выполнить OpenAI запрос с автоматическим перебором ключей."""

    loop = asyncio.get_running_loop()
    while True:
        idx, _key_id, client = key_manager.acquire_client()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: factory(client)), timeout
            )
        except AuthenticationError:
            key_manager.report_failure(idx)
            continue
        except OpenAIConfigError:
            raise
        except Exception:
            # неудачи OpenAI не связанные с ключами не переключают ключ
            raise
        else:
            key_manager.report_success(idx)
            return result

@contextmanager
def file_lock():
    lock = PROCESSED_FILE.with_suffix(".lock")
    if fcntl is not None:
        with open(lock, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    else:  # Windows fallback using msvcrt
        with open(lock, "a") as lf:
            try:
                msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
                yield
            finally:
                try:
                    lf.seek(0)
                    msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass

# Defaults for the filter prompt are defined before the dataclass so the
# attributes can use them directly without a NameError.
DEFAULT_PROMPT_YES = (
    "текст действительно полезен (кейсы, стратегии, апдейты, арбитраж…)"
)
DEFAULT_PROMPT_NO = "реклама, эфир, подкаст, мерч, вакансии, мем, оффтоп."


@dataclass
class ChatConfig:
    channels: list[str] = field(default_factory=list)
    ids: dict[str, deque] = field(default_factory=dict)
    auto_last: dict[str, int] = field(default_factory=dict)
    prompt_yes: str = DEFAULT_PROMPT_YES
    prompt_no: str = DEFAULT_PROMPT_NO
    log_enabled: bool = False

    def filter_prompt(self) -> str:
        return build_filter_prompt(self.prompt_yes, self.prompt_no)

ALL_CHATS: dict[str, ChatConfig] = {}

async def log(ctx: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    cfg = get_cfg(ctx)
    if cfg.log_enabled and ctx.chat_data.get("target_chat"):
        try:
            await ctx.bot.send_message(ctx.chat_data["target_chat"], f"#log {text}")
        except Exception:
            logging.exception("Failed to send log message")


# ────────────── UTIL: журнал постов ────────────────────────────────────────

def build_filter_prompt(p_yes: str, p_no: str) -> str:
    return (
        "Ты эксперт по SEO и iGaming-маркетингу. "
        "Отвечай только 'yes' или 'no'. "
        f"'Yes' — {p_yes} "
        f"'No' — {p_no}"
    )

def load_data() -> dict[str, ChatConfig]:
    """Load per-chat settings from ``processed_ids.json``."""

    data: dict[str, ChatConfig] = {}

    if PROCESSED_FILE.exists():
        try:
            with file_lock():
                with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
        except json.JSONDecodeError:
            raw = {}
        except Exception:
            logging.exception("Failed to read data")
            raw = {}
    else:
        raw = {}

    if isinstance(raw, dict) and "chats" in raw:
        items = raw.get("chats", {})
    else:
        # backwards compatibility with old structure
        items = {"default": raw}

    for chat_id, cfg in items.items():
        if not isinstance(cfg, dict):
            continue
        chs = [str(c).split(":")[0].lstrip("@") for c in cfg.get("channels", [])]
        ids = {
            c: deque(map(int, v), maxlen=PROCESSED_LIMIT)
            for c, v in cfg.get("ids", {}).items()
        }
        raw_auto = cfg.get("auto_last", {})
        auto_last: dict[str, int] = {}
        if isinstance(raw_auto, dict):
            for c, v in raw_auto.items():
                key = str(c).split(":")[0].lstrip("@")
                try:
                    auto_last[key] = int(v)
                except (TypeError, ValueError):
                    continue
        p_yes = cfg.get("prompt_if_yes", DEFAULT_PROMPT_YES)
        p_no = cfg.get("prompt_if_no", DEFAULT_PROMPT_NO)
        log_en = bool(cfg.get("log_enabled", False))
        data[str(chat_id)] = ChatConfig(
            channels=chs,
            ids=ids,
            auto_last=auto_last,
            prompt_yes=p_yes,
            prompt_no=p_no,
            log_enabled=log_en,
        )

    return data

def save_all() -> None:
    """Persist ``ALL_CHATS`` to disk."""
    data: dict[str, dict] = {}
    for cid, cfg in ALL_CHATS.items():
        # keep channel list unique and sanitized but preserve order
        seen: set[str] = set()
        uniq: list[str] = []
        for c in cfg.channels:
            c = str(c).split(":")[0].lstrip("@")
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        cfg.channels = uniq
        auto_last = {}
        for c in cfg.auto_last:
            key = str(c).split(":")[0].lstrip("@")
            if key in cfg.channels:
                try:
                    auto_last[key] = int(cfg.auto_last[c])
                except (TypeError, ValueError):
                    continue
        data[cid] = {
            "channels": cfg.channels,
            "ids": {c: list(v) for c, v in cfg.ids.items()},
            "auto_last": auto_last,
            "prompt_if_yes": cfg.prompt_yes,
            "prompt_if_no": cfg.prompt_no,
            "log_enabled": cfg.log_enabled,
        }

    tmp = PROCESSED_FILE.with_suffix(".tmp")
    with file_lock():
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"chats": data}, f, ensure_ascii=False, indent=2)
        tmp.replace(PROCESSED_FILE)

ALL_CHATS = load_data()

def get_cfg(ctx: ContextTypes.DEFAULT_TYPE) -> ChatConfig:
    chat_id = str(ctx.chat_data.get("chat_id", ctx.chat_data.get("target_chat")))
    if chat_id not in ALL_CHATS:
        ALL_CHATS[chat_id] = ChatConfig()
    return ALL_CHATS[chat_id]


def mark_processed(cfg: ChatConfig, key: str, msg_id: int) -> bool:
    """Запомнить ID обработанного поста и обновить последний авто-порог."""

    seen = cfg.ids.setdefault(key, deque(maxlen=PROCESSED_LIMIT))
    already = msg_id in seen
    changed = False
    if not already:
        seen.append(msg_id)
        changed = True
    try:
        numeric_id = int(msg_id)
    except (TypeError, ValueError):
        numeric_id = 0
    if numeric_id and numeric_id > cfg.auto_last.get(key, 0):
        cfg.auto_last[key] = numeric_id
        changed = True
    return changed


def task_running(ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    t = ctx.chat_data.get("task")
    return bool(t) and not t.done()


async def stop_auto_cycle(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    wait: bool = False,
    timeout: float = 5.0,
) -> None:
    """Request graceful shutdown of the авто-выжимка loop.

    The helper sets the common stop flags and optionally waits for the
    background task to finish after sending it a cancellation signal.
    """

    ctx.chat_data["stop"] = True
    ctx.chat_data["auto_stop"] = True

    task = ctx.chat_data.get("auto_task")
    if not task or task.done():
        return

    task.cancel()

    if not wait:
        return

    try:
        await asyncio.wait_for(task, timeout)
    except asyncio.TimeoutError:
        logging.warning("Не удалось остановить авто-выжимку за %s сек", timeout)
    except asyncio.CancelledError:
        pass
    except Exception:
        logging.exception("Ошибка при ожидании остановки авто-выжимки")

def launch_task(ctx: ContextTypes.DEFAULT_TYPE, coro) -> None:
    task = ctx.application.create_task(coro)
    ctx.chat_data["task"] = task
    task.add_done_callback(lambda _: ctx.chat_data.pop("task", None))


async def clear_history(
    ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, last_id: int, prog_id: int | None = None
) -> None:
    """Delete all messages in the chat using the bot account."""

    start_id = ctx.chat_data.get("start_id", last_id)
    for msg_id in range(last_id, start_id - 1, -1):
        try:
            await ctx.bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except Exception:
            logging.exception("Failed to delete message")

    ctx.chat_data.pop("start_id", None)

    if prog_id:
        try:
            await ctx.bot.delete_message(chat_id=chat_id, message_id=prog_id)
        except Exception:
            logging.exception("Failed to delete progress message")

# ────────────── AI helpers ─────────────────────────────────────────────────

async def ai_check(cfg: ChatConfig, text: str) -> tuple[bool, str | None]:
    """Return ``(True, None)`` if the text is relevant.

    If the answer is "NO" and logging is enabled, also request a short
    explanation and return it as the second tuple element. When logging is
    disabled, the additional request is skipped to save tokens.
    """

    base = [
        {"role": "system", "content": cfg.filter_prompt()},
        {"role": "user", "content": text[:4000]},
    ]
    rsp = await openai_call(
        lambda client: client.chat.completions.create(
            model=MODEL_NAME, messages=base
        )
    )
    answer = rsp.choices[0].message.content.strip()
    ok = answer.lower().startswith("y")
    reason = None

    if not ok and cfg.log_enabled:
        rsp2 = await openai_call(
            lambda client: client.chat.completions.create(
                model=MODEL_NAME,
                messages=base
                + [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": "Кратко объясни почему NO"},
                ],
            )
        )
        reason = rsp2.choices[0].message.content.strip()

    return ok, reason

async def paraphrase(text: str) -> str:
    rsp = await openai_call(
        lambda client: client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты русскоязычный SEO-журналист. Перепиши текст лёгким "
                        "рерайтом, сохрани факты."
                    ),
                },
                {"role": "user", "content": text.strip()},
            ],
        )
    )
    return rsp.choices[0].message.content.strip()


async def build_digest_payload(text: str) -> dict:
    """Сформировать компактную выжимку поста через OpenAI."""

    instruction = (
        "Ты работаешь редактором Telegram-канала про SEO и маркетинг. "
        "Проанализируй оригинальную публикацию и составь короткий анонс "
        "в деловом стиле. Выдели главную мысль и конкретные полезные факты. "
        "Ответ верни в строгом JSON-формате со следующими полями: "
        "emoji (один подходящий эмодзи), title (до 100 символов), "
        "summary (2–3 предложения с ключевыми фактами), "
        "highlights (список из 0–4 коротких выводов или рекомендаций). "
        "Не добавляй никакого текста вне JSON."
    )

    rsp = await openai_call(
        lambda client: client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": text.strip()},
            ],
        )
    )
    raw = rsp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logging.warning("Не удалось разобрать JSON от OpenAI, используем запасной формат")
        data = {
            "emoji": "📝",
            "title": raw.splitlines()[0][:100] if raw else "Новая публикация",
            "summary": raw,
            "highlights": [],
        }

    if not isinstance(data, dict):
        data = {
            "emoji": "📝",
            "title": "Новая публикация",
            "summary": raw,
            "highlights": [],
        }

    data.setdefault("emoji", "📝")
    data.setdefault("title", "Новая публикация")
    data.setdefault("summary", "")
    hl = data.get("highlights")
    if not isinstance(hl, list):
        hl = []
    data["highlights"] = [str(item) for item in hl[:4]]
    return data


def render_digest(data: dict, source: str, link: str) -> str:
    emoji = html.escape(str(data.get("emoji", "📝")))
    title = html.escape(str(data.get("title", "Новая публикация")))
    summary = html.escape(str(data.get("summary", "")))
    highlights = [html.escape(str(item)) for item in data.get("highlights", []) if item]

    parts: list[str] = []
    parts.append(f"{emoji} <b>{title}</b>")
    if summary:
        parts.append("")
        parts.append(summary)
    if highlights:
        parts.append("")
        parts.append("\n".join(f"• {item}" for item in highlights))

    src = html.escape(source)
    if link:
        link_attr = html.escape(link)
        source_line = f"Источник: <a href='{link_attr}'>{src}</a>"
    else:
        source_line = f"Источник: {src}"
    parts.append("")
    parts.append(source_line)
    body = "\n".join(part for part in parts if part is not None)
    return body[:4090]

# ────────────── TELETHON helpers ───────────────────────────────────────────
def quick_score(m: Message) -> int:
    return (m.views or 0) + (m.forwards or 0) * 10

async def fetch_posts(
    channel: str | int,
    from_dt: datetime | None,
    to_dt: datetime | None,
    limit: int | None = None,
    by_popularity: bool = False,
) -> list[Message]:
    async def collect(target) -> list[Message]:
        collected: list[Message] = []
        iter_kwargs: dict[str, int] = {}
        if limit:
            # Берём небольшой запас, чтобы после фильтрации по дате осталось
            # достаточно сообщений.
            iter_kwargs["limit"] = max(limit * 2, 50)
        try:
            async for msg in tg_client.iter_messages(target, **iter_kwargs):
                if not msg.text:
                    continue
                dt = msg.date.replace(tzinfo=None)
                if to_dt and dt > to_dt:
                    continue
                if from_dt and dt < from_dt:
                    # iter_messages возвращает посты от новых к старым, поэтому
                    # можно остановиться, когда ушли ниже диапазона.
                    break
                collected.append(msg)
                if limit and len(collected) >= limit:
                    break
        except Exception:
            raise
        return collected

    try:
        msgs = await collect(channel)
    except ValueError:
        if isinstance(channel, str) and channel.lstrip("-").isdigit():
            msgs = await collect(int(channel))
        else:
            raise
    except Exception:
        logging.exception("Failed to fetch posts")
        return []

    if by_popularity:
        msgs.sort(key=quick_score, reverse=True)
    else:
        msgs.sort(key=lambda m: m.date)
    if limit and not by_popularity:
        msgs = msgs[-limit:]
    return msgs

async def process_and_send(
    ctx: ContextTypes.DEFAULT_TYPE,
    msg: Message,
    chan: str,
    *,
    mode: str | None = None,
    notify: bool = True,
):
    if ctx.chat_data.get("stop"):
        return False
    cfg = get_cfg(ctx)
    key = chan.lstrip("@")
    seen = cfg.ids.setdefault(key, deque(maxlen=PROCESSED_LIMIT))
    if msg.id in seen:
        await log(ctx, f"Пропускаю {msg.id}: уже обработан")
        return False
    await log(ctx, f"Проверяю пост {msg.id} из {chan}")
    try:
        ai_ok, reason = await ai_check(cfg, msg.text)
    except asyncio.CancelledError:
        raise
    except OpenAIConfigError as exc:
        await log(ctx, str(exc))
        mark_processed(cfg, key, msg.id)
        save_all()
        return False
    except Exception:
        logging.exception("AI filter failed")
        await log(ctx, f"AI ошибка при обработке {msg.id}")
        mark_processed(cfg, key, msg.id)
        save_all()
        return False
    if ctx.chat_data.get("stop"):
        return False
    await log(ctx, f"AI ответ для {msg.id}: {'YES' if ai_ok else 'NO'}")
    if not ai_ok:
        if reason:
            await log(ctx, reason)
        mark_processed(cfg, key, msg.id)
        save_all()
        return False
    if ctx.chat_data.get("stop"):
        return False
    mode = mode or ctx.chat_data.get("output_mode", "rewrite")
    username = getattr(msg.chat, "username", None) or chan.lstrip("@")
    link = (
        f"https://t.me/{username}/{msg.id}" if username and not username.lstrip("-").isdigit() else ""
    )
    raw_src = f"@{username}" if username and not username.lstrip("-").isdigit() else chan
    escaped_src = html.escape(raw_src)
    parse_mode = tg_const.ParseMode.HTML
    disable_preview = True

    if mode == "digest":
        await log(ctx, f"Формируем выжимку для поста {msg.id}")
        if ctx.chat_data.get("stop"):
            return False
        try:
            digest = await build_digest_payload(msg.text)
        except asyncio.CancelledError:
            raise
        except OpenAIConfigError as exc:
            await log(ctx, str(exc))
            mark_processed(cfg, key, msg.id)
            save_all()
            return False
        except Exception:
            logging.exception("Failed to build digest")
            await log(ctx, f"Не удалось построить выжимку для {msg.id}")
            mark_processed(cfg, key, msg.id)
            save_all()
            return False
        if ctx.chat_data.get("stop"):
            return False
        body = render_digest(digest, raw_src, link)
    else:
        await log(ctx, f"Перефразируем пост {msg.id}")
        if ctx.chat_data.get("stop"):
            return False
        try:
            rewritten = html.escape(await paraphrase(msg.text))
        except asyncio.CancelledError:
            raise
        except OpenAIConfigError as exc:
            await log(ctx, str(exc))
            mark_processed(cfg, key, msg.id)
            save_all()
            return False
        except Exception:
            logging.exception("Failed to paraphrase message")
            await log(ctx, f"Не удалось перефразировать {msg.id}")
            mark_processed(cfg, key, msg.id)
            save_all()
            return False
        if ctx.chat_data.get("stop"):
            return False
        footer = (
            f"\n\n<b>Дата публикации:</b> {msg.date.strftime('%d.%m.%Y %H:%M')} "
            f"| <b>Просмотров:</b> {msg.views or 0}"
        )
        link_attr = f" href='{html.escape(link)}'" if link else ""
        body = (
            f"<b>Источник:</b> <a{link_attr}>{escaped_src}</a>\n\n"
            f"{rewritten}{footer}"
        )[:4090]
    try:
        await ctx.bot.send_message(
            chat_id=ctx.chat_data["target_chat"],
            text=body,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_preview,
        )
    except Exception:
        logging.exception("Failed to send processed message to target chat")
        await log(ctx, f"Отправка сообщения {msg.id} завершилась ошибкой")
        mark_processed(cfg, key, msg.id)
        save_all()
        return False
    await log(ctx, f"Отправлено сообщение {msg.id}")
    if ctx.chat_data.get("stop"):
        return True
    log_count = ctx.chat_data.get("sent", 0) + 1
    ctx.chat_data["sent"] = log_count
    if notify:
        try:
            await ctx.bot.send_message(
                chat_id=ctx.chat_data["target_chat"],
                text=f"✅ Сообщение {log_count} из канала {chan} отправлено",
            )
        except Exception:
            logging.exception("Failed to send confirmation message")
    mark_processed(cfg, key, msg.id)
    save_all()
    return True


async def send_filtered_posts(
    ctx: ContextTypes.DEFAULT_TYPE,
    chan: str,
    posts: list[Message],
    need: int,
    *,
    mode: str | None = None,
    notify: bool = True,
) -> tuple[int, int]:
    """Send posts that pass the AI filter until ``need`` is reached.

    Returns a tuple ``(sent, attempts)`` where ``attempts`` counts how many
    messages were checked by the AI.
    """
    sent = 0
    attempts = 0
    cfg = get_cfg(ctx)
    key = chan.lstrip("@")
    seen = cfg.ids.setdefault(key, deque(maxlen=PROCESSED_LIMIT))
    if mode is None:
        mode = ctx.chat_data.get("output_mode", "rewrite")
    await log(ctx, f"Начинаю проверку {len(posts)} постов из {chan}")
    sem = asyncio.Semaphore(CONCURRENCY)
    async def worker(m: Message):
        nonlocal sent, attempts
        if ctx.chat_data.get("stop") or (sent >= need and need):
            return
        if m.id in seen:
            return
        async with sem:
            attempts += 1
            if await process_and_send(ctx, m, chan, mode=mode, notify=notify):
                sent += 1
    tasks = [asyncio.create_task(worker(m)) for m in posts]
    if tasks:
        for t in asyncio.as_completed(tasks):
            try:
                await t
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("send_filtered_posts worker failed")
                await log(ctx, "Ошибка в рабочем таске, продолжаю")
            if ctx.chat_data.get("stop") or (
                sent >= need and need
            ) or (attempts >= ATTEMPT_LIMIT and sent == 0):
                for x in tasks:
                    x.cancel()
                break
        await asyncio.gather(*tasks, return_exceptions=True)
    else:
        await log(ctx, f"Нет новых постов для обработки в {chan}")
    await log(ctx, f"Закончена проверка {chan}: отправлено {sent} из {attempts}")
    return sent, attempts

# ────────────── TELEGRAM BOT HANDLERS ──────────────────────────────────────
BASE_MENU_ROWS = [
    ["🔍 Парсить конкретный канал", "➡️ По очереди все каналы"],
    ["⭐ Популярные посты всех каналов"],
    ["⭐ Популярные посты конкретного канала"],
    ["Поиск постов за последние дни в конкретном канале"],
    ["Поиск постов за последние дни во всех каналах"],
    ["📅 Диапазон дат (канал)", "📅 Диапазон дат (все)"],
    ["➕ Добавить каналы", "➖ Удалить каналы"],
    ["📰 Авто-выжимка"],
    ["Настройки"],
    ["Очистить чат"],
    ["⏹ Остановить"],
    ["ℹ️ Инструкция"],
]


def _clone_rows(rows: list[list[str]]) -> list[list[str]]:
    return [row[:] for row in rows]


ADMIN_MENU_ROWS = _clone_rows(BASE_MENU_ROWS)
try:
    settings_index = next(
        idx for idx, row in enumerate(ADMIN_MENU_ROWS) if "Настройки" in row
    )
except StopIteration:
    settings_index = len(ADMIN_MENU_ROWS) - 1
ADMIN_MENU_ROWS.insert(settings_index + 1, ["Меню API ключей"])

USER_MAIN_KB = ReplyKeyboardMarkup(BASE_MENU_ROWS, resize_keyboard=True)
ADMIN_MAIN_KB = ReplyKeyboardMarkup(ADMIN_MENU_ROWS, resize_keyboard=True)

SETTINGS_KB = ReplyKeyboardMarkup(
    [
        ["🗑 Очистить историю ID"],
        ["Отображать лог ?"],
        ["📝 Заменить фильтр-промпт"],
        ["Отмена"],
    ],
    resize_keyboard=True,
)

CANCEL_KB = ReplyKeyboardMarkup([["Отмена"]], resize_keyboard=True)

API_MENU_KB = ReplyKeyboardMarkup(
    [
        ["➕ Добавить API ключи"],
        ["🗑 Очистить API ключи"],
        ["🔙 Назад"],
    ],
    resize_keyboard=True,
    one_time_keyboard=True,
)


def main_menu(ctx: ContextTypes.DEFAULT_TYPE) -> ReplyKeyboardMarkup:
    return ADMIN_MAIN_KB if ctx.chat_data.get("is_admin") else USER_MAIN_KB


async def make_channel_kb(chans: list[str], ctx: ContextTypes.DEFAULT_TYPE) -> ReplyKeyboardMarkup:
    await tg_client.start()
    lookup: dict[str, str] = {}
    names: list[str] = []
    for idx, c in enumerate(chans, 1):
        try:
            ent = await tg_client.get_entity(c)
            name = f"@{ent.username}" if getattr(ent, "username", None) else ent.title
        except Exception:
            logging.exception("Failed to get entity for channel")
            name = c
        label = f"{idx}. {name}"
        names.append(label)
        lookup[label] = c
    await tg_client.disconnect()
    ctx.user_data["chan_lookup"] = lookup
    rows = [[n] for n in names]
    rows.append(["Отмена"])
    return ReplyKeyboardMarkup(rows, resize_keyboard=True, one_time_keyboard=True)

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user = update.effective_user
    if user:
        ctx.chat_data["is_admin"] = user.id in ADMIN_IDS
    ctx.chat_data["target_chat"] = update.effective_chat.id
    ctx.chat_data["chat_id"] = chat_id
    ctx.chat_data["start_id"] = update.message.message_id
    get_cfg(ctx)  # ensure config exists
    await update.message.reply_text("Выберите действие:", reply_markup=main_menu(ctx))


# -------------- текстовый ввод после кнопок --------------------------------
DATE_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4})-(\d{2}\.\d{2}\.\d{4})")

async def text_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    ctx.chat_data.setdefault("target_chat", update.effective_chat.id)
    ctx.chat_data.setdefault("chat_id", str(update.effective_chat.id))
    ctx.chat_data.setdefault("start_id", update.message.message_id)
    user = update.effective_user
    if user:
        ctx.chat_data["is_admin"] = user.id in ADMIN_IDS
    cfg = get_cfg(ctx)
    mode = ctx.user_data.get("mode")

    if text == "Отмена":
        if mode == "api_add_keys":
            ctx.user_data["mode"] = "api_menu"
            await update.message.reply_text(
                "Добавление ключей отменено.", reply_markup=API_MENU_KB
            )
        else:
            ctx.user_data.clear()
            await update.message.reply_text("Отменено", reply_markup=main_menu(ctx))
        return

    if text == "⏹ Остановить":
        await stop_auto_cycle(ctx)
        task = ctx.chat_data.get("task")
        if task and not task.done():
            task.cancel()
        await update.message.reply_text(
            "Парсинг будет остановлен", reply_markup=main_menu(ctx)
        )
        return

    if text == "ℹ️ Инструкция":
        ctx.user_data.clear()
        instruction = (
            "<b>Инструкция по использованию бота</b>\n\n"
            "1. Через кнопку ➕ <b>Добавить каналы</b> занесите интересующие каналы "
            "(через запятую). Все данные и ID просмотренных постов хранятся в файле"
            " processed_ids.json.\n"
            "2. Бот фильтрует контент при помощи ИИ. На каждый текст отправляется "
            "запрос: подходит ли он по тематике? Ответ 'Yes' или 'No' определяется"
            " промптами. Их можно изменить через 📝 <b>Заменить фильтр-промпт</b>. "
            "Если ответ 'Yes', текст перефразируется и отправляется в этот чат.\n"
            "3. <b>🔍 Парсить конкретный канал</b> — выберите канал и укажите, сколь"
            "ко постов проверить. Бот берёт самые свежие сообщения и проходит их п"
            "о порядку.\n"
            "4. <b>➡️ По очереди все каналы</b> — каждый канал из списка обрабатыва"
            "ется по очереди, также по хронологии.\n"
            "5. <b>⭐ Популярные посты всех каналов</b> или <b>⭐ Популярные посты конкретного канала</b> — для каждого канала вычисляется среднее число просмотров. Посты с показателем выше среднего считаются популярными и сортируются по убыванию просмотров. Если самые популярные уже обработаны, бот берёт следующие по рейтингу.\n"
            "6. <b>Поиск постов за последние дни</b> — просматривает сообщения только из указанного числа последних дней.\n"
            "7. <b>📅 Диапазон дат</b> — задайте даты в формате 07.06.2025-07.03.2025,"
            " затем количество постов. Бот ищет сообщения в заданном интервале и перебирает их по очереди.\n"
            "8. Меню <b>Настройки</b> позволяет очистить историю ID, заменить фильтр-промпт и включить или выключить лог (по умолчанию лог отключён).\n"
            "9. <b>Очистить чат</b> — бот удалит все сообщения в этом диалоге.\n"
            "10. Кнопка ⏹ <b>Остановить</b> прерывает любой текущий парсинг.\n"
            "11. <b>📰 Авто-выжимка</b> — бот в реальном времени отслеживает появление новых публикаций, сразу делает по ним выжимки и присылает на апрув. Повторное нажатие останавливает мониторинг, также его можно прервать кнопкой ⏹.\n"
        )
        await update.message.reply_text(
            instruction, reply_markup=main_menu(ctx), parse_mode=tg_const.ParseMode.HTML
        )
        return

    if text == "Меню API ключей":
        if not ctx.chat_data.get("is_admin"):
            await update.message.reply_text(
                "Кнопка доступна только администратору.", reply_markup=main_menu(ctx)
            )
            return
        ctx.user_data.clear()
        ctx.user_data["mode"] = "api_menu"
        masked = key_manager.masked_keys()
        if masked:
            listing = "\n".join(f"{idx + 1}. {mask}" for idx, mask in enumerate(masked))
            msg = (
                f"Сохранено {len(masked)} ключей:\n{listing}\n\n"
                "Выберите действие:"
            )
        else:
            msg = "Ключей пока нет. Добавьте новые ключи.\n\nВыберите действие:"
        await update.message.reply_text(msg, reply_markup=API_MENU_KB)
        return

    auto_task = ctx.chat_data.get("auto_task")
    auto_running = bool(auto_task) and not auto_task.done()

    if text == "📰 Авто-выжимка":
        if auto_running:
            await stop_auto_cycle(ctx)
            await update.message.reply_text(
                "Останавливаю авто-выжимку…", reply_markup=main_menu(ctx)
            )
        else:
            if not cfg.channels:
                await update.message.reply_text("Список каналов пуст.", reply_markup=main_menu(ctx))
                return
            if task_running(ctx):
                await update.message.reply_text(
                    "Уже выполняется задача. Нажмите ⏹ Остановить",
                    reply_markup=main_menu(ctx),
                )
                return
            ctx.user_data.clear()
            ctx.chat_data["auto_stop"] = False
            task = ctx.application.create_task(auto_monitor(ctx))
            ctx.chat_data["auto_task"] = task
            await update.message.reply_text(
                "Авто-выжимка запущена. Мониторю новые посты в реальном времени.",
                reply_markup=main_menu(ctx),
            )
        return

    auto_task = ctx.chat_data.get("auto_task")
    auto_running = bool(auto_task) and not auto_task.done()

    if auto_running and text not in {"⏹ Остановить"}:
        await update.message.reply_text(
            "Сейчас работает авто-выжимка. Остановите её через кнопку 📰 Авто-выжимка, чтобы выполнить другие действия.",
            reply_markup=main_menu(ctx),
        )
        return

    # block new actions while a task is running so menus don't mix
    if task_running(ctx):
        await update.message.reply_text(
            "Уже выполняется задача. Нажмите ⏹ Остановить",
            reply_markup=main_menu(ctx),
        )
        return

    if mode == "api_menu":
        if text == "➕ Добавить API ключи":
            ctx.user_data["mode"] = "api_add_keys"
            await update.message.reply_text(
                "Вставьте OpenAI API ключи (каждый на новой строке).",
                reply_markup=CANCEL_KB,
            )
        elif text == "🗑 Очистить API ключи":
            key_manager.clear_keys()
            ctx.user_data["mode"] = "api_menu"
            await update.message.reply_text(
                "Все ключи удалены. Добавьте новые ключи.", reply_markup=API_MENU_KB
            )
        elif text == "🔙 Назад":
            ctx.user_data.clear()
            await update.message.reply_text(
                "Возвращаюсь в главное меню.", reply_markup=main_menu(ctx)
            )
        else:
            await update.message.reply_text(
                "Выберите действие из меню API ключей.", reply_markup=API_MENU_KB
            )
        return

    if mode == "api_add_keys":
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        keys = raw_lines or []
        added = key_manager.add_keys(keys)
        total = key_manager.count()
        if not keys:
            message = "Не удалось распознать ключи. Вставьте каждый ключ на новой строке."
        elif added:
            skipped = len(keys) - len(added)
            extra = f" (пропущено дублей: {skipped})" if skipped else ""
            message = f"Добавлено {len(added)} ключей{extra}. Всего сохранено: {total}."
        else:
            message = (
                "Новые ключи не добавлены. Возможно, такие ключи уже есть или формат неверный."
                f" Всего сохранено: {total}."
            )
        ctx.user_data["mode"] = "api_menu"
        await update.message.reply_text(message, reply_markup=API_MENU_KB)
        return

    # base menu actions should override pending modes
    if text == "🔍 Парсить конкретный канал":
        ctx.user_data.clear()
        if not cfg.channels:
            await update.message.reply_text("Список каналов пуст.")
            return
        ctx.user_data["mode"] = "by_channel"
        await update.message.reply_text(
            "Выберите канал:",
            reply_markup=await make_channel_kb(cfg.channels, ctx),
        )
        return
    if text == "➡️ По очереди все каналы":
        ctx.user_data.clear()
        ctx.user_data["mode"] = "seq_count"
        await update.message.reply_text(
            "Сколько постов проверить?", reply_markup=ReplyKeyboardRemove()
        )
        return
    if text == "⭐ Популярные посты всех каналов":
        ctx.user_data.clear()
        ctx.user_data["mode"] = "pop_count"
        await update.message.reply_text(
            "Сколько постов проверить?", reply_markup=ReplyKeyboardRemove()
        )
        return
    if text == "⭐ Популярные посты конкретного канала":
        ctx.user_data.clear()
        if not cfg.channels:
            await update.message.reply_text("Список каналов пуст.")
            return
        ctx.user_data["mode"] = "pop_chan_select"
        await update.message.reply_text(
            "Выберите канал:",
            reply_markup=await make_channel_kb(cfg.channels, ctx),
        )
        return
    if text == "Поиск постов за последние дни в конкретном канале":
        ctx.user_data.clear()
        if not cfg.channels:
            await update.message.reply_text("Список каналов пуст.")
            return
        ctx.user_data["mode"] = "recent_chan_select"
        await update.message.reply_text(
            "Выберите какой канал",
            reply_markup=await make_channel_kb(cfg.channels, ctx),
        )
        return
    if text == "Поиск постов за последние дни во всех каналах":
        ctx.user_data.clear()
        ctx.user_data["mode"] = "recent_all_days"
        await update.message.reply_text(
            "Введите число за последние сколько дней искать контент",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    if text == "📅 Диапазон дат (канал)":
        ctx.user_data.clear()
        if not cfg.channels:
            await update.message.reply_text("Список каналов пуст.")
            return
        ctx.user_data["mode"] = "range_channel_select"
        await update.message.reply_text(
            "Выберите канал:",
            reply_markup=await make_channel_kb(cfg.channels, ctx),
        )
        return
    if text == "📅 Диапазон дат (все)":
        ctx.user_data.clear()
        ctx.user_data["mode"] = "range_all_wait"
        await update.message.reply_text(
            'Введите диапазон дат "07.06.2025-07.03.2025"',
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    if text == "➕ Добавить каналы":
        ctx.user_data.clear()
        ctx.user_data["mode"] = "add_channels"
        await update.message.reply_text(
            "Введите каналы через запятую:", reply_markup=ReplyKeyboardRemove()
        )
        return
    if text == "➖ Удалить каналы":
        ctx.user_data.clear()
        if not cfg.channels:
            await update.message.reply_text("Список каналов пуст.")
            return
        ctx.user_data["mode"] = "delete_channel"
        await update.message.reply_text(
            "Нажмите канал который хотите удалить:",
            reply_markup=await make_channel_kb(cfg.channels, ctx),
        )
        return
    if text == "Настройки":
        ctx.user_data.clear()
        await update.message.reply_text(
            "Выберите настройку:", reply_markup=SETTINGS_KB
        )
        return
    if text == "🗑 Очистить историю ID":
        ctx.user_data.clear()
        ctx.user_data["mode"] = "clear_menu"
        await update.message.reply_text(
            "Что очистить?",
            reply_markup=ReplyKeyboardMarkup(
                [
                    ["Очистить конкретные ID постов"],
                    ["Очистить все сохраненные ID постов"],
                    ["Отмена"],
                ],
                resize_keyboard=True,
                one_time_keyboard=True,
            ),
        )
        return
    if text == "📝 Заменить фильтр-промпт":
        ctx.user_data.clear()
        ctx.user_data["mode"] = "replace_prompt_yes"
        await update.message.reply_text(
            "Вот текущий промпт:\n"
            "Отвечай только 'yes' или 'no'.\n"
            f"'Yes' — {cfg.prompt_yes}\n"
            f"'No' — {cfg.prompt_no}",
            reply_markup=CANCEL_KB,
        )
        await update.message.reply_text(
            "Введите промпт в случае если YES:", reply_markup=CANCEL_KB
        )
        return
    if text == "Отображать лог ?":
        ctx.user_data.clear()
        ctx.user_data["mode"] = "toggle_log"
        await update.message.reply_text(
            "Показывать технический лог?",
            reply_markup=ReplyKeyboardMarkup(
                [["Да", "Нет"], ["Отмена"]],
                resize_keyboard=True,
                one_time_keyboard=True,
            ),
        )
        return
    if text == "Очистить чат":
        chat_id = update.effective_chat.id
        ctx.user_data.clear()
        prog = await update.message.reply_text("Очищаю чат…")
        launch_task(
            ctx,
            clear_history(
                ctx, chat_id, update.message.message_id, prog.message_id
            ),
        )
        return

    if mode == "add_channels":
        new_channels = [c.strip().lstrip("@") for c in text.split(",") if c.strip()]
        added = []
        await tg_client.start()
        for c in new_channels:
            try:
                ent = await tg_client.get_entity(c)
                c = ent.username or str(c)
            except Exception:
                logging.exception("Failed to resolve channel")
                c = str(c)
            if c and c not in cfg.channels:
                cfg.channels.append(c)
                cfg.ids.setdefault(c, deque(maxlen=PROCESSED_LIMIT))
                added.append(c)
        await tg_client.disconnect()
        save_all()
        ctx.user_data.clear()
        await update.message.reply_text(
            ("Добавлено: " + ", ".join(added)) if added else "Нет новых каналов",
            reply_markup=main_menu(ctx),
        )
        return

    if mode == "replace_prompt_yes":
        cfg.prompt_yes = text
        ctx.user_data["mode"] = "replace_prompt_no"
        await update.message.reply_text(
            "Введите промпт в случае если NO:", reply_markup=CANCEL_KB
        )
        return

    if mode == "replace_prompt_no":
        cfg.prompt_no = text
        save_all()
        ctx.user_data.clear()
        await update.message.reply_text("Фильтр-промпт обновлён", reply_markup=main_menu(ctx))
        return

    if mode == "toggle_log":
        if text in {"Да", "Нет"}:
            cfg.log_enabled = text == "Да"
            save_all()
            ctx.user_data.clear()
            await update.message.reply_text(
                f"Лог {'включен' if cfg.log_enabled else 'выключен'}",
                reply_markup=main_menu(ctx),
            )
        else:
            await update.message.reply_text(
                "Выберите Да или Нет",
                reply_markup=ReplyKeyboardMarkup(
                    [["Да", "Нет"], ["Отмена"]],
                    resize_keyboard=True,
                    one_time_keyboard=True,
                ),
            )
        return

    if mode == "delete_channel":
        lookup = ctx.user_data.get("chan_lookup", {})
        chan = lookup.get(text, text).lstrip("@")
        if chan in cfg.channels:
            cfg.channels.remove(chan)
            cfg.ids.pop(chan, None)
            save_all()
            msg = f"Канал {chan} удалён"
        else:
            msg = "Неизвестный канал"
        ctx.user_data.clear()
        await update.message.reply_text(msg, reply_markup=main_menu(ctx))
        return

    if mode == "clear_menu":
        if text == "Очистить конкретные ID постов":
            if not cfg.channels:
                await update.message.reply_text("Список каналов пуст.", reply_markup=main_menu(ctx))
                ctx.user_data.clear()
            else:
                ctx.user_data["mode"] = "clear_ids"
                await update.message.reply_text(
                    "Какой канал очистить?",
                    reply_markup=await make_channel_kb(cfg.channels, ctx),
                )
            return
        if text == "Очистить все сохраненные ID постов":
            for c in cfg.channels:
                cfg.ids[c] = deque(maxlen=PROCESSED_LIMIT)
            save_all()
            ctx.user_data.clear()
            await update.message.reply_text("Все ID очищены", reply_markup=main_menu(ctx))
            return
        if text == "Отмена":
            ctx.user_data.clear()
            await update.message.reply_text("Отменено", reply_markup=main_menu(ctx))
            return

    if mode == "clear_ids":
        lookup = ctx.user_data.get("chan_lookup", {})
        chan = lookup.get(text, text).lstrip("@")
        if chan in cfg.channels:
            cfg.ids[chan] = deque(maxlen=PROCESSED_LIMIT)
            save_all()
            msg = f"История {chan} очищена"
        else:
            msg = "Неизвестный канал"
        ctx.user_data.clear()
        await update.message.reply_text(msg, reply_markup=main_menu(ctx))
        return

    if mode == "by_channel":
        lookup = ctx.user_data.get("chan_lookup", {})
        chan = lookup.get(text, text).lstrip("@")
        if chan not in cfg.channels:
            await update.message.reply_text("Канал не в списке.")
            ctx.user_data.clear()
        else:
            ctx.user_data["channel"] = chan
            ctx.user_data["mode"] = "by_channel_count"
            await update.message.reply_text(
                "Сколько постов проверить?", reply_markup=ReplyKeyboardRemove()
            )
        return

    if mode == "by_channel_count":
        if not text.isdigit():
            await update.message.reply_text("Нужно число")
            return
        if task_running(ctx):
            await update.message.reply_text(
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=main_menu(ctx)
            )
            return
        chan = ctx.user_data.get("channel")
        limit = int(text)
        await update.message.reply_text("Обрабатываю…", reply_markup=main_menu(ctx))
        launch_task(ctx, run_channel(ctx, chan, None, None, limit))
        return

    if mode == "range_channel_select":
        lookup = ctx.user_data.get("chan_lookup", {})
        chan = lookup.get(text, text).lstrip("@")
        if chan in cfg.channels:
            ctx.user_data["mode"] = "range_channel_wait"
            ctx.user_data["channel"] = chan
            await update.message.reply_text(
                'Введите диапазон дат "07.06.2025-07.03.2025"',
                reply_markup=ReplyKeyboardRemove(),
            )
        else:
            await update.message.reply_text("Канал не в списке.", reply_markup=main_menu(ctx))
            ctx.user_data.clear()
        return

    if mode == "range_channel_wait":
        m = DATE_RE.match(text)
        if m:
            ctx.user_data["from"] = m.group(1)
            ctx.user_data["to"] = m.group(2)
            ctx.user_data["mode"] = "range_channel_count"
            await update.message.reply_text(
                "Сколько постов проверить?", reply_markup=ReplyKeyboardRemove()
            )
        else:
            await update.message.reply_text("Формат неверен.")
        return

    if mode == "range_channel_count":
        if not text.isdigit():
            await update.message.reply_text("Нужно число")
            return
        if task_running(ctx):
            await update.message.reply_text(
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=main_menu(ctx)
            )
            return
        chan = ctx.user_data.get("channel")
        limit = int(text)
        from_d = ctx.user_data.get("from")
        to_d = ctx.user_data.get("to")
        await update.message.reply_text("Обрабатываю…", reply_markup=main_menu(ctx))
        launch_task(ctx, run_channel(ctx, chan, from_d, to_d, limit))
        return

    if mode == "range_all_wait":
        m = DATE_RE.match(text)
        if m:
            ctx.user_data["from"] = m.group(1)
            ctx.user_data["to"] = m.group(2)
            ctx.user_data["mode"] = "range_all_count"
            await update.message.reply_text(
                "Сколько постов проверить?", reply_markup=ReplyKeyboardRemove()
            )
        else:
            await update.message.reply_text("Формат неверен.")
        return

    if mode == "range_all_count":
        if not text.isdigit():
            await update.message.reply_text("Нужно число")
            return
        if task_running(ctx):
            await update.message.reply_text(
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=main_menu(ctx)
            )
            return
        limit = int(text)
        from_d = ctx.user_data.get("from")
        to_d = ctx.user_data.get("to")
        await update.message.reply_text("Обрабатываю…", reply_markup=main_menu(ctx))
        launch_task(ctx, run_seq_all(ctx, from_d, to_d, limit))
        return

    if mode == "seq_count":
        if not text.isdigit():
            await update.message.reply_text("Нужно число")
            return
        if task_running(ctx):
            await update.message.reply_text(
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=main_menu(ctx)
            )
            return
        limit = int(text)
        await update.message.reply_text("Стартуем…", reply_markup=main_menu(ctx))
        launch_task(ctx, run_seq_all(ctx, None, None, limit))
        return

    if mode == "pop_count":
        if not text.isdigit():
            await update.message.reply_text("Нужно число")
            return
        if task_running(ctx):
            await update.message.reply_text(
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=main_menu(ctx)
            )
            return
        limit = int(text)
        await update.message.reply_text("Ищем популярные…", reply_markup=main_menu(ctx))
        launch_task(ctx, run_pop_all(ctx, limit))
        return

    if mode == "pop_chan_select":
        lookup = ctx.user_data.get("chan_lookup", {})
        chan = lookup.get(text, text).lstrip("@")
        if chan in cfg.channels:
            ctx.user_data["channel"] = chan
            ctx.user_data["mode"] = "pop_chan_count"
            await update.message.reply_text(
                "Сколько постов проверить?", reply_markup=ReplyKeyboardRemove()
            )
        else:
            await update.message.reply_text("Канал не в списке.", reply_markup=main_menu(ctx))
            ctx.user_data.clear()
        return

    if mode == "pop_chan_count":
        if not text.isdigit():
            await update.message.reply_text("Нужно число")
            return
        if task_running(ctx):
            await update.message.reply_text(
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=main_menu(ctx)
            )
            return
        chan = ctx.user_data.get("channel")
        limit = int(text)
        await update.message.reply_text("Ищем популярные…", reply_markup=main_menu(ctx))
        launch_task(ctx, run_pop_channel(ctx, chan, limit))
        return

    if mode == "recent_chan_select":
        lookup = ctx.user_data.get("chan_lookup", {})
        chan = lookup.get(text, text).lstrip("@")
        if chan in cfg.channels:
            ctx.user_data["channel"] = chan
            ctx.user_data["mode"] = "recent_chan_days"
            await update.message.reply_text(
                "Введите число за последние сколько дней искать контент",
                reply_markup=ReplyKeyboardRemove(),
            )
        else:
            await update.message.reply_text("Канал не в списке.", reply_markup=main_menu(ctx))
            ctx.user_data.clear()
        return

    if mode == "recent_chan_days":
        if not text.isdigit():
            await update.message.reply_text("Нужно число")
            return
        if task_running(ctx):
            await update.message.reply_text(
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=main_menu(ctx)
            )
            return
        chan = ctx.user_data.get("channel")
        days = int(text)
        await update.message.reply_text("Обрабатываю…", reply_markup=main_menu(ctx))
        launch_task(ctx, run_recent_channel(ctx, chan, days))
        return

    if mode == "recent_all_days":
        if not text.isdigit():
            await update.message.reply_text("Нужно число")
            return
        if task_running(ctx):
            await update.message.reply_text(
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=main_menu(ctx)
            )
            return
        days = int(text)
        await update.message.reply_text("Обрабатываю…", reply_markup=main_menu(ctx))
        launch_task(ctx, run_recent_all(ctx, days))
        return
    if mode:
        await update.message.reply_text("Не понимаю ответ, начните заново", reply_markup=main_menu(ctx))
        ctx.user_data.clear()
    else:
        await update.message.reply_text("Неизвестная команда", reply_markup=main_menu(ctx))


# -------------- задачи -----------------------------------------------------


async def auto_cycle(
    ctx: ContextTypes.DEFAULT_TYPE, *, client_ready: bool = False
) -> int:
    cfg = get_cfg(ctx)
    if not cfg.channels:
        return 0

    total_sent = 0
    started_here = False
    if not client_ready:
        await tg_client.start()
        started_here = True
    ctx.chat_data["stop"] = False
    ctx.chat_data["sent"] = 0
    try:
        for chan in cfg.channels:
            if ctx.chat_data.get("auto_stop"):
                break
            posts = await fetch_posts(chan, None, None, 50)
            if not posts:
                continue
            key = chan.lstrip("@")
            seen = cfg.ids.setdefault(key, deque(maxlen=PROCESSED_LIMIT))
            last_seen = cfg.auto_last.get(key, 0)
            if not last_seen:
                baseline_added = False
                for msg in posts:
                    if mark_processed(cfg, key, msg.id):
                        baseline_added = True
                if baseline_added:
                    save_all()
                    await log(ctx, f"Синхронизировал стартовую точку для {chan}")
                continue
            fresh = [m for m in posts if m.id not in seen and m.id > last_seen]
            if not fresh:
                continue
            try:
                sent, _ = await send_filtered_posts(
                    ctx,
                    chan,
                    fresh,
                    0,
                    mode="digest",
                    notify=False,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Auto cycle failed for channel")
                await log(ctx, f"Ошибка при обработке канала {chan}")
                continue
            total_sent += sent
    finally:
        if started_here:
            await tg_client.disconnect()
    return total_sent


async def prime_auto_seen(
    ctx: ContextTypes.DEFAULT_TYPE, *, client_ready: bool = False
) -> int:
    """Зафиксировать текущие посты, чтобы авто-выжимка начинала только с новых."""

    cfg = get_cfg(ctx)
    if not cfg.channels:
        return 0

    added_total = 0
    started_here = False
    if not client_ready:
        await tg_client.start()
        started_here = True
    try:
        modified = False
        for chan in cfg.channels:
            posts = await fetch_posts(chan, None, None, 50)
            if not posts:
                continue
            key = chan.lstrip("@")
            added_here = 0
            for msg in posts:
                if mark_processed(cfg, key, msg.id):
                    added_here += 1
            if added_here:
                modified = True
                added_total += added_here
                await log(ctx, f"Зафиксировал последние {added_here} постов из {chan}")
        if modified:
            save_all()
    finally:
        if started_here:
            await tg_client.disconnect()

    return added_total


async def auto_monitor(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = ctx.chat_data.get("target_chat")
    if chat_id is None:
        return
    try:
        await tg_client.start()
    except Exception:
        logging.exception("Не удалось запустить Telethon клиент для авто-выжимки")
        await ctx.bot.send_message(
            chat_id,
            "Не удалось подключиться к Telegram. Попробуйте запустить авто-выжимку позже.",
        )
        return
    try:
        primed = await prime_auto_seen(ctx, client_ready=True)
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.exception("Не удалось подготовить авто-выжимку")
        primed = 0
    await ctx.bot.send_message(
        chat_id,
        "Авто-выжимка активирована. Новые релевантные публикации будут приходить сразу после выхода.",
    )
    if primed:
        await ctx.bot.send_message(
            chat_id,
            "Текущие публикации помечены как просмотренные, начну с новых постов.",
        )
    try:
        while not ctx.chat_data.get("auto_stop"):
            total = await auto_cycle(ctx, client_ready=True)
            if ctx.chat_data.get("auto_stop"):
                break
            if total:
                await ctx.bot.send_message(
                    chat_id,
                    f"Авто-выжимка: прислано {total} новостей на апрув.",
                )
            sleep_seconds = 5 if total else 10
            for _ in range(sleep_seconds):
                if ctx.chat_data.get("auto_stop"):
                    break
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        logging.info("Авто-выжимка принудительно остановлена")
    except Exception:
        logging.exception("Ошибка в автоматическом мониторинге")
        await ctx.bot.send_message(
            chat_id,
            "Авто-выжимка остановлена из-за ошибки. Проверьте логи.",
        )
    finally:
        ctx.chat_data.pop("auto_stop", None)
        ctx.chat_data.pop("auto_task", None)
        try:
            await asyncio.shield(
                ctx.bot.send_message(chat_id, "Авто-выжимка остановлена.")
            )
        except Exception:
            logging.exception(
                "Не удалось отправить уведомление об остановке авто-выжимки"
            )
        finally:
            try:
                await tg_client.disconnect()
            except Exception:
                logging.exception("Не удалось отключить Telethon клиент после авто-выжимки")


async def run_seq_all(ctx, from_d: str | None = None, to_d: str | None = None, limit: int | None = None) -> bool:
    await tg_client.start()
    await log(ctx, "Запускаю обход всех каналов")
    from_dt = datetime.strptime(from_d, "%d.%m.%Y") if from_d else None
    to_dt   = datetime.strptime(to_d, "%d.%m.%Y") if to_d else None
    ctx.chat_data["stop"] = False
    ctx.chat_data["sent"] = 0
    cfg = get_cfg(ctx)
    attempts_total = 0
    for c in cfg.channels:
        await log(ctx, f"Читаю канал {c}")
        if ctx.chat_data.get("stop"):
            break
        msgs = await fetch_posts(c, from_dt, to_dt, None)
        await log(ctx, f"Получено {len(msgs)} сообщений из {c}")
        sent, attempts = await send_filtered_posts(ctx, c, msgs, limit or 0)
        attempts_total += attempts
    await tg_client.disconnect()
    if ctx.chat_data.get("stop"):
        msg = "Остановлено"
    elif ctx.chat_data.get("sent"):
        msg = "Готово ✅"
    elif attempts_total >= ATTEMPT_LIMIT:
        msg = ATTEMPT_MSG
    else:
        msg = "Введите количество постов больше, бот ничего не нашёл"
    await ctx.bot.send_message(ctx.chat_data["target_chat"], msg)
    ctx.user_data.clear()
    return msg not in {"Введите количество постов больше, бот ничего не нашёл", ATTEMPT_MSG}

async def run_pop_all(ctx, limit: int | None = None) -> bool:
    await tg_client.start()
    await log(ctx, "Ищем популярные посты во всех каналах")
    cfg = get_cfg(ctx)
    ctx.chat_data["stop"] = False
    ctx.chat_data["sent"] = 0
    for c in cfg.channels:
        if ctx.chat_data.get("stop"):
            break
        await log(ctx, f"Читаю канал {c}")
        posts = await fetch_posts(c, None, None, None)
        if not posts:
            continue
        await log(ctx, f"Найдено {len(posts)} постов, выбираем популярные")
        avg = sum(p.views or 0 for p in posts) / len(posts)
        popular = [p for p in posts if (p.views or 0) > avg]
        popular.sort(key=lambda m: m.views or 0, reverse=True)
        sent, _ = await send_filtered_posts(ctx, c, popular, limit or 0)
        await log(ctx, f"Канал {c} обработан: отправлено {sent}")
    await tg_client.disconnect()
    if ctx.chat_data.get("stop"):
        msg = "Остановлено"
    elif ctx.chat_data.get("sent"):
        msg = "Готово ✅"
    else:
        msg = "Введите количество постов больше, бот ничего не нашёл"
    await ctx.bot.send_message(ctx.chat_data["target_chat"], msg)
    ctx.user_data.clear()
    return msg != "Введите количество постов больше, бот ничего не нашёл"

async def run_pop_channel(ctx, chan: str, limit: int | None = None) -> bool:
    cfg = get_cfg(ctx)
    if chan not in cfg.channels:
        await ctx.bot.send_message(ctx.chat_data["target_chat"], "Канал не в списке.")
        return True
    await tg_client.start()
    await log(ctx, f"Ищем популярные посты в {chan}")
    ctx.chat_data["stop"] = False
    ctx.chat_data["sent"] = 0
    posts = await fetch_posts(chan, None, None, None)
    if posts:
        await log(ctx, f"Найдено {len(posts)} постов, выбираем популярные")
        avg = sum(p.views or 0 for p in posts) / len(posts)
        popular = [p for p in posts if (p.views or 0) > avg]
        popular.sort(key=lambda m: m.views or 0, reverse=True)
        sent, _ = await send_filtered_posts(ctx, chan, popular, limit or 0)
        await log(ctx, f"Популярные посты канала {chan} обработаны: отправлено {sent}")
    await tg_client.disconnect()
    if ctx.chat_data.get("stop"):
        msg = "Остановлено"
    elif ctx.chat_data.get("sent"):
        msg = "Готово ✅"
    else:
        msg = "Введите количество постов больше, бот ничего не нашёл"
    await ctx.bot.send_message(ctx.chat_data["target_chat"], msg)
    ctx.user_data.clear()
    return msg != "Введите количество постов больше, бот ничего не нашёл"

async def run_recent_all(ctx, days: int) -> bool:
    await tg_client.start()
    await log(ctx, f"Поиск постов за последние {days} дней во всех каналах")
    from_dt = datetime.now() - timedelta(days=days)
    to_dt = datetime.now()
    ctx.chat_data["stop"] = False
    ctx.chat_data["sent"] = 0
    cfg = get_cfg(ctx)
    attempts_total = 0
    for c in cfg.channels:
        if ctx.chat_data.get("stop"):
            break
        await log(ctx, f"Читаю канал {c}")
        msgs = await fetch_posts(c, from_dt, to_dt, None)
        await log(ctx, f"Получено {len(msgs)} сообщений из {c}")
        sent, attempts = await send_filtered_posts(ctx, c, msgs, len(msgs))
        attempts_total += attempts
    await tg_client.disconnect()
    if ctx.chat_data.get("stop"):
        msg = "Остановлено"
    elif ctx.chat_data.get("sent"):
        msg = "Готово ✅"
    elif attempts_total >= ATTEMPT_LIMIT:
        msg = ATTEMPT_MSG
    else:
        msg = "Бот ничего не нашёл"
    await ctx.bot.send_message(ctx.chat_data["target_chat"], msg)
    ctx.user_data.clear()
    return msg not in {"Бот ничего не нашёл", ATTEMPT_MSG}

async def run_recent_channel(ctx, chan: str, days: int) -> bool:
    cfg = get_cfg(ctx)
    if chan not in cfg.channels:
        await ctx.bot.send_message(ctx.chat_data["target_chat"], "Канал не в списке.")
        return True
    await tg_client.start()
    await log(ctx, f"Поиск постов за последние {days} дней в {chan}")
    from_dt = datetime.now() - timedelta(days=days)
    to_dt = datetime.now()
    ctx.chat_data["stop"] = False
    ctx.chat_data["sent"] = 0
    msgs = await fetch_posts(chan, from_dt, to_dt, None)
    await log(ctx, f"Получено {len(msgs)} сообщений")
    sent, attempts = await send_filtered_posts(ctx, chan, msgs, len(msgs))
    await log(ctx, f"Канал {chan} обработан: отправлено {sent} из {attempts}")
    await tg_client.disconnect()
    if ctx.chat_data.get("stop"):
        msg = "Остановлено"
    elif sent and attempts:
        msg = "Готово ✅"
    elif attempts >= ATTEMPT_LIMIT:
        msg = ATTEMPT_MSG
    else:
        msg = "Бот ничего не нашёл"
    await ctx.bot.send_message(ctx.chat_data["target_chat"], msg)
    ctx.user_data.clear()
    return msg not in {"Бот ничего не нашёл", ATTEMPT_MSG}

async def run_channel(
    ctx,
    chan: str,
    from_d: str | None = None,
    to_d: str | None = None,
    limit: int | None = None,
):
    cfg = get_cfg(ctx)
    if chan not in cfg.channels:
        await ctx.bot.send_message(ctx.chat_data["target_chat"], "Канал не в списке.")
        return True
    await tg_client.start()
    await log(ctx, f"Парсим {chan}")
    from_dt = datetime.strptime(from_d, "%d.%m.%Y") if from_d else None
    to_dt   = datetime.strptime(to_d, "%d.%m.%Y") if to_d else None
    ctx.chat_data["stop"] = False
    ctx.chat_data["sent"] = 0
    msgs = await fetch_posts(chan, from_dt, to_dt, None)
    await log(ctx, f"Получено {len(msgs)} сообщений")
    sent, attempts = await send_filtered_posts(ctx, chan, msgs, limit or 0)
    await log(ctx, f"Канал {chan} обработан: отправлено {sent} из {attempts}")
    await tg_client.disconnect()
    if ctx.chat_data.get("stop"):
        msg = "Остановлено"
    elif sent >= (limit or 0) and sent:
        msg = "Готово ✅"
    elif sent == 0 and attempts >= ATTEMPT_LIMIT:
        msg = ATTEMPT_MSG
    else:
        msg = "Введите количество постов больше, бот ничего не нашёл"
    await ctx.bot.send_message(ctx.chat_data["target_chat"], msg)
    ctx.user_data.clear()
    return msg not in {"Введите количество постов больше, бот ничего не нашёл", ATTEMPT_MSG}

# ────────────── MAIN ───────────────────────────────────────────────────────
def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))

    print("Bot running…  (Ctrl+C to stop)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
if __name__ == "__main__":    main()