# -*- coding: utf-8 -*-
"""
DOCX Validator Telegram Bot — Articles & EEAT (Zoho Writer API)
ВЕРСИЯ: 2.1.0 — GPT‑5 mini integration, responses API fallback, admin-only keys menu, robust DOCX patch, improved prompts, promo-code scan
Дата: 2025‑09‑15
"""

# =========================================================
# ===============  НАСТРОЙКА (ОБЯЗАТЕЛЬНО)  ===============
# =========================================================
import os  # <— важно: os должен быть раньше использования
import copy

# 1) Токен телеграм‑бота
BOT_TOKEN = "7506878864:AAEsjLOa0yT-WfB-4AKhrhFA3xopuJzaHPY"

# 2) ID администратора (целое число).
#    Только этот пользователь увидит кнопку/команду «Добавить OpenAI Keys».
#    Пример: ALLOWED_USER_ID = 123456789
ADMIN_TG_ID = 408198196  # <-- введите свой Telegram ID

# 3) Модель OpenAI (GPT‑5 mini по умолчанию)
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

# 4) Путь к локальной БД (сюда сохраняются ключи и служебное состояние)
DB_PATH = "bot_state.sqlite3"

# 5) Zoho OAuth — нужно, только если качаете .docx с Zoho Writer.
ZOHO_CLIENT_ID     = "1000.OVLXDJNC2W9FG0TP9MX16VQ4F9ZA9T"
ZOHO_CLIENT_SECRET = "1c82ede3caade03b28e9cbdef636c8d48a0e1894cd"
ZOHO_REFRESH_TOKEN = "1000.be990ad5b2db465263fcd0e2d44dcc9d.814765cf10392c7af6c7336ac521438e"
ZOHO_ACCOUNTS_URL  = os.environ.get("ZOHO_ACCOUNTS_URL", "https://accounts.zoho.com").rstrip("/")
ZOHOAPIS_DOMAIN_DEFAULT = os.environ.get("ZOHOAPIS_DOMAIN", "www.zohoapis.com")

# 6) Производительность / лимиты
OPENAI_TIMEOUT_SEC     = float(os.environ.get("OPENAI_TIMEOUT", "25.0"))
OPENAI_POOL_MAXCONN    = int(os.environ.get("OPENAI_POOL_MAXCONN", "12"))
LLM_MAX_RETRIES_LOCAL  = int(os.environ.get("OPENAI_MAX_RETRIES", "1"))
PROMO_MAX_CHARS        = int(os.environ.get("PROMO_MAX_CHARS", "12000"))
EEAT_PROMPT_MAX_CHARS  = int(os.environ.get("EEAT_PROMPT_MAX_CHARS", "6000"))
ART_PROMPT_MAX_CHARS   = int(os.environ.get("ART_PROMPT_MAX_CHARS", "6000"))

# Разрешенные служебные метки, которые не считаются промокодами.
PROMO_IGNORE_CODES = {
    "MT", "MD", "MK", "MB", "H1", "H2", "H3", "H4",
    "CTA", "FAQ", "SEO", "URL", "WWW", "HTTP", "HTTPS",
}
EEAT_LANGUAGE_IGNORE_CODES = {"MK", "MT"}
VALIDATOR_CONCURRENCY  = int(os.environ.get("VALIDATOR_CONCURRENCY", "8"))

# --- Новые параметры (Zoho OAuth: кеш и задержки) ---
ZOHO_OAUTH_MAX_RETRIES       = int(os.environ.get("ZOHO_OAUTH_MAX_RETRIES", "3"))
ZOHO_OAUTH_RETRY_BASE_SEC    = float(os.environ.get("ZOHO_OAUTH_RETRY_BASE_SEC", "2.0"))
ZOHO_OAUTH_MIN_INTERVAL_SEC  = float(os.environ.get("ZOHO_OAUTH_MIN_INTERVAL_SEC", "2.5"))
ZOHO_OAUTH_TTL_SKEW_SEC      = int(os.environ.get("ZOHO_OAUTH_TTL_SKEW_SEC", "60"))
ZOHO_TOKEN_DEFAULT_TTL_SEC   = int(os.environ.get("ZOHO_TOKEN_DEFAULT_TTL_SEC", "3600"))

# =========================================================
# ================== ИМПОРТЫ/КОНСТАНТЫ ====================
# =========================================================
import re
import io
import json
import time
import math
import sqlite3
import tempfile
import datetime
import unicodedata
import zipfile
import shutil
import posixpath as pp
import traceback
import asyncio
import random
import threading
from typing import List, Tuple, Dict, Optional, Callable, Any, Sequence, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (Message, FSInputFile, CallbackQuery, BotCommand,
                           BotCommandScopeDefault, BotCommandScopeChat)

from lxml import etree
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from PIL import Image

import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry as _URLLIB3_Retry
except Exception:
    _URLLIB3_Retry = None

import httpx
from openai import OpenAI

BAR = "═" * 26
WORD_JOINER = "\u2060"

# =========================================================
# ======================  УТИЛИТЫ  ========================
# =========================================================
def now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def trim(s: str, n: int = 800) -> str:
    if s is None: return ""
    s = s.strip()
    return s if len(s) <= n else (s[:n].rstrip() + " …(обрезано)")

def mask_secret(v: Optional[str], head: int = 6, tail: int = 4) -> str:
    if not v: return ""
    if len(v) <= head + tail + 3: return v[:1] + "***"
    return f"{v[:head]}...{v[-tail:]}"

def strip_invisible(s: str) -> str:
    if not s: return s
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r'[\u00AD\u034F\u061C\u180E\u2000-\u200A\u200B-\u200F\u202F\u205F\u2060-\u2064\u3000\uFE00-\uFE0F]', "", s)

def clean_for_prompt(text: str) -> str:
    s = strip_invisible((text or ""))
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[:n].rstrip() + " …")

def neutralize_slash_tokens(s: str) -> str:
    return re.sub(r'(^|[\s\(\[#>\-])/(?!/)', lambda m: m.group(1) + "/" + WORD_JOINER, s or "")

def first_words(text: str, n: int = 3) -> str:
    words = re.findall(r'\w+', (text or "").strip())
    return (" ".join(words[:n]) + "...") if words else ""

# =========================================================
# =====================  ОПРЕДЕЛЕНИЕ ПИСЬМА  ==============
# =========================================================
_SCRIPT_RANGES = {
    "latin": (
        (0x0041, 0x005A), (0x0061, 0x007A),
        (0x00C0, 0x00FF), (0x0100, 0x017F), (0x0180, 0x024F),
    ),
    "cyrillic": (
        (0x0400, 0x04FF), (0x0500, 0x052F),
        (0x2DE0, 0x2DFF), (0xA640, 0xA69F),
    ),
    "greek": (
        (0x0370, 0x03FF), (0x1F00, 0x1FFF),
    ),
    "hebrew": ((0x0590, 0x05FF),),
    "arabic": (
        (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),
    ),
}

_SCRIPT_LABELS = {
    "latin": "латиница",
    "cyrillic": "кириллица",
    "greek": "греческий алфавит",
    "hebrew": "иврит",
    "arabic": "арабский алфавит",
}


def _char_script(ch: str) -> Optional[str]:
    code = ord(ch)
    for script, ranges in _SCRIPT_RANGES.items():
        for start, end in ranges:
            if start <= code <= end:
                return script
    return None


def _dominant_script(text: str) -> Optional[str]:
    counts: Dict[str, int] = {}
    total = 0
    for ch in strip_invisible(text or ""):
        if not ch.isalpha():
            continue
        script = _char_script(ch)
        if not script:
            continue
        counts[script] = counts.get(script, 0) + 1
        total += 1
    if not counts or total == 0:
        return None
    script, count = max(counts.items(), key=lambda kv: kv[1])
    if count < 3:
        return None
    if (count / total) < 0.6:
        return None
    return script


def _script_label(script: Optional[str]) -> str:
    if not script:
        return "неизвестный алфавит"
    return _SCRIPT_LABELS.get(script, script)

# =========================================================
# ======================  БАЗА ДАННЫХ  ====================
# =========================================================
def db_init():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kv (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS openai_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sk TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            )
        """)
        try:
            cur = conn.execute("PRAGMA table_info(openai_keys)")
            cols = {r[1] for r in cur.fetchall()}
            if "sk" not in cols:
                conn.execute(f"ALTER TABLE openai_keys RENAME TO openai_keys_backup_{int(time.time())}")
                conn.execute("""
                    CREATE TABLE openai_keys (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sk TEXT NOT NULL UNIQUE,
                        created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
                    )
                """)
        except Exception:
            pass
        try:
            cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='gemini_keys'")
            if cur.fetchone():
                conn.execute("DROP TABLE gemini_keys")
        except Exception:
            pass
        conn.commit()

def db_add_keys_bulk(text: str) -> int:
    text = text or ""
    candidates = re.findall(r'\b(?:sk|rk)-[A-Za-z0-9_\-]{15,}\b', text)
    added = 0
    if not candidates:
        return 0
    with sqlite3.connect(DB_PATH) as conn:
        for sk in candidates:
            try:
                conn.execute("INSERT OR IGNORE INTO openai_keys(sk) VALUES(?)", (sk.strip(),))
                added += 1
            except Exception:
                pass
        conn.commit()
    return added

def db_list_keys() -> List[str]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT sk FROM openai_keys ORDER BY id ASC")
        return [r[0] for r in cur.fetchall()]

def db_clear_keys() -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("DELETE FROM openai_keys")
        conn.commit()
        return cur.rowcount

def db_get_kv(key: str) -> Optional[str]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT v FROM kv WHERE k=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

def db_set_kv(key: str, value: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("REPLACE INTO kv(k, v) VALUES(?, ?)", (key, value))
        conn.commit()

_ADMIN_DEBUG_LOCK = threading.Lock()
_ADMIN_DEBUG_ENABLED: Optional[bool] = None
_ADMIN_DEBUG_LOGS: List[str] = []
_ADMIN_DEBUG_SEQ: int = 0

def admin_debug_enabled() -> bool:
    global _ADMIN_DEBUG_ENABLED
    with _ADMIN_DEBUG_LOCK:
        if _ADMIN_DEBUG_ENABLED is None:
            db_init()
            val = db_get_kv("admin_debug_log_enabled")
            _ADMIN_DEBUG_ENABLED = (val == "1")
        return bool(_ADMIN_DEBUG_ENABLED)

def admin_debug_set(enabled: bool) -> None:
    global _ADMIN_DEBUG_ENABLED
    db_init()
    with _ADMIN_DEBUG_LOCK:
        _ADMIN_DEBUG_ENABLED = bool(enabled)
        if not enabled:
            _ADMIN_DEBUG_LOGS.clear()
    db_set_kv("admin_debug_log_enabled", "1" if enabled else "0")

def admin_debug_log(message: str) -> None:
    if not admin_debug_enabled():
        return
    global _ADMIN_DEBUG_SEQ
    timestamp = now_str()
    with _ADMIN_DEBUG_LOCK:
        _ADMIN_DEBUG_SEQ += 1
        seq = _ADMIN_DEBUG_SEQ
        line = f"[{timestamp} #{seq:04d}] {message}"
        _ADMIN_DEBUG_LOGS.append(line)
        if len(_ADMIN_DEBUG_LOGS) > 200:
            del _ADMIN_DEBUG_LOGS[:-200]

def admin_debug_drain() -> List[str]:
    with _ADMIN_DEBUG_LOCK:
        logs = list(_ADMIN_DEBUG_LOGS)
        _ADMIN_DEBUG_LOGS.clear()
    return logs


def _split_text_for_telegram(text: str, limit: int = 3500) -> List[str]:
    if not text:
        return []
    try:
        limit_val = int(limit)
    except (TypeError, ValueError):
        limit_val = 3500
    limit = max(1, limit_val)
    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    lines = text.split('\n')
    for idx, line in enumerate(lines):
        suffix = '\n' if idx < len(lines) - 1 else ''
        chunk = line + suffix
        while chunk:
            available = limit - current_len
            if available <= 0:
                parts.append("".join(current))
                current = []
                current_len = 0
                available = limit
            take = min(len(chunk), available)
            current.append(chunk[:take])
            current_len += take
            chunk = chunk[take:]
            if current_len >= limit:
                parts.append("".join(current))
                current = []
                current_len = 0
    if current:
        parts.append("".join(current))

    return [p for p in parts if p]

# =========================================================
# ==============  OPENAI (карусель ключей)  ===============
# =========================================================
import contextlib

class KeysExhaustedError(RuntimeError):
    pass

def _httpx_client():
    return httpx.Client(
        timeout=OPENAI_TIMEOUT_SEC,
        limits=httpx.Limits(max_connections=OPENAI_POOL_MAXCONN, max_keepalive_connections=OPENAI_POOL_MAXCONN),
        follow_redirects=True,
    )

def _openai_client_for_key(sk: str) -> OpenAI:
    return OpenAI(api_key=sk, http_client=_httpx_client())

def _messages_preview(messages: List[Dict[str, Any]], limit: int = 600) -> str:
    parts: List[str] = []
    for msg in messages or []:
        role = str(msg.get("role") or "?")
        content = msg.get("content")
        if isinstance(content, list):
            texts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    txt = str(part.get("text") or "")
                else:
                    txt = str(part)
                if txt:
                    texts.append(txt.strip())
            text = " ".join(texts)
        else:
            text = str(content or "")
        parts.append(f"{role}: {text.strip()}")
    joined = " | ".join(parts)
    return clip(joined, limit)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                txt = str(part.get("text") or "")
            else:
                txt = str(part)
            if txt:
                texts.append(txt.strip())
        text = " ".join(texts)
    else:
        text = str(content or "")
    return text.strip()


def _format_messages_for_log(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in messages or []:
        role = str(msg.get("role") or "?")
        text = _message_content_to_text(msg.get("content"))
        if not text:
            text = "(пусто)"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _format_exception(ex: BaseException) -> str:
    name = type(ex).__name__
    return f"{name}: {ex}"


def _log_llm_chat(label: str,
                  key_masked: str,
                  round_idx: int,
                  attempt_idx: int,
                  messages: List[Dict[str, Any]],
                  response_text: Optional[str] = None,
                  error: Optional[BaseException] = None,
                  method: Optional[str] = None) -> None:
    header = [f"LLM[{label}] (ключ {key_masked}, попытка {attempt_idx}, раунд {round_idx}"]
    if method:
        header.append(f", метод {method}")
    header.append(")")
    prompt_text = _format_messages_for_log(messages)
    if not prompt_text:
        prompt_text = "(пусто)"
    if error is None:
        resp = (response_text or "").strip()
        if not resp:
            resp = "(пустой ответ)"
        admin_debug_log(
            "\n".join(
                [
                    "".join(header),
                    BAR,
                    "Запрос:",
                    prompt_text,
                    "",
                    "Ответ:",
                    resp,
                ]
            )
        )
    else:
        err_txt = _format_exception(error)
        admin_debug_log(
            "\n".join(
                [
                    "".join(header),
                    BAR,
                    "Запрос:",
                    prompt_text,
                    "",
                    "Ответ:",
                    f"Ошибка: {err_txt}",
                    "",
                    "Почему так?",
                ]
            )
        )

def _responses_input_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for msg in messages or []:
        role = str(msg.get("role") or "user")
        content = msg.get("content", "")
        blocks: List[Dict[str, str]] = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") and "text" in part:
                    blocks.append({"type": part.get("type", "text"), "text": str(part.get("text", ""))})
                elif isinstance(part, str):
                    blocks.append({"type": "text", "text": part})
                else:
                    blocks.append({"type": "text", "text": str(part)})
        else:
            blocks.append({"type": "text", "text": str(content or "")})
        formatted.append({"role": role, "content": blocks})
    return formatted

def _extract_text_from_openai_response(resp: Any) -> Optional[str]:
    if resp is None:
        return None

    text_attr = getattr(resp, "output_text", None)
    if isinstance(text_attr, str) and text_attr.strip():
        return text_attr.strip()

    output = getattr(resp, "output", None)
    texts: List[str] = []

    def _gather(obj: Any):
        if obj is None:
            return
        if isinstance(obj, str):
            texts.append(obj)
            return
        if isinstance(obj, list):
            for item in obj:
                _gather(item)
            return
        if isinstance(obj, dict):
            typ = obj.get("type")
            text_val = obj.get("text")
            if isinstance(typ, str) and typ.lower() in {"output_text", "text"} and isinstance(text_val, str):
                texts.append(text_val)
            elif text_val is not None:
                _gather(text_val)
            value_val = obj.get("value")
            if isinstance(value_val, str):
                texts.append(value_val)
            elif value_val is not None:
                _gather(value_val)
            for key in ("content", "contents", "items", "data", "output", "outputs", "values"):
                if key in obj:
                    _gather(obj[key])
            return
        for attr_name in ("text", "value"):
            if hasattr(obj, attr_name):
                val = getattr(obj, attr_name)
                if isinstance(val, str):
                    texts.append(val)
                elif val is not None:
                    _gather(val)
        for attr_name in ("content", "contents"):
            if hasattr(obj, attr_name):
                _gather(getattr(obj, attr_name))

    if output:
        _gather(output)
    if texts:
        combined = "".join(texts).strip()
        if combined:
            return combined

    dumped = _object_to_builtins(resp)
    if dumped is not None:
        _gather(dumped)
        if texts:
            combined = "".join(texts).strip()
            if combined:
                return combined
    return None


def _content_to_text(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        text = content.strip()
        return text or None

    texts: List[str] = []

    def _gather(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            stripped = obj.strip()
            if stripped:
                texts.append(stripped)
            return
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                _gather(item)
            return
        if isinstance(obj, dict):
            typ = obj.get("type")
            text_val = obj.get("text")
            if isinstance(text_val, str):
                if not isinstance(typ, str) or typ.lower() not in {"input_text", "user_message"}:
                    stripped = text_val.strip()
                    if stripped:
                        texts.append(stripped)
            elif text_val is not None:
                _gather(text_val)
            value_val = obj.get("value")
            if isinstance(value_val, str):
                stripped = value_val.strip()
                if stripped:
                    texts.append(stripped)
            elif value_val is not None:
                _gather(value_val)
            for key in (
                "content",
                "contents",
                "data",
                "values",
                "items",
                "parts",
                "choices",
                "message",
                "messages",
                "output",
                "outputs",
                "result",
                "response",
                "segments",
                "tool_calls",
            ):
                if key in obj:
                    _gather(obj[key])
            return
        for attr in (
            "text",
            "content",
            "value",
            "data",
            "parts",
            "choices",
            "message",
            "messages",
            "output",
            "outputs",
            "result",
            "response",
            "tool_calls",
        ):
            if hasattr(obj, attr):
                try:
                    _gather(getattr(obj, attr))
                except Exception:
                    continue

    _gather(content)
    if not texts:
        return None
    combined = "".join(texts).strip()
    return combined or None


def _object_to_builtins(obj: Any) -> Optional[Any]:
    if obj is None:
        return None
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        for kwargs in ({}, {"exclude_none": True}):
            try:
                return model_dump(**kwargs)
            except TypeError:
                continue
            except Exception:
                break
    for name in ("model_dump_json", "dict", "to_dict"):
        method = getattr(obj, name, None)
        if not callable(method):
            continue
        if name == "dict" and callable(model_dump):
            continue
        try:
            data = method()
            if name == "model_dump_json" and isinstance(data, str):
                try:
                    return json.loads(data)
                except Exception:
                    continue
            return data
        except TypeError:
            try:
                data = method(exclude_none=True)
            except Exception:
                continue
            if name == "model_dump_json" and isinstance(data, str):
                try:
                    return json.loads(data)
                except Exception:
                    continue
            return data
        except Exception:
            continue
    return None


def _extract_text_from_chat_choice(choice: Any) -> Optional[str]:
    candidate = None
    if isinstance(choice, dict):
        if "message" in choice:
            candidate = _extract_text_from_chat_choice(choice["message"])
            if candidate:
                return candidate
        text = choice.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        candidate = _content_to_text(choice.get("content"))
        if candidate:
            return candidate
        for key in ("delta", "result", "response"):
            if key in choice:
                candidate = _extract_text_from_chat_choice(choice[key])
                if candidate:
                    return candidate
        return None

    message_attr = getattr(choice, "message", None)
    if message_attr is not None:
        candidate = _extract_text_from_chat_choice(message_attr)
        if candidate:
            return candidate

    text_attr = getattr(choice, "text", None)
    if isinstance(text_attr, str) and text_attr.strip():
        return text_attr.strip()

    content_attr = getattr(choice, "content", None)
    candidate = _content_to_text(content_attr)
    if candidate:
        return candidate

    for attr in ("delta", "result", "response"):
        nested = getattr(choice, attr, None)
        if nested is not None:
            candidate = _extract_text_from_chat_choice(nested)
            if candidate:
                return candidate

    return None


def _invoke_openai(client: OpenAI, messages: List[Dict[str, Any]], max_tokens: int, temperature: float) -> Tuple[str, str]:
    last_exc: Optional[Exception] = None
    responses_api = getattr(client, "responses", None)
    if responses_api is not None:
        try:
            resp = responses_api.create(
                model=OPENAI_MODEL,
                input=_responses_input_from_messages(messages),
            )
            text = _extract_text_from_openai_response(resp)
            if text:
                return text, "responses"
            raise RuntimeError("Пустой ответ от Responses API")
        except Exception as ex:
            last_exc = ex

    chat_api = getattr(client, "chat", None)
    completions_api = getattr(chat_api, "completions", None) if chat_api else None
    if completions_api is not None:
        try:
            resp = completions_api.create(
                model=OPENAI_MODEL,
                messages=messages,
            )
        except Exception as ex:
            last_exc = ex
        else:
            choices = getattr(resp, "choices", None)
            if isinstance(choices, list) and choices:
                for choice in choices:
                    content = _extract_text_from_chat_choice(choice)
                    if content:
                        return content, "chat.completions"

            fallback = _content_to_text(resp)
            if fallback:
                return fallback, "chat.completions"

            dumped = _object_to_builtins(resp)
            if dumped is not None:
                fallback = _content_to_text(dumped)
                if fallback:
                    return fallback, "chat.completions"

            debug_payload = dumped if dumped is not None else resp
            try:
                if isinstance(debug_payload, (dict, list)):
                    serialized = json.dumps(debug_payload, ensure_ascii=False)
                else:
                    serialized = str(debug_payload)
            except Exception:
                serialized = repr(debug_payload)
            admin_debug_log(
                "LLM[chat.completions] Не удалось извлечь текст. Сырой ответ: "
                f"{trim(serialized, 800)}"
            )
            raise RuntimeError("Пустой ответ от chat.completions")

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("У клиента OpenAI нет поддерживаемых методов Responses/chat")

def _call_openai_with_keys(messages: List[Dict],
                           max_tokens: int = 64,
                           temperature: float = 0.0,
                           purpose: str = "") -> str:
    label = purpose or "general"

    keys = db_list_keys()
    env_sk = os.environ.get("OPENAI_API_KEY")
    if env_sk and env_sk not in keys:
        keys.append(env_sk)

    if not keys:
        admin_debug_log(
            f"LLM[{label}] (без ключей)\nЗапрос:\n(нет доступных ключей)\n"
            "Ответ:\nОшибка: Нет ни одного OpenAI API Key\nПочему так?"
        )
        raise KeysExhaustedError("Нет ни одного OpenAI API Key")

    last_exc = None
    for round_idx in (1, 2):
        for sk in keys:
            key_masked = mask_secret(sk, 8, 6)
            try:
                client = _openai_client_for_key(sk)
            except Exception as ex:
                last_exc = ex
                admin_debug_log(f"LLM[{label}] Ошибка при инициализации ключа {key_masked}: {ex}")
                continue

            local_exc: Optional[Exception] = None
            for attempt in range(LLM_MAX_RETRIES_LOCAL + 1):
                payload = copy.deepcopy(messages or [])
                try:
                    answer, method = _invoke_openai(client, payload, max_tokens, temperature)
                    if isinstance(answer, str) and answer.strip():
                        text = answer.strip()
                        _log_llm_chat(
                            label=label,
                            key_masked=key_masked,
                            round_idx=round_idx,
                            attempt_idx=attempt + 1,
                            messages=payload,
                            response_text=text,
                            method=method,
                        )
                        return text
                    raise RuntimeError("Пустой ответ модели OpenAI")
                except Exception as ex:
                    local_exc = ex
                    _log_llm_chat(
                        label=label,
                        key_masked=key_masked,
                        round_idx=round_idx,
                        attempt_idx=attempt + 1,
                        messages=payload,
                        error=ex,
                    )
                    time.sleep(0.2 * (attempt + 1))

            last_exc = local_exc
    err_text = trim(_format_exception(last_exc) if last_exc else "Неизвестная ошибка", 400)
    admin_debug_log(
        f"LLM[{label}] (исчерпаны ключи)\nЗапрос:\n(повторяющийся запрос)\n"
        f"Ответ:\nОшибка: {err_text}\nПочему так?"
    )
    raise KeysExhaustedError(f"Не удалось получить ответ от модели: {last_exc}")

def _normalize_yesno(s: str) -> Optional[str]:
    if not s: return None
    t = (s or "").strip().lower()
    t = re.sub(r'[^\wа-яё-]+', ' ', t, flags=re.I).strip()
    t = t.split()[0] if t else ""
    yes = {"да","yes","y","ok","true","так","oui","si","ja"}
    no  = {"нет","no","n","false","nie","nein","non"}
    if t in yes: return "Да"
    if t in no:  return "Нет"
    if t.startswith("да"):  return "Да"
    if t.startswith("нет"): return "Нет"
    return None


def _split_yesno_details(text: str) -> Tuple[Optional[str], str]:
    if text is None:
        return None, ""
    stripped = (text or "").strip()
    if not stripped:
        return None, ""
    m = re.match(r'^(да|нет)\b', stripped, flags=re.I)
    if not m:
        return None, stripped
    answer = "Да" if m.group(1).lower().startswith("да") else "Нет"
    remainder = stripped[m.end():]
    remainder = re.sub(r'^[\s\.:;,\-–—]+', '', remainder)
    return answer, remainder.strip()


def _parse_yesno_list(raw: str) -> Tuple[Optional[str], List[str]]:
    answer, remainder = _split_yesno_details(raw)
    if answer == "Нет":
        return answer, []
    text = remainder if remainder else ((raw or "").strip())
    if not text:
        return answer, []
    tokens = [t.strip() for t in re.split(r"[|,\n]+", text) if t and t.strip()]
    cleaned: List[str] = []
    seen: set = set()
    for token in tokens:
        if token not in seen:
            cleaned.append(token)
            seen.add(token)
    return answer, cleaned


def llm_yesno(question: str, purpose: str) -> str:
    system_prompt = (
        "Ты аккуратный русскоязычный ассистент контроля качества. "
        "Отвечай ТОЛЬКО одним словом: «Да» или «Нет». Никаких комментариев."
    )
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    raw = _call_openai_with_keys(messages=base_messages, max_tokens=4, temperature=0.0, purpose=purpose)
    yn = _normalize_yesno(raw)
    if yn in ("Да", "Нет"):
        return yn
    reminder_text = (question or "").strip()
    if reminder_text:
        reminder_text += "\n\nОтветь одним словом: «Да» или «Нет»."
    else:
        reminder_text = "Ответь одним словом: «Да» или «Нет»."
    reminder_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": reminder_text}
    ]
    raw2 = _call_openai_with_keys(messages=reminder_messages, max_tokens=4, temperature=0.0, purpose=f"{purpose} (retry)")
    yn2 = _normalize_yesno(raw2)
    if yn2 in ("Да", "Нет"):
        return yn2
    raise RuntimeError(
        f"[{purpose}] Не удалось интерпретировать ответы модели: первичный={raw!r}, повторный={raw2!r}"
    )

def llm_text(question: str, purpose: str, max_tokens: int = 256) -> str:
    messages = [
        {"role": "system",
         "content": "Ты краткий русскоязычный ассистент. Пиши только то, что просят, без лишних слов."},
        {"role": "user", "content": question}
    ]
    return _call_openai_with_keys(messages=messages, max_tokens=max_tokens, temperature=0.0, purpose=purpose)

# =========================================================
# ================= DOCX без KeyError (patch) =============
# =========================================================
COMMENTS_XML_STUB = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'''
COMMENTS_EXT_XML_STUB = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w15:commentsEx xmlns:w15="http://schemas.microsoft.com/office/2012/wordml"/>'''

def _ensure_content_type_override(ct_root, part_name: str, content_type: str):
    ns = "{http://schemas.openxmlformats.org/package/2006/content-types}"
    exists = ct_root.xpath(f'./*[@PartName="{part_name}"]')
    if not exists:
        ov = etree.SubElement(ct_root, f"{ns}Override")
        ov.set("PartName", part_name)
        ov.set("ContentType", content_type)

def patch_docx_for_python_docx(path: str) -> str:
    with zipfile.ZipFile(path, "r") as zin:
        names = set(zin.namelist())
        need_comments    = ("word/comments.xml" not in names)
        need_comments_ex = ("word/commentsExtended.xml" not in names)
        if not (need_comments or need_comments_ex or ("[Content_Types].xml" not in names)):
            return path

        out_path = os.path.join(os.path.dirname(path), f"patched_{os.path.basename(path)}")
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "[Content_Types].xml":
                    try:
                        root = etree.fromstring(data)
                        if need_comments:
                            _ensure_content_type_override(root, "/word/comments.xml",
                                "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml")
                        if need_comments_ex:
                            _ensure_content_type_override(root, "/word/commentsExtended.xml",
                                "application/vnd.ms-word.commentsExt+xml")
                        data = etree.tostring(root, encoding="utf-8", xml_declaration=True)
                    except Exception:
                        pass
                zout.writestr(item, data)

            if need_comments:
                zout.writestr("word/comments.xml", COMMENTS_XML_STUB)
            if need_comments_ex:
                zout.writestr("word/commentsExtended.xml", COMMENTS_EXT_XML_STUB)
    return out_path

# =========================================================
# =========== СБРОС РАЗМЕРОВ КАРТИНОК ДЛЯ ZIP-ВЫГРУЗКИ ====
# =========================================================
def reset_docx_image_sizes_to_native(src_path: str) -> str:
    """
    Создаёт копию DOCX, где размеры всех картинок выставлены в 100% от исходника.
    Обновляются DrawingML (wp:extent, a:xfrm/a:ext) и VML (width/height в стиле).
    Возвращает путь к новому файлу (imgfix_*.docx). При ошибке вернёт исходный путь.
    """
    out_path = os.path.join(os.path.dirname(src_path), f"imgfix_{os.path.basename(src_path)}")
    try:
        with zipfile.ZipFile(src_path, "r") as zin:
            names = set(zin.namelist())

            # реальные размеры изображений в word/media/*
            img_dims: Dict[str, Dict[str, Any]] = {}
            for name in names:
                if not name.lower().startswith("word/media/"):
                    continue
                if name.endswith("/"):
                    continue
                try:
                    with Image.open(io.BytesIO(zin.read(name))) as im:
                        w, h = im.size  # px
                        img_dims[name] = {
                            "cx": int(round(w * 9525)),       # EMU (96dpi)
                            "cy": int(round(h * 9525)),
                            "wpt": float(w) * 72.0 / 96.0,    # pt (для VML)
                            "hpt": float(h) * 72.0 / 96.0,
                        }
                except Exception:
                    continue

            if not img_dims:
                with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                    for it in zin.infolist():
                        zout.writestr(it, zin.read(it.filename))
                return out_path

            def parse_rels(part_xml_path: str) -> Dict[str, str]:
                """rid -> абсолютный путь внутри zip (word/.../media/...)"""
                base_dir = "word"
                rels_path = f"word/_rels/{pp.basename(part_xml_path)}.rels"
                m: Dict[str, str] = {}
                if rels_path not in names:
                    return m
                try:
                    root = etree.fromstring(zin.read(rels_path))
                    rels = root.xpath('/*[local-name()="Relationships"]/*[local-name()="Relationship"]')
                    for rnode in rels:
                        rid = rnode.get("Id")
                        tgt = rnode.get("Target")
                        mode = (rnode.get("TargetMode") or "").lower()
                        typ = (rnode.get("Type") or "").lower()
                        if not rid or not tgt or mode == "external":
                            continue
                        if ("/image" in typ) or ("/media/" in tgt.lower()):
                            t = tgt.lstrip("/")
                            abs_path = pp.normpath(pp.join(base_dir, t))
                            m[rid] = abs_path
                except Exception:
                    pass
                return m

            def update_vml_style(style: str, wpt: float, hpt: float) -> str:
                s = style or ""
                if re.search(r'(?i)width:', s):
                    s = re.sub(r'(?i)width:[^;]+', f'width:{wpt:.2f}pt', s)
                else:
                    s = f'width:{wpt:.2f}pt;' + s
                if re.search(r'(?i)height:', s):
                    s = re.sub(r'(?i)height:[^;]+', f'height:{hpt:.2f}pt', s)
                else:
                    s = f'height:{hpt:.2f}pt;' + s
                s = re.sub(r'(?i)mso-(?:width|height)-percent:[^;]+;?', '', s)
                s = ";".join([p for p in [x.strip() for x in s.split(";")] if p])
                return s

            def process_xml(name: str) -> Optional[bytes]:
                try:
                    raw = zin.read(name)
                    if (b"<a:blip" not in raw) and (b"<v:imagedata" not in raw):
                        return None
                    rid_map = parse_rels(name)
                    root = etree.fromstring(raw)
                    changed = False

                    # DrawingML
                    for blip in root.xpath('.//a:blip', namespaces=NS):
                        rid = blip.get('{%s}embed' % NS['r']) or blip.get('{%s}link' % NS['r'])
                        if not rid:
                            continue
                        target = rid_map.get(rid)
                        dims = img_dims.get(target or "")
                        if not dims:
                            continue
                        cx, cy = dims["cx"], dims["cy"]

                        par = blip.getparent()
                        while par is not None:
                            q = etree.QName(par)
                            if q.namespace == NS['wp'] and q.localname in ("inline", "anchor"):
                                break
                            par = par.getparent()
                        if par is not None:
                            for ext in par.xpath('./wp:extent', namespaces=NS):
                                if ext.get("cx") != str(cx) or ext.get("cy") != str(cy):
                                    ext.set("cx", str(cx))
                                    ext.set("cy", str(cy))
                                    changed = True

                        node = blip.getparent()
                        pic_node = None
                        while node is not None:
                            q = etree.QName(node)
                            if q.namespace == NS['pic'] and q.localname == "pic":
                                pic_node = node
                                break
                            node = node.getparent()
                        if pic_node is not None:
                            for ex in pic_node.xpath('.//pic:spPr/a:xfrm/a:ext', namespaces=NS):
                                if ex.get("cx") != str(cx) or ex.get("cy") != str(cy):
                                    ex.set("cx", str(cx))
                                    ex.set("cy", str(cy))
                                    changed = True

                    # VML
                    for imd in root.xpath('.//v:imagedata', namespaces=NS):
                        rid = imd.get('{%s}id' % NS['r']) or imd.get('{%s}relid' % NS['o'])
                        if not rid:
                            continue
                        target = rid_map.get(rid)
                        dims = img_dims.get(target or "")
                        if not dims:
                            continue
                        wpt, hpt = dims["wpt"], dims["hpt"]
                        shape = imd.getparent()
                        if shape is not None:
                            old_style = shape.get("style") or ""
                            new_style = update_vml_style(old_style, wpt, hpt)
                            if new_style != old_style:
                                shape.set("style", new_style)
                                changed = True

                    if changed:
                        return etree.tostring(root, encoding="utf-8", xml_declaration=True)
                    return None
                except Exception:
                    return None

            # модифицируем нужные XML
            mods: Dict[str, bytes] = {}
            xml_parts = [
                n for n in names
                if n.startswith("word/") and n.endswith(".xml") and (
                    n.endswith("document.xml")
                    or pp.basename(n).startswith("header")
                    or pp.basename(n).startswith("footer")
                )
            ]
            for n in xml_parts:
                m = process_xml(n)
                if m is not None:
                    mods[n] = m

            with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for it in zin.infolist():
                    data = mods.get(it.filename)
                    if data is not None:
                        zout.writestr(it, data)
                    else:
                        zout.writestr(it, zin.read(it.filename))
        return out_path
    except Exception:
        return src_path

# =========================================================
# ================= DOCX helpers (read) ===================
# =========================================================
NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'wp': 'http://schemas.openxmlformats.org/wordprocessingDrawing',
    'v': 'urn:schemas-microsoft-com:vml',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'o': 'urn:schemas-microsoft-com:office:office',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
}

def _oxml(node):
    if hasattr(node, "element"): return node.element
    if hasattr(node, "_element"): return node._element
    if hasattr(node, "_p"): return node._p
    if hasattr(node, "_tbl"): return node._tbl
    return None

def iter_block_items(parent):
    from docx.oxml.ns import qn
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)

def element_image_count(el) -> int:
    root = _oxml(el)
    try:
        blips = root.xpath('.//*[local-name()="blip"]')
        vmls  = root.xpath('.//*[local-name()="imagedata"]')
        return len(blips) + len(vmls)
    except Exception:
        return 0

def document_total_images(doc: Document) -> int:
    try:
        return len(doc.element.body.xpath('.//*[local-name()="blip"]')) + len(doc.element.body.xpath('.//*[local-name()="imagedata"]'))
    except Exception:
        return 0

def get_heading_level(p: Paragraph) -> Optional[int]:
    try:
        style = p.style
        sid = (style.style_id or "").lower() if style else ""
        m = re.search(r'heading\s*([1-6])', sid)
        if m: return int(m.group(1))
        sname = (style.name or "").lower() if style else ""
        if any(tok in sname for tok in ["heading","заголовок","überschrift","encabezado","titre","rubrique","nadpis","naslov","intestazione","encabezamiento","επικεφαλίδα","标题","見出し","제목"]):
            m2 = re.search(r'([1-6])', sname)
            if m2: return int(m2.group(1))
        pPr = p._p.pPr
        if pPr is not None and pPr.outlineLvl is not None:
            return int(pPr.outlineLvl.val) + 1
    except Exception:
        pass
    return None

def get_heading_level_strict(p: Paragraph) -> Optional[int]:
    try:
        style = p.style
        sid = (style.style_id or "").lower() if style else ""
        m = re.search(r'heading\s*([1-6])', sid)
        if m: return int(m.group(1))
        sname = (style.name or "").lower() if style else ""
        if any(tok in sname for tok in ["heading","заголовок"]):
            m2 = re.search(r'([1-6])', sname)
            if m2: return int(m2.group(1))
    except Exception:
        pass
    return None

def _norm_text(s: str) -> str:
    return strip_invisible((s or "").replace("\xa0", " "))

# =========================================================
# ===================== EEAT парсинг ======================
# =========================================================
_MARKER_RE = re.compile(r'(MT:|MD:|MK:)\s*')

def _split_markers_payloads(s: str) -> List[Tuple[str, str, str]]:
    s2 = _norm_text(s)
    out: List[Tuple[str, str, str]] = []
    it = list(_MARKER_RE.finditer(s2))
    for i, m in enumerate(it):
        marker = m.group(1)
        start = m.end()
        end = it[i+1].start() if i + 1 < len(it) else len(s2)
        payload = s2[start:end].strip()
        raw_from_marker = s2[m.start():]
        out.append((marker[:-1], payload, raw_from_marker))
    return out

def _extract_slug_and_rest(line: str) -> Tuple[Optional[str], str, bool]:
    s = _norm_text(line)
    m = re.match(r'^\s*/(?P<core>[^ \t\r\n]+?)(?=(?:MT:|MD:|MK:)|\s|$)', s)
    if not m:
        return None, s, False
    slug = "/" + m.group('core')
    rest = s[m.end():]
    had_leading_space = bool(re.match(r'^\s+/', s))
    return slug, rest, had_leading_space

def parse_eeat_sections(doc: Document):
    sections = []
    current = None

    def flush_current():
        nonlocal current
        if current:
            for key in ("MT","MD","MK"):
                parts = current.get(key+"_parts") or []
                payload = " ".join([_norm_text(x) for x in parts if x]).strip()
                payload = re.sub(r"\s+", " ", payload)
                current[key] = payload or None
            current.setdefault('content_images', 0)
            current.setdefault('content_texts', [])
            sections.append(current)
            current = None

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            raw_text = block.text or ""
            orig_lines = re.split(r'(?:\r|\n)+', raw_text.replace('\xa0',' '))
            lvl = get_heading_level(block)

            # НОВОЕ: MK заканчивается на первом заголовке
            if current and current.get("state") == "MK" and lvl is not None:
                current["state"] = None  # прекращаем сбор MK на первом заголовке после MK:

            for ln_raw in orig_lines:
                if not (ln_raw or "").strip():
                    continue

                slug, rest, had_leading_space = _extract_slug_and_rest(ln_raw)

                if slug:
                    s_norm = _norm_text(ln_raw)
                    next_marker_pos = len(s_norm)
                    for mk in ("MT:", "MD:", "MK:"):
                        pos = s_norm.find(mk)
                        if pos != -1:
                            next_marker_pos = min(next_marker_pos, pos)
                    raw_before_markers = s_norm[:next_marker_pos]
                    has_space_in_slug_area = bool(re.search(r'^/\S+\s+\S', raw_before_markers))

                    flush_current()
                    current = {
                        "slug": slug, "slug_heading_level": lvl,
                        "slug_leading_space": had_leading_space,
                        "slug_has_space": has_space_in_slug_area,
                        "MT": None, "MD": None, "MK": None,
                        "MT_raw": None, "MD_raw": None, "MK_raw": None,
                        "mt_h": None, "md_h": None, "mk_h": None,
                        "MT_parts": [], "MD_parts": [], "MK_parts": [],
                        "state": None, "content_images": 0, "content_texts": [],
                        "h1": None,
                        # ======= НОВОЕ: флаги форматирования маркеров =======
                        "MT_leading_space": False, "MD_leading_space": False, "MK_leading_space": False,
                        "MT_no_space_after_colon": False, "MD_no_space_after_colon": False, "MK_no_space_after_colon": False,
                    }
                    ln_for_markers = rest
                else:
                    ln_for_markers = ln_raw

                if not current:
                    continue

                if current.get("h1") is None and lvl == 1 and ln_for_markers.strip():
                    current["h1"] = _norm_text(ln_for_markers.strip())

                chunks = _split_markers_payloads(ln_for_markers)
                if chunks:
                    for marker, payload, raw_from_marker in chunks:
                        # ======= НОВОЕ (исправлено): проверка пробела до и после «:» =======
                        s2_ext = _norm_text(ln_for_markers)
                        idx_start = s2_ext.find(raw_from_marker)
                        # 1) Ведущий пробел перед маркером
                        if idx_start > 0 and s2_ext[:idx_start].strip() == "":
                            current[f"{marker}_leading_space"] = True
                        # Доп. надёжная проверка (quirks Word): начало строки + пробелы + маркер
                        if re.match(r'^\s+' + re.escape(marker) + r':', s2_ext):
                            current[f"{marker}_leading_space"] = True
                        # 2) Пробел обязателен после двоеточия
                        if raw_from_marker.startswith(marker + ":"):
                            after = raw_from_marker[len(marker) + 1:]
                            if after and not after.startswith(" "):
                                current[f"{marker}_no_space_after_colon"] = True
                        # ======= конец блока =======

                        current["state"] = marker
                        key = marker + "_parts"
                        raw_key = marker + "_raw"
                        lvl_key = marker.lower() + "_h"
                        if current.get(raw_key) is None:
                            current[raw_key] = raw_from_marker
                        if current.get(lvl_key) is None and lvl:
                            current[lvl_key] = lvl
                        if payload:
                            current[key].append(payload)
                    continue

                if current["state"] == "MT":
                    current["MT_parts"].append(_norm_text(ln_for_markers))
                elif current["state"] == "MD":
                    current["MD_parts"].append(_norm_text(ln_for_markers))
                elif current["state"] == "MK":
                    current["MK_parts"].append(_norm_text(ln_for_markers))
                else:
                    current["content_texts"].append(_norm_text(ln_for_markers))

            if current:
                imgs = element_image_count(block)
                if imgs > 0:
                    current["content_images"] += imgs

        elif isinstance(block, Table):
            if current:
                current["content_images"] += element_image_count(block)

    flush_current()
    return sections

# =========================================================
# ===================== ПРОМПТЫ LLM =======================
# =========================================================
def _prepare_text_lines_for_prompt(text: Union[str, Sequence[str], None]) -> List[str]:
    if text is None:
        return []
    if isinstance(text, str):
        candidates = [text]
    else:
        candidates = []
        for chunk in text:
            if chunk is None:
                continue
            candidates.append(str(chunk))
    lines: List[str] = []
    for chunk in candidates:
        for line in re.split(r'(?:\r?\n)+', chunk):
            cleaned = clean_for_prompt(line)
            if cleaned:
                lines.append(cleaned)
    return lines


def _format_text_for_prompt(
    text: Union[str, Sequence[str], None],
    prefix: str = "Текст:",
    max_chars: Optional[int] = None,
) -> str:
    lines = _prepare_text_lines_for_prompt(text)
    if not lines:
        body = "—"
    else:
        body = "\n".join(lines)
        if max_chars is not None:
            body = clip(body, max_chars)
    return f"{prefix}\n{body}"


def _format_headings_for_prompt(headings: Sequence[Tuple[Optional[int], str]]) -> str:
    formatted: List[str] = []
    for entry in headings:
        lvl: Optional[int]
        text: Optional[str]
        if isinstance(entry, tuple) and len(entry) >= 2:
            lvl, text = entry[0], entry[1]
        else:
            lvl, text = None, entry  # type: ignore[assignment]
        label = f"H{lvl}" if isinstance(lvl, int) and 1 <= lvl <= 6 else "H?"
        text_clean = clean_for_prompt(text)
        if text_clean:
            text_clean = clip(text_clean, 400)
        formatted.append(f"{label}: {text_clean or '—'}")
    return "\n".join(formatted) if formatted else "—"


def build_article_headings_vs_text_prompt(
    headings: Sequence[Tuple[Optional[int], str]],
    body: Union[str, Sequence[str], None],
) -> str:
    htxt = _format_headings_for_prompt(headings)
    body_block = _format_text_for_prompt(body, prefix="Текст:", max_chars=ART_PROMPT_MAX_CHARS)
    return (
        "Ты проверяешь совпадение языка заголовков и основного текста.\n"
        "Каждая строка в блоке <HEADINGS> начинается с H1:, H2: и т.д.\n"
        "Смотри только в блоки <HEADINGS> и <TEXT> ниже и игнорируй язык этих инструкций.\n"
        "Определи доминирующий язык блока <TEXT> по грамматике и служебным словам (диакритику игнорируй: á→a, ü→u и т.п.).\n"
        "Игнорируй бренды, домены/URL, аббревиатуры, числа, валюты и одиночные заимствования.\n"
        "Сравни язык каждого заголовка из <HEADINGS> с языком <TEXT>.\n"
        "Если хотя бы один заголовок на другом языке — ответь «Нет». Если все совпадает — ответь «Да».\n"
        "Отвечай строго одним словом «Да» или «Нет». Никаких комментариев.\n"
        f"\n<HEADINGS>\n{htxt}\n</HEADINGS>"
        f"\n\n<TEXT>\n{body_block}\n</TEXT>"
    )

def build_article_list_problem_headings_prompt(
    headings: Sequence[Tuple[Optional[int], str]],
    body: Union[str, Sequence[str], None],
) -> str:
    htxt = _format_headings_for_prompt(headings)
    body_block = _format_text_for_prompt(body, prefix="Текст:", max_chars=ART_PROMPT_MAX_CHARS)
    return (
        "Определи ДОМИНИРУЮЩИЙ язык <TEXT> СТРОГО по блоку <TEXT> ниже. Язык промпта на котором это написано не причем. Не упоминай его."
        "Игнорируй язык этих инструкций и всё, что вне тегов. Только язык с <TEXT>. "
        "Диакритику игнорируй (á→a, ü→u и т.п.). "
        "Игнорируй бренды/имена, домены/URL, аббревиатуры, числа/валюты и одиночные англ. заимствования.\n"
        "Сравни каждый заголовок из <HEADINGS> с языком <TEXT>. "
        "Каждая строка в <HEADINGS> имеет вид H1:, H2: и т.д.\n"
        "Если заголовок совпадает по языку — пропусти. "
        "Если нет — укажи точную проблему.\n"
        "Начни ответ с «Да», если найдены ошибки, иначе ответь ровно «Нет».\n"
        "Если ответ «Да», на следующих строках перечисли проблемные заголовки.\n"
        "Каждый оформляй так: 'проблемный заголовок' — краткое пояснение.\n"
        f"\n<HEADINGS>\n{htxt}\n</HEADINGS>"
        f"\n\n<TEXT>\n{body_block}\n</TEXT>"
    )


def build_slug_consistency_prompt(
    slug: str,
    mt: str,
    md: str,
    mk: str,
    h1: str,
    content: Union[str, Sequence[str], None],
) -> str:
    slug = (slug or "").strip()
    mt = (mt or "").strip()
    md = (md or "").strip()
    mk = (mk or "").strip()
    h1 = (h1 or "").strip()
    body = []
    body.append("Проверь, отражает ли SLUG тему и язык секции.")
    body.append("SLUG пишут без диакритики — это нормально, если он всё равно про ту же тему.")
    body.append("Если SLUG заметно про другую тему или язык — ответь «Нет». Иначе ответь «Да».")
    body.append("Отвечай строго одним словом «Да» или «Нет». Никаких пояснений.")
    body.append("")
    body.append(slug or "—")
    body.append("")
    if mt: body.append(f"MT: {mt}")
    if md: body.append(f"MD: {md}")
    if mk: body.append(f"MK: {mk}")
    if h1: body.append(f"H1: {h1}")
    body.append("")
    body.append(_format_text_for_prompt(content, prefix="Текст:", max_chars=EEAT_PROMPT_MAX_CHARS))
    return "\n".join(body)

def _format_meta_block(mt: str, md: str, mk: str, h1: str) -> str:
    lines = ["MT: " + ((mt or "").strip() or "—"),
             "MD: " + ((md or "").strip() or "—"),
             "MK: " + ((mk or "").strip() or "—"),
             "H1: " + ((h1 or "").strip() or "—")]
    return "\n".join(lines)


def build_section_lang_match_prompt(mt: str, md: str, mk: str, h1: str, content: Union[str, Sequence[str], None]) -> str:
    content_block = _format_text_for_prompt(content, prefix="Текст:", max_chars=EEAT_PROMPT_MAX_CHARS)
    meta_block = _format_meta_block(mt, md, mk, h1)
    return (
        "Ты проверяешь, совпадает ли язык MT/MD/MK/H1 с языком текста секции.\n"
        "Анализируй только содержимое блоков <META> и <CONTENT> ниже и игнорируй язык этих инструкций.\n"
        "Определи доминирующий язык <CONTENT> по грамматике и служебным словам (диакритику игнорируй: á→a, ü→u и т.п.).\n"
        "Игнорируй бренды, домены/URL, числа, валюты и одиночные заимствования.\n"
        "Сравни язык MT, MD, MK и H1 с языком <CONTENT>.\n"
        "Если хотя бы один элемент явно на другом языке — ответь «Нет». Если всё совпадает — ответь «Да».\n"
        "Отвечай строго одним словом «Да» или «Нет». Никаких пояснений.\n"
        f"\n<META>\n{meta_block}\n</META>"
        f"\n\n<CONTENT>\n{content_block}\n</CONTENT>"
    )


def build_section_lang_list_bad_prompt(mt: str, md: str, mk: str, h1: str, content: Union[str, Sequence[str], None]) -> str:
    content_block = _format_text_for_prompt(content, prefix="Текст:", max_chars=EEAT_PROMPT_MAX_CHARS)
    meta_block = _format_meta_block(mt, md, mk, h1)
    return (
        "Если найдёшь несоответствие языков — ответь «Да», иначе ответь ровно «Нет».\n"
        "Если ответ «Да», перечисли ТОЛЬКО ярлыки из набора MT | MD | MK | H1, разделяя их через « | ». Никаких пояснений.\n"
        "Анализируй исключительно данные внутри тегов <META> и <CONTENT> ниже. Игнорируй язык этих инструкций."
        "\nПомни: бренды, домены и одиночные заимствованные слова не считаются сменой языка; оценивай основную часть текста.\n"
        f"\n<META>\n{meta_block}\n</META>"
        f"\n\n<CONTENT>\n{content_block}\n</CONTENT>"
    )

def build_promo_scan_prompt(full_text: str) -> str:
    full_text = clip(clean_for_prompt(full_text), PROMO_MAX_CHARS)
    return (
        "Определи, есть ли в тексте настоящий ПРОМОКОД — заглавный буквенно-цифровой шаблон вроде «CODE123».\n"
        "Игнорируй служебные метки («MT», «MD», «MK», «H1», «H2», «H3», «H4») и единицы измерения («MB», «GB», «TB» и т.п.).\n"
        "Если нашёл хотя бы один промокод — ответь «Да». Если не нашёл — ответь «Нет».\n"
        "Отвечай строго одним словом «Да» или «Нет». Никаких комментариев.\n\n" + full_text
    )

def build_promo_extract_prompt(full_text: str) -> str:
    full_text = clip(clean_for_prompt(full_text), PROMO_MAX_CHARS)
    return (
        "Выпиши из текста только настоящие ПРОМОКОДЫ (заглавные буквенно-цифровые шаблоны). "
        "Игнорируй служебные метки («MT», «MD», «MK», «H1», «H2», «H3», «H4») и единицы "
        "измерения («MB», «GB», «TB» и т.п.). Если промокоды найдены, начни ответ с «Да» и на той же строке "
        "после двоеточия перечисли их через « | ». Если промокодов нет — ответь ровно «Нет».\n\n" + full_text
    )

def filter_promo_codes(raw: str) -> List[str]:
    if not raw:
        return []
    answer, tokens = _parse_yesno_list(raw)
    if answer == "Нет":
        return []
    parts = tokens
    if not parts:
        return []
    cleaned: List[str] = []
    seen: set = set()
    for part in parts:
        token = part.strip()
        if not token:
            continue
        norm = re.sub(r"[^0-9A-Z]", "", token.upper())
        if not norm:
            continue
        if norm in PROMO_IGNORE_CODES:
            continue
        if norm in seen:
            continue
        cleaned.append(token)
        seen.add(norm)
    return cleaned

# =========================================================
# ==================== ВАЛИДАЦИЯ СТАТЕЙ ===================
# =========================================================

# ===== НОВОЕ: строгий формат списка ключевых слов MK =====
_MK_LIST_STRICT_RE = re.compile(r'^\s*[^,;]+(?:, [^,;]+)*\s*$')

def _mk_list_is_strict(s: str) -> bool:
    if not (s or "").strip():
        return True
    return bool(_MK_LIST_STRICT_RE.fullmatch(s))

# ===== НОВОЕ: сканер маркеров во всех текстах (для статей) =====
def _validate_markers_in_plain_doc(doc: Document) -> List[str]:
    """
    Проверяет формат MT/MD/MK во всем документе (статьи):
    - запрет пробела перед маркером в начале строки
    - обязателен пробел после «:»
    - границы: MT=[MT:..до MD:], MD=[MD:..до MK:], MK=[MK:..до первого заголовка]
    - длины: MT<=65, MD<=156
    - MK: ключевые слова строго «a, b, c»
    """
    errors: List[str] = []
    state: Optional[str] = None  # 'MT'|'MD'|'MK'|None
    acc: Dict[str, List[str]] = {"MT": [], "MD": [], "MK": []}
    have: Dict[str, bool] = {"MT": False, "MD": False, "MK": False}

    def _finalize(kind: str, _reason: str):
        nonlocal state
        txt = " ".join(acc[kind]).strip()
        acc[kind].clear()
        have[kind] = False  # позволяем встретить тот же маркер позже
        if not txt:
            state = None
            return
        if kind == "MT" and len(txt) > 65:
            errors.append(f"Ошибка, MT: длина превышает 65 символов (сейчас {len(txt)}).")
        if kind == "MD" and len(txt) > 156:
            errors.append(f"Ошибка, MD: длина превышает 156 символов (сейчас {len(txt)}).")
        if kind == "MK":
            if not _mk_list_is_strict(txt):
                errors.append("Ошибка, после MK: ключевые слова должны быть строго через запятую с пробелом после нее, например: MK: key1, key2.")
            elif (' ' in txt) and (',' not in txt):
                errors.append("MK: Ключевые слова перечисляются через запятую.")
        state = None

    for block in iter_block_items(doc):
        lvl = get_heading_level(block) if isinstance(block, Paragraph) else None

        # На первом заголовке после MK — завершаем MK
        if state == "MK" and lvl is not None:
            _finalize("MK", "heading")

        if isinstance(block, Paragraph):
            # разбиваем абзац по мягким переносам/enter
            for ln in re.split(r'(?:\r|\n)+', (block.text or "").replace('\xa0', ' ')):
                s = _norm_text(ln)
                if not s:
                    continue

                # ВАЖНО: разбиваем строку на все маркеры внутри неё
                chunks = _split_markers_payloads(s)  # -> [(marker, payload, raw_from_marker), ...]
                if chunks:
                    for marker, payload, raw_from_marker in chunks:
                        # 1) проверка пробела ДО маркера в начале строки
                        idx_start = s.find(raw_from_marker)
                        if idx_start > 0 and s[:idx_start].strip() == "":
                            errors.append(f"Ошибка, перед {marker}: обнаружен пробел.")
                        # 2) обязателен пробел после «:»
                        if raw_from_marker.startswith(marker + ":"):
                            after = raw_from_marker[len(marker) + 1:]
                            if after and not after.startswith(" "):
                                errors.append(f"Ошибка, после {marker}: должен быть пробел.")

                        # переключение состояний и финализация предыдущего
                        if marker == "MD" and have["MT"]:
                            _finalize("MT", "MD start")
                        if marker == "MK" and have["MD"]:
                            _finalize("MD", "MK start")
                        if marker == "MT" and have["MK"]:
                            _finalize("MK", "new MT")

                        state = marker
                        have[marker] = True
                        if payload:
                            acc[marker].append(_norm_text(payload))
                    continue  # маркеры в этой строке уже разобрали

                # накапливаем текст в текущем состоянии
                if state in ("MT", "MD", "MK"):
                    acc[state].append(s)

        elif isinstance(block, Table):
            # заголовков у таблицы нет — MK не завершаем
            pass

    # Финализация на конце документа — завершаем всё, что было начато
    for _kind in ("MK", "MD", "MT"):
        if have[_kind]:
            _finalize(_kind, "eof")

    return errors

def validate_text_article(doc: Document, tag: Optional[str] = None) -> List[str]:
    errors: List[str] = []
    headings: List[Tuple[int, str]] = []
    body_texts: List[str] = []

    tag_clean = (tag or "").strip()
    section_label = f"Статья {tag_clean}" if tag_clean else "Статья"

    def purpose(task: str) -> str:
        return f"{section_label}: {task}"

    def build_numbering_maps(doc: Document):
        maps = {"num_to_abs": {}, "abs_to_fmt": {}}
        try:
            numbering_part = getattr(doc.part, "numbering_part", None)
            if numbering_part is None: return maps
            numbering = numbering_part.element
            nums = numbering.xpath('//*[local-name()="num"]')
            for num in nums:
                num_id = num.get('{%s}numId' % NS['w'])
                abs_nodes = num.xpath('./*[local-name()="abstractNumId"]')
                abs_val = None
                if abs_nodes: abs_val = abs_nodes[0].get('{%s}val' % NS['w'])
                if num_id and abs_val: maps["num_to_abs"][num_id] = abs_val
            abs_list = numbering.xpath('//*[local-name()="abstractNum"]')
            for an in abs_list:
                aid = an.get('{%s}abstractNumId' % NS['w'])
                if not aid: continue
                lvl_map = {}
                lvls = an.xpath('./*[local-name()="lvl"]')
                for lvl in lvls:
                    ilvl = lvl.get('{%s}ilvl' % NS['w'])
                    fmt_nodes = lvl.xpath('./*[local-name()="numFmt"]')
                    numfmt = fmt_nodes[0].get('{%s}val' % NS['w']) if fmt_nodes else None
                    if ilvl and numfmt: lvl_map[ilvl] = numfmt
                maps["abs_to_fmt"][aid] = lvl_map
        except Exception:
            pass
        return maps

    def paragraph_list_type(p: Paragraph, maps) -> Optional[str]:
        try:
            pPr = p._p.pPr
            if pPr is None or pPr.numPr is None: return None
            numPr = pPr.numPr
            numId = getattr(getattr(numPr, 'numId', None), 'val', None)
            ilvl = getattr(getattr(numPr, 'ilvl', None), 'val', None)
            if numId is None or ilvl is None: return None
            abs_id = maps["num_to_abs"].get(str(numId))
            if not abs_id: return None
            fmt = maps["abs_to_fmt"].get(abs_id, {}).get(str(ilvl))
            if not fmt: return None
            return "bullet" if fmt == "bullet" else "numbered"
        except Exception:
            return None

    num_maps = build_numbering_maps(doc)
    bullets = numbers = 0

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = (block.text or "").strip()
            lvl = get_heading_level(block)
            if lvl:
                headings.append((lvl, text))
                if len(text) > 350:
                    errors.append(f'Ошибка, текст отмечен как заголовок: "{first_words(text, 3)}"')
            else:
                body_texts.append(text)

            ltype = paragraph_list_type(block, num_maps)
            if ltype == "bullet":   bullets += 1
            elif ltype == "numbered": numbers += 1

        elif isinstance(block, Table):
            try:
                if len(block.rows) in (1,2):
                    errors.append("Таблица содержит слишком мало строк (1–2).")
            except Exception:
                pass

    if not headings:
        errors.append("Нет ни одного заголовка.")
    else:
        has_nonempty_h1 = any(lvl == 1 and (txt or "").strip() for lvl, txt in headings)
        if not has_nonempty_h1:
            errors.append("Нет ни одного заголовка H1 (или все H1 пустые).")
        first_nonempty = next(((lvl, txt) for (lvl, txt) in headings if (txt or "").strip()), None)
        if first_nonempty and first_nonempty[0] != 1:
            errors.append("Первый заголовок должен быть H1.")
        if not any(lvl == 2 for lvl,_ in headings): errors.append("Нет ни одного заголовка H2.")
        if not any(lvl == 3 for lvl,_ in headings): errors.append("Нет ни одного заголовка H3.")

    if bullets == 0: errors.append("Нет ни одного маркированного списка (bullet).")
    if numbers == 0: errors.append("Нет ни одного нумерованного списка (numbered).")
    if document_total_images(doc) < 2: errors.append("Картинок меньше двух (минимум 2).")

    try:
        heading_pairs = [(lvl, txt) for (lvl, txt) in headings if (txt or "").strip()]
        body_blocks = [t for t in body_texts if (t or "").strip()]
        body_sample_lines = _prepare_text_lines_for_prompt(body_blocks)
        body_join = clip(" ".join(body_sample_lines), ART_PROMPT_MAX_CHARS)
        script_mismatch_found = False
        if heading_pairs and body_join:
            body_script = _dominant_script(body_join)
            if body_script:
                label_body = _script_label(body_script)
                seen_script_notes: set = set()
                for _, heading in heading_pairs:
                    h_script = _dominant_script(heading)
                    if not h_script or h_script == body_script:
                        continue
                    script_mismatch_found = True
                    display = first_words(heading, 6)
                    if not display:
                        display = clip((heading or "").strip(), 60)
                    if not display:
                        display = "—"
                    key = (display, h_script)
                    if key in seen_script_notes:
                        continue
                    seen_script_notes.add(key)
                    errors.append(
                        "Язык: заголовок «{}» написан другим алфавитом ({}), "
                        "чем основной текст ({})".format(display, _script_label(h_script), label_body)
                    )
            prompt = build_article_headings_vs_text_prompt(heading_pairs, body_blocks)
            yn = llm_yesno(prompt, purpose=purpose("Язык заголовков vs текст"))
            if yn == "Нет" or script_mismatch_found:
                prompt2 = build_article_list_problem_headings_prompt(heading_pairs, body_blocks)
                bad_raw = llm_text(prompt2, purpose=purpose("Проблемные заголовки"), max_tokens=256)
                bad_raw = bad_raw.strip()
                yn_bad, bad_details = _split_yesno_details(bad_raw)
                bad_details = re.sub(r'\s*\|\s*', ' | ', (bad_details or "").strip())
                if yn_bad == "Нет" or not bad_details:
                    if not script_mismatch_found:
                        errors.append("Язык: заголовки не совпадают с языком основного текста.")
                else:
                    errors.append(
                        "Язык: заголовки не совпадают с языком основного текста. "
                        f"(Проблемные заголовки: {bad_details}.)"
                    )
    except KeysExhaustedError:
        raise
    except Exception as ex:
        errors.append(f"Нейросетевая проверка языка заголовков/текста не выполнена: {ex}")

    try:
        full_text = " ".join([t for t in (body_texts + [t for _, t in headings]) if t])
        q1 = build_promo_scan_prompt(full_text)
        has_code = llm_yesno(q1, purpose=purpose("Промокоды — скан"))
        if has_code == "Да":
            q2 = build_promo_extract_prompt(full_text)
            codes = llm_text(q2, purpose=purpose("Промокоды — извлечение"), max_tokens=256)
            codes = codes.strip()
            filtered = filter_promo_codes(codes)
            if filtered:
                errors.append(f"Обнаружен посторонний промокод: {' | '.join(filtered)}")
    except KeysExhaustedError:
        raise
    except Exception:
        pass

    # ===== НОВОЕ: строгие проверки MT/MD/MK во всех статьях =====
    try:
        errors.extend(_validate_markers_in_plain_doc(doc))
    except Exception:
        # не роняем проверку статей из-за служебной ошибки
        pass

    return errors

# =========================================================
# ===================== ВАЛИДАЦИЯ EEAT ====================
# =========================================================
def validate_eeat(doc: Document) -> Tuple[List[str], Dict]:
    errors: List[str] = []
    details = {"summary": {}, "section_blocks": []}

    eeat_scope_label = "EEAT (все секции)"

    def eeat_purpose(task: str) -> str:
        return f"{eeat_scope_label}: {task}"

    # Жёсткие правила по заголовкам
    headings_strict = []
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            t = block.text.strip()
            lvl = get_heading_level_strict(block)
            if lvl:
                headings_strict.append((lvl, t))
                if len(t) > 350:
                    errors.append(f'Ошибка, текст отмечен как заголовок: "{first_words(t, 3)}"')

    h1_texts = [t for lvl,t in headings_strict if lvl == 1 and t.strip()]
    if len(h1_texts) != 5:
        lst = "\n".join([f"      {i+1}) {txt}" for i, txt in enumerate(h1_texts)])
        errors.append(f"В EEAT должно быть ровно 5 H1; найдено {len(h1_texts)}.\n    Найдены H1:\n{lst if lst else '      —'}")
    if any(lvl == 3 for lvl,_ in headings_strict): errors.append("В EEAT не допускаются заголовки H3.")
    if any(lvl == 4 for lvl,_ in headings_strict): errors.append("В EEAT не допускаются заголовки H4.")

    # Парсим секции
    sections = parse_eeat_sections(doc)
    for i, sec in enumerate(sections):
        if i < len(h1_texts) and not sec.get("h1"):
            sec["h1"] = h1_texts[i]

    # По секциям (параллельно) — БЕЗ прямого индексирования списка sections
    slug_by_idx = {i + 1: (sec.get('slug') or '') for i, sec in enumerate(sections)}

    def check_section(idx: int, sec: Dict) -> Tuple[int, List[str]]:
        se: List[str] = []
        slug_raw = sec.get("slug") or ""
        slug_clean = strip_invisible(slug_raw)

        # SLUG форма
        if sec.get("slug_leading_space"):
            se.append("SLUG должен начинаться строго с '/' без пробела в начале строки.")
        if sec.get("slug_has_space"):
            se.append("В SLUG недопустимы пробелы — слова разделяются дефисами '-'.")
        if not re.match(r'^/[a-z0-9-]+$', slug_clean.strip().lower()):
            se.append("SLUG должен содержать только ASCII-символы [a-z0-9-] (без диакритики).")

        mt = (sec.get("MT") or "")
        md = (sec.get("MD") or "")
        mk = (sec.get("MK") or "")
        h1 = (sec.get("h1") or "")
        content_lines = [t for t in (sec.get("content_texts") or []) if (t or "").strip()]
        content_sample_lines = _prepare_text_lines_for_prompt(content_lines)
        content_text_sample = clip(" ".join(content_sample_lines), EEAT_PROMPT_MAX_CHARS)

        section_descriptor = f"EEAT секция #{idx}"
        extra = (slug_clean.strip() or h1.strip())
        if extra:
            section_descriptor += f" ({extra})"

        def section_purpose(task: str) -> str:
            return f"{section_descriptor}: {task}"

        content_script = _dominant_script(content_text_sample)
        if content_script:
            h1_script = _dominant_script(h1)
            if h1_script and h1_script != content_script:
                se.append(
                    "H1 написан другим алфавитом ({}) чем основной текст секции ({}).".format(
                        _script_label(h1_script), _script_label(content_script)
                    )
                )

        # ===== НОВОЕ: строгие проверки форматирования маркеров для EEAT (исправлено/группировка) =====
        lead_list = [m for m in ("MT", "MD", "MK") if sec.get(f"{m}_leading_space")]
        if lead_list:
            if len(lead_list) == 1:
                se.append(f"Ошибка, перед {lead_list[0]}: обнаружен пробел.")
            else:
                se.append("Ошибка, перед " + ", ".join([f"{m}:" for m in lead_list]) + " обнаружен пробел.")

        # отсутствие пробела после «:» — отдельные ошибки по каждому
        if sec.get("MT_no_space_after_colon"): se.append("Ошибка, после MT: должен быть пробел.")
        if sec.get("MD_no_space_after_colon"): se.append("Ошибка, после MD: должен быть пробел.")
        if sec.get("MK_no_space_after_colon"): se.append("Ошибка, после MK: должен быть пробел.")

        # длины MT/MD в EEAT
        if mt and len(mt) > 65:
            se.append(f"Ошибка, MT: длина превышает 65 символов (сейчас {len(mt)}).")
        if md and len(md) > 156:
            se.append(f"Ошибка, MD: длина превышает 156 символов (сейчас {len(md)}).")

        # формат списка MK
        if mk:
            if not _mk_list_is_strict(mk):
                se.append("Ошибка, после MK: ключевые слова должны быть строго через запятую с пробелом после нее, например: MK: key1, key2.")
            elif (' ' in mk) and (',' not in mk):
                se.append("MK: Ключевые слова перечисляются через запятую.")

        # 1) slug ↔ содержимое
        try:
            prompt = build_slug_consistency_prompt(slug_clean, mt, md, mk, h1, content_lines)
            yn = llm_yesno(prompt, purpose=section_purpose("SLUG ↔ содержимое"))
            if yn == "Нет":
                se.append("SLUG не соответствует содержимому секции либо написан некорректно\n(Совет: сделайте /slug с заголовка или уточните у ChatGPT).")
        except KeysExhaustedError:
            raise
        except Exception as ex:
            se.append(f"Проверка соответствия SLUG содержимому не выполнена: {ex}")

        # 2) MT/MD/MK/H1 ↔ язык текста секции
        try:
            prompt = build_section_lang_match_prompt(mt, md, mk, h1, content_lines)
            yn = llm_yesno(prompt, purpose=section_purpose("Язык элементов"))
            if yn == "Нет":
                prompt2 = build_section_lang_list_bad_prompt(mt, md, mk, h1, content_lines)
                bad_raw = llm_text(prompt2, purpose=section_purpose("Проблемные элементы"), max_tokens=64).strip()
                ans_bad, tokens = _parse_yesno_list(bad_raw)
                filtered = [t for t in tokens if t.upper() not in EEAT_LANGUAGE_IGNORE_CODES]
                if ans_bad == "Да" and not filtered:
                    se.append("Язык заголовков/меты не совпадает с языком текста секции.")
                elif filtered:
                    se.append(
                        "Язык заголовков/меты не совпадает с языком текста секции. "
                        f"(Проблемные элементы: {' | '.join(filtered)}.)"
                    )
        except KeysExhaustedError:
            raise
        except Exception as ex:
            se.append(f"Проверка языка элементов секции не выполнена: {ex}")

        # 3) Правила по изображениям
        imgs = int(sec.get("content_images", 0))
        if idx == 5:
            if imgs != 1: se.append(f"В секции #5 должно быть ровно 1 картинка; найдено {imgs}.")
        else:
            if imgs != 0: se.append(f"В секции #{idx} изображения быть не должно; найдено {imgs}.")

        return idx, se

    out_errors: List[str] = []
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(sections)))) as pool:
        futs = [pool.submit(check_section, i+1, sec) for i, sec in enumerate(sections)]
        for fut in as_completed(futs):
            try:
                idx, se = fut.result()
            except Exception as ex:
                idx, se = -1, [f"внутренняя ошибка проверки секции: {ex}"]
            if se:
                slug = slug_by_idx.get(idx, "")
                for e in se:
                    out_errors.append(f"Секция #{idx}{f' ({slug})' if slug else ''}: {e}")
    errors.extend(out_errors)

    # Промокод — скан по всем секциям
    try:
        full = []
        for sec in sections:
            for k in ("MT","MD","MK","h1"):
                v = (sec.get(k) or "")
                if v: full.append(f"{k.upper()}: {v}")
            for ln in (sec.get("content_texts") or []):
                if ln: full.append(ln)
        all_text = " ".join(full)
        q1 = build_promo_scan_prompt(all_text)
        has_code = llm_yesno(q1, purpose=eeat_purpose("Промокоды — скан"))
        if has_code == "Да":
            q2 = build_promo_extract_prompt(all_text)
            codes = llm_text(q2, purpose=eeat_purpose("Промокоды — извлечение"), max_tokens=256)
            codes = codes.strip()
            filtered = filter_promo_codes(codes)
            if filtered:
                errors.append(f"EEAT: обнаружен посторонний промокод: {' | '.join(filtered)}")
    except KeysExhaustedError:
        raise
    except Exception:
        pass

    return errors, details

# =========================================================
# ==================== ЗАГРУЗКА ZOHO DOCX =================
# =========================================================
def _mk_retry(total=2, backoff=0.25, statuses=(429, 502, 503, 504)):
    if _URLLIB3_Retry is None:
        return None
    try:
        return _URLLIB3_Retry(
            total=total, backoff_factor=backoff,
            status_forcelist=statuses,
            allowed_methods=frozenset(["GET", "POST"])
        )
    except TypeError:
        return _URLLIB3_Retry(
            total=total, backoff_factor=backoff,
            status_forcelist=statuses,
            method_whitelist=frozenset(["GET", "POST"])
        )

def _mk_session(pool_maxsize=8, retries=2, backoff=0.25):
    s = requests.Session()
    r = _mk_retry(total=retries, backoff=backoff)
    adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize, max_retries=r)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"Connection": "keep-alive"})
    return s

_ZOHO_SESSION = _mk_session(pool_maxsize=16, retries=2, backoff=0.10)

# --- Кеш и синхронизация OAuth токена ---
_ZOHO_TOKEN_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_ZOHO_TOKEN_LOCK = threading.Lock()

def zoho_doc_id_from_url(url: str) -> Optional[str]:
    m = re.search(r"https?://writer\.zoho\.[^/]+/writer/open/([A-Za-z0-9]+)", url)
    if m: return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9]+)", url)
    return m.group(1) if m else None

def is_workdrive_public(url: str) -> bool:
    return "zohopublic" in url or "/external/" in url

def zoho_domains_from_writer_url(url: str) -> Tuple[str, str]:
    m = re.search(r"https?://writer\.zoho\.([^/]+)/", url)
    tld = m.group(1) if m else ""
    tld_map = {
        "com":    ("www.zohoapis.com",    "https://accounts.zoho.com"),
        "eu":     ("www.zohoapis.eu",     "https://accounts.zoho.eu"),
        "in":     ("www.zohoapis.in",     "https://accounts.zoho.in"),
        "com.au": ("www.zohoapis.com.au", "https://accounts.zoho.com.au"),
        "jp":     ("www.zohoapis.jp",     "https://accounts.zoho.jp"),
        "sa":     ("www.zohoapis.sa",     "https://accounts.zoho.sa"),
        "ca":     ("www.zohoapis.ca",     "https://accounts.zoho.ca"),
        "com.cn": ("www.zohoapis.com.cn","https://accounts.zoho.com.cn"),
    }
    return tld_map.get(tld, (ZOHOAPIS_DOMAIN_DEFAULT, ZOHO_ACCOUNTS_URL))

def _text_preview_from_bytes(b: bytes, limit: int = 600) -> str:
    try: t = b.decode("utf-8", "ignore")
    except Exception:
        try: t = b.decode("latin-1", "ignore")
        except Exception: return ""
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= limit else t[:limit] + " …(обрезано)"

def _retry_after_seconds(headers: Dict[str, str]) -> Optional[float]:
    ra = None
    for k in ("Retry-After", "retry-after"):
        if k in headers and headers[k]:
            ra = headers[k].strip()
            break
    if not ra:
        return None
    if re.fullmatch(r"\d+", ra):
        try:
            return float(int(ra))
        except Exception:
            return None
    return None

def _extract_ttl_seconds(js: Dict[str, Any]) -> int:
    ttl = None
    for k in ("expires_in", "expires_in_sec", "expires_in_secs"):
        if k in js:
            try:
                ttl = int(js[k])
                break
            except Exception:
                pass
    if not ttl or ttl <= 0:
        ttl = ZOHO_TOKEN_DEFAULT_TTL_SEC
    return max(ttl, 300)

def _get_zoho_access_token(accounts_url: str) -> Tuple[Optional[str], Optional[str]]:
    if not (ZOHO_CLIENT_ID and ZOHO_CLIENT_SECRET and ZOHO_REFRESH_TOKEN):
        return None, "\n".join([
            "[API] Скачивание Zoho DOCX — ОШИБКА",
            "Не задан OAuth Zoho (ZOHO_CLIENT_ID / ZOHO_CLIENT_SECRET / ZOHO_REFRESH_TOKEN)."
        ])

    key = (accounts_url, ZOHO_CLIENT_ID, ZOHO_REFRESH_TOKEN)
    attempts = max(1, ZOHO_OAUTH_MAX_RETRIES)
    last_err_text = None

    for attempt in range(attempts):
        with _ZOHO_TOKEN_LOCK:
            entry = _ZOHO_TOKEN_CACHE.get(key, {})
            now = time.time()
            token = entry.get("token")
            exp = float(entry.get("exp", 0))
            next_allowed = float(entry.get("next_allowed", 0))
            if token and now < exp:
                return token, None
            wait_sec = max(0.0, next_allowed - now)
        if wait_sec > 0:
            time.sleep(wait_sec)

        with _ZOHO_TOKEN_LOCK:
            entry = _ZOHO_TOKEN_CACHE.get(key, {})
            now = time.time()
            token = entry.get("token")
            exp = float(entry.get("exp", 0))
            if token and now < exp:
                return token, None
            _ZOHO_TOKEN_CACHE[key] = {**entry, "next_allowed": now + ZOHO_OAUTH_MIN_INTERVAL_SEC}

            data = {
                "grant_type": "refresh_token",
                "client_id": ZOHO_CLIENT_ID,
                "client_secret": ZOHO_CLIENT_SECRET,
                "refresh_token": ZOHO_REFRESH_TOKEN,
            }
            try:
                resp = _ZOHO_SESSION.post(f"{accounts_url}/oauth/v2/token", data=data, timeout=30)
            except Exception as ex:
                last_err_text = f"[API] OAuth Zoho сеть: {ex}"
                delay = min(60.0, ZOHO_OAUTH_RETRY_BASE_SEC * (2 ** attempt) + random.uniform(0.0, 0.5))
                _ZOHO_TOKEN_CACHE[key]["next_allowed"] = time.time() + max(delay, ZOHO_OAUTH_MIN_INTERVAL_SEC)
                time.sleep(delay)
                continue

            ctype = resp.headers.get("Content-Type", "")
            js = {}
            if isinstance(ctype, str) and ctype.startswith("application/json"):
                try:
                    js = resp.json()
                except Exception:
                    js = {}
            ok = (resp.status_code == 200 and "access_token" in js)

            if ok:
                token = js["access_token"]
                ttl = _extract_ttl_seconds(js)
                exp_ts = time.time() + max(10, ttl - ZOHO_OAUTH_TTL_SKEW_SEC)
                _ZOHO_TOKEN_CACHE[key] = {"token": token, "exp": exp_ts, "next_allowed": time.time() + ZOHO_OAUTH_MIN_INTERVAL_SEC}
                return token, None

            body_preview = trim(str(js) or resp.text, 400)
            last_err_text = f"[API] OAuth Zoho ошибка: HTTP {resp.status_code}; {body_preview}"
            too_many = ("too many requests" in (body_preview or "").lower())
            if resp.status_code in (400, 429) and too_many:
                retry_after = _retry_after_seconds(resp.headers) or 0.0
                delay = retry_after if retry_after > 0 else (ZOHO_OAUTH_RETRY_BASE_SEC * (2 ** attempt) + random.uniform(0.25, 0.75))
                delay = max(ZOHO_OAUTH_MIN_INTERVAL_SEC, min(delay, 60.0))
                _ZOHO_TOKEN_CACHE[key]["next_allowed"] = time.time() + delay
                time.sleep(delay)
                continue
            break

    return None, (last_err_text or "[API] OAuth Zoho ошибка: неизвестно")

def download_zoho_docx(url: str, tmpdir: str, tag: str) -> Tuple[Optional[str], Optional[str]]:
    if not url.startswith("http") and os.path.exists(url):
        return url, None
    if is_workdrive_public(url):
        return None, "\n".join([
            "[API] Скачивание Zoho DOCX — ОШИБКА",
            "Публичные WorkDrive‑ссылки (zohopublic/external) Writer API не поддерживает.",
            "Нужна ссылка: https://writer.zoho.{dc}/writer/open/<document_id>"
        ])

    doc_id = zoho_doc_id_from_url(url) if url.startswith("http") else None
    if not doc_id:
        return None, "\n".join([
            "[API] Скачивание Zoho DOCX — ОШИБКА",
            "Некорректная ссылка (ожидаю https://writer.zoho.../writer/open/<id>)",
            f"URL: {url}"
        ])

    zohoapis_domain, accounts_url = zoho_domains_from_writer_url(url)

    if not (ZOHO_CLIENT_ID and ZOHO_CLIENT_SECRET and ZOHO_REFRESH_TOKEN):
        return None, "\n".join([
            "[API] Скачивание Zoho DOCX — ОШИБКА",
            "Не задан OAuth Zoho (ZOHO_CLIENT_ID / ZOHO_CLIENT_SECRET / ZOHO_REFRESH_TOKEN)."
        ])

    token, oauth_err = _get_zoho_access_token(accounts_url)
    if oauth_err:
        return None, oauth_err

    download_url = f"https://{zohoapis_domain}/writer/api/v1/download/{doc_id}?format=docx"
    headers = {"Authorization": f"Zoho-oauthtoken {token}", "User-Agent": "DOCX-Validator/1.9.0"}

    try:
        resp = _ZOHO_SESSION.get(download_url, timeout=45, allow_redirects=True, headers=headers)
        ct = (resp.headers.get("Content-Type") or "").lower()
        data = resp.content or b""
        if resp.status_code != 200 or not data:
            preview = _text_preview_from_bytes(data) if ct.startswith(("application/json","text/")) else ""
            return None, f"[API] Скачивание Zoho DOCX — HTTP {resp.status_code}; {preview}"
        if not data.startswith(b"PK"):
            preview = _text_preview_from_bytes(data)
            return None, f"[API] Скачивание Zoho DOCX — 200 OK, но не ZIP/DOCX. {preview}"
        out_path = os.path.join(tmpdir, f"zdoc_{int(time.time()*1000)}.docx")
        with open(out_path, "wb") as f:
            f.write(data)
        return out_path, None
    except Exception as ex:
        return None, f"[API] Скачивание Zoho DOCX — сеть: {ex}"

# =========================================================
# ============== ЯДРО: открыть/проверить DOCX =============
# =========================================================
def _open_doc_safe(path: str) -> Document:
    patched = patch_docx_for_python_docx(path)
    return Document(patched)

def validate_article_from_url(tag: str, url: str, tmpdir: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
    own_tmp = False
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="docxv_"); own_tmp = True
    lines = []; ok = True; zip_src_path: Optional[str] = None

    try:
        orig_path, err = download_zoho_docx(url, tmpdir, tag)
        if err:
            lines.append(f"🛑 [{tag}] ОШИБКА ЗАГРУЗКИ")
            for ln in err.splitlines():
                if ln.strip(): lines.append(f"   - {ln}")
            return False, "\n".join(lines), None

        try:
            doc = _open_doc_safe(orig_path)
            errors = validate_text_article(doc, tag)
            if errors:
                ok = False
                lines.append(f"🛑 [{tag}] ОШИБКИ:")
                for e in errors: lines.append(f"   - {e}")
            else:
                lines.append(f"✅ [{tag}] ПРОЙДЕНО.")
        except KeysExhaustedError:
            raise
        except Exception as ex:
            ok = False
            tb = " | ".join(traceback.format_exc().splitlines()[-3:])
            lines.append(f"🛑 [{tag}] Не удалось прочитать/проверить DOCX: {ex}")
            if tb: lines.append(f"   • Traceback (последние строки): {tb}")

        try:
            fixed = reset_docx_image_sizes_to_native(orig_path)
            zip_src_path = fixed or orig_path
        except Exception:
            zip_src_path = orig_path
    finally:
        if own_tmp:
            pass
    return ok, "\n".join(lines), zip_src_path

def validate_eeat_from_url(url: str, tmpdir: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
    own_tmp = False
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="docxv_"); own_tmp = True
    lines = []; ok = True; zip_src_path: Optional[str] = None

    try:
        orig_path, err = download_zoho_docx(url, tmpdir, "EEAT")
        if err:
            lines.append(f"🛑 EEAT: ошибка загрузки")
            for ln in err.splitlines():
                if ln.strip(): lines.append(f"   - {ln}")
            return False, "\n".join(lines), None

        try:
            doc = _open_doc_safe(orig_path)
            eeat_errors, _details = validate_eeat(doc)
            if eeat_errors:
                ok = False
                lines.append("🛑 EEAT: ОШИБКИ:")
                for ge in eeat_errors: lines.append(f"   - {ge}")
            else:
                lines.append("✅ EEAT: ПРОЙДЕНО.")
        except KeysExhaustedError:
            raise
        except Exception as ex:
            ok = False
            lines.append(f"🛑 EEAT: Не удалось прочитать/проверить DOCX: {ex}")

        try:
            fixed = reset_docx_image_sizes_to_native(orig_path)
            zip_src_path = fixed or orig_path
        except Exception:
            zip_src_path = orig_path
    finally:
        if own_tmp:
            pass
    return ok, "\n".join(lines), zip_src_path

async def run_full_validation_async(project: str,
                                    articles: List[Tuple[str, str]],
                                    eeat_link: Optional[str] = None) -> Tuple[bool, str, List[Tuple[str, str]], Optional[str]]:
    tmpdir = tempfile.mkdtemp(prefix="docxv_")
    sem = asyncio.Semaphore(max(1, VALIDATOR_CONCURRENCY))
    per_item_logs: List[Tuple[str, str]] = []
    zip_items: List[Tuple[str, str]] = []

    async def run_article(tag: str, url: str):
        async with sem:
            return await asyncio.to_thread(validate_article_from_url, tag, url, tmpdir)

    async def run_eeat(url: str):
        async with sem:
            return await asyncio.to_thread(validate_eeat_from_url, url, tmpdir)

    try:
        tasks = [asyncio.create_task(run_article(tag, url)) for tag, url in articles]
        eeat_task = asyncio.create_task(run_eeat(eeat_link)) if eeat_link else None

        results = await asyncio.gather(*tasks, return_exceptions=True)
        eeat_result = await asyncio.gather(eeat_task, return_exceptions=True) if eeat_task else []

        overall_ok = True
        report_lines: List[str] = []

        for i, res in enumerate(results):
            tag = articles[i][0]
            if isinstance(res, KeysExhaustedError):
                raise res
            if isinstance(res, Exception):
                per_item_logs.append((tag, f"🛑 Ошибка выполнения: {res!r}"))
                overall_ok = False
                continue
            ok, log_text, zip_src_path = res
            overall_ok = overall_ok and ok
            per_item_logs.append((tag, log_text))
            report_lines.append(log_text); report_lines.append("")
            if ok and zip_src_path:
                zip_items.append((zip_src_path, f"{tag}.docx"))

        report_lines += [BAR, "📑 EEAT", BAR]
        if eeat_link:
            er = eeat_result[0]
            if isinstance(er, KeysExhaustedError):
                raise er
            if isinstance(er, Exception):
                eeat_log = f"🛑 EEAT: ошибка выполнения — {er!r}"
                per_item_logs.append(("EEAT", eeat_log))
                report_lines.append(eeat_log)
                overall_ok = False
            else:
                ok, eeat_log, eeat_zip_src = er
                overall_ok = overall_ok and ok
                per_item_logs.append(("EEAT", eeat_log))
                report_lines.append(eeat_log)
                if ok and eeat_zip_src:
                    zip_items.append((eeat_zip_src, "eeat.docx"))
        else:
            report_lines.append("EEAT: ссылка не задана.")
        report_lines.append(BAR)
        full_report = "\n".join(report_lines)

        zip_path: Optional[str] = None
        if overall_ok and zip_items:
            fd, tmp_zip = tempfile.mkstemp(prefix="docs_", suffix=".zip")
            os.close(fd)
            with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for src, arcname in zip_items:
                    zf.write(src, arcname)
            zip_path = tmp_zip

        return overall_ok, full_report, per_item_logs, zip_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# =========================================================
# ================== REPORT / РАЗМЕТКА ====================
# =========================================================
def extract_errors_from_log_text(tag: str, text: str) -> List[str]:
    pattern = rf"^🛑 \[{re.escape(tag)}\] ОШИБКИ:\s*(?P<blk>(?:\s{{3,}}- .*(?:\n|$))+)"
    m = re.search(pattern, text, flags=re.M | re.U)
    items: List[str] = []
    if m:
        for ln in m.group("blk").splitlines():
            ln = ln.strip()
            if ln.startswith("- "):
                items.append(ln[2:].strip())
    if not items:
        for ln in text.splitlines():
            if re.match(r"^\s{2,}-\s+", ln):
                items.append(ln.strip()[2:].strip())
    return items[:12]

def parse_article_log(tag: str, text: str) -> Dict:
    ok = "ПРОЙДЕНО" in text
    errors = extract_errors_from_log_text(tag, text)
    return {"tag": tag, "ok": ok and not errors, "errors": errors}

def build_single_message(project: str, per_item_logs: List[Tuple[str, str]], overall_ok: bool) -> str:
    lines: List[str] = []
    lines.append(BAR)

    article_rows = [(tag, txt) for tag, txt in per_item_logs if tag != "EEAT"]
    parsed_articles = [parse_article_log(tag, txt) for tag, txt in article_rows]
    total = len(parsed_articles)
    failed = sum(1 for a in parsed_articles if not a["ok"])
    passed = total - failed

    lines.append("📄 Статьи")
    for a in parsed_articles:
        if a["ok"]:
            lines.append(f"— {a['tag']}: ✅ ПРОЙДЕНО")
        else:
            lines.append(f"— {a['tag']}: ❌ Ошибки")
            for e in a["errors"]:
                lines.append(f"   • {e}")
    if total == 0:
        lines.append("— (нет статей)")
    lines.append(f"\nИтог по статьям: всего {total}; ✅ {passed}; ❌ {failed}")
    lines.append(BAR)

    eeat_pair = next(((tag, txt) for tag, txt in per_item_logs if tag == "EEAT"), None)
    lines.append("📑 EEAT")
    lines.append("")
    if eeat_pair is None:
        lines.append("EEAT: ссылка не задана.")
    else:
        for ln in eeat_pair[1].splitlines():
            lines.append(neutralize_slash_tokens(ln))
    lines.append(BAR)

    text = "\n".join(lines)
    if len(text) > 4000:
        text = text[:3990].rstrip() + "\n…(сообщение обрезано)"
    return text

# =========================================================
# ========================  BOT  ==========================
# =========================================================
dp = Dispatcher()
_PENDING_KEYS: Dict[int, bool] = {}

def is_admin(uid: Optional[int]) -> bool:
    return int(uid or 0) == int(ADMIN_TG_ID)

async def maybe_send_admin_debug_logs(message: Message, heading: str = "🛠 Лог LLM") -> None:
    if not is_admin(getattr(message.from_user, "id", None)):
        return
    if not admin_debug_enabled():
        return
    logs = admin_debug_drain()
    if not logs:
        return
    log_text = "\n\n".join(logs)
    fd, tmp_path = tempfile.mkstemp(prefix="llm_log_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(log_text)
            if not log_text.endswith("\n"):
                fh.write("\n")
        await message.answer_document(
            FSInputFile(tmp_path, filename="log.txt"),
            caption=heading,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

async def setup_menu_commands(bot: Bot, chat_id: Optional[int] = None):
    await bot.set_my_commands(
        commands=[
            BotCommand(command="start", description="Запустить бота"),
            BotCommand(command="help", description="Как пользоваться валидатором"),
        ],
        scope=BotCommandScopeDefault()
    )
    if ADMIN_TG_ID:
        try:
            await bot.set_my_commands(
                commands=[
                    BotCommand(command="start", description="Запустить бота"),
                    BotCommand(command="help", description="Как пользоваться валидатором"),
                    BotCommand(command="keys_add", description="Добавить OpenAI ключи"),
                    BotCommand(command="keys_list", description="Список ключей"),
                    BotCommand(command="keys_clear", description="Удалить все ключи"),
                    BotCommand(command="logs_toggle", description="Вкл/выкл лог LLM"),
                ],
                scope=BotCommandScopeChat(chat_id=ADMIN_TG_ID)
            )
        except Exception:
            pass

def parse_lines_to_pairs(text: str) -> Tuple[List[Tuple[str, str]], Optional[str], List[str]]:
    """
    Форматы:
      1) 'название<пробел/таб>URL_или_путь'  — в одну строку
      2) 'название' и на следующей строке 'URL_или_путь' — в две строки
    'eeat' — ссылка на EEAT.
    """
    articles: List[Tuple[str, str]] = []
    eeat_link: Optional[str] = None
    errors: List[str] = []

    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    i = 0

    def is_url_or_path(s: str) -> bool:
        return bool(re.match(r"^https?://", s, flags=re.I)) or os.path.exists(s)

    while i < len(lines):
        ln = lines[i].strip()

        m = re.match(r'^(\S+)\s+(.+)$', ln)
        if m and is_url_or_path(m.group(2).strip()):
            tag = m.group(1).strip()
            url = m.group(2).strip()
            if tag.lower() == "eeat":
                eeat_link = url
            else:
                articles.append((tag, url))
            i += 1
            continue

        if i + 1 < len(lines) and is_url_or_path(lines[i + 1].strip()):
            tag = ln
            url = lines[i + 1].strip()
            if tag.lower() == "eeat":
                eeat_link = url
            else:
                articles.append((tag, url))
            i += 2
            continue

        errors.append(f"Строка {i+1}: не распознана пара 'название\\tссылка' → «{ln}»")
        i += 1

    return articles, eeat_link, errors

@dp.message(CommandStart())
async def on_start(message: Message):
    db_init()
    await setup_menu_commands(message.bot)

    if not is_admin(getattr(message.from_user, "id", None)):
        await message.answer(
            "Привет! Я валидатор DOCX (Zoho Writer) с нейро‑проверками.\n\n"
            "Отправьте построчно пары `название<TAB/пробел>ссылка/путь`.\n"
            "`eeat <ссылка/путь>` — EEAT‑проверка.\n"
            "Ссылки: `https://writer.zoho.{dc}/writer/open/<id>` или локальные пути к .docx."
        )
        return

    await message.answer(
        "Привет, админ! Кнопки управления ключами доступны в меню у строки ввода:\n"
        "— /keys_add — добавить ключи (многострочный ввод)\n"
        "— /keys_list — просмотр списка (замаскировано)\n"
        "— /keys_clear — удалить все ключи\n\n"
        "Для запуска проверок просто пришлите пары `название  ссылка` (строка на файл/Zoho)."
    )

@dp.message(Command("help"))
async def on_help(message: Message):
    await message.answer(
        "Отправьте построчно пары `название<TAB/пробел>ссылка/путь`.\n"
        "`eeat <ссылка/путь>` — EEAT‑проверка.\n"
        "Ссылки: `https://writer.zoho.{dc}/writer/open/<id>` или локальные пути к .docx.\n\n"
        "Администратор управляет ключами OpenAI через меню команд возле строки ввода."
    )

@dp.message(Command("keys_add"))
async def on_keys_add(message: Message):
    if not is_admin(getattr(message.from_user, "id", None)):
        await message.answer("⛔️ Недостаточно прав.")
        return
    _PENDING_KEYS[message.chat.id] = True
    await message.answer(
        "Пришлите ключи OpenAI (каждый в новой строке). Пример:\n"
        "sk-proj-...\nsk-...\n\n"
        "Будут сохранены только корректные строки, дубликаты игнорируются."
    )

@dp.message(Command("keys_list"))
async def on_keys_list(message: Message):
    if not is_admin(getattr(message.from_user, "id", None)):
        await message.answer("⛔️ Недостаточно прав.")
        return
    db_init()
    keys = db_list_keys()
    if not keys:
        await message.answer("Ключей нет.")
        return
    masked = [mask_secret(k, 8, 6) for k in keys]
    await message.answer("Ключи:\n• " + "\n• ".join(masked))

@dp.message(Command("keys_clear"))
async def on_keys_clear(message: Message):
    if not is_admin(getattr(message.from_user, "id", None)):
        await message.answer("⛔️ Недостаточно прав.")
        return
    n = db_clear_keys()
    await message.answer(f"Удалено ключей: {n}")

@dp.message(Command("logs_toggle"))
async def on_logs_toggle(message: Message):
    if not is_admin(getattr(message.from_user, "id", None)):
        await message.answer("⛔️ Недостаточно прав.")
        return
    db_init()
    current = admin_debug_enabled()
    admin_debug_set(not current)
    if current:
        await message.answer(
            "Лог LLM выключен. Новые ответы OpenAI не будут отправляться до повторного включения."
        )
    else:
        await message.answer(
            "Лог LLM включен. После каждой проверки будут приходить ответы и ошибки OpenAI."
        )

@dp.message(F.text)
async def on_text(message: Message):
    db_init()
    admin_user = is_admin(getattr(message.from_user, "id", None))

    try:
        if _PENDING_KEYS.get(message.chat.id) and admin_user:
            added = db_add_keys_bulk(message.text or "")
            _PENDING_KEYS.pop(message.chat.id, None)
            await message.answer(f"✅ Добавлено ключей: {added}.")
            return

        raw = message.text or ""
        articles, eeat_link, parse_errors = parse_lines_to_pairs(raw)
        if parse_errors and not articles and not eeat_link:
            await message.answer("Не удалось распознать ни одной строки:\n• " + "\n• ".join(parse_errors))
            return

        keys = db_list_keys()
        if not keys and not os.environ.get("OPENAI_API_KEY"):
            if admin_user:
                await message.answer("Бот не настроен: нет ключей OpenAI. Добавьте через /keys_add.")
            else:
                await message.answer("Бот временно не настроен. Сообщите администратору.")
            return

        try:
            overall_ok, _full_report, per_item_logs, zip_path = await run_full_validation_async(
                "Project", articles, eeat_link
            )
            final_msg = build_single_message("Project", per_item_logs, overall_ok)

            if overall_ok and zip_path and os.path.exists(zip_path):
                try:
                    await message.bot.send_document(
                        chat_id=message.chat.id,
                        document=FSInputFile(zip_path, filename="docs.zip"),
                        caption=final_msg
                    )
                finally:
                    with contextlib.suppress(Exception):
                        os.remove(zip_path)
            else:
                await message.bot.send_message(
                    chat_id=message.chat.id,
                    text=final_msg,
                    disable_web_page_preview=True
                )
        except KeysExhaustedError as ex:
            await message.answer("Закончились ключи, пишите @locosd")
            if admin_user:
                await message.answer(f"🛠 Детали: {ex}")
        except Exception as ex:
            await message.answer(f"🛑 Внутренняя ошибка: {ex}")
    finally:
        await maybe_send_admin_debug_logs(message)

async def main():
    try:
        import uvloop
        uvloop.install()
    except Exception:
        pass

    if not BOT_TOKEN or BOT_TOKEN == "PASTE_YOUR_TELEGRAM_BOT_TOKEN":
        raise RuntimeError("Не задан BOT_TOKEN (см. начало файла).")
    bot = Bot(BOT_TOKEN, parse_mode=None)
    await setup_menu_commands(bot)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())