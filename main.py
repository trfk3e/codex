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
from typing import List, Tuple, Dict, Optional, Callable, Any
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
    line = f"[{now_str()}] {message}"
    with _ADMIN_DEBUG_LOCK:
        _ADMIN_DEBUG_LOGS.append(line)
        if len(_ADMIN_DEBUG_LOGS) > 200:
            del _ADMIN_DEBUG_LOGS[:-200]

def admin_debug_drain() -> List[str]:
    with _ADMIN_DEBUG_LOCK:
        logs = list(_ADMIN_DEBUG_LOGS)
        _ADMIN_DEBUG_LOGS.clear()
    return logs

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
            if isinstance(typ, str) and typ.lower() in {"output_text", "text"} and isinstance(obj.get("text"), str):
                texts.append(obj["text"])
            for key in ("content", "contents", "items", "data", "output"):
                if key in obj:
                    _gather(obj[key])
            return
        for attr_name in ("text", "value"):
            if hasattr(obj, attr_name):
                val = getattr(obj, attr_name)
                if isinstance(val, str):
                    texts.append(val)
        for attr_name in ("content", "contents"):
            if hasattr(obj, attr_name):
                _gather(getattr(obj, attr_name))

    if output:
        _gather(output)
    if texts:
        combined = "".join(texts).strip()
        if combined:
            return combined

    for attr in ("model_dump", "dict", "to_dict"):
        if hasattr(resp, attr):
            try:
                data = getattr(resp, attr)()
            except Exception:
                continue
            _gather(data)
            if texts:
                combined = "".join(texts).strip()
                if combined:
                    return combined
    return None

def _invoke_openai(client: OpenAI, messages: List[Dict[str, Any]], max_tokens: int, temperature: float) -> Tuple[str, str]:
    last_exc: Optional[Exception] = None
    responses_api = getattr(client, "responses", None)
    if responses_api is not None:
        try:
            resp = responses_api.create(
                model=OPENAI_MODEL,
                input=_responses_input_from_messages(messages),
                temperature=temperature,
                max_output_tokens=max_tokens,
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
        resp = completions_api.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice0 = getattr(resp, "choices", None)
        if isinstance(choice0, list) and choice0:
            msg0 = choice0[0]
            content = None
            if isinstance(msg0, dict):
                content = ((msg0.get("message") or {}).get("content")) or msg0.get("text")
            else:
                message_attr = getattr(msg0, "message", None)
                if message_attr is not None:
                    content = getattr(message_attr, "content", None)
                if content is None:
                    content = getattr(msg0, "text", None)
            if isinstance(content, str) and content.strip():
                return content.strip(), "chat.completions"
        raise RuntimeError("Пустой ответ от chat.completions")

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("У клиента OpenAI нет поддерживаемых методов Responses/chat")

def _call_openai_with_keys(messages: List[Dict],
                           max_tokens: int = 64,
                           temperature: float = 0.0,
                           purpose: str = "") -> str:
    label = purpose or "general"
    prompt_preview = _messages_preview(messages, limit=500)
    if prompt_preview:
        admin_debug_log(f"LLM[{label}] Prompt: {prompt_preview}")
    keys = db_list_keys()
    env_sk = os.environ.get("OPENAI_API_KEY")
    if env_sk and env_sk not in keys:
        keys.append(env_sk)

    if not keys:
        admin_debug_log(f"LLM[{label}] Нет доступных ключей.")
        raise KeysExhaustedError("Нет ни одного OpenAI API Key")

    last_exc = None
    for round_idx in (1, 2):
        for sk in keys:
            try:
                client = _openai_client_for_key(sk)
                local_exc = None
                key_masked = mask_secret(sk, 8, 6)
                for attempt in range(LLM_MAX_RETRIES_LOCAL + 1):
                    try:
                        admin_debug_log(
                            f"LLM[{label}] Ключ {key_masked}: попытка {attempt + 1} (раунд {round_idx})"
                        )
                        answer, method = _invoke_openai(client, messages, max_tokens, temperature)
                        if isinstance(answer, str) and answer.strip():
                            text = answer.strip()
                            admin_debug_log(
                                f"LLM[{label}] Ключ {key_masked}: успех через {method}, ответ: {trim(text, 500)}"
                            )
                            return text
                        raise RuntimeError("Пустой ответ модели OpenAI")
                    except Exception as ex:
                        local_exc = ex
                        admin_debug_log(
                            f"LLM[{label}] Ключ {key_masked}: ошибка попытки {attempt + 1}: {ex}"
                        )
                        time.sleep(0.2 * (attempt + 1))
                last_exc = local_exc
            except Exception as ex:
                last_exc = ex
                admin_debug_log(f"LLM[{label}] Ошибка при использовании ключа {key_masked}: {ex}")
                continue
    admin_debug_log(f"LLM[{label}] Ключи исчерпаны: {last_exc}")
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

def llm_yesno(question: str, purpose: str) -> str:
    messages = [
        {"role": "system",
         "content": "Ты аккуратный русскоязычный ассистент контроля качества. "
                    "Отвечай ТОЛЬКО одним словом: «Да» или «Нет». Никаких комментариев."},
        {"role": "user", "content": question}
    ]
    raw = _call_openai_with_keys(messages=messages, max_tokens=4, temperature=0.0, purpose=purpose)
    yn = _normalize_yesno(raw)
    if yn in ("Да", "Нет"):
        return yn
    messages.append({"role": "user", "content": "Напомню: нужно одно слово — «Да» или «Нет»."})
    raw2 = _call_openai_with_keys(messages=messages, max_tokens=4, temperature=0.0, purpose=f"{purpose} (retry)")
    yn2 = _normalize_yesno(raw2)
    if yn2 in ("Да", "Нет"):
        return yn2
    raise RuntimeError(f"[{purpose}] Не удалось интерпретировать ответ модели: {raw!r}")

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
def build_article_headings_vs_text_prompt(headings: List[str], body: str) -> str:
    htxt = "\n".join([f"- {clean_for_prompt(h)}" for h in headings if (h or "").strip()])
    body = clip(clean_for_prompt(body), ART_PROMPT_MAX_CHARS)
    return (
        "Ты — валидатор совпадения языка заголовков и основного текста. Язык промпта на котором это написано не причем. Не упоминай его.\n"
        "Анализируй СТРОГО только содержимое внутри тегов <HEADINGS> и <TEXT> ниже. "
        "Игнорируй язык этих инструкций и всё, что вне тегов.\n"
        "1) Определи ДОМИНИРУЮЩИЙ язык блока <TEXT> по грамматике/служебным словам "
        "(диакритику игнорируй: á→a, ü→u и т.п.).\n"
        "2) Игнорируй бренды/имена, домены/URL, аббревиатуры, числа/валюты и одиночные англ. заимствования "
        "(например: login, casino, bonus).\n"
        "3) Сравни каждый заголовок из <HEADINGS> с языком <TEXT>.\n"
        "Ответ строго одним словом: «Да» (совпадает) или «Нет» (есть заголовки на другом языке). Никаких комментариев.  Главное не ошибайся, думай столько сколько нужно.\n"
        f"\n<HEADINGS>\n{htxt or '—'}\n</HEADINGS>"
        f"\n\n<TEXT>\n{body}\n</TEXT>"
    )

def build_article_list_problem_headings_prompt(headings: List[str], body: str) -> str:
    htxt = "\n".join([f"- {clean_for_prompt(h)}" for h in headings if (h or "").strip()])
    body = clip(clean_for_prompt(body), ART_PROMPT_MAX_CHARS)
    return (
        "Определи ДОМИНИРУЮЩИЙ язык <TEXT> СТРОГО по блоку <TEXT> ниже. Язык промпта на котором это написано не причем. Не упоминай его."
        "Игнорируй язык этих инструкций и всё, что вне тегов. Только язык с <TEXT>. "
        "Диакритику игнорируй (á→a, ü→u и т.п.). "
        "Игнорируй бренды/имена, домены/URL, аббревиатуры, числа/валюты и одиночные англ. заимствования.\n"
        "Сравни каждый заголовок из <HEADINGS> с языком <TEXT>. "
        "Если заголовок совпадает по языку — пропусти. "
        "Если нет — укажи точную проблему.\n"
        "Строгий формат для каждой строки: 'проблемный заголовок' - краткое пояснение что не так и конкретно нужно определить язык <TEXT>. "
        "Если ошибок нет — выведи «—». Главное не ошибайся, думай столько сколько нужно.\n"
        f"\n<HEADINGS>\n{htxt or '—'}\n</HEADINGS>"
        f"\n\n<TEXT>\n{body}\n</TEXT>"
    )

def build_slug_consistency_prompt(slug: str, mt: str, md: str, mk: str, h1: str, content: str) -> str:
    slug = (slug or "").strip()
    mt = (mt or "").strip()
    md = (md or "").strip()
    mk = (mk or "").strip()
    h1 = (h1 or "").strip()
    content = clip((content or "").strip(), EEAT_PROMPT_MAX_CHARS)
    body = []
    body.append("Соответствует ли SLUG тематике и языку текста? При проверке учитывать, что SLUG обычно пишется без диакритики (это нормально).")
    body.append("Акцент — на СООТВЕТСТВИИ СОДЕРЖАНИЯ: slug должен отражать тему раздела. Допустимы мелкие отличия или заимствованные слова (например iGaming, бренды).")
    body.append("Ответь строго «Да» или «Нет». Никаких пояснений.")
    body.append("")
    body.append(slug)
    body.append("")
    if mt: body.append(f"MT: {mt}")
    if md: body.append(f"MD: {md}")
    if mk: body.append(f"MK: {mk}")
    if h1: body.append(f"H1: {h1}")
    body.append("")
    if content:
        body.append("Текст секции:")
        body.append(content)
    return "\n".join(body)

def build_section_lang_match_prompt(mt: str, md: str, mk: str, h1: str, content: str) -> str:
    content = clip((content or ""), EEAT_PROMPT_MAX_CHARS)
    msg = []
    msg.append(
        "Ты — строгий валидатор совпадения языка (диактритику не учитываем)."
        "Есть четыре элемента (MT, MD, MK, H1) и основной текст секции. "
        "Сначала мысленно определи ДОМИНИРУЮЩИЙ ЯЗЫК секции по грамматике/морфологии и частотной лексике. "
        "ОЧЕНЬ ВАЖНО: полностью игнорируй диакритику (например, á=а, ó=о, ü=u и т.д.), "
        "то есть при проверке языка все символы считаются без акцентов. "
        "Также игнорируй бренды, имена собственные, домены/URL, аббревиатуры, числа/валюты и отдельные заимствованные слова."
        "\nПравила:\n"
        "• Считай язык секции определённым по грамматике/служебным словам.\n"
        "• Элемент совпадает, если он на том же языке (без учёта диакритики).\n"
        "• Отвечай «Нет» только если элемент явно на другом языке.\n"
        "• В сомнительных случаях отвечай «Да».\n"
        "Ответ строго одним словом: «Да» или «Нет»."
    )
    msg.append("")
    if mt: msg.append(f"MT: {mt}")
    if md: msg.append(f"MD: {md}")
    if mk: msg.append(f"MK: {mk}")
    if h1: msg.append(f"H1: {h1}")
    msg.append("")
    msg.append("Текст секции:")
    msg.append(content)
    return "\n".join(msg)

def build_section_lang_list_bad_prompt(mt: str, md: str, mk: str, h1: str, content: str) -> str:
    content = clip((content or ""), EEAT_PROMPT_MAX_CHARS)
    msg = []
    msg.append("Если языки НЕ совпадают, выпиши какие элементы не совпадают, строго из набора: MT | MD | MK | H1.")
    msg.append("Выведи только эти ярлыки через « | » в ОДНУ строку. Если всё ок — напиши «—».")
    if mt: msg.append(f"MT: {mt}")
    if md: msg.append(f"MD: {md}")
    if mk: msg.append(f"MK: {mk}")
    if h1: msg.append(f"H1: {h1}")
    msg.append("")
    msg.append("Текст секции:")
    msg.append(content)
    return "\n".join(msg)

def build_promo_scan_prompt(full_text: str) -> str:
    full_text = clip(clean_for_prompt(full_text), PROMO_MAX_CHARS)
    return ("Перепроверь данный текст от А до Я, есть ли в нем какой-то ПРОМОКОД СТРОГО написанный заглавными буквами, либо типичный буквенно-цифровой шаблон?.\n Например 'COD', 'CODE', 'MONEY' и т.п. только капслоком, тащательно изучи текст от А до Я. \n Выпиши его даже если он упоминается один раз в тексте."
            "Строго ответь «Да» или «Нет». Никаких комментариев.\n\n" + full_text)

def build_promo_extract_prompt(full_text: str) -> str:
    full_text = clip(clean_for_prompt(full_text), PROMO_MAX_CHARS)
    return ("Выпиши из текста все найденные ПРОМОКОДЫ (только которые написаны заглавными буквами), без комментариев. "
            "Каждый код — отдельно, через « | » в одной строке. Если ничего нет — «—».\n\n" + full_text)

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

def validate_text_article(doc: Document) -> List[str]:
    errors: List[str] = []
    headings: List[Tuple[int, str]] = []
    body_texts: List[str] = []

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
        heading_texts = [t for _, t in headings if (t or "").strip()]
        body_join = clip(" ".join([t for t in body_texts if (t or "").strip()]), ART_PROMPT_MAX_CHARS)
        if heading_texts and body_join:
            prompt = build_article_headings_vs_text_prompt(heading_texts, body_join)
            yn = llm_yesno(prompt, purpose="Headings vs Text language (article)")
            if yn == "Нет":
                prompt2 = build_article_list_problem_headings_prompt(heading_texts, body_join)
                bad = llm_text(prompt2, purpose="List problem headings", max_tokens=256)
                bad = re.sub(r'\s*\|\s*', ' | ', bad.strip())
                if bad in ("—", "-", ""):
                    errors.append("Язык: заголовки не совпадают с языком основного текста.")
                else:
                    errors.append(f"Язык: заголовки не совпадают с языком основного текста. "
                                  f"(Проблемные заголовки: {bad}.)")
    except KeysExhaustedError:
        raise
    except Exception as ex:
        errors.append(f"Нейросетевая проверка языка заголовков/текста не выполнена: {ex}")

    try:
        full_text = " ".join([t for t in (body_texts + [t for _, t in headings]) if t])
        q1 = build_promo_scan_prompt(full_text)
        has_code = llm_yesno(q1, purpose="Promo code scan (article)")
        if has_code == "Да":
            q2 = build_promo_extract_prompt(full_text)
            codes = llm_text(q2, purpose="Promo code extract", max_tokens=256)
            codes = codes.strip()
            if codes and codes != "—":
                errors.append(f"Обнаружен посторонний промокод: {codes}")
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
        content_texts = " ".join([t for t in (sec.get("content_texts") or []) if (t or "").strip()])

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
            prompt = build_slug_consistency_prompt(slug_clean, mt, md, mk, h1, content_texts)
            yn = llm_yesno(prompt, purpose=f"EEAT slug consistency #{idx}")
            if yn == "Нет":
                se.append("SLUG не соответствует содержимому секции либо написан некорректно\n(Совет: сделайте /slug с заголовка или уточните у ChatGPT).")
        except KeysExhaustedError:
            raise
        except Exception as ex:
            se.append(f"Проверка соответствия SLUG содержимому не выполнена: {ex}")

        # 2) MT/MD/MK/H1 ↔ язык текста секции
        try:
            prompt = build_section_lang_match_prompt(mt, md, mk, h1, content_texts)
            yn = llm_yesno(prompt, purpose=f"EEAT section language match #{idx}")
            if yn == "Нет":
                prompt2 = build_section_lang_list_bad_prompt(mt, md, mk, h1, content_texts)
                bad = llm_text(prompt2, purpose="EEAT list bad items", max_tokens=64).strip()
                bad = re.sub(r'\s*\|\s*', ' | ', bad)
                if bad in ("—", "-", ""):
                    se.append("Язык заголовков/меты не совпадает с языком текста секции.")
                else:
                    se.append(f"Язык заголовков/меты не совпадает с языком текста секции. (Проблемные элементы: {bad}.)")
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
        has_code = llm_yesno(q1, purpose="Promo code scan (EEAT)")
        if has_code == "Да":
            q2 = build_promo_extract_prompt(all_text)
            codes = llm_text(q2, purpose="Promo code extract (EEAT)", max_tokens=256)
            codes = codes.strip()
            if codes and codes != "—":
                errors.append(f"EEAT: обнаружен посторонний промокод: {codes}")
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
            errors = validate_text_article(doc)
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
    max_len = 3500
    chunk: List[str] = []
    current_len = 0
    title = heading
    for line in logs:
        if current_len + len(line) + 1 > max_len and chunk:
            await message.answer(f"{title}:\n" + "\n".join(chunk))
            chunk = []
            current_len = 0
            title = "🛠 Лог LLM (продолжение)"
        chunk.append(line)
        current_len += len(line) + 1
    if chunk:
        await message.answer(f"{title}:\n" + "\n".join(chunk))

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