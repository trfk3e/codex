import os
import sys
import re
import json
import random
import requests
import threading
import time
import logging
import configparser
import ctypes
from typing import Optional

import customtkinter as ctk
from tkinter import filedialog, messagebox

# === исправление путей для PyInstaller ===
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =========================================

# Пути
API_FILE       = os.path.join(BASE_DIR, "API.txt")
BAD_API_FILE   = os.path.join(BASE_DIR, "BadAPI.txt")
SETTINGS_FILE  = os.path.join(BASE_DIR, "settings.ini")
LOG_TEMPLATE   = os.path.join(BASE_DIR, "log_{}.txt")

# Плейсхолдер для пустых строк в редакторе плана
PLACEHOLDER = "➖➖➖"

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "debug.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────── Настройки ───────────────────

def load_settings(path=SETTINGS_FILE):
    config = configparser.ConfigParser()
    if os.path.exists(path):
        config.read(path, encoding="utf-8")
    if "main" not in config:
        config["main"] = {}
    # значения по умолчанию
    config["main"].setdefault("plan_path", "")
    config["main"].setdefault("api_path", API_FILE)
    config["main"].setdefault("output_dir", "")
    config["main"].setdefault("language", "Lang")
    config["main"].setdefault("currency", "EUR")
    config["main"].setdefault("h1", "100-180")
    config["main"].setdefault("h2", "100-150")
    config["main"].setdefault("h3", "100-150")
    # новые параметры анти‑429
    config["main"].setdefault("tpm_limit", "30000")      # tokens per minute org-wide
    config["main"].setdefault("max_tokens_part", "1100") # потолок вывода на часть
    return config


def save_settings(config, path=SETTINGS_FILE):
    with open(path, "w", encoding="utf-8") as f:
        config.write(f)


def prune_logs(directory=BASE_DIR, keep=5):
    pattern = re.compile(r"log_\d+\.txt")
    files = [f for f in os.listdir(directory) if pattern.match(f)]
    files.sort(reverse=True)
    for old in files[keep:]:
        try:
            os.remove(os.path.join(directory, old))
        except Exception:
            pass


# ─────────────────── Исключения ───────────────────

class APIError(Exception):
    """Базовый класс ошибок API."""


class InvalidAPIKeyError(APIError):
    """Неверный/отозванный ключ (401)."""


class RateLimitError(APIError):
    """429: временный лимит (токены/мин или запросы/мин)."""
    def __init__(self, retry_after: Optional[float] = None, is_org_level: bool = True, raw: str = ""):
        super().__init__(f"Rate limit (retry_after={retry_after}, org_level={is_org_level})")
        self.retry_after = retry_after
        self.is_org_level = is_org_level
        self.raw = raw


class QuotaExceededError(APIError):
    """429: исчерпана квота/биллинг (не временно)."""


# ─────────────────── Работа с ключами ───────────────────

def load_api_keys(path=API_FILE):
    """Загрузить API ключи из файла."""
    if not os.path.exists(path):
        open(path, "a", encoding="utf-8").close()
    keys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            k = line.strip().lstrip("\ufeff")
            if k:
                keys.append(k)
    logger.debug("Loaded %d API keys from %s", len(keys), path)
    return keys


def save_bad_keys(bad, path=BAD_API_FILE):
    """Добавить «плохие» ключи в файл."""
    if not os.path.exists(path):
        open(path, "a", encoding="utf-8").close()
    if not bad:
        return
    with open(path, "a", encoding="utf-8") as f:
        for k in bad:
            f.write(k + "\n")
    logger.info("Saved %d bad API keys to %s", len(bad), path)


def update_bad_api_key(bad_key, api_keys_file=API_FILE):
    """Удалить bad_key из API.txt и добавить в BadAPI.txt."""
    try:
        if os.path.exists(api_keys_file):
            with open(api_keys_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            new_lines = [line for line in lines if bad_key not in line]
            with open(api_keys_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            logger.info("Removed bad key %s from %s", bad_key[:8] + "…", api_keys_file)
    except Exception as e:
        logger.error("Ошибка при обновлении API ключей: %s", e)
    try:
        with open(BAD_API_FILE, "a", encoding="utf-8") as f:
            f.write(bad_key + "\n")
        logger.info("Added bad key %s to %s", bad_key[:8] + "…", BAD_API_FILE)
    except Exception as e:
        logger.error("Ошибка при записи bad API ключа: %s", e)


# ─────────────────── Анти‑429 утилиты ───────────────────

def _parse_retry_after(response) -> Optional[float]:
    # 1) заголовок Retry-After
    ra = response.headers.get("Retry-After")
    if ra:
        try:
            return float(ra)
        except Exception:
            pass
    # 2) из тела ошибки "Please try again in X ms/s"
    try:
        data = response.json()
        msg = ((data or {}).get("error") or {}).get("message", "")
        m = re.search(r"try again in\s+([\d\.]+)\s*(ms|s)", msg, flags=re.I)
        if m:
            val = float(m.group(1))
            return val / 1000.0 if m.group(2).lower() == "ms" else val
    except Exception:
        pass
    return None


def _is_org_level_limit(response_json: dict) -> bool:
    msg = ((response_json or {}).get("error") or {}).get("message", "").lower()
    return "in organization" in msg or "org-" in msg


def estimate_tokens_from_messages(messages: list, extra_completion_tokens: int = 0) -> int:
    """Грубая оценка токенов (без tiktoken): ~3.6 символа/токен + накладные."""
    total_chars = 0
    for m in messages:
        total_chars += len(m.get("content", ""))
    approx = int(total_chars / 3.6) + 8 * len(messages)
    return approx + int(extra_completion_tokens or 0)


class TokenBucket:
    """Простой token bucket по TPM (tokens per minute)."""
    def __init__(self, capacity: int, refill_per_sec: float):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill = float(refill_per_sec)
        self.t = time.monotonic()
        self.lock = threading.Lock()

    def reserve(self, need: int) -> float:
        with self.lock:
            now = time.monotonic()
            dt = now - self.t
            self.t = now
            self.tokens = min(self.capacity, self.tokens + dt * self.refill)
            if need <= self.tokens:
                self.tokens -= need
                return 0.0
            deficit = need - self.tokens
            wait = deficit / self.refill
            self.tokens = 0.0
            return wait


# ─────────────────── Вызов API ───────────────────

def call_api(messages, api_key, log_file=None, model="o3", max_tokens=None, timeout=180):
    """Отправить запрос в OpenAI Chat Completions и вернуть текст ответа."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {"model": model, "messages": messages}
    if max_tokens is not None:
        data["max_tokens"] = int(max_tokens)

    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("USER:\n" + messages[-1]["content"] + "\n")
        except Exception as e:
            logger.debug("Failed to write user log: %s", e)

    logger.debug("Calling API with key %s", api_key[:8] + "…")
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=timeout,
        )
    except Exception as exc:
        logger.error("Network error: %s", exc)
        raise APIError(f"Network error: {exc}") from exc

    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        if log_file:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write("ASSISTANT:\n" + content + "\n")
            except Exception as e:
                logger.debug("Failed to write assistant log: %s", e)
        return content

    # --- ошибки ---
    try:
        payload = response.json()
    except Exception:
        payload = {}

    if response.status_code == 401:
        logger.error("Invalid API key")
        raise InvalidAPIKeyError("Invalid API key")

    if response.status_code == 429:
        code = ((payload.get("error") or {}).get("code") or "").lower()
        if "insufficient_quota" in code or "billing_hard_limit" in code:
            raise QuotaExceededError(response.text)
        retry_after = _parse_retry_after(response)
        is_org = _is_org_level_limit(payload)
        raise RateLimitError(retry_after=retry_after, is_org_level=is_org, raw=response.text)

    logger.error("API error %s: %s", response.status_code, response.text)
    raise APIError(f"Ошибка API: {response.status_code} - {response.text}")


def validate_api_keys(keys):
    """Вернуть первый «живой» ключ и список гарантированно плохих.
    Важно: 429 считаем признаком «ключ валидный, просто лимит».
    """
    bad = []
    for key in keys:
        logger.info("Validating API key %s...", key[:8] + "…")
        try:
            # минимизируем расход токенов при проверке
            call_api([{"role": "user", "content": "ping"}], key, model="o3", max_tokens=1)
            logger.info("API key %s is valid", key[:8] + "…")
            return key, bad
        except RateLimitError:
            logger.info("API key %s is valid but currently rate-limited", key[:8] + "…")
            return key, bad
        except QuotaExceededError:
            logger.warning("API key %s quota exceeded", key[:8] + "…")
            bad.append(key)
        except InvalidAPIKeyError:
            logger.warning("API key %s is invalid", key[:8] + "…")
            bad.append(key)
        except Exception as e:
            logger.error("Error validating key %s: %s", key[:8] + "…", e)
    return None, bad


# ─────────────────── Разбор плана / утилиты ───────────────────

def parse_plan(plan_path):
    """Вернуть список частей из файла плана (разделитель — пустая строка)."""
    if not os.path.exists(plan_path):
        open(plan_path, "a", encoding="utf-8").close()
        return []
    parts = []
    current = []
    with open(plan_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()
            if line == "":
                if current:
                    parts.append("\n".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        parts.append("\n".join(current))
    logger.debug("Parsed %d plan parts from %s", len(parts), plan_path)
    return parts


def sanitize_filename(name):
    name = re.sub(r"[^a-zA-Z0-9\-_ ]", "", name)
    return name.strip().replace(" ", "_")


def write_html(folder, title, content):
    safe = sanitize_filename(title)
    filename = os.path.join(folder, safe + ".html")
    count = 1
    while os.path.exists(filename):
        filename = os.path.join(folder, f"{safe}_{count}.html")
        count += 1
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename


def normalize_html(html):
    """Очистка/нормализация HTML из API."""
    cleaned = re.sub(r"</?html.*?>", "", html, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?body.*?>", "", cleaned, flags=re.IGNORECASE)

    def split_header(match):
        level = match.group(1)
        text = match.group(2).strip()
        if len(text.split()) > 12:
            for sep in [".", "?", "!"]:
                if sep in text:
                    idx = text.find(sep) + 1
                    header = text[:idx].strip()
                    rest = text[idx:].strip()
                    if rest:
                        return f"<h{level}>{header}</h{level}>\n<p>{rest}</p>"
        return match.group(0)

    cleaned = re.sub(r"<h([1-6])>(.*?)</h\1>", split_header, cleaned, flags=re.DOTALL)
    cleaned = "\n".join(line for line in cleaned.splitlines() if "`html" not in line and "```" not in line)
    cleaned = cleaned.strip()
    return "<html>\n<body>\n" + cleaned + "\n</body>\n</html>"


# ─────────────────── Генерация: промпты ───────────────────

def generate_initial_message(total, language, currency, h1amount, h2amount, h3amount):
    return (
        f"Часть 0/{total}\n"
        "Привет. Твоя роль - ты профессиональный SEO копирайтер в сфере I-Gaming специализирующийся писать уникальные текста которые будут без роботных слов для ZeroGPT на основе реальных фактов и реальной информации.\n"
        "Сейчас мы будем писать HTML текст по частям. Он будет состоять из N частей\n"
        f"Делай, думай и пиши только на - {language}. Валюта - {currency} - не упоминай слишком часто валюту, она указана только для таблиц.\n\n"
        "Инструкция:\n\n"
        f"H1: Заголовок - <h1></h1>\nНиже текст в <p></p>. Не больше {h1amount} слов важно чтобы рандомно. НЕ МЕНЬШЕ, ЭТО ОЧЕНЬ ВАЖНО\n"
        f"H2: - Заголовок - <h2></h2>\nНиже текст в <p></p> Не больше {h2amount} слов важно чтобы рандомно. НЕ МЕНЬШЕ, ЭТО ОЧЕНЬ ВАЖНО\n"
        f"H3: - Заголовок - <h3></h3>\nНиже текст в <p></p> Не больше {h3amount} слов важно чтобы рандомно. НЕ МЕНЬШЕ, ЭТО ОЧЕНЬ ВАЖНО\n\n"
        "Если упоминается что нужно сделать таблицу используем данный набор для таблиц:\n\n"
        "Таблицу делаем в - <table style=\"border:0.5px solid black;border-collapse:collapse;\"><tr><th style=\"border:0.5px solid black;\"></th></table>\n"
        "Очень важно, не делай пустые таблицы, либо таблицы одной строкой. Таблица должно быть сформирована ровно, хорошо заполнена и качественно. Попробуй найти реальную информацию. Если уже крайний случай - бери информацию со страниц конкурентов\n"
        "Очень важно чтобы таблица была ровной, списки слева а ответы справа, и заполненной не одной строкой и не с одним или 2 столбиками. Не давай комментарии к таблицам, не нужно им придумывать подписи! Таблица должна иметь минимум 2 колонки максимум 5-6 колонок\n"
        "Таблицы не должны быть огромными или состоять из 1-2 полоски горизонтальной или вертикальной или иметь столько колонок, что она не будет помещаться на странице или быть кривой и плыть аж вниз - у нее должен быть размер - стандарт.\n\n"
        "Если упоминается маркированный список, используем данный набор для маркированных списков:\n\n"
        "Маркированный список в HTML создается с помощью тега <ul>, а каждый элемент списка оформляется с помощью тега <li>.\n\n"
        "Если упоминается нумерованный список, используем данный набор для нумерованных списков:\n\n"
        "<ol>\n  <li>Первый элемент</li>\n  <li>Второй элемент</li>\n  <li>Третий элемент</li>\n</ol>\n\n"
        f"В конце в последнем {total} разделе FAQ в H2 заголовке не должно быть текста. В ответах должно быть не больше 30-50 слов! Это очень важно!\n\n"
        "Пиши исключительно в человеческом собственном стиле, чтобы текст не распознавался как AI generated\n"
        "После точки '.' текст должен всегда начинаться с большой буквы!\n"
        "Прорабатывай таблицы, так-как они почти всегда кривые, а мне нужны красивые таблички для сайта\n"
        "НЕ ВСТАВЛЯЙ НИКАКИЕ ССЫЛКИ\n"
        "Ты готов получать задания к написанию текста? Пиши исключительно в человеческом собственном стиле, чтобы текст не распознавался как AI generated"
        "\n\n⭑ LANGUAGE-AGNOSTIC ⭑\n"
        "При генерации текста ChatGPT обязан применять WHITELIST и чистить BLACKLIST независимо от языка исходного задания.\n"
        "Если контент не на английском — найди и перефразируй локальные эквиваленты запрещённых клише.\n\n"
        "Правила:\n"
        "1) Выполни WHITELIST до генерации.\n"
        "2) Просканируй черновик; любое совпадение с BLACKLIST → перепиши/удали.\n"
        "3) Минимум 10 % ручного рерайта + AI-check перед выдачей.\n\n"
        "═══════════  ✅  WHITELIST — ОБЯЗАТЕЛЬНО ДЕЛАЕМ  ═══════════\n"
        "• Живой вводный абзац; свежие даты/факты.\n"
        "• Только проверяемые факты: лицензия, рейтинг, дата публикации и т. п.\n"
        "• Ситуативные H2/H3-вопросы — о реально изменившихся вещах.\n"
        "• Мини-наблюдение/деталь (подвох в бонусах и т. п.).\n"
        "• «Что было / что стало» для апдейтов.\n"
        "• Структура: H1 → H2 плюсы/минусы → H3 FAQ → CTA.\n"
        "• Техника: ≥ 10 % ручного рерайта + AI probability check.\n\n"
        "═══════════  ⛔  BLACKLIST — КАТЕГОРИЧЕСКИ УБИРАЕМ  ═══════════\n"
        "1. Штампы, ложные гарантии, спам-эпитеты.\n"
        "2. Срочные призывы и кликбейты.\n"
        "3. Ложный авторитет, keyword-stuffing и пр.\n\n"
        "Алгоритм: сгенерируй → проверь WHITELIST → очисти BLACKLIST → AI-чек → выдай текст."
    )


def generate_part_message(num, total, part, h1amount, h2amount, h3amount):
    base = (
        f"Часть {num}/{total}\n"
        "Пиши только внутри HTML кода. Ищи информацию в интернете. Если ее там нету выдумай сам - главное чтобы это было близко к реальности. Используй только настоящую информацию для заполнения, пиши исключительно по теме заголовков ничего более.\n"
        "Ключевые слова должны входить гармонично и красиво в текст, выдели их жирным шрифтом <strong></strong>, не используй \"**\".\n"
        "Ключевые слова не должны встречаться в одном предложении, каждому ключу — своё предложение.\n"
        "После точки '.' текст всегда с большой буквы. Не вставляй больше чем нужно ключевых слов.\n\n"
        f"Если заголовок H1: Ниже текст в <p></p>. Не больше {h1amount} слов (рандомно). НЕ МЕНЬШЕ, ЭТО ОЧЕНЬ ВАЖНО\n"
        f"Если заголовок H2: Ниже текст в <p></p>. Не больше {h2amount} слов (рандомно). НЕ МЕНЬШЕ, ЭТО ОЧЕНЬ ВАЖНО\n"
        f"Если заголовок H3: Ниже текст в <p></p>. Не больше {h3amount} слов (рандомно). НЕ МЕНЬШЕ, ЭТО ОЧЕНЬ ВАЖНО\n\n"
        "Пиши исключительно в человеческом стиле, без шаблонных фраз. ZeroGPT не должен распознавать текст как роботный.\n\n"
        "Вот сами заголовки:\n"
    )
    return base + part


# ─────────────────── Класс приложения ───────────────────

DEFAULT_TPM_LIMIT = 30000
DEFAULT_MAX_TOKENS_PER_PART = 1100
BACKOFF_BASE = 1.5
BACKOFF_MAX = 30.0
JITTER_FRAC = 0.15

def _with_jitter(seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    span = seconds * JITTER_FRAC
    return max(0.0, seconds + random.uniform(-span, span))


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        global API_FILE
        self.title("Generator")
        self.minsize(800, 500)
        self._bind_clipboard_shortcuts()
        self.after(0, lambda: self.state("zoomed"))

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.config_parser = load_settings()

        self.plan_path_var = ctk.StringVar(value=self.config_parser["main"].get("plan_path", ""))
        self.api_path_var = ctk.StringVar(value=self.config_parser["main"].get("api_path", API_FILE))
        self.output_dir_var = ctk.StringVar(value=self.config_parser["main"].get("output_dir", ""))

        if not self.output_dir_var.get() and self.plan_path_var.get():
            self.output_dir_var.set(os.path.dirname(self.plan_path_var.get()))

        self.language_var = ctk.StringVar(value=self.config_parser["main"].get("language", "Lang"))
        self.currency_var = ctk.StringVar(value=self.config_parser["main"].get("currency", "EUR"))
        self.h1_var = ctk.StringVar(value=self.config_parser["main"].get("h1", "100-180"))
        self.h2_var = ctk.StringVar(value=self.config_parser["main"].get("h2", "100-150"))
        self.h3_var = ctk.StringVar(value=self.config_parser["main"].get("h3", "100-150"))

        # Анти‑429 настройки
        self.tpm_limit_var = ctk.StringVar(value=self.config_parser["main"].get("tpm_limit", str(DEFAULT_TPM_LIMIT)))
        self.max_tokens_part_var = ctk.StringVar(value=self.config_parser["main"].get("max_tokens_part", str(DEFAULT_MAX_TOKENS_PER_PART)))

        # обновляем глобальный путь к ключам
        API_FILE = self.api_path_var.get()

        self.log_file = None
        self._progress = {"idx": 1, "html_parts": []}  # прогресс генерации

        # Сетка
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=280)
        self.grid_columnconfigure(1, weight=1)

        settings_btn = ctk.CTkButton(self, text="⚙", width=40, height=32, command=self.open_settings)
        settings_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)

        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left.grid_columnconfigure(1, weight=1)
        left.grid_rowconfigure(8, weight=1)

        ctk.CTkLabel(left, text="Язык").grid(row=0, column=0, padx=5, pady=(5,0), sticky="w")
        ctk.CTkEntry(left, textvariable=self.language_var).grid(row=0, column=1, padx=5, pady=(5,0), sticky="ew")

        ctk.CTkLabel(left, text="Валюта").grid(row=1, column=0, padx=5, pady=(5,0), sticky="w")
        ctk.CTkEntry(left, textvariable=self.currency_var).grid(row=1, column=1, padx=5, pady=(5,0), sticky="ew")

        ctk.CTkLabel(left, text="Слов в H1").grid(row=2, column=0, padx=5, pady=(5,0), sticky="w")
        ctk.CTkEntry(left, textvariable=self.h1_var).grid(row=2, column=1, padx=5, pady=(5,0), sticky="ew")

        ctk.CTkLabel(left, text="Слов в H2").grid(row=3, column=0, padx=5, pady=(5,0), sticky="w")
        ctk.CTkEntry(left, textvariable=self.h2_var).grid(row=3, column=1, padx=5, pady=(5,0), sticky="ew")

        ctk.CTkLabel(left, text="Слов в H3").grid(row=4, column=0, padx=5, pady=(5,0), sticky="w")
        ctk.CTkEntry(left, textvariable=self.h3_var).grid(row=4, column=1, padx=5, pady=(5,0), sticky="ew")

        # Новые поля анти‑429
        ctk.CTkLabel(left, text="TPM лимит (org)").grid(row=5, column=0, padx=5, pady=(5,0), sticky="w")
        ctk.CTkEntry(left, textvariable=self.tpm_limit_var).grid(row=5, column=1, padx=5, pady=(5,0), sticky="ew")

        ctk.CTkLabel(left, text="Max tokens/часть").grid(row=6, column=0, padx=5, pady=(5,0), sticky="w")
        ctk.CTkEntry(left, textvariable=self.max_tokens_part_var).grid(row=6, column=1, padx=5, pady=(5,0), sticky="ew")

        ctk.CTkButton(left, text="Настройки", command=self.open_settings).grid(row=7, column=0, columnspan=2, pady=(5,0))
        ctk.CTkButton(left, text="Старт", command=self.start).grid(row=8, column=0, columnspan=2, pady=10)

        self.log_box = ctk.CTkTextbox(left)
        self.log_box.grid(row=9, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.log_box.configure(state="disabled")

        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.plan_text = ctk.CTkTextbox(right)
        self.plan_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.text_widget = getattr(self.plan_text, "_textbox", self.plan_text)
        if hasattr(self.text_widget, "tag_configure"):
            self.text_widget.tag_configure("placeholder")
        if hasattr(self.text_widget, "configure"):
            self.text_widget.configure(undo=True)

        self.text_widget.bind("<<Modified>>", self._on_text_modified)
        self.text_widget.bind("<Key>", self._protect_placeholders)
        self.text_widget.bind("<<Undo>>", lambda e: self.text_widget.edit_undo())
        self.text_widget.bind("<<Redo>>", lambda e: self.text_widget.edit_redo())
        self._updating_placeholders = False

        ctk.CTkButton(right, text="Сохранить", command=self.save_plan_file).grid(row=1, column=0, pady=5)

        self.load_plan_file()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ───── UI/вспомогательные методы ─────

    def center_window(self, win=None):
        target = win or self
        target.update_idletasks()
        width = target.winfo_width()
        height = target.winfo_height()
        if width <= 1 or height <= 1:
            geom = target.geometry()
            m = re.match(r"(\d+)x(\d+)", geom)
            if m:
                width = int(m.group(1))
                height = int(m.group(2))
        x = (target.winfo_screenwidth() - width) // 2
        y = (target.winfo_screenheight() - height) // 2
        target.geometry(f"{width}x{height}+{x}+{y}")

    def _bind_clipboard_shortcuts(self):
        """Горячие клавиши Ctrl на любой раскладке (Windows)."""
        def is_english_layout():
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            thread_id = ctypes.windll.user32.GetWindowThreadProcessId(hwnd, 0)
            hkl = ctypes.windll.user32.GetKeyboardLayout(thread_id)
            langid = hkl & 0xFFFF
            return langid & 0x3FF == 0x09

        def _handle_ctrl(event):
            if is_english_layout():
                return
            if event.state & 0x4:
                kc = event.keycode
                if kc == 67:
                    event.widget.event_generate('<<Copy>>');  return 'break'
                if kc == 88:
                    event.widget.event_generate('<<Cut>>');   return 'break'
                if kc == 86:
                    event.widget.event_generate('<<Paste>>'); return 'break'
                if kc == 65:
                    event.widget.event_generate('<<SelectAll>>'); return 'break'
                if kc == 90:
                    event.widget.event_generate('<<Undo>>');  return 'break'
                if kc == 89:
                    event.widget.event_generate('<<Redo>>');  return 'break'
        self.bind_all('<Control-KeyPress>', _handle_ctrl)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
        if file_path:
            self.plan_path_var.set(file_path)
            self.load_plan_file()
            self.save_settings()

    def load_plan_file(self):
        path = self.plan_path_var.get()
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.plan_text.delete("1.0", "end")
            self.plan_text.insert("1.0", content)
            self.update_placeholders()

    def save_plan_file(self):
        path = self.plan_path_var.get()
        if not path:
            path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
            if not path:
                return
            self.plan_path_var.set(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.get_plan_content())
        self.save_settings()

    def get_plan_content(self):
        text = self.plan_text.get("1.0", "end-1c")
        lines = []
        for line in text.split("\n"):
            if line.strip() == PLACEHOLDER:
                lines.append("")
            else:
                lines.append(line)
        return "\n".join(lines).rstrip()

    def _on_text_modified(self, event=None):
        if getattr(self, "_updating_placeholders", False):
            return
        widget = self.text_widget
        if hasattr(widget, "edit_modified") and widget.edit_modified():
            widget.edit_modified(False)
            self.update_placeholders()

    def update_placeholders(self):
        """Подставляем плейсхолдеры вместо пустых строк (для наглядности)."""
        widget = self.text_widget
        widget.mark_set("_restore", "insert")
        if self.plan_text.tag_ranges("sel"):
            widget.mark_set("_sel_start", "sel.first")
            widget.mark_set("_sel_end", "sel.last")
        self._updating_placeholders = True
        lines = int(widget.index("end-1c").split(".")[0])
        for i in range(1, lines + 1):
            start = f"{i}.0"
            end = f"{i}.end"
            text = widget.get(start, end)

            if i == 1:
                if text == PLACEHOLDER:
                    widget.delete(start, end)
                if "placeholder" in widget.tag_names(start):
                    widget.tag_remove("placeholder", start, end)
                continue

            if text.strip() == "":
                if text != PLACEHOLDER:
                    widget.delete(start, end)
                    widget.insert(start, PLACEHOLDER, "placeholder")
                else:
                    widget.tag_add("placeholder", start, end)
            else:
                if "placeholder" in widget.tag_names(start):
                    widget.tag_remove("placeholder", start, end)
        if hasattr(widget, "edit_modified"):
            widget.edit_modified(False)
        widget.mark_set("insert", "_restore")
        widget.mark_unset("_restore")
        if "_sel_start" in widget.mark_names():
            widget.tag_remove("sel", "1.0", "end")
            widget.tag_add("sel", "_sel_start", "_sel_end")
            widget.mark_unset("_sel_start")
            widget.mark_unset("_sel_end")
        self._updating_placeholders = False

    def _protect_placeholders(self, event):
        widget = self.text_widget
        index = widget.index("insert")

        def line_is_placeholder(num):
            return widget.get(f"{num}.0", f"{num}.end") == PLACEHOLDER

        if event.keysym in ("BackSpace", "Delete"):
            if self.plan_text.tag_ranges("sel"):
                return
            line, col = map(int, index.split("."))
            if line_is_placeholder(line):
                widget.delete(f"{line}.0", f"{line}.end+1c")
                self.update_placeholders()
                return "break"
            if event.keysym == "BackSpace" and col == 0 and line > 1 and line_is_placeholder(line - 1):
                widget.delete(f"{line-1}.0", f"{line-1}.end+1c"); self.update_placeholders(); return "break"
            if event.keysym == "Delete" and index == widget.index(f"{line}.end") and line_is_placeholder(line + 1):
                widget.delete(f"{line+1}.0", f"{line+1}.end+1c"); self.update_placeholders(); return "break"
        else:
            if widget.get(f"{index.split('.')[0]}.0", f"{index.split('.')[0]}.end") == PLACEHOLDER:
                if event.keysym not in ("Left", "Right", "Up", "Down", "Home", "End"):
                    return "break"
            if self.plan_text.tag_ranges("sel") and event.char:
                start = widget.index("sel.first")
                end = widget.index("sel.last")
                rng = widget.tag_nextrange("placeholder", start, end)
                if rng:
                    return "break"

    def log(self, msg):
        def _append():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
        self.log_box.after(0, _append)

    # ───── Работа с чекпоинтом ─────

    def _checkpoint_path(self, folder):
        return os.path.join(folder, ".generator_checkpoint.json")

    def _load_checkpoint(self, folder):
        path = self._checkpoint_path(folder)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "idx" in data and "html_parts" in data:
                    self._progress = {"idx": int(data["idx"]), "html_parts": list(data["html_parts"])}
                    self.log(f"Обнаружен чекпоинт. Продолжаем с части {self._progress['idx']}.")
                    return True
            except Exception as e:
                logger.warning("Не удалось загрузить чекпоинт: %s", e)
        self._progress = {"idx": 1, "html_parts": []}
        return False

    def _save_checkpoint(self, folder):
        path = self._checkpoint_path(folder)
        data = {"idx": self._progress.get("idx", 1), "html_parts": self._progress.get("html_parts", [])}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.debug("Не удалось сохранить чекпоинт: %s", e)

    def _clear_checkpoint(self, folder):
        path = self._checkpoint_path(folder)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    # ───── Кнопки ─────

    def start(self):
        if hasattr(self, "_running") and self._running:
            return
        missing = []
        if not self.plan_path_var.get():
            missing.append("H-plan")
        if not self.api_path_var.get():
            missing.append("API")
        if missing:
            self.log("Не заполнены настройки: " + ", ".join(missing))
            messagebox.showerror("Ошибка", "Не заполнены настройки: " + ", ".join(missing))
            return
        self._running = True
        threading.Thread(target=self.run_generation, daemon=True).start()

    def open_settings(self):
        win = ctk.CTkToplevel(self)
        win.title("Settings")
        win.geometry("560x300")
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()
        self.center_window(win)

        win.grid_columnconfigure(1, weight=1)

        plan_var = ctk.StringVar(value=self.plan_path_var.get())
        api_var = ctk.StringVar(value=self.api_path_var.get())
        out_var = ctk.StringVar(value=self.output_dir_var.get())
        tpm_var = ctk.StringVar(value=self.tpm_limit_var.get())
        max_tok_var = ctk.StringVar(value=self.max_tokens_part_var.get())

        def choose_plan():
            p = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
            if p:
                plan_var.set(p)

        def choose_api():
            p = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
            if p:
                api_var.set(p)

        def choose_out():
            p = filedialog.askdirectory()
            if p:
                out_var.set(p)

        ctk.CTkLabel(win, text="H-plan путь").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(win, textvariable=plan_var, width=300).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(win, text="...", command=choose_plan, width=30).grid(row=0, column=2, padx=5)

        ctk.CTkLabel(win, text="API путь").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(win, textvariable=api_var, width=300).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(win, text="...", command=choose_api, width=30).grid(row=1, column=2, padx=5)

        ctk.CTkLabel(win, text="Папка HTML").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(win, textvariable=out_var, width=300).grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(win, text="...", command=choose_out, width=30).grid(row=2, column=2, padx=5)

        ctk.CTkLabel(win, text="TPM лимит (org)").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(win, textvariable=tpm_var, width=150).grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(win, text="Max tokens/часть").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(win, textvariable=max_tok_var, width=150).grid(row=4, column=1, padx=5, pady=5, sticky="w")

        def save_and_close():
            self.plan_path_var.set(plan_var.get())
            self.api_path_var.set(api_var.get())
            self.output_dir_var.set(out_var.get())
            self.tpm_limit_var.set(tpm_var.get())
            self.max_tokens_part_var.set(max_tok_var.get())
            global API_FILE
            API_FILE = self.api_path_var.get()
            self.load_plan_file()
            self.save_settings()
            win.destroy()

        ctk.CTkButton(win, text="Сохранить", command=save_and_close).grid(row=5, column=0, columnspan=3, pady=10)

    def save_settings(self):
        cfg = self.config_parser
        cfg["main"]["plan_path"] = self.plan_path_var.get()
        cfg["main"]["api_path"] = self.api_path_var.get()
        cfg["main"]["language"] = self.language_var.get()
        cfg["main"]["currency"] = self.currency_var.get()
        cfg["main"]["h1"] = self.h1_var.get()
        cfg["main"]["h2"] = self.h2_var.get()
        cfg["main"]["h3"] = self.h3_var.get()
        cfg["main"]["output_dir"] = self.output_dir_var.get()
        cfg["main"]["tpm_limit"] = self.tpm_limit_var.get()
        cfg["main"]["max_tokens_part"] = self.max_tokens_part_var.get()
        save_settings(cfg)

    def on_close(self):
        self.save_settings()
        self.destroy()

    # ───── Генерация с учётом анти‑429 и резюма ─────

    def _generate_with_key(self, key, parts, folder):
        """
        Генерация всех частей данным ключом.
        Каждая часть — независимый запрос (system + user).
        При 429/временных ошибках повторяется ТЕКУЩАЯ часть без отката прогресса.
        При InvalidAPIKey/Quota — поднимается исключение; прогресс сохранён в self._progress.
        """
        total = len(parts)

        # Общий system для всех частей
        system_msg = {
            "role": "system",
            "content": generate_initial_message(
                total,
                self.language_var.get(),
                self.currency_var.get(),
                self.h1_var.get(),
                self.h2_var.get(),
                self.h3_var.get(),
            )
        }

        # Троттлер по TPM
        try:
            tpm_limit_val = int(float(self.tpm_limit_var.get()))
        except Exception:
            tpm_limit_val = DEFAULT_TPM_LIMIT
        bucket = TokenBucket(capacity=tpm_limit_val, refill_per_sec=tpm_limit_val / 60.0)

        # Порог max_tokens на часть
        try:
            max_tokens = int(float(self.max_tokens_part_var.get()))
        except Exception:
            max_tokens = DEFAULT_MAX_TOKENS_PER_PART

        # Прогресс (может быть восстановлен из чекпоинта)
        idx = int(self._progress.get("idx", 1))
        html_parts = list(self._progress.get("html_parts", []))

        while idx <= total:
            part = parts[idx - 1]
            user_msg = {
                "role": "user",
                "content": generate_part_message(
                    idx, total, part,
                    self.h1_var.get(), self.h2_var.get(), self.h3_var.get()
                )
            }
            messages = [system_msg, user_msg]

            # Превентивный троттлинг
            est_req_tokens = estimate_tokens_from_messages(messages, extra_completion_tokens=max_tokens)
            wait_pre = bucket.reserve(est_req_tokens)
            if wait_pre > 0:
                w = _with_jitter(wait_pre)
                self.log(f"TPM троттлинг: ждём {w:.2f} с перед частью {idx}")
                time.sleep(w)

            attempt = 0
            while True:
                attempt += 1
                try:
                    result = call_api(messages, key, self.log_file, model="o3", max_tokens=max_tokens)
                    html_parts.append(result + "\n")
                    self._progress["html_parts"] = html_parts
                    self._progress["idx"] = idx + 1
                    self._save_checkpoint(folder)

                    self.log(f"Часть {idx} сгенерирована")
                    logger.info("Part %d generated", idx)
                    idx += 1
                    break  # к следующей части

                except InvalidAPIKeyError:
                    # сохраняем прогресс и пробрасываем — пусть верхний слой сменит ключ
                    self._progress["idx"] = idx
                    self._progress["html_parts"] = html_parts
                    self._save_checkpoint(folder)
                    raise

                except QuotaExceededError as e:
                    # квота исчерпана — ключ бесполезен; сохраняем прогресс и выходим
                    self._progress["idx"] = idx
                    self._progress["html_parts"] = html_parts
                    self._save_checkpoint(folder)
                    self.log(f"Квота исчерпана для ключа {key[:8]}… на части {idx}")
                    logger.warning("Quota exceeded for key %s at part %d", key[:8] + "…", idx)
                    raise

                except RateLimitError as e:
                    # временный 429 — ждём и повторяем ЭТУ ЖЕ часть
                    suggested = e.retry_after if e.retry_after is not None else 0.0
                    backoff = min(BACKOFF_MAX, BACKOFF_BASE ** min(attempt, 8))
                    sleep_for = _with_jitter(max(suggested, backoff))
                    self.log(
                        f"429 (TPM) на части {idx}. Повтор через {sleep_for:.2f} с "
                        f"(org={e.is_org_level}, попытка {attempt})"
                    )
                    logger.info("Rate limited on part %d; sleeping %.2fs", idx, sleep_for)
                    time.sleep(sleep_for)
                    # повторим ту же часть

                except APIError as e:
                    # сеть/прочее — несколько попыток, затем отдаём управление выше (смена ключа/повтор)
                    if attempt < 3:
                        self.log(f"Ошибка на части {idx}: {e}. Повтор попытки ({attempt+1})...")
                        time.sleep(_with_jitter(2.0 * attempt))
                        continue
                    else:
                        self._progress["idx"] = idx
                        self._progress["html_parts"] = html_parts
                        self._save_checkpoint(folder)
                        self.log(f"Ошибка при генерации части {idx}: {e}")
                        logger.error("Part %d failed with non-rate-limit error: %s", idx, e)
                        return None  # пусть верхний уровень решит, что делать дальше

        # Готово — пишем файл
        first_header = ''
        for line in parts[0].splitlines():
            if line.startswith("H1:"):
                first_header = line[3:].strip()
                break
        if not first_header:
            first_header = "result"

        cleaned = normalize_html("".join(html_parts))
        filename = write_html(folder, first_header, cleaned)
        self._clear_checkpoint(folder)
        logger.info("File saved: %s", filename)
        return filename

    def run_generation(self):
        # лог-файл на запуск
        self.log_file = LOG_TEMPLATE.format(int(time.time()))
        open(self.log_file, "a", encoding="utf-8").close()
        logger.info("Run log file: %s", self.log_file)
        prune_logs()

        plan_file = self.plan_path_var.get()
        if not plan_file:
            self.after(0, lambda: messagebox.showerror("Ошибка", "Не указан файл плана"))
            self.log("Не указан файл плана")
            self._running = False
            return

        api_file = self.api_path_var.get()
        if not api_file:
            self.after(0, lambda: messagebox.showerror("Ошибка", "Не указан файл API"))
            self.log("Не указан файл API")
            self._running = False
            return

        global API_FILE
        API_FILE = api_file
        self.save_plan_file()

        folder = self.output_dir_var.get().strip() or (os.path.dirname(plan_file) or ".")
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        if not os.path.exists(plan_file):
            open(plan_file, "a", encoding="utf-8").close()
            self.log(f"Создан файл плана: {plan_file}")
            logger.debug("Created plan file %s", plan_file)

        parts = parse_plan(plan_file)
        if not parts:
            self.after(0, lambda: messagebox.showerror("Ошибка", "План пуст"))
            self.log("План пуст")
            self._running = False
            return

        # Загружаем чекпоинт, если есть
        self._load_checkpoint(folder)

        self.log("Загрузка API ключей...")
        logger.info("Loading API keys")
        keys = load_api_keys()
        if not keys:
            self.after(0, lambda: messagebox.showerror("Ошибка", "Нет API ключей"))
            self.log("Нет API ключей")
            logger.error("No API keys found")
            self._running = False
            return

        while True:
            if not keys:
                self.after(0, lambda: messagebox.showerror("Ошибка", "Нет рабочих API ключей"))
                self.log("Нет рабочих ключей")
                logger.error("No working API keys")
                self._running = False
                return

            self.log("Проверка ключей...")
            logger.info("Validating API keys")
            key, bad = validate_api_keys(keys)
            for b in bad:
                update_bad_api_key(b, API_FILE)
                if b in keys:
                    keys.remove(b)
                self.log(f"Ключ не работает: {b[:8]}…")
                logger.info("Bad key %s moved to %s", b[:8] + "…", BAD_API_FILE)

            if not key:
                # все ключи некорректны/закончилась квота
                continue

            self.log(f"Используется ключ: {key[:8]}…")
            logger.info("Using API key %s", key[:8] + "…")

            try:
                filename = self._generate_with_key(key, parts, folder)
                if filename:
                    self.after(0, lambda fname=filename: messagebox.showinfo("Готово", f"Файл сохранен: {fname}"))
                    self._running = False
                    return
                else:
                    # Нефатальная ошибка внутри — попробуем ещё раз с этим же ключом
                    self.log("Ошибка при генерации, повторяем с тем же ключом…")
                    time.sleep(1.0)
                    continue

            except InvalidAPIKeyError:
                self.log(f"Ключ недействителен: {key[:8]}…")
                logger.warning("API key %s is invalid during generation", key[:8] + "…")
                update_bad_api_key(key, API_FILE)
                if key in keys:
                    keys.remove(key)
                continue

            except QuotaExceededError:
                self.log(f"Квота исчерпана у ключа: {key[:8]}… Пытаемся другим ключом.")
                logger.warning("Quota exceeded for key %s, switching", key[:8] + "…")
                update_bad_api_key(key, API_FILE)
                if key in keys:
                    keys.remove(key)
                continue

            except Exception as e:
                # Непредвиденное — завершаем
                self.after(0, lambda: messagebox.showerror("Ошибка", f"Критическая ошибка: {e}"))
                logger.exception("Critical error: %s", e)
                self._running = False
                return


def main():
    logger.info("Application started")
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
