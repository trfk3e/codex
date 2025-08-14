import os
import sys
import re
import requests
import threading
import time
import logging
import configparser
import ctypes
import customtkinter as ctk
from tkinter import filedialog, messagebox

# === исправление путей для PyInstaller ===
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =========================================

# теперь определяем остальные пути ОДИН раз
API_FILE       = os.path.join(BASE_DIR, "API.txt")
BAD_API_FILE   = os.path.join(BASE_DIR, "BadAPI.txt")
SETTINGS_FILE  = os.path.join(BASE_DIR, "settings.ini")
LOG_TEMPLATE   = os.path.join(BASE_DIR, "log_{}.txt")

# Placeholder text used to indicate blank lines in the plan editor
PLACEHOLDER = "➖➖➖"

# Configure basic logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "debug.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def load_settings(path=SETTINGS_FILE):
    config = configparser.ConfigParser()
    if os.path.exists(path):
        config.read(path, encoding="utf-8")
    if "main" not in config:
        config["main"] = {}
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


class APIError(Exception):
    """Base class for API related errors."""


class InvalidAPIKeyError(APIError):
    """Raised when the API reports an invalid key."""


def load_api_keys(path=API_FILE):
    """Load API keys from the given file."""
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


def call_api(messages, api_key, log_file=None):
    """Send a request to the OpenAI API and return the response text."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {"model": "o3", "messages": messages}

    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("USER:\n" + messages[-1]["content"] + "\n")
        except Exception as e:
            logger.debug("Failed to write user log: %s", e)

    logger.debug("Calling API with key %s", api_key[:8])
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=180,
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
    elif response.status_code == 401:
        logger.error("Invalid API key")
        raise InvalidAPIKeyError("Invalid API key")
    else:
        logger.error("API error %s: %s", response.status_code, response.text)
        raise APIError(f"Ошибка API: {response.status_code} - {response.text}")


def validate_api_keys(keys):
    """Return the first valid API key and list of definitely bad keys."""
    bad = []
    for key in keys:
        logger.info("Validating API key %s...", key[:8])
        try:
            call_api([{"role": "user", "content": "ping"}], key)
            logger.info("API key %s is valid", key[:8])
            return key, bad
        except InvalidAPIKeyError:
            logger.warning("API key %s is invalid", key[:8])
            bad.append(key)
        except Exception as e:
            logger.error("Error validating key %s: %s", key[:8], e)
    return None, bad


def save_bad_keys(bad, path=BAD_API_FILE):
    """Append bad API keys to the specified file."""
    if not os.path.exists(path):
        open(path, "a", encoding="utf-8").close()
    if not bad:
        return
    with open(path, "a", encoding="utf-8") as f:
        for k in bad:
            f.write(k + "\n")
    logger.info("Saved %d bad API keys to %s", len(bad), path)


def update_bad_api_key(bad_key, api_keys_file=API_FILE):
    """Remove bad_key from api_keys_file and append to BAD_API_FILE."""
    try:
        if os.path.exists(api_keys_file):
            with open(api_keys_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            new_lines = [line for line in lines if bad_key not in line]
            with open(api_keys_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            logger.info("Removed bad key %s from %s", bad_key[:8], api_keys_file)
    except Exception as e:
        logger.error("Ошибка при обновлении API ключей: %s", e)
    try:
        with open(BAD_API_FILE, "a", encoding="utf-8") as f:
            f.write(bad_key + "\n")
        logger.info("Added bad key %s to %s", bad_key[:8], BAD_API_FILE)
    except Exception as e:
        logger.error("Ошибка при записи bad API ключа: %s", e)


def parse_plan(plan_path):
    """Return list of parts from the plan file separated by blank lines."""
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


# ────────────────────────────────  ЧАСТЬ 1  ────────────────────────────────
# Стартовый промпт (добавление блока WHITELIST / BLACKLIST)

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

        # ─── НОВЫЙ БЛОК: WHITELIST / BLACKLIST ───
        "\n\n⭑ LANGUAGE-AGNOSTIC ⭑\n"
        "При генерации текста ChatGPT обязан применять WHITELIST и чистить BLACKLIST\n"
        "независимо от языка исходного задания.\n"
        "Если контент не на английском — найди и перефразируй локальные эквиваленты\n"
        "запрещённых клише, сохранив естественный стиль.\n\n"
        "Правила работы движка\n"
        "1) Выполни WHITELIST до генерации.\n"
        "2) Просканируй черновик; любое совпадение с BLACKLIST → перепиши/удали.\n"
        "3) Минимум 10 % ручного рерайта + AI-check перед выдачей.\n\n"
        "═══════════  ✅  WHITELIST — ОБЯЗАТЕЛЬНО ДЕЛАЕМ  ═══════════\n"
        "• Живой вводный абзац (2-3 строки): свежие даты, гео, событие.\n"
        "• Только проверяемые факты: лицензия, рейтинг Trustpilot, дата публикации на авторитетном портале и т. д.\n"
        "• Ситуативные H2/H3-вопросы — о реально изменившихся вещах.\n"
        "• Мини-наблюдение / личная деталь (напр., подвох в бонусах).\n"
        "• Формат «что было / что стало» для любых апдейтов.\n"
        "• Структура: H1 бренд → H2 плюсы/минусы → H3 FAQ → CTA «Перейти на сайт».\n"
        "• Целевые метрики (по тесту 30 стр.): индекс ≥ 90 %, позиция ≤ 18.4, dwell ≥ 71 с.\n"
        "• Техника: ≥ 10 % ручного рерайта + проверка «AI probability» до публикации.\n\n"
        "═══════════  ⛔  BLACKLIST — КАТЕГОРИЧЕСКИ УБИРАЕМ  ═══════════\n"
        "1. Финальные/начальные штампы и обобщения\n"
        "2. Срочный призыв к клику / бонусу\n"
        "3. Ложный авторитет / experts-speak\n"
        "4. Безосновательные гарантии\n"
        "5. Переспамленные эпитеты и триггер-слова\n"
        "6. Шаблонные заголовки-причины\n"
        "7. Типовые CTA-якоря\n"
        "8. Любое keyword-stuffing и все семь экспериментальных триггер-фраз\n\n"
        "────────────────────────────────────────────────────────────────────────\n"
        "Алгоритм: сгенерируй → проверь WHITELIST → очисти BLACKLIST (на любом языке) → AI-чек → выдай готовый текст."
    )


# ────────────────────────────────  ЧАСТЬ 2  ────────────────────────────────
# Промпт для каждой части (тот же принцип)

def generate_part_message(num, total, part, h1amount, h2amount, h3amount):
    base = (
        f"Часть {num}/{total}\n"
        "Пиши только внутри HTML кода. Ищи информацию в интернете. Если ее там нету выдумай сам - главное чтобы это было близко к реальности. Используй только настоящую информацию для заполнения, пиши исключительно по теме заголовков ничего более.\n"
        "Ключевые слова должны входить очень гармонично и красиво в текст, выдели их жирным шрифтом <strong></strong>, ни в коем случае не выделяй их с помощью \"**\".\n"
        "Очень важно чтобы ключевые слова не использовались в одном и том же предложении, они должны быть только в разных предложениях, для каждого ключевого слова свое предложение!\n"
        "После точки '.' текст должен всегда начинаться с большой буквы!\n"
        "Не вставляй больше чем нужно ключевых слов, следи за этим\n\n"
        f"Если заголовок H1: Ниже текст в <p></p>. Не больше {h1amount} слов важно чтобы рандомно. НЕ МЕНЬШЕ, ЭТО ОЧЕНЬ ВАЖНО\n"
        f"Если заголовок H2: Ниже текст в <p></p> Не больше {h2amount} слов важно чтобы рандомно. НЕ МЕНЬШЕ, ЭТО ОЧЕНЬ ВАЖНО\n"
        f"Если заголовок H3: Ниже текст в <p></p> Не больше {h3amount} слов важно чтобы рандомно. НЕ МЕНЬШЕ, ЭТО ОЧЕНЬ ВАЖНО\n\n"
        "Пиши исключительно в человеческом собственном стиле, чтобы текст не распознавался как AI generated\n"
        "Следи за написанием и вхождением ключевых слов, пиши красиво и человечно, пожалуйста. Текст должен быть написан без шаблонных фраз и терминов которые используются в копирайтинге. Сервис ZeroGPT не должен распознавать текст роботным.\n\n"

        # ─── НОВЫЙ БЛОК ───
        "⭑ LANGUAGE-AGNOSTIC ⭑\n"
        "При генерации текста ChatGPT обязан применять WHITELIST и чистить BLACKLIST … (полный текст блока из первого промпта)\n\n"

        "Вот сами заголовки: \n"
    )
    return base + part

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
    """Clean and normalize HTML returned from the API."""
    # remove stray html/body tags
    cleaned = re.sub(r"</?html.*?>", "", html, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?body.*?>", "", cleaned, flags=re.IGNORECASE)

    def split_header(match):
        level = match.group(1)
        text = match.group(2).strip()
        # If header text appears to contain an entire paragraph, split at first period
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

    # Remove any lines containing `html or ``` to strip code fences from the output
    cleaned = "\n".join(
        line
        for line in cleaned.splitlines()
        if "`html" not in line and "```" not in line
    )

    cleaned = cleaned.strip()
    return "<html>\n<body>\n" + cleaned + "\n</body>\n</html>"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        global API_FILE
        self.title("Generator")
        self.minsize(800, 500)
        self._bind_clipboard_shortcuts()
        # Open the main window maximized
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

        # update global api file path
        API_FILE = self.api_path_var.get()

        self.log_file = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=280)
        self.grid_columnconfigure(1, weight=1)
        # gear button in the top-right corner opens the settings window
        settings_btn = ctk.CTkButton(
            self,
            text="⚙",
            width=40,
            height=32,
            command=self.open_settings,
        )
        settings_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)

        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left.grid_columnconfigure(1, weight=1)
        left.grid_rowconfigure(7, weight=1)

        ctk.CTkLabel(left, text="Язык").grid(row=0, column=0, padx=5, pady=(5,0), sticky="w")
        ctk.CTkEntry(left, textvariable=self.language_var).grid(row=0, column=1, padx=5, pady=(5,0), sticky="ew")

        ctk.CTkLabel(left, text="Валюта").grid(row=1, column=0, padx=5, pady=(5,0), sticky="w")
        ctk.CTkEntry(left, textvariable=self.currency_var).grid(row=1, column=1, padx=5, pady=(5,0), sticky="ew")

        ctk.CTkLabel(left, text="Слов в заголовке H1").grid(row=2, column=0, padx=5, pady=(5,0), sticky="w")
        ctk.CTkEntry(left, textvariable=self.h1_var).grid(row=2, column=1, padx=5, pady=(5,0), sticky="ew")

        ctk.CTkLabel(left, text="Слов в заголовке H2").grid(row=3, column=0, padx=5, pady=(5,0), sticky="w")
        ctk.CTkEntry(left, textvariable=self.h2_var).grid(row=3, column=1, padx=5, pady=(5,0), sticky="ew")

        ctk.CTkLabel(left, text="Слов в заголовке H3").grid(row=4, column=0, padx=5, pady=(5,0), sticky="w")
        ctk.CTkEntry(left, textvariable=self.h3_var).grid(row=4, column=1, padx=5, pady=(5,0), sticky="ew")

        ctk.CTkButton(left, text="Настройки", command=self.open_settings).grid(row=5, column=0, columnspan=2, pady=(5,0))
        ctk.CTkButton(left, text="Старт", command=self.start).grid(row=6, column=0, columnspan=2, pady=10)

        self.log_box = ctk.CTkTextbox(left)
        self.log_box.grid(row=7, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.log_box.configure(state="disabled")

        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.plan_text = ctk.CTkTextbox(right)
        self.plan_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # underlying tkinter.Text widget for tag operations
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
        """Bind Ctrl shortcuts so they work on any keyboard layout."""

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
                if kc == 67:  # C
                    event.widget.event_generate('<<Copy>>')
                    return 'break'
                if kc == 88:  # X
                    event.widget.event_generate('<<Cut>>')
                    return 'break'
                if kc == 86:  # V
                    event.widget.event_generate('<<Paste>>')
                    return 'break'
                if kc == 65:  # A
                    event.widget.event_generate('<<SelectAll>>')
                    return 'break'
                if kc == 90:  # Z
                    event.widget.event_generate('<<Undo>>')
                    return 'break'
                if kc == 89:  # Y
                    event.widget.event_generate('<<Redo>>')
                    return 'break'

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
        """Insert placeholders for blank lines without disturbing the cursor."""
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
                widget.delete(f"{line-1}.0", f"{line-1}.end+1c")
                self.update_placeholders()
                return "break"

            if event.keysym == "Delete" and index == widget.index(f"{line}.end") and line_is_placeholder(line + 1):
                widget.delete(f"{line+1}.0", f"{line+1}.end+1c")
                self.update_placeholders()
                return "break"

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
        win.geometry("480x220")  # make dialog large enough to show all fields
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()
        self.center_window(win)

        win.grid_columnconfigure(1, weight=1)

        plan_var = ctk.StringVar(value=self.plan_path_var.get())
        api_var = ctk.StringVar(value=self.api_path_var.get())
        out_var = ctk.StringVar(value=self.output_dir_var.get())

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

        ctk.CTkLabel(win, text="H-plan путь").grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkEntry(win, textvariable=plan_var, width=250).grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkButton(win, text="...", command=choose_plan, width=30).grid(row=0, column=2, padx=5)

        ctk.CTkLabel(win, text="API путь").grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkEntry(win, textvariable=api_var, width=250).grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkButton(win, text="...", command=choose_api, width=30).grid(row=1, column=2, padx=5)

        ctk.CTkLabel(win, text="Папка HTML").grid(row=2, column=0, padx=5, pady=5)
        ctk.CTkEntry(win, textvariable=out_var, width=250).grid(row=2, column=1, padx=5, pady=5)
        ctk.CTkButton(win, text="...", command=choose_out, width=30).grid(row=2, column=2, padx=5)

        def save_and_close():
            self.plan_path_var.set(plan_var.get())
            self.api_path_var.set(api_var.get())
            self.output_dir_var.set(out_var.get())
            global API_FILE
            API_FILE = self.api_path_var.get()
            self.load_plan_file()
            self.save_settings()
            win.destroy()

        ctk.CTkButton(win, text="Сохранить", command=save_and_close).grid(row=3, column=0, columnspan=3, pady=10)

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
        save_settings(cfg)

    def on_close(self):
        self.save_settings()
        self.destroy()

    def _generate_with_key(self, key, parts, folder):
        """Attempt to generate the full article using the provided API key.

        Returns the output filename on success or ``None`` on failure.  Any
        ``InvalidAPIKeyError`` is propagated to the caller so the key can be
        marked as bad immediately.
        """
        total = len(parts)
        messages = []
        initial = generate_initial_message(
            total,
            self.language_var.get(),
            self.currency_var.get(),
            self.h1_var.get(),
            self.h2_var.get(),
            self.h3_var.get(),
        )
        messages.append({"role": "user", "content": initial})
        try:
            reply = call_api(messages, key, self.log_file)
            messages.append({"role": "assistant", "content": reply})
        except InvalidAPIKeyError:
            raise
        except Exception as e:
            self.log(f"Ошибка при инициализации: {e}")
            logger.error("Initialization error: %s", e)
            return None

        html_content = ""
        for idx, part in enumerate(parts, start=1):
            user_msg = generate_part_message(
                idx,
                total,
                part,
                self.h1_var.get(),
                self.h2_var.get(),
                self.h3_var.get(),
            )
            messages.append({"role": "user", "content": user_msg})
            try:
                result = call_api(messages, key, self.log_file)
                messages.append({"role": "assistant", "content": result})
                html_content += result + "\n"
                self.log(f"Часть {idx} сгенерирована")
                logger.info("Part %d generated", idx)
            except InvalidAPIKeyError:
                raise
            except Exception as e:
                self.log(f"Ошибка при генерации части {idx}: {e}")
                logger.error("Error generating part %d: %s", idx, e)
                return None

        first_header = ''
        for line in parts[0].splitlines():
            if line.startswith("H1:"):
                first_header = line[3:].strip()
                break
        if not first_header:
            first_header = "result"
        cleaned = normalize_html(html_content)
        filename = write_html(folder, first_header, cleaned)
        logger.info("File saved: %s", filename)
        return filename

    def run_generation(self):
        # create unique log file per run
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
        # ensure plan on disk is up to date
        self.save_plan_file()

        folder = self.output_dir_var.get().strip()
        if not folder:
            folder = os.path.dirname(plan_file) or "."
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
                self.log(f"Ключ не работает: {b}")
                logger.info("Bad key %s moved to %s", b[:8], BAD_API_FILE)

            if not key:
                continue

            self.log(f"Используется ключ: {key[:8]}...")
            logger.info("Using API key %s", key[:8])

            success = False
            for attempt in range(3):
                try:
                    filename = self._generate_with_key(key, parts, folder)
                except InvalidAPIKeyError:
                    self.log(f"Ключ недействителен: {key}")
                    logger.warning("API key %s is invalid during generation", key[:8])
                    filename = None
                    attempt = 2  # force marking key as bad
                if filename:
                    self.after(0, lambda fname=filename: messagebox.showinfo("Готово", f"Файл сохранен: {fname}"))
                    self._running = False
                    return
                else:
                    if attempt < 2:
                        self.log("Ошибка при генерации, перезапуск...")
                        logger.info("Retrying generation (attempt %d) with same key", attempt + 2)
                    time.sleep(1)

            update_bad_api_key(key, API_FILE)
            if key in keys:
                keys.remove(key)
            self.log(f"Ключ не работает: {key}")
            logger.info("Bad key %s moved to %s after retries", key[:8], BAD_API_FILE)


def main():
    logger.info("Application started")
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()