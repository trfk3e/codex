import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog  # ttk может понадобиться для таблицы, но постараемся обойтись CTk
import requests

try:
    from customtkinter.windows import ctk_toplevel
    _orig_iconbitmap = ctk_toplevel.CTkToplevel.iconbitmap

    def _iconbitmap_safe(self, *args, **kwargs):
        success = False
        try:
            _orig_iconbitmap(self, *args, **kwargs)
        except Exception as exc:
            if hasattr(self, "log_message"):
                self.log_message(f"Не удалось установить иконку окна: {exc}", "DEBUG")

    ctk_toplevel.CTkToplevel.iconbitmap = _iconbitmap_safe
except Exception:
    pass
import configparser
import threading
import os
import time
import re
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
from queue import Queue, Empty

import random
import traceback
import json  # НОВОВВЕДЕНИЕ: Для работы с файлом статусов API ключей
import datetime  # НОВОВВЕДЕНИЕ: Для работы со временем сброса лимитов
from dateutil.parser import parse as parse_datetime  # Для парсинга ISO дат, если понадобится при чтении
from dateutil.relativedelta import relativedelta  # Для парсинга "1m", "60s"
# Utilities for HTML parsing and shutdown handling
from bs4 import BeautifulSoup, NavigableString
import sys
import atexit
import signal
# from multiprocessing import Process, freeze_support  # Multiprocessing no longer used
from collections import deque, defaultdict
import itertools

if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

def app_path(name: str) -> str:
    return os.path.join(APP_DIR, name)

DEFAULT_CONFIG_FILE = "settings.ini"
DEFAULT_MODEL = "gpt-4o-mini"
MAX_FILENAME_LENGTH = 100
MAX_RETRY_PASSES = 3
# Количество попыток генерации для одного ключевого слова
# Увеличено до 10, чтобы снизить вероятность недостающих файлов
# Limit of generation attempts per keyword. If None, retries continue
# until the required number of files is produced.
MAX_KEYWORD_ATTEMPTS = None
# Максимальное количество потоков
MAX_THREADS = 200
# Limit how many projects can run generation simultaneously
MAX_CONCURRENT_PROJECTS = 3
# Global limit of concurrently running worker threads across all tabs
GLOBAL_THREAD_SEMAPHORE = threading.Semaphore(MAX_THREADS)
GLOBAL_PROJECT_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_PROJECTS)
GEMINI_KEY_MAX_REQUESTS = 10  # allow up to 10 calls per key
GEMINI_KEY_SLOT_SECONDS = 65  # slots replenish after 65 seconds
BAD_API_KEYS_FILE = "BAD_API.txt"
# Limit the amount of lines kept in the GUI log to avoid slowdown
MAX_LOG_LINES = 500
# Interval between log textbox updates in milliseconds
LOG_FLUSH_INTERVAL_MS = 100
# Max number of log messages processed per flush to avoid UI freezes
LOG_FLUSH_BATCH_SIZE = 50
# Maximum queued log messages before newer ones are dropped
MAX_LOG_QUEUE_SIZE = 5000
# Минимальный и максимальный допустимые размеры сгенерированных файлов
MIN_ARTICLE_SIZE = 7 * 1024
MAX_ARTICLE_SIZE = 45 * 1024
# Максимальная допустимая длина заголовка в оглавлении
MAX_TOC_ITEM_CHARS = 200
# Максимальная суммарная длина всех заголовков в оглавлении
MAX_TOC_TOTAL_CHARS = 2000
# Минимальный интервал между запросами одним и тем же API ключом (секунды)
PER_KEY_CALL_INTERVAL = 0.05
# НОВОВВЕДЕНИЕ: Имя файла для статусов API ключей и мьютекс для доступа к нему
API_KEY_STATUSES_FILE = "api_key_statuses.json"  # НОВОВВЕДЕНИЕ

# Поддерживаемые провайдеры
PROVIDER_OPENAI = "OpenAI"
PROVIDER_GEMINI = "Gemini 2.5 Flash"

# Файлы для провайдера Gemini
GEMINI_USAGE_FILE = "gemini_key_usage.json"
EXHAUSTED_GEMINI_KEYS_FILE = "exhausted-limit-keys.txt"
GEMINI_USAGE_LOCK = threading.RLock()

# Файлы для совместного использования API ключей и списка проектов
SHARED_KEYS_FILE = "shared_keys.txt"
PROJECTS_FILE = "projects.txt"

# Глобальный кэш плохих API ключей и мьютекс для него
BAD_API_KEYS_CACHE = None
BAD_API_KEYS_CACHE_LOCK = threading.Lock()
# Храним время последнего запроса к API для каждого ключа
api_key_last_call_time = {}
api_key_last_call_time_lock = threading.Lock()

# Global API key statuses shared across all project windows
GLOBAL_API_KEY_STATUSES = None
GLOBAL_API_KEY_STATUSES_LOCK = threading.RLock()
GLOBAL_API_KEY_STATUSES_LOADED = False

# Telegram integration settings file
TELEGRAM_SETTINGS_FILE = "telegram_settings.json"
AUTH_FILE = "auth.lock"

def _get_system_fingerprint() -> str:
    """Return a hash representing the current machine."""
    import hashlib
    import platform
    import uuid

    data = [
        platform.system(),
        platform.release(),
        platform.version(),
        platform.machine(),
        platform.processor(),
        str(uuid.getnode()),
    ]
    joined = "||".join(data)
    return hashlib.sha256(joined.encode("utf-8", "ignore")).hexdigest()

def check_first_run_password():
    auth_path = app_path(AUTH_FILE)
    current_fp = _get_system_fingerprint()
    stored_fp = None
    if os.path.exists(auth_path):
        try:
            with open(auth_path, "r", encoding="utf-8") as f:
                stored_fp = f.read().strip()
        except Exception:
            stored_fp = None
    if stored_fp == current_fp:
        return
    root = tk.Tk()
    root.withdraw()
    pwd = simpledialog.askstring("Первый запуск", "Введите пароль", show="*")
    if pwd != "locosd228":
        messagebox.showerror("Ошибка", "Неверный пароль. Программа завершит работу.")
        root.destroy()
        sys.exit(1)
    try:
        with open(auth_path, "w", encoding="utf-8") as f:
            f.write(current_fp)
    except Exception:
        pass
    finally:
        root.destroy()


HELP_TEXT = (
    "Программа генерирует статьи с использованием ChatGPT.\n\n"

    "1. В поле API ключей внесите ваши ключи OpenAI, каждый с новой строкой.\n\n"

    "2. Укажите папку для сохранения файлов.\n\n"

    "3. Загрузите файл ключевых слов в формате 'фраза\tколичество'.\n"
    "   Дополнительные строки могут задавать параметры:\n"
    "   - '!==+ ссылка' — URL, куда будет вести ссылка.\n"
    "   - '!===+ язык' — язык создаваемых текстов.\n"
    "   - '!====+ тема' — например '!====+ Краш игра'. Добавляется перед каждой\n"
    "     ключевой фразой, если ChatGPT не знает игру.\n\n"

    "4. Если ключевая фраза отсутствует в ответе, она вставляется случайно\n"
    "   в один из абзацев до конца текста.\n\n"

    "5. Слайдер \"Количество потоков\" задаёт число одновременно выполняемых\n"
    "   задач. Каждый ключ может использоваться максимум в 10 потоках,\n"
    "   поэтому предел равен числу активных ключей × 10 (но не больше 200).\n\n"

    "6. 'Статусы API Ключей' открывает таблицу со столбцами:\n"
    "   'Ключ (хвост)' — последние символы ключа;\n"
    "   'Статус' — active, cooldown или bad;\n"
    "   'Запросов (Ост/Лимит)' и 'Сброс Запросов' — оставшиеся запросы и время их сброса;\n"
    "   'Токенов (Ост/Лимит)' и 'Сброс Токенов' — аналогичные данные по токенам;\n"
    "   'Обновлено' — время последней проверки;\n"
    "   'Ошибка' — последнее сообщение об ошибке.\n\n"

    "7. Кнопка 'Остановить' мгновенно очищает очередь задач и прекращает\n"
    "   дальнейшую генерацию.\n\n"

    "8. Нажмите 'Начать генерацию' и следите за логом снизу.")


def load_bad_api_keys(log_func=None):
    """Load bad API keys once and return the cached set."""
    global BAD_API_KEYS_CACHE
    with BAD_API_KEYS_CACHE_LOCK:
        if BAD_API_KEYS_CACHE is not None:
            return BAD_API_KEYS_CACHE
        loaded = set()
        bad_file = app_path(BAD_API_KEYS_FILE)
        if os.path.exists(bad_file):
            try:
                with open(bad_file, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped:
                            loaded.add(stripped)
                if loaded and log_func:
                    log_func(
                        f"Загружено {len(loaded)} плохих API ключей из {BAD_API_KEYS_FILE}.",
                        "INFO",
                    )
            except Exception as e:
                if log_func:
                    log_func(
                        f"Ошибка загрузки файла плохих ключей ({BAD_API_KEYS_FILE}): {e}",
                        "WARNING",
                    )
        BAD_API_KEYS_CACHE = loaded
        return BAD_API_KEYS_CACHE


def load_shared_keys():
    file_path = app_path(SHARED_KEYS_FILE)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return []


def save_shared_keys(keys):
    file_path = app_path(SHARED_KEYS_FILE)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(keys))
    except Exception as e:
        print(f"Ошибка сохранения общих ключей: {e}")


def load_project_folders():
    file_path = app_path(PROJECTS_FILE)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return []


def save_project_folders(folders):
    file_path = app_path(PROJECTS_FILE)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(folders))
    except Exception as e:
        print(f"Ошибка сохранения списка проектов: {e}")


def load_telegram_settings(log_func=None):
    """Load Telegram bot settings from TELEGRAM_SETTINGS_FILE."""
    default_data = {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}
    file_path = app_path(TELEGRAM_SETTINGS_FILE)
    if not os.path.exists(file_path):
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(default_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if log_func:
                log_func(f"Не удалось создать {TELEGRAM_SETTINGS_FILE}: {e}", "WARNING")
        return None, None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        token = str(data.get("TELEGRAM_BOT_TOKEN", "")).strip()
        chat_id = str(data.get("TELEGRAM_CHAT_ID", "")).strip()
        if token and chat_id:
            return token, chat_id
    except Exception as e:
        if log_func:
            log_func(f"Ошибка чтения {TELEGRAM_SETTINGS_FILE}: {e}", "WARNING")
    return None, None


def send_telegram_message(token, chat_id, text, log_func=None):
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text}, timeout=10,
        )
    except Exception as e:
        if log_func:
            log_func(f"Ошибка отправки сообщения в Telegram: {e}", "WARNING")


# --- Global Telegram credentials and shutdown notification ---
# These are loaded once so we can notify about abnormal exits.
GLOBAL_TELEGRAM_TOKEN, GLOBAL_TELEGRAM_CHAT_ID = load_telegram_settings()
PROGRAM_EXITED_VIA_UI = False


def _notify_program_closed():
    if not PROGRAM_EXITED_VIA_UI:
        send_telegram_message(
            GLOBAL_TELEGRAM_TOKEN,
            GLOBAL_TELEGRAM_CHAT_ID,
            "Программа закрыта",
        )


atexit.register(_notify_program_closed)


def _signal_handler(signum, frame):
    sys.exit(0)


for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, _signal_handler)
    except Exception:
        pass


class TextGeneratorApp(ctk.CTkFrame):
    def __init__(self, master, config_file: str = DEFAULT_CONFIG_FILE):
        super().__init__(master)
        self.master = master
        self.config_file = config_file
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.api_keys_list = []
        self.output_folder = tk.StringVar()
        self.keywords_file_path = tk.StringVar()
        self.num_threads_var = tk.IntVar(value=5)
        self.generation_active = False
        self.waiting_for_project_slot = False
        self.stop_event = threading.Event()
        self.silent_stop = False
        self.project_slot_acquired = False

        # Share API key statuses across all tabs to reduce I/O
        global GLOBAL_API_KEY_STATUSES
        self.api_key_statuses_lock = GLOBAL_API_KEY_STATUSES_LOCK
        if GLOBAL_API_KEY_STATUSES is None:
            GLOBAL_API_KEY_STATUSES = {}
        self.api_key_statuses = GLOBAL_API_KEY_STATUSES
        self.api_key_queue = Queue()

        self.log_queue = Queue()
        self.log_overflow_notified = False

        self.task_creation_queue = Queue()
        self.generation_language_var = tk.StringVar(value="Русский")
        self.supported_languages = ["Русский", "English", "Українська", "Deutsch", "Français", "Español", "Português"]
        self.language_codes = {"Русский": "ru", "English": "en", "Українська": "uk", "Deutsch": "de", "Français": "fr",
                               "Español": "es", "Português": "pt"}
        self.target_link_var = tk.StringVar()
        self.topic_word = ""
        self.output_format_var = tk.StringVar(value="TXT")

        self.provider_var = tk.StringVar(value=PROVIDER_OPENAI)
        self.openai_keys_file_var = tk.StringVar()
        self.gemini_keys_file_var = tk.StringVar()
        self.proxy_file_var = tk.StringVar()
        self.api_key_proxy_map = {}
        self.proxy_countries = {}

        self.gemini_usage = {}
        self._load_gemini_usage()
        self.gemini_key_locks = defaultdict(threading.Lock)
        # Квота 10 запросов на ключ с восстановлением слота через 65 секунд
        self.gemini_key_quota = defaultdict(
            lambda: threading.BoundedSemaphore(GEMINI_KEY_MAX_REQUESTS)
        )

        self.article_toc_background_colors = [
            "#f9f9f9", "#fcf3f2", "#fcfcfe", "#fff5f3", "#f5f8ff", "#f8fcf3",
            "#f6f7f7", "#fafcf5", "#fdfbfb", "#f9f9f2", "#fcfbf6", "#f4f8f8",
            "#f6f5fa", "#f7fef4", "#fbf8fa", "#f5f5f6", "#f9f7fc", "#fef8f8",
            "#f7f7f9", "#f8f4f4", "#f6fcfb", "#f7f6f6", "#fef7f4", "#f5f7fa",
            "#fcf5fc", "#fbf6f4", "#f3f3f3", "#fafafb", "#f6f8f5", "#f8f7fa",
            "#f7f3f3", "#f9fbfa", "#f7fcfd", "#fdfdfb", "#f4f9f9", "#fefcfc",
            "#fcfbfa", "#f3f8f6", "#f4f5f9", "#fefdf7", "#fbfcf9", "#f7f5f7",
            "#f9f4f6", "#f3f9f3", "#fdfdf9", "#f4f7f8", "#fcf7f8", "#f5f3fa",
            "#f9fef7", "#f8fcfa", "#f3f5f3", "#f3f6f9", "#f8f8f7", "#fdf6f6",
            "#fcfafb", "#fbf7fb", "#f5f8f6", "#f7fbfb", "#f9f9fb", "#fdfbfa",
            "#f5f9fa", "#f8f3f6", "#f3f4f9", "#fcf3f9", "#f9f9f7", "#fcfdfb",
            "#f5f6f6", "#f7f4fa", "#fcf8f8", "#f4f4f4", "#fbfbfc", "#fcf6f6",
            "#f5fbf9", "#f9f3f5", "#fdfcfd", "#f3f6f6", "#f5f6f9", "#f4f8f6",
            "#f9f7f5", "#fcfcfb", "#f7f8f4", "#fbfbf9", "#f5f7f7", "#f3f9fa",
            "#fcf7fc", "#fdfdfc", "#fcfcf6", "#f3f6f8", "#fcfbfc", "#f4f3f9",
            "#f7f7f4", "#f5f9f3", "#f7f8f8", "#f3f3f6", "#f8f6fa", "#f7f9f9",
            "#f5f5f3", "#f9f5fc", "#f7f7f7", "#f6f3f3", "#fbfbf8", "#fcf4f4"
        ]

        self.all_task_definitions = []
        self.successful_task_ids = set()
        self.success_lock = threading.Lock()
        self.api_key_management_lock = threading.RLock()
        self.api_key_usage_stats = {}
        self.api_stats_lock = threading.Lock()
        self.completed_task_ids = set()
        self.previous_h1_text = None
        self.previous_h1_lock = threading.Lock()
        self.progress_data = {}
        self.expected_total_tasks = 0
        self.used_filepaths = set()
        self.used_filepaths_lock = threading.Lock()
        self.output_file_counter = 0
        self.output_count_lock = threading.Lock()
        self.last_output_scan_time = 0.0

        self._load_api_key_statuses()
        self.load_settings()
        self.load_progress_data()
        self._initial_check_and_revive_keys()
        self.create_widgets()
        self.update_threads_label()

        self.telegram_bot_token = None
        self.telegram_chat_id = None
        self._load_telegram_settings()

    def _run_task_batch(self):
        """Run generation loops for current self.all_task_definitions."""
        self._initial_check_and_revive_keys()
        if self.api_key_queue.empty():
            self.log_message("Нет доступных API ключей для начала генерации. Проверьте статусы.", "ERROR")
            return
        for pass_num in range(MAX_RETRY_PASSES):
            if self.stop_event.is_set():
                self.log_message("Процесс генерации остановлен до начала нового прохода.", "INFO")
                break
            if pass_num > 0:
                self.log_message(f"Проход {pass_num + 1}: Проверка и оживление cooldown ключей...", "DEBUG")
                self._initial_check_and_revive_keys()
                if self.api_key_queue.empty():
                    self.log_message(f"Проход {pass_num + 1}: Нет активных ключей. Пропускаем.", "WARNING")
                    if not self.stop_event.is_set():
                        time.sleep(1)
                    continue

            tasks_for_this_pass = [d for d in self.all_task_definitions if d["id"] not in self.successful_task_ids]
            if not tasks_for_this_pass:
                self.log_message("Все запланированные статьи успешно сгенерированы.", "INFO")
                break
            self.log_message(f"--- Проход генерации {pass_num + 1}/{MAX_RETRY_PASSES} ---")
            self.log_message(f"Задач к выполнению на этом проходе: {len(tasks_for_this_pass)}")

            while not self.task_creation_queue.empty():
                try:
                    self.task_creation_queue.get_nowait()
                    self.task_creation_queue.task_done()
                except Empty:
                    break
                except Exception as e_q_clear:
                    self.log_message(f"Ошибка при очистке очереди задач: {e_q_clear}", "DEBUG")

            current_pass_task_idx_counter = 0
            for task_def in tasks_for_this_pass:
                if self.stop_event.is_set():
                    break
                current_pass_task_idx_counter += 1
                self.task_creation_queue.put((task_def["id"], task_def["keyword"], task_def["num_for_kw"],
                                              task_def["total_for_kw"], current_pass_task_idx_counter,
                                              len(tasks_for_this_pass)))
            if self.stop_event.is_set():
                self.log_message("Генерация остановлена во время заполнения очереди задач.", "INFO")
                break

            threads = []
            num_available_keys_for_threads = self.api_key_queue.qsize()
            num_active_threads_to_start = min(self.num_threads_var.get(), num_available_keys_for_threads,
                                              len(tasks_for_this_pass))
            if num_active_threads_to_start < self.num_threads_var.get() and num_active_threads_to_start > 0:
                self.log_message(
                    f"Количество потоков уменьшено до {num_active_threads_to_start} (ограничено ключами/задачами).",
                    "INFO")
            if num_active_threads_to_start == 0 and len(tasks_for_this_pass) > 0:
                self.log_message(f"Нет доступных API ключей или задач для запуска потоков на проходе {pass_num + 1}.",
                                 "WARNING")
                if not self.stop_event.is_set():
                    time.sleep(1)
                continue
            for i in range(num_active_threads_to_start):
                if self.stop_event.is_set():
                    break
                acquired = GLOBAL_THREAD_SEMAPHORE.acquire(blocking=False)
                if not acquired:
                    self.log_message(
                        "Достигнут глобальный предел потоков. Ожидание освобождения слотов...",
                        "INFO",
                    )
                    while not acquired and not self.stop_event.is_set():
                        acquired = GLOBAL_THREAD_SEMAPHORE.acquire(timeout=1)
                    if not acquired:
                        break
                t = threading.Thread(target=self.worker_thread, name=f"Worker-{i + 1}", daemon=True)
                threads.append(t)
                t.start()
                time.sleep(0.01)
            if self.stop_event.is_set():
                self.log_message("Генерация остановлена во время запуска потоков.", "INFO")
            self._monitor_task_queue_and_threads(threads)
            if self.stop_event.is_set():
                self.log_message(f"Проход генерации {pass_num + 1} прерван.", "INFO")
                break
        else:
            remaining_tasks_count = len([d for d in self.all_task_definitions if d['id'] not in self.successful_task_ids])
            if remaining_tasks_count == 0:
                self.log_message("Все задачи успешно выполнены после всех проходов.", "INFO")
            else:
                self.log_message(
                    f"После {MAX_RETRY_PASSES} проходов не удалось сгенерировать {remaining_tasks_count} из {len(self.all_task_definitions)} статей.",
                    "WARNING")

    def log_message(self, message, level="INFO"):
        if level == "DEBUG":
            return
        if getattr(self, 'silent_stop', False) and self.stop_event.is_set():
            allowed = [
                "Генерация остановлена",
                "Генерация завершена",
                "Итоговое количество",
                "Итого для",
                "Файлов для",
            ]
            if not any(sub in message for sub in allowed):
                return
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] [{level}] {message}\n"
        print(formatted_message.strip())
        if hasattr(self, 'log_queue'):
            if self.log_queue.qsize() < MAX_LOG_QUEUE_SIZE:
                self.log_queue.put(formatted_message)
                self.log_overflow_notified = False
            elif not self.log_overflow_notified:
                self.log_overflow_notified = True
                overflow_msg = f"[{timestamp}] [WARNING] Очередь логов переполнена, часть сообщений будет потеряна\n"
                try:
                    self.log_queue.put_nowait(overflow_msg)
                except Exception:
                    pass

    def _update_log_textbox(self, formatted_message):
        if hasattr(self, 'log_textbox') and self.log_textbox.winfo_exists():
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert(tk.END, formatted_message)
            try:
                total_lines = int(float(self.log_textbox.index('end-1c').split('.')[0]))
                if total_lines > MAX_LOG_LINES:
                    self.log_textbox.delete('1.0', f'{total_lines - MAX_LOG_LINES + 1}.0')
            except Exception:
                pass
            self.log_textbox.configure(state="disabled")
            self.log_textbox.see(tk.END)

    def flush_log_queue(self):
        if hasattr(self, 'log_queue') and not self.log_queue.empty():
            batch = []
            for _ in range(LOG_FLUSH_BATCH_SIZE):
                try:
                    batch.append(self.log_queue.get_nowait())
                except Empty:
                    break
            if batch:
                self._update_log_textbox(''.join(batch))
        self.after(LOG_FLUSH_INTERVAL_MS, self.flush_log_queue)

    def _load_api_key_statuses(self):
        global GLOBAL_API_KEY_STATUSES_LOADED, GLOBAL_API_KEY_STATUSES
        with self.api_key_statuses_lock:
            if not GLOBAL_API_KEY_STATUSES_LOADED:
                status_file = app_path(API_KEY_STATUSES_FILE)
                if os.path.exists(status_file):
                    try:
                        with open(status_file, "r", encoding="utf-8") as f:
                            loaded_statuses = json.load(f)
                            for key, status_data in loaded_statuses.items():
                                if "reset_requests_at" in status_data and status_data["reset_requests_at"]:
                                    status_data["reset_requests_at"] = parse_datetime(status_data["reset_requests_at"])
                                if "reset_tokens_at" in status_data and status_data["reset_tokens_at"]:
                                    status_data["reset_tokens_at"] = parse_datetime(status_data["reset_tokens_at"])
                                if "last_updated" in status_data and status_data["last_updated"]:
                                    status_data["last_updated"] = parse_datetime(status_data["last_updated"])
                            GLOBAL_API_KEY_STATUSES.update(loaded_statuses)
                        GLOBAL_API_KEY_STATUSES_LOADED = True
                        self.log_message(f"Статусы API ключей загружены из {API_KEY_STATUSES_FILE}.")
                    except Exception as e:
                        self.log_message(
                            f"Ошибка загрузки статусов API ключей ({API_KEY_STATUSES_FILE}): {e}", "ERROR")
                        GLOBAL_API_KEY_STATUSES.clear()
                        GLOBAL_API_KEY_STATUSES_LOADED = True
                else:
                    self.log_message(
                        f"Файл статусов API ключей ({API_KEY_STATUSES_FILE}) не найден. Будет создан новый.", "INFO")
                    GLOBAL_API_KEY_STATUSES_LOADED = True

    def _save_api_key_statuses(self):
        with self.api_key_statuses_lock:
            try:
                savable_statuses = {}
                for key, status_data in GLOBAL_API_KEY_STATUSES.items():
                    s_data = status_data.copy()
                    if "reset_requests_at" in s_data and isinstance(s_data["reset_requests_at"], datetime.datetime):
                        s_data["reset_requests_at"] = s_data["reset_requests_at"].isoformat()
                    if "reset_tokens_at" in s_data and isinstance(s_data["reset_tokens_at"], datetime.datetime):
                        s_data["reset_tokens_at"] = s_data["reset_tokens_at"].isoformat()
                    if "last_updated" in s_data and isinstance(s_data["last_updated"], datetime.datetime):
                        s_data["last_updated"] = s_data["last_updated"].isoformat()
                    savable_statuses[key] = s_data
                status_file = app_path(API_KEY_STATUSES_FILE)
                with open(status_file, "w", encoding="utf-8") as f:
                    json.dump(savable_statuses, f, indent=4, ensure_ascii=False)
                self.log_message(f"Статусы API ключей сохранены в {API_KEY_STATUSES_FILE}.", "DEBUG")
            except Exception as e:
                self.log_message(f"Ошибка сохранения статусов API ключей в {API_KEY_STATUSES_FILE}: {e}", "ERROR")

    def _load_gemini_usage(self):
        path = app_path(GEMINI_USAGE_FILE)
        today = datetime.date.today().isoformat()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        with GEMINI_USAGE_LOCK:
            for key, info in data.items():
                if info.get("date") != today:
                    info = {"date": today, "used_today": 0}
                else:
                    info = {"date": today, "used_today": info.get("used_today", 0)}
                self.gemini_usage[key] = info

    def _save_gemini_usage(self):
        with GEMINI_USAGE_LOCK:
            path = app_path(GEMINI_USAGE_FILE)
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.gemini_usage, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    def _mark_gemini_key_exhausted(self, api_key):
        with GEMINI_USAGE_LOCK:
            info = self.gemini_usage.get(api_key)
            if info:
                info["used_today"] = 250
        with open(app_path(EXHAUSTED_GEMINI_KEYS_FILE), "a", encoding="utf-8") as f:
            f.write(api_key + "\n")
        self._remove_key_from_queue(api_key)
        self.log_message(f"Ключ {api_key[:7]}... исчерпан по лимиту сегодня.", "INFO")
        self._save_gemini_usage()

    def _remove_key_from_queue(self, api_key):
        with self.api_key_queue.mutex:
            self.api_key_queue.queue = deque([k for k in self.api_key_queue.queue if k != api_key])

    def _before_gemini_call(self, api_key):
        today = datetime.date.today().isoformat()
        with GEMINI_USAGE_LOCK:
            info = self.gemini_usage.setdefault(api_key, {"date": today, "used_today": 0})
            if info.get("date") != today:
                info.update({"date": today, "used_today": 0})
            if info["used_today"] >= 250:
                self.log_message(
                    f"Ключ {api_key[:7]} исчерпал дневной лимит Gemini",
                    "INFO",
                )
                return False
            info["used_today"] += 1
            self._save_gemini_usage()
            self.log_message(
                f"Предварительная проверка Gemini пройдена для ключа {api_key[:7]}",
                "DEBUG",
            )
            return True

    def _after_gemini_call(self, api_key):
        with GEMINI_USAGE_LOCK:
            info = self.gemini_usage.setdefault(
                api_key,
                {"date": datetime.date.today().isoformat(), "used_today": 0},
            )
            if info["used_today"] >= 250:
                self._mark_gemini_key_exhausted(api_key)
            else:
                self._save_gemini_usage()

    def load_progress_data(self):
        """Initialize in-memory progress tracking (file persistence removed)."""
        self.progress_data = {}
        self.completed_task_ids = set()

    def save_progress_data(self):
        """No-op since progress persistence is removed."""
        return

    def update_progress(self, task_id, keyword, num_for_kw):
        self.completed_task_ids.add(task_id)
        if keyword not in self.progress_data:
            self.progress_data[keyword] = []
        if num_for_kw not in self.progress_data[keyword]:
            self.progress_data[keyword].append(num_for_kw)
            self.progress_data[keyword] = sorted(self.progress_data[keyword])
        # Progress persistence removed

    def _get_default_api_key_status(self):
        return {
            "status": "active",
            "limit_requests": None, "remaining_requests": None, "reset_requests_at": None,
            "limit_tokens": None, "remaining_tokens": None, "reset_tokens_at": None,
            "last_updated": None, "error_message": None
        }

    def _initial_check_and_revive_keys(self):
        with self.api_key_management_lock:
            keys_to_check = self.api_keys_list[:]
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        refreshed_any_key = False
        with self.api_key_statuses_lock:
            for key_str in keys_to_check:
                status_entry = self.api_key_statuses.get(key_str, self._get_default_api_key_status())
                changed = False
                if status_entry["status"] in ["cooldown_requests", "cooldown_both"]:
                    if status_entry["reset_requests_at"] and now_utc >= status_entry["reset_requests_at"]:
                        status_entry["reset_requests_at"] = None
                        status_entry["remaining_requests"] = status_entry["limit_requests"]
                        if status_entry["status"] == "cooldown_requests":
                            status_entry["status"] = "active"
                        elif status_entry["status"] == "cooldown_both" and (
                                not status_entry["reset_tokens_at"] or now_utc >= status_entry["reset_tokens_at"]):
                            status_entry["status"] = "active"
                        changed = True
                        self.log_message(f"Ключ {key_str[:7]}... вышел из cooldown по запросам.", "INFO")
                if status_entry["status"] in ["cooldown_tokens", "cooldown_both"]:
                    if status_entry["reset_tokens_at"] and now_utc >= status_entry["reset_tokens_at"]:
                        status_entry["reset_tokens_at"] = None
                        status_entry["remaining_tokens"] = status_entry["limit_tokens"]
                        if status_entry["status"] == "cooldown_tokens":
                            status_entry["status"] = "active"
                        elif status_entry["status"] == "cooldown_both" and (
                                not status_entry["reset_requests_at"] or now_utc >= status_entry["reset_requests_at"]):
                            status_entry["status"] = "active"
                        changed = True
                        self.log_message(f"Ключ {key_str[:7]}... вышел из cooldown по токенам.", "INFO")
                if changed:
                    refreshed_any_key = True
                    status_entry["last_updated"] = now_utc
                self.api_key_statuses[key_str] = status_entry
        if refreshed_any_key:
            self._repopulate_available_api_key_queue()
        self._save_api_key_statuses()

    def _current_per_key_concurrency(self):
        return GEMINI_KEY_MAX_REQUESTS

    def _repopulate_available_api_key_queue(self):
        with self.api_key_management_lock:
            current_master_keys = self.api_keys_list[:]
        new_queue = Queue()
        active_keys_for_queue = []
        with self.api_key_statuses_lock:
            for key_str in current_master_keys:
                status_data = self.api_key_statuses.get(key_str)
                if status_data and status_data.get("status") == "active":
                    active_keys_for_queue.append(key_str)
                elif not status_data:
                    self.api_key_statuses[key_str] = self._get_default_api_key_status()
                    active_keys_for_queue.append(key_str)
        per_key = self._current_per_key_concurrency()
        if active_keys_for_queue:
            random.shuffle(active_keys_for_queue)
            rr_cycle = itertools.cycle(active_keys_for_queue)
            total_entries = per_key * len(active_keys_for_queue)
            for _ in range(total_entries):
                new_queue.put(next(rr_cycle))
            self.log_message(
                f"Очередь API ключей обновлена. Активных ключей: {len(active_keys_for_queue)}",
                "INFO",
            )
        else:
            self.log_message("Нет активных API ключей для добавления в очередь.", "WARNING")
        self.api_key_queue = new_queue
        self.update_threads_label()

    def _parse_ratelimit_reset_time(self, reset_value_str):
        if not reset_value_str: return None
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        match = re.match(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)", reset_value_str.lower())
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                self.log_message(
                    f"Не удалось преобразовать значение времени '{match.group(1)}' в число для '{reset_value_str}'",
                    "WARNING")
                return now_utc + datetime.timedelta(minutes=5)
            unit = match.group(2)
            delta = None
            if unit == "ms":
                delta = datetime.timedelta(milliseconds=value)
            elif unit == "s":
                delta = datetime.timedelta(seconds=value)
            elif unit == "m":
                delta = datetime.timedelta(minutes=value)
            elif unit == "h":
                delta = datetime.timedelta(hours=value)
            elif unit == "d":
                delta = datetime.timedelta(days=value)
            if delta: return now_utc + delta
        self.log_message(f"Не удалось распознать формат времени сброса: {reset_value_str}", "WARNING")
        return now_utc + datetime.timedelta(minutes=5)

    def _update_api_key_status_from_headers(self, api_key, headers, is_error=False, status_code=None):
        if not api_key or not headers: return
        with self.api_key_statuses_lock:
            status_entry = self.api_key_statuses.get(api_key, self._get_default_api_key_status())
            changed_in_function = False
            try:
                limit_req_hdr = headers.get('x-ratelimit-limit-requests')
                remaining_req_hdr = headers.get('x-ratelimit-remaining-requests')
                reset_req_hdr = headers.get('x-ratelimit-reset-requests')
                limit_tok_hdr = headers.get('x-ratelimit-limit-tokens')
                remaining_tok_hdr = headers.get('x-ratelimit-remaining-tokens')
                reset_tok_hdr = headers.get('x-ratelimit-reset-tokens')
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                status_entry["last_updated"] = now_utc
                if limit_req_hdr: status_entry["limit_requests"] = int(limit_req_hdr)
                if remaining_req_hdr: status_entry["remaining_requests"] = int(remaining_req_hdr)
                if reset_req_hdr: status_entry["reset_requests_at"] = self._parse_ratelimit_reset_time(reset_req_hdr)
                if limit_tok_hdr: status_entry["limit_tokens"] = int(limit_tok_hdr)
                if remaining_tok_hdr: status_entry["remaining_tokens"] = int(remaining_tok_hdr)
                if reset_tok_hdr: status_entry["reset_tokens_at"] = self._parse_ratelimit_reset_time(reset_tok_hdr)

                was_active = status_entry["status"] == "active"
                if is_error and status_code == 429:
                    if remaining_req_hdr and int(remaining_req_hdr) <= 1:
                        status_entry["status"] = "cooldown_requests"
                        self.log_message(
                            f"Ключ {api_key[:7]}... переведен в cooldown (запросы, код 429). Сброс: {status_entry.get('reset_requests_at')}",
                            "WARNING")
                        changed_in_function = True
                    elif remaining_tok_hdr and int(remaining_tok_hdr) <= 100:
                        status_entry["status"] = "cooldown_tokens"
                        self.log_message(
                            f"Ключ {api_key[:7]}... переведен в cooldown (токены, код 429). Сброс: {status_entry.get('reset_tokens_at')}",
                            "WARNING")
                        changed_in_function = True
                    else:
                        if status_entry["status"] == "active":
                            status_entry["status"] = "cooldown_requests"
                            status_entry["reset_requests_at"] = now_utc + datetime.timedelta(minutes=1)
                if status_entry["status"] == "active":
                    if status_entry.get("remaining_requests") is not None and status_entry["remaining_requests"] <= 1:
                        status_entry["status"] = "cooldown_requests"
                        self.log_message(
                            f"Ключ {api_key[:7]}... переведен в cooldown (запросы, остаток <= 1). Сброс: {status_entry.get('reset_requests_at')}",
                            "WARNING")
                        changed_in_function = True
                    elif status_entry.get("remaining_tokens") is not None and status_entry["remaining_tokens"] <= 200:
                        status_entry["status"] = "cooldown_tokens"
                        self.log_message(
                            f"Ключ {api_key[:7]}... переведен в cooldown (токены, остаток <= 200). Сброс: {status_entry.get('reset_tokens_at')}",
                            "WARNING")
                        changed_in_function = True
                self.api_key_statuses[api_key] = status_entry
            except Exception as e:
                self.log_message(f"Ошибка при обновлении статуса ключа {api_key[:7]} из заголовков: {e}", "ERROR")
                status_entry["status"] = "error"
                status_entry["error_message"] = str(e)
                self.api_key_statuses[api_key] = status_entry
                changed_in_function = True
        if changed_in_function:
            self._save_api_key_statuses()
            if status_entry["status"] != "active" and was_active:
                self.log_message(f"Ключ {api_key[:7]}... более не активен. Обновление очереди...", "INFO")
                self._repopulate_available_api_key_queue()

    def create_widgets(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        provider_frame = ctk.CTkFrame(main_frame)
        provider_frame.pack(pady=(10, 5), padx=10, fill="x")
        ctk.CTkLabel(provider_frame, text="Провайдер:").pack(side="left", padx=(0, 10))
        self.provider_combobox = ctk.CTkComboBox(provider_frame,
                                                 values=[PROVIDER_OPENAI, PROVIDER_GEMINI],
                                                 variable=self.provider_var,
                                                 command=self._on_provider_change)
        self.provider_combobox.pack(side="left", fill="x", expand=True)

        self.openai_keys_frame = ctk.CTkFrame(main_frame)
        self.openai_keys_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(self.openai_keys_frame, text="Файл с ключами OpenAI (.txt):").pack(anchor="w")
        self.openai_keys_entry = ctk.CTkEntry(self.openai_keys_frame, textvariable=self.openai_keys_file_var, width=350)
        self.openai_keys_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.openai_keys_browse = ctk.CTkButton(self.openai_keys_frame, text="Выбрать...", command=self._browse_openai_keys_file)
        self.openai_keys_browse.pack(side="left")

        self.gemini_keys_frame = ctk.CTkFrame(main_frame)
        self.gemini_keys_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(self.gemini_keys_frame, text="Файл с ключами Gemini (.txt):").pack(anchor="w")
        self.gemini_keys_entry = ctk.CTkEntry(self.gemini_keys_frame, textvariable=self.gemini_keys_file_var, width=350)
        self.gemini_keys_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.gemini_keys_browse = ctk.CTkButton(self.gemini_keys_frame, text="Выбрать...", command=self._browse_gemini_keys_file)
        self.gemini_keys_browse.pack(side="left")

        self.proxy_frame = ctk.CTkFrame(main_frame)
        self.proxy_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(self.proxy_frame, text="Файл с прокси (.txt):").pack(anchor="w")
        self.proxy_entry = ctk.CTkEntry(self.proxy_frame, textvariable=self.proxy_file_var, width=350)
        self.proxy_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.proxy_browse = ctk.CTkButton(self.proxy_frame, text="Выбрать...", command=self._browse_proxy_file)
        self.proxy_browse.pack(side="left")
        folder_frame = ctk.CTkFrame(main_frame)
        folder_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(folder_frame, text="Папка для сохранения файлов:").pack(anchor="w")
        self.folder_entry = ctk.CTkEntry(folder_frame, textvariable=self.output_folder, width=350)
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.browse_button = ctk.CTkButton(folder_frame, text="Выбрать...", command=self.browse_folder)
        self.browse_button.pack(side="left")
        keywords_frame = ctk.CTkFrame(main_frame)
        keywords_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(keywords_frame, text="Файл с ключевыми словами (.txt, формат: 'ключ<TAB>количество'):").pack(
            anchor="w")
        self.keywords_entry = ctk.CTkEntry(keywords_frame, textvariable=self.keywords_file_path, width=350)
        self.keywords_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.browse_keywords_button = ctk.CTkButton(keywords_frame, text="Выбрать...",
                                                    command=self.browse_keywords_file)
        self.browse_keywords_button.pack(side="left")
        lang_frame = ctk.CTkFrame(main_frame)
        lang_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(lang_frame, text="Язык генерации текстов:").pack(side="left", padx=(0, 10))
        self.language_combobox = ctk.CTkComboBox(lang_frame, values=self.supported_languages,
                                                 variable=self.generation_language_var)
        self.language_combobox.pack(side="left", fill="x", expand=True)
        link_frame = ctk.CTkFrame(main_frame)
        link_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(link_frame, text="Ссылка для ключевого слова (URL):").pack(side="left", padx=(0, 10))
        self.target_link_entry = ctk.CTkEntry(link_frame, textvariable=self.target_link_var)
        self.target_link_entry.pack(side="left", fill="x", expand=True)
        format_frame = ctk.CTkFrame(main_frame)
        format_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(format_frame, text="Формат выходного файла:").pack(side="left", padx=(0, 10))
        self.format_segmented_button = ctk.CTkSegmentedButton(format_frame, values=["TXT", "HTML"],
                                                              variable=self.output_format_var)
        self.format_segmented_button.pack(side="left", fill="x", expand=True)
        self.threads_frame = ctk.CTkFrame(main_frame)
        self.threads_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(self.threads_frame, text="Количество потоков:").pack(side="left", padx=(0, 10))
        self.threads_slider = ctk.CTkSlider(self.threads_frame, from_=1, to=MAX_THREADS,
                                            variable=self.num_threads_var, number_of_steps=MAX_THREADS - 1)
        self.threads_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.threads_label = ctk.CTkLabel(self.threads_frame, text=str(self.num_threads_var.get()))
        self.threads_label.pack(side="left")
        self.num_threads_var.trace_add("write", self.update_threads_label)
        action_frame = ctk.CTkFrame(main_frame)
        action_frame.pack(pady=(10, 5), padx=10, fill="x")
        self.start_button = ctk.CTkButton(action_frame, text="Начать генерацию", command=self.start_generation_thread)
        self.start_button.pack(side="left", padx=5)
        self.stop_button = ctk.CTkButton(action_frame, text="Остановить", command=self.stop_generation,
                                         state="disabled")
        self.stop_button.pack(side="left", padx=5)
        self.api_status_button = ctk.CTkButton(action_frame, text="Статусы API Ключей",
                                               command=self._open_api_key_status_window)
        self.api_status_button.pack(side="left", padx=10)
        self.help_button = ctk.CTkButton(action_frame, text="!", width=30,
                                         command=self._open_help_window)
        self.help_button.pack(side="left")
        log_frame = ctk.CTkFrame(main_frame)
        log_frame.pack(pady=10, padx=10, fill="both", expand=True)
        ctk.CTkLabel(log_frame, text="Лог выполнения:").pack(anchor="w")
        self.log_textbox = ctk.CTkTextbox(log_frame, state="disabled", height=150, wrap="word")
        self.log_textbox.pack(fill="both", expand=True)
        self.after(100, self._repopulate_available_api_key_queue)
        self.after(LOG_FLUSH_INTERVAL_MS, self.flush_log_queue)
        self._on_provider_change()

    def _browse_openai_keys_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            self.openai_keys_file_var.set(path)

    def _browse_gemini_keys_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            self.gemini_keys_file_var.set(path)

    def _browse_proxy_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            self.proxy_file_var.set(path)

    def _on_provider_change(self, *_):
        provider = self.provider_var.get()
        if provider == PROVIDER_OPENAI:
            self.gemini_keys_frame.pack_forget()
            self.proxy_frame.pack_forget()
            self.openai_keys_frame.pack(pady=5, padx=10, fill="x")
            self.threads_frame.pack(pady=5, padx=10, fill="x")
        else:
            self.openai_keys_frame.pack_forget()
            self.gemini_keys_frame.pack(pady=5, padx=10, fill="x")
            self.proxy_frame.pack(pady=5, padx=10, fill="x")
            self.threads_frame.pack_forget()


    def _load_api_keys_from_file(self, path):
        if not path or not os.path.exists(path):
            messagebox.showerror("Ошибка", f"Файл с ключами не найден: {path}")
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                keys=[line.strip() for line in f if line.strip()]
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл ключей: {e}")
            return []
        if not keys:
            messagebox.showerror("Ошибка", f"В файле {path} нет валидных строк с ключами")
        return keys

    def _load_proxies_from_file(self, path):
        if not path or not os.path.exists(path):
            messagebox.showerror("Ошибка", f"Файл с прокси не найден: {path}")
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                proxies = [line.strip() for line in f if line.strip()]
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл прокси: {e}")
            return []
        if not proxies:
            messagebox.showerror("Ошибка", f"В файле {path} нет валидных строк с прокси")
        return proxies

    def _proxy_line_to_url(self, proxy_line):
        try:
            host, port, user, password = proxy_line.split(":")
            return f"http://{user}:{password}@{host}:{port}"
        except ValueError:
            return None

    def _get_proxy_country(self, proxy_line):
        proxy_url = self._proxy_line_to_url(proxy_line)
        if not proxy_url:
            return "Unknown"
        proxies = {"http": proxy_url, "https": proxy_url}
        try:
            resp = requests.get("http://ip-api.com/json", proxies=proxies, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("country", "Unknown")
        except Exception:
            pass
        return "Unknown"

    def _prepare_api_keys(self):
        provider=self.provider_var.get()
        if provider==PROVIDER_OPENAI:
            keys=self._load_api_keys_from_file(self.openai_keys_file_var.get())
            self.api_key_proxy_map = {}
        else:
            keys=self._load_api_keys_from_file(self.gemini_keys_file_var.get())
            proxies=self._load_proxies_from_file(self.proxy_file_var.get())
            if len(proxies)<len(keys):
                msg = f"В папке .txt - {len(keys)} АПИ, в папке с прокси - {len(proxies)} ПРОКСИ"
                messagebox.showerror("Ошибка", msg)
                self.log_message(msg, "ERROR")
                return False
            with self.api_key_management_lock:
                self.api_key_proxy_map=dict(zip(keys,proxies))
        if not keys:
            return False
        with self.api_key_management_lock:
            self.api_keys_list=keys
        self._repopulate_available_api_key_queue()
        return True

    def handle_api_keys_textbox_change(self, event=None):
        self.update_api_keys_from_textbox()
        self._initial_check_and_revive_keys()
        self._repopulate_available_api_key_queue()
        self.update_threads_label()

    def update_threads_label(self, *args):
        current_var_val = self.num_threads_var.get()
        if hasattr(self, 'threads_label') and self.threads_label.winfo_exists():
            self.threads_label.configure(text=str(current_var_val))
        if hasattr(self, 'threads_slider') and self.threads_slider.winfo_exists():
            with self.api_key_queue.mutex:
                num_active_keys = len(set(self.api_key_queue.queue))
            per_key = self._current_per_key_concurrency()
            if num_active_keys == 0:
                s_max_logical_limit = max(1, current_var_val)
            else:
                s_max_logical_limit = min(MAX_THREADS, num_active_keys * per_key)
            s_max_logical_limit = max(1, s_max_logical_limit)
            slider_from_value = 1
            slider_actual_to = slider_from_value + 1 if s_max_logical_limit == slider_from_value else s_max_logical_limit
            slider_actual_to = max(slider_actual_to, slider_from_value + 1)
            slider_number_of_steps = max(1, slider_actual_to - slider_from_value)
            self.threads_slider.configure(to=slider_actual_to, number_of_steps=slider_number_of_steps)
            final_var_val = current_var_val
            if num_active_keys > 0:
                if final_var_val > s_max_logical_limit:
                    final_var_val = s_max_logical_limit
                if final_var_val < slider_from_value:
                    final_var_val = slider_from_value
            if self.num_threads_var.get() != final_var_val:
                self.num_threads_var.set(final_var_val)
            if hasattr(self, 'threads_label') and self.threads_label.winfo_exists():
                self.threads_label.configure(text=str(self.num_threads_var.get()))

    def _load_telegram_settings(self):
        token, chat_id = load_telegram_settings(self.log_message)
        self.telegram_bot_token = token
        self.telegram_chat_id = chat_id
        global GLOBAL_TELEGRAM_TOKEN, GLOBAL_TELEGRAM_CHAT_ID
        if token and chat_id:
            GLOBAL_TELEGRAM_TOKEN = token
            GLOBAL_TELEGRAM_CHAT_ID = chat_id

    def send_telegram_notification(self, text):
        send_telegram_message(self.telegram_bot_token, self.telegram_chat_id, text, self.log_message)

    def browse_folder(self):
        fld = filedialog.askdirectory()
        if fld: self.output_folder.set(fld); self.log_message(f"Папка для сохранения: {fld}")

    def browse_keywords_file(self):
        f_path = filedialog.askopenfilename(title="Выберите файл с ключевыми словами",
                                            filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        if f_path:
            self.keywords_file_path.set(f_path)
            self.log_message(f"Файл с ключевыми словами: {f_path}")
            self._apply_special_params_from_keywords_file(f_path)

    def _apply_special_params_from_keywords_file(self, path):
        """Extract link, language and topic markers from the keywords file."""
        link_marker = "!==+"
        lang_marker = "!===+"
        topic_markers = ["!====+"]
        detected_link = None
        detected_lang = None
        detected_topic = None
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith(lang_marker):
                        lang_value = stripped[len(lang_marker):].strip()
                        if lang_value:
                            detected_lang = lang_value
                    elif stripped.startswith(link_marker):
                        link_value = stripped[len(link_marker):].strip()
                        if link_value:
                            detected_link = link_value
                    elif any(stripped.startswith(m) for m in topic_markers):
                        for t_marker in topic_markers:
                            if stripped.startswith(t_marker):
                                topic_value = stripped[len(t_marker):].strip()
                                if topic_value:
                                    detected_topic = topic_value
                                break
        except Exception as e:
            self.log_message(f"Ошибка чтения файла ключевых слов: {e}", "ERROR")
            return

        if detected_link:
            self.target_link_var.set(detected_link)
            self.log_message(f"Ссылка из файла загружена: {detected_link}")
        if detected_lang:
            self.generation_language_var.set(detected_lang)
            self.log_message(f"Язык из файла загружен: {detected_lang}")
        if detected_topic is not None:
            self.topic_word = detected_topic
            self.log_message(f"Тема из файла загружена: {detected_topic}")

    def _count_total_keywords(self, path):
        """Return total quantity of keywords from file."""
        total = 0
        if not path:
            return total
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if stripped.startswith("!==+") or stripped.startswith("!===+") or stripped.startswith("!====+"):
                        continue
                    parts = stripped.split("\t")
                    if len(parts) == 2:
                        try:
                            qty = int(parts[1].strip())
                            if qty > 0:
                                total += qty
                        except ValueError:
                            continue
        except Exception as e:
            self.log_message(f"Ошибка подсчета количества ключевых слов: {e}", "ERROR")
        return total

    def load_settings(self):
        cfg = configparser.ConfigParser()
        if os.path.exists(self.config_file):
            try:
                cfg.read(self.config_file, encoding='utf-8')
                if "Settings" in cfg:
                    s = cfg["Settings"]
                    self.output_folder.set(s.get("OutputFolder", ""))
                    self.provider_var.set(s.get("Provider", PROVIDER_OPENAI))
                    self.openai_keys_file_var.set(s.get("OpenAIKeysFile", ""))
                    self.gemini_keys_file_var.set(s.get("GeminiKeysFile", ""))
                    self.proxy_file_var.set(s.get("ProxyFile", ""))
                    self.num_threads_var.set(s.getint("NumThreads", 5))
                    self.generation_language_var.set(s.get("GenerationLanguage", "Русский"))
                    self.target_link_var.set(s.get("TargetLink", ""))
                    self.output_format_var.set(s.get("OutputFormat", "TXT"))
                    self.log_message("Настройки загружены.")
                else:
                    self.log_message("Секция 'Settings' не найдена в файле конфигурации.", "WARNING")
            except Exception as e:
                self.log_message(f"Ошибка загрузки настроек: {e}", "ERROR")
        else:
            self.log_message("Файл настроек не найден. Будут использованы значения по умолчанию.", "INFO")

    def save_settings(self):
        self._save_api_key_statuses()
        cfg = configparser.ConfigParser()
        cfg["Settings"] = {
            "OutputFolder": self.output_folder.get(),
            "Provider": self.provider_var.get(),
            "OpenAIKeysFile": self.openai_keys_file_var.get(),
            "GeminiKeysFile": self.gemini_keys_file_var.get(),
            "ProxyFile": self.proxy_file_var.get(),
            "NumThreads": str(self.num_threads_var.get()),
            "GenerationLanguage": self.generation_language_var.get(),
            "TargetLink": self.target_link_var.get(),
            "OutputFormat": self.output_format_var.get(),
        }
        try:
            with open(self.config_file, "w", encoding="utf-8") as cf:
                cfg.write(cf)
            self.log_message("Настройки сохранены.")
        except Exception as e:
            self.log_message(f"Ошибка сохранения настроек: {e}", "ERROR")

    def on_closing(self):
        if self.generation_active:
            if messagebox.askyesno("Подтверждение",
                                   "Генерация все еще активна. Вы уверены, что хотите выйти? \nТекущие задачи будут прерваны (но API вызовы могут завершиться)."):
                self.log_message("Запрос на выход во время активной генерации. Остановка...", "WARNING")
                self.stop_generation()
                self.after(1500, self._perform_shutdown)
            else:
                return
        else:
            self._perform_shutdown()

    def _perform_shutdown(self):
        global PROGRAM_EXITED_VIA_UI
        PROGRAM_EXITED_VIA_UI = True
        self.save_settings()
        self.destroy()

    def update_api_keys_from_textbox(self):
        if not hasattr(self, 'api_keys_textbox') or not self.api_keys_textbox.winfo_exists(): return
        keys_str = self.api_keys_textbox.get("1.0", tk.END).strip()
        new_keys_list = [k.strip() for k in keys_str.split("\n") if k.strip()]
        with self.api_key_management_lock:
            old_keys_set = set(self.api_keys_list)
            new_keys_set = set(new_keys_list)
            added_keys = new_keys_set - old_keys_set
            removed_keys = old_keys_set - new_keys_set
            self.api_keys_list = new_keys_list
        with self.api_key_statuses_lock:
            for key_to_remove in removed_keys:
                if key_to_remove in self.api_key_statuses:
                    del self.api_key_statuses[key_to_remove]
                    self.log_message(f"Статус для удаленного ключа {key_to_remove[:7]}... очищен.", "DEBUG")
            for key_to_add in added_keys:
                if key_to_add not in self.api_key_statuses:
                    self.api_key_statuses[key_to_add] = self._get_default_api_key_status()
                    self.log_message(f"Добавлен новый ключ {key_to_add[:7]}... в отслеживание статусов.", "DEBUG")
        if added_keys or removed_keys:
            self._save_api_key_statuses()
            save_shared_keys(self.api_keys_list)
            try:
                self.save_settings()
            except Exception as e_save:
                self.log_message(f"Ошибка автосохранения настроек: {e_save}", "ERROR")

    def _update_gui_and_log_bad_key(self, bad_key):
        key_actually_removed_from_master = False
        with self.api_key_management_lock:
            if bad_key in self.api_keys_list:
                self.api_keys_list.remove(bad_key)
                self.log_message(f"Ключ {bad_key[:7]}... удален из активного списка (признан плохим).", "INFO")
                key_actually_removed_from_master = True
        with self.api_key_statuses_lock:
            if bad_key in self.api_key_statuses:
                del self.api_key_statuses[bad_key]
                self.log_message(f"Статус для плохого ключа {bad_key[:7]}... удален.", "DEBUG")
        if key_actually_removed_from_master:
            try:
                existing_bad_keys = set()
                bad_file = app_path(BAD_API_KEYS_FILE)
                if os.path.exists(bad_file):
                    with open(bad_file, "r", encoding="utf-8") as f_read_bad:
                        for line in f_read_bad: existing_bad_keys.add(line.strip())
                if bad_key not in existing_bad_keys:
                    with open(bad_file, "a", encoding="utf-8") as f_bad:
                        f_bad.write(bad_key + "\n")
                    self.log_message(f"Ключ {bad_key[:7]}... добавлен в {BAD_API_KEYS_FILE}", "INFO")
                    with BAD_API_KEYS_CACHE_LOCK:
                        if BAD_API_KEYS_CACHE is not None:
                            BAD_API_KEYS_CACHE.add(bad_key)
            except Exception as e:
                self.log_message(f"Не удалось записать плохой ключ {bad_key[:7]}... в файл: {e}", "ERROR")
            if hasattr(self, 'api_keys_textbox') and self.api_keys_textbox.winfo_exists():
                self.api_keys_textbox.delete("1.0", tk.END)
                if self.api_keys_list: self.api_keys_textbox.insert("1.0", "\n".join(self.api_keys_list))
            self._repopulate_available_api_key_queue()
            self._save_api_key_statuses()

    def validate_inputs(self):
        if self.api_key_queue.empty():
            messagebox.showerror("Ошибка валидации",
                                 "Нет доступных API ключей для работы. Проверьте введенные ключи и их статусы.")
            return False
        if not self.output_folder.get():
            messagebox.showerror("Ошибка валидации", "Пожалуйста, выберите папку для сохранения файлов.")
            return False
        if not os.path.isdir(self.output_folder.get()):
            messagebox.showerror("Ошибка валидации",
                                 f"Указанная папка для сохранения не существует: {self.output_folder.get()}")
            return False
        if not self.keywords_file_path.get():
            messagebox.showerror("Ошибка валидации", "Пожалуйста, выберите файл с ключевыми словами.")
            return False
        if not os.path.isfile(self.keywords_file_path.get()):
            messagebox.showerror("Ошибка валидации",
                                 f"Указанный файл с ключевыми словами не существует: {self.keywords_file_path.get()}")
            return False
        if not self.generation_language_var.get():
            messagebox.showerror("Ошибка валидации", "Пожалуйста, выберите язык генерации текстов.")
            return False
        if not self.output_format_var.get() in ["TXT", "HTML"]:
            messagebox.showerror("Ошибка валидации",
                                 "Пожалуйста, выберите корректный формат выходного файла (TXT или HTML).")
            return False
        if self.target_link_var.get():
            if not (self.target_link_var.get().startswith("http://") or self.target_link_var.get().startswith(
                    "https://")):
                if messagebox.askyesno("Предупреждение: Ссылка",
                                       "Введенная ссылка не начинается с http:// или https://. \nЭто может быть некорректный URL. Продолжить?"):
                    pass
                else:
                    return False
        return True

    def set_ui_for_generation(self, active):
        self.generation_active = active
        widgets_to_disable = [self.folder_entry, self.browse_button, self.keywords_entry,
                              self.browse_keywords_button, self.threads_slider, self.language_combobox,
                              self.target_link_entry, self.format_segmented_button]
        button_state_if_active = "disabled"
        button_state_if_inactive = "normal"
        element_state = "disabled" if active else "normal"
        self.start_button.configure(state=button_state_if_active if active else button_state_if_inactive,
                                    text="Генерация..." if active else "Начать генерацию")
        self.stop_button.configure(state=button_state_if_inactive if active else button_state_if_active)
        for widget in widgets_to_disable:
            if hasattr(widget, 'configure') and widget.winfo_exists():
                widget.configure(state=element_state)

    def start_generation_thread(self):
        self.stop_event.clear()
        self.silent_stop = False
        # Reset topic to avoid leftover value from previous runs
        self.topic_word = ""
        if not self._prepare_api_keys():
            return
        if not self.validate_inputs():
            return
        target_link = self.target_link_var.get().strip()
        if target_link:
            total_kw = self._count_total_keywords(self.keywords_file_path.get())
            sanitized_folder_name = self.sanitize_filename(target_link.replace("https://", "").replace("http://", ""))
            base_folder = os.path.join(self.output_folder.get(), f"{sanitized_folder_name}_{total_kw}")
            dest_folder = base_folder
            lang_suffix = self.sanitize_filename(self.generation_language_var.get())
            try:
                os.makedirs(dest_folder, exist_ok=False)
            except FileExistsError:
                dest_folder = os.path.join(self.output_folder.get(), f"{sanitized_folder_name}_{lang_suffix}_{total_kw}")
                suffix_counter = 1
                while True:
                    try:
                        os.makedirs(dest_folder, exist_ok=False)
                        break
                    except FileExistsError:
                        dest_folder = os.path.join(
                            self.output_folder.get(),
                            f"{sanitized_folder_name}_{lang_suffix}_{suffix_counter}_{total_kw}"
                        )
                        suffix_counter += 1
            self.output_folder.set(dest_folder)
            if self.folder_entry.winfo_exists():
                self.folder_entry.delete(0, tk.END)
                self.folder_entry.insert(0, dest_folder)
        self.load_progress_data()
        with self.api_stats_lock:
            self.api_key_usage_stats.clear()
        self._initial_check_and_revive_keys()
        if self.api_key_queue.empty():
            self.log_message(
                "Нет доступных API ключей в очереди после проверки. Генерация не может быть запущена.",
                "ERROR",
            )
            messagebox.showerror(
                "Ошибка API ключей",
                "Нет доступных API ключей. Проверьте статусы ключей или добавьте новые.",
            )
            self.set_ui_for_generation(False)
            return
        self.set_ui_for_generation(True)
        self.waiting_for_project_slot = True
        self.log_message("Ожидание свободного слота генерации...", "INFO")
        self._wait_for_project_slot()

    def _wait_for_project_slot(self):
        if self.stop_event.is_set():
            self.waiting_for_project_slot = False
            self.set_ui_for_generation(False)
            return
        acquired = GLOBAL_PROJECT_SEMAPHORE.acquire(blocking=False)
        if acquired:
            self.project_slot_acquired = True
            self.waiting_for_project_slot = False
            self._begin_generation_after_slot()
        else:
            self.after(1000, self._wait_for_project_slot)

    def _begin_generation_after_slot(self):
        with self.api_key_queue.mutex:
            queued_keys = list(self.api_key_queue.queue)
        active_key_count = len(set(queued_keys))
        # Как минимум по одному потоку на каждый активный ключ,
        # но не больше глобального лимита потоков.
        target_threads = max(self.num_threads_var.get(), active_key_count) or 1
        target_threads = min(target_threads, MAX_THREADS)
        self.num_threads_var.set(target_threads)
        self.update_threads_label()
        self.output_file_counter = self._ensure_output_file_count(force=True)
        self.log_message(
            f"Запуск генерации: {self.num_threads_var.get()} поток(а/ов), активных ключей в очереди: {active_key_count}."
        )
        self.successful_task_ids.clear()
        self.all_task_definitions.clear()
        threading.Thread(target=self.process_all_keywords, daemon=True).start()
    def stop_generation(self):
        if self.generation_active or self.waiting_for_project_slot:
            self.silent_stop = True
            self.stop_event.set()
            self.log_message("Генерация остановлена")
            with self.task_creation_queue.mutex:
                self.task_creation_queue.queue.clear()
            self.all_task_definitions.clear()
            self._save_api_key_statuses()
            if self.project_slot_acquired:
                GLOBAL_PROJECT_SEMAPHORE.release()
                self.project_slot_acquired = False
            if self.waiting_for_project_slot and not self.project_slot_acquired:
                self.waiting_for_project_slot = False
            self.set_ui_for_generation(False)

    def sanitize_filename(self, fname_base):
        s = re.sub(r'[\\/*?:"<>|]', "", fname_base)
        s = s.strip()
        return s[:MAX_FILENAME_LENGTH]

    def get_unique_filepath(self, h1_text):
        folder = self.output_folder.get()
        sanitized = self.sanitize_filename(h1_text)
        if not sanitized:
            sanitized = "untitled_article"
        chosen_format = self.output_format_var.get()
        extension = ".txt" if chosen_format == "TXT" else ".html"

        with self.used_filepaths_lock:
            base_filepath = os.path.join(folder, f"{sanitized}{extension}")
            if base_filepath not in self.used_filepaths and not os.path.exists(base_filepath):
                self.used_filepaths.add(base_filepath)
                return base_filepath
            i = 1
            while True:
                indexed_filepath = os.path.join(folder, f"{sanitized}_{i}{extension}")
                if indexed_filepath not in self.used_filepaths and not os.path.exists(indexed_filepath):
                    self.used_filepaths.add(indexed_filepath)
                    return indexed_filepath
                i += 1
                if i > 1000:
                    self.log_message(
                        f"Превышено количество попыток создания уникального имени файла для {sanitized}",
                        "ERROR",
                    )
                    fallback = os.path.join(folder, f"{sanitized}_{random.randint(1001, 9999)}{extension}")
                    self.used_filepaths.add(fallback)
                    return fallback

    def release_filepath(self, path):
        """Remove a filepath from the reserved set."""
        with self.used_filepaths_lock:
            self.used_filepaths.discard(path)

    def _ensure_output_file_count(self, force=False):
        """Check output folder for small files and return count of valid ones."""
        now = time.time()
        if not force and now - self.last_output_scan_time < 2:
            return self.output_file_counter
        valid_count = 0
        try:
            folder = self.output_folder.get()
            if not folder:
                return 0
            exts = (".txt", ".html")
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(exts):
                    try:
                        size = os.path.getsize(fpath)
                        if size < MIN_ARTICLE_SIZE:
                            os.remove(fpath)
                            self.log_message(f"Удален слишком маленький файл: {fname}", "WARNING")
                        elif size > MAX_ARTICLE_SIZE:
                            os.remove(fpath)
                            self.log_message(f"Удален слишком большой файл: {fname}", "WARNING")
                        else:
                            valid_count += 1
                    except Exception as e_check:
                        self.log_message(f"Ошибка проверки файла {fname}: {e_check}", "ERROR")
        except Exception as e:
            self.log_message(f"Ошибка проверки количества файлов: {e}", "ERROR")
        with self.output_count_lock:
            self.output_file_counter = valid_count
        self.last_output_scan_time = now
        return valid_count

    def _has_text_under_h1(self, body_root):
        """Return True if there's textual content before the first H2."""
        if not body_root:
            return False
        for child in body_root.children:
            if not hasattr(child, "name"):
                if str(child).strip():
                    return True
                continue
            if child.name == "h2":
                break
            if child.name in ["p", "li"]:
                if child.get_text(strip=True):
                    return True
            elif hasattr(child, "find_all"):
                for sub in child.find_all(["p", "li"], recursive=False):
                    if sub.get_text(strip=True):
                        return True
        return False

    def call_openai_api(self, client_instance, messages, api_key_used_for_call, retries=3, delay_seconds=0.5):
        for attempt in range(retries):
            if self.stop_event.is_set():
                self.log_message("API вызов прерван сигналом остановки.", "WARNING")
                return None
            with api_key_last_call_time_lock:
                last_ts = api_key_last_call_time.get(api_key_used_for_call, 0.0)
            to_wait = PER_KEY_CALL_INTERVAL - (time.time() - last_ts)
            if to_wait > 0:
                time.sleep(to_wait)
            self.log_message("Глобальная задержка 10с перед вызовом OpenAI", "DEBUG")
            if self.stop_event.wait(10):
                self.log_message("Глобальная задержка перед OpenAI прервана", "DEBUG")
                return None
            try:
                raw_response = client_instance.chat.completions.with_raw_response.create(model=DEFAULT_MODEL,
        messages=messages, timeout=300)
                completion = raw_response.parse()
                if hasattr(raw_response, 'headers'):
                    self._update_api_key_status_from_headers(api_key_used_for_call, raw_response.headers)
                with api_key_last_call_time_lock:
                    api_key_last_call_time[api_key_used_for_call] = time.time()
                return completion.choices[0].message.content.strip()
            except RateLimitError as rle:
                log_level = "ERROR" if attempt + 1 == retries else "WARNING"
                self.log_message(f"OpenAI API RateLimitError: {rle}. Попытка {attempt + 1}/{retries}.", log_level)
                if hasattr(rle, 'response') and rle.response is not None and hasattr(rle.response, 'headers'):
                    self._update_api_key_status_from_headers(api_key_used_for_call, rle.response.headers, is_error=True,
                                                             status_code=429)
                specific_error_type = None
                try:
                    if hasattr(rle, 'response') and rle.response is not None:
                        error_details = rle.response.json().get("error", {}); specific_error_type = error_details.get(
                            "type")
                    elif hasattr(rle, 'body') and rle.body is not None and 'error' in rle.body:
                        specific_error_type = rle.body.get('error', {}).get('type')
                except Exception as e_parse:
                    self.log_message(f"Не удалось извлечь specific_error_type из RateLimitError: {e_parse}", "DEBUG")
                if specific_error_type in ['billing_not_active', 'insufficient_quota']:
                    self.log_message(
                        f"RateLimitError тип '{specific_error_type}'. Ключ {api_key_used_for_call[:7]}... будет обработан как невалидный.",
                        "ERROR")
                    return "INVALID_API_KEY_ERROR"
                current_delay = delay_seconds * (2 ** attempt)
                if attempt + 1 < retries:
                    self.log_message(f"Ожидание {current_delay} секунд перед следующей попыткой...",
                                     "INFO"); time.sleep(current_delay)
                else:
                    return None
            except APIStatusError as ase:
                log_level = "ERROR" if attempt + 1 == retries else "WARNING"
                self.log_message(
                    f"OpenAI API StatusError: {ase}. Status Code: {ase.status_code}. Попытка {attempt + 1}/{retries}.",
                    log_level)
                if hasattr(ase, 'response') and ase.response is not None and hasattr(ase.response, 'headers'):
                    self._update_api_key_status_from_headers(api_key_used_for_call, ase.response.headers, is_error=True,
                                                             status_code=ase.status_code)
                if ase.status_code == 401:
                    self.log_message(
                        f"Ошибка 401: Недействительный API ключ {api_key_used_for_call[:7]}.... Ключ будет обработан как невалидный.",
                        "ERROR")
                    return "INVALID_API_KEY_ERROR"
                if ase.status_code == 429:
                    specific_error_type_ase = None
                    try:
                        if hasattr(ase, 'response') and ase.response is not None:
                            error_details_ase = ase.response.json().get("error",
                                                                        {}); specific_error_type_ase = error_details_ase.get(
                                "type")
                        elif hasattr(ase, 'body') and ase.body is not None and 'error' in ase.body:
                            specific_error_type_ase = ase.body.get('error', {}).get('type')
                    except Exception as e_parse_ase:
                        self.log_message(f"Не удалось извлечь specific_error_type из APIStatusError 429: {e_parse_ase}",
                                         "DEBUG")
                    if specific_error_type_ase in ['billing_not_active', 'insufficient_quota']:
                        self.log_message(
                            f"APIStatusError 429 тип '{specific_error_type_ase}'. Ключ {api_key_used_for_call[:7]}... будет обработан как невалидный.",
                            "ERROR")
                        return "INVALID_API_KEY_ERROR"
                    self.log_message(f"Получен статус 429 (Rate Limit) как APIStatusError. Увеличенная задержка.",
                                     "WARNING")
                    current_delay = delay_seconds * (2 ** attempt) * 1.5
                else:
                    current_delay = delay_seconds * (2 ** attempt)
                if attempt + 1 < retries:
                    self.log_message(f"Ожидание {current_delay:.2f} секунд перед следующей попыткой...",
                                     "INFO"); time.sleep(current_delay)
                else:
                    return None
            except APIConnectionError as ace:
                log_level = "ERROR" if attempt + 1 == retries else "WARNING"
                self.log_message(f"OpenAI API ConnectionError: {ace}. Попытка {attempt + 1}/{retries}.", log_level)
                current_delay = delay_seconds * (attempt + 1)
                if attempt + 1 < retries:
                    time.sleep(current_delay)
                else:
                    return None
            except Exception as e:
                log_level = "ERROR" if attempt + 1 == retries else "WARNING"
                self.log_message(
                    f"Неожиданная ошибка OpenAI API ({type(e).__name__}): {e}. Попытка {attempt + 1}/{retries}.",
                    log_level)
                current_delay = delay_seconds * (attempt + 1)
                if attempt + 1 < retries:
                    time.sleep(current_delay)
                else:
                    return None
        return None

    def call_gemini_api(self, api_key_used_for_call, prompt_text, retries=3, delay_seconds=0.5, context=""):
        key_short = api_key_used_for_call[:7]
        token_delay = GEMINI_KEY_SLOT_SECONDS
        lock = self.gemini_key_locks[api_key_used_for_call]
        bucket = self.gemini_key_quota[api_key_used_for_call]
        self.log_message(f"[{context}] Ожидание ключа {key_short}", "DEBUG")
        while not self.stop_event.is_set():
            if lock.acquire(timeout=1):
                break
        else:
            return None
        with api_key_last_call_time_lock:
            last = api_key_last_call_time.get(api_key_used_for_call, 0.0)
        wait_needed = last + 10.0 - time.time()
        if wait_needed > 0:
            self.log_message(
                f"[{context}] Пауза {wait_needed:.2f}с перед запросом Gemini ключом {key_short}",
                "DEBUG",
            )
            if self.stop_event.wait(wait_needed):
                lock.release()
                return None
        self.log_message(f"[{context}] Ожидание квоты ключа {key_short}", "DEBUG")
        while not self.stop_event.is_set():
            if bucket.acquire(timeout=1):
                break
        else:
            lock.release()
            return None
        try:
            self.log_message(
                f"[{context}] Подготовка запроса к Gemini для ключа {key_short}, длина промпта {len(prompt_text)}",
                "DEBUG",
            )
            if not self._before_gemini_call(api_key_used_for_call):
                self.log_message(
                    f"[{context}] Ключ {key_short} не прошёл предварительную проверку Gemini",
                    "DEBUG",
                )
                return "GEMINI_KEY_EXHAUSTED"
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
            params = {"key": api_key_used_for_call}
            headers = {"Content-Type": "application/json"}
            payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
            proxy_line = self.api_key_proxy_map.get(api_key_used_for_call)
            proxies = None
            if proxy_line:
                proxy_url = self._proxy_line_to_url(proxy_line)
                if proxy_url:
                    proxies = {"http": proxy_url, "https": proxy_url}
                    country = self.proxy_countries.get(proxy_line)
                    if not country:
                        country = self._get_proxy_country(proxy_line)
                        self.proxy_countries[proxy_line] = country
                    host, port, *_ = proxy_line.split(":")
                    self.log_message(
                        f"[{context}] Используется прокси {host}:{port} ({country})",
                        "INFO",
                    )
            for attempt in range(retries):
                if self.stop_event.is_set():
                    self.log_message(f"[{context}] Прекращено из-за стоп-сигнала", "DEBUG")
                    return None
                try:
                    self.log_message(
                        f"[{context}] Попытка {attempt + 1}/{retries} отправки запроса Gemini ключом {key_short}",
                        "DEBUG",
                    )
                    with api_key_last_call_time_lock:
                        api_key_last_call_time[api_key_used_for_call] = time.time()
                    resp = requests.post(
                        url,
                        params=params,
                        headers=headers,
                        json=payload,
                        timeout=(10, 120),
                        proxies=proxies,
                    )
                    self.log_message(
                        f"[{context}] Ответ Gemini {resp.status_code} за {getattr(resp, 'elapsed', datetime.timedelta(0)).total_seconds():.2f}с",
                        "DEBUG",
                    )
                    self.log_message(
                        f"[{context}] Тело ответа Gemini (первые 200 символов): {resp.text[:200]}",
                        "DEBUG",
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        text = (
                            data.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        if text:
                            self._after_gemini_call(api_key_used_for_call)
                            self.log_message(
                                f"[{context}] Получен текст длиной {len(text)} символов от Gemini ключом {key_short}",
                                "DEBUG",
                            )
                            return text
                    else:
                        err_msg = resp.text
                        retry_wait = None
                        if resp.headers.get("Content-Type", "").startswith("application/json"):
                            try:
                                data = resp.json()
                                err_msg = data.get("error", {}).get("message", err_msg)
                                for detail in data.get("error", {}).get("details", []):
                                    if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                                        delay = detail.get("retryDelay", "0s")
                                        secs = 0.0
                                        if delay.endswith("s"):
                                            try:
                                                secs = float(delay[:-1])
                                            except Exception:
                                                secs = 0.0
                                        retry_wait = secs + 10.0
                            except Exception:
                                err_msg = err_msg[:200]
                        self.log_message(
                            f"[{context}] Gemini API error {resp.status_code}: {err_msg[:200]}",
                            "ERROR",
                        )
                        if resp.status_code == 429:
                            wait_for = retry_wait if retry_wait is not None else 10.0
                            self.log_message(
                                f"[{context}] Повтор через {wait_for:.1f}с после 429",
                                "WARNING",
                            )
                            token_delay = max(token_delay, wait_for)
                            if self.stop_event.wait(wait_for):
                                self.log_message(
                                    f"[{context}] Ожидание после 429 прервано стоп-сигналом",
                                    "DEBUG",
                                )
                                return None
                except requests.exceptions.Timeout as e:
                    self.log_message(
                        f"[{context}] Таймаут Gemini API: {e}",
                        "WARNING",
                    )
                except Exception as e:
                    self.log_message(f"[{context}] Gemini API request failed: {e}", "ERROR")
                if attempt + 1 < retries:
                    pause = delay_seconds * (attempt + 1)
                    self.log_message(
                        f"[{context}] Повтор через {pause:.2f}с", "DEBUG",
                    )
                    if self.stop_event.wait(pause):
                        return None
            self._after_gemini_call(api_key_used_for_call)
            return ""
        finally:
            lock.release()
            self.log_message(f"[{context}] Ключ {key_short} освобождён", "DEBUG")

            def restore():
                bucket.release()
                self.log_message(
                    f"[{context}] Квота для ключа {key_short} восстановлена",
                    "DEBUG",
                )

            threading.Timer(token_delay, restore).start()

    # ================================================================================
    # НОВЫЙ МЕТОД ДЛЯ ОЧИСТКИ HTML (УБЕДИТЕСЬ, ЧТО ОН ВНУТРИ КЛАССА TextGeneratorApp)
    # ================================================================================
    def _clean_html_structure(self, html_content, log_prefix=""):
        """
        Очищает HTML-структуру от распространенных ошибок генерации,
        особенно некорректных тегов <p> внутри/вокруг таблиц или списков.
        """
        if not html_content:
            self.log_message(f"{log_prefix} HTML для очистки пуст.", "DEBUG")
            return ""
        try:
            # self.log_message(f"{log_prefix} Исходный HTML для очистки:\n{html_content[:1000]}...", "TRACE")
            soup = BeautifulSoup(html_content, 'html.parser')
            changes_made_overall = False

            # Используем цикл while для повторных проходов, так как unwrap может создавать новые ситуации
            max_passes = 5  # Предотвращение бесконечного цикла
            current_pass = 0
            while current_pass < max_passes:
                changes_made_this_pass = False
                current_pass += 1
                # self.log_message(f"{log_prefix} Очистка HTML, проход {current_pass}", "TRACE")

                # PASS 1: <p> является прямым потомком table, ul, ol, tr, thead, tbody, tfoot
                # Пример: <table><p>содержимое</p></table>  -> <table>содержимое</table>
                for p_tag in list(soup.find_all('p')):  # list() для безопасной итерации при изменении дерева
                    if not p_tag.parent: continue

                    if p_tag.parent.name in ['table', 'thead', 'tbody', 'tfoot', 'tr', 'ul', 'ol']:
                        # self.log_message(f"{log_prefix} [Pass {current_pass}, Rule 1A] Обнаружен <p> внутри <{p_tag.parent.name}>. Удаление <p>-обертки.", "DEBUG")
                        p_tag.unwrap()
                        changes_made_this_pass = True
                        changes_made_overall = True
                        continue  # Переходим к следующему <p> из первоначального списка

                # PASS 2: <p> содержит блочные элементы (table, ul, ol, tr, li, h1-h6, div и т.д.)
                # Пример: <p><table>...</table></p> -> <table>...</table>
                # Пример: <p><tr>...</tr></p> -> <tr>...</tr>
                block_elements_that_should_not_be_in_p = [
                    'table', 'ul', 'ol', 'dl',
                    'tr', 'th', 'td', 'li', 'dt', 'dd',
                    'thead', 'tbody', 'tfoot', 'caption', 'colgroup', 'col',
                    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                    'div', 'blockquote', 'pre', 'figure', 'hr',
                    'section', 'article', 'aside', 'nav', 'header', 'footer', 'main'
                ]
                for p_tag in list(soup.find_all('p')):
                    if not p_tag.parent: continue

                    contains_invalid_child = False
                    offending_child_name = "N/A"
                    if hasattr(p_tag, 'contents'):
                        for child in p_tag.contents:  # Проверяем прямых потомков
                            if hasattr(child, 'name') and child.name in block_elements_that_should_not_be_in_p:
                                contains_invalid_child = True
                                offending_child_name = child.name
                                break

                    if contains_invalid_child:
                        # self.log_message(f"{log_prefix} [Pass {current_pass}, Rule 1B] Обнаружен <p>, содержащий блочный элемент <{offending_child_name}>. Удаление <p>-обертки.", "DEBUG")
                        p_tag.unwrap()
                        changes_made_this_pass = True
                        changes_made_overall = True
                        continue

                # PASS 3: <li> содержит <p> как единственный значимый дочерний элемент
                # Пример: <li><p>текст</p></li> -> <li>текст</li>
                for li_tag in list(soup.find_all('li')):
                    if not li_tag.parent: continue

                    meaningful_children = [child for child in li_tag.contents if
                                           child.name or (isinstance(child, str) and child.strip())]
                    if len(meaningful_children) == 1 and meaningful_children[0].name == 'p':
                        p_to_unwrap = meaningful_children[0]
                        # self.log_message(f"{log_prefix} [Pass {current_pass}, Rule 2] Обнаружен единственный <p> внутри <li>. Удаление <p>-обертки.", "DEBUG")
                        p_to_unwrap.unwrap()
                        changes_made_this_pass = True
                        changes_made_overall = True

                # PASS 4: Очистка пустых тегов <p> или тегов <p>, содержащих только пробелы/&nbsp
                for p_tag in list(soup.find_all('p')):
                    if not p_tag.parent: continue

                    text_content = p_tag.get_text(separator="", strip=False)  # Получаем весь текст, включая пробелы
                    is_effectively_empty = not text_content.strip()  # Если после strip() ничего не осталось

                    # Дополнительная проверка на &nbsp; если strip=False оставляет его
                    if not is_effectively_empty:
                        is_nbsp_only = all(c in '\xa0\u00a0 ' for c in text_content)
                        if is_nbsp_only:
                            is_effectively_empty = True

                    if is_effectively_empty and not p_tag.find(True, recursive=False):  # Нет дочерних тегов
                        # self.log_message(f"{log_prefix} [Pass {current_pass}, Rule 3] Удаление пустого или содержащего только пробелы тега <p>.", "DEBUG")
                        p_tag.decompose()
                        changes_made_this_pass = True
                        changes_made_overall = True

                if not changes_made_this_pass:
                    # Если за проход не было изменений, дальнейшие проходы не нужны
                    # self.log_message(f"{log_prefix} На проходе {current_pass} изменений не было, завершаем очистку.", "TRACE")
                    break

            if changes_made_overall:
                self.log_message(
                    f"{log_prefix} Структура HTML была очищена/изменена после {current_pass} проход(а/ов).", "INFO")

            # self.log_message(f"{log_prefix} Очищенный HTML:\n{str(soup)[:1000]}...", "TRACE")
            return str(soup)
        except Exception as e:
            self.log_message(f"{log_prefix} КРИТИЧЕСКАЯ ОШИБКА при очистке HTML: {e}\n{traceback.format_exc()}",
                             "ERROR")
            return html_content  # Возвращаем исходный контент в случае ошибки

    # КОНЕЦ НОВОГО МЕТОДА _clean_html_structure
    # ================================================================================

    # ================================================================================
    # ОБНОВЛЕННЫЙ МЕТОД generate_single_article_content
    # (ЗАМЕНИТЕ СУЩЕСТВУЮЩИЙ ПОЛНОСТЬЮ)
    # ================================================================================
    def generate_single_article_content(self, task_id, keyword_phrase, task_num_for_keyword, total_tasks_for_keyword,
                                        global_task_num, total_global_tasks):
        if self.stop_event.is_set(): return False

        retrieved_api_key_str = None
        openai_client = None
        key_marked_as_bad_in_this_task = False
        key_went_to_cooldown_in_this_task = False

        selected_lang = self.generation_language_var.get()
        target_link = self.target_link_var.get().strip()
        kw_for_prompt = f"язык: {selected_lang}, тема {self.topic_word} для ключевого слова: {keyword_phrase}".strip() if self.topic_word else keyword_phrase
        log_prefix_base = f"[{task_id} ({task_num_for_keyword}/{total_tasks_for_keyword} для '{keyword_phrase}', {selected_lang}), Общая {global_task_num}/{total_global_tasks}]"
        log_prefix = log_prefix_base

        # Инициализация переменных, которые раньше были в области видимости цикла по body_lines
        heading_id_counter = 1
        link_inserted_flag = False

        try:
            retrieved_api_key_str = self.api_key_queue.get(block=True, timeout=20)
            with self.api_stats_lock:
                self.api_key_usage_stats[retrieved_api_key_str] = self.api_key_usage_stats.get(retrieved_api_key_str,
                                                                                               0) + 1
            key_short_display = f"...{retrieved_api_key_str[-5:]}" if len(
                retrieved_api_key_str) > 5 else retrieved_api_key_str
            log_prefix = f"[{task_id} ({task_num_for_keyword}/{total_tasks_for_keyword} для '{keyword_phrase}', {selected_lang}, ключ {key_short_display}), Общая {global_task_num}/{total_global_tasks}]"
            if self.provider_var.get() == PROVIDER_OPENAI:
                openai_client = OpenAI(api_key=retrieved_api_key_str, timeout=30.0, max_retries=0)
                if openai_client is None: raise ValueError("Клиент OpenAI не был инициализирован.")
            else:
                openai_client = None
        except Empty:
            self.log_message(f"{log_prefix_base} Ошибка: Таймаут получения API ключа из очереди.", "ERROR")
            self._initial_check_and_revive_keys()
            return False
        except Exception as e_init:
            self.log_message(
                f"{log_prefix_base} Ошибка инициализации клиента OpenAI ({retrieved_api_key_str[:7] if retrieved_api_key_str else 'N/A'}...): {e_init}. Пропуск.",
                "ERROR")
            if retrieved_api_key_str:
                with self.api_key_statuses_lock:
                    status_data = self.api_key_statuses.get(retrieved_api_key_str, self._get_default_api_key_status())
                if status_data.get("status") == "active":
                    self.api_key_queue.put(retrieved_api_key_str)
                else:
                    self.log_message(
                        f"Ключ {key_short_display} не возвращен в очередь, статус: {status_data.get('status')}.",
                        "DEBUG")
            return False

        try:
            self.log_message(f"{log_prefix} Начало генерации...")
            h1_user_prompt_variations = [
                f"Язык: {selected_lang}. Тема: {self.topic_word} - Придумай короткий, ясный, интригующий SEO H1 для статьи на тему I-Gaming (Не упоминай термин 'I-Gaming' в заголовке): {keyword_phrase}. Ответ должен содержать только текст заголовка, без HTML-тегов или кавычек. Ключевое слово {keyword_phrase} не должно быть в начале и конце, оно должно быть гармонично вставлено в средину заголовка ЭТО ВАЖНО!! Заголовок не должен быть такой как у всех!!! Только одно предложение в заголовке!",
                f"Язык: {selected_lang}. Тема: {self.topic_word} - Создай привлекающий внимание SEO заголовок H1 для текста о {keyword_phrase}. Ответ должен содержать только текст заголовка, без HTML-тегов или кавычек. Ключевое слово {keyword_phrase} не должно быть в начале и конце, оно должно быть гармонично вставлено в средину заголовка ЭТО ВАЖНО!! Заголовок не должен быть такой как у всех!!! Только одно предложение в заголовке!",
                f"Язык: {selected_lang}. Тема: {self.topic_word} - Какой SEO H1 лучше всего подойдет для статьи о {keyword_phrase}? Заголовок должен быть цепляющим. Ответ должен содержать только текст заголовка, без HTML-тегов или кавычек. Ключевое слово {keyword_phrase} не должно быть в начале и конце, оно должно быть гармонично вставлено в средину заголовка ЭТО ВАЖНО!! Заголовок не должен быть такой как у всех!!! Только одно предложение в заголовке!",
                f"Язык: {selected_lang}. Тема: {self.topic_word} - Сформулируй основной SEO заголовок (H1) для материала по {keyword_phrase}. Кратко и по существу. Ответ должен содержать только текст заголовка, без HTML-тегов или кавычек. Ключевое слово {keyword_phrase} не должно быть в начале и конце, оно должно быть гармонично вставлено в средину заголовка ЭТО ВАЖНО!! Заголовок не должен быть такой как у всех!!! Только одно предложение в заголовке!",
                f"Язык: {selected_lang}. Тема: {self.topic_word} - Напиши вопросительный SEO H1 для статьи, раскрывающей {keyword_phrase}. Ответ должен содержать только текст заголовка, без HTML-тегов или кавычек. Ключевое слово {keyword_phrase} не должно быть в начале и конце, оно должно быть гармонично вставлено в средину заголовка!! Заголовок не должен быть такой как у всех!!! Только одно предложение в заголовке!",
                f"Язык: {selected_lang}. Тема: {self.topic_word} - Предложи SEO H1, который подчеркивает пользу или решение проблемы, связанной с {keyword_phrase}. Ответ должен содержать только текст заголовка, без HTML-тегов или кавычек. Ключевое слово {keyword_phrase} не должно быть в начале и конце, оно должно быть гармонично вставлено в средину заголовка ЭТО ВАЖНО!! Заголовок не должен быть такой как у всех!!! Только одно предложение в заголовке!",
                f"Язык: {selected_lang}. Тема: {self.topic_word} - Сгенерируй SEO H1, который содержит цифру или статистический факт (если уместно для темы I-Gaming (Не упоминай термин 'I-Gaming' в заголовке) {keyword_phrase}). Если неуместно, то просто интригующий H1. Ответ должен содержать только текст заголовка, без HTML-тегов или кавычек. Ключевое слово {keyword_phrase} не должно быть в начале и конце, оно должно быть гармонично вставлено в средину заголовка!! Заголовок не должен быть такой как у всех!!! Только одно предложение в заголовке!"
            ]
            selected_h1_user_prompt = random.choice(h1_user_prompt_variations)
            base_h1_prompt = [
                {
                    "role": "system",
                    "content": f"Ты SEO-копирайтер. Придумай I-Gaming (Не упоминай термин 'I-Gaming' в тексте) заголовок H1. Ответ на {selected_lang} языке. Начало не должно быть типичным: избегай слов 'Discover', 'Dive', 'How', 'Ultimate', 'What', 'Reel', 'Is', 'Unlocking' и подобных. Заголовок должен начинаться уникально. Ключевое слово {keyword_phrase} не должно быть в начале!"
                },
                {"role": "user", "content": selected_h1_user_prompt}
            ]
            with self.previous_h1_lock:
                if self.previous_h1_text:
                    base_h1_prompt.append({
                        "role": "user",
                        "content": f"Сделай так, чтобы главный первый заголовок не был похож вообще на этот, проработай тщательно начало и конец, чтобы не было повторений: \"{self.previous_h1_text}\". ОБЗЯТАТЕЛЬНО НАЧАЛО НЕ ДОЛЖНО СОВПАДАТЬ!!!"
                    })

            if self.provider_var.get() == PROVIDER_OPENAI:
                original_h1_text_raw = self.call_openai_api(openai_client, base_h1_prompt, retrieved_api_key_str)
            else:
                prompt_text = "\n".join([m["content"] for m in base_h1_prompt]) + "\nСделай текст не слишком длинным и используй настоящую, проверяемую информацию; избегай вымышленных фактов."
                self.log_message(
                    f"{log_prefix} [H1] Отправка запроса к Gemini, длина промпта {len(prompt_text)}",
                    "DEBUG",
                )
                original_h1_text_raw = self.call_gemini_api(
                    retrieved_api_key_str, prompt_text, context=f"H1 {task_id}"
                )
                if original_h1_text_raw == "GEMINI_KEY_EXHAUSTED":
                    self._mark_gemini_key_exhausted(retrieved_api_key_str)
                    key_marked_as_bad_in_this_task = True
                    return False
                self.log_message(
                    f"{log_prefix} [H1] Получено {len(original_h1_text_raw)} символов",
                    "DEBUG",
                )

            if original_h1_text_raw == "INVALID_API_KEY_ERROR":
                self.log_message(f"{log_prefix} API ключ {key_short_display} невалиден (H1). Обработка...", "ERROR")
                self.after(0, self._update_gui_and_log_bad_key, retrieved_api_key_str)
                key_marked_as_bad_in_this_task = True
                return False

            with self.api_key_statuses_lock:
                current_key_status_after_h1 = self.api_key_statuses.get(retrieved_api_key_str, {}).get("status", "active")
            if current_key_status_after_h1 != "active":
                key_went_to_cooldown_in_this_task = True
                self.log_message(
                    f"{log_prefix} Ключ {key_short_display} перешел в '{current_key_status_after_h1}' после H1.",
                    "INFO")

            if not original_h1_text_raw or self.stop_event.is_set():
                self.log_message(
                    f"{log_prefix} H1 не сгенерирован ({'пусто' if not original_h1_text_raw else 'остановлено'}).",
                    "ERROR" if not original_h1_text_raw else "WARNING")
                return False
            original_h1_text = original_h1_text_raw.replace('"', '').replace("'", "").strip()
            if original_h1_text.lower().startswith('h1:'):
                self.log_message(
                    f"{log_prefix} H1 содержит префикс 'H1:' и будет пересоздан.",
                    "WARNING",
                )
                return False
            if not original_h1_text:
                original_h1_text = keyword_phrase
            self.log_message(f"{log_prefix} H1 (оригинал): '{original_h1_text}'")
            filepath_to_save = self.get_unique_filepath(original_h1_text)

            # Return key to queue while waiting before body generation
            with self.api_key_statuses_lock:
                status_after_h1 = self.api_key_statuses.get(retrieved_api_key_str, {}).get("status", "active")
            if status_after_h1 == "active":
                self.api_key_queue.put(retrieved_api_key_str)
            retrieved_api_key_str = None

            # Pause to respect Gemini rate limits after plan (H1) generation
            self.log_message(
                f"{log_prefix} Ожидание 65 секунд после генерации плана перед созданием тела статьи...",
                "DEBUG",
            )
            if self.stop_event.wait(65):
                self.log_message(
                    f"{log_prefix} Ожидание после плана прервано сигналом остановки.",
                    "DEBUG",
                )
                return False

            def generate_random_body_prompt(current_selected_lang, current_keyword_phrase, current_original_h1_text):
                # Рандомизация объема вступления
                intro_word_count = random.randint(150, 300)  # Вступление может быть от 150 до 300 слов
                intro_paragraphs = intro_word_count // 50  # Примерное количество абзацев (по 50 слов на абзац)

                # Рандомизация количества разделов H2
                num_h2_sections = random.randint(6, 9)  # Количество основных разделов от 6 до 9

                # Исправление для h3_per_section: random.randint принимает только два аргумента (a, b)
                # Если нужно выбрать из списка, используйте random.choice
                h3_per_section_options = [2, 3, 4]
                h3_per_section = random.choice(h3_per_section_options)
                total_h3 = num_h2_sections * h3_per_section

                # Рандомизация объема текста в каждом разделе
                # Исправление для section_word_count:
                section_word_count_options = [150, 200, 250]
                section_word_count = random.choice(section_word_count_options)
                section_paragraphs = section_word_count // 50  # Примерное количество абзацев для каждого раздела

                # Рандомизация для добавления HTML-элементов (таблицы и списков)
                # Исправление для num_tables и num_lists:
                num_tables_options = [1, 2, 3]  # Используем _options для ясности, что это список для random.choice
                num_tables = random.choice(num_tables_options)  # Количество таблиц
                num_lists_options = [1, 2, 3]
                num_lists = random.choice(num_lists_options)  # Количество списков (маркированных и нумерованных)

                body_prompt_user_str = (
                    f"Язык: {selected_lang}\n. Думай и пиши только на нем, избегай символов других языков, придерживайся правильности написания и грамматики языка {selected_lang}."
                    f"Твоя главная цель — создать **максимально подробный, объемный и всесторонне раскрывающий тему казино - {self.topic_word} (Не упоминай термин 'I-Gaming' в тексте) текст-статью с уникальным заголовком H1**. "
                    f"Не экономь на словах, каждый аспект должен быть объяснен глубоко и детально."
                    f"**Критически важно: стремись к верхней границе указанных диапазонов слов, а не к нижней.** "
                    f"Напиши статью на {selected_lang} языке только на казино тематику - {self.topic_word} (Не упоминай термин 'I-Gaming' в тексте) на тему '{kw_for_prompt}' (определи что это, не пиши билиберду) под основным заголовком (H1) '{original_h1_text}'.\n\n"

                    f"**ТРЕБОВАНИЯ К HTML-СТРУКТУРЕ И РАЗМЕЩЕНИЮ ЭЛЕМЕНТОВ (согласно системным инструкциям):**\n"
                    f"Помни, что статья ОБЯЗАТЕЛЬНО должна содержать HTML-таблицы и HTML-списки, как указано в системных инструкциях. "
                    f"**КЛЮЧЕВОЕ УСЛОВИЕ:** Эти элементы (таблицы и списки) должны быть **логично интегрированы ВНУТРИ** соответствующих разделов H2 или H3. "
                    f"Они НЕ должны группироваться в конце статьи или после заключения. Размещай их после одного или нескольких абзацев текста внутри раздела, но до следующего подзаголовка, чтобы они дополняли и иллюстрировали изложенный материал.\n"
                    f"Используй для выделения ключевых слов теги <strong></strong>. Не используй Markdown-разметку (типа `**текст**`, `__текст__`, `*текст*` или `_текст_`) для этих целей.\n"
                    f"В маркированных и нумерованных списках для выделения также используй <strong></strong>, а не Markdown.\n\n"

                    f"**СОДЕРЖАНИЕ СТАТЬИ:**\n"
                    f"Язык: {selected_lang}\n. Думай и пиши только на нем, избегай символов других языков, придерживайся правильности написания и грамматики языка {selected_lang}."
                    f"1.  **Вступление:** Напиши подробное вступление ({intro_word_count} слов, {intro_paragraphs} абзацев), плавно вводящее в тему.\n"
                    f"2.  **Основные разделы (H2) и Интеграция Элементов:** Создай примерно {num_h2_sections} основных разделов с подзаголовками H2 (<h2>…</h2>). Каждый раздел должен быть объемом около {section_word_count} слов ({section_paragraphs} абзацев).\n"
                    f"    * **Во время написания этих разделов H2 (или их H3 подразделах), ты ДОЛЖЕН интегрировать HTML-элементы:**\n"
                    f"        * **Первую HTML-таблицу** постарайся разместить внутри одного из первых 2-3 разделов H2 (или его H3 подраздела), после нескольких абзацев текста.\n"
                    f"        * **HTML-маркированный список** интегрируй в один из следующих разделов H2/H3, также внутри текстового контента.\n"
                    f"        * **HTML-нумерованный список** (и вторую таблицу, если создаешь две) размести в последующих разделах H2/H3, аналогично интегрируя в текст.\n"
                    f"    * **Не добавляй все элементы в один раздел и не оставляй их на конец статьи.**\n"
                    f"3.  **Подразделы (H3):** Где это уместно, под каждым H2 добавь {h3_per_section} подзаголовков H3 (<h3>…</h3>). Каждый подраздел H3 должен содержать примерно {random.randint(150, 180)} слов ({random.randint(2, 3)} абзаца).\n"
                    f"    * **Общее количество H2 и H3:** Убедись, что общее число заголовков H2 и H3 составляет от 8 до 10 (не считая H1). Распредели {total_h3} подзаголовков H3 по разделам H2.\n\n"
                    f"4. **Подводящее итоги статьи** Напиши содержательно (1-2 абзаца, примерно 100-150 слов), подводящее итоги статьи. **HTML-таблицы и списки НЕ должны идти ПОСЛЕ итогов статьи, не употребляй слово 'Заключение'**\n\n"

                    f"**ДОПОЛНИТЕЛЬНЫЕ ТРЕБОВАНИЯ К ТЕКСТУ:**\n"
                    f"Язык: {selected_lang}\n. Думай и пиши только на нем, избегай символов других языков, придерживайся правильности написания и грамматики языка {selected_lang}."
                    f"-   **Абзацы:** Каждый абзац во всей статье должен состоять минимум из 3–5 полных предложений.\n"
                    f"-   **Ключевая фраза:** Включи ключевую фразу '{keyword_phrase}' 1–2 раза **внутри тегов <p>…</p>** так, чтобы она выглядела естественно в тексте.\n"
                    f"-   **Оригинальность:** Не повторяй основной H1 в других заголовках или абзацах. Текст должен быть оригинальным, с уникальной структурой заголовков.\n"
                    f"-   **Наполнение:** Проверь, чтобы не было пустых или слишком коротких разделов.\n"
                    f"-   **Форматирование:** Для выделения жирным шрифтом всегда используй теги `<strong></strong>`. Для курсивного начертания всегда используй теги `<em></em>`. Не используй Markdown-разметку.\n"
                    f"-   **Не нумеруй** заголовки цифрами типа '1.', '2.', '3.' и т.д. Заголовки должны быть текстовыми, без префиксов из чисел.\n"
                    f"-   **Перепроверь** чтобы не было ошибки с заголовками и они не повторялись одинаково между собой, например два H1. H1 должен быть только один в тексте.\n"
                )

                body_prompt_system_str = (
                    f"Язык: {selected_lang}\n. Думай и пиши только на нем, избегай символов других языков, придерживайся правильности написания и грамматики языка {selected_lang}."
                    f"Ты опытный SEO-копирайтер. Твоя задача — сгенерировать **высококачественный, объёмный и подробный** HTML-контент для сайта. "
                    f"**Строго следуй инструкциям пользователя по содержанию, объему текста (вступление, H2, H3, заключение) и количеству абзацев.**\n\n"

                    f"**ГЛАВНОЕ ТРЕБОВАНИЕ: ГЕНЕРАЦИЯ И ИНТЕГРАЦИЯ HTML-ЭЛЕМЕНТОВ:**\n"
                    f"1.  **ОБЯЗАТЕЛЬНО СОЗДАЙ И ВКЛЮЧИ В ТЕКСТ СТАТЬИ:**\n"
                    f"    * **Одну или две (1-2) HTML-таблицы.** Таблицы должны быть информативными и релевантными теме. Структура таблицы: `<table><thead><tr><th>Заголовок1</th><th>Заголовок2</th>...</tr></thead><tbody><tr><td>Данные1</td><td>Данные2</td>...</tr><tr><td>Данные3</td><td>Данные4</td>...</tr>...</tbody></table>`. Заголовки таблицы (теги `<th>`) ОБЯЗАТЕЛЬНО внутри `<thead>`. Данные таблицы (теги `<td>`) ОБЯЗАТЕЛЬНО внутри `<tbody>`.\n"
                    f"    * **Один (1) HTML-маркированный список.** Формат: `<ul><li>Пункт 1</li><li>Пункт 2</li>...</ul>`.\n"
                    f"    * **Один (1) HTML-нумерованный список.** Формат: `<ol><li>Пункт A</li><li>Пункт B</li>...</ol>`.\n"
                    f"2.  **КРИТИЧЕСКИ ВАЖНОЕ ПРАВИЛО РАЗМЕЩЕНИЯ ЭЛЕМЕНТОВ:**\n"
                    f"    * **ИНТЕГРИРУЙ** эти HTML-элементы (таблицы и списки) **НЕПОСРЕДСТВЕННО ВНУТРЬ ТЕКСТА** соответствующих разделов H2 или H3, как указано в пользовательском промпте (например, таблица в одном из первых H2, списки в последующих). Они должны быть окружены абзацами текста этих разделов.\n"
                    f"    * **АБСОЛЮТНО ЗАПРЕЩЕНО:** Размещать таблицы или списки ПОСЛЕ заключительного раздела статьи. Они также НЕ должны быть сгруппированы вместе в конце статьи или перед заключением.\n"
                    f"    * Элементы должны логически дополнять и иллюстрировать текст раздела, в который они вставлены.\n\n"

                    f"**ПРАВИЛА ФОРМАТИРОВАНИЯ HTML:**\n"
                    f"-   **Только HTML:** Весь контент должен быть представлен в виде чистой HTML-разметки. Ответ должен начинаться непосредственно с первого HTML-тега основного контента (например, `<p>` из вступления или `<h2>` первого раздела, если вступление отсутствует в ТЗ) и заканчиваться последним HTML-тегом (например, `</p>` из заключения).\n"  # Уточнено начало
                    f"-   **Без оберток и посторонних элементов:** НЕ включай `<!DOCTYPE>`, `<html>`, `<head>`, `<body>`. НЕ включай в свой ответ заголовок H1 (он будет добавлен отдельно). НЕ генерируй таблицы содержания (ToC) или любые другие навигационные блоки, если это не указано явно.\n"  # Добавлен запрет на ToC
                    f"-   **НЕ оборачивай HTML-ответ** в блоки ```html ... ```, ``` ```, или любые другие markdown-конструкции.\n"
                    f"-   **Заголовки:** Используй `<h2>` и `<h3>` для заголовков.\n"
                    f"-   **Абзацы:** Каждый абзац должен быть обернут в теги `<p>…</p>`.\n"
                    f"-   **Выделение текста:**\n"
                    f"    * Для выделения жирным шрифтом **ВСЕГДА** используй теги `<strong></strong>`.\n"
                    f"    * Для курсивного начертания **ВСЕГДА** используй теги `<em></em>`.\n"
                    f"    * **ЗАПРЕЩЕНО** использовать Markdown для выделения (например, `**текст**`, `__текст__`, `*текст*`, `_текст_`). Это касается как основного текста, так и текста внутри списков и таблиц.\n"
                    f"-   **Списки и таблицы:** Убедись, что все HTML-теги (`<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`, `<ul>`, `<ol>`, `<li>`) корректно открыты и закрыты. Каждый элемент `<li>` должен быть отдельным пунктом. Таблица должна быть минимум 2x2.\n\n"

                    f"**СТРУКТУРА КОНТЕНТА (краткое напоминание из пользовательского запроса):**\n"
                    f"-   Вступление, затем разделы H2/H3, затем заключение, как указано пользователем.\n"
                    f"-   **Интеграция HTML-элементов (таблиц, списков) должна происходить ВНУТРИ разделов H2/H3, ДО заключения.**\n\n"

                    f"**ФИНАЛЬНАЯ ПРОВЕРКА ПЕРЕД ОТВЕТОМ (выполни мысленно):**\n"
                    f"1. Все ли запрошенные HTML-элементы (таблицы, списки) присутствуют?\n"
                    f"2. Интегрированы ли они ВНУТРИ различных разделов H2/H3, а НЕ в конце статьи и НЕ после заключения?\n"
                    f"3. Соответствует ли весь текст HTML-форматированию и другим требованиям (отсутствие Markdown, правильные теги и т.д.)?\n"
                    f"4. Начинается ли ответ с `<p>` или `<h2>` и не содержит ли он H1 или блоков ToC?\n"
                    f"5. Нету ли в тексте символов которых нету в языке {selected_lang}, все ли правильно ?\n"
                    f"**Если какой-либо из этих пунктов нарушен, исправь свой ответ ПЕРЕД тем, как его предоставить.**"
                )

                return body_prompt_user_str, body_prompt_system_str

                # Вызываем функцию и получаем строки промптов

            body_prompt_user, body_prompt_system = generate_random_body_prompt(selected_lang, kw_for_prompt,
                                                                               original_h1_text)
            # --- КОНЕЦ ВАЖНОГО ИЗМЕНЕНИЯ ---

            if retrieved_api_key_str is None:
                try:
                    retrieved_api_key_str = self.api_key_queue.get(block=True, timeout=20)
                    with self.api_stats_lock:
                        self.api_key_usage_stats[retrieved_api_key_str] = self.api_key_usage_stats.get(retrieved_api_key_str, 0) + 1
                    key_short_display = f"...{retrieved_api_key_str[-5:]}" if len(retrieved_api_key_str) > 5 else retrieved_api_key_str
                except Empty:
                    self.log_message(f"{log_prefix} Ошибка: Таймаут получения API ключа из очереди.", "ERROR")
                    return False

            if self.provider_var.get() == PROVIDER_OPENAI:
                article_body_raw_from_api = self.call_openai_api(openai_client,
                                                                 [{"role": "system", "content": body_prompt_system},
                                                                  {"role": "user", "content": body_prompt_user}],
                                                                 retrieved_api_key_str)
            else:
                prompt_text = body_prompt_system + "\n" + body_prompt_user + "\nСделай текст не слишком длинным и используй настоящую, проверяемую информацию; избегай вымышленных фактов."
                self.log_message(
                    f"{log_prefix} [BODY] Отправка запроса к Gemini, длина промпта {len(prompt_text)}",
                    "DEBUG",
                )
                article_body_raw_from_api = self.call_gemini_api(
                    retrieved_api_key_str, prompt_text, context=f"BODY {task_id}"
                )
                if article_body_raw_from_api == "GEMINI_KEY_EXHAUSTED":
                    self._mark_gemini_key_exhausted(retrieved_api_key_str)
                    key_marked_as_bad_in_this_task = True
                    return False
                self.log_message(
                    f"{log_prefix} [BODY] Получено {len(article_body_raw_from_api)} символов",
                    "DEBUG",
                )

            # ... (остальной код вашего метода)

            if article_body_raw_from_api == "INVALID_API_KEY_ERROR":  # Обработка невалидного ключа
                self.log_message(f"{log_prefix} API ключ {key_short_display} невалиден (тело). Обработка...", "ERROR")
                self.after(0, self._update_gui_and_log_bad_key, retrieved_api_key_str)
                key_marked_as_bad_in_this_task = True
                return False

            with self.api_key_statuses_lock:  # Проверка статуса ключа
                current_key_status_after_body = self.api_key_statuses.get(retrieved_api_key_str, {}).get("status",
                                                                                                         "active")
            if current_key_status_after_body != "active" and not key_went_to_cooldown_in_this_task:
                key_went_to_cooldown_in_this_task = True
                self.log_message(
                    f"{log_prefix} Ключ {key_short_display} перешел в '{current_key_status_after_body}' после тела.",
                    "INFO")

            if not article_body_raw_from_api or self.stop_event.is_set():
                self.log_message(
                    f"{log_prefix} Тело не сгенерировано ({'пусто' if not article_body_raw_from_api else 'остановлено'}).",
                    "ERROR" if not article_body_raw_from_api else "WARNING")
                return False

            # Начальная очистка от Markdown-оберток кода
            cleaned_api_response_for_wrappers = article_body_raw_from_api
            # ... (логика удаления ```html ... ``` как в вашем коде) ...
            cleaned_api_response_for_wrappers = re.sub(r'^\s*<p>\s*```html\s*</p>\s*\n?', '',
                                                       cleaned_api_response_for_wrappers,
                                                       flags=re.IGNORECASE | re.MULTILINE)
            cleaned_api_response_for_wrappers = re.sub(r'\n?\s*<p>\s*```\s*</p>\s*$', '',
                                                       cleaned_api_response_for_wrappers,
                                                       flags=re.IGNORECASE | re.MULTILINE)
            cleaned_api_response_for_wrappers = re.sub(r'^\s*```html\s*\n?', '', cleaned_api_response_for_wrappers,
                                                       flags=re.IGNORECASE)
            cleaned_api_response_for_wrappers = re.sub(r'\n?\s*```\s*$', '', cleaned_api_response_for_wrappers,
                                                       flags=re.IGNORECASE)

            article_to_clean_further = cleaned_api_response_for_wrappers.strip()

            # Основная очистка HTML с помощью _clean_html_structure (многопроходного)
            cleaned_body_html_str = self._clean_html_structure(article_to_clean_further, log_prefix)

            # --- НАЧАЛО НОВОЙ DOM-ОРИЕНТИРОВАННОЙ ЛОГИКИ ---
            # Оборачиваем в div для безопасной работы с BeautifulSoup, если контент не имеет единого корня
            # или для упрощения поиска (.div)
            soup_container = BeautifulSoup(f"<div>{cleaned_body_html_str}</div>", 'html.parser')
            actual_body_root = soup_container.div

            # Удаление H1 из тела ответа API, если он там есть (работаем с DOM)
            if actual_body_root and actual_body_root.contents:
                first_significant_child = None
                for node in actual_body_root.contents:
                    if hasattr(node, 'name') and node.name:  # Ищем первый тег
                        first_significant_child = node
                        break

                if first_significant_child and first_significant_child.name == 'h1':
                    h1_text_in_body = first_significant_child.get_text(strip=True)
                    # Сравнение с original_h1_text для подтверждения, что это "эхо" H1
                    if h1_text_in_body.lower() == original_h1_text.lower() or \
                            (len(h1_text_in_body) > 3 and len(h1_text_in_body) < (
                                    len(original_h1_text) + 30) and original_h1_text.lower() in h1_text_in_body.lower()):  # Более мягкое сравнение
                        self.log_message(
                            f"{log_prefix} Обнаружен и удален тег H1 ('{h1_text_in_body[:60]}...') из тела ответа API (DOM).",
                            "INFO")
                        first_significant_child.extract()

            if actual_body_root and actual_body_root.find('h1'):
                self.log_message(
                    f"{log_prefix} Обнаружен лишний тег H1 в тексте после очиcтки. Статья будет пересоздана.",
                    "WARNING",
                )
                return False

            toc_items = []
            # Генерируемый нами H1
            main_h1_id = f"t{heading_id_counter}"
            toc_items.append({"level": 1, "text": original_h1_text, "id": main_h1_id})
            generated_h1_html = f'<h1 id="{main_h1_id}">{original_h1_text}</h1>'
            heading_id_counter += 1

            # Обработка H2/H3 в DOM для ToC и добавления ID
            if actual_body_root:
                for h_tag in actual_body_root.find_all(['h2', 'h3']):
                    level = int(h_tag.name[1:])
                    heading_text_clean = h_tag.get_text(strip=True)
                    if not heading_text_clean: heading_text_clean = f"Подзаголовок {level}"

                    h_id = f"t{heading_id_counter}"
                    h_tag['id'] = h_id  # Добавляем/заменяем ID прямо в теге
                    toc_items.append({"level": level, "text": heading_text_clean, "id": h_id})
                    heading_id_counter += 1

                    # Вставка ссылок (DOM-based, под H1, до первого H2, с улучшенной рандомизацией)
                    link_inserted_flag_ref = [link_inserted_flag]
                    if actual_body_root and target_link and keyword_phrase and not link_inserted_flag_ref[0]:

                        candidate_tags_in_first_section = []
                        # Собираем теги <p> и <li> только до первого <h2> в actual_body_root
                        for child_node in actual_body_root.children:
                            if not hasattr(child_node, 'name'):
                                continue
                            if child_node.name == 'h2':
                                break

                            if child_node.name in ['p', 'li']:
                                candidate_tags_in_first_section.append(child_node)
                            # ИСПРАВЛЕНИЕ ЗДЕСЬ:
                            elif hasattr(child_node, 'find_all') and child_node.find_all(['p', 'li'], recursive=False):
                                for sub_tag in child_node.find_all(['p', 'li'], recursive=False):
                                    if hasattr(sub_tag, 'find') and not sub_tag.find('h2'):
                                        candidate_tags_in_first_section.append(sub_tag)

                        if not candidate_tags_in_first_section:
                            self.log_message(
                                f"{log_prefix} Не найдено подходящих тегов (<p> или <li>) под H1 (до первого H2) для вставки ссылки.",
                                "WARNING")
                        else:
                            temp_shuffled_candidates_for_search = list(candidate_tags_in_first_section)
                            random.shuffle(temp_shuffled_candidates_for_search)

                            for tag in temp_shuffled_candidates_for_search:
                                if link_inserted_flag_ref[0]: break

                                existing_keyword_links = tag.find_all('a', string=re.compile(re.escape(keyword_phrase),
                                                                                             re.IGNORECASE))
                                already_linked_correctly = False
                                for existing_link_tag in existing_keyword_links:
                                    if existing_link_tag.get('href') == target_link:
                                        already_linked_correctly = True
                                        break
                                if already_linked_correctly:
                                    self.log_message(
                                        f"{log_prefix} Ссылка для '{keyword_phrase}' на '{target_link}' уже есть в <{tag.name}> (зона под H1).",
                                        "DEBUG")
                                    link_inserted_flag_ref[0] = True
                                    break

                                text_nodes = tag.find_all(string=True, recursive=True)
                                for text_node in text_nodes:
                                    if text_node.parent.name == 'a': continue

                                    node_text = str(text_node)
                                    match = re.search(r'(?i)\b(' + re.escape(keyword_phrase) + r')\b', node_text)
                                    if match:
                                        original_kw_casing = match.group(1)
                                        text_before_keyword = node_text[:match.start()]
                                        text_after_keyword = node_text[match.end():]

                                        link_tag_obj = soup_container.new_tag('a', href=target_link)
                                        link_tag_obj.string = original_kw_casing

                                        current_node_to_operate_on = text_node
                                        if text_after_keyword:
                                            current_node_to_operate_on.insert_after(NavigableString(text_after_keyword))
                                        current_node_to_operate_on.insert_after(link_tag_obj)
                                        if text_before_keyword:
                                            current_node_to_operate_on.insert_after(
                                                NavigableString(text_before_keyword))
                                        current_node_to_operate_on.extract()

                                        link_inserted_flag_ref[0] = True
                                        self.log_message(
                                            f"{log_prefix} Ссылка для '{keyword_phrase}' успешно вставлена в <{tag.name}> (зона под H1, ключ найден).",
                                            "INFO")
                                        break
                                if link_inserted_flag_ref[0]: break

                    link_inserted_flag = link_inserted_flag_ref[0]

                    if actual_body_root and target_link and keyword_phrase and not link_inserted_flag:
                        if 'candidate_tags_in_first_section' not in locals() or not candidate_tags_in_first_section:
                            candidate_tags_in_first_section = []
                            for child_node in actual_body_root.children:
                                if not hasattr(child_node, 'name'): continue
                                if child_node.name == 'h2': break
                                if child_node.name in ['p', 'li']:
                                    candidate_tags_in_first_section.append(child_node)
                                # ИСПРАВЛЕНИЕ ЗДЕСЬ (второе место):
                                elif hasattr(child_node, 'find_all') and child_node.find_all(['p', 'li'],
                                                                                             recursive=False):
                                    for sub_tag in child_node.find_all(['p', 'li'], recursive=False):
                                        if hasattr(sub_tag, 'find') and not sub_tag.find('h2'):
                                            candidate_tags_in_first_section.append(sub_tag)

                        if candidate_tags_in_first_section:
                            self.log_message(
                                f"{log_prefix} Ключ '{keyword_phrase}' не найден в зоне под H1. Принудительная рандомная вставка.",
                                "WARNING")

                            target_tag_for_forced_insertion = None
                            if len(candidate_tags_in_first_section) > 2:
                                middle_candidates = list(candidate_tags_in_first_section[1:-1])
                                if middle_candidates:  # Убедимся, что список не пуст после среза
                                    random.shuffle(middle_candidates)
                                    target_tag_for_forced_insertion = middle_candidates[0]

                            if not target_tag_for_forced_insertion and candidate_tags_in_first_section:
                                temp_candidates_for_random_choice = list(
                                    candidate_tags_in_first_section)  # Копируем для перемешивания
                                random.shuffle(temp_candidates_for_random_choice)
                                target_tag_for_forced_insertion = temp_candidates_for_random_choice[0]

                            if target_tag_for_forced_insertion:
                                link_element = soup_container.new_tag('a', href=target_link)
                                link_element.string = keyword_phrase

                                direct_text_nodes = [tn for tn in target_tag_for_forced_insertion.contents if
                                                     isinstance(tn, NavigableString) and str(tn).strip()]

                                inserted_strategically = False
                                if direct_text_nodes:
                                    text_node_to_modify = random.choice(direct_text_nodes)
                                    original_text = str(text_node_to_modify)
                                    words = original_text.split()

                                    if len(words) > 1:
                                        position_to_insert_after_word_index = 0

                                        text_part_before_link = " ".join(
                                            words[:position_to_insert_after_word_index + 1])
                                        text_part_after_link = " ".join(words[position_to_insert_after_word_index + 1:])

                                        replacement_parts = []
                                        replacement_parts.append(NavigableString(text_part_before_link + " "))
                                        replacement_parts.append(link_element)
                                        if text_part_after_link:
                                            replacement_parts.append(NavigableString(" " + text_part_after_link))

                                        text_node_to_modify.replace_with(*replacement_parts)
                                        inserted_strategically = True
                                        self.log_message(
                                            f"{log_prefix} Ссылка принудительно вставлена после первого слова в тексте тега <{target_tag_for_forced_insertion.name}>.",
                                            "INFO")
                                    elif words:
                                        text_node_to_modify.replace_with(NavigableString(words[0] + " "), link_element)
                                        inserted_strategically = True
                                        self.log_message(
                                            f"{log_prefix} Ссылка принудительно вставлена после единственного слова в теге <{target_tag_for_forced_insertion.name}>.",
                                            "INFO")

                                if not inserted_strategically:
                                    target_tag_for_forced_insertion.insert(0, link_element)
                                    target_tag_for_forced_insertion.insert(1, NavigableString(" "))
                                    self.log_message(
                                        f"{log_prefix} Ссылка принудительно вставлена в начало тега <{target_tag_for_forced_insertion.name}> (запасной вариант).",
                                        "INFO")

                                link_inserted_flag = True
                            else:
                                self.log_message(
                                    f"{log_prefix} Не удалось выбрать тег для принудительной вставки ссылки (зона под H1).",
                                    "ERROR")
                        else:
                                self.log_message(
                                    f"{log_prefix} Нет подходящих тегов в зоне под H1 для принудительной вставки ссылки.",
                                    "ERROR")
            if actual_body_root and keyword_phrase:
                body_text_lower = actual_body_root.get_text(separator=" ", strip=True).lower()
                if keyword_phrase.lower() not in body_text_lower:
                    candidate_tags = [t for t in actual_body_root.find_all(['p', 'li']) if t.get_text(strip=True)]
                    if candidate_tags:
                        tgt = random.choice(candidate_tags)
                        direct_nodes = [tn for tn in tgt.contents if isinstance(tn, NavigableString) and str(tn).strip()]
                        if direct_nodes:
                            node_to_mod = random.choice(direct_nodes)
                            words = str(node_to_mod).split()
                            if len(words) > 1:
                                idx = random.randint(0, len(words) - 2)
                            else:
                                idx = 0
                            before = " ".join(words[:idx + 1])
                            after = " ".join(words[idx + 1:])
                            replacement = []
                            replacement.append(NavigableString(before + " "))
                            replacement.append(NavigableString(keyword_phrase))
                            if after:
                                replacement.append(NavigableString(" " + after))
                            node_to_mod.replace_with(*replacement)
                        else:
                            words = tgt.get_text(" ", strip=True).split()
                            if words:
                                idx = random.randint(0, max(0, len(words) - 2))
                                words.insert(idx + 1, keyword_phrase)
                                for child in list(tgt.children):
                                    child.extract()
                                tgt.append(NavigableString(" ".join(words)))
                        self.log_message(f"{log_prefix} Ключевая фраза добавлена в тег <{tgt.name}>.", "INFO")

            # Получаем итоговую строку HTML из обработанного actual_body_root
            article_body_html_processed = "".join(
                str(content_node) for content_node in actual_body_root.contents) if actual_body_root else ""
            # --- КОНЕЦ НОВОЙ DOM-ОРИЕНТИРОВАННОЙ ЛОГИКИ ---

            if not self._has_text_under_h1(actual_body_root):
                self.log_message(
                    f"{log_prefix} Нет текста под H1. Статья будет пересоздана.",
                    "WARNING",
                )
                return False

            # Проверка корректности оглавления
            total_toc_len = sum(len(it["text"]) for it in toc_items)
            if any(len(it["text"]) > MAX_TOC_ITEM_CHARS for it in toc_items) or total_toc_len > MAX_TOC_TOTAL_CHARS:
                self.log_message(
                    f"{log_prefix} Некорректное оглавление (слишком длинные пункты). Статья будет пересоздана.",
                    "WARNING",
                )
                return False

            # Формирование оглавления (ToC)
            toc_li_html_parts = []
            for item in toc_items:  # toc_items уже заполнен
                toc_li_html_parts.append(f'<li><a href="#{item["id"]}">{item["text"]}</a></li>')  # Корректный формат
            toc_li_html = "".join(toc_li_html_parts)
            chosen_toc_bg_color = random.choice(self.article_toc_background_colors)
            toc_div = f'''<div id="texter" style="background: {chosen_toc_bg_color};border: 1px solid #aaa;display: table;margin-bottom: 1em;padding: 1em;width: 350px;">
                                <p class="toctitle" style="font-weight: 700; text-align: center">
                    </p>
                    <ul class="toc_list">{toc_li_html}</ul>
            </div>'''

            # Собираем финальный контент
            final_body_content_parts = [toc_div, generated_h1_html, article_body_html_processed]
            final_body_content = "\n".join(final_body_content_parts)

            # filepath_to_save was determined during H1 generation
            output_format = self.output_format_var.get()

            with open(filepath_to_save, "w", encoding="utf-8") as f:
                f.write(final_body_content)
            try:
                size_written = os.path.getsize(filepath_to_save)
                if size_written < MIN_ARTICLE_SIZE:
                    os.remove(filepath_to_save)
                    self.log_message(
                        f"{log_prefix} Файл слишком маленький и будет пересоздан: {os.path.basename(filepath_to_save)}",
                        "WARNING",
                    )
                    return False
                if size_written > MAX_ARTICLE_SIZE:
                    os.remove(filepath_to_save)
                    self.log_message(
                        f"Удален слишком большой файл: {os.path.basename(filepath_to_save)}",
                        "WARNING",
                    )
                    return False
            except Exception as e_size:
                self.log_message(
                    f"{log_prefix} Ошибка проверки размера файла {os.path.basename(filepath_to_save)}: {e_size}",
                    "ERROR",
                )
                return False

            self.log_message(f"{log_prefix} Файл ('{output_format}') сохранен: {os.path.basename(filepath_to_save)}")
            if target_link and not link_inserted_flag:
                self.log_message(f"{log_prefix} ВНИМАНИЕ: Ссылка для '{keyword_phrase}' НЕ БЫЛА ВСТАВЛЕНА.", "WARNING")
            with self.success_lock:
                self.successful_task_ids.add(task_id)
            with self.output_count_lock:
                self.output_file_counter += 1
            # record progress for this keyword/index
            self.update_progress(task_id, keyword_phrase, task_num_for_keyword)
            with self.previous_h1_lock:
                self.previous_h1_text = original_h1_text
            success = True
            return True
        except Exception as e_gen:
            self.log_message(f"{log_prefix} Общая ошибка генерации статьи: {type(e_gen).__name__} - {e_gen}", "ERROR")
            self.log_message(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return False
        finally:  # Возвращение ключа в очередь и освобождение пути
            if 'filepath_to_save' in locals():
                self.release_filepath(filepath_to_save)
            if retrieved_api_key_str:
                if key_marked_as_bad_in_this_task:
                    pass
                elif key_went_to_cooldown_in_this_task:
                    self.log_message(f"{log_prefix} Ключ {key_short_display} ушел в cooldown, не возвращается.",
                                     "DEBUG")
                else:
                    with self.api_key_statuses_lock:
                        final_status_check = self.api_key_statuses.get(retrieved_api_key_str, {}).get("status", "error")
                    if final_status_check == "active":
                        self.api_key_queue.put(retrieved_api_key_str)
                    else:
                        self.log_message(
                            f"{log_prefix} Ключ {key_short_display} статус '{final_status_check}', не возвращен.",
                            "DEBUG")

    # ================================================================================
    # КОНЕЦ ОБНОВЛЕННОГО МЕТОДА generate_single_article_content
    # ================================================================================

    def worker_thread(self):
        try:
            while not self.stop_event.is_set():
                task_payload = None
                task_dequeued_this_iteration = False
                try:
                    task_payload = self.task_creation_queue.get(timeout=1)
                    task_dequeued_this_iteration = True
                    if self.stop_event.is_set():
                        self.log_message(
                            f"Поток {threading.current_thread().name} получил задачу, но обнаружен сигнал остановки.",
                            "DEBUG",
                        )
                        break
                    task_id, keyword, num_for_kw, total_for_kw, global_num, total_global = task_payload
                    self.generate_single_article_content(
                        task_id,
                        keyword,
                        num_for_kw,
                        total_for_kw,
                        global_num,
                        total_global,
                    )
                except Empty:
                    if self.task_creation_queue.empty():
                        self.log_message(
                            f"Поток {threading.current_thread().name} завершает работу: очередь задач пуста.",
                            "DEBUG",
                        )
                        break
                    if self.stop_event.is_set():
                        self.log_message(
                            f"Поток {threading.current_thread().name} завершает работу по сигналу остановки.",
                            "DEBUG",
                        )
                        break
                    continue
                except Exception as e_worker_critical:
                    task_id_for_log = task_payload[0] if task_payload and isinstance(task_payload, tuple) and len(
                        task_payload) > 0 else "UNKNOWN_TASK_ID"
                    self.log_message(
                        f"Критическая ошибка в потоке {threading.current_thread().name} при обработке задачи {task_id_for_log}: {type(e_worker_critical).__name__} - {e_worker_critical}",
                        "CRITICAL")
                    self.log_message(f"Traceback: {traceback.format_exc()}", "DEBUG")
                finally:
                    if task_dequeued_this_iteration:
                        self.task_creation_queue.task_done()
        finally:
            GLOBAL_THREAD_SEMAPHORE.release()
            self.log_message(
                f"Рабочий поток ({threading.current_thread().name}) завершил свою работу.",
                "DEBUG",
            )

    def process_all_keywords(self):
        keywords_input_list_local = []
        total_expected = 0
        try:
            with open(self.keywords_file_path.get(), "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if self.stop_event.is_set():
                        break
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("!===+"):
                        lang_val = line[len("!===+"):].strip()
                        if lang_val:
                            self.generation_language_var.set(lang_val)
                            self.log_message(f"Автообнаружен язык из файла: {lang_val}")
                        continue
                    if line.startswith("!==+"):
                        link_val = line[len("!==+"):].strip()
                        if link_val:
                            self.target_link_var.set(link_val)
                            self.log_message(f"Автообнаружена ссылка из файла: {link_val}")
                        continue
                    if line.startswith("!====+"):
                        topic_val = line.split("+", 1)[-1].strip()
                        if topic_val:
                            self.topic_word = topic_val
                            self.log_message(f"Автообнаружена тема из файла: {topic_val}")
                        continue
                    parts = line.split("\t")
                    if len(parts) == 2:
                        keyword = parts[0].strip()
                        quantity_str = parts[1].strip()
                        if not keyword:
                            self.log_message(f"Пустое ключевое слово в строке {line_num}. Пропуск.", "WARNING")
                            continue
                        try:
                            qty = int(quantity_str)
                            if qty > 0:
                                keywords_input_list_local.append({"keyword": keyword, "quantity": qty})
                                total_expected += qty
                        except ValueError:
                            self.log_message(
                                f"Некорректное количество для '{keyword}' (строка {line_num}): '{quantity_str}'.",
                                "ERROR")
        except Exception as e:
            self.log_message(f"Ошибка чтения файла ключевых слов: {e}", "ERROR")
            self.after(0, self.set_ui_for_generation, False)
            if self.project_slot_acquired:
                GLOBAL_PROJECT_SEMAPHORE.release()
                self.project_slot_acquired = False
            return
        if self.stop_event.is_set() or not keywords_input_list_local:
            self.after(0, self.set_ui_for_generation, False)
            if self.project_slot_acquired:
                GLOBAL_PROJECT_SEMAPHORE.release()
                self.project_slot_acquired = False
            return

        self.send_telegram_notification(f"Генерируем статьи в кол-ве: {total_expected}")

        self.output_file_counter = self._ensure_output_file_count(force=True)
        kw_results = {}
        for item in keywords_input_list_local:
            if self.stop_event.is_set():
                break
            keyword = item["keyword"]
            quantity = item["quantity"]

            kw_start_count = self.output_file_counter
            attempt = 0

            while (self.output_file_counter - kw_start_count) < quantity and not self.stop_event.is_set():
                self._initial_check_and_revive_keys()
                if self.api_key_queue.empty():
                    self.log_message(
                        f"Нет активных API ключей для '{keyword}'. Ожидание восстановления...",
                        "WARNING",
                    )
                    time.sleep(1)
                    continue

                attempt += 1
                if MAX_KEYWORD_ATTEMPTS and attempt > MAX_KEYWORD_ATTEMPTS:
                    missing = quantity - (self.output_file_counter - kw_start_count)
                    self.log_message(
                        f"Достигнут лимит попыток для '{keyword}'. Осталось {missing} файлов. Переходим далее.",
                        "WARNING",
                    )
                    break

                remaining = quantity - (self.output_file_counter - kw_start_count)
                self.log_message(f"Генерация для '{keyword}' - попытка {attempt}, осталось {remaining} файлов")
                self.all_task_definitions = []
                task_id_set = set()
                for i in range(remaining):
                    base_id = re.sub(r'[^a-zA-Z0-9\-_]', '', keyword.replace(" ", "_"))
                    cand = f"{base_id}_{i + 1}_try{attempt}"
                    orig = cand
                    suffix_counter = 1
                    while cand in task_id_set:
                        cand = f"{orig}_dup{suffix_counter}"
                        suffix_counter += 1
                    task_id_set.add(cand)
                    self.all_task_definitions.append({
                        "id": cand,
                        "keyword": keyword,
                        "num_for_kw": i + 1,
                        "total_for_kw": remaining,
                    })

                self.successful_task_ids.clear()
                self._run_task_batch()
                # Пересчитываем количество файлов после каждой попытки,
                # чтобы гарантировать точный учет созданных статей.
                self.output_file_counter = self._ensure_output_file_count(force=True)

            # Финальное пересчитывание после выхода из цикла
            self.output_file_counter = self._ensure_output_file_count(force=True)
            kw_final_count = self.output_file_counter - kw_start_count
            kw_results[keyword] = kw_final_count
            self.log_message(f"Файлов для '{keyword}': {kw_final_count}/{quantity}", "INFO")

        # Сводка по количеству файлов для каждого ключа
        for item in keywords_input_list_local:
            kw = item["keyword"]
            qty = item["quantity"]
            done = kw_results.get(kw, 0)
            self.log_message(f"Итого для '{kw}': {done}/{qty}")

        self.output_file_counter = self._ensure_output_file_count(force=True)
        final_count = self.output_file_counter
        self.log_message("Генерация завершена (или остановлена).", "INFO")
        self.log_message(f"Итоговое количество файлов: {final_count} из {total_expected}", "INFO")
        self.send_telegram_notification(
            f"Генерация завершена. Итоговое количество файлов: {final_count} из {total_expected}"
        )
        self.after(0, self.set_ui_for_generation, False)
        if self.project_slot_acquired:
            GLOBAL_PROJECT_SEMAPHORE.release()
            self.project_slot_acquired = False

    def _monitor_task_queue_and_threads(self, threads_list):
        while not self.task_creation_queue.empty() and not self.stop_event.is_set():
            time.sleep(0.05)
        if self.stop_event.is_set():
            self.log_message("Мониторинг: процесс остановлен. Завершаем без ожидания потоков.", "WARNING")
            return
        if self.task_creation_queue.empty():
            self.log_message("Мониторинг: все задачи переданы. Ожидание выполнения...", "INFO")
        self.log_message(f"Ожидание завершения {len(threads_list)} рабочих потоков...", "INFO")
        for t in threads_list:
            while t.is_alive():
                t.join(timeout=1)
                if self.stop_event.is_set():
                    break
            if t.is_alive():
                self.log_message(f"Поток {t.name} не завершился корректно.", "WARNING")
        if self.stop_event.is_set():
            self.log_message("Проход генерации был прерван.", "WARNING")
        else:
            self.log_message("Все рабочие потоки текущего прохода завершили работу.", "INFO")

    def _open_api_key_status_window(self):
        if hasattr(self, 'api_status_win') and self.api_status_win is not None and self.api_status_win.winfo_exists():
            self.api_status_win.focus()
        else:
            self.api_status_win = ApiKeyStatusWindow(self)
        if self.api_status_win and self.api_status_win.winfo_exists():
            self.api_status_win.lift()
            self.api_status_win.attributes('-topmost', True)
            self.api_status_win.after(100, lambda: self.api_status_win.attributes('-topmost', False))

    def _open_help_window(self):
        if hasattr(self, 'help_win') and self.help_win is not None and self.help_win.winfo_exists():
            self.help_win.focus()
        else:
            self.help_win = HelpWindow(self)
        if self.help_win and self.help_win.winfo_exists():
            self.help_win.lift()
            self.help_win.attributes('-topmost', True)
            self.help_win.after(100, lambda: self.help_win.attributes('-topmost', False))


# Класс ApiKeyStatusWindow остается без изменений, предполагается, что он корректен
class ApiKeyStatusWindow(ctk.CTkToplevel):
    def __init__(self, master_app):
        super().__init__(master_app)
        self.master_app = master_app
        self.title("Статусы API Ключей")
        self.geometry("1000x550")

        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        ctk.CTkLabel(self.main_frame, text="Состояние API ключей OpenAI:",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(0, 10))

        self.scrollable_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.scrollable_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.data_row_map = {}
        self.header_labels = []
        self._create_table_headers()

        self.refresh_interval_ms = 1000
        self._after_id_refresh = None

        self.populate_table_data()  # Первоначальное заполнение
        self._schedule_refresh()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_table_headers(self):
        headers = ["Ключ (хвост)", "Статус", "Запросов (Ост/Лимит)", "Сброс Запросов",
                   "Токенов (Ост/Лимит)", "Сброс Токенов", "Обновлено", "Ошибка"]
        for col, header_text in enumerate(headers):
            header_label = ctk.CTkLabel(self.scrollable_frame, text=header_text, font=ctk.CTkFont(weight="bold"),
                                        anchor="w")
            header_label.grid(row=0, column=col, padx=5, pady=(5, 10), sticky="ew")
            self.header_labels.append(header_label)
            self.scrollable_frame.grid_columnconfigure(col, weight=1 if col == 7 else 0)  # Колонка ошибок растягивается

    def populate_table_data(self):
        if not (hasattr(self, 'scrollable_frame') and self.scrollable_frame.winfo_exists()):
            return

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        statuses_copy = {}
        with self.master_app.api_key_statuses_lock:
            # Сортируем ключи для консистентного отображения, если они добавляются/удаляются
            sorted_keys_from_master = sorted(self.master_app.api_key_statuses.keys())
            for key in sorted_keys_from_master:
                if key in self.master_app.api_key_statuses:  # Доп. проверка на случай изменения словаря в другом потоке
                    statuses_copy[key] = self.master_app.api_key_statuses[key].copy()

        current_displayed_keys = set(self.data_row_map.keys())
        keys_in_new_data = set(statuses_copy.keys())

        keys_to_remove = current_displayed_keys - keys_in_new_data
        for api_key_to_remove in keys_to_remove:
            if api_key_to_remove in self.data_row_map:
                for label_widget in self.data_row_map[api_key_to_remove]:
                    if label_widget.winfo_exists(): label_widget.destroy()
                del self.data_row_map[api_key_to_remove]

        row_idx_counter = 1
        for api_key_str in sorted_keys_from_master:  # Используем отсортированный список
            status_data = statuses_copy.get(api_key_str)
            if not status_data: continue

            key_display = f"...{api_key_str[-7:]}" if len(api_key_str) > 7 else api_key_str
            status_val = status_data.get("status", "N/A")
            rem_req = status_data.get("remaining_requests")
            lim_req = status_data.get("limit_requests")
            req_display = f"{rem_req}/{lim_req}" if lim_req is not None and rem_req is not None else "N/A"

            reset_req_at = status_data.get("reset_requests_at")
            reset_req_display = "N/A"
            if isinstance(reset_req_at, datetime.datetime):
                if reset_req_at > now_utc:
                    delta = relativedelta(reset_req_at, now_utc)
                    parts = []
                    if delta.days > 0: parts.append(f"{delta.days}д")
                    if delta.hours > 0: parts.append(f"{delta.hours}ч")
                    if delta.minutes > 0: parts.append(f"{delta.minutes}м")
                    if delta.seconds > 0 or not parts: parts.append(f"{delta.seconds}с")
                    reset_req_display = " ".join(parts) if parts else "~0с"
                else:
                    reset_req_display = "Сброшен"
            elif reset_req_at:
                reset_req_display = str(reset_req_at)[:19]

            rem_tok = status_data.get("remaining_tokens")
            lim_tok = status_data.get("limit_tokens")
            tok_display = f"{rem_tok}/{lim_tok}" if lim_tok is not None and rem_tok is not None else "N/A"

            reset_tok_at = status_data.get("reset_tokens_at")
            reset_tok_display = "N/A"
            if isinstance(reset_tok_at, datetime.datetime):
                if reset_tok_at > now_utc:
                    delta = relativedelta(reset_tok_at, now_utc)
                    parts = []
                    if delta.days > 0: parts.append(f"{delta.days}д")
                    if delta.hours > 0: parts.append(f"{delta.hours}ч")
                    if delta.minutes > 0: parts.append(f"{delta.minutes}м")
                    if delta.seconds > 0 or not parts: parts.append(f"{delta.seconds}с")
                    reset_tok_display = " ".join(parts) if parts else "~0с"
                else:
                    reset_tok_display = "Сброшен"
            elif reset_tok_at:
                reset_tok_display = str(reset_tok_at)[:19]

            last_upd = status_data.get("last_updated")
            last_upd_display = "N/A"
            if isinstance(last_upd, datetime.datetime):
                try:
                    last_upd_local = last_upd.astimezone() if last_upd.tzinfo else last_upd; last_upd_display = last_upd_local.strftime(
                        "%H:%M:%S")
                except Exception:
                    last_upd_display = last_upd.strftime("%H:%M:%S (UTC?)")
            error_msg = status_data.get("error_message", "")
            values_to_display = [key_display, status_val, req_display, reset_req_display, tok_display,
                                 reset_tok_display, last_upd_display, error_msg]

            if api_key_str in self.data_row_map:
                labels_for_key = self.data_row_map[api_key_str]
                for col, new_text in enumerate(values_to_display):
                    if labels_for_key[col].winfo_exists(): labels_for_key[col].configure(text=str(new_text))
                for col, label_widget in enumerate(labels_for_key):  # Перегридовка для сохранения порядка
                    if label_widget.winfo_exists(): label_widget.grid(row=row_idx_counter, column=col, padx=5, pady=2,
                                                                      sticky="ew")
            else:
                new_labels_row = []
                for col, val_text in enumerate(values_to_display):
                    cell_label = ctk.CTkLabel(self.scrollable_frame, text=str(val_text),
                                              wraplength=150 if col == 7 else 120, anchor="w")  # Увеличил wraplength
                    cell_label.grid(row=row_idx_counter, column=col, padx=5, pady=2, sticky="ew")
                    new_labels_row.append(cell_label)
                self.data_row_map[api_key_str] = new_labels_row
            row_idx_counter += 1

        # Обработка случая, когда нет ключей для отображения
        if "_no_data_message_" in self.data_row_map and keys_in_new_data:  # Если были "нет данных", а теперь есть
            for widget in self.data_row_map["_no_data_message_"]: widget.destroy()
            del self.data_row_map["_no_data_message_"]
        elif not keys_in_new_data and "_no_data_message_" not in self.data_row_map:  # Если данных нет и сообщения тоже
            no_keys_label = ctk.CTkLabel(self.scrollable_frame, text="Нет API ключей для отображения.")
            no_keys_label.grid(row=1, column=0, columnspan=len(self.header_labels), padx=5, pady=10, sticky="ew")
            self.data_row_map["_no_data_message_"] = [no_keys_label]

    def _schedule_refresh(self):
        if not (self.winfo_exists() and hasattr(self, 'scrollable_frame') and self.scrollable_frame.winfo_exists()):
            if self._after_id_refresh:
                try:
                    self.after_cancel(self._after_id_refresh)
                except tk.TclError:
                    pass  # Может быть ошибка, если окно уже уничтожено
                self._after_id_refresh = None
            return
        self.populate_table_data()
        self._after_id_refresh = self.after(self.refresh_interval_ms, self._schedule_refresh)

    def on_close(self):
        if self._after_id_refresh:
            try:
                self.after_cancel(self._after_id_refresh)
            except tk.TclError:
                pass
            self._after_id_refresh = None
        self.master_app.api_status_win = None  # Сбрасываем ссылку в основном приложении
        self.destroy()

    def populate_table(self):
        # Очищаем предыдущие виджеты, если они есть
        for widget_row in self.table_widgets:
            for widget in widget_row:
                widget.destroy()
        self.table_widgets = []

        headers = ["Ключ (хвост)", "Статус", "Запросов (Ост/Лимит)", "Сброс Запросов",
                   "Токенов (Ост/Лимит)", "Сброс Токенов", "Обновлено", "Ошибка"]

        for col, header_text in enumerate(headers):
            header_label = ctk.CTkLabel(self.scrollable_frame, text=header_text, font=ctk.CTkFont(weight="bold"))
            header_label.grid(row=0, column=col, padx=5, pady=5, sticky="w")
            # self.table_widgets.append([header_label]) # Не добавляем заголовки в список очистки

        row_idx = 1
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        with self.master_app.api_key_statuses_lock:  # Доступ к статусам ключей основного приложения
            sorted_keys = sorted(self.master_app.api_key_statuses.keys())

            for api_key_str in sorted_keys:
                status_data = self.master_app.api_key_statuses.get(api_key_str)
                if not status_data: continue

                row_widgets = []

                key_display = f"...{api_key_str[-7:]}" if len(api_key_str) > 7 else api_key_str
                status_val = status_data.get("status", "N/A")

                rem_req = status_data.get("remaining_requests", "N/A")
                lim_req = status_data.get("limit_requests", "N/A")
                req_display = f"{rem_req}/{lim_req}"

                reset_req_at = status_data.get("reset_requests_at")
                reset_req_display = "N/A"
                if isinstance(reset_req_at, datetime.datetime):
                    if reset_req_at > now_utc:
                        delta = relativedelta(reset_req_at, now_utc)
                        reset_req_display = ""
                        if delta.days > 0: reset_req_display += f"{delta.days}д "
                        if delta.hours > 0: reset_req_display += f"{delta.hours}ч "
                        if delta.minutes > 0: reset_req_display += f"{delta.minutes}м "
                        if delta.seconds > 0 or not reset_req_display: reset_req_display += f"{delta.seconds}с"
                        reset_req_display = reset_req_display.strip()
                    else:
                        reset_req_display = "Прошло"
                elif reset_req_at:  # Если это строка (не должно быть, но на всякий)
                    reset_req_display = str(reset_req_at)[:19]

                rem_tok = status_data.get("remaining_tokens", "N/A")
                lim_tok = status_data.get("limit_tokens", "N/A")
                tok_display = f"{rem_tok}/{lim_tok}"

                reset_tok_at = status_data.get("reset_tokens_at")
                reset_tok_display = "N/A"
                if isinstance(reset_tok_at, datetime.datetime):
                    if reset_tok_at > now_utc:
                        delta = relativedelta(reset_tok_at, now_utc)
                        reset_tok_display = ""
                        if delta.days > 0: reset_tok_display += f"{delta.days}д "
                        if delta.hours > 0: reset_tok_display += f"{delta.hours}ч "
                        if delta.minutes > 0: reset_tok_display += f"{delta.minutes}м "
                        if delta.seconds > 0 or not reset_tok_display: reset_tok_display += f"{delta.seconds}с"
                        reset_tok_display = reset_tok_display.strip()
                    else:
                        reset_tok_display = "Прошло"
                elif reset_tok_at:
                    reset_tok_display = str(reset_tok_at)[:19]

                last_upd = status_data.get("last_updated")
                last_upd_display = "N/A"
                if isinstance(last_upd, datetime.datetime):
                    last_upd_display = last_upd.strftime("%H:%M:%S")  # Только время для краткости

                error_msg = status_data.get("error_message", "")

                values = [key_display, status_val, req_display, reset_req_display,
                          tok_display, reset_tok_display, last_upd_display, error_msg]

                for col, val_text in enumerate(values):
                    cell_label = ctk.CTkLabel(self.scrollable_frame, text=str(val_text),
                                              wraplength=100)  # wraplength для error_message
                    cell_label.grid(row=row_idx, column=col, padx=5, pady=2, sticky="w")
                    row_widgets.append(cell_label)

                self.table_widgets.append(row_widgets)
                row_idx += 1

        if not sorted_keys:
            no_keys_label = ctk.CTkLabel(self.scrollable_frame,
                                         text="Нет API ключей для отображения или статусы еще не загружены.")
            no_keys_label.grid(row=1, column=0, columnspan=len(headers), padx=5, pady=10)
            self.table_widgets.append([no_keys_label])


class HelpWindow(ctk.CTkToplevel):
    def __init__(self, master_app):
        super().__init__(master_app)
        self.title("Помощь")
        win_w, win_h = 600, 400
        self.geometry(f"{win_w}x{win_h}")
        self.update_idletasks()
        try:
            m_x = master_app.winfo_rootx()
            m_y = master_app.winfo_rooty()
            m_w = master_app.winfo_width()
            m_h = master_app.winfo_height()
            pos_x = m_x + (m_w - win_w) // 2
            pos_y = m_y + (m_h - win_h) // 2
            self.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        except Exception:
            pass

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        textbox = ctk.CTkTextbox(frame, state="normal", wrap="word")
        textbox.insert("1.0", HELP_TEXT)
        textbox.configure(state="disabled")
        textbox.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self.destroy)


class TabbedApp(ctk.CTk):
    def __init__(self, folders):
        super().__init__()
        self.title("Генератор Текстов v2.6 (Tabs)")
        self.geometry("1300x731")
        self.folders = folders
        self.apps = []

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True)

        for folder in self.folders:
            self._add_tab(folder)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _add_tab(self, folder):
        os.makedirs(folder, exist_ok=True)
        tab = self.tabview.add(os.path.basename(folder))
        cfg_path = os.path.join(folder, DEFAULT_CONFIG_FILE)
        app = TextGeneratorApp(tab, config_file=cfg_path)
        app.output_folder.set(folder)
        app.pack(fill="both", expand=True)
        app.save_settings()
        self.apps.append(app)

    def on_close(self):
        global PROGRAM_EXITED_VIA_UI
        PROGRAM_EXITED_VIA_UI = True
        for app in self.apps:
            app.save_settings()
        save_project_folders(self.folders)
        self.destroy()


def prompt_for_projects():
    prompt_root = tk.Tk()
    prompt_root.withdraw()

    num = simpledialog.askinteger(
        "Количество вкладок",
        "Сколько открыть вкладок?",
        minvalue=1,
        initialvalue=1,
        parent=prompt_root,
    )
    if not num:
        sys.exit(0)

    script_dir = APP_DIR
    folders = []
    for i in range(num):
        name = simpledialog.askstring(
            "Название папки",
            f"Введите название папки для вкладки {i + 1}:",
            parent=prompt_root,
        )
        if not name:
            name = f"project_{i + 1}"
        path = os.path.join(script_dir, name)
        os.makedirs(path, exist_ok=True)
        folders.append(path)

    prompt_root.destroy()
    return folders


if __name__ == "__main__":
    check_first_run_password()
    folders = load_project_folders()
    if not folders:
        folders = prompt_for_projects()
        save_project_folders(folders)

    app = TabbedApp(folders)
    app.mainloop()