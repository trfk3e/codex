# bot.py ────────────────────────────────────────────────────────────────────
import json, re, html
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import logging
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
from openai import OpenAI

# ────────────── КОНФИГ ─────────────────────────────────────────────────────
OPENAI_API_KEY = "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"
TELEGRAM_BOT_TOKEN   = "7621000604:AAHrWFyNx8JCrPkCtmtC4MWAV2Ri5-EpOQo"
TG_API_ID     = "28511990"
TG_API_HASH = "f51873d6f1402467b6188a37100754ae"


PROCESSED_FILE = Path("processed_ids.json")
PROCESSED_LIMIT = 1000
CONCURRENCY = 5
MODEL_NAME     = "o3-mini"
ATTEMPT_LIMIT  = 50
ATTEMPT_MSG    = (
    "После 50 попыток все ответы AI были - NO, повторите ещё раз, "
    "поменяйте промпт, или выберите другой канал"
)

# ────────────── ГЛОБАЛЬНЫЕ КЛИЕНТЫ ─────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
tg_client     = TelegramClient(
    "seo_news_session", TG_API_ID, TG_API_HASH, timeout=10
)

async def openai_call(method, *args, timeout=60, **kwargs):
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, lambda: method(*args, **kwargs)), timeout
    )

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

logging.basicConfig(level=logging.INFO)

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
        ids = {c: deque(map(int, v), maxlen=PROCESSED_LIMIT) for c, v in cfg.get("ids", {}).items()}
        p_yes = cfg.get("prompt_if_yes", DEFAULT_PROMPT_YES)
        p_no = cfg.get("prompt_if_no", DEFAULT_PROMPT_NO)
        log_en = bool(cfg.get("log_enabled", False))
        data[str(chat_id)] = ChatConfig(chs, ids, p_yes, p_no, log_en)

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
        data[cid] = {
            "channels": cfg.channels,
            "ids": {c: list(v) for c, v in cfg.ids.items()},
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


def task_running(ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    t = ctx.chat_data.get("task")
    return bool(t) and not t.done()


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
    rsp = await openai_call(openai_client.chat.completions.create, model=MODEL_NAME, messages=base)
    answer = rsp.choices[0].message.content.strip()
    ok = answer.lower().startswith("y")
    reason = None

    if not ok and cfg.log_enabled:
        rsp2 = await openai_call(
            openai_client.chat.completions.create,
            model=MODEL_NAME,
            messages=base
            + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "Кратко объясни почему NO"},
            ],
        )
        reason = rsp2.choices[0].message.content.strip()

    return ok, reason

async def paraphrase(text: str) -> str:
    rsp = await openai_call(
        openai_client.chat.completions.create,
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
        openai_client.chat.completions.create,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": text.strip()},
        ],
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
    try:
        msgs = [
            m
            async for m in tg_client.iter_messages(channel)
            if m.text
        ]
    except ValueError:
        if isinstance(channel, str) and channel.lstrip("-").isdigit():
            msgs = [
                m
                async for m in tg_client.iter_messages(int(channel))
                if m.text
            ]
        else:
            raise
    except Exception:
        logging.exception("Failed to fetch posts")
        return []
    if from_dt or to_dt:
        msgs = [
            m
            for m in msgs
            if (from_dt or datetime.min)
            <= m.date.replace(tzinfo=None)
            <= (to_dt or datetime.max)
        ]
    if by_popularity:
        msgs.sort(key=quick_score, reverse=True)
    else:
        msgs.sort(key=lambda m: m.date)
    if limit:
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
    except Exception:
        logging.exception("AI filter failed")
        await log(ctx, f"AI ошибка при обработке {msg.id}")
        seen.append(msg.id)
        save_all()
        return False
    if ctx.chat_data.get("stop"):
        return False
    await log(ctx, f"AI ответ для {msg.id}: {'YES' if ai_ok else 'NO'}")
    if not ai_ok:
        if reason:
            await log(ctx, reason)
        seen.append(msg.id)
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
        except Exception:
            logging.exception("Failed to build digest")
            await log(ctx, f"Не удалось построить выжимку для {msg.id}")
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
        except Exception:
            logging.exception("Failed to paraphrase message")
            await log(ctx, f"Не удалось перефразировать {msg.id}")
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
    seen.append(msg.id)
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
MAIN_KB = ReplyKeyboardMarkup(
    [
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
    ],
    resize_keyboard=True,
)

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
    ctx.chat_data["target_chat"] = update.effective_chat.id
    ctx.chat_data["chat_id"] = chat_id
    ctx.chat_data["start_id"] = update.message.message_id
    get_cfg(ctx)  # ensure config exists
    await update.message.reply_text("Выберите действие:", reply_markup=MAIN_KB)


# -------------- текстовый ввод после кнопок --------------------------------
DATE_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4})-(\d{2}\.\d{2}\.\d{4})")

async def text_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    ctx.chat_data.setdefault("target_chat", update.effective_chat.id)
    ctx.chat_data.setdefault("chat_id", str(update.effective_chat.id))
    ctx.chat_data.setdefault("start_id", update.message.message_id)
    cfg = get_cfg(ctx)
    mode = ctx.user_data.get("mode")

    if text == "Отмена":
        ctx.user_data.clear()
        await update.message.reply_text("Отменено", reply_markup=MAIN_KB)
        return

    if text == "⏹ Остановить":
        ctx.chat_data["stop"] = True
        ctx.chat_data["auto_stop"] = True
        await update.message.reply_text(
            "Парсинг будет остановлен", reply_markup=MAIN_KB
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
            "11. <b>📰 Авто-выжимка</b> — бот раз в час ищет новые публикации, сразу делает по ним выжимки и присылает на апрув. Повторное нажатие останавливает мониторинг, также его можно прервать кнопкой ⏹.\n"
        )
        await update.message.reply_text(
            instruction, reply_markup=MAIN_KB, parse_mode=tg_const.ParseMode.HTML
        )
        return

    auto_task = ctx.chat_data.get("auto_task")
    auto_running = bool(auto_task) and not auto_task.done()

    if text == "📰 Авто-выжимка":
        if auto_running:
            ctx.chat_data["auto_stop"] = True
            ctx.chat_data["stop"] = True
            await update.message.reply_text(
                "Останавливаю авто-выжимку…", reply_markup=MAIN_KB
            )
        else:
            if not cfg.channels:
                await update.message.reply_text("Список каналов пуст.", reply_markup=MAIN_KB)
                return
            if task_running(ctx):
                await update.message.reply_text(
                    "Уже выполняется задача. Нажмите ⏹ Остановить",
                    reply_markup=MAIN_KB,
                )
                return
            ctx.user_data.clear()
            ctx.chat_data["auto_stop"] = False
            task = ctx.application.create_task(auto_monitor(ctx))
            ctx.chat_data["auto_task"] = task
            await update.message.reply_text(
                "Авто-выжимка запущена. Проверяю каналы каждый час.",
                reply_markup=MAIN_KB,
            )
        return

    auto_task = ctx.chat_data.get("auto_task")
    auto_running = bool(auto_task) and not auto_task.done()

    if auto_running and text not in {"⏹ Остановить"}:
        await update.message.reply_text(
            "Сейчас работает авто-выжимка. Остановите её через кнопку 📰 Авто-выжимка, чтобы выполнить другие действия.",
            reply_markup=MAIN_KB,
        )
        return

    # block new actions while a task is running so menus don't mix
    if task_running(ctx):
        await update.message.reply_text(
            "Уже выполняется задача. Нажмите ⏹ Остановить",
            reply_markup=MAIN_KB,
        )
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
            reply_markup=MAIN_KB,
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
        await update.message.reply_text("Фильтр-промпт обновлён", reply_markup=MAIN_KB)
        return

    if mode == "toggle_log":
        if text in {"Да", "Нет"}:
            cfg.log_enabled = text == "Да"
            save_all()
            ctx.user_data.clear()
            await update.message.reply_text(
                f"Лог {'включен' if cfg.log_enabled else 'выключен'}",
                reply_markup=MAIN_KB,
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
        await update.message.reply_text(msg, reply_markup=MAIN_KB)
        return

    if mode == "clear_menu":
        if text == "Очистить конкретные ID постов":
            if not cfg.channels:
                await update.message.reply_text("Список каналов пуст.", reply_markup=MAIN_KB)
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
            await update.message.reply_text("Все ID очищены", reply_markup=MAIN_KB)
            return
        if text == "Отмена":
            ctx.user_data.clear()
            await update.message.reply_text("Отменено", reply_markup=MAIN_KB)
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
        await update.message.reply_text(msg, reply_markup=MAIN_KB)
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
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=MAIN_KB
            )
            return
        chan = ctx.user_data.get("channel")
        limit = int(text)
        await update.message.reply_text("Обрабатываю…", reply_markup=MAIN_KB)
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
            await update.message.reply_text("Канал не в списке.", reply_markup=MAIN_KB)
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
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=MAIN_KB
            )
            return
        chan = ctx.user_data.get("channel")
        limit = int(text)
        from_d = ctx.user_data.get("from")
        to_d = ctx.user_data.get("to")
        await update.message.reply_text("Обрабатываю…", reply_markup=MAIN_KB)
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
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=MAIN_KB
            )
            return
        limit = int(text)
        from_d = ctx.user_data.get("from")
        to_d = ctx.user_data.get("to")
        await update.message.reply_text("Обрабатываю…", reply_markup=MAIN_KB)
        launch_task(ctx, run_seq_all(ctx, from_d, to_d, limit))
        return

    if mode == "seq_count":
        if not text.isdigit():
            await update.message.reply_text("Нужно число")
            return
        if task_running(ctx):
            await update.message.reply_text(
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=MAIN_KB
            )
            return
        limit = int(text)
        await update.message.reply_text("Стартуем…", reply_markup=MAIN_KB)
        launch_task(ctx, run_seq_all(ctx, None, None, limit))
        return

    if mode == "pop_count":
        if not text.isdigit():
            await update.message.reply_text("Нужно число")
            return
        if task_running(ctx):
            await update.message.reply_text(
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=MAIN_KB
            )
            return
        limit = int(text)
        await update.message.reply_text("Ищем популярные…", reply_markup=MAIN_KB)
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
            await update.message.reply_text("Канал не в списке.", reply_markup=MAIN_KB)
            ctx.user_data.clear()
        return

    if mode == "pop_chan_count":
        if not text.isdigit():
            await update.message.reply_text("Нужно число")
            return
        if task_running(ctx):
            await update.message.reply_text(
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=MAIN_KB
            )
            return
        chan = ctx.user_data.get("channel")
        limit = int(text)
        await update.message.reply_text("Ищем популярные…", reply_markup=MAIN_KB)
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
            await update.message.reply_text("Канал не в списке.", reply_markup=MAIN_KB)
            ctx.user_data.clear()
        return

    if mode == "recent_chan_days":
        if not text.isdigit():
            await update.message.reply_text("Нужно число")
            return
        if task_running(ctx):
            await update.message.reply_text(
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=MAIN_KB
            )
            return
        chan = ctx.user_data.get("channel")
        days = int(text)
        await update.message.reply_text("Обрабатываю…", reply_markup=MAIN_KB)
        launch_task(ctx, run_recent_channel(ctx, chan, days))
        return

    if mode == "recent_all_days":
        if not text.isdigit():
            await update.message.reply_text("Нужно число")
            return
        if task_running(ctx):
            await update.message.reply_text(
                "Уже выполняется задача. Нажмите ⏹ Остановить", reply_markup=MAIN_KB
            )
            return
        days = int(text)
        await update.message.reply_text("Обрабатываю…", reply_markup=MAIN_KB)
        launch_task(ctx, run_recent_all(ctx, days))
        return
    if mode:
        await update.message.reply_text("Не понимаю ответ, начните заново", reply_markup=MAIN_KB)
        ctx.user_data.clear()
    else:
        await update.message.reply_text("Неизвестная команда", reply_markup=MAIN_KB)


# -------------- задачи -----------------------------------------------------


async def auto_cycle(ctx: ContextTypes.DEFAULT_TYPE) -> int:
    cfg = get_cfg(ctx)
    if not cfg.channels:
        return 0

    total_sent = 0
    await tg_client.start()
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
            fresh = [m for m in posts if m.id not in seen]
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
        await tg_client.disconnect()
    return total_sent


async def prime_auto_seen(ctx: ContextTypes.DEFAULT_TYPE) -> int:
    """Зафиксировать текущие посты, чтобы авто-выжимка начинала только с новых."""

    cfg = get_cfg(ctx)
    if not cfg.channels:
        return 0

    added_total = 0
    await tg_client.start()
    try:
        for chan in cfg.channels:
            posts = await fetch_posts(chan, None, None, 1)
            if not posts:
                continue
            key = chan.lstrip("@")
            seen = cfg.ids.setdefault(key, deque(maxlen=PROCESSED_LIMIT))
            latest = posts[-1]
            if latest.id not in seen:
                seen.append(latest.id)
                added_total += 1
                await log(ctx, f"Зафиксировал последний пост {chan}: {latest.id}")
    finally:
        await tg_client.disconnect()

    if added_total:
        save_all()

    return added_total


async def auto_monitor(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = ctx.chat_data.get("target_chat")
    if chat_id is None:
        return
    try:
        primed = await prime_auto_seen(ctx)
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.exception("Не удалось подготовить авто-выжимку")
        primed = 0
    await ctx.bot.send_message(
        chat_id,
        "Авто-выжимка активирована. Буду присылать новые релевантные публикации каждый час.",
    )
    if primed:
        await ctx.bot.send_message(
            chat_id,
            "Текущие публикации помечены как просмотренные, начну с новых постов.",
        )
    try:
        while not ctx.chat_data.get("auto_stop"):
            total = await auto_cycle(ctx)
            if ctx.chat_data.get("auto_stop"):
                break
            if total:
                await ctx.bot.send_message(
                    chat_id,
                    f"Авто-выжимка: прислано {total} новостей на апрув.",
                )
            for _ in range(60):
                if ctx.chat_data.get("auto_stop"):
                    break
                await asyncio.sleep(60)
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.exception("Ошибка в автоматическом мониторинге")
        await ctx.bot.send_message(
            chat_id,
            "Авто-выжимка остановлена из-за ошибки. Проверьте логи.",
        )
    finally:
        ctx.chat_data.pop("auto_stop", None)
        ctx.chat_data.pop("auto_task", None)
        await ctx.bot.send_message(chat_id, "Авто-выжимка остановлена.")


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