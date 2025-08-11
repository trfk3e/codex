# check_openai_keys.py
"""
Проверяет список API-ключей OpenAI из файла **KEYS.txt** на работоспособность с моделью **o3-mini**.

Создаёт два файла:
  • **bad-api.txt** — содержит ключи, по которым вызов не удался.
  • **log.txt**      — для каждого проблемного ключа строка вида
                      `<ключ>: <сообщение_ошибки>` (между записями пустая строка).

### Использование
1. Положите рядом `KEYS.txt` (по одному ключу в строке).
2. Установите/обновите библиотеку:
   ```bash
   pip install --upgrade openai
   ```
3. Запустите:
   ```bash
   python check_openai_keys.py
   ```

### Почему был `ModuleNotFoundError`
Начиная с версии **openai-python >= 1.0** модуль `openai.error` удалён.
Все исключения теперь доступны прямо из корневого пакета, поэтому скрипт
обновлён на `from openai import OpenAIError`.
"""
from __future__ import annotations

import pathlib
import time
from typing import List

# --- OpenAI SDK ----------------------------------------------------------------
# Для новых версий (>1.0)
try:
    from openai import OpenAI, OpenAIError  # type: ignore
except ImportError:  # на случай старой 0.x — используем совместимую схему
    import openai  # type: ignore

    class OpenAIError(openai.error.OpenAIError):  # type: ignore
        """Простой адаптер, чтобы код ниже не менялся."""

    class _LegacyClient:  # адаптер старого клиента к новому интерфейсу
        def __init__(self, api_key: str):
            self.api_key = api_key

        def chat(self):  # noqa: D401
            return self

        def completions(self):  # noqa: D401
            return self

        def create(self, **kwargs):  # noqa: ANN001
            return openai.ChatCompletion.create(api_key=self.api_key, **kwargs)

    def OpenAI(api_key: str):  # type: ignore  # noqa: N802
        return _LegacyClient(api_key)
# ------------------------------------------------------------------------------

# === Настройки ===
KEYS_FILE = "KEYS.txt"
BAD_KEYS_FILE = "bad-api.txt"
LOG_FILE = "log.txt"
MODEL = "o3-mini"      # какую модель пингуем
TIMEOUT = 20           # секунд на запрос
PAUSE = 1              # секунд между запросами (избегаем rate limit)
# ==================


def load_keys(path: pathlib.Path) -> List[str]:
    """Читает ключи из файла, убирая пустые строки и пробелы."""
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл {path.absolute()}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_key(client: OpenAI) -> None:  # type: ignore
    """Проверяет ключ, бросает исключение, если что-то не так."""
    client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Привет"}],
        timeout=TIMEOUT,
    )


def main() -> None:
    keys_path = pathlib.Path(KEYS_FILE)
    keys = load_keys(keys_path)

    bad_keys: List[str] = []
    log_lines: List[str] = []

    total = len(keys)
    print(f"Найдено {total} ключей. Проверяем…\n")

    for idx, key in enumerate(keys, 1):
        client = OpenAI(api_key=key)
        try:
            check_key(client)
            print(f"[{idx}/{total}] ✅ OK")
        except OpenAIError as exc:  # ошибки SDK
            print(f"[{idx}/{total}] ❌ {exc.__class__.__name__}")
            bad_keys.append(key)
            log_lines.append(f"{key}: {exc}")
        except Exception as exc:  # прочие сбои (сеть и пр.)
            print(f"[{idx}/{total}] ⚠️  {exc.__class__.__name__}")
            bad_keys.append(key)
            log_lines.append(f"{key}: {exc}")
        time.sleep(PAUSE)

    pathlib.Path(BAD_KEYS_FILE).write_text("\n".join(bad_keys), encoding="utf-8")
    pathlib.Path(LOG_FILE).write_text("\n\n".join(log_lines), encoding="utf-8")

    print("\n— Готово —")
    print(f"Не прошло: {len(bad_keys)} из {total} (см. {BAD_KEYS_FILE} и {LOG_FILE})")


if __name__ == "__main__":
    main()
