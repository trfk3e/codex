import os
import logging
from typing import List, Dict
import requests


def load_api_keys(folder: str) -> List[str]:
    """Load API keys from all .txt files inside *folder*.
    Each line is treated as a separate key.
    """
    keys: List[str] = []
    if not os.path.isdir(folder):
        logging.warning("Папка с API ключами не найдена: %s", folder)
        return keys
    for name in os.listdir(folder):
        if name.lower().endswith('.txt'):
            path = os.path.join(folder, name)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        keys.append(line)
    return keys


def load_proxies(file_path: str) -> List[str]:
    """Load proxies from a file, one per line."""
    proxies: List[str] = []
    if not os.path.isfile(file_path):
        logging.warning("Файл прокси не найден: %s", file_path)
        return proxies
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                proxies.append(line)
    return proxies


def get_proxy_country(proxy: str) -> str:
    """Return country for proxy using ip-api.com."
    # proxy format assumed: ip:port:user:pass
    """
    ip = proxy.split(':')[0]
    try:
        resp = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)
        if resp.ok:
            data = resp.json()
            return data.get('country', 'Unknown')
    except requests.RequestException:
        pass
    return 'Unknown'


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return key
    return f"{key[:4]}...{key[-4:]}"


def assign_proxies(api_keys_folder: str, proxies_file: str) -> Dict[str, str]:
    keys = load_api_keys(api_keys_folder)
    proxies = load_proxies(proxies_file)
    if len(proxies) < len(keys):
        logging.error(
            "В папке .txt - %d АПИ, в папке с прокси - %d ПРОКСИ",
            len(keys), len(proxies),
        )
    mapping: Dict[str, str] = {}
    for key, proxy in zip(keys, proxies):
        country = get_proxy_country(proxy)
        logging.info("Подключение к прокси %s (страна: %s) для ключа %s", proxy, country, mask_key(key))
        mapping[key] = proxy
    return mapping


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Сопоставление API ключей и прокси")
    parser.add_argument("api_keys_folder", help="Папка, содержащая .txt файлы с API ключами")
    parser.add_argument("proxies_file", help="Файл со списком прокси")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    assign_proxies(args.api_keys_folder, args.proxies_file)
