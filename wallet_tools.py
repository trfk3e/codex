import requests

# --- Получение сумм транзакций --------------------------------------------------

# Каждая функция возвращает общую полученную сумму в стандартных единицах (BTC, ETH, BNB, TRX)


def _btc_total_received(address: str) -> float:
    """Возвращает общий объём BTC, полученный адресом через BlockCypher."""
    url = f"https://api.blockcypher.com/v1/btc/main/addrs/{address}"
    data = requests.get(url, timeout=10).json()
    return data.get("total_received", 0) / 1e8


def _eth_total_received(address: str) -> float:
    """Возвращает общий объём ETH, полученный адресом через BlockCypher."""
    url = f"https://api.blockcypher.com/v1/eth/main/addrs/{address}"
    data = requests.get(url, timeout=10).json()
    return data.get("total_received", 0) / 1e18


def _bsc_total_received(address: str, api_key: str | None = None) -> float:
    """Возвращает общий объём BNB, полученный через BscScan."""
    url = (
        "https://api.bscscan.com/api"
        "?module=account&action=txlist&startblock=0&endblock=99999999&sort=asc"
        f"&address={address}"
    )
    if api_key:
        url += f"&apikey={api_key}"
    data = requests.get(url, timeout=10).json()
    if data.get("status") != "1":
        return 0.0
    total = sum(int(tx.get("value", 0)) for tx in data.get("result", []))
    return total / 1e18


def _trx_total_received(address: str) -> float:
    """Возвращает общий объём TRX, полученный адресом через TronScan."""
    url = f"https://apilist.tronscan.org/api/transaction?address={address}"
    data = requests.get(url, timeout=10).json()
    total = 0
    for tx in data.get("data", []):
        try:
            total += int(tx.get("amount", 0))
        except ValueError:
            continue
    return total / 1e6  # SUN to TRX


NETWORK_TOTAL_RECEIVED = {
    "BTC": _btc_total_received,
    "ETH": _eth_total_received,
    "BNB": _bsc_total_received,
    "TRX": _trx_total_received,
}


def filter_addresses_by_minimum(network: str, addresses: list[str], minimum: float, api_key: str | None = None):
    """Возвращает адреса, получившие сумму не менее указанного минимума."""
    func = NETWORK_TOTAL_RECEIVED[network]
    result: list[tuple[str, float]] = []
    for addr in addresses:
        try:
            total = func(addr) if network != "BNB" else func(addr, api_key)
            if total >= minimum:
                result.append((addr, total))
        except Exception:
            continue
    return result


# --- Заготовка для отправки транзакций -----------------------------------------


def send_transaction(network: str, from_priv_key: str, to_address: str, amount: float):
    """Заглушка для отправки транзакции.

    Полноценная реализация потребует сетевых методов подписи и отправки.
    Функция намеренно не реализована.
    """
    raise NotImplementedError("Отправка транзакций не реализована.")
