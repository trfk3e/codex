import os
import time
import threading
import queue
import csv
import customtkinter as ctk
import hashlib
from ecdsa import SECP256k1, SigningKey
import base58
from Crypto.Hash import keccak


def keccak_256(data: bytes):
    return keccak.new(digest_bits=256, data=data)

# --- Генераторы для сетей -------------------------------------------------------


def _compress_pubkey(vk):
    pk_bytes = vk.to_string()
    prefix = b"\x02" if pk_bytes[63] % 2 == 0 else b"\x03"
    return prefix + pk_bytes[:32]


def _generate_btc_like(priv_key, p2pkh_prefix, wif_prefix):
    vk = priv_key.get_verifying_key()
    pubkey = _compress_pubkey(vk)
    sha = hashlib.sha256(pubkey).digest()
    ripe = hashlib.new("ripemd160", sha).digest()
    vh160 = p2pkh_prefix + ripe
    checksum = hashlib.sha256(hashlib.sha256(vh160).digest()).digest()[:4]
    address = base58.b58encode(vh160 + checksum).decode()

    wif_payload = wif_prefix + priv_key.to_string() + b"\x01"
    wif_check = hashlib.sha256(hashlib.sha256(wif_payload).digest()).digest()[:4]
    wif = base58.b58encode(wif_payload + wif_check).decode()
    return address, wif


def _generate_bitcoin(priv_key):
    return _generate_btc_like(priv_key, b"\x00", b"\x80")


def _generate_dogecoin(priv_key):
    return _generate_btc_like(priv_key, b"\x1e", b"\x9e")


def _generate_litecoin(priv_key):
    return _generate_btc_like(priv_key, b"\x30", b"\xb0")


def _generate_ethereum(priv_key):
    vk = priv_key.get_verifying_key()
    pubkey = vk.to_string()
    address = "0x" + keccak_256(pubkey).hexdigest()[-40:]
    return address, priv_key.to_string().hex()


def _generate_tron(priv_key):
    vk = priv_key.get_verifying_key()
    pubkey = vk.to_string()
    eth_address_bytes = keccak_256(pubkey).digest()[-20:]
    addr_payload = b"\x41" + eth_address_bytes
    checksum = hashlib.sha256(hashlib.sha256(addr_payload).digest()).digest()[:4]
    address = base58.b58encode(addr_payload + checksum).decode()
    return address, priv_key.to_string().hex()


NETWORK_GENERATORS = {
    "Биткоин": _generate_bitcoin,
    "Эфириум": _generate_ethereum,
    "Трон": _generate_tron,
    "Догикоин": _generate_dogecoin,
    "Лайткоин": _generate_litecoin,
}

# --- Рабочий поток --------------------------------------------------------------

def generate_address(network: str):
    priv_key = SigningKey.from_string(os.urandom(32), curve=SECP256k1)
    return NETWORK_GENERATORS[network](priv_key)


def worker(network, prefix, suffix, case_sensitive, result_queue, counter, target, stop_event):
    prefix_cmp = prefix if case_sensitive else prefix.lower()
    suffix_cmp = suffix if case_sensitive else suffix.lower()

    while not stop_event.is_set():
        addr, key = generate_address(network)
        addr_cmp = addr if case_sensitive else addr.lower()

        with counter["lock"]:
            counter["total"] += 1

        if (not prefix_cmp or addr_cmp.startswith(prefix_cmp)) and (
            not suffix_cmp or addr_cmp.endswith(suffix_cmp)
        ):
            result_queue.put((addr, key))
            with counter["lock"]:
                counter["found"] += 1
                if counter["found"] >= target:
                    stop_event.set()
                    return


# --- Графический интерфейс ------------------------------------------------------

class VanityApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Генератор ванити-адресов")
        self.geometry("700x500")

        ctk.set_default_color_theme("blue")

        self.network_var = ctk.StringVar(value="Биткоин")
        self.prefix_var = ctk.StringVar()
        self.suffix_var = ctk.StringVar()
        self.case_var = ctk.BooleanVar(value=False)
        # используем StringVar, чтобы пустой ввод не вызывал TclError в IntVar/DoubleVar
        self.threads_var = ctk.StringVar(value="1")
        self.target_var = ctk.StringVar(value="1")

        self._build_ui()

        self.result_queue = queue.Queue()
        self.counter = {"total": 0, "found": 0, "lock": threading.Lock()}
        self.stop_event = threading.Event()
        self.threads = []
        self.start_time = None
        self.output_file = "vanity_results.csv"

    # Размещение элементов
    def _build_ui(self):
        options_frame = ctk.CTkFrame(self)
        options_frame.pack(padx=10, pady=10, fill="x")

        ctk.CTkLabel(options_frame, text="Сеть").grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkOptionMenu(
            options_frame,
            values=list(NETWORK_GENERATORS.keys()),
            variable=self.network_var,
        ).grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(options_frame, text="Префикс").grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkEntry(options_frame, textvariable=self.prefix_var).grid(
            row=1, column=1, padx=5, pady=5
        )

        ctk.CTkLabel(options_frame, text="Суффикс").grid(row=2, column=0, padx=5, pady=5)
        ctk.CTkEntry(options_frame, textvariable=self.suffix_var).grid(
            row=2, column=1, padx=5, pady=5
        )

        ctk.CTkCheckBox(options_frame, text="Учитывать регистр", variable=self.case_var).grid(
            row=3, column=0, columnspan=2, pady=5
        )

        ctk.CTkLabel(options_frame, text="Потоки").grid(row=4, column=0, padx=5, pady=5)
        ctk.CTkEntry(options_frame, textvariable=self.threads_var).grid(
            row=4, column=1, padx=5, pady=5
        )

        ctk.CTkLabel(options_frame, text="Количество адресов").grid(row=5, column=0, padx=5, pady=5)
        ctk.CTkEntry(options_frame, textvariable=self.target_var).grid(
            row=5, column=1, padx=5, pady=5
        )

        control_frame = ctk.CTkFrame(self)
        control_frame.pack(padx=10, fill="x")

        self.start_btn = ctk.CTkButton(control_frame, text="Старт", command=self.start)
        self.start_btn.pack(side="left", padx=5, pady=5)

        self.stop_btn = ctk.CTkButton(control_frame, text="Стоп", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=5, pady=5)

        self.status_label = ctk.CTkLabel(control_frame, text="Ожидание")
        self.status_label.pack(side="left", padx=10)

        self.textbox = ctk.CTkTextbox(self, height=250)
        self.textbox.pack(padx=10, pady=10, fill="both", expand=True)

    def start(self):
        if self.threads:
            return

        network = self.network_var.get()
        prefix = self.prefix_var.get()
        suffix = self.suffix_var.get()
        case_sensitive = self.case_var.get()
        try:
            threads = max(1, int(self.threads_var.get()))
        except (TypeError, ValueError):
            threads = 1
        try:
            target = max(1, int(self.target_var.get()))
        except (TypeError, ValueError):
            target = 1

        # запуск новой сессии поиска
        self.result_queue = queue.Queue()
        self.counter = {"total": 0, "found": 0, "lock": threading.Lock()}
        self.stop_event = threading.Event()
        self.threads = []
        self.start_time = time.perf_counter()

        for _ in range(threads):
            t = threading.Thread(
                target=worker,
                args=(network, prefix, suffix, case_sensitive, self.result_queue, self.counter, target, self.stop_event),
                daemon=True,
            )
            t.start()
            self.threads.append(t)

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="Выполняется...")
        self.after(500, self._update_gui)

    def stop(self):
        # не блокируем главный поток ожиданием завершения рабочих потоков
        self.stop_event.set()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Остановка...")

    def _update_gui(self):
        while not self.result_queue.empty():
            addr, key = self.result_queue.get()
            line = f"{self.network_var.get()},{addr},{key}\n"
            self.textbox.insert("end", line)
            with open(self.output_file, "a", newline="") as f:
                f.write(line)

        elapsed = max(time.perf_counter() - self.start_time, 1e-6)
        with self.counter["lock"]:
            total = self.counter["total"]
            found = self.counter["found"]
        rate = total / elapsed
        self.status_label.configure(
            text=f"Сгенерировано {total}, найдено {found}, {rate:.2f} адресов/с"
        )

        if self.stop_event.is_set():
            # проверяем, все ли потоки завершены
            if not any(t.is_alive() for t in self.threads):
                self.threads = []
                self.start_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                self.status_label.configure(text="Остановлено")
            else:
                self.after(500, self._update_gui)
        else:
            self.after(500, self._update_gui)


if __name__ == "__main__":
    app = VanityApp()
    app.mainloop()
