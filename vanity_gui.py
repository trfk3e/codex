import os
import time
import threading
import queue
import csv
import customtkinter as ctk
import hashlib
from ecdsa import SECP256k1, SigningKey
import base58
from sha3 import keccak_256

# --- Network specific generators -------------------------------------------------


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
    "Bitcoin": _generate_bitcoin,
    "Ethereum": _generate_ethereum,
    "Tron": _generate_tron,
    "Dogecoin": _generate_dogecoin,
    "Litecoin": _generate_litecoin,
}

# --- Worker ---------------------------------------------------------------------

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


# --- GUI -----------------------------------------------------------------------

class VanityApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Vanity Address Generator")
        self.geometry("700x500")

        ctk.set_default_color_theme("blue")

        self.network_var = ctk.StringVar(value="Bitcoin")
        self.prefix_var = ctk.StringVar()
        self.suffix_var = ctk.StringVar()
        self.case_var = ctk.BooleanVar(value=False)
        self.threads_var = ctk.IntVar(value=1)
        self.target_var = ctk.IntVar(value=1)

        self._build_ui()

        self.result_queue = queue.Queue()
        self.counter = {"total": 0, "found": 0, "lock": threading.Lock()}
        self.stop_event = threading.Event()
        self.threads = []
        self.start_time = None
        self.output_file = "vanity_results.csv"

    # UI layout
    def _build_ui(self):
        options_frame = ctk.CTkFrame(self)
        options_frame.pack(padx=10, pady=10, fill="x")

        ctk.CTkLabel(options_frame, text="Network").grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkOptionMenu(
            options_frame,
            values=list(NETWORK_GENERATORS.keys()),
            variable=self.network_var,
        ).grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(options_frame, text="Prefix").grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkEntry(options_frame, textvariable=self.prefix_var).grid(
            row=1, column=1, padx=5, pady=5
        )

        ctk.CTkLabel(options_frame, text="Suffix").grid(row=2, column=0, padx=5, pady=5)
        ctk.CTkEntry(options_frame, textvariable=self.suffix_var).grid(
            row=2, column=1, padx=5, pady=5
        )

        ctk.CTkCheckBox(options_frame, text="Case sensitive", variable=self.case_var).grid(
            row=3, column=0, columnspan=2, pady=5
        )

        ctk.CTkLabel(options_frame, text="Threads").grid(row=4, column=0, padx=5, pady=5)
        ctk.CTkEntry(options_frame, textvariable=self.threads_var).grid(
            row=4, column=1, padx=5, pady=5
        )

        ctk.CTkLabel(options_frame, text="Targets").grid(row=5, column=0, padx=5, pady=5)
        ctk.CTkEntry(options_frame, textvariable=self.target_var).grid(
            row=5, column=1, padx=5, pady=5
        )

        control_frame = ctk.CTkFrame(self)
        control_frame.pack(padx=10, fill="x")

        self.start_btn = ctk.CTkButton(control_frame, text="Start", command=self.start)
        self.start_btn.pack(side="left", padx=5, pady=5)

        self.stop_btn = ctk.CTkButton(control_frame, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=5, pady=5)

        self.status_label = ctk.CTkLabel(control_frame, text="Idle")
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
        threads = max(1, self.threads_var.get())
        target = max(1, self.target_var.get())

        self.result_queue = queue.Queue()
        self.counter = {"total": 0, "found": 0, "lock": threading.Lock()}
        self.stop_event.clear()
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
        self.status_label.configure(text="Running...")
        self.after(500, self._update_gui)

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join()
        self.threads = []
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Stopped")

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
            text=f"{total} generated, {found} found, {rate:.2f} addr/s"
        )

        if self.stop_event.is_set():
            self.stop()
        else:
            self.after(500, self._update_gui)


if __name__ == "__main__":
    app = VanityApp()
    app.mainloop()
