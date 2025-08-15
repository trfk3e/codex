import os
import time
import threading
import queue
import csv
import customtkinter as ctk
from bip_utils import (
    Secp256k1PrivateKey,
    P2PKHAddrEncoder,
    WifEncoder,
    CoinsConf,
    EthAddr,
    TrxAddr,
)

# --- Network specific generators -------------------------------------------------

def _generate_bitcoin(priv_key):
    address = P2PKHAddrEncoder.EncodeKey(
        priv_key.PublicKey(),
        net_ver=CoinsConf.BitcoinMainNet.ParamByKey("p2pkh_net_ver"),
    )
    wif = WifEncoder.Encode(
        priv_key.Raw().ToBytes(),
        net_ver=CoinsConf.BitcoinMainNet.ParamByKey("wif_net_ver"),
    )
    return address, wif


def _generate_dogecoin(priv_key):
    address = P2PKHAddrEncoder.EncodeKey(
        priv_key.PublicKey(),
        net_ver=CoinsConf.DogecoinMainNet.ParamByKey("p2pkh_net_ver"),
    )
    wif = WifEncoder.Encode(
        priv_key.Raw().ToBytes(),
        net_ver=CoinsConf.DogecoinMainNet.ParamByKey("wif_net_ver"),
    )
    return address, wif


def _generate_litecoin(priv_key):
    address = P2PKHAddrEncoder.EncodeKey(
        priv_key.PublicKey(),
        net_ver=CoinsConf.LitecoinMainNet.ParamByKey("p2pkh_net_ver"),
    )
    wif = WifEncoder.Encode(
        priv_key.Raw().ToBytes(),
        net_ver=CoinsConf.LitecoinMainNet.ParamByKey("wif_net_ver"),
    )
    return address, wif


def _generate_ethereum(priv_key):
    address = EthAddr.EncodeKey(priv_key.PublicKey())
    return address, priv_key.Raw().ToHex()


def _generate_tron(priv_key):
    address = TrxAddr.EncodeKey(priv_key.PublicKey())
    return address, priv_key.Raw().ToHex()


NETWORK_GENERATORS = {
    "Bitcoin": _generate_bitcoin,
    "Ethereum": _generate_ethereum,
    "Tron": _generate_tron,
    "Dogecoin": _generate_dogecoin,
    "Litecoin": _generate_litecoin,
}

# --- Worker ---------------------------------------------------------------------

def generate_address(network: str):
    priv_key = Secp256k1PrivateKey.FromBytes(os.urandom(32))
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
