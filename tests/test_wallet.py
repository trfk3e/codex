import pytest
from ecdsa import SigningKey, SECP256k1
from vanity_gui import _generate_bitcoin, _generate_ethereum
import wallet_tools


def test_generate_bitcoin_known_key():
    sk = SigningKey.from_string(b"\x01" * 32, curve=SECP256k1)
    addr, wif = _generate_bitcoin(sk)
    assert addr == "1C6Rc3w25VHud3dLDamutaqfKWqhrLRTaD"
    assert wif == "KwFfNUhSDaASSAwtG7ssQM1uVX8RgX5GHWnnLfhfiQDigjioWXHH"


def test_generate_ethereum_known_key():
    sk = SigningKey.from_string(b"\x01" * 32, curve=SECP256k1)
    addr, key_hex = _generate_ethereum(sk)
    assert addr == "0x1a642f0e3c3af545e7acbd38b07251b3990914f1"
    assert key_hex == "01" * 32


def test_filter_addresses_by_minimum(monkeypatch):
    def fake_total(addr):
        return {"good": 1.5, "bad": 0.5}[addr]
    monkeypatch.setitem(wallet_tools.NETWORK_TOTAL_RECEIVED, "BTC", fake_total)
    result = wallet_tools.filter_addresses_by_minimum("BTC", ["good", "bad"], 1.0)
    assert result == [("good", 1.5)]
