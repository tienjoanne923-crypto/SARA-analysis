"""ctypes wrapper of the MathType 7 API (64-bit MT6.dll).

Signatures follow the declarations of MathType's own Word add-in (VBA Integer = int16, Long = int32, String =
ANSI char*).  All strings passed in must be ASCII (the ANSI code page here is 936); output paths are kept under
an ASCII temp folder.  The DLL drives MathType.exe through window messages and may use the clipboard, so a
session backs up and restores the text on the clipboard.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

DLL_PATH = r"C:\Program Files (x86)\MathType\System\64\MT6.dll"
PREFS_DIR = Path(r"C:\Program Files (x86)\MathType\Preferences")
TMP = Path(os.environ.get("TEMP", r"C:\Users\user\AppData\Local\Temp")) / "mtpipe"

LOCAL, FILE, PREVIOUS = -3, -4, -1
F_MTEF, F_PICT, F_TEXT, F_GIF = 4, 6, 7, 9
PREF_EXISTING, PREF_MTDEFAULT, PREF_USER = 1, 2, 3
STAT_ACTUAL_LEN = -1
ERRORS = {-1: "MT_NOT_FOUND", -2: "CANT_RUN", -3: "BAD_VERSION", -4: "IN_USE", -5: "NOT_RUNNING", -6: "RUN_TIMEOUT",
          -7: "NOT_EQUATION", -8: "FILE_NOT_FOUND", -9: "MEMORY", -10: "BAD_FILE", -11: "DATA_NOT_FOUND",
          -12: "TOO_MANY_SESSIONS", -13: "SUBSTITUTION_ERROR", -14: "TRANSLATOR_ERROR", -15: "PREFERENCE_ERROR",
          -16: "BAD_PATH", -17: "FILE_ACCESS", -18: "FILE_WRITE_ERROR", -9999: "mtERROR"}


class RECT(C.Structure):
    _fields_ = [("left", C.c_int32), ("top", C.c_int32), ("right", C.c_int32), ("bottom", C.c_int32)]


class DIMS(C.Structure):
    _fields_ = [("baseline", C.c_int16), ("bounds", RECT)]


class MTError(RuntimeError):
    pass


def _clip_backup():
    try:
        import win32clipboard as wc
        wc.OpenClipboard()
        try:
            return wc.GetClipboardData(wc.CF_UNICODETEXT) if wc.IsClipboardFormatAvailable(wc.CF_UNICODETEXT) else None
        finally:
            wc.CloseClipboard()
    except Exception:
        return None


def _clip_restore(text):
    if text is None:
        return
    try:
        import win32clipboard as wc
        wc.OpenClipboard()
        try:
            wc.EmptyClipboard()
            wc.SetClipboardText(text, wc.CF_UNICODETEXT)
        finally:
            wc.CloseClipboard()
    except Exception:
        pass


class MathType:
    """One API session: connect, convert text/MTEF, render WMF, translate MTEF back to MathML/TeX."""

    def __init__(self, timeout: int = 30):
        self.dll = C.WinDLL(DLL_PATH)
        S, L, P, V = C.c_int16, C.c_int32, C.c_char_p, C.c_void_p
        sig = {"MTAPIConnect": ([S, S], L), "MTAPIDisconnect": ([], L), "MTAPIVersion": ([S], L),
               "MTAPIGetAppFunctionality": ([], L), "MTXFormReset": ([], L), "MTXFormGetStatus": ([S], L),
               "MTXFormSetTranslator": ([S, P], L), "MTXFormSetPrefs": ([S, P], L), "MTGetLastDimension": ([S], L),
               "MTXFormEqn": ([S, S, V, L, S, S, V, L, P, C.POINTER(DIMS)], L), "MTGetPrefsFromFile": ([P, P, S], L)}
        for name, (args, res) in sig.items():
            f = getattr(self.dll, name)
            f.argtypes, f.restype = args, res
        self.clip = _clip_backup()
        r = self.dll.MTAPIConnect(1, timeout)            # mtinitLAUNCH_NOW: start MathType.exe if needed
        self.connect_result = r
        if r < 0:
            raise MTError(f"MTAPIConnect failed: {r} {ERRORS.get(r, '')}")
        self.version = self.dll.MTAPIVersion(5)
        self.functionality = None                          # MTAPIGetAppFunctionality has another signature on x64
        TMP.mkdir(parents=True, exist_ok=True)
        self.prefs = {}

    def close(self):
        try:
            self.dll.MTAPIDisconnect()
        finally:
            _clip_restore(self.clip)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # -------------------------------------------------------------- helpers
    def prefs_from_file(self, name: str) -> bytes:
        if name not in self.prefs:
            buf = C.create_string_buffer(8192)
            r = self.dll.MTGetPrefsFromFile(str(PREFS_DIR / name).encode("mbcs"), buf, 8192)
            if r < 0:
                raise MTError(f"MTGetPrefsFromFile({name}) = {r} {ERRORS.get(r, '')}")
            self.prefs[name] = buf.value
        return self.prefs[name]

    def _xform(self, src_fmt, data: bytes, dst, dst_fmt, cap=1 << 18, path=b" "):
        dims = DIMS()
        src = C.create_string_buffer(data, len(data))
        while True:
            out = C.create_string_buffer(cap) if dst == LOCAL else C.create_string_buffer(1)
            r = self.dll.MTXFormEqn(LOCAL, src_fmt, src, len(data), dst, dst_fmt, out,
                                    cap if dst == LOCAL else 1, path, C.byref(dims))
            if r == -9 and dst == LOCAL and cap < (1 << 24):
                cap *= 4
                continue
            if r != 0:
                raise MTError(f"MTXFormEqn({src_fmt}->{dst_fmt}) = {r} {ERRORS.get(r, '')}")
            n = self.dll.MTXFormGetStatus(STAT_ACTUAL_LEN) if dst == LOCAL else 0
            return (out.raw[:n] if dst == LOCAL and n > 0 else out.raw if dst == LOCAL else b""), dims

    def text_to_mtef(self, text: str, prefs: str = "Times+Symbol 12.eqp") -> bytes:
        self.dll.MTXFormReset()
        if prefs:
            r = self.dll.MTXFormSetPrefs(PREF_USER, self.prefs_from_file(prefs))
            if r != 0:
                raise MTError(f"MTXFormSetPrefs = {r} {ERRORS.get(r, '')}")
        self.dll.MTXFormSetTranslator(0, None)
        mtef, _ = self._xform(F_TEXT, text.encode("ascii"), LOCAL, F_MTEF)
        if not mtef.startswith(b"\x05\x01"):
            # the actual length status may be absent: trim at the first run of trailing zeros is unsafe, so fail
            raise MTError(f"not MTEF: {mtef[:8].hex()}")
        return mtef

    def mtef_to_wmf(self, mtef: bytes, path: Path) -> dict:
        self.dll.MTXFormReset()
        _, dims = self._xform(F_MTEF, mtef, FILE, F_PICT, path=str(path).encode("mbcs"))
        return {"dims": [self.dll.MTGetLastDimension(i) for i in range(1, 6)],
                "baseline": dims.baseline, "bounds": [dims.bounds.left, dims.bounds.top, dims.bounds.right, dims.bounds.bottom]}

    def mtef_to_text(self, mtef: bytes, translator: str = "MathML3 (no namespace).tdl") -> str:
        self.dll.MTXFormReset()
        r = self.dll.MTXFormSetTranslator(0, translator.encode("ascii"))
        if r != 0:
            raise MTError(f"MTXFormSetTranslator({translator}) = {r} {ERRORS.get(r, '')}")
        out, _ = self._xform(F_MTEF, mtef, LOCAL, F_TEXT, cap=1 << 16)
        return out.split(b"\x00", 1)[0].decode("mbcs", errors="replace")
