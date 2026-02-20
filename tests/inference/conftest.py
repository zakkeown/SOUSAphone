"""Conftest for inference tests - patches missing _lzma if needed."""

import sys
import types


def _ensure_lzma():
    """Provide a stub _lzma module if the C extension is missing.

    This happens on pyenv-built Pythons where xz/lzma dev headers were not
    available at compile time.  librosa -> pooch -> lzma -> _lzma is the
    import chain that triggers the error.
    """
    try:
        import _lzma  # noqa: F401

        return  # already available
    except ImportError:
        pass

    _lzma = types.ModuleType("_lzma")

    # Constants expected by the stdlib lzma module
    _lzma.CHECK_NONE = 0
    _lzma.CHECK_CRC32 = 1
    _lzma.CHECK_CRC64 = 4
    _lzma.CHECK_SHA256 = 10
    _lzma.CHECK_ID_MAX = 15
    _lzma.CHECK_UNKNOWN = 16
    _lzma.FILTER_LZMA1 = 0x4000000000000001
    _lzma.FILTER_LZMA2 = 0x21
    _lzma.FILTER_DELTA = 0x03
    _lzma.FILTER_X86 = 0x04
    _lzma.FILTER_IA64 = 0x06
    _lzma.FILTER_ARM = 0x07
    _lzma.FILTER_ARMTHUMB = 0x08
    _lzma.FILTER_POWERPC = 0x05
    _lzma.FILTER_SPARC = 0x09
    _lzma.FORMAT_AUTO = 0
    _lzma.FORMAT_XZ = 1
    _lzma.FORMAT_ALONE = 2
    _lzma.FORMAT_RAW = 3
    _lzma.MF_HC3 = 0x03
    _lzma.MF_HC4 = 0x04
    _lzma.MF_BT2 = 0x12
    _lzma.MF_BT3 = 0x13
    _lzma.MF_BT4 = 0x14
    _lzma.MODE_FAST = 1
    _lzma.MODE_NORMAL = 2
    _lzma.PRESET_DEFAULT = 6
    _lzma.PRESET_EXTREME = 1 << 31
    _lzma._encode_filter_properties = lambda f: b""
    _lzma._decode_filter_properties = lambda fid, data: {}
    _lzma.is_check_supported = lambda check_id: True

    class LZMAError(Exception):
        pass

    _lzma.LZMAError = LZMAError

    class LZMACompressor:
        def __init__(self, *args, **kwargs):
            pass

        def compress(self, data):
            return data

        def flush(self):
            return b""

    class LZMADecompressor:
        def __init__(self, *args, **kwargs):
            self.eof = True
            self.needs_input = False
            self.unused_data = b""

        def decompress(self, data, max_length=-1):
            return data

    _lzma.LZMACompressor = LZMACompressor
    _lzma.LZMADecompressor = LZMADecompressor

    sys.modules["_lzma"] = _lzma


# Patch before any test imports librosa
_ensure_lzma()
