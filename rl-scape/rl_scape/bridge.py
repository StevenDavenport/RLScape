import socket
import time
import struct


class RLBridgeClient:
    def __init__(self, host="127.0.0.1", port=5656, timeout=10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock = None
        self._file = None

    def connect(self):
        if self._sock is not None:
            return
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._file = self._sock.makefile("rb")

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _send_line(self, line: str):
        if self._sock is None:
            raise RuntimeError("Not connected")
        self._sock.sendall((line + "\n").encode("utf-8"))

    def _readline(self) -> bytes:
        if self._file is None:
            raise RuntimeError("Not connected")
        line = self._file.readline()
        if not line:
            raise RuntimeError("Connection closed")
        return line

    def ping(self) -> str:
        self._send_line("PING")
        return self._readline().decode("utf-8").strip()

    def move(self, x: int, y: int):
        self._send_line(f"MOVE {x} {y}")
        return self._readline().decode("utf-8").strip()

    def down(self, button: int):
        self._send_line(f"DOWN {button}")
        return self._readline().decode("utf-8").strip()

    def up(self, button: int):
        self._send_line(f"UP {button}")
        return self._readline().decode("utf-8").strip()

    def drag(self, dx: int, dy: int):
        self._send_line(f"DRAG {dx} {dy}")
        return self._readline().decode("utf-8").strip()

    def step(self):
        self._send_line("STEP")
        return self._read_frame_with_retry()

    def frame(self):
        self._send_line("FRAME")
        return self._read_frame_with_retry()

    def state(self):
        self._send_line("STATE")
        line = self._readline().decode("utf-8").strip()
        parts = line.split()
        if len(parts) < 10 or parts[0] != "STATE":
            raise RuntimeError(f"Bad state header: {line}")
        return {
            "total_xp": int(parts[1]),
            "total_levels": int(parts[2]),
            "hp": int(parts[3]),
            "max_hp": int(parts[4]),
            "anim": int(parts[5]),
            "interacting": int(parts[6]),
            "loop_cycle": int(parts[7]),
            "skill_index": int(parts[8]),
            "skill_delta": int(parts[9]),
        }

    def ready(self) -> bool:
        self._send_line("READY")
        line = self._readline().decode("utf-8").strip()
        parts = line.split()
        if len(parts) < 2 or parts[0] != "READY":
            raise RuntimeError(f"Bad ready header: {line}")
        return parts[1] == "1"

    def _read_frame(self):
        header = self._readline().decode("utf-8").strip()
        if header == "ERR no-headless":
            # allow a short retry window during startup
            for _ in range(20):
                time.sleep(0.05)
                header = self._readline().decode("utf-8").strip()
                if header and not header.startswith("ERR"):
                    break
        parts = header.split()
        if len(parts) < 5 or parts[0] != "FRAME":
            raise RuntimeError(f"Bad frame header: {header}")
        width = int(parts[1])
        height = int(parts[2])
        channels = int(parts[3])
        length = int(parts[4])
        data = self._file.read(length)
        if data is None or len(data) != length:
            raise RuntimeError("Incomplete frame data")
        return width, height, channels, data

    def _read_frame_with_retry(self, timeout_s=15.0):
        deadline = time.time() + timeout_s
        last_err = None
        while time.time() < deadline:
            try:
                return self._read_frame()
            except (TimeoutError, RuntimeError, OSError) as err:
                last_err = err
                try:
                    self.close()
                except Exception:
                    pass
                time.sleep(0.1)
                try:
                    self.connect()
                except Exception:
                    time.sleep(0.2)
        if last_err is not None:
            raise last_err
        raise RuntimeError("Failed to read frame")
