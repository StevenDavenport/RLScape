import json
import os
import subprocess
import time


def _env_or(default, key):
    value = os.environ.get(key)
    return value if value else default


DEFAULT_PASSWORD = "rl"


def _default_paths():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    server_dir = os.path.join(base, "third_party", "2006scape", "2006Scape Server")
    client_dir = os.path.join(base, "third_party", "2006scape", "2006Scape Client")
    java_home = _env_or("/home/staff/steven/local/jdk8/jdk8u482-b08", "RL_SCAPE_JAVA_HOME")
    mvn_path = _env_or("/home/staff/steven/maven/apache-maven-3.8.6/bin/mvn", "RL_SCAPE_MVN")
    return server_dir, client_dir, java_home


class RLScapeLauncher:
    def __init__(
        self,
        server_dir=None,
        client_dir=None,
        server_config="ServerConfig.json",
        java_home=None,
        mvn_path=None,
        port=5656,
        username="agent",
        local=True,
        headless=True,
        auto_tune=True,
        tune_cycle_times=None,
        tune_timeout_s=8.0,
        tune_avg_ratio=0.9,
    ):
        default_server, default_client, default_java = _default_paths()
        self.server_dir = _env_or(server_dir or default_server, "RL_SCAPE_SERVER_DIR")
        self.client_dir = _env_or(client_dir or default_client, "RL_SCAPE_CLIENT_DIR")
        self.server_config = server_config
        self.java_home = _env_or(java_home or default_java, "RL_SCAPE_JAVA_HOME")
        self.mvn_path = _env_or(mvn_path or "/home/staff/steven/maven/apache-maven-3.8.6/bin/mvn", "RL_SCAPE_MVN")
        self.port = int(port)
        self.username = _env_or(username, "RL_SCAPE_USERNAME")
        self.password = DEFAULT_PASSWORD
        self.local = local
        self.headless = headless
        self.auto_tune = auto_tune
        self.tune_cycle_times = tune_cycle_times
        self.tune_timeout_s = float(tune_timeout_s)
        self.tune_avg_ratio = float(tune_avg_ratio)
        self._server_proc = None
        self._client_proc = None
        self._built_modules = set()
        self._auto_tuned = False

    def _env(self):
        env = os.environ.copy()
        if self.java_home:
            env["JAVA_HOME"] = self.java_home
            env["PATH"] = os.path.join(self.java_home, "bin") + os.pathsep + env.get("PATH", "")
        return env

    def start_server(self):
        if self._server_proc is not None:
            return
        if not os.path.isdir(self.server_dir):
            raise FileNotFoundError(f"Server dir not found: {self.server_dir}")
        self._build("2006Scape Server")
        cmd = [
            "java",
            "-jar",
            "target/server-1.0-jar-with-dependencies.jar",
            "-c",
            self.server_config,
        ]
        self._server_proc = subprocess.Popen(
            cmd,
            cwd=self.server_dir,
            env=self._env(),
        )
        time.sleep(2.0)

    def start_client(self):
        if self._client_proc is not None:
            return
        if not os.path.isdir(self.client_dir):
            raise FileNotFoundError(f"Client dir not found: {self.client_dir}")
        self._build("2006Scape Client")
        cmd = [
            "java",
            "-jar",
            "target/client-1.0-jar-with-dependencies.jar",
            "-rl",
            "-rl-port",
            str(self.port),
        ]
        if self.headless:
            cmd.append("-headless")
        if self.local:
            cmd.append("-local")
        cmd += ["-u", self.username, "-p", self.password]
        print(f"[rl-scape] launching client: cwd={self.client_dir} cmd={' '.join(cmd)}")
        self._client_proc = subprocess.Popen(
            cmd,
            cwd=self.client_dir,
            env=self._env(),
        )
        time.sleep(2.0)

    def _build(self, module_name):
        if module_name in self._built_modules:
            return
        if not self.mvn_path or not os.path.isfile(self.mvn_path):
            raise FileNotFoundError(f"Maven not found: {self.mvn_path}")
        cmd = [
            self.mvn_path,
            "-pl",
            module_name,
            "-am",
            "package",
        ]
        subprocess.run(cmd, cwd=os.path.dirname(self.server_dir), env=self._env(), check=True)
        self._built_modules.add(module_name)

    def _config_path(self):
        return os.path.join(self.server_dir, self.server_config)

    def _load_config(self):
        path = self._config_path()
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_config(self, data):
        path = self._config_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=False)

    def _set_cycle_time_ms(self, ms):
        data = self._load_config()
        data["cycle_time_ms"] = int(ms)
        self._write_config(data)

    def _tick_stats_path(self):
        data = self._load_config()
        rel = data.get("rl_tick_report_file", "data/rl_tick.json")
        return os.path.join(self.server_dir, rel)

    def _read_tick_stats(self):
        path = self._tick_stats_path()
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _wait_for_tick_stats(self, timeout_s):
        start = time.time()
        last = None
        while time.time() - start < timeout_s:
            stats = self._read_tick_stats()
            if stats is not None and stats != last:
                return stats
            time.sleep(0.2)
        return None

    def _auto_tune_cycle_time(self):
        if not self.auto_tune or self._auto_tuned:
            return
        candidates = self.tune_cycle_times or [60, 80, 100, 120, 150, 200, 300, 400, 600]
        print("[rl-scape] auto-tuning server tick...")
        chosen = None
        for ms in candidates:
            print(f"[rl-scape] testing cycle_time_ms={ms}")
            self._set_cycle_time_ms(ms)
            self.start_server()
            stats = self._wait_for_tick_stats(self.tune_timeout_s)
            self.stop()
            if stats is None:
                print("[rl-scape] no tick stats, skipping")
                continue
            overruns = int(stats.get("overruns", 0))
            avg_ms = float(stats.get("avg_ms", ms + 1))
            if overruns == 0 and avg_ms <= ms * self.tune_avg_ratio:
                chosen = ms
                print(f"[rl-scape] stable at {ms}ms (avg {avg_ms:.2f}ms)")
                break
            print(f"[rl-scape] unstable at {ms}ms (avg {avg_ms:.2f}ms, overruns {overruns})")
        if chosen is None:
            chosen = candidates[-1]
            print(f"[rl-scape] falling back to {chosen}ms")
        self._set_cycle_time_ms(chosen)
        self._auto_tuned = True

    def start(self):
        self._auto_tune_cycle_time()
        self.start_server()
        self.start_client()

    def stop(self):
        for proc in (self._client_proc, self._server_proc):
            if proc is None:
                continue
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in (self._client_proc, self._server_proc):
            if proc is None:
                continue
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._client_proc = None
        self._server_proc = None
