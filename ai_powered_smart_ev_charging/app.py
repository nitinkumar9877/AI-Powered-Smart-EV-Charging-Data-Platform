import importlib.util
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
APP_FILE = APP_DIR / "streamlit_app.py"
DEFAULT_PORT = 8501

REQUIRED_PACKAGES = {
    "streamlit": "streamlit>=1.36.0",
    "plotly": "plotly>=5.20.0",
    "folium": "folium>=0.16.0",
    "streamlit_folium": "streamlit-folium>=0.22.0",
}


def ensure_dependencies() -> None:
    missing = [spec for module, spec in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return
    print("Installing web application dependencies:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def available_port(start: int = DEFAULT_PORT) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("localhost", port)) != 0:
                return port
    raise RuntimeError("No available localhost port found for the Streamlit app.")


def main() -> None:
    ensure_dependencies()
    port = available_port()
    url = f"http://localhost:{port}"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.port",
        str(port),
        "--server.address",
        "localhost",
        "--browser.gatherUsageStats",
        "false",
    ]
    process = subprocess.Popen(cmd, cwd=APP_DIR)
    time.sleep(2)
    webbrowser.open(url)
    print(f"EV Charging Intelligence Platform is running at {url}")
    process.wait()


if __name__ == "__main__":
    main()
