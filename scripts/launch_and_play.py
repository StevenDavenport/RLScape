import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rl_scape.launcher import RLScapeLauncher
from rl_scape.bridge import RLBridgeClient


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/launch_and_play.py <server_dir> <client_dir> [java_home] [name]")
        sys.exit(1)

    server_dir = sys.argv[1]
    client_dir = sys.argv[2]
    java_home = sys.argv[3] if len(sys.argv) > 3 else None
    name = sys.argv[4] if len(sys.argv) > 4 else "agent"

    launcher = RLScapeLauncher(
        server_dir=server_dir,
        client_dir=client_dir,
        java_home=java_home,
        port=5656,
        username=name,
        local=True,
    )
    launcher.start()

    client = RLBridgeClient(host="127.0.0.1", port=5656)
    client.connect()
    print(client.ping())
    client.close()

    print("Server + client started. Now run manual play in another terminal:")
    print("  python scripts/manual_play.py 127.0.0.1 5656 1")

    try:
        input("Press Enter to stop server/client... ")
    finally:
        launcher.stop()


if __name__ == "__main__":
    main()
