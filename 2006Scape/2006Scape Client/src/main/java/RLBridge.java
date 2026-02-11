import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

final class RLBridge implements Runnable {

	private static RLBridge instance;

	private final Game game;
	private final int port;
	private final Object frameLock = new Object();
	private volatile boolean running = true;
	private long frameCounter = 0L;
	private byte[] lastRgb;
	private int lastWidth;
	private int lastHeight;
	private int[] lastExp;

	private RLBridge(Game game, int port) {
		this.game = game;
		this.port = port;
	}

	public static void start(Game game, int port) {
		if (instance != null) {
			return;
		}
		instance = new RLBridge(game, port);
		Thread thread = new Thread(instance, "RLBridge");
		thread.setDaemon(true);
		thread.start();
	}

	public static void onFrame() {
		if (instance == null) {
			return;
		}
		instance.signalFrame();
	}

	private void signalFrame() {
		synchronized (frameLock) {
			frameCounter++;
			frameLock.notifyAll();
		}
	}

	@Override
	public void run() {
		try (ServerSocket server = new ServerSocket(port)) {
			while (running) {
				try (Socket socket = server.accept()) {
					handleConnection(socket);
				} catch (IOException e) {
					if (running) {
						e.printStackTrace();
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void handleConnection(Socket socket) throws IOException {
		BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
		DataOutputStream out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
		String line;
		long lastFrame = frameCounter;
		while ((line = in.readLine()) != null) {
			line = line.trim();
			if (line.isEmpty()) {
				continue;
			}
			String[] parts = line.split("\\s+");
			String cmd = parts[0].toUpperCase();
			switch (cmd) {
				case "PING":
					writeLine(out, "PONG");
					break;
				case "MOVE":
					if (parts.length >= 3) {
						game.rlMouseMove(parseInt(parts[1]), parseInt(parts[2]));
						writeLine(out, "OK");
					} else {
						writeLine(out, "ERR");
					}
					break;
				case "DOWN":
					if (parts.length >= 2) {
						game.rlMousePress(parseInt(parts[1]));
						writeLine(out, "OK");
					} else {
						writeLine(out, "ERR");
					}
					break;
				case "UP":
					if (parts.length >= 2) {
						game.rlMouseRelease(parseInt(parts[1]));
						writeLine(out, "OK");
					} else {
						writeLine(out, "ERR");
					}
					break;
				case "DRAG":
					if (parts.length >= 3) {
						game.mouseWheelDragged(parseInt(parts[1]), parseInt(parts[2]));
						writeLine(out, "OK");
					} else {
						writeLine(out, "ERR");
					}
					break;
				case "STEP":
					lastFrame = waitForNextFrame(lastFrame);
					sendFrame(out);
					break;
				case "FRAME":
					sendFrame(out);
					lastFrame = frameCounter;
					break;
				case "STATE":
					sendState(out);
					break;
				case "READY":
					sendReady(out);
					break;
				case "QUIT":
					writeLine(out, "BYE");
					return;
				default:
					writeLine(out, "ERR");
					break;
			}
		}
	}

	private long waitForNextFrame(long lastFrame) {
		synchronized (frameLock) {
			while (frameCounter <= lastFrame) {
				try {
					frameLock.wait(1000L);
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
					break;
				}
			}
			return frameCounter;
		}
	}

	private void sendFrame(DataOutputStream out) throws IOException {
		int[] pixels = game.getHeadlessPixels();
		if (pixels == null) {
			if (lastRgb != null) {
				int len = lastWidth * lastHeight * 3;
				writeLine(out, "FRAME " + lastWidth + " " + lastHeight + " 3 " + len);
				out.write(lastRgb);
				out.flush();
			} else {
				writeLine(out, "ERR no-headless");
			}
			return;
		}
		int width = game.getHeadlessWidth();
		int height = game.getHeadlessHeight();
		int len = width * height * 3;
		byte[] rgb = new byte[len];
		int idx = 0;
		for (int p : pixels) {
			rgb[idx++] = (byte) ((p >> 16) & 0xff);
			rgb[idx++] = (byte) ((p >> 8) & 0xff);
			rgb[idx++] = (byte) (p & 0xff);
		}
		lastRgb = rgb;
		lastWidth = width;
		lastHeight = height;
		writeLine(out, "FRAME " + width + " " + height + " 3 " + len);
		out.write(rgb);
		out.flush();
	}

	private void sendState(DataOutputStream out) throws IOException {
		long totalExp = game.getRlTotalExp();
		int totalLevels = game.getRlTotalLevels();
		int hp = game.getRlCurrentHp();
		int maxHp = game.getRlMaxHp();
		int anim = game.getRlAnim();
		int interacting = game.getRlInteractingEntity();
		int loopCycle = game.getRlLoopCycle();
		int skillIndex = -1;
		int skillDelta = 0;
		int[] currentExp = game.getRlCurrentExp();
		if (currentExp != null) {
			if (lastExp == null || lastExp.length != currentExp.length) {
				lastExp = new int[currentExp.length];
				for (int i = 0; i < currentExp.length; i++) {
					lastExp[i] = currentExp[i];
				}
			} else {
				for (int i = 0; i < currentExp.length; i++) {
					int delta = currentExp[i] - lastExp[i];
					if (delta > 0) {
						if (delta > skillDelta) {
							skillDelta = delta;
							skillIndex = i;
						}
						lastExp[i] = currentExp[i];
					}
				}
			}
		}
		writeLine(out, "STATE " + totalExp + " " + totalLevels + " " + hp + " " + maxHp + " " + anim + " " + interacting + " " + loopCycle + " " + skillIndex + " " + skillDelta);
	}

	private void sendReady(DataOutputStream out) throws IOException {
		boolean ready = game.isRlReady();
		writeLine(out, "READY " + (ready ? "1" : "0"));
	}

	private void writeLine(DataOutputStream out, String line) throws IOException {
		out.write((line + "\n").getBytes("UTF-8"));
		out.flush();
	}

	private int parseInt(String value) {
		try {
			return Integer.parseInt(value);
		} catch (NumberFormatException e) {
			return 0;
		}
	}
}
