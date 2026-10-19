# iNysha desktop client

Thin Python client for the cloud iNysha agent. It exposes **screen,
microphone and speakers** and keeps the actual reasoning, VAD, STT and TTS on
the VDS.

The same code runs on **Ubuntu and Windows**. The only platform-specific parts
are hidden behind cross-platform libraries:

| Capability | Library |
|---|---|
| Microphone / speaker | `sounddevice` (PortAudio) |
| Screen capture | `mss` + Pillow (X11; Wayland fallback via `grim`) |
| Global hotkey | `pynput` |
| Transport | `websockets` |

## Install

Use Python 3.11+.

### Ubuntu

```bash
sudo apt update
sudo apt install -y libportaudio2 libx11-6
python3 -m venv .venv-client
source .venv-client/bin/activate
pip install -r requirements-client.txt
```

`pynput` global hotkeys and `mss` screen capture need an **X11 session**.
On a Wayland session the client still works, but the global hotkey may be
blocked; press `v` in the console instead. Screen capture falls back to
`grim`, `gnome-screenshot` or `spectacle` if one is installed.

### Windows

```powershell
py -3.11 -m venv .venv-client
.venv-client\Scripts\activate
pip install -r requirements-client.txt
```

The `sounddevice` wheel bundles PortAudio on Windows; no system package is
needed. Allow Python to use the microphone in Windows privacy settings.

## Configure

Copy `client/.env.example` to `client/.env` (recommended for a desktop copy)
or to the directory you run the client from, and fill in:

```dotenv
SERVER_URL=ws://2.24.160.163:8010
CLIENT_TOKEN=<same value as the server's CLIENT_TOKEN>
CLIENT_ID=desktop-andrey
CLIENT_NAME=Andrey desktop
CLIENT_HOTKEY=<f8>
```

`CLIENT_TOKEN` is **not** the MAX webhook secret. It is a separate shared
secret between the VDS and desktop clients.

## Run

From the project root:

```bash
python -m client
```

or:

```bash
python -m client.main
```

Console commands:

- `v` — start/stop a voice session
- `s` — take a local test screenshot
- `d` — list audio devices
- `q` — quit
- any other line of **2 or more characters** — sent to the server agent as a
  normal text prompt; the answer is printed in the console

The global hotkey (default `F8`) toggles the same voice session.

## How voice works

1. The client streams raw 16 kHz / 16-bit / mono PCM from the microphone.
2. The server runs WebRTC VAD and segments utterances.
3. Groq Whisper transcribes the utterance.
4. The shared agent answers through the serialized queue.
5. ElevenLabs streams `pcm_16000` back; the client plays it and reports
   `playback_done`, which re-opens the microphone.

The mic is muted while the agent is thinking/speaking, so the assistant never
transcribes its own voice.

## Troubleshooting

- **No microphone**: run `python -m client` and `d`, then set
  `CLIENT_INPUT_DEVICE=...` to the index or a name substring.
- **No sound**: same for `CLIENT_OUTPUT_DEVICE`.
- **`PortAudio` missing on Ubuntu**: install `libportaudio2`.
- **Hotkey does nothing on Ubuntu**: use an X11 session, or use `v`.
- **`screen capture failed`**: install `grim` (Wayland) or run an X11 session.
- **401/close 4001**: `CLIENT_TOKEN` does not match the server.
