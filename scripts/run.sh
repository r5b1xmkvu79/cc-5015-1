#!/usr/bin/env bash
# Launch the mesh repair tool.  Adds the locally extracted Qt runtime libs
# (see .bootstrap) to the loader path when present so the app works on hosts
# without system libEGL/libGL/libdbus/libxkbcommon.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

QT_LIBS="$ROOT/.bootstrap/debs/root/usr/lib/aarch64-linux-gnu"
QT_LIBS_2="$ROOT/.bootstrap/debs/root/lib/aarch64-linux-gnu"
if [[ -d "$QT_LIBS" ]]; then
  export LD_LIBRARY_PATH="$QT_LIBS:$QT_LIBS_2${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Offscreen is the safe default on headless hosts; override QT_QPA_PLATFORM
# to xcb/wayland when a display is available.
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
fi

cd "$ROOT"
exec python3 -m mesh_repair.app "$@"
