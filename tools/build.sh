#!/bin/bash
# Build the Race env.
#
# race/ is an overlay: it is symlinked into the pinned PufferLib checkout at
# third_party/pufferlib so upstream stays byte-identical and `git submodule
# status` stays clean. Nothing in third_party/ is ever edited.
#
# Usage:
#   tools/build.sh eval      # headless eval CLI          -> build/race_eval
#   tools/build.sh play      # replay client (mp4 export)  -> build/race_play
#   tools/build.sh demo      # interactive raylib demo    -> build/race
#   tools/build.sh module    # python extension (_C.so) via pufferlib build.sh
#   tools/build.sh link      # just refresh the overlay symlinks
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUFFER="$ROOT/third_party/pufferlib"
TARGET="${1:-eval}"

if [ ! -d "$PUFFER/ocean" ]; then
    echo "error: $PUFFER is empty. Run: git submodule update --init --recursive" >&2
    exit 1
fi

# --- overlay ---------------------------------------------------------------
mkdir -p "$ROOT/build"
rm -rf "$PUFFER/ocean/race"
ln -s "$ROOT/race" "$PUFFER/ocean/race"
ln -sf "$ROOT/config/race.ini" "$PUFFER/config/race.ini"
[ "$TARGET" = "link" ] && { echo "overlay linked"; exit 0; }

# --- raylib ----------------------------------------------------------------
PLATFORM="$(uname -s)"
ARCH="$(uname -m)"
if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
    FRAMEWORKS=(-lGL)
else
    RAYLIB_NAME='raylib-5.5_macos'
    FRAMEWORKS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
fi
if [ ! -d "$PUFFER/$RAYLIB_NAME" ]; then
    echo "Downloading $RAYLIB_NAME..."
    curl -sL "https://github.com/raysan5/raylib/releases/download/5.5/$RAYLIB_NAME.tar.gz" \
        -o "$PUFFER/$RAYLIB_NAME.tar.gz"
    tar xf "$PUFFER/$RAYLIB_NAME.tar.gz" -C "$PUFFER"
    rm "$PUFFER/$RAYLIB_NAME.tar.gz"
fi

# PufferLib's build.sh hardcodes -mavx2 -mfma (x86 only; drive.h and bf16.h use
# AVX2 intrinsics directly) and -fopenmp (Apple clang rejects it outright).
# Race needs neither, so the standalone targets compile directly.
SIMD=()
[ "$ARCH" = "x86_64" ] && SIMD=(-mavx2 -mfma)

INCLUDES=(-I"$PUFFER/$RAYLIB_NAME/include" -I"$PUFFER/src" -I"$PUFFER/vendor" -I"$ROOT/race")
WARN=(-Wall -Werror=return-type -Werror=incompatible-pointer-types -ferror-limit=5)

case "$TARGET" in
  eval)
    echo "Compiling race_eval ($ARCH)..."
    clang -O2 -DNDEBUG "${WARN[@]}" ${SIMD[@]+"${SIMD[@]}"} "${INCLUDES[@]}" \
        "$ROOT/race/race_eval.c" -o "$ROOT/build/race_eval" \
        "$PUFFER/$RAYLIB_NAME/lib/libraylib.a" "${FRAMEWORKS[@]}" \
        -lm -lpthread -DPLATFORM_DESKTOP
    echo "Built: build/race_eval"
    ;;
  capi)
    echo "Compiling libracecapi ($ARCH)..."
    clang -O2 -DNDEBUG -fPIC -shared "${WARN[@]}" ${SIMD[@]+"${SIMD[@]}"} "${INCLUDES[@]}" \
        "$ROOT/race/race_capi.c" -o "$ROOT/build/libracecapi.dylib" \
        "$PUFFER/$RAYLIB_NAME/lib/libraylib.a" "${FRAMEWORKS[@]}" \
        -lm -lpthread -DPLATFORM_DESKTOP 2>/dev/null \
      || clang -O2 -DNDEBUG -fPIC -shared "${WARN[@]}" ${SIMD[@]+"${SIMD[@]}"} "${INCLUDES[@]}" \
        "$ROOT/race/race_capi.c" -o "$ROOT/build/libracecapi.so" \
        "$PUFFER/$RAYLIB_NAME/lib/libraylib.a" "${FRAMEWORKS[@]}" \
        -lm -lpthread -DPLATFORM_DESKTOP
    echo "Built: build/libracecapi"
    ;;
  play)
    echo "Compiling race_play ($ARCH)..."
    clang -O2 -DNDEBUG "${WARN[@]}" ${SIMD[@]+"${SIMD[@]}"} "${INCLUDES[@]}" \
        "$ROOT/race/race_play.c" -o "$ROOT/build/race_play" \
        "$PUFFER/$RAYLIB_NAME/lib/libraylib.a" "${FRAMEWORKS[@]}" \
        -lm -lpthread -DPLATFORM_DESKTOP
    echo "Built: build/race_play"
    ;;
  demo)
    echo "Compiling race demo ($ARCH)..."
    clang -O2 -DNDEBUG "${WARN[@]}" ${SIMD[@]+"${SIMD[@]}"} "${INCLUDES[@]}" \
        "$ROOT/race/race.c" -o "$ROOT/build/race" \
        "$PUFFER/$RAYLIB_NAME/lib/libraylib.a" "${FRAMEWORKS[@]}" \
        -lm -lpthread -DPLATFORM_DESKTOP
    echo "Built: build/race"
    ;;
  module)
    # Delegates to pufferlib's build.sh, which needs OpenMP. Apple clang has no
    # OpenMP, so this target is Linux/container only (see docker/).
    echo "Building python module via pufferlib build.sh..."
    ( cd "$PUFFER" && bash build.sh race --cpu )
    ;;
  *)
    echo "usage: tools/build.sh [eval|play|capi|demo|module|link]" >&2
    exit 1
    ;;
esac
