#!/usr/bin/env bash
# Download the Elasticsearch distribution for this machine into vendor/.
#
# The distribution is ~450MB and platform-specific, so it is not committed to
# the repo. This script picks the right build for the current OS/architecture.
set -euo pipefail

VERSION="${ES_VERSION:-8.17.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor"

if [[ -d "$VENDOR/elasticsearch-$VERSION" ]]; then
  echo "Elasticsearch $VERSION already present at $VENDOR/elasticsearch-$VERSION"
  exit 0
fi

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)   PLATFORM="darwin-aarch64" ;;
  Darwin-x86_64)  PLATFORM="darwin-x86_64"  ;;
  Linux-aarch64)  PLATFORM="linux-aarch64"  ;;
  Linux-x86_64)   PLATFORM="linux-x86_64"   ;;
  *)
    echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2
    echo "Download manually from https://www.elastic.co/downloads/elasticsearch" >&2
    echo "and extract into $VENDOR/" >&2
    exit 1
    ;;
esac

URL="https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-${VERSION}-${PLATFORM}.tar.gz"

mkdir -p "$VENDOR"
echo "downloading elasticsearch ${VERSION} (${PLATFORM}) ..."
echo "  $URL"
echo "  ~450MB - this takes a few minutes"

curl -fL --retry 3 --max-time 1800 -o "$VENDOR/es.tar.gz" "$URL"
tar xzf "$VENDOR/es.tar.gz" -C "$VENDOR"
rm -f "$VENDOR/es.tar.gz"

echo "extracted to $VENDOR/elasticsearch-$VERSION"
echo "next: ./scripts/start_search.sh && make search"
