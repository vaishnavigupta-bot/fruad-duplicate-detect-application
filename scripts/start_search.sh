#!/usr/bin/env bash
# Start a local single-node Elasticsearch for the dedupe benchmark.
#
# Security is disabled on purpose: this node listens on localhost only, holds
# nothing but synthetic data, and exists to measure query latency. Do not use
# this configuration for anything real.
#
#   ./scripts/start_search.sh          start (foreground logs tailed)
#   ./scripts/start_search.sh stop     stop
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ES_HOME="$(find "$ROOT/vendor" -maxdepth 1 -type d -name 'elasticsearch-*' | head -1)"
PID_FILE="$ROOT/vendor/es.pid"
LOG_FILE="$ROOT/vendor/es.log"

if [[ "${1:-start}" == "stop" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "stopped."
  else
    echo "not running (no pid file)."
  fi
  exit 0
fi

if [[ -z "$ES_HOME" ]]; then
  echo "Elasticsearch not found under $ROOT/vendor/." >&2
  echo "Run ./scripts/fetch_search.sh to download it for this platform." >&2
  exit 1
fi

CONFIG="$ES_HOME/config/elasticsearch.yml"
if ! grep -q "^xpack.security.enabled" "$CONFIG" 2>/dev/null; then
  cat >> "$CONFIG" <<'YAML'

# --- local benchmark configuration (synthetic data, localhost only) ---
xpack.security.enabled: false
xpack.security.enrollment.enabled: false
xpack.security.http.ssl.enabled: false
xpack.security.transport.ssl.enabled: false
discovery.type: single-node
network.host: 127.0.0.1
http.port: 9200
YAML
fi

# 2GB heap is plenty for 10k documents and keeps the footprint modest.
export ES_JAVA_OPTS="${ES_JAVA_OPTS:--Xms2g -Xmx2g}"

echo "starting Elasticsearch from $ES_HOME ..."
"$ES_HOME/bin/elasticsearch" -d -p "$PID_FILE" > "$LOG_FILE" 2>&1

printf "waiting for http://127.0.0.1:9200 "
for _ in $(seq 1 90); do
  if curl -fs http://127.0.0.1:9200 >/dev/null 2>&1; then
    echo ""
    curl -s http://127.0.0.1:9200 | head -12
    exit 0
  fi
  printf "."
  sleep 2
done

echo ""
echo "did not come up in time; last 40 log lines:" >&2
tail -40 "$LOG_FILE" >&2
exit 1
