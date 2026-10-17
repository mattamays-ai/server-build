#!/bin/sh
# Smart DJ Music Assistant Home Assistant app installer.
# Run from the Home Assistant Terminal & SSH app, with Supervisor access.
set -eu

REPOSITORY="https://github.com/mattamays-ai/server"
SLUG="music_assistant_smartdj"
NIGHTLY_SLUG="music_assistant_nightly"
SUPERVISOR="${SUPERVISOR:-http://supervisor}"
TOKEN="${SUPERVISOR_TOKEN:-}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[ -n "$TOKEN" ] || die "SUPERVISOR_TOKEN is missing. Run inside Home Assistant Terminal & SSH with Supervisor access."
command -v curl >/dev/null 2>&1 || die "curl is required."
command -v grep >/dev/null 2>&1 || die "grep is required."

api() {
  method=$1
  path=$2
  body=${3:-}
  if [ -n "$body" ]; then
    curl --connect-timeout 10 --max-time 120 -fsS -X "$method" \
      -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
      --data "$body" "$SUPERVISOR$path"
  else
    curl --connect-timeout 10 --max-time 120 -fsS -X "$method" \
      -H "Authorization: Bearer $TOKEN" "$SUPERVISOR$path"
  fi
}

is_ok() { printf '%s' "$1" | grep -Eq '"result"[[:space:]]*:[[:space:]]*"ok"'; }

case "${1:-}" in
  --logs)
    exec curl -fsS -N -H "Authorization: Bearer $TOKEN" "$SUPERVISOR/addons/$SLUG/logs/follow"
    ;;
  --rollback)
    echo "Stopping Smart DJ and restarting Music Assistant Nightly..."
    api POST "/addons/$SLUG/stop" >/dev/null 2>&1 || true
    api POST "/addons/$NIGHTLY_SLUG/start" >/dev/null 2>&1 || true
    echo "Rollback commands sent. Check both app states in Home Assistant."
    exit 0
    ;;
  --yes) ASSUME_YES=1 ;;
  "") ASSUME_YES=0 ;;
  *) die "Usage: sh install-smartdj-ma.sh [--yes|--logs|--rollback]" ;;
esac

echo "==> Smart DJ Home Assistant installer"
echo "    Repository: $REPOSITORY"
echo "    App slug:   $SLUG"
echo "    Supervisor: $SUPERVISOR"

# Register this repository (idempotently).
repos=$(api GET "/store/repositories") || die "Cannot reach the Supervisor API at $SUPERVISOR."
if ! printf '%s' "$repos" | grep -Fq "$REPOSITORY"; then
  echo "==> Adding the Smart DJ repository"
  resp=$(api POST "/store/repositories" "{\"repository\":\"$REPOSITORY\"}") || die "Could not register repository: $resp"
  is_ok "$resp" || die "Supervisor rejected repository registration: $resp"
fi

echo "==> Refreshing app store"
api POST "/store/reload" >/dev/null || die "Could not refresh the Home Assistant app store."
sleep 3

# Verify the app is actually visible before trying to install it.
store=$(api GET "/store/addons") || die "Could not list Home Assistant apps."
if ! printf '%s' "$store" | grep -Fq "$SLUG"; then
  echo "ERROR: '$SLUG' is not visible in the app store after adding $REPOSITORY." >&2
  echo "Check that the repository's smartdj-addon/config.yaml is accepted by this Home Assistant version." >&2
  exit 1
fi

# Install only if not already installed; update an existing install to the
# latest store version.
if api GET "/addons/$SLUG/info" >/dev/null 2>&1; then
  echo "==> Smart DJ app is already installed; updating to the latest store version"
  if resp=$(api POST "/store/addons/$SLUG/update"); then
    is_ok "$resp" || echo "NOTE: Supervisor answered $resp"
  else
    echo "NOTE: update call failed (possibly already on the latest version); continuing."
  fi
else
  echo "==> Installing $SLUG (first image download may take several minutes)"
  resp=$(api POST "/store/addons/$SLUG/install") || die "App installation failed: $resp"
  is_ok "$resp" || die "Supervisor did not confirm installation: $resp"
fi

# Confirm install before making any state changes.
info=$(api GET "/addons/$SLUG/info") || die "Install was requested but $SLUG is not available under installed apps."
is_ok "$info" || die "Unexpected app info response: $info"

if [ "$ASSUME_YES" -ne 1 ]; then
  printf "Stop Music Assistant Nightly to avoid port conflicts? [y/N] "
  read -r answer
  case "$answer" in y|Y|yes|YES) ASSUME_YES=1 ;; esac
fi
if [ "$ASSUME_YES" -eq 1 ]; then
  echo "==> Stopping $NIGHTLY_SLUG (if installed)"
  api POST "/addons/$NIGHTLY_SLUG/stop" >/dev/null 2>&1 || echo "Nightly was not running or could not be stopped; check for port conflicts."
else
  echo "NOTE: Nightly was left running. The two apps may conflict on host-network ports."
fi

echo "==> Starting Smart DJ"
resp=$(api POST "/addons/$SLUG/start") || die "Could not start Smart DJ: $resp"
is_ok "$resp" || die "Supervisor did not confirm start: $resp"

echo "==> Waiting for startup logs"
sleep 5
api GET "/addons/$SLUG/logs" 2>/dev/null | tail -n 50 || echo "Logs are not available yet. Run: sh install-smartdj-ma.sh --logs"
echo
echo "Installer requests completed. Check Settings > Apps > Music Assistant Smart DJ for status."
echo "Follow logs: sh install-smartdj-ma.sh --logs"
echo "Rollback:    sh install-smartdj-ma.sh --rollback"
