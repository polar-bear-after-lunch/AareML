#!/bin/bash
# sync_to_ubelix.sh — Push local AareML code to UBELIX
# Usage: bash sync_to_ubelix.sh
# Run from anywhere; the script finds the AareML directory automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="ubelix:~/AareML/"

echo "Syncing AareML to UBELIX..."
echo "  Local:  $SCRIPT_DIR"
echo "  Remote: $REMOTE"
echo ""

rsync -avz --progress \
  --exclude 'data/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude '.pytest_cache/' \
  --exclude 'results/' \
  --exclude 'figures/' \
  --exclude '*.pdf' \
  --exclude '*.zip' \
  --exclude 'AareML-*.pdf' \
  --exclude 'AareML-*.zip' \
  --exclude 'create_notebook_*.py' \
  --exclude 'README.md' \
  -e "ssh -i ~/.ssh/id_ed25519" \
  "$SCRIPT_DIR/" \
  "$REMOTE"

echo ""
echo "Sync complete."
