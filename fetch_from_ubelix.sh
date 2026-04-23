#!/bin/bash
# fetch_from_ubelix.sh — Pull results from UBELIX back to local Mac
# Usage: bash fetch_from_ubelix.sh
# Run from anywhere; only downloads results/ and figures/ folders.

set -euo pipefail

LOCAL_DIR="/Users/amber/VS Code/polar-bear-after-lunch/AareML"
REMOTE="ubelix:~/AareML/"

echo "Fetching results from UBELIX..."
echo "  Remote: $REMOTE"
echo "  Local:  $LOCAL_DIR"
echo ""

rsync -avz --progress \
  -e "ssh -i ~/.ssh/id_ed25519" \
  "$REMOTE"results/ \
  "$LOCAL_DIR/results/"

rsync -avz --progress \
  -e "ssh -i ~/.ssh/id_ed25519" \
  "$REMOTE"figures/ \
  "$LOCAL_DIR/figures/"

rsync -avz --progress \
  -e "ssh -i ~/.ssh/id_ed25519" \
  "$REMOTE"notebooks/ \
  "$LOCAL_DIR/notebooks/"

echo ""
echo "Fetch complete."
