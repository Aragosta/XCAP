#!/usr/bin/env bash
# Fetch the public data this project measures. Nothing here is synthetic.
#
#   FlyWire v783 connectome  - Codex public snapshot (Dorkenwald et al. 2024,
#                              Schlegel et al. 2024)
#   NLTK corpora             - Project Gutenberg, Brown, Reuters
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/flywire data/text

base=https://storage.googleapis.com/flywire-data/codex/data/fafb/783
for f in connections.csv.gz coordinates.csv.gz classification.csv.gz \
         neurons.csv.gz consolidated_cell_types.csv.gz; do
  [ -f "data/flywire/$f" ] || curl -sSL --retry 4 -o "data/flywire/$f" "$base/$f"
done

nltk=https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora
for p in gutenberg brown reuters; do
  [ -d "data/text/$p" ] || {
    curl -sSL --retry 4 -o "data/text/$p.zip" "$nltk/$p.zip"
    (cd data/text && unzip -qo "$p.zip")
  }
done
echo "data ready"
