#!/usr/bin/env bash
# Downloads the third-party data and the ganak binary used by the artefact. Nothing here is versioned.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/data"
mkdir -p "$D/dimacs" "$D/bin"

# Whole-population measurements and variability models (GPL-2.0)
if [ ! -d "$D/Distance-Based_Data" ]; then
  git clone --depth 1 https://github.com/ChristianKaltenecker/Distance-Based_Data "$D/Distance-Based_Data"
fi
M="$D/Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues"
# JavaGC and VP9 ship their measurements compressed
for s in JavaGC VP9; do
  if [ -f "$M/$s/measurements.tar.gz" ] && [ ! -f "$M/$s/.extracted" ]; then
    tar -xzf "$M/$s/measurements.tar.gz" -C "$M/$s" && touch "$M/$s/.extracted"
  fi
done

# Large models in DIMACS from the feature-model benchmark (MIT), pinned to one commit
FMB=https://raw.githubusercontent.com/SoftVarE-Group/feature-model-benchmark/afa60ee2c836e7bdc4068e0f4f128ea31158d2ad/feature_models/dimacs
fetch() { [ -s "$D/dimacs/$1" ] || curl -fsSL -o "$D/dimacs/$1" "$FMB/$2"; }
fetch berkeleydb.dimacs     database/berkeleydb/Hierons2020.dimacs
fetch financial01.dimacs    finance/FinancialServices01/Nieke2018-2018-05-09.dimacs
fetch busybox-1.18.0.dimacs systems_software/BusyBox/Kn%C3%BCppel2017-1.18.0.dimacs
fetch linux-2.6.33.3.dimacs systems_software/Linux/Kn%C3%BCppel2017-2.6.33.3.dimacs

# ganak 2.8.0 release binary for this platform; use it with --prob 0 for deterministic exact counts
if [ ! -x "$D/bin/ganak" ]; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  G=linux-amd64 ;;
    Linux-aarch64) G=linux-arm64 ;;
    Darwin-arm64)  G=mac-arm64 ;;
    Darwin-x86_64) G=mac-x86_64 ;;
    *) echo "No ganak build for $(uname -s)-$(uname -m); use Linux, macOS or WSL2" >&2; exit 1 ;;
  esac
  curl -fsSL "https://github.com/meelgroup/ganak/releases/download/release/v2.8.0/ganak-v2.8.0-$G.tar.gz" \
    | tar -xz -C "$D/bin" ganak
fi
"$D/bin/ganak" --version 2>&1 | sed -n 1p
echo "Data ready under $D"
