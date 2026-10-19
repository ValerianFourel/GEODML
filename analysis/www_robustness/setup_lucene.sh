#!/bin/bash
# ARR author response: one-time setup on a HoreKa LOGIN node (compute nodes need no network) for lucene_check.py.
# Downloads a standalone Java 21 (Eclipse Temurin) and Anserini's fat jar from Maven Central into
#   $W/environment/lucene/{jdk,anserini-<version>-fatjar.jar}
# Nothing else is installed and no existing environment is changed. About 350 MB. Prints the sha256 of both downloads.
# Usage: bash setup_lucene.sh [ANSERINI_VERSION]   (default: the latest release on Maven Central, printed)
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
META=https://repo1.maven.org/maven2/io/anserini/anserini/maven-metadata.xml
V=${1:-$(curl -fsL "$META" | sed -n 's:.*<release>\(.*\)</release>.*:\1:p')}
[ -n "$V" ] || { echo "no Anserini version found at $META" >&2; exit 2; }
echo "Anserini $V"
D="$W/environment/lucene"
mkdir -p "$D"
if [ ! -x "$D/jdk/bin/java" ]; then
  curl -fL --retry 3 -o "$D/jdk21.tar.gz" \
    "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse?project=jdk"
  mkdir -p "$D/jdk" && tar -xzf "$D/jdk21.tar.gz" -C "$D/jdk" --strip-components=1
fi
JAR="$D/anserini-$V-fatjar.jar"
[ -s "$JAR" ] || curl -fL --retry 3 -o "$JAR" "https://repo1.maven.org/maven2/io/anserini/anserini/$V/anserini-$V-fatjar.jar"
"$D/jdk/bin/java" -version 2>&1 | head -1
sha256sum "$D/jdk21.tar.gz" "$JAR"
"$D/jdk/bin/java" -cp "$JAR" io.anserini.index.IndexCollection -options 2>&1 | head -3 || true
echo "LUCENE_JAR=$JAR"
