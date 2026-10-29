#!/usr/bin/env bash
# Hash list for the freeze message (section 12): sha256sum of every file, paths relative to the package root, and the
# git commit on its own line. Text files over 1 MB are listed separately so they can be declared.
set -euo pipefail
cd "$(dirname "$0")"
find . -type f ! -path './.git/*' ! -name 'MANIFEST.txt' -print0 | sort -z | xargs -0 sha256sum > MANIFEST.txt
git rev-parse HEAD 2>/dev/null >> MANIFEST.txt || echo "no-git" >> MANIFEST.txt
echo "files: $(( $(wc -l < MANIFEST.txt) - 1 ))"
echo "text files over 1 MB:"; find . -type f -size +1M ! -path './.git/*' -exec file {} \; | grep -i text || echo "  none"
