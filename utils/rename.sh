# batch rename
for f in *.jpg; do mv "$f" "${f#reid_}"; done