#!/usr/bin/env bash

printf "WARNING: this will overwrite existing file(s). Proceed? [y/N] "
read -r ans

case "$ans" in
    [Yy]*) ;;                       # yes → continue
    *) echo "Aborted."; exit 1 ;;   # no or empty → bail out
esac

scp ../njyaa_db.* j@revistaea.org:/home/j/Documents/projects/njyaa
