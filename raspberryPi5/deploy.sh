#!/bin/bash
set -e

PI="${1:-root@172.16.68.111}"
DEST="${2:-/opt/HaoYao}"
DIR="$(cd "$(dirname "$0")" && pwd)"

rsync -avzc --delete --dry-run \
    --exclude={.git/,.gitattributes,.gitignore,__pycache__/,*.pyc,*.pyo,.idea/,defect.db,files/,original_files/,detect_files/,fpga/,deploy.sh,CLAUDE.md,README.md} \
    "$DIR/" "$PI:$DEST/"

read -p "确认同步? [y/N] " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rsync -avz --delete \
        --exclude={.git/,.gitattributes,.gitignore,__pycache__/,*.pyc,*.pyo,.idea/,defect.db,files/,original_files/,detect_files/,fpga/,deploy.sh,CLAUDE.md,README.md} \
        "$DIR/" "$PI:$DEST/"
fi
