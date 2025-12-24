#!/bin/bash

#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <target_dir>"
  exit 1
fi

TARGET_DIR="$1"
PYTHON_SCRIPT="eval_alce.py"

for filepath in "$TARGET_DIR"/alce*.json; do
  echo "Executing for: $filepath"
  python "$PYTHON_SCRIPT" --f "$filepath"
done

