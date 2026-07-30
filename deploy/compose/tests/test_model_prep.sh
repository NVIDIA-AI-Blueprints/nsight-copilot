#!/usr/bin/env bash
# Regression test for deploy/compose/model-prep.sh HF_HOME default.
set -euo pipefail

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
SCRIPT=$SCRIPT_DIR/../model-prep.sh
TMPDIR=$(mktemp -d)
trap 'rm -rf $TMPDIR' EXIT

export HOME=$TMPDIR/home
mkdir -p $HOME

mkdir -p $TMPDIR/bin
cat > $TMPDIR/bin/hf <<'EOF'
#!/usr/bin/env bash
if [[ ${1:-} != download ]]; then
  echo 'fake hf: unexpected subcommand: '${1:-} >&2
  exit 1
fi
LOCAL_DIR=
for ((i = 1; i <= $#; i++)); do
  if [[ ${!i} == --local-dir ]]; then
    j=$((i + 1))
    LOCAL_DIR=${!j}
  fi
done
if [[ -z $LOCAL_DIR ]]; then
  echo 'fake hf: --local-dir not found' >&2
  exit 1
fi
mkdir -p $LOCAL_DIR
EOF
chmod +x $TMPDIR/bin/hf
export PATH=$TMPDIR/bin:$PATH

unset HF_HOME
cd $TMPDIR
bash $SCRIPT

[[ -f $TMPDIR/.cache/huggingface/.check_for_update_done ]] || { echo 'FAIL: .check_for_update_done missing'; exit 1; }
[[ -d $TMPDIR/models/bge-m3 ]] || { echo 'FAIL: models/bge-m3 missing'; exit 1; }
echo 'PASS'
