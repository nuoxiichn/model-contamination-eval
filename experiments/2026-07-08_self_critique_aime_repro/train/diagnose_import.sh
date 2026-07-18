#!/usr/bin/env bash
# 定位 verl import 卡在哪：分别测 3 种 PYTHONPATH 组合，看哪个卡（timeout=卡）。
# 用法：bash diagnose_import.sh
set -u
V=/mnt/public/code/chennuoxi/slow_thinking_sc/thirdparty/verl
T=/mnt/public/code/chennuoxi/slow_thinking_sc/train
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
cd /tmp

echo "=== train/ 目录里有没有自动加载文件 ==="
ls -la "$T"/sitecustomize.py "$T"/usercustomize.py "$T"/conftest.py "$T"/__init__.py 2>/dev/null || echo "  （无 sitecustomize/usercustomize/conftest/__init__）"
echo

echo "=== A) 只 verl 本体（脚本现在用的配置）——期望能过或秒报缺包 ==="
PYTHONPATH="$V" timeout 90 python -c "import verl; print('A OK')"
echo "A exit=$?"
echo

echo "=== B) verl + train/（上一版卡住的配置）——若卡=train 是元凶 ==="
PYTHONPATH="$T:$V" timeout 90 python -c "import verl; print('B OK')"
echo "B exit=$?"
echo

echo "exit 124 = 卡住(timeout)；0 = 成功；1 = 报错(看是不是只缺包)"
