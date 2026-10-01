# 完整功能測試命令

這份文件用來逐項驗證目前專案。請依照順序執行；前一項結束後再開始
下一項，避免 Isaac、bridge 8100 port 或 GPU 資源互相衝突。

所有測試產物都寫入 `outputs/test/full_test/`。除非段落明確要求，請勿加入
`--enable-motion`。

## 執行環境分工

三種環境不可混用。開啟新的 terminal 後，依工作內容選擇其中一種：

| 工作 | 環境 | 啟動方式 |
| --- | --- | --- |
| Isaac、Remote、H5、ROS | 共用 Isaac/ROS | `source /opt/isaac_ros2/setup.bash` |
| TFDS/RLDS build | `rlds_env` | `conda activate rlds_env` |
| OpenVLA server／訓練 | `openvla` | `conda activate openvla` |

不要在共用 Isaac 環境安裝 OpenVLA 或 TensorFlow 套件。`run_vla.py` 本身是
Isaac 程式，仍使用共用 Isaac 環境；只有另一個 terminal 中的
`vla_server.py` 使用 `openvla`。

標示為「需要實體」的段落只在有人位於機器人旁、急停可用且工作區清空時
執行。其餘段落不會發送實體運動命令。

## 0. 測試前確認

先載入共用 Isaac 環境，再確認版本、工作目錄、單元測試與基本 Python
語法：

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash

git status --short
git branch --show-current
git log -1 --oneline

git diff --check
python -m unittest discover -s tests -v
python -m compileall -q vla_sim scripts
```

預期：

- `git status --short` 沒有輸出。
- commit 與本次要驗證或交接的版本一致。
- 單元測試全部顯示 `ok`。
- `git diff --check` 與 `compileall` 都沒有錯誤輸出。

確認 GPU 與 8100、8000 port：

```bash
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
ss -ltnp | grep -E ':(8100|8000)\b' || true
```

開始前不應有舊的 Isaac 或 VLA server 佔用這兩個 port。

## 1. Isaac Lab 共用環境

每一個要執行 Isaac 的遠端 terminal 都先執行：

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash
```

一般功能測試只需要上述指令。新帳號首次使用時，可在自己的終端
確認共用安裝可讀、ROS overlay 與 CUDA 可用：

```bash
python3 -c 'import isaaclab, rclpy, torch; print(isaaclab.__file__, rclpy.__file__, torch.cuda.get_device_name(0))'
ros2 pkg prefix vla_ros_bridge
```

共用工作站的安裝、升級與跨帳號驗證屬於管理員操作，與專案功能測試分開。

## 2. 純場景

### 2.1 Headless smoke test

```bash
isaaclab scripts/run_scene.py \
  --headless --enable_cameras --max-steps 180
```

預期看到以下訊息，接著程式自行結束：

```text
Canonical scene running. Ctrl+C closes Isaac.
```

### 2.2 遠端桌面畫面確認

在遠端桌面的 terminal 中確認 `DISPLAY`：

```bash
echo "$DISPLAY"
```

接著啟動 GUI：

```bash
cd ~/IsaacLab/ur3e_vla
isaaclab scripts/run_scene.py \
  --enable_cameras
```

確認：

- 湯匙、紅杯、深藍灰碗完整出現在場景。
- 三個彩色目標點預設隱藏。
- YOLO 視角構圖正確。
- 檢查完按 `Ctrl+C`，等待程式完整結束。

只有要除錯目標點時才使用：

```bash
isaaclab scripts/run_scene.py \
  --enable_cameras --show-markers
```

## 3. Bridge、直播與三條正式任務

### 3.1 Terminal A：啟動 Isaac 與 bridge

```bash
isaaclab scripts/run_remote_pick_place.py \
  --headless --enable_cameras --seed 42
```

保持 Terminal A 開啟。bridge 位於遠端 `127.0.0.1:8100`。

### 3.2 Terminal B：健康狀態

先執行「Isaac Lab 共用環境」，再執行：

```bash
cd ~/IsaacLab/ur3e_vla

curl -fsS http://127.0.0.1:8100/health | python3 -m json.tool
python3 scripts/tools/test_bridge_client.py \
  --server http://127.0.0.1:8100
```

預期：

- `/health` 的 `status` 為 `ok`。
- 等待初始化後 `state` 為 `waiting`。
- `/status` 包含 `object_poses`、`yolo_visibility`、`gripper` 與
  `frames_sent`。
- `yolo_visibility` 中三個物件均為可見。

### 3.3 本地端持續直播

在本地端另開一個 terminal，建立 SSH tunnel：

```bash
REMOTE_USER=your_user
REMOTE_HOST=your_host
SSH_PORT=your_ssh_port
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -L 127.0.0.1:18100:127.0.0.1:8100 \
  -p "$SSH_PORT" "$REMOTE_USER@$REMOTE_HOST"
```

第一次使用時，把測試 client 複製到本地：

```bash
scp -P "$SSH_PORT" \
  "$REMOTE_USER@$REMOTE_HOST:IsaacLab/ur3e_vla/scripts/tools/test_bridge_client.py" \
  ./test_bridge_client.py
```

本地 Python 需要：

```bash
python3 -m pip install requests websockets opencv-python numpy
```

開啟直播：

```bash
python3 test_bridge_client.py \
  --server http://127.0.0.1:18100 \
  --watch
```

直播視窗應持續更新。按 `q` 或 `Esc` 只會關閉 viewer，不會停止遠端
Isaac。

Python viewer 通過後，正式交接仍需以對方的 Unity 專案連線同一個
`ws://127.0.0.1:18100/unity` 與 HTTP API，確認畫面、state event 與
complete event。Python client 只驗證協定，不取代 Unity 驗收。

### 3.4 pause、resume、reset

Terminal B：

```bash
cd ~/IsaacLab/ur3e_vla
SERVER=http://127.0.0.1:8100

python3 scripts/tools/test_bridge_client.py \
  --server "$SERVER" --obj 0 --dest 0

while [ "$(curl -fsS "$SERVER/status" | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["state"])')" != "running" ]; do
  sleep 0.2
done

python3 scripts/tools/test_bridge_client.py \
  --server "$SERVER" --control pause

python3 scripts/tools/test_bridge_client.py --server "$SERVER"

sleep 2
python3 scripts/tools/test_bridge_client.py \
  --server "$SERVER" --control resume
```

確認直播不中斷，並在 viewer 看到 `running → paused → running → done`。
任務結束後：

```bash
python3 scripts/tools/test_bridge_client.py \
  --server "$SERVER" --control reset --seed 42
```

### 3.5 固定 seed 三條正式配對

編號：

```text
obj 0 = spoon
obj 1 = red_mug
obj 2 = bowl
dest 0 = 餐具區
dest 1 = 杯子區
dest 2 = 疊碗區
```

正式保證的配對只有 `0→0`、`1→1`、`2→2`。API 仍接受其他組合，但其他
六種組合不列入成功率驗收。Terminal B 貼上以下完整區塊：

```bash
cd ~/IsaacLab/ur3e_vla
SERVER=http://127.0.0.1:8100

wait_for_state() {
  expected="$1"
  while true; do
    current="$(
      curl -fsS "$SERVER/status" |
      python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])'
    )"
    echo "state=$current, waiting_for=$expected"
    [ "$current" = "$expected" ] && break
    sleep 1
  done
}

for task in 0 1 2; do
  echo "===== obj=$task dest=$task ====="
  wait_for_state waiting
  python3 scripts/tools/test_bridge_client.py \
    --server "$SERVER" --obj "$task" --dest "$task"
  wait_for_state done
  status="$(curl -fsS "$SERVER/status")"
  echo "$status" | python3 -m json.tool
  echo "$status" | python3 -c \
    'import json,sys; assert json.load(sys.stdin)["result"]["success"] is True'
  python3 scripts/tools/test_bridge_client.py \
    --server "$SERVER" --control reset --seed 42
  wait_for_state waiting
done
```

確認三次結果的 `result.success=true`。直播在全部任務期間都必須保持連線。

完成後回到 Terminal A 按 `Ctrl+C`，並確認：

```bash
ss -ltnp | grep ':8100\b' || true
```

應無輸出。

## 4. 單環境 H5 收集

### 4.1 三物件 no-save smoke test

```bash
cd ~/IsaacLab/ur3e_vla

for target in spoon red_mug bowl; do
  isaaclab scripts/collect_demos.py \
    --headless --enable_cameras \
    --target "$target" \
    --episodes 1 --max-episodes-tried 1 \
    --output-dir ~/IsaacLab/ur3e_vla/outputs/test/full_test/single_smoke \
    --no-save-h5
done
```

三次都應顯示 `episode 1/1: success`。

### 4.2 實際寫入小型 H5

```bash
isaaclab scripts/collect_demos.py \
  --headless --enable_cameras \
  --target red_mug \
  --episodes 3 --max-episodes-tried 5 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/test/full_test/single_h5 \
  --overwrite
```

H5 路徑：

```bash
export TEST_H5=~/IsaacLab/ur3e_vla/outputs/test/full_test/single_h5/red_mug/demos.h5
ls -lh "$TEST_H5"
```

驗證 schema、影像尺寸、action 與 5 Hz metadata：

```bash
python3 - "$TEST_H5" <<'PY'
import sys
import h5py

path = sys.argv[1]
with h5py.File(path, "r") as f:
    print("file attrs:", dict(f.attrs))
    assert f.attrs["scene_profile"] == "canonical_scene_v1"
    assert f.attrs["num_demos"] == 3
    assert sorted(f["data"].keys()) == ["demo_0", "demo_1", "demo_2"]

    for name, demo in f["data"].items():
        print(name, {key: value.shape for key, value in demo.items()
                     if hasattr(value, "shape")})
        assert demo["image"].shape[1:] == (480, 640, 3)
        assert demo["other/hand_image"].shape[1:] == (480, 640, 3)
        assert demo["robot_state"].shape[1] == 15
        assert demo["action"].shape[1] == 8
        assert demo.attrs["record_hz"] == 5.0
        assert demo.attrs["gripper_action_encoding"] == (
            "logical_binary_0_open_1_closed"
        )
print("PASS: canonical H5 schema")
PY
```

## 5. Multi-env 收集

### 5.1 no-save smoke test

```bash
cd ~/IsaacLab/ur3e_vla

isaaclab scripts/collect_demos_multi_env.py \
  --headless --enable_cameras \
  --target red_mug \
  --episodes 2 --num-envs 2 --max-episodes-tried 2 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/test/full_test/multi_smoke \
  --no-save-h5
```

預期 `2/2 successes`。

### 5.2 實際寫入 multi-env H5

```bash
isaaclab scripts/collect_demos_multi_env.py \
  --headless --enable_cameras \
  --target red_mug \
  --episodes 2 --num-envs 2 --max-episodes-tried 2 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/test/full_test/multi_h5 \
  --overwrite
```

```bash
export MULTI_H5=~/IsaacLab/ur3e_vla/outputs/test/full_test/multi_h5/red_mug/demos.h5
ls -lh "$MULTI_H5"
```

驗證 multi-env H5：

```bash
python3 - "$MULTI_H5" <<'PY'
import sys
import h5py

with h5py.File(sys.argv[1], "r") as f:
    assert f.attrs["scene_profile"] == "canonical_scene_v1"
    assert f.attrs["num_demos"] == 2
    assert sorted(f["data"].keys()) == ["demo_0", "demo_1"]
    for demo in f["data"].values():
        assert demo["image"].shape[1:] == (480, 640, 3)
        assert demo["other/hand_image"].shape[1:] == (480, 640, 3)
        assert demo["robot_state"].shape[1] == 15
        assert demo["action"].shape[1] == 8
        assert demo.attrs["record_hz"] == 5.0
        assert demo.attrs["gripper_action_encoding"] == (
            "logical_binary_0_open_1_closed"
        )
print("PASS: multi-env canonical H5 schema")
PY
```

不加 `--overwrite` 重跑收集時，預期出現 `FileExistsError`，既有 H5 必須保持
不變。所有收集入口只有收滿指定的成功 episode 數量才會以退出碼 0 結束；
相機讀取或 H5 寫入發生錯誤時會停止，不會用黑圖代替訓練影像。
共用 `close_app()` 會將例外狀態傳給 Isaac Sim 6.0.1 的
[`close(exit_code=...)`](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/py/source/extensions/isaacsim.simulation_app/docs/index.html)，
避免 fast shutdown 把失敗回報為退出碼 0。

可刻意設定最多兩次嘗試、要求三筆資料，檢查不足數量時會回傳非零退出碼：

```bash
if isaaclab scripts/collect_demos_multi_env.py \
  --headless --enable_cameras --target red_mug \
  --episodes 3 --num-envs 2 --max-episodes-tried 2 --no-save-h5; then
  echo "FAIL: incomplete collection returned success" >&2
else
  echo "確認 log 顯示 collected 2/3（或更少），且程式已結束"
fi
```

此指令是預期失敗測試，不會產生新的 H5。

### 5.3 Multi-env 預覽影片

```bash
isaaclab scripts/collect_demos_multi_env.py \
  --headless --enable_cameras \
  --target red_mug \
  --episodes 1 --num-envs 2 --max-episodes-tried 2 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/test/full_test/multi_video \
  --no-save-h5 \
  --record-video \
  --video-env -1 \
  --video-camera camera_policy \
  --video-path ~/IsaacLab/ur3e_vla/outputs/test/full_test/multi_video.mp4
```

```bash
ls -lh ~/IsaacLab/ur3e_vla/outputs/test/full_test/multi_video.mp4
```

### 5.4 MultiView H5 與五個相機

MultiView 是獨立實驗功能，但繼承同一個 base scene。只執行與驗證，不修改
`vla_sim/multiview/`：

```bash
isaaclab scripts/collect_demos_multi_view.py \
  --headless --enable_cameras \
  --target red_mug \
  --episodes 1 --max-episodes-tried 2 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/test/full_test/multiview_h5 \
  --overwrite
```

```bash
export MULTIVIEW_H5=~/IsaacLab/ur3e_vla/outputs/test/full_test/multiview_h5/red_mug/demos.h5

python - "$MULTIVIEW_H5" <<'PY'
import sys
import h5py

with h5py.File(sys.argv[1], "r") as f:
    assert f.attrs["scene_profile"] == "multiview_scene_v1"
    assert f.attrs["num_demos"] == 1
    demo = f["data/demo_0"]
    paths = (
        "image",
        "other/hand_image",
        "other/top_image",
        "other/left_image",
        "other/right_image",
    )
    lengths = []
    for path in paths:
        dataset = demo[path]
        print(path, dataset.shape)
        assert dataset.shape[1:] == (480, 640, 3)
        lengths.append(dataset.shape[0])
    assert len(set(lengths)) == 1
    assert demo["action"].shape == (lengths[0], 8)
    assert demo["robot_state"].shape == (lengths[0], 15)
    assert demo["gripper_tip_position"].shape == (lengths[0], 3)
    for name in ("spoon", "red_mug", "bowl"):
        assert demo[f"object_poses/{name}"].shape == (lengths[0], 7)
    assert demo.attrs["record_hz"] == 5.0
print("PASS: MultiView H5 schema and five synchronized streams")
PY
```

需要人工確認五個視角時，使用 GUI 並只錄一個 episode：

```bash
isaaclab scripts/collect_demos_multi_view.py \
  --enable_cameras --show-gui \
  --target red_mug \
  --episodes 1 --max-episodes-tried 2 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/test/full_test/multiview_video \
  --no-save-h5 --save-video
```

確認 `main`、`wrist`、`top`、`left`、`right` 五支影片存在、構圖合理且沒有
物件被裁切。

## 6. 維護工具

### 6.1 重建 assembled robot asset 到測試目錄

不要在 smoke test 中覆蓋正式 asset，改寫入 `outputs/test/`：

```bash
cd ~/IsaacLab/ur3e_vla

isaaclab scripts/tools/export_robot_asset.py \
  --headless \
  --output ~/IsaacLab/ur3e_vla/outputs/test/full_test/assets/assembled_robot.usda \
  --overwrite
```

```bash
ls -lh ~/IsaacLab/ur3e_vla/outputs/test/full_test/assets/assembled_robot.usda
```

用剛輸出的 asset 啟動 multi-env：

```bash
isaaclab scripts/collect_demos_multi_env.py \
  --headless --enable_cameras \
  --asset ~/IsaacLab/ur3e_vla/outputs/test/full_test/assets/assembled_robot.usda \
  --target red_mug \
  --episodes 1 --num-envs 1 --max-episodes-tried 1 \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/test/full_test/exported_asset_smoke \
  --no-save-h5
```

### 6.2 碰撞資產產生器

```bash
isaaclab scripts/tools/make_usd_collision_asset.py \
  --headless \
  --source /Props/YCB/Axis_Aligned/025_mug.usd \
  --output ~/IsaacLab/ur3e_vla/outputs/test/full_test/assets/mug_collision.usda \
  --approximation convexDecomposition \
  --mass 0.20
```

```bash
ls -lh ~/IsaacLab/ur3e_vla/outputs/test/full_test/assets/mug_collision.usda
```

### 6.3 匯出 3D 列印 STL

這會匯出場景中實際顯示的 spoon、mug、bowl、餐具盒與杯墊，單位為 mm：

```bash
isaaclab scripts/tools/export_printable_assets.py \
  --headless \
  --output-dir ~/IsaacLab/ur3e_vla/outputs/test/full_test/printable \
  --overwrite
```

確認 manifest、尺寸與 watertight 檢查：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path.home() / "IsaacLab/ur3e_vla/outputs/test/full_test/printable"
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["units"] == "millimetres"
assert set(manifest["parts"]) == {
    "spoon", "mug", "bowl", "cutlery_tray", "coaster"
}
for name, part in manifest["parts"].items():
    path = root / part["file"]
    assert path.is_file() and path.stat().st_size > 0
    assert part["watertight"] is True
    assert all(value > 0 for value in part["size_mm"])
    print(name, part["size_mm"], path)
print("PASS: printable assets")
PY
```

STL 通過幾何檢查不代表已設定好印表機參數；正式列印前仍需在 slicer 確認
擺放方向、支撐、縮放、壁厚與公差。

## 7. H5 工具

先設定第四節產生的 H5：

```bash
cd ~/IsaacLab/ur3e_vla
export TEST_H5=~/IsaacLab/ur3e_vla/outputs/test/full_test/single_h5/red_mug/demos.h5
```

### 7.1 建立 subset

```bash
python3 scripts/tools/subset_h5_demos.py \
  --input "$TEST_H5" \
  --output outputs/test/full_test/subset/demos.h5 \
  --episodes 1 \
  --seed 42 \
  --overwrite
```

### 7.2 匯出圖片與影片

```bash
python3 scripts/tools/export_h5_frames.py \
  --h5 "$TEST_H5" \
  --demos demo_0 \
  --every-n 5 \
  --video \
  --fps 5 \
  --out-dir outputs/test/full_test/h5_export
```

```bash
find outputs/test/full_test/h5_export -maxdepth 3 -type f | sort
```

## 8. TFDS/RLDS build

使用另一個 terminal：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlds_env
cd ~/IsaacLab/ur3e_vla/rlds_builder/ur3e_vla_dataset

export UR3E_VLA_H5_PATH=~/IsaacLab/ur3e_vla/outputs/test/full_test/single_h5/red_mug/demos.h5
export UR3E_VLA_VAL_RATIO=0.0
export UR3E_VLA_SPLIT_SEED=42

tfds build --overwrite \
  --data_dir ~/IsaacLab/ur3e_vla/outputs/test/full_test/tfds
```

完成後：

```bash
find ~/IsaacLab/ur3e_vla/outputs/test/full_test/tfds -maxdepth 4 -type f | sort
```

注意：builder 第一次執行可能需要下載 Universal Sentence Encoder。

## 9. OpenVLA server 與模擬 rollout

這一節需要已訓練、包含 `ur3e_vla_dataset` normalization statistics 的模型。
若尚未建立模型，先略過第 9 節；不要把模型下載快取當作已微調的模型。

### 9.0 乾淨 OpenVLA checkout 的 UR3e 整合

訓練前先在乾淨的 OpenVLA checkout 套用專案提供的固定 patch。完整
H5 → TFDS → OpenVLA 訓練指令見 [command_reference.md](command_reference.md)。
以下只檢查外部 checkout 的資料整合；不要複製另一個帳號的未提交修改：

```bash
PROJECT_ROOT=/path/to/ur3e_vla
OPENVLA_SRC=/path/to/openvla
cd "$OPENVLA_SRC"
test -z "$(git status --porcelain)" || { echo 'OpenVLA checkout is not clean'; false; }
git switch --detach c8f03f48af692657d3060c19588038c7220e9af9
git apply --check "$PROJECT_ROOT/scripts/tools/openvla_ur3e.patch"
git apply "$PROJECT_ROOT/scripts/tools/openvla_ur3e.patch"
git diff --check
```

這個 patch 註冊 `ur3e_vla_dataset`、把 UR3e state/action 轉成 OpenVLA
介面，並提供明確的最終模型輸出目錄。若 checkout 已有未提交修改，先
停止並另建乾淨 checkout；不要用 `git reset` 覆蓋他人的工作。

### 9.1 Terminal A：啟動 VLA server

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openvla
cd ~/IsaacLab/ur3e_vla

export MODEL_PATH=/srv/isaac_ros2/models/pick_place_v1
test -f "$MODEL_PATH/dataset_statistics.json" || { echo "尚無可測試的微調模型：$MODEL_PATH"; false; }

python3 scripts/vla_server.py \
  --model-path "$MODEL_PATH" \
  --unnorm-key ur3e_vla_dataset \
  --host 127.0.0.1 \
  --port 8000
```

等待出現：

```text
[VLA Server] Ready.
```

### 9.2 Terminal B：server API

```bash
curl -fsS http://127.0.0.1:8000/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8000/config | python3 -m json.tool
```

預期 `status=ok`，且 `available_unnorm_keys` 包含
`ur3e_vla_dataset`。

### 9.3 使用 H5 比對 prediction

```bash
cd ~/IsaacLab/ur3e_vla

python3 scripts/tools/check_vla_prediction.py \
  --h5 outputs/test/full_test/single_h5/red_mug/demos.h5 \
  --server http://127.0.0.1:8000 \
  --demo demo_0 \
  --steps 0 5 10 \
  --unnorm-key ur3e_vla_dataset
```

確認每一個 step 都回傳 HTTP 200、7 維 action 與 latency。

### 9.4 Terminal C：模擬 VLA rollout

先執行「Isaac Lab 共用環境」，再執行：

```bash
isaaclab scripts/run_vla.py \
  --headless --enable_cameras \
  --target red_mug \
  --instruction "pick up the red mug" \
  --vla-server http://127.0.0.1:8000 \
  --camera camera_policy \
  --unnorm-key ur3e_vla_dataset \
  --action-scale 0.5 \
  --vla-step-interval 12 \
  --max-steps 600
```

確認 VLA server 持續收到 `/predict`，Isaac runner 沒有 timeout 或 action
shape 錯誤。測完先停止 Isaac，再停止 VLA server。

## 10. 實體關節控制虛擬手臂

### 10.1 不接實體的 fake JointState 測試

Terminal A 先執行「Isaac Lab 共用環境」：

```bash
isaaclab scripts/sync_sim_real.py \
  --headless --enable_cameras \
  --joint-states-topic /joint_states \
  --joint-state-timeout 0.5
```

Terminal B 先載入 ROS：

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash
```

故意使用亂序 joint names，持續發布：

```bash
ros2 topic pub -r 10 /joint_states sensor_msgs/msg/JointState \
  "{name: [wrist_3_joint, shoulder_pan_joint, elbow_joint, wrist_1_joint, shoulder_lift_joint, wrist_2_joint], position: [0.0, 0.57, 1.57, -1.57, -1.57, -1.57]}"
```

Terminal C：

```bash
curl -fsS http://127.0.0.1:8100/status | python3 -m json.tool
```

預期：

- `state` 變成 `running`。
- `joint_sync.state` 為 `live`。
- YOLO 直播持續更新。

停止 Terminal B 的 publisher 後等待超過 0.5 秒：

```bash
sleep 1
curl -fsS http://127.0.0.1:8100/status | python3 -m json.tool
```

預期 `state=hold`、`joint_sync.detail=stale_joint_state`，虛擬手臂保持最後
姿勢。

測試控制 API：

```bash
python3 ~/IsaacLab/ur3e_vla/scripts/tools/test_bridge_client.py \
  --server http://127.0.0.1:8100 --control pause

python3 ~/IsaacLab/ur3e_vla/scripts/tools/test_bridge_client.py \
  --server http://127.0.0.1:8100 --control resume

python3 ~/IsaacLab/ur3e_vla/scripts/tools/test_bridge_client.py \
  --server http://127.0.0.1:8100 --control reset --seed 42
```

確認此模式沒有任何 Servo 或 trajectory publisher：

```bash
ros2 node list
ros2 topic info /servo_node/delta_twist_cmds --verbose
```

最後停止 publisher 與 Isaac。

### 10.2 接實體 UR3e（需要實體）

此段只能在現場執行。先確認實體手臂工作區清空、急停可用，再啟動 driver。
本模式只訂閱關節，不發布 Servo 或 trajectory 指令。

確認 `/joint_states` 存在：

```bash
ros2 topic echo /joint_states --once
```

再執行：

```bash
isaaclab scripts/sync_sim_real.py \
  --headless --enable_cameras \
  --joint-states-topic /joint_states \
  --joint-state-timeout 0.5
```

手動慢速調整實體手臂，確認虛擬六軸跟隨；此 runner 不會發送任何實體
運動命令。

### 10.3 虛擬軌跡驅動實機（需要實體）

目前只支援 `obj=1, dest=2` 與 reset。完整且唯一的操作方式請見
[`docs/sim_real_sync.md`](sim_real_sync.md)。其他物品與位置尚未開放。

不要把 10.1 的 fake JointState 測試通過解讀成此模式已通過。此段包含實體
運動，必須另行完成該文件中的 dry-run、安全檢查與現場驗收。

## 11. Real VLA Servo dry-run（需要相機與 ROS，不啟用運動）

這一項需要：

- VLA server 已啟動。
- ROS camera topic 正常。
- 不加入 `--enable-motion`。

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash

python3 scripts/real_vla_servo.py \
  --image-topic /image_raw \
  --instruction "pick up the red mug" \
  --vla-server http://127.0.0.1:8000 \
  --unnorm-key ur3e_vla_dataset \
  --query-rate-hz 1 \
  --duration 5 \
  --diagnostics
```

預期：

- 顯示 `DRY RUN ONLY`。
- 能收到影像並印出 VLA action。
- `published_twist=0`。
- 沒有實體手臂或 gripper 動作。

本文件不包含 `--enable-motion` 測試。實體運動仍需依
`docs/real_robot_vla_runbook.md` 的安全檢查執行。

## 12. 最後清理與確認

確認沒有程序或 port 留在背景：

```bash
pgrep -af '[i]saac-sim|[r]un_scene|[r]un_remote_pick_place|[r]un_vla|[c]ollect_demos|[s]ync_sim_real|[v]la_server' || true
ss -ltnp | grep -E ':(8100|8000)\b' || true
```

確認 Git 沒有因測試被修改：

```bash
cd ~/IsaacLab/ur3e_vla
git status --short
```

測試輸出都位於：

```text
outputs/test/full_test/
```

全部確認後，如不再需要這些測試產物才刪除：

```bash
rm -rf ~/IsaacLab/ur3e_vla/outputs/test/full_test
```

## 測試結果紀錄

```text
[ ] Git diff check / unit tests / Python compile
[ ] 純場景 headless
[ ] 純場景 GUI / YOLO 畫面
[ ] Bridge health / status
[ ] 本地持續直播
[ ] pause / resume / reset
[ ] 固定 seed 0→0、1→1、2→2
[ ] 單環境三物件 smoke test
[ ] 單環境 H5 schema
[ ] Multi-env smoke / H5 / video
[ ] H5 不覆寫既有資料 / 未收滿回傳非零退出碼
[ ] MultiView H5 / 五視角影片
[ ] Assembled robot exporter
[ ] Collision asset generator
[ ] 3D 列印 STL / manifest / watertight
[ ] H5 subset / image / video tools
[ ] TFDS build
[ ] VLA server health / prediction
[ ] VLA simulation rollout（requests > 0、errors = 0）
[ ] Fake JointState live / stale HOLD
[ ] 實體 UR3e → Isaac read-only sync（需要實體）
[ ] obj=1,dest=2 sim-to-real（需要實體）
[ ] Real VLA Servo dry-run
[ ] 無背景程序、Git 工作目錄乾淨
```

`VLA simulation rollout` 的通過條件是 server、影像、action 與 Isaac 控制通路
正常；它不代表模型已完成抓取。模型成功率必須另外執行多次完整 rollout 並
記錄成功／失敗。

## 仍需外部或人工驗收的項目

以下項目無法只靠本文件的自動命令宣告完成：

- Unity 正式程式與遠端 bridge 的長時間連線、控制和事件回傳。
- 實際 YOLO 模型的辨識率；`yolo_visibility` 只檢查物件是否完整入鏡，不是
  YOLO inference 結果。
- OpenVLA 完整任務成功率；短版 smoke test 只驗證推論通路。
- STL 的 slicer 設定、實際列印尺寸、公差、顏色與真實場景擺放。
- Real-to-sim 與 sim-to-real 的現場安全、延遲、MoveIt singularity 和路徑。
