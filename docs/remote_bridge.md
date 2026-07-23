# Unity–Isaac Remote Bridge

## 目前完成的功能

遠端 RTX 工作站會執行 Isaac Sim，並在 `127.0.0.1:8100` 提供：

- `CameraYolo` 的持續 JPEG 直播。
- red mug、spoon、bowl 的 pick-and-place 任務。
- 三個目標位置。
- pause、resume、reset 控制。
- 任務狀態與成功／失敗結果。

本地端透過 SSH tunnel 使用本機 `127.0.0.1:18100`。直播使用一條持續開啟的
WebSocket；任務與控制使用 HTTP。HTTP request 不會中斷直播。

~~~text
Unity
  WebSocket  ws://127.0.0.1:18100/unity
  HTTP       http://127.0.0.1:18100
                  |
                  | SSH tunnel
                  v
Remote Isaac 127.0.0.1:8100
~~~

## 通用啟動方式

以下方式適用於任何有 OpenSSH client 的終端，不限定作業系統或 shell。

先在本地終端建立 SSH tunnel 並登入遠端：

~~~console
ssh -tt -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L 127.0.0.1:18100:127.0.0.1:8100 -p 6002 acolab@140.112.42.35
~~~

登入遠端後，在同一個 SSH session 執行：

~~~bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab_ros2
cd ~/IsaacLab

./isaaclab.sh -p ./ur3e_vla/scripts/run_remote_pick_place.py \
  --headless \
  --enable_cameras
~~~

保持這個 SSH session 開啟。在另一個本地終端確認 bridge：

~~~console
curl http://127.0.0.1:18100/health
~~~

如果本地沒有 `curl`，也可以用瀏覽器開啟相同 URL。看到 `waiting` 後即可
啟動 Unity：

~~~json
{"status":"ok","state":"waiting","protocol":1}
~~~

## API 總覽

所有 HTTP JSON 都使用 UTF-8 與 `Content-Type: application/json`。

| Method | Path | 用途 | 成功狀態碼 |
|---|---|---|---:|
| GET | `/health` | 確認 bridge 與 protocol | 200 |
| GET | `/status` | 取得完整狀態 snapshot | 200 |
| POST | `/pickplace` | 送出物件與目標位置 | 202 |
| POST | `/control` | pause、resume 或 reset | 200 |
| WebSocket | `/unity` | 持續接收 JPEG 與 JSON event | 101 upgrade |

物件編號：

| obj | 物件 |
|---:|---|
| 0 | spoon |
| 1 | red_mug |
| 2 | bowl |

物件編號依 `CameraYolo` 畫面由左到右排列。`dest` 也使用 `0`、`1`、`2`，
依序對應場景中由左到右的三個固定目標座標。彩色目標點在等待、執行、
暫停、完成與 reset 的所有階段都保持隱藏。

需要從直播確認位置時，在啟動 Isaac 的命令加上 `--show-markers`。此參數只
改變標示的可見性，不會改變 `dest` 座標；正式執行時拿掉即可恢復全程隱藏。

## GET `/health`

Request 沒有 body。

Response：

~~~json
{
  "status": "ok",
  "state": "waiting",
  "protocol": 1
}
~~~

- `status`：bridge 正常時為 `ok`。
- `state`：目前 Isaac 任務狀態。
- `protocol`：目前協定版本為整數 `1`。

## GET `/status`

Request 沒有 body。這個 API 用於查看目前完整狀態，例如：

~~~json
{
  "state": "running",
  "trial_id": 7,
  "object": "red_mug",
  "task_index": 1,
  "position_index": 1,
  "progress": {
    "t": 3.2,
    "total": 11.8,
    "gripper_command": 0.0,
    "finger_joint": 0.0
  },
  "result": null,
  "object_poses": null,
  "seed": 42,
  "frames_sent": 1234,
  "gripper": {
    "command": 0.0,
    "finger_joint": 0.0,
    "soft_limits": [0.0, 0.8]
  },
  "yolo_visibility": null,
  "goal_visibility": null
}
~~~

主要欄位格式：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `state` | string | `starting`、`resetting`、`waiting`、`running`、`paused` 或 `done` |
| `trial_id` | integer 或 null | 本次任務編號；沒有任務時為 null |
| `object` | string 或 null | 目前物件名稱 |
| `task_index` | integer 或 null | `obj` 編號 |
| `position_index` | integer 或 null | `dest` 編號 |
| `progress` | object 或 null | 任務時間與夾爪數值 |
| `result` | object 或 null | 任務結束後的結果 |
| `object_poses` | object 或 null | waiting 時各物件的 `[x,y,z]` |
| `seed` | integer 或 null | reset 使用的 seed |
| `frames_sent` | integer | server 已編碼的影格數 |
| `gripper` | object（waiting 後提供） | 夾爪命令、關節值與 soft limits |
| `yolo_visibility` | object 或 null | 物件入鏡診斷 |
| `goal_visibility` | object 或 null | 使用 `--show-markers` 時的目標入鏡診斷 |

`result` 的格式：

~~~json
{
  "success": true,
  "reason": "completed",
  "detail": {
    "obj_lifted": true,
    "obj_at_place": true,
    "ee_safe": true,
    "obj_pos_final": [0.34, -0.02, 1.10],
    "ee_pos_final": [0.30, 0.13, 1.35],
    "best_lift_height": 0.08,
    "obj_place_xy_dist": 0.02,
    "place_pos": [0.34, -0.025, 0.0]
  }
}
~~~

`detail` 是診斷資料，可能包含更多欄位。

## POST `/pickplace`

URL：

~~~text
http://127.0.0.1:18100/pickplace
~~~

Request body：

~~~json
{"obj":0,"dest":1}
~~~

- `obj`：integer，範圍 `0..2`。
- `dest`：integer，範圍 `0..2`。

Server 接受後回傳 HTTP 202：

~~~json
{
  "accepted": true,
  "trial_id": 7,
  "obj": 0,
  "dest": 1
}
~~~

HTTP 202 只表示任務已接受。真正的成功或失敗會由 `/unity` WebSocket 的
`complete` event 回傳。

## POST `/control`

URL：

~~~text
http://127.0.0.1:18100/control
~~~

Request body 三選一：

~~~json
{"action":"pause"}
{"action":"resume"}
{"action":"reset"}
~~~

Reset 也可以指定整數 seed：

~~~json
{"action":"reset","seed":42}
~~~

成功接受後回傳 HTTP 200：

~~~json
{"accepted":true}
~~~

## HTTP 錯誤格式

- HTTP 400：`obj`、`dest`、`action` 或 `seed` 無效。
- HTTP 409：目前狀態不能執行該命令，或上一個 command 尚未套用。
- HTTP 422：request body 不是可解析的 JSON object。

Error response：

~~~json
{"detail":"error description"}
~~~

## WebSocket `/unity`

Unity 使用：

~~~text
ws://127.0.0.1:18100/unity
~~~

這是一條 server-to-client 的持續串流；Unity 不需要傳送 WebSocket message。
每個 WebSocket message 都是以下其中一種格式。

### Binary message：JPEG

- 一個 binary message 就是一張完整 JPEG。
- JPEG 開頭為 `FF D8`，結尾為 `FF D9`。
- 預設解析度為 `1280x720`。
- 預設 JPEG quality 為 `85`。
- 可直接交給 Unity `Texture2D.LoadImage(byte[])`。

### Text message：state

每個 text message 是一個完整的 UTF-8 JSON object。連線後 server 會先送出
目前狀態，之後在狀態改變時繼續發送：

~~~json
{
  "type": "state",
  "state": "running",
  "trial_id": 7
}
~~~

- `type`：固定為 `state`。
- `state`：目前狀態字串。
- `trial_id`：目前任務編號；沒有任務時為 `0`。

進入 `waiting` 時，同一個 state message 還會附上 `object_poses`、`seed`、
`yolo_visibility` 與 `goal_visibility`，格式與 GET `/status` 的同名欄位相同。

### Text message：complete

任務結束時發送：

~~~json
{
  "type": "complete",
  "trial_id": 7,
  "success": true,
  "detail": {
    "obj_lifted": true,
    "obj_at_place": true,
    "ee_safe": true
  }
}
~~~

- `type`：固定為 `complete`。
- `trial_id`：對應 `/pickplace` response 的任務編號。
- `success`：boolean，任務是否成功。
- `detail`：object，成功判定的診斷資料。

JPEG binary message、state JSON 與 complete JSON 會出現在同一條 WebSocket，
但各自保留完整的 WebSocket message boundary。

## 與現有 C# 的對應

`IsaacLabVideoSource.cs`：

- `serverUrl` 使用 `ws://127.0.0.1:18100/unity`。
- `OnMessage(byte[] bytes)` 收到 `FF D8` 開頭時，當作 JPEG。
- 其他資料以 UTF-8 轉成 JSON，依 `type` 處理 `state` 或 `complete`。
- 同一條 WebSocket 持續用於多個任務與 reset。

`IsaacLabClient.cs`：

- `endpointUrl` 使用 `http://127.0.0.1:18100/pickplace`。
- `SendPickPlace(int obj, int dest)` 對應 POST `/pickplace`。
- pause、resume、reset 對應 POST `/control`。
- `/pickplace` 的 `trial_id` 對應 WebSocket `complete.trial_id`。

`scripts/tools/test_bridge_client.py` 使用完全相同的 HTTP 與 WebSocket 格式，可作為目前
server 互動方式的參考；正式 Unity 直接由 C# 連線。

## 一次任務的資料流

~~~text
Unity -> HTTP POST /pickplace {obj, dest}
Isaac -> HTTP 202 {accepted, trial_id, obj, dest}
Isaac -> WS state {state:"running", trial_id}
Isaac -> WS binary JPEG ... 持續直播
Isaac -> WS complete {trial_id, success, detail}
Isaac -> WS state {state:"done", trial_id}

Unity -> HTTP POST /control {action:"reset"}
Isaac -> HTTP 200 {accepted:true}
Isaac -> WS state {state:"resetting", trial_id}
Isaac -> WS binary JPEG ... 持續直播
Isaac -> WS state {state:"waiting", trial_id:0}
~~~

收到 `waiting` 後可送下一個任務。任務進行中送 pause 會收到 `paused`，送
resume 會再次收到 `running`，期間 JPEG 直播仍維持。

所有工作結束時，在執行 Isaac 的遠端 shell 按一次 Ctrl+C；Isaac 關閉後再
輸入 `exit` 結束 SSH session 與 tunnel。
