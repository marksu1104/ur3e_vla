# Remote 實驗前確認（驗證日期：2026-10-05）

分支為 `refactor/unified-runtime`，修正起點是 `450661f`；本頁記錄當時
包含工作樹修正的 Remote 驗證證據，現行提交以 `git log -1 --oneline` 為準。
結論限定於 Remote Isaac／Unity 模擬橋接，不包含實體機器人或 real-to-sim 控制。

## 指令與場景契約

原本的 Remote 啟動方式維持不變：

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash
isaaclab scripts/run_remote_pick_place.py \
  --headless --enable_cameras --seed 42
```

SSH tunnel 與測試 client 的使用方式也維持不變：

```bash
ssh -tt -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18100:127.0.0.1:8100 \
  -p "$SSH_PORT" "$REMOTE_USER@$REMOTE_HOST"

python3 scripts/tools/test_bridge_client.py \
  --server http://127.0.0.1:18100 --watch
```

以 AST 比對 `450661f`，runner 的八個額外參數名稱、型別與預設值完全相同：
`--bridge-port`、`--stream-width`、`--stream-height`、`--stream-quality`、
`--stream-every-n-steps`、`--max-trial-seconds`、`--show-markers`、`--seed`。

以下來源與 `450661f` 無差異：canonical 場景與相機、物件與目的地設定、
simulation runtime、pick-and-place 軌跡與成功判準、action、YOLO visibility，
以及 `test_bridge_client.py`。沒有更改 0／1／2 的物件、目的地編號或 HTTP／
WebSocket 路徑。

## Remote 的三項修正

- `run_remote_pick_place.py` 在 local 與公開 state 都完成轉移後才解除 command
  reservation，避免 trajectory 建立期間第二個 task 收到 202，之後卻被略過。
- `/control` 對 list／dict 等非字串 action 回傳 HTTP 400，不再因 unhashable
  payload 變成 500。
- BridgeServer 新增依模式限制 control 的能力；Remote 預設仍完整接受
  pause／resume／reset，只有 sim-to-real 明確拒絕未實作的 pause／resume。

這三項沒有改啟動參數、任務配對、軌跡或成功判準。針對實際 command AST、
main AST 與 ASGI handler 的 6 個交錯請求測試，在 `450661f` 全數重現失敗，
目前版本 6 個全數通過；完整 CPU suite 為 134 個測試、0 skips。

## 30 分鐘實際 headless 長測

從第一個 `waiting` 開始連續執行 1800.005 秒，結果 PASS：

| 檢查 | 結果 |
| --- | --- |
| 正式任務 | `0→0`、`1→1`、`2→2` 各兩次，共 6 次，內建成功判準全部成功，complete event 各一次 |
| pause／resume／reset | 每一輪都驗證；pause 時 progress 保持且影像繼續產生；另取消 trial 6，沒有錯誤 complete event |
| 串流 | 43,438 幀 JPEG 全部可解碼且非單色；同一 observer segment 最大 gap 0.3799 秒 |
| 慢 client | 47 秒不讀 frame 時，健康 viewer 仍收到 1,144 幀；慢 client 在 2.001 秒內關閉，重連成功 |
| 無 viewer／重連 | 無 viewer 時 server 持續產生 frame；running 重連成功；無 viewer 完成後重連只 replay 一次 complete |
| 來源穩定 | runner、bridge、scene、runtime 等 15 個來源 SHA 在測試前後相同，且仍與目前工作樹一致 |
| 自有資源 | 60 個樣本；RSS 8171.05→8184.17 MiB，範圍 13.72 MiB；owned GPU 固定為 5559 MiB |

RSS 在這個單次 30 分鐘樣本的 5 分鐘後線性斜率約 0.7164 MiB／min；這是
描述性數值，不能據此宣稱永遠沒有 memory leak。測試完成後自有 process 正常
停止，8100 已釋放；其他人的 Ollama process 沒有停止或納入 owned 資源數字。

完整證據：

- [Remote readiness 彙整](../outputs/test/remote/remote_readiness_20261005/results.json)
- [30 分鐘長測](../outputs/test/remote/remote_soak_20261004_v2/results.json)
- [競態修正前後](../outputs/test/remote/bridge_audit_20261004/regression_results.json)
- [134 個測試與穩定版本前檢](../outputs/test/preflight/cleanup_20261005/report.json)

## 驗收邊界

- 這次使用 Python client 驗證相同 HTTP／WebSocket protocol，正式 Unity 畫面與
  Unity 端事件處理仍由人工驗收。
- `seed` 目前只保存及回報 metadata；reset 恢復固定 canonical spawn，不能將
  不同 seed 稱為不同隨機場景覆蓋。
- frame queue 上限為一幀、採 latest-wins；每個 client 的 event queue 仍無上限。
  本次低頻事件與 47 秒慢 reader 正常，不能推論任意事件洪水下記憶體有界。
- 同步 runner 的 reservation 順序與 sim-to-real 未實作 control 已用實際 command
  block 的純 CPU 測試修正；沒有啟動 ROS 或實體機器人，現場同步仍待人工驗收。

在目前來源與環境下，可接著做 Remote Unity 人工驗收及固定 canonical 場景
實驗。正式實驗前仍應執行 preflight、確認 8100 未被舊程序占用，再啟動 runner。
請使用本分支最新提交，不要切回修正起點 `450661f` 取代最新版本；若有
額外現場修改，先保存再整理 working tree。這次長測只涵蓋正式配對
`0→0`、`1→1`、`2→2`；其餘六種組合沒有包含在 readiness 結論中。
