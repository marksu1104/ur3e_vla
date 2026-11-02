# 歷史驗證紀錄

六份舊快照已合併；這裡僅用來追查當時結果，現行操作請看 [完整指令](../command_reference.md)，Remote 範圍請看 [實驗前確認](../remote_readiness.md)；本機交接計畫 `docs/acceptance.md` 已被 Git 忽略。旧前檢與第一次 Remote 長測的 JSON 已合併到各組 history.json；一次性腳本已由正式工具、tests 和現行指令取代。

<a id="2026-10-03-manual-checklist"></a>

## 2026-10-03 展示與資料人工驗收

這份流程供人工驗收時直接操作。歷史自動測試結果見
[acceptance_450661f.md](validation_history.md#450661f)，最新修正與整合結果見
[acceptance_20261004.md](validation_history.md#2026-10-04-integration) 及
[MultiView 寫入補測](validation_history.md#2026-10-04-multiview-rollback)。Remote 指令與 30 分鐘長測
見 [Remote readiness](../remote_readiness.md)。最新三物件五視角本機頁在
[`outputs/test/data/acceptance_gallery_20261004/index.html`](../../outputs/test/data/acceptance_gallery_20261004/index.html)，
本機影片與填表頁在
[index.html](../../outputs/test/data/acceptance_20261002_followup_450661f/index.html)。
頁面可一起播放五個視角，下載人工驗收 Markdown 紀錄；本機產物不在 Git。

### 1. 開啟影片與紀錄頁

在工作站桌面的瀏覽器開啟上述 HTML 檔，先看單環境影片與五支 MultiView
影片。單環境 overlay 的 `gripper_raw` 是 logical 值：0 開、1 閉。
記錄每個視角是否清楚、動作與落點是否可接受；資料欄位的自動檢查通過
仍需這一步視覺確認。表單填寫不會自動保存，完成後下載紀錄。

### 2. 啟動 Remote 展示

在新的工作站 terminal 執行；GUI 需使用已有 DISPLAY 的桌面 terminal：

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash
git status --short --branch
git log -1 --oneline
isaaclab scripts/run_remote_pick_place.py --enable_cameras --seed 42
```

等到 `REMOTE READY`，確認場景、三個物件、Robotiq 和 YOLO 構圖，目標
marker 預設隱藏。若從 SSH 啟動，以 `--headless` 跑展示，再用本地 viewer
看畫面；GUI 檢查仍需另在桌面進行。

原始程式基準是 `450661f`，目前另有資料工具、TFDS、VLA 錯誤處理與訓練
patch 的未提交修正；驗收時記錄實際 HEAD 與未提交差異，詳見最新紀錄。

### 3. 連線、控制與三條正式任務

工作站端先確認：

```bash
curl -fsS http://127.0.0.1:8100/health
```

遠端連線請依 [remote_bridge.md](../remote_bridge.md) 建立 SSH tunnel；本地
viewer 與 Unity 都使用 tunnel 的 HTTP 與 `/unity` WebSocket 地址。
完整操作命令見 [full_test_commands.md](../full_test_commands.md) 第 3 節。

每條任務只在 `waiting` 時送出；正式配對為 `0→0`（spoon）、`1→1`
（red_mug）、`2→2`（bowl）。以 Python client 測試時，例如：

```bash
python3 scripts/tools/test_bridge_client.py --server http://127.0.0.1:8100 --obj 0 --dest 0
python3 scripts/tools/test_bridge_client.py --server http://127.0.0.1:8100 --control pause
python3 scripts/tools/test_bridge_client.py --server http://127.0.0.1:8100 --control resume
python3 scripts/tools/test_bridge_client.py --server http://127.0.0.1:8100
```

pause 需在 `running`、resume 需在 `paused` 時送出。等任務 `done`，確認
`result.success=true` 並看實際落點，再 reset：

```bash
python3 scripts/tools/test_bridge_client.py --server http://127.0.0.1:8100 --control reset --seed 42
```

送出 reset 只表示命令已接受；確認 `resetting → waiting` 後再送下一條
命令。直播需在整段操作中持續更新。正式 Unity 還需確認 JPEG 顯示、state
與 complete event、trial ID 對應和 HTTP 控制，不以 Python client 代驗。

### 4. 一行驗證 H5

另開新的 terminal、載入共用環境後，可以只讀檢查既有資料：

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash
python scripts/tools/validate_h5.py outputs/test/data/acceptance_20261002_450661f/single/red_mug/demos.h5 --expected-profile canonical_scene_v1 --expected-episodes 1
python scripts/tools/validate_h5.py outputs/test/data/acceptance_20261002_450661f/multiview/red_mug/demos.h5 --expected-profile multiview_scene_v1 --expected-episodes 1
```

工具回傳 0 表示儲存的 schema、長度、資料值與 metadata 檢查通過；失敗
回傳 1，並指出欄位或影格。5 Hz 檢查的是 metadata，沒有量測硬體時間。
若要重收資料，使用新的輸出目錄，沿用完整測試文件第 4–5 節命令。

### 5. 記錄與結束

在影片頁填操作者、日期和問題；只有親自確認通過的項目才勾選。下載紀錄
後放入 `outputs/test/`，保留 Unity 版本、截圖或 log 路徑、實際 HEAD 與
Git 差異。遇到問題先記重現步驟，不需為了驗收臨時改場景。

停止 viewer 後，在 Remote terminal 按 Ctrl+C；確認 8100 已釋放：

```bash
ss -ltnp | rg ':8100\b' || true
```

本流程完成後，可記錄「展示與資料人工驗收」結果。OpenVLA 新 checkout
訓練／推論、模型成功率、實體同步與部署仍有獨立驗收工作，本次不操作
實體機器人。

<a id="2026-10-04-integration"></a>

## 2026-10-04 補強與整合驗證

分支：`refactor/unified-runtime`。HEAD 仍是
`450661f6b6227e224ce55e54e0f2a9f560a019f1`；下列修正尚未提交。
本紀錄涵蓋當日凌晨收集及下午接續驗證，沒有把下午工作列為九點前完成。
場景與 MultiView 程式維持原樣；全程只使用本機模擬，未操作實體機器人。
以上是當日 16:12 前的測試範圍。之後的 MultiView 寫入回復修正及新一輪
驗證另見 [multiview_rollback_20261004.md](validation_history.md#2026-10-04-multiview-rollback)，
原有 98 個測試與檔案指紋保留為前一輪證據。

### 已補上的問題

- H5 core 寫入遇到 Python 例外時，移除本次新增的半集並還原 root metadata，
  保留舊集且允許重試；先拒絕 NULL instruction 與非有限 gripper 資料。
  這不涵蓋斷電、程序被強制殺掉或損壞的儲存裝置。
- H5 subset 拒絕來源與目的為同一檔案、symlink 或 hardlink；驗證來源與集數，
  完成暫存副本後才發佈。複製失敗保留既有輸出；非 overwrite 發佈也不覆寫
  複製期間新出現的目的檔案。
- H5 驗證工具只讀檔，會拒絕不完整 MultiView、錯誤欄位或 metadata、NaN、
  黑圖及失效的 dataset 連結。預覽改讀 logical gripper 欄位 13；錯誤抽樣
  間隔、FPS 或無法建立影片 writer 會回報失敗。
- TFDS 更新至 `1.1.0`：來源路徑正規化、去重與排序，episode key 區分不同
  收集來源；failed episodes 在分割前移除，拒絕混用 scene profile 或 gripper
  encoding。至少兩集且 val ratio 非零時，train 與 val 各保留至少一集。
  舊 `1.0.0` 保留；移動來源檔案會改變 key，需記錄輸入路徑與 split 設定。
- VLA client 僅接受有限的七維 action；失敗時回傳零位移並保留最後有效 grip，
  不重播先前的 delta。server 也拒絕錯誤模型輸出；Remote `/control` 的非字串
  action 回傳 400。實體控制程式沒有執行。
- VLA runner 將背景例外傳回主流程；bounded 短測需完成指定步數、無推論錯誤，
  且至少消費一個 action。只在結束 join 後收到結果不算通過；合法全零 action
  可以通過。Ctrl+C 保留中斷與清理行為。
- OpenVLA pinned patch 修正 optimizer 更新計數與 gradient accumulation，
  最後一次更新一定匯出，包含保留 checkpoint 模式；step 參數需為正數。

### 實際結果與證據

| 項目 | 結果 | 範圍 |
| --- | --- | --- |
| 單元與格式檢查 | PASS | 98 個測試通過；全 Python compile 與 Git diff check 通過。先前新增連結檢查測試的首跑錯誤已修正，初次 log 保留 |
| 三物件 × 三種收集模式 | PASS | single、Multi-env（2 env）各物件各 2 集，MultiView 各物件 1 集；九組共 15 集、761 步，全部 H5 驗證通過 |
| TFDS 1.1.0 建置與逐步讀回 | PASS | 三個 single 來源共 6 集、296 步，train 4／val 2；雙視角 RGB、state、action、instruction 全數與 H5 完全相同；RLDS flags、reward、discount、512D embedding 正確 |
| 真實 builder split | PASS | train／val 互斥且完整覆蓋來源；重排、重複來源與 symlink alias 不改 split／key；來源 H5 SHA-256 不變 |
| 真實 Isaac＋HTTP 替身 | PASS | 合法零 action 退出 0；錯誤六維 action 與 timeout 均退出非零。這三組只驗證錯誤處理 |
| OpenVLA patch | PASS | 獨立乾淨 pinned `c8f03f48af692657d3060c19588038c7220e9af9` 副本套用成功；實際迴圈 AST 的 CPU 測試驗證更新數、累積與最後一步儲存 |
| 真實 GPU 短訓練／匯出 | PASS | 本機快取 OpenVLA 7B、BF16 LoRA rank 4、batch 1，恰一個 optimizer update；save interval 250 仍匯出最終模型。426／439 個 LoRA B tensor 非零，全部有限，action statistics 為 7D |
| 匯出模型 HTTP 推論 | PASS | spoon、red_mug、bowl 三種指令各一次；回傳有限七維 action，單次記錄 latency 約 120–138 ms，非效能 benchmark |
| 真實模型 Isaac rollout | PASS | red_mug，240 步，20 requests、0 errors、20 consumed actions，退出 0；沒有量測抓取成功率 |

詳細產物集中於
[`outputs/test/model/pre9_20261004_450661f/`](../../outputs/test/model/pre9_20261004_450661f)；
資料夾名稱沿用凌晨工作，不能以名稱推斷完成時間。重點證據：

- [九組收集結果](../../outputs/test/model/pre9_20261004_450661f/results.json)
- [TFDS 全資料讀回與 split](../../outputs/test/model/pre9_20261004_450661f/tfds_multitask_readback.json)
- [模擬 HTTP 錯誤短測](../../outputs/test/model/pre9_20261004_450661f/rollout_http/rollout_http_results.json)
- [真實訓練與權重檢查](../../outputs/test/model/pre9_20261004_450661f/training_evidence.json)、[訓練 log](../../outputs/test/model/pre9_20261004_450661f/training_short.log)
- [真實模型推論／rollout](../../outputs/test/model/pre9_20261004_450661f/openvla_end_to_end/results.json)
- [patch 套用與 CPU 驗證](../../outputs/test/model/training_audit_20261004/patch_verification.json)
- [完整單元測試摘要](../../outputs/test/model/pre9_20261004_450661f/unit_tests_final.log)、[版本與檔案指紋](../../outputs/test/model/pre9_20261004_450661f/final_state.json)

OpenVLA 原外部 checkout 沒有修改；獨立副本及模型都在 ignored outputs，
沒有修改 Python 環境或下載新依賴。測試結束後自有程序均停止。

### 尚待驗收與後續工作

GUI／YOLO 畫面、正式 Unity 顯示與事件、人工判斷五視角和落點，仍按
[人工操作流程](validation_history.md#2026-10-03-manual-checklist) 執行。先前 Remote 重複任務與重連
結果仍見 [歷史紀錄](validation_history.md#450661f)，本次沒有數小時直播壓力測試。

一個更新只證明模型訓練、匯出、推論和模擬控制通路可用。仍需較完整資料、
正式訓練、保留集與多 seed 任務成功率評估，才可判斷策略品質。

MultiView 額外視角與 object poses 的寫入仍在 core H5 寫入之後；此後段若
出錯可能留下不完整集，驗證工具會拒絕，目前沒有宣稱整段皆可 rollback。
這是本輪當時的限制；接續已修正並完成故障注入、三物件收集與 TFDS 讀回，
結果見上述新紀錄。

實體唯讀同步與現場路徑驗收、較長時間穩定性、問題修正後重測，以及提交／
合併審查仍是獨立工作。此紀錄不表示只剩人工驗收，也不表示全專案完成。

<a id="2026-10-04-multiview-rollback"></a>

## 2026-10-04 MultiView 寫入回復補強

分支仍為 `refactor/unified-runtime`，HEAD 仍為 `450661f`，修正尚未提交。
這是 [前一輪整合驗證](validation_history.md#2026-10-04-integration) 之後的接續工作。
全程未操作實體機器人；場景、相機、runtime、收集時序與動作流程未修改。

### 修正結果

原先 MultiView 先完成 core H5，再另開檔案寫入 top／left／right、三物件
姿態與 gripper tip；後段失敗可能留下被 `num_demos` 計入的不完整集。

現在 core writer 提供選用的 keyword-only `write_extensions(group)`。
MultiView 所有額外資料在同一次開檔、既有回復範圍內寫入；全部成功後才
更新集數並 flush。發生 Python 寫入例外或 KeyboardInterrupt 時，移除本次
新增集並還原 tracked root metadata，保留舊集且可用同 ID 重試。
這項回復不涵蓋強制殺程序、斷電或儲存裝置失效。

寫入前核對五視角相同 `(T,H,W,3)` shape 與有限值、各物件 `(T,7)` 姿態及
`(T,3)` tip。原有四參數 core 呼叫維持相容；dataset 名稱、數值、dtype、
gzip 4 與 chunks 保留。`fields` metadata 補列本來已有的 tip dataset。

### 驗證

| 項目 | 結果與範圍 |
| --- | --- |
| 完整 CPU 測試 | 108 個通過；Python compile、Git diff check 通過 |
| 新增 MultiView 測試 | 10 個，含七個 extension dataset 的寫入後 OSError／KeyboardInterrupt、新檔／舊集、fields／count／flush 失敗、同 ID 重試、防覆寫及非法輸入；舊 16 個 H5 測試也通過 |
| 獨立審查與相容性 | 三物件既有資料共 155 幀經新 writer 重建，各 dataset 值、shape、dtype、壓縮與 chunks 一致，來源 SHA 不變 |
| 真實 buffer 故障注入 | 52 幀五視角資料重播完全一致；在最後 tip dataset 寫入後注入失敗，舊集及 root metadata 保留、半集移除、同 ID 重試成功 |
| 實際 MultiView 收集 | spoon 44、red_mug 52、bowl 59 步，各 1 集，五視角、物件姿態與 tip 的 H5 驗證均通過 |
| 五視角影片 | 三物件共 15 支，逐幀可解碼且非空畫面，640×480、5 FPS，各自幀數與 H5 相同；此輪沒有重做瀏覽器播放驗證 |
| 共用 writer 實際回歸 | single red_mug 1 集／48 步；Multi-env（2 env）red_mug 2 集／104 步，退出碼與 H5 驗證通過 |
| MultiView → TFDS 1.1.0 | 真實 CPU build：train 2 集／103 步、val 1 集／52 步；全部影像、state15、action8、指令、metadata、RLDS flags、reward、discount 與 H5 相符，512D embedding 有限且同集一致 |
| TFDS split 與來源保存 | train／val 互斥且完整涵蓋三來源，key 唯一；重排、重複及 symlink alias 不改分割；三份 H5 的建置與讀回前後 SHA 不變 |

TFDS 維持 **main／hand 兩視角**；top／left／right、物件姿態及 tip 仍完整
保留在 H5，沒有擴充模型輸入。5 Hz 是既有模擬取樣時序與 metadata，
影片的 5 FPS 已讀取驗證；沒有量測實體相機時間。

本輪產物集中於
[`outputs/test/data/multiview_rollback_20261004/`](../../outputs/test/data/multiview_rollback_20261004)：

- [收集與 15 支影片結果](../../outputs/test/data/multiview_rollback_20261004/results.json)
- [52 幀真實資料回復與重試](../../outputs/test/data/multiview_rollback_20261004/real_buffer_results.json)
- [TFDS 全資料讀回](../../outputs/test/data/multiview_rollback_20261004/tfds_check/readback_results.json)
- [獨立審查、相容性與 source SHA](../../outputs/test/data/multiview_rollback_20261004/review.json)
- [單元測試](../../outputs/test/data/multiview_rollback_20261004/unit_tests.log)、[最後版本與檔案指紋](../../outputs/test/data/multiview_rollback_20261004/final_state.json)

自有模擬程序已結束；當時常駐的兩個 Ollama 模型程序保留。
GUI／Unity、人工畫面與落點、正式模型成功率、長時間穩定性及現場同步仍需
各自驗收，這輪沒有重跑前一輪的 GPU 模型訓練或宣告全專案完成。

<a id="2026-10-04-remote-readiness"></a>

## 2026-10-04 Remote 實驗前確認

分支為 `refactor/unified-runtime`，HEAD 仍是 `450661f`；本頁涵蓋目前未提交
修正。結論限定於 Remote Isaac／Unity 模擬橋接，不包含實體機器人或
real-to-sim 控制。

### 指令與場景契約

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

### Remote 的兩項修正

- `run_remote_pick_place.py` 在 local 與公開 state 都完成轉移後才解除 command
  reservation，避免 trajectory 建立期間第二個 task 收到 202，之後卻被略過。
- `/control` 對 list／dict 等非字串 action 回傳 HTTP 400，不再因 unhashable
  payload 變成 500。

這兩項沒有改啟動參數、任務配對、軌跡或成功判準。針對實際 command AST、
main AST 與 ASGI handler 的 6 個交錯請求測試，在 `450661f` 全數重現失敗，
目前版本 6 個全數通過；完整 CPU suite 為 119 個測試、0 skips。

### 30 分鐘實際 headless 長測

從第一個 `waiting` 開始連續執行 1800.007 秒，結果 PASS：

| 檢查 | 結果 |
| --- | --- |
| 正式任務 | `0→0`、`1→1`、`2→2` 各兩次，共 6 次，內建成功判準全部成功，complete event 各一次 |
| pause／resume／reset | 每一輪都驗證；pause 時 progress 保持且影像繼續產生；另取消 trial 6，沒有錯誤 complete event |
| 串流 | 40,755 幀 JPEG 全部可解碼且非單色；同一 observer segment 最大 gap 0.4301 秒 |
| 慢 client | 47 秒不讀 frame 時，健康 viewer 仍收到 1,145 幀；慢 client 在 2.001 秒內關閉，重連成功 |
| 無 viewer／重連 | 無 viewer 時 server 持續產生 frame；running 重連成功；無 viewer 完成後重連只 replay 一次 complete |
| 來源穩定 | runner、bridge、scene、runtime 等 15 個來源 SHA 在測試前後相同 |
| 自有資源 | 60 個樣本；RSS 8200.60→8216.82 MiB，範圍 16.22 MiB；owned GPU 5557→5559 MiB |

RSS 在這個單次 30 分鐘樣本的 5 分鐘後線性斜率約 0.7703 MiB／min；這是
描述性數值，不能據此宣稱永遠沒有 memory leak。測試完成後自有 process 正常
停止，8100 已釋放；其他人的 Ollama process 沒有停止或納入 owned 資源數字。

完整證據：

- [Remote readiness 彙整](../../outputs/test/remote/history.json)
- [30 分鐘長測](../../outputs/test/remote/history.json)
- [競態修正前後](../../outputs/test/remote/bridge_audit_20261004/regression_results.json)
- [119 個測試與穩定版本前檢](../../outputs/test/preflight/history.json)

### 驗收邊界

- 這次使用 Python client 驗證相同 HTTP／WebSocket protocol，正式 Unity 畫面與
  Unity 端事件處理仍由人工驗收。
- `seed` 目前只保存及回報 metadata；reset 恢復固定 canonical spawn，不能將
  不同 seed 稱為不同隨機場景覆蓋。
- frame queue 上限為一幀、採 latest-wins；每個 client 的 event queue 仍無上限。
  本次低頻事件與 47 秒慢 reader 正常，不能推論任意事件洪水下記憶體有界。
- 實體同步 runner 沒有執行。只讀審查發現另兩個實體／同步路徑也有較早解除
  reservation 的寫法，需另做 CPU 重現及現場驗證，不能沿用這份 Remote 結論。

在目前來源與環境下，可接著做 Remote Unity 人工驗收及固定 canonical 場景
實驗。正式實驗前仍應執行 preflight、確認 8100 未被舊程序占用，再啟動 runner。
目前修正尚未提交；實驗前不要用 `git reset`、切回乾淨 `450661f` 或清除
working tree，否則會失去上述兩項 Remote 修正。這次長測只涵蓋正式配對
`0→0`、`1→1`、`2→2`；其餘六種組合沒有包含在 readiness 結論中。

<a id="2026-10-05-status"></a>

## 2026-10-05 自動驗證與交接狀態

目前分支為 **refactor/unified-runtime**，HEAD 仍是 **450661f**；本頁涵蓋尚未提交的工作樹修正。這些結果支持進入人工驗收，不代表整個專案或實體機器人路徑已完成。

### 已完成

- Remote 啟動參數、HTTP／WebSocket 路徑、canonical 場景、三個物件與目的地編號、軌跡及成功判準均未改。原指令與第二次 30 分鐘長測見 [Remote readiness](../remote_readiness.md)。
- 目前 Remote 來源完成 6 次正式配對任務、pause／resume／reset、慢 client、無 viewer 與重連驗證；43,438 幀可解碼，測試後 8100 已釋放。
- 完整純 CPU suite 為 134 tests、0 skips；compile 與 Git diff check 通過。
- spoon、red_mug、bowl 三份實際 MultiView H5 通過只讀驗證，檢查前後 SHA 相同。最終報告在 [preflight report](../../outputs/test/preflight/history.json)。
- H5 writer 現在會在開檔前拒絕錯誤 shape、非數值、非有限值、浮點或超出 uint8 契約的影像；single、Multi-env 與五視角共用同一規則。
- sim-to-real 僅接受已實作的 reset；pause／resume 回 HTTP 409。兩條同步 loop 的 command reservation 會等狀態套用成功後才解除，相關測試只使用 CPU mock。
- OpenVLA patch 會把最終模型寫入 sibling staging，確認 weights／shards、processor、tokenizer、config 與 dataset statistics 完整後才發布；既有 final path 直接拒絕，失敗不會混用舊權重與新 statistics。

### 尚待人工或外部環境驗收

- Unity 實際畫面、任務指令與 result event。
- [三物件五視角驗收頁](../../outputs/test/data/acceptance_gallery_20261004/index.html) 的人工視覺檢查。
- 用更新後的 OpenVLA atomic export patch 在乾淨外部 checkout 再做短訓練與匯出；2026-10-04 的真實一更新訓練證據早於這項產物發布補強。
- real-to-sim 現場唯讀同步，以及限定 obj=1、dest=2 的 sim-to-real 實體路徑。
- Remote 長測只涵蓋 0→0、1→1、2→2；其餘六種配對沒有包含在 readiness 結論。

### 交接注意

目前修正仍未提交。Remote 實驗前不要執行 git reset --hard、git clean 或切回乾淨的 450661f，否則會失去本頁驗證的修正。先執行 acceptance preflight、確認 8100 未被占用，再使用 [Remote readiness](../remote_readiness.md) 內原本的啟動與 SSH 指令。

<a id="450661f"></a>

## 450661f 自動測試與人工交接紀錄

日期：2026-10-02（Asia/Taipei）。驗證分支：`refactor/unified-runtime`。
程式版本：`450661f6b6227e224ce55e54e0f2a9f560a019f1`。

第一輪只執行本機 headless 模擬、HTTP/WebSocket 與資料轉換，當時只修改
文件。後續補測和工具準備另記於下方。全程未啟動 ROS driver、實體控制或
Unity，未修改場景、MultiView、runtime 或收集程式。

### 本次已驗證

| 項目 | 結果 | 證據與範圍 |
| --- | --- | --- |
| 單元測試 | PASS | 20 個測試通過；Python compile 與 Git diff check 通過 |
| Canonical scene | PASS | `run_scene.py --headless --enable_cameras --max-steps 180`，退出碼 0 |
| Remote HTTP | PASS | `/health` 回傳 `ok`、protocol 1；初始化到 waiting，具有 object poses、gripper 與 visibility 資料 |
| Remote 正式任務 | PASS | seed 42；`0→0`、`1→1`、`2→2` 各 1 次，內建判準均回報 `result.success=true` |
| Remote 串流與事件 | PASS | 同一條 WebSocket 收到 1,050 張可解碼且像素並非全相同的 JPEG；最大相鄰收訊間隔 0.369 秒；各任務收到恰好一次、trial ID 與結果相符的 complete event |
| pause / resume / reset | PASS | pause 持續 2 秒，progress.t 保持不變、影像繼續收到；resume 回到 running；三次任務後 reset 均回到 waiting、seed 42 |
| 單環境 H5 | PASS | red_mug 1 筆、48 幀；main/wrist 640×480 RGB、state 15D、action 8D、5 Hz metadata |
| MultiView H5 | PASS | red_mug 1 筆、52 幀；五路 640×480 RGB 與 state/action 長度一致；三個物件的 7D pose、單位 quaternion、3D gripper tip 均為有限值 |
| MultiView 影片 | PASS | main、wrist、top、left、right 五支 MP4 全部可逐幀解碼，各 52 幀、640×480、5 fps |
| MultiView 防覆寫 | PASS | 不加 overwrite 重跑回報 FileExistsError、退出碼 1；既有 H5 的 SHA-256 完全相同 |
| MultiView 未收滿 | PASS | 要求 2 筆、最多嘗試 1 次：log 顯示 collected 1/2、退出碼 1；保留的 1 筆 H5 仍通過五路 schema 檢查 |
| TFDS build 與讀回 | PASS | 本次單環境 H5 轉成 train 1 episode、48 steps；逐步讀回 image、hand_image、state、action、instruction 與 H5 完全一致 |
| RLDS 欄位 | PASS | 512D language embedding 為有限值；first/last/terminal、reward、discount 與來源 episode metadata 正確 |
| 結束清理 | PASS | 自動 runner 已結束；8100/8000 無 listener，本次模擬與轉換程序均已退出 |

兩種 H5 都檢查 state/action 的有限值、logical binary gripper、只在最後
一幀設置 terminal flag，以及末幀六個運動分量歸零。五路長度一致只證明
儲存資料對齊；各視角的構圖與動作內容仍需人工看影片確認。

Remote runner 在測試完成後主動送 SIGINT 結束程序，取得退出碼 130；
這是本次測試的預期停止方式。其他正常收集與場景短測退出碼均為 0。
Isaac 啟動 log 仍有 headless GLFW 和 `omni.anim.graph.core` 註冊警告；
上述功能檢查通過，本次沒有變更相容性設定或處理這些警告。

### 原始證據與重現方式

本機產物位於 `outputs/test/data/acceptance_20261002_450661f/`，Git 忽略此目錄。
文件可以隨 Git 交接；影片、H5、TFDS 與 log 不會隨 Git clone 出現。

- [自動檢查結果](../../outputs/test/data/acceptance_20261002_450661f/results.json)、[TFDS 讀回結果](../../outputs/test/data/acceptance_20261002_450661f/tfds_results.json)。
- [Remote 各任務結果](../../outputs/test/data/acceptance_20261002_450661f/remote_trials.json)、[事件與影格收訊紀錄](../../outputs/test/data/acceptance_20261002_450661f/remote_events.json)、[首張直播影格](../../outputs/test/data/acceptance_20261002_450661f/remote_first_frame.jpg)。
- [MultiView 影片目錄](../../outputs/test/data/acceptance_20261002_450661f/multiview/red_mug/videos)。
- `unit_tests.log`、`remote.log`、`scene.log`、`single.log`、`multiview.log`、
  `multiview_existing.log`、`multiview_partial.log`、`tfds_build.log`、
  `tfds_verify.log` 保留完整輸出，`progress.log` 僅列摘要。
- `run_acceptance.py` 與 `verify_tfds.py` 保留本次自動檢查程式，僅存在
  測試目錄；前者拒絕重用已產生的正常收集 H5。

正常重測請依 [full_test_commands.md](../full_test_commands.md) 第 2–5、8 節
執行，改用新的測試目錄；本次正常收集均未加入 `--overwrite`。單環境與
MultiView 都用 `--target red_mug --episodes 1 --max-episodes-tried 2`；
MultiView 另加 `--save-video`。防覆寫測試重用 MultiView 目錄、不加
`--overwrite`；未收滿測試用另一個新目錄、`--episodes 2` 搭配
`--max-episodes-tried 1`。

TFDS 使用獨立的 `rlds_env`（TensorFlow 2.13.0、TFDS 4.9.2、TF Hub 0.14.0、
h5py 3.9.0），禁用 GPU。輸入為本次 `single/red_mug/demos.h5`，
`UR3E_VLA_VAL_RATIO=0.0`、`UR3E_VLA_SPLIT_SEED=42`，產物在本次測試
目錄的 `tfds/ur3e_vla_dataset/1.0.0/`。這次沒有驗證 val split 或多物件 build。

### 2026-10-03 人工驗收清單

以下仍未驗收，明天依序確認並記錄操作者、版本、結果與證據：

- [ ] GUI 確認湯匙、紅杯、碗、Robotiq 與 YOLO 構圖，目標 marker 預設隱藏。
- [ ] 透過實際 SSH tunnel 持續觀看 Remote 畫面，確認 pause / resume / reset
  及三條正式配對；檢查視覺落點與動作是否符合展示需求。
- [ ] 對接者使用正式 Unity 專案，確認 JPEG 顯示、state、complete、trial ID
  與 HTTP 控制；Python 協定測試不取代 Unity 驗收。
- [ ] 觀看五支 MultiView 影片，確認每個視角、動作與資料用途，記錄是否接受。
- [ ] 乾淨的外部 OpenVLA checkout 固定在
  `c8f03f48af692657d3060c19588038c7220e9af9`，先以 `git apply --check`
  驗證專案 patch，再做短訓練、匯出、server prediction 與模擬 rollout；
  依 [command_reference.md](../command_reference.md) 的環境與輸出路徑操作。
- [ ] 另訂多次完整 rollout 的物件、seed、成功判準與成功率統計；短版推論
  通路測試不作為模型抓取成功率。
- [ ] 現場另行驗證唯讀關節同步，以及限定的 `obj=1,dest=2` sim-to-real；
  依 [sim_real_sync.md](../sim_real_sync.md) 執行，本次沒有操作實體機器人。
- [ ] 僅修驗收發現的問題，重跑受影響項目；完成文件與提交審查後再決定
  是否合併 main。

本次是固定版本的短測證據；未驗證長時間直播、其他六種配對、YOLO 模型
辨識率、OpenVLA 新 checkout 的訓練／推論或實體路徑，不宣告全專案完成。

### 後續補測與驗收準備

同日補測沿用 `450661f` 的 runtime；另新增只讀的
`scripts/tools/validate_h5.py`、8 個資料檢查測試，並修正
`scripts/tools/export_h5_frames.py` 的夾爪 overlay：原先讀 reserved 欄位
14，改為 logical gripper 欄位 13。原有輸出欄位名稱 `gripper_raw` 保留。

| 項目 | 結果 | 證據與範圍 |
| --- | --- | --- |
| 單元測試 | PASS | 28 個測試通過；新增缺少視角、影像長度錯誤、NaN action、過早 terminal、全黑影格、筆數與預覽標示檢查 |
| Remote 重複任務 | PASS | 同一程序三輪正式配對、共 9 次成功；總測試時間 290.59 秒，含 120 秒待機直播 |
| Remote 收訊 | PASS | 6,355 張可解碼且像素並非全相同的 JPEG，最大相鄰收訊間隔 0.380 秒 |
| 重連與命令 | PASS | running 新連線有影像；done 重連回放 complete 恰好一次；busy 再送 task 得 409；執行中 reset 取消任務且不發成功 complete；另做 5 次連續 reset |
| 唯一 viewer 關閉後重連 | PASS | 所有 viewer 離線時仍持續編碼影像；任務 running 可重新連線，無人觀看時完成的 red_mug 任務在 done 重連回放 complete 恰好一次 |
| 資源採樣 | PASS | 此測試中程序群 RSS 採樣 8,187.5–8,189.6 MiB，GPU 5,946–5,948 MiB，未見持續增加；這不是數小時壓力測試或記憶體洩漏的完整證明 |
| 單環境防覆寫 | PASS | 重用既有單環境 H5、不加 overwrite：FileExistsError、退出碼 1，SHA-256 不變 |
| 單環境未收滿 | PASS | 要求 2 筆、最多嘗試 1 次：collected 1/2、退出碼 1，保留的 48 幀 H5 通過驗證 |
| 共用 H5 驗證工具 | PASS | 實際讀過單環境、Multi-env、MultiView 與部分收集 H5，檢查儲存的 schema、數值、黑圖與 metadata，檔案未被修改 |
| 預覽與人工表單 | PASS | 六支 H.264 預覽副本在隔離的 headless Chrome 載入；五視角播放／暫停／歸零與 Markdown 下載均實測，人工項目保持未勾選 |

補測產物位於 `outputs/test/data/acceptance_20261002_followup_450661f/`：

- [補測結果](../../outputs/test/data/acceptance_20261002_followup_450661f/results.json)、[完整 Remote 收訊與資源採樣](../../outputs/test/data/acceptance_20261002_followup_450661f/remote_retry/remote_repeat_evidence.json)、[唯一 viewer 重連結果](../../outputs/test/data/acceptance_20261002_followup_450661f/last_viewer/results.json)。
- [H5 驗證與 SHA-256 紀錄](../../outputs/test/data/acceptance_20261002_followup_450661f/h5_validation_results.json)。
- [瀏覽器檢查結果](../../outputs/test/data/acceptance_20261002_followup_450661f/preview_results.json)、[影片與填表頁](../../outputs/test/data/acceptance_20261002_followup_450661f/index.html)。
- 首次補測腳本誤把 reset 前的 waiting 當成 reset 已完成，連送命令收到
  409。腳本改為等待 resetting → waiting 後重跑通過；runtime 未修改。
  `first_attempt_results.json` 與首次 log 保留。
- 原始 OpenCV MP4 可解碼，但 Chrome 無法播放其 codec；另製作 H.264
  預覽副本，沒有替換原始影片或 H5。`preview_first_attempt.log` 保留。
- 人工操作步驟見 [acceptance_20261003.md](validation_history.md#2026-10-03-manual-checklist)。

上述程式與文件變更尚未提交；驗收時需另記實際 HEAD 和工作目錄差異。
