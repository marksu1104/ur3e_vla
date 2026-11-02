# Luna 接手計畫與驗收狀態

更新：2026-11-03。這是接手工作的單一入口；後續更新這份文件，不另增 README 或日期版計畫。

## 直接交給 Luna 的任務

> 在 `~/IsaacLab/ur3e_vla` 接續 `refactor/unified-runtime`。先完整讀本文件並完成 L0，再按 L1、L2 推進離線工作。功能基線是已推送的 `0a5d895`；後續文件提交的 HEAD 可能不同，不要 reset 回基線。保留 Remote 指令、共用場景、MultiView、資料與模型。預設只做純 CPU／Mock／AST、唯讀盤點及小範圍修正。不得自行連接 ROS 或操作實體機器人；dry-run 也不算離線。下載、GPU、長時間收集／訓練依下列階段取得授權後才執行。每項工作先重現問題、補必要測試，再修正並回報證據。不要降低測試門檻、掩蓋失敗或宣稱整個專案完成。

本文件授權範圍內的例行唯讀、離線修正、測試與同分支提交可以持續做，不必每個命令再問一次；超出範圍時說明具體要做什麼。現場操作預設交給使用者，Luna 只準備命令並分析使用者提供的結果。

## 目前基線：已完成什麼、尚未完成什麼

| 項目 | 截至功能基線的證據／界限 |
| --- | --- |
| Git | 工作分支 `refactor/unified-runtime`；`0a5d895` 已 push。`main` 尚未合併，不能當成含有最新修正的版本 |
| CPU | 16 個測試模組、191 tests、0 skips；語法與 Git diff check 通過 |
| Remote | 原啟動參數、HTTP／WebSocket 路由與共用場景保留。歷史 30 分鐘長測三組正式配對各兩次成功、43,438 幀；來源 SHA 已比對。這不是新跑的長測，Unity 仍待人工驗收 |
| H5／MultiView | 防覆寫、Python 例外回滾與嚴格資料輸入檢查；五視角 H5／影片保留。6 份真實 H5、9 episodes 通過新版 parser，來源 SHA 未變。不是大量資料或人工視覺品質驗收 |
| 實體 VLA 程式 | 單一背景推論；過期影像／action、晚到結果不套用；異常退出先零速清理。只有 CPU 證據，未驗 ROS 排程或物理停止 |
| 同步夾爪程式 | SetIO 接受狀態、request token、請求後新鮮 DI 與 timeout 已補強；沒有做實體操作 |
| TFDS | builder **1.1.1**，schema 不變；已補原始 dtype、shape、finite、四元數、gripper、terminal、黑幀與 task 檢查。真正序列化 build 未做：歷史 USE cache 已不存在，沒有另下載 |
| 模型 | 專案可稽核整合證據只有一次 optimizer update 與通路 rollout，早於最新 atomic-export 補強。使用者另外訓練／實機試過的模型，尚未與本專案版本對照 |
| 實體現況 | 使用者已安裝夾爪，VLA 曾勉強對準但難以抓起。不能再寫「沒有夾爪／忽略夾爪」，也不能把此現象直接判成只需重訓 |

本機歷史證據：[最新前檢](../outputs/test/preflight/hardening_20261103/report.json)、[真實 H5 parser 相容性](../outputs/test/preflight/hardening_20261103/parser_compatibility.json)、[推送版本紀錄](../outputs/test/preflight/hardening_20261103/commit.json)。它們在 ignored outputs，新的 clone 不一定有；連結不存在不能偽造結果。

## 可做／不可做

| 分類 | Luna 的執行範圍 |
| --- | --- |
| 可以自行做 | Git／程式／文件盤點、既有資料唯讀檢查、純 CPU／Mock／AST、重現錯誤、小範圍修正、更新既有文件、保留證據；通過檢查後 commit／普通 push 到目前工作分支 |
| 可以準備、需資源授權後執行 | USE／模型下載、純 Isaac 模擬、GPU server／短訓練、批次收集。先列出模型／資料、大小或資源上限、預計執行時間、全新輸出路徑及停止方式。使用者已授權該階段後，不重複詢問每一條命令 |
| 只由使用者現場執行 | ROS topic／服務驗證、real-to-sim、sim-to-real 或實體 VLA dry-run、夾爪開關、實際抓放。Luna 預設不執行 `rclpy.init()`、啟 UR driver／Servo、切 controller、呼叫 SetIO、發送實體 task/reset，或加入 `--enable-motion` |
| 未另行授權禁止 | merge／push `main`、force push、刪除分支、清除或 reset 他人修改、覆寫原始 H5／模型、變動共用環境／驅動／資產根目錄、任意修改場景／路徑／物件編號、停止他人程序 |

**dry-run 不是離線模式。** `sync_sim_real.py` 與 `real_vla_servo.py` 即使沒有 `--enable-motion`，仍可能初始化 ROS、訂閱或建立服務／publisher；純離線驗證只能用測試中的替代物件。

**8100 的健康回應不能證明它是純模擬。** Remote 和同步共用 port；`test_bridge_client.py --obj` 或 `--control` 連錯 server 可能觸發實體命令。必須核對程序的啟動命令、擁有者及 `/status`／supported tasks；只有 `/health` 或 port 號不足以判斷。禁止把不明 server 當成測試對象。

## 設定與功能契約：不要為了通過而放寬

| 設定 | 目前契約 |
| --- | --- |
| 場景與物件 | `spoon`、`red_mug`、`bowl`；編號 `0、1、2`。正式 Remote 配對 `0→0、1→1、2→2`，其他六組沒有相同驗證覆蓋 |
| Remote 指令 | 保留下方原指令與原八個額外參數；pause／resume／reset 全保留。bridge `127.0.0.1:8100`、SSH tunnel 本地 `18100` |
| 場景來源 | 共用數值 `vla_sim/config.py`，場景 `vla_sim/scene/`。不要複製另一套 scene／常數，不因實機問題改掉 canonical 場景 |
| 相機 | `camera_yolo` 用於直播／檢視；canonical policy 輸入是 `camera_policy`，手部是 `camera_wrist`。MultiView 再加 top／left／right，不改視角或輸出欄位 |
| MultiView 意義 | H5 保存五視角、pose／tip；TFDS main／wrist；目前 OpenVLA 主視角＋文字。不是五視角 VLA，也不是已使用 proprioception |
| seed | seed 42 保存 metadata，reset 恢復固定 canonical 位置；不能說已做隨機場景泛化驗證 |
| 動作 | policy 7D `[dx,dy,dz,droll,dpitch,dyaw,gripper]`；H5 8D 另加 terminal。邏輯夾爪 `0=open、1=closed`，由現有 controller 對應 Robotiq `finger_joint`，不能混用實體關節值 |
| 座標／四元數 | 保留既有 frame 補償、`pose_wxyz_from_sim` 與 quaternion adapter；不能直接把 Isaac API 原始 pose 當專案固定格式，也不能把模擬 workspace 當成實機標定 |
| 實體 VLA defaults | query 1 Hz、publish 10 Hz、duration 3 秒（初始化完成後起算）、HTTP timeout 10 秒；linear gain 1、angular gain 0、max linear 0.01 m/s、max angular 0.03 rad/s |
| VLA freshness | `--image-timeout=0.5` 秒；`--action-timeout=1.5` 秒，從輸入影像的本機接收時間起算，不能用回應到達時間刷新。worker 最多一筆、不直接呼叫 ROS；過期保持夾爪，不重播舊速度 |
| 同步限制 | 只支援 `obj=1、dest=2` 與 reset；pause／resume 應回 409。只做現有平移／yaw 與 frame 補償，非通用 TCP 控制或 VLA 同步 |
| 同步時間／IO | joint timeout 0.5 秒、DI freshness 0.5 秒、gripper timeout 3 秒、waypoint timeout 90 秒。DI 必須是請求後的新樣本且 SetIO 已接受；本機收到的時間不等於感測器採樣時間 |
| IO 與速度 | DO pin 0 使用 0=open、1=closed；DI0=True 表示開爪、DI2=True 表示夾持確認。這是目前 cell 假設，實際語意由使用者核對。`--allow-high-speed` 是相容旗標，沒有額外隱藏保護；明確速度上限才是上限。Runbook 的 20 Hz／0.05 m/s 範例不是現場標定值 |
| TFDS／OpenVLA | TFDS **1.1.1**；固定外部 OpenVLA revision `c8f03f48af692657d3060c19588038c7220e9af9`；normalization key `ur3e_vla_dataset` 必須與實際模型 statistics 對上 |

原 Remote 指令只供已確認的純模擬程序使用；不要為了準備計畫而啟動它：

```bash
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash
isaaclab scripts/run_remote_pick_place.py \
  --headless --enable_cameras --seed 42
```

## L0：接手版本與離線基線（第一個工作）

1. 看 Git 狀態與最新提交。若工作樹有未知修改，先辨識來源並保留；不要 stash／reset／clean 掩蓋問題。
2. 只在乾淨工作樹切到指定分支、正常 fast-forward 更新；發生衝突或分歧就停下說明，不自行合併 main 或覆蓋遠端。
3. `0a5d895` 是功能基線，不是每次強制切回的 HEAD；必須包含它及後續交接文件提交。

```bash
(
set -euo pipefail
cd ~/IsaacLab/ur3e_vla
test -z "$(git status --porcelain)" || { echo 'Working tree has changes; preserve and inspect them first'; exit 1; }
git switch refactor/unified-runtime
git fetch origin
git pull --ff-only
git merge-base --is-ancestor 0a5d895 HEAD
git status --short --branch
git log -3 --oneline
)
```

在新 terminal 執行 CPU 基線；不要先啟動 Conda，也不要執行 ROS 節點：

```bash
(
set -e
cd ~/IsaacLab/ur3e_vla
source /opt/isaac_ros2/setup.bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python -B scripts/tools/acceptance_preflight.py \
  --h5 outputs/test/data/multiview_rollback_20261004/h5/multiview/spoon/demos.h5 \
  --h5 outputs/test/data/multiview_rollback_20261004/h5/multiview/red_mug/demos.h5 \
  --h5 outputs/test/data/multiview_rollback_20261004/h5/multiview/bowl/demos.h5
)
```

完成判準：目前至少 191 tests、0 skips；diff／compile 通過；三份 H5 前後 SHA 相同；保存工具印出的新 report 路徑。工具自動產生唯一目錄，不重用舊 output dir。

若新 clone 沒有 sample H5，先做不含 `--h5` 的 CPU 前檢並明確回報未驗 H5，繼續不依賴資料的工作。OpenVLA patch 測試若缺來源會 skip：確認本機固定 revision checkout，必要時設定 `UR3E_OPENVLA_SOURCE=/path/to/openvla`。不要降低門檻、刪測試、改成 skip 或用 `--skip-unit-tests` 製造綠燈。新增有意義的案例後，記錄實際測試數並維護門檻。

## L1：模型／資料來源盤點（可立即離線做）

目標是對上「使用者當時實機試的模型」，不是重新下載一個 base model。

- 在使用者指定的模型目錄、專案 outputs 與既有共享目錄唯讀查看；不要遍歷帳號／憑證目錄或載入 7B 權重到 GPU。
- 記錄模型路徑／版本、config、processor／tokenizer、statistics 的 SHA、weight shard 名稱／大小、OpenVLA revision／patch SHA、TFDS version、來源 H5 resolved paths／SHA、訓練參數與相機、unnorm key、action units／frame／頻率與執行增益。
- 大權重不反覆 hash 或載入；必要的完整 fingerprint 只做一次並留紀錄。含本機路徑的 manifest 留在 ignored outputs，公開摘要用相對路徑或變數。
- 不知道使用者模型在哪裡，就一次詢問「上次實機使用的 model path、訓練輸出目錄和執行命令」，同時繼續 L2；不要把專案的一更新 smoke model 當成使用者唯一模型。

完成判準：可追溯的 manifest，或清楚標出哪些欄位缺少、需要使用者提供。不得猜測 normalization、修改 statistics 或將不明 checkpoint 當正式模型。

## L2：成功判準與離線缺口（不需實機即可推進）

- 先定義 `通路正常`、`軌跡完成`、`抓起`、`搬運未掉落`、`釋放`、`落點成功` 的區別及需要哪些證據。未取得證據的階段標為 unknown，不補成 True。
- 為模型評估準備逐次記錄欄位與小型離線測試；`complete` event、HTTP 200、exit 0 都不能直接當物件抓放成功。保留既有 Remote protocol，不為新增指標破壞 Unity 格式。
- 模擬 VLA 目前關爪 latch 後不放開；如需完整 pick-and-place 評估，先提出局部改法與回歸案例，不能順手改 Remote 軌跡或成功判準。
- 對 workspace／桌面界線先整理所需 frame、TCP、限制來源與離線測試案例；沒有現場校準值時只做設計／測試，不把猜測座標或模擬值寫成實機預設。
- 每次只處理一個具體缺口，先有失敗案例再最小修正。先跑相關模組，最後一次完整 suite；文件小改只檢查連結／命令語法，不反覆重跑整套。

完成判準：可審查的指標規格／局部實作、對應測試、明確未做項目。不能提交讓整套測試失敗的版本，也不能用 expectedFailure 或忽略錯誤冒充完成。

## L3：USE 快取與 TFDS 1.1.1 真正 build（條件式）

先檢查既有環境與持久快取。已知 `rlds_env` 存在，但原 `/tmp/tfhub_modules` 不在。若需下載，先提出來源、大小／磁碟需求與持久個人 cache 位置，取得該階段授權；只使用原設定的 `universal-sentence-encoder-large/5`，不要替換 embedding 模型或安裝到共用 Isaac runtime。

快取可用後：

1. 新 terminal 啟用 `rlds_env`；設定明確 `TFHUB_CACHE_DIR`。不要把暫存 `/tmp` 當長期依賴。
2. 新建 `outputs/test/data/luna_<RUN_ID>/`，以現有已驗證的三物件 MultiView H5 build 到 TFDS **1.1.1**，不使用 `--overwrite`。`UR3E_VLA_H5_PATHS` 優先於單一路徑；使用一種時 unset 另一種，固定 val ratio 0.1、seed 42。
3. 真的序列化後重新載入：目前三份 sample 預期 **3 episodes／155 steps、2 train／1 val**；檢查 main／wrist、state／action、dtype／shape、first／last／terminal，與 H5 對照。
4. 記錄來源 H5 在 build 前後 SHA 一致；保留 resolved input list，搬動 H5 會改 episode key／split。

完成判準：真實 TFDS build＋讀回證據。Mock embedding／parser 相容性不能代替。缺 cache 時回報 blocked-by-resource，繼續 L1／L2，不偽造 build 通過。

## L4：最新 OpenVLA patch 的真實短訓練與匯出（需 GPU 授權）

先有 L3 產物、固定 base weights、獨立 `openvla` 環境，以及已核准的 GPU／執行時間。下載與長訓練不能包含在「先做短測」的假設內。

- 用外部 OpenVLA 的新 clean checkout／worktree，固定 revision；不 reset 現有 dirty checkout，不重複 apply patch。先 `git apply --check`，失敗就停止查明。
- 先做有界 **1 optimizer update、batch 1**；不要盲貼正式範例的 2000 steps、500 episodes 或開並行工作。記錄 gradient accumulation，max_steps 是 optimizer updates。
- `final_model_dir` 必須全新、與 run directory 不同，不共用舊路徑。驗證 config、processor／tokenizer、statistics、所有 weight shards 完整，失敗不得留下看似成功的 final export。
- server 明確指定本機 `--model-path`、`--unnorm-key ur3e_vla_dataset`、`--host 127.0.0.1`；不要無參數啟動去拉 base model。以既有 H5 影像做一次有界 HTTP 推論，確認 7D finite action 和正確 statistics。
- 如另做模擬 rollout，先確認只啟動自己的純 Isaac 程序；保存實際步數／action／errors。它仍不是抓取成功率驗證。

完成判準：最新 patch 的真實 update、原子匯出、載入及推論均有證據。一次更新只驗通路，不宣稱學會抓取。完整命令見 [Command Reference](command_reference.md)；檢查失敗必須停止，不得繼續下一行。

## L5：人工驗收與現場階段（Luna 不代替使用者操作）

| 順序 | 使用者執行／確認 | Luna 的工作與通過標準 |
| --- | --- | --- |
| 1 | Unity 連線、直播、pause／resume／reset、三條正式配對、state／complete／trial ID、重連 | 準備 [Remote 指令](remote_readiness.md)、分析紀錄；Python client 測過不代替 Unity |
| 2 | 看 [三物件五視角頁](../outputs/test/data/acceptance_gallery_20261004/index.html)，確認構圖、夾爪、動作、落點 | 保留未勾選模板，只有使用者真的確認才記通過，不把自動表單測試當人工結果 |
| 3 | 六軸唯讀同步，確認 joint names、FK／TCP、桌面與虛實 frame；單獨確認夾爪 open／close、DI 語意和更新頻率 | 核對 log／參數；不能因訊息存在就宣稱標定完成 |
| 4 | dry-run，再驗限定紅杯→dest2 固定路徑與 reset 的現場流程 | 逐次記錄抓起、搬運、掉落、釋放、落點；不要只計 trajectory complete |
| 5 | 固定路徑穩定後，再做 VLA 的接近／下降／閉爪／抬升分階段比較 | 對照 L1 模型與相機、raw action、applied velocity、回授；先定位問題，再決定補資料重訓 |

sim-to-real motion enabled 的 reset **會開實體夾爪，並從當前位置直線回 HOME**；不是清除虛擬物件而已。其他物件／目的地、通用 TCP、VLA 同時帶動虛實都屬後續開發，不能把限定路徑通過當全面支援。現場命令只由使用者依 [同步 runbook](sim_real_sync.md)／[實體 VLA runbook](real_robot_vla_runbook.md)執行。

## 曾踩過的坑：不可用這些方式「修好」

1. **環境混用／就地升級。** Isaac／ROS 用新 terminal `source /opt/isaac_ros2/setup.bash`；TFDS 用另一個 `rlds_env`；訓練／server 用另一個 `openvla`。不往共用 runtime 裝 TensorFlow／OpenVLA、不改 `/opt/isaac_ros2/current`，不獨立升降驅動、Isaac、Torch 或 ROS。基線版本見 [README](../README.md)。
2. **重建另一套場景。** 保留共用 scene options、相機、三物件、marker、軌跡與 MultiView。Multi-env 依賴 assembled robot USD；未改 robot／gripper 時不必重匯資產。不要復活舊 mug scene 或錯誤 home pose。
3. **API／四元數錯配。** 使用現有姿態與狀態 adapter，不直接替換舊 API accessor 或猜 quaternion 排序。
4. **H5 覆寫。** collector 的 `--overwrite` 會刪舊 H5；同 episode ID 不可覆蓋。每次新 dataset／run 名稱，保留 single、Multi-env、MultiView、legacy 的 profile／action 差異。例外回滾不等於斷電／程序崩潰保護。
5. **數值 cast 掩蓋錯誤。** float image 不能 cast 成 uint8「修正」；NaN、非二元 gripper、錯 terminal 都應拒絕。獨立 H5 validator 仍須檢查 metadata 與 MultiView 額外資料；5 Hz metadata 不代表已量測採樣 jitter。
6. **資料／模型名稱當證據。** `mugs_500` 目前實際只有 3 episodes，缺新 metadata；新版正式目錄仍未備妥。不得混入新 profile、補假的 metadata 或把一更新 smoke model 當正式策略。
7. **舊 TFDS 快取當新結果。** 1.0.0／1.1.0 不會自動得到 1.1.1 驗證；來源路徑改動會改 key／split，必須保存 input manifest。
8. **OpenVLA dirty checkout／舊 final model。** 不 reset 別人的 patch，不重複 apply，不混舊權重／新 statistics，不讓兩次訓練共用同一 final path；缺檔不是改路徑或刪檢查就能通過。
9. **shell 檢查失敗卻繼續。** 一般 shell 的 `test ... || false` 不保證阻止下一行。危險多行步驟用 fail-fast subshell；遇到錯誤停下，不要只複製後半段。文件中的模板路徑先確認，不靠腳本錯誤猜位置。
10. **假綠燈。** 不刪／skip／降低至少 191 tests 的門檻，不把 parser 當完整 TFDS、不把 exit 0／HTTP 200／complete 當抓放成功、不拿舊 report 套到新版本。
11. **實機與模擬誤接。** dry-run 仍連 ROS；8100 也可能是實體同步。中央 motion gate、單一背景推論與 shutdown zero 不得繞過；worker 不直接操作 ROS。
12. **executor／回授回退。** joint 和 IO node 用目前 shared executor；各自重複 spin_once 曾造成互相飢餓。保留 request token、新鮮 DI、晚到結果丟棄，不能靠放寬 timeout 或重播舊資料續跑。
13. **瞎調控制。** 5 Hz 訓練位移增量與實機速度命令尚待校準，不能直接乘 5、反轉 gain、關閉 frame compensation 或提高速度來碰運氣；callback 新鮮不代表物理抓起。
14. **刪到依賴或洩露本機資訊。** 模型目錄的 Python 檔可能是載入必要實作，不是一次性測試。`outputs/test/model/openvla_clean_20261004` 是 patch 測試來源，沒有替代來源不能刪。不要 `git clean`、整包刪 outputs 或清使用者／其他程序的資料。

帳號、IP、cell 的 ROS_DOMAIN_ID／ROBOT_IP、具體路徑只取本機 ignored `docs/local_commands.md`／`docs/shared_environment.md`，不要猜、不要推到 Git、不要貼進公開報告。完整本機指令必須保留；新 clone 缺這些檔時另行私下交接。

## 每項工作怎麼交付

- 開始前記 branch／HEAD／工作樹差異；結束後記修改目的、檔案、命令、退出碼、測試數／skips、輸入／輸出與 report 路徑、H5 SHA、未做項目和剩餘風險。
- 產物只放 `outputs/test/{remote,data,model,preflight}/luna_<RUN_ID>/` 的必要檔，不把大 log 貼在對話。既有工具已會輸出精簡摘要，失敗再看對應 log。
- 一個小問題一筆 commit，包含必要測試與本文件狀態更新；完成檢查後普通 push 到同分支。不 merge main、不 force push、不刪歷史分支。不為了交接再增加 README／同內容快照。
- 測試保護有效但不等於全專案驗收。最終回覆明確分開「已做」「未做」「需要使用者現場提供／執行」，不寫沒有證據的全部完成。

| 工作 | 本次交接時狀態 | Luna 更新欄位 |
| --- | --- | --- |
| L0 基線 | 0a5d895 的 191 tests 通過；待接手重新確認環境 | 日期／HEAD／report／結果 |
| L1 模型來源對照 | 使用者實測模型未完成對照 | manifest 或缺少欄位 |
| L2 指標與離線缺口 | 可立即開始；物理成功判準／界線尚待定義 | 單一工作／測試／commit／限制 |
| L3 TFDS 1.1.1 | 待 USE cache 與資源授權；實際 build 尚未驗 | cache／輸出版本／讀回／H5 SHA |
| L4 新 patch 訓練匯出 | 待資料、weights、GPU 授權；真實新匯出未驗 | revision／patch／update／export／推論 |
| L5 人工／現場 | Unity、視覺品質、同步、實體抓放待使用者驗證 | 操作者紀錄／逐階段結果／失敗案例 |

必要的專用文件：[日常完整指令](command_reference.md)、[完整測試順序](full_test_commands.md)、[資料與產物位置](project_layout.md)、[歷史證據](archive/validation_history.md)。現行狀態與後續計畫以本頁為準。
