# 驗收與目前狀態

最後更新：2026-11-03。分支為 **refactor/unified-runtime**，本輪修正起點為 **450661f**，現行提交以 `git log -1 --oneline` 為準。本分支保存已通過 CPU 與資料檢查的進度，不代表全專案或實體抓取已驗收完成；尚未合併 main。

## 自動驗證已完成

- 2026-10-05 基準為 134 tests，2435856 checkpoint 為 142 tests；本輪在既有測試檔補足過期／晚到推論、同步 IO 回授與 TFDS 驗證，完整 suite 現為 191 tests、0 skips。
- 目前 Remote 來源完成第二次 30 分鐘 headless 長測：正式配對 0→0、1→1、2→2 各兩次成功，pause／resume／reset、慢 client、無 viewer 與重連均通過，共解碼 43,438 幀。
- Remote 啟動參數、HTTP／WebSocket 路徑、canonical 場景、相機、三個物件與目的地編號、軌跡及成功判準未改。
- spoon、red_mug、bowl 三份實際 MultiView H5 通過只讀驗證，檢查前後 SHA 相同；single、Multi-env 與五視角 writer 共用嚴格 RGB uint8 規則。
- 同步 command reservation 的競態已修正；sim-to-real 只接受目前已實作的 reset，pause／resume 會回 HTTP 409。
- OpenVLA patch 會先在 sibling staging 產生並驗證完整模型，再發布到全新的 final path；失敗不會混合舊權重與新 statistics。

歷史自動報告：[2026-10-05 preflight](../outputs/test/preflight/cleanup_20261005/report.json)、[2026-10-30 CPU 與 H5 前檢](../outputs/test/preflight/quick_prep_20261030/report.json)。2435856 提交前執行了 142 個 CPU 測試、三份 MultiView H5 唯讀驗證及 Git 差異檢查；這些本機歷史產物不隨 Git clone 提供。Remote 的來源比對、原指令與長測數字見 [Remote 實驗前確認](remote_readiness.md)。

## 2026-10-30 可先完成的準備

- 依使用者 10 月 14 日回報修正實體文件：夾爪已安裝，VLA 曾實測但難以完成抓取；不是沒有夾爪。
- 推論失敗時，實體 VLA handler 清空上一筆平滑速度並走既有 zero-command 路徑，保留夾爪狀態；下一筆有效回應可恢復。
- 夾爪 SetIO 回傳拒絕或 future 失敗不再記為成功，匹配的請求可重試；較舊失敗不清除新的相反命令快取。
- 動作增益、頻率、Remote 指令、場景、MultiView 和同步控制流程均未改。
- 上述只做 CPU 測試；相機／action 過期停止、實體抓取回授與完成判準仍待補齊，現場效果未驗證。

## 2026-11-03 離線補強

最新本機報告：[191-test 與 H5 前檢](../outputs/test/preflight/hardening_20261103/report.json)。

- 實體 VLA HTTP 推論改為單一背景 worker，主 callback 仍能在等待期間檢查停止；影像接收超過 0.5 秒、action 所用影像接收超過 1.5 秒或 duration 屆滿就不套用結果、不改夾爪。一般例外退出也先送零速再清理，沒有改既有增益／頻率或執行任何實體動作。
- 同步夾爪 SetIO 需真正回傳 accepted；token 保護新的請求。DI 需為請求後收到、非未來且 0.5 秒內的樣本；未確認就保持零速並依既有 timeout 中止。
- TFDS 升 1.1.1，schema 不變；在 embedding 前檢查原始 dtype、shape、finite、四元數、binary gripper、reserved／terminal、末步零位移、主／手部黑幀及 task，禁止靠 cast 掩蓋壞資料。舊版 TFDS 不會自動取得新驗證，需另行 build。
- 全程 CPU 測試；時間戳是本機 callback 接收時間，沒有驗證來源採樣時間、ROS 實際排程或物理停止／抓放效果。未新增 README、工具或測試模組。

## 已知未修缺口與交接範圍

這是進度 checkpoint，不是實體抓取 release。下列仍未完成的範圍保留作為後續工作，沒有執行實體控制：

- 本機 callback 新鮮度不等於相機／IO 來源採樣時間；實機排程、停止距離、工作區／桌面限制與模型 action 到速度的校準仍需驗證。
- 同步 complete 主要代表軌跡走完，沒有真實物件抓起、途中掉落與落點判準。模擬 VLA 的成功退出代表通路運作，夾爪 latch 也不適合直接評估完整釋放。
- TFDS 1.1.1 的真正序列化 build／最新 OpenVLA export 的真實訓練尚待執行。本次檢查發現 rlds_env 存在，但歷史 USE cache 已不在，未另外下載；CPU parser 測試與真實 H5 相容性不能代替完整 build。獨立 H5 validator 仍需檢查完整 metadata 與 MultiView 額外欄位。
- MultiView 指五視角收集；目前 TFDS 只有 main／wrist，而固定 OpenVLA 訓練輸入是主視角與文字，未使用五視角或 proprioception。
- 本機 mugs_500 舊 H5 實際只有 3 episodes、242 steps，且缺新版必要 metadata；檔名或收集指令範例不代表已建立正式訓練集。使用者實機使用的外部模型仍需對照資料、normalization、相機與控制參數。

可先交接共用場景、Remote、收集工具與已驗證的 CPU 契約。實體相關程式供後續開發與受控現場驗證；先完成現場基礎核對與 dry run，再做限定路徑實驗。外部 OpenVLA、三種執行環境、資料與模型另行管理；patch 測試需有固定 revision 的本機 OpenVLA Git source。

## 你接下來要驗收的順序

1. **Remote／Unity**：依 remote_readiness.md 啟動原本指令，實際確認直播、pause／resume／reset、三條正式配對、state／complete event 與 trial ID。
2. **MultiView 視覺品質**：打開 [三物件五視角驗收頁](../outputs/test/data/acceptance_gallery_20261004/index.html)，逐一確認五視角畫面、動作、夾爪與落點，下載人工紀錄。
3. **資料抽查**：必要時依 command_reference.md 重跑 H5 validator；自動檢查不能代替影像內容判讀。
4. **OpenVLA**：在乾淨的外部 checkout 套用目前 patch，重新做短訓練、完整模型匯出、server inference 與模擬 rollout。2026-10-04 的真實一更新訓練早於最新 atomic export 補強。
5. **現場同步**：先驗 real-to-sim 唯讀同步、夾爪 IO 語意／更新頻率與 dry run，再驗目前限定 obj=1、dest=2 的 sim-to-real 路徑。

## 驗收邊界

| 項目 | 現況 |
| --- | --- |
| Remote Python protocol 與 30 分鐘模擬 | 已通過 |
| Unity 實際畫面與事件處理 | 待人工驗收 |
| 三物件 MultiView H5 結構與雜湊 | 已通過 |
| 五視角視覺品質 | 待人工驗收 |
| 更新後 OpenVLA patch 的純 CPU 契約 | 已通過 |
| 更新後 patch 的真實短訓練／匯出 | 待外部 OpenVLA 環境驗收 |
| real-to-sim／sim-to-real 實體路徑 | 待現場驗收 |

## 交接注意

- 使用本分支最新已提交版本；git status --short 應能清楚辨認現場額外修改。450661f 只是修正起點，切回該版本會失去本輪修正。不要用 git reset --hard 或 git clean 清除尚未保存的工作。
- seed 42 目前是固定 canonical reset 的 metadata，不代表隨機場景覆蓋。
- Remote 長測只涵蓋 0→0、1→1、2→2；其餘六種配對尚未納入 readiness 結論。
- 本輪沒有啟動 ROS 或操作實體機器人。
- 需要完整、較耗時的測試順序時才看 [full_test_commands.md](full_test_commands.md)；平常操作以 [command_reference.md](command_reference.md) 為準。
