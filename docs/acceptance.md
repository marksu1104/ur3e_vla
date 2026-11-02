# 驗收與目前狀態

最後更新：2026-11-03。分支為 **refactor/unified-runtime**，本輪修正起點為 **450661f**，現行提交以 `git log -1 --oneline` 為準。本分支保存已通過 CPU 與資料檢查的進度，不代表全專案或實體抓取已驗收完成；尚未合併 main。

## 自動驗證已完成

- 2026-10-05 基準為 134 tests、0 skips；本輪另在既有 test_vla_client.py 補上 8 個實體 VLA handler 的 CPU 回歸測試，最新完整結果見下方報告。
- 目前 Remote 來源完成第二次 30 分鐘 headless 長測：正式配對 0→0、1→1、2→2 各兩次成功，pause／resume／reset、慢 client、無 viewer 與重連均通過，共解碼 43,438 幀。
- Remote 啟動參數、HTTP／WebSocket 路徑、canonical 場景、相機、三個物件與目的地編號、軌跡及成功判準未改。
- spoon、red_mug、bowl 三份實際 MultiView H5 通過只讀驗證，檢查前後 SHA 相同；single、Multi-env 與五視角 writer 共用嚴格 RGB uint8 規則。
- 同步 command reservation 的競態已修正；sim-to-real 只接受目前已實作的 reset，pause／resume 會回 HTTP 409。
- OpenVLA patch 會先在 sibling staging 產生並驗證完整模型，再發布到全新的 final path；失敗不會混合舊權重與新 statistics。

歷史自動報告：[2026-10-05 preflight](../outputs/test/preflight/cleanup_20261005/report.json)、[2026-10-30 CPU 與 H5 前檢](../outputs/test/preflight/quick_prep_20261030/report.json)。提交前重新執行 142 個 CPU 測試、三份 MultiView H5 唯讀驗證及 Git 差異檢查；這些本機歷史產物不隨 Git clone 提供。Remote 的來源比對、原指令與長測數字見 [Remote 實驗前確認](remote_readiness.md)。

## 2026-10-30 可先完成的準備

- 依使用者 10 月 14 日回報修正實體文件：夾爪已安裝，VLA 曾實測但難以完成抓取；不是沒有夾爪。
- 推論失敗時，實體 VLA handler 清空上一筆平滑速度並走既有 zero-command 路徑，保留夾爪狀態；下一筆有效回應可恢復。
- 夾爪 SetIO 回傳拒絕或 future 失敗不再記為成功，匹配的請求可重試；較舊失敗不清除新的相反命令快取。
- 動作增益、頻率、Remote 指令、場景、MultiView 和同步控制流程均未改。
- 上述只做 CPU 測試；相機／action 過期停止、實體抓取回授與完成判準仍待補齊，現場效果未驗證。

## 已知未修缺口與交接範圍

這是進度 checkpoint，不是實體抓取 release。下列缺口來自 2026-10-30 的靜態審查與純 CPU 重現，沒有執行實體控制：

- 實體 VLA 尚未依相機／action 新鮮度停止；推論晚於 duration 返回後，仍可能先套用 action 或發出夾爪命令。一般例外退出的零速收尾也需補齊。
- 同步使用的 RealArmCommander 尚未檢查 SetIO 的 success=False；DI 回授缺少接收時間。10 月 30 日只修了 real_vla_servo.py 的對應錯誤分支，沒有修到同步路徑。
- 同步 complete 主要代表軌跡走完，沒有真實物件抓起、途中掉落與落點判準。模擬 VLA 的成功退出代表通路運作，夾爪 latch 也不適合直接評估完整釋放。
- TFDS parser 仍直接轉換影像／state／action dtype，未完整拒絕 NaN、非二元夾爪等輸入；正式 build 前必須先跑 H5 validator，後續應補 builder 本身的驗證。
- MultiView 指五視角收集；目前 TFDS 只有 main／wrist，而固定 OpenVLA 訓練輸入是主視角與文字，未使用五視角或 proprioception。
- 本機 mugs_500 舊 H5 實際只有 3 episodes、242 steps，且缺新版必要 metadata；檔名或收集指令範例不代表已建立正式訓練集。使用者實機使用的外部模型仍需對照資料、normalization、相機與控制參數。

可先交接共用場景、Remote、收集工具與已驗證的 CPU 契約。實體相關程式供後續開發與受控現場驗證；先補上述控制／回授缺口，再做限定路徑實驗。外部 OpenVLA、三種執行環境、資料與模型另行管理；patch 測試需有固定 revision 的本機 OpenVLA Git source。

## 你接下來要驗收的順序

1. **Remote／Unity**：依 remote_readiness.md 啟動原本指令，實際確認直播、pause／resume／reset、三條正式配對、state／complete event 與 trial ID。
2. **MultiView 視覺品質**：打開 [三物件五視角驗收頁](../outputs/test/data/acceptance_gallery_20261004/index.html)，逐一確認五視角畫面、動作、夾爪與落點，下載人工紀錄。
3. **資料抽查**：必要時依 command_reference.md 重跑 H5 validator；自動檢查不能代替影像內容判讀。
4. **OpenVLA**：在乾淨的外部 checkout 套用目前 patch，重新做短訓練、完整模型匯出、server inference 與模擬 rollout。2026-10-04 的真實一更新訓練早於最新 atomic export 補強。
5. **現場同步**：先補上述控制與回授缺口，再做 real-to-sim 唯讀同步及目前限定 obj=1、dest=2 的 sim-to-real 路徑。

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
