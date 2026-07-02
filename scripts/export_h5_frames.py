"""
export_h5_frames.py
─────────────────────────────────────────────────────────────
把 h5 demo 的影像匯出成：
  1. images/  每幀一張 jpg，檔名含 frame index
  2. video/   每個 demo 一支 mp4（需要 opencv）

執行範例：
  # 匯出所有 demo 的圖片
  python scripts/export_h5_frames.py --h5 ./outputs/h5/mugs_5/red_mug/demos.h5

  # 只匯出 demo_0，每 5 幀一張
  python scripts/export_h5_frames.py --h5 ./outputs/h5/mugs_5/red_mug/demos.h5 --demos demo_0 --every-n 5

  # 同時輸出影片
  python scripts/export_h5_frames.py --h5 ./outputs/h5/mugs_5/red_mug/demos.h5 --video

  # 指定輸出目錄
  python scripts/export_h5_frames.py --h5 ./outputs/h5/mugs_5/red_mug/demos.h5 --out-dir ./my_export
"""

import argparse
import base64
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── 文字 overlay（在圖片上印 frame index 和 gripper 狀態）────────
def draw_overlay(pil_img: Image.Image, frame_idx: int,
                 gripper_raw: float, demo_key: str) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)

    gripper_str = f"gripper_raw: {gripper_raw:.3f}"
    text = f"{demo_key}  frame={frame_idx:04d}  {gripper_str}"

    # 半透明黑底
    w, h = img.size
    draw.rectangle([0, 0, w, 24], fill=(0, 0, 0, 180))
    draw.text((4, 4), text, fill=(255, 255, 0))

    return img


# ── 匯出圖片 ─────────────────────────────────────────────────────
def export_images(h5_path: str, demo_keys: list,
                  every_n: int, out_dir: Path):
    with h5py.File(h5_path, "r") as f:
        if not demo_keys:
            demo_keys = sorted(f["data"].keys())

        for demo_key in demo_keys:
            demo      = f[f"data/{demo_key}"]
            images    = demo["image"][:]        # (T, H, W, 3)
            robot_state = demo["robot_state"][:]  # (T, 15)
            num_steps = demo.attrs["num_steps"]
            instruction = demo.attrs.get("instruction", "")

            save_dir = out_dir / "images" / demo_key
            save_dir.mkdir(parents=True, exist_ok=True)

            exported = []
            for i in range(0, num_steps, every_n):
                gripper_raw = float(robot_state[i, 14])
                pil_img = Image.fromarray(images[i])
                pil_img = draw_overlay(pil_img, i, gripper_raw, demo_key)

                fname = save_dir / f"frame_{i:04d}.jpg"
                pil_img.save(fname, quality=90)
                exported.append({
                    "frame_idx":   i,
                    "gripper_raw": round(gripper_raw, 4),
                    "file":        str(fname)
                })

            # 存一份 index json，方便對照標注
            index_path = save_dir / "frame_index.json"
            index_path.write_text(
                json.dumps({
                    "demo":        demo_key,
                    "instruction": instruction,
                    "num_steps":   int(num_steps),
                    "every_n":     every_n,
                    "frames":      exported
                }, indent=2),
                encoding="utf-8"
            )

            print(f"  [{demo_key}] {len(exported)} frames → {save_dir}")
            print(f"           index → {index_path}")


# ── 匯出影片 ─────────────────────────────────────────────────────
def export_videos(h5_path: str, demo_keys: list, out_dir: Path, fps: int = 10):
    try:
        import cv2
    except ImportError:
        print("[ERROR] 需要 opencv：pip install opencv-python")
        return

    with h5py.File(h5_path, "r") as f:
        if not demo_keys:
            demo_keys = sorted(f["data"].keys())

        for demo_key in demo_keys:
            demo        = f[f"data/{demo_key}"]
            images      = demo["image"][:]
            robot_state = demo["robot_state"][:]
            num_steps   = demo.attrs["num_steps"]

            save_dir = out_dir / "video"
            save_dir.mkdir(parents=True, exist_ok=True)

            h, w = images[0].shape[:2]
            out_path = save_dir / f"{demo_key}.mp4"
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (w, h)
            )

            for i in range(num_steps):
                gripper_raw = float(robot_state[i, 14])
                pil_img = Image.fromarray(images[i])
                pil_img = draw_overlay(pil_img, i, gripper_raw, demo_key)

                # PIL RGB → cv2 BGR
                frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            writer.release()
            print(f"  [{demo_key}] {num_steps} frames → {out_path}")


# ── main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",      required=True, help="demos.h5 路徑")
    parser.add_argument("--demos",   nargs="+",     help="指定 demo key，不填就全部")
    parser.add_argument("--every-n", type=int, default=1,
                        help="每 N 幀輸出一張圖（預設 1 = 全部）")
    parser.add_argument("--video",   action="store_true", help="同時輸出 mp4 影片")
    parser.add_argument("--fps",     type=int, default=10, help="影片 fps（預設 10）")
    parser.add_argument("--out-dir", default="./export",  help="輸出根目錄")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n h5:     {args.h5}")
    print(f" out:    {out_dir}")
    print(f" every:  {args.every_n} frames")
    print(f" video:  {args.video}\n")

    export_images(args.h5, args.demos or [], args.every_n, out_dir)

    if args.video:
        export_videos(args.h5, args.demos or [], out_dir, fps=args.fps)

    print("\n完成。")
    print(f"圖片：{out_dir}/images/<demo_key>/frame_XXXX.jpg")
    print(f"Index：{out_dir}/images/<demo_key>/frame_index.json")
    if args.video:
        print(f"影片：{out_dir}/video/<demo_key>.mp4")
    print("\n看完圖片後，參考 frame_index.json 的 frame_idx 填寫 ground_truth.json 的 skill_segments。")


if __name__ == "__main__":
    main()