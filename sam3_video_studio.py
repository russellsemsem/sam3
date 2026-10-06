"""SAM 3 Video Studio: click objects in a video, track them with SAM 3 on Apple Silicon,
then blur, recolor, replace or remove the background.
    python sam3_video_studio.py [--max-seconds 10] [--work-size 1024] [--dtype float32|bfloat16]
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # run any op MPS lacks on the CPU instead of crashing

import argparse, re, subprocess, sys, tempfile, time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np
import torch
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--max-seconds", type=float, default=3, help="load only the first N seconds (0 = all)")
ap.add_argument("--work-size", type=int, default=1024, help="longest side of frames used for clicking/preview")
ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
ARGS = ap.parse_args()

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = getattr(torch, ARGS.dtype)
print(f"Loading SAM 3 tracker on {DEVICE} ({ARGS.dtype}). First run downloads a few GB of weights.")
try:
    MODEL = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", dtype=DTYPE).to(DEVICE).eval()
    PROC = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
except OSError as err:
    sys.exit(f"{err}\n\nRun `hf auth login` with a token from the account that was granted facebook/sam3.")
MODEL.requires_grad_(False)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_enc = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
H264 = (["-c:v", "libx264", "-crf", "18", "-preset", "medium"] if "libx264" in _enc
        else ["-c:v", "h264_videotoolbox", "-b:v", "20M"])
OUT_DIR = Path("exports").resolve()
OUT_DIR.mkdir(exist_ok=True)
COLORS = [(52, 120, 246), (255, 149, 0), (52, 199, 89), (255, 59, 48), (175, 82, 222), (255, 204, 0)]
FX = dict(mode="Blur", color="#00b140", image=None, strength=40, grow=0, soften=2, invert=False,
          highlight=False, hl_opacity=40)


class LazyFrames(Mapping):
    """Feeds frames to SAM 3 on demand instead of pre-processing the whole clip (saves ~12 MB RAM per frame)."""
    def __init__(self, frames):
        self.frames, self.last = frames, (None, None)
    def __len__(self):
        return len(self.frames)
    def __iter__(self):
        return iter(range(len(self.frames)))
    def __getitem__(self, i):
        if self.last[0] != i:
            px = PROC.video_processor(videos=[self.frames[i][None]], return_tensors="pt").pixel_values_videos[0][0]
            self.last = (i, px.to(DTYPE))
        return self.last[1]


class State:
    def __init__(self):
        self.path, self.frames, self.fps, self.size = None, [], 30.0, (1, 1)
        self.session, self.tracked = None, False
        self.clicks, self.history = {}, []  # clicks: obj -> {frame: [(x, y, label)]}
        self.logits = {}                    # frame -> {obj: low-res mask logits}

S = State()


def fmt(sec):
    return f"{sec / 60:.0f} min" if sec >= 90 else f"{sec:.0f} s"


# ---------- model ----------
def load_video(path):
    global S
    S = State()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    if not path:
        return None, gr.update(maximum=1, value=0), "Load a video to start."
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    S.fps = fps if 1 <= fps <= 240 else 30.0
    limit = int(ARGS.max_seconds * S.fps) if ARGS.max_seconds > 0 else 10**9
    while len(S.frames) < limit:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        S.size, s = (w, h), min(1.0, ARGS.work_size / max(h, w))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        S.frames.append(rgb if s == 1 else cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA))
    cap.release()
    if not S.frames:
        raise gr.Error("Couldn't read any frames from that file. Try an MP4 or MOV.")
    S.path = path
    S.session = PROC.init_video_session(video=None, inference_device=DEVICE, inference_state_device=DEVICE,
                                        video_storage_device="cpu", processing_device="cpu", dtype=DTYPE)
    S.session.processed_frames = LazyFrames(S.frames)
    S.session.video_height, S.session.video_width = S.frames[0].shape[:2]
    n = len(S.frames)
    return (render(0, "Selection"), gr.update(maximum=max(n - 1, 1), value=0),
            f"Loaded {n} frames ({n / S.fps:.1f} s, {S.size[0]}×{S.size[1]}). Click the object you want to keep.")


def store(out):
    lr = out.pred_masks[:, 0].float().cpu().numpy().astype(np.float16)
    slot = S.logits.setdefault(int(out.frame_idx), {})
    for i, oid in enumerate(S.session.obj_ids):
        slot[int(oid)] = lr[i]


def send_points(f, oids):
    PROC.add_inputs_to_inference_session(
        inference_session=S.session, frame_idx=f, obj_ids=oids,
        input_points=[[[[x, y] for x, y, _ in S.clicks[o][f]] for o in oids]],
        input_labels=[[[lab for _, _, lab in S.clicks[o][f]] for o in oids]])
    with torch.inference_mode():
        store(MODEL(inference_session=S.session, frame_idx=f))


def resync():
    """Rebuild SAM 3's memory from the remaining clicks so undo is exact."""
    S.clicks = {o: {f: p for f, p in per.items() if p} for o, per in S.clicks.items()}
    S.clicks = {o: per for o, per in S.clicks.items() if per}
    for slot in S.logits.values():
        for o in [o for o in slot if o not in S.clicks]:
            del slot[o]
    if not S.clicks:
        S.tracked = False
    S.session.reset_tracking_data()
    for f in sorted({f for per in S.clicks.values() for f in per}):
        send_points(f, [o for o, per in S.clicks.items() if f in per])


def on_click(f, obj, mode, view, evt: gr.SelectData):
    if S.session is None:
        raise gr.Error("Load a video first.")
    f, oid = int(f), int(obj)
    x, y = evt.index[0], evt.index[1]
    S.clicks.setdefault(oid, {}).setdefault(f, []).append((float(x), float(y), 1 if mode.startswith("Add") else 0))
    S.history.append((oid, f))
    t = time.time()
    send_points(f, [oid])
    hint = " Track again to apply this to the whole clip." if S.tracked else ""
    return render(f, view), f"Object {oid} updated on frame {f} ({time.time() - t:.1f} s).{hint}"


def on_undo(f, view):
    if not S.history:
        raise gr.Error("Nothing to undo.")
    oid, fr = S.history.pop()
    S.clicks[oid][fr].pop()
    resync()
    return render(f, view), f"Removed the last click (object {oid}, frame {fr})."


def on_clear(obj, f, view):
    oid = int(obj)
    S.clicks.pop(oid, None)
    S.history = [h for h in S.history if h[0] != oid]
    if S.session is not None:
        resync()
    return render(f, view), f"Cleared object {oid}."


def on_track(view):
    prompted = sorted({f for per in S.clicks.values() for f in per})
    if S.session is None or not prompted:
        raise gr.Error("Click on an object first.")
    start, n = prompted[0], len(S.frames)
    total = n - start + (start + 1 if start else 0)
    S.tracked, done, t0, last = False, 0, time.time(), 0.0
    for reverse in ([False, True] if start else [False]):  # forward to the end, then back to frame 0
        it = MODEL.propagate_in_video_iterator(S.session, start_frame_idx=start, reverse=reverse)
        while True:
            with torch.inference_mode():
                out = next(it, None)
                if out is not None:
                    store(out)
            if out is None:
                break
            done += 1
            if time.time() - last > 0.7:
                last = time.time()
                rate = (last - t0) / done
                yield (render(out.frame_idx, view), gr.update(value=out.frame_idx),
                       f"Tracking: {done}/{total} frames, {rate:.1f} s per frame, about {fmt(rate * (total - done))} left.")
    S.tracked = True
    yield (render(start, view), gr.update(value=start),
           f"Tracked {n} frames in {fmt(time.time() - t0)}. Scrub to check; click any frame to fix a mistake, then track again.")


# ---------- effects ----------
def hex_rgb(v):
    m = re.search(r"#?([0-9a-fA-F]{6})", str(v or ""))
    if m:
        return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    nums = re.findall(r"[\d.]+", str(v or ""))
    return tuple(int(float(c)) for c in nums[:3]) if len(nums) >= 3 else (0, 177, 64)


def masks_at(f, w, h):
    return {o: cv2.resize(lr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0
            for o, lr in dict(S.logits.get(f, {})).items()}


def alpha(f, w, h, scale):
    """0..1 matte of all selected objects (inverted if requested). Edge settings are in source pixels."""
    a = np.zeros((h, w), np.uint8)
    for m in masks_at(f, w, h).values():
        a[m] = 255
    g = round(FX["grow"] * scale)
    if g:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(g) + 1, 2 * abs(g) + 1))
        a = cv2.dilate(a, k) if g > 0 else cv2.erode(a, k)
    a = a.astype(np.float32) / 255
    if FX["soften"] * scale > 0.3:
        a = cv2.GaussianBlur(a, (0, 0), FX["soften"] * scale)
    return 1 - a if FX["invert"] else a


def cover(img, w, h):
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (max(w, round(iw * s)), max(h, round(ih * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y:y + h, x:x + w]


def soft_blur(rgb, weight, sigma):
    """Blur that only gathers color from the region being blurred, so the subject doesn't smear into it."""
    h, w = rgb.shape[:2]
    d = max(1, int(sigma // 6))
    sw, sh = max(1, w // d), max(1, h // d)
    img = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    wt = cv2.resize(weight, (sw, sh), interpolation=cv2.INTER_AREA)
    s = max(sigma / d, 0.5)
    num = cv2.GaussianBlur(img * wt[..., None], (0, 0), s)
    den = cv2.GaussianBlur(wt, (0, 0), s)[..., None]
    out = np.where(den > 1e-3, num / np.maximum(den, 1e-3), cv2.GaussianBlur(img, (0, 0), s))
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def composite(rgb, f, scale):
    h, w = rgb.shape[:2]
    a = alpha(f, w, h, scale)
    mode = FX["mode"]
    if mode == "Transparent":
        return np.dstack([rgb, (a * 255 + 0.5).astype(np.uint8)])
    if mode == "Color":
        bg = np.empty_like(rgb)
        bg[:] = hex_rgb(FX["color"])
    elif mode == "Image":
        bg = cover(FX["image"], w, h) if FX["image"] is not None else np.full_like(rgb, 128)
    elif mode == "Black & white":
        bg = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    else:
        bg = soft_blur(rgb, 1 - a, FX["strength"] / 100 * 0.04 * max(h, w))
    a3 = a[..., None]
    return np.clip(rgb * a3 + bg * (1 - a3) + 0.5, 0, 255).astype(np.uint8)


def highlight(out, f, scale):
    """Draw each object's colored tint and outline on a finished frame."""
    if not FX["highlight"]:
        return out
    h, w = out.shape[:2]
    rgb = np.ascontiguousarray(out[..., :3])
    op = FX["hl_opacity"] / 100
    thickness = max(1, round(3 * scale))  # about 3 px at the original resolution
    for o, m in masks_at(f, w, h).items():
        c = COLORS[(o - 1) % len(COLORS)]
        rgb[m] = (rgb[m] * (1 - op) + np.array(c) * op).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cnts, -1, c, thickness, cv2.LINE_AA)
    if out.shape[2] == 4:  # keep the alpha channel for Transparent exports
        out[..., :3] = rgb
        return out
    return rgb


def render(f, view):
    if not S.frames:
        return None
    f = int(np.clip(f, 0, len(S.frames) - 1))
    rgb = S.frames[f]
    h, w = rgb.shape[:2]
    if view == "Result":
        scale = w / S.size[0]
        out = highlight(composite(rgb, f, scale), f, scale)
        if out.shape[2] == 4:  # show transparency on a checkerboard
            yy, xx = np.mgrid[:h, :w]
            board = np.where(((yy // 16 + xx // 16) % 2 == 0)[..., None], 190, 250)
            a = out[..., 3:4] / 255.0
            out = (out[..., :3] * a + board * (1 - a)).astype(np.uint8)
        return out
    out = rgb.copy()
    for o, m in masks_at(f, w, h).items():
        c = COLORS[(o - 1) % len(COLORS)]
        out[m] = (out[m] * 0.5 + np.array(c) * 0.5).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, c, 2, cv2.LINE_AA)
    r = max(5, w // 150)
    for per in S.clicks.values():
        for x, y, lab in per.get(f, []):
            cv2.circle(out, (int(x), int(y)), r + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), r, (34, 197, 94) if lab else (239, 68, 68), -1, cv2.LINE_AA)
    return out


def on_fx(mode, color, image, strength, grow, soften, invert, highlight_on, hl_opacity, f):
    if image is not None:
        image = np.dstack([image] * 3) if image.ndim == 2 else image[..., :3]
    FX.update(mode=mode, color=color, image=image, strength=strength, grow=grow, soften=soften,
              invert=invert, highlight=highlight_on, hl_opacity=hl_opacity)
    return (render(f, "Result"), gr.update(value="Result"), gr.update(visible=mode == "Color"),
            gr.update(visible=mode == "Image"), gr.update(visible=mode == "Blur"),
            gr.update(visible=bool(highlight_on)))


# ---------- export ----------
class Writer:
    """Pipes raw frames into ffmpeg (bundled with imageio-ffmpeg)."""
    def __init__(self, path, w, h, rgba=False, audio=False):
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba" if rgba else "rgb24",
               "-s", f"{w}x{h}", "-r", str(S.fps), "-i", "-"]
        if audio:  # keep the clip's soundtrack, trimmed to the exported length
            cmd += ["-t", f"{len(S.frames) / S.fps:.3f}", "-i", S.path, "-map", "0:v:0", "-map", "1:a:0?", "-shortest"]
        if rgba:
            cmd += ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-c:a", "pcm_s16le"]
        else:
            cmd += H264 + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "192k"]
        self.err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd + [str(path)], stdin=subprocess.PIPE, stderr=self.err)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            self.err.seek(0)
            raise gr.Error("ffmpeg failed: " + self.err.read().decode(errors="replace")[-400:])


def on_export(want_matte, progress=gr.Progress()):
    if not S.tracked:
        raise gr.Error("Track the selection through the video first (step 3).")
    if FX["mode"] == "Image" and FX["image"] is None:
        raise gr.Error("Choose a background image first.")
    w, h = S.size[0] // 2 * 2, S.size[1] // 2 * 2  # H.264 needs even sizes
    rgba = FX["mode"] == "Transparent"
    name = f"{Path(S.path).stem}-{datetime.now():%Y%m%d-%H%M%S}"
    main_path, matte_path = OUT_DIR / f"{name}.{'mov' if rgba else 'mp4'}", OUT_DIR / f"{name}-matte.mp4"
    main = Writer(main_path, w, h, rgba=rgba, audio=True)
    matte = Writer(matte_path, w, h) if want_matte else None
    cap, n = cv2.VideoCapture(S.path), len(S.frames)
    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[:h, :w])
        main.write(highlight(composite(rgb, i, 1.0), i, 1.0))  # full resolution, masks upscaled from SAM's output
        if matte:
            m = (alpha(i, w, h, 1.0) * 255 + 0.5).astype(np.uint8)
            matte.write(np.dstack([m, m, m]))
        progress((i + 1) / n, desc=f"Exporting frame {i + 1} of {n}")
    cap.release()
    main.close()
    files = [str(main_path)]
    if matte:
        matte.close()
        files.append(str(matte_path))
    note = ("ProRes 4444 with transparency; open it in Final Cut, Premiere, Resolve or After Effects."
            if rgba else "Saved.")
    return (None if rgba else str(main_path)), files, f"{note} Files are in {OUT_DIR}."


# ---------- UI ----------
with gr.Blocks(title="SAM 3 Video Studio") as demo:
    gr.Markdown("## SAM 3 Video Studio\nClick objects in a clip, track them with SAM 3 on your Mac, "
                "then swap or restyle the background.")
    with gr.Row():
        with gr.Column(scale=3):
            preview = gr.Image(label="Click to select (green dot adds, red dot removes)", interactive=False)
            frame = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="Frame")
            view = gr.Radio(choices=["Selection", "Result"], value="Selection", label="Preview")
            status = gr.Markdown("Load a video to start.")
        with gr.Column(scale=2):
            video = gr.Video(label="1. Load a clip", sources=["upload"])
            obj = gr.Radio(choices=["1", "2", "3", "4", "5", "6"], value="1", label="2. Object (each has its own color)")
            mode = gr.Radio(choices=["Add to object", "Remove from object"], value="Add to object", label="Click mode")
            with gr.Row():
                undo_btn = gr.Button("Undo last click")
                clear_btn = gr.Button("Clear object")
            with gr.Row():
                track_btn = gr.Button("3. Track through video", variant="primary")
                stop_btn = gr.Button("Stop")
            fx_mode = gr.Dropdown(choices=["Blur", "Color", "Image", "Black & white", "Transparent"],
                                  value="Blur", label="4. Background effect")
            fx_color = gr.ColorPicker(value="#00b140", label="Color", visible=False)
            fx_image = gr.Image(label="Background image", type="numpy", visible=False)
            fx_strength = gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Blur strength")
            fx_grow = gr.Slider(minimum=-20, maximum=20, value=0, step=1, label="Grow or shrink the edge (px)")
            fx_soft = gr.Slider(minimum=0, maximum=20, value=2, step=1, label="Edge softness (px)")
            fx_invert = gr.Checkbox(label="Apply the effect to the selected objects instead")
            fx_highlight = gr.Checkbox(label="Show the colored highlight on objects in the result")
            fx_hl_opacity = gr.Slider(minimum=0, maximum=100, value=40, step=1,
                                      label="Highlight opacity (0 = outline only)", visible=False)
            matte_chk = gr.Checkbox(label="Also export a black-and-white matte")
            export_btn = gr.Button("5. Export video", variant="primary")
            result = gr.Video(label="Exported video", interactive=False)
            files = gr.File(label="Files", file_count="multiple")

    M = dict(concurrency_id="model", concurrency_limit=1)
    video.change(load_video, video, [preview, frame, status], **M)
    preview.select(on_click, [frame, obj, mode, view], [preview, status], **M)
    frame.input(render, [frame, view], preview, trigger_mode="always_last")
    view.input(render, [frame, view], preview)
    undo_btn.click(on_undo, [frame, view], [preview, status], **M)
    clear_btn.click(on_clear, [obj, frame, view], [preview, status], **M)
    run = track_btn.click(on_track, view, [preview, frame, status], **M)
    stop_btn.click(lambda: "Stopped. Tracked frames keep their masks; track again to finish.", None, status, cancels=[run])
    fx_in = [fx_mode, fx_color, fx_image, fx_strength, fx_grow, fx_soft, fx_invert,
             fx_highlight, fx_hl_opacity, frame]
    for c in fx_in[:-1]:
        c.change(on_fx, fx_in, [preview, view, fx_color, fx_image, fx_strength, fx_hl_opacity])
    export_btn.click(on_export, matte_chk, [result, files, status], **M)

demo.queue().launch(inbrowser=True, allowed_paths=[str(OUT_DIR)])
