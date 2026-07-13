import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Unsafe behavior detection - foto/video/webcam/RTSP")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path ke file .pt hasil training")
    parser.add_argument("--source", type=str, default="0",
                         help="'0'/'1'.. = webcam, path foto, path video, atau URL RTSP")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "image", "stream"],
                         help="Paksa mode kalau auto-detect salah nebak")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran input model (samain sama pas training)")
    parser.add_argument("--device", type=str, default="", help="'cpu', '0' (index GPU), kosongin buat auto")
    parser.add_argument("--save", action="store_true", help="Simpan hasil (nama default 'output')")
    parser.add_argument("--output", type=str, default=None,
                         help="Simpan hasil dengan nama ini (nulis --output aja udah otomatis nyimpen, --save opsional)")
    parser.add_argument("--no-alert", action="store_true", help="Matiin bunyi alert")
    parser.add_argument("--alert-classes", type=str, default="smoking,vaping,Phone",
                         help="Class yang bakal bunyi kalau kedeteksi, pisah pakai koma")
    parser.add_argument("--alert-cooldown", type=float, default=3.0,
                         help="Jeda minimum antar bunyi alert (detik), biar nggak bunyi terus tiap frame")
    parser.add_argument("--debug", action="store_true",
                         help="Print semua class yang kedeteksi tiap frame, buat diagnosis kalau alert nggak bunyi")
    parser.add_argument("--alert-sound", type=str, default=None,
                         help="Path ke file audio (.mp3/.wav) buat alert. Kosongin buat pakai system beep bawaan Windows.")
    return parser.parse_args()


def detect_mode(source_str, forced_mode):
    if forced_mode != "auto":
        return forced_mode
    if source_str.isdigit():
        return "stream"  # webcam
    suffix = Path(source_str).suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    return "stream"  
    
def init_alert_sound(sound_path):
    """Load file audio sekali di awal (bukan tiap alert, biar nggak ada delay)."""
    if not sound_path:
        return
    import pygame
    pygame.mixer.init()
    pygame.mixer.music.load(sound_path)
    print(f"Alert sound dimuat: {sound_path}")

def play_alert(sound_path=None):
    if sound_path:
        import pygame
        pygame.mixer.music.play()
    elif sys.platform == "win32":
        import winsound
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    else:
        print("\a", end="", flush=True)


def mux_alert_audio(video_path, alert_timestamps, duration_sec, alert_sound_path):
    """Tempel bunyi alert ke video hasil save, di detik-detik alert kedeteksi.
    Pakai ffmpeg portable dari imageio-ffmpeg (auto-download sekali pas pip install,
    nggak perlu install ffmpeg manual atau ubah PATH sama sekali)."""
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine
    except ImportError:
        print("[audio] Skip: install dulu 'pip install pydub imageio-ffmpeg' buat nempelin alert ke video.")
        return

    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("[audio] Skip: install dulu 'pip install imageio-ffmpeg' (ffmpeg portable otomatis, nggak perlu setup PATH).")
        return

    AudioSegment.converter = ffmpeg_bin

    import subprocess
    import os

    total_ms = int(duration_sec * 1000) + 500
    track = AudioSegment.silent(duration=total_ms, frame_rate=44100)

    if alert_sound_path:
        beep = AudioSegment.from_file(alert_sound_path)
    else:
        beep = Sine(1000).to_audio_segment(duration=300).apply_gain(-5)

    for t in alert_timestamps:
        track = track.overlay(beep, position=int(t * 1000))

    base, ext = video_path.rsplit(".", 1)
    audio_path = f"{base}_alert_audio.wav"
    final_path = f"{base}_with_audio.{ext}"
    track.export(audio_path, format="wav")

    subprocess.run(
        [ffmpeg_bin, "-y", "-i", video_path, "-i", audio_path,
         "-c:v", "copy", "-c:a", "aac", "-shortest", final_path],
        check=True,
    )
    os.remove(audio_path)
    print(f"[audio] Video dengan alert audio disimpan ke: {final_path}")


def run_image(model, source, args):
    results = model.predict(
        source, conf=args.conf, imgsz=args.imgsz,
        device=args.device if args.device else None, verbose=False,
    )
    annotated = results[0].plot()

    alert_classes = set(c.strip() for c in args.alert_classes.split(","))
    boxes = results[0].boxes
    detected_names = []
    if boxes is None or len(boxes) == 0:
        print("Tidak ada objek terdeteksi (smoking/vaping/Phone) di foto ini.")
    else:
        for box in boxes:
            cls_name = model.names[int(box.cls)]
            detected_names.append(cls_name)
            print(f"Terdeteksi: {cls_name} ({float(box.conf):.2f})")

    if not args.no_alert and any(name in alert_classes for name in detected_names):
        play_alert(args.alert_sound)

    if args.save or args.output:
        out_path = f"{args.output or 'output'}.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"Disimpan ke: {out_path}")

    cv2.imshow("Unsafe Behavior Detection", annotated)
    print("Tekan sembarang tombol di window buat nutup.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_stream(model, source, args):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal buka source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = None
    out_path = None
    if args.save or args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = f"{args.output or 'output'}.mp4"
        writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))
        print(f"Hasil bakal disimpan ke: {out_path}")

    alert_classes = set(c.strip() for c in args.alert_classes.split(","))
    alert_timestamps = []
    last_alert_time = 0.0
    frame_idx = 0

    prev_time = time.time()
    print("Jalan... tekan 'q' di window video buat keluar.")
    if not args.no_alert:
        print(f"Alert aktif buat class: {sorted(alert_classes)} (cooldown {args.alert_cooldown}s)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream/video selesai atau gagal baca frame.")
            break

        results = model.predict(
            frame, conf=args.conf, imgsz=args.imgsz,
            device=args.device if args.device else None, verbose=False,
        )
        annotated = results[0].plot()

        # cek apakah ada class yang perlu di-alert di frame ini
        boxes = results[0].boxes
        detected_names = [model.names[int(b.cls)] for b in boxes] if boxes is not None else []
        triggered = [n for n in detected_names if n in alert_classes]

        if args.debug:
            if detected_names:
                confs = [f"{float(b.conf):.2f}" for b in boxes]
                print(f"[DEBUG] Terdeteksi: {list(zip(detected_names, confs))}")
            else:
                print("[DEBUG] Nggak ada objek terdeteksi frame ini")

        now = time.time()
        if not args.no_alert and triggered and (now - last_alert_time) >= args.alert_cooldown:
            print(f"[ALERT] Terdeteksi: {', '.join(sorted(set(triggered)))}")
            play_alert(args.alert_sound)
            last_alert_time = now
            alert_timestamps.append(frame_idx / fps_in)

        # FPS counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(
            annotated, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        cv2.imshow("Unsafe Behavior Detection", annotated)

        if writer is not None:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if out_path and alert_timestamps and not args.no_alert:
        duration_sec = frame_idx / fps_in
        mux_alert_audio(out_path, alert_timestamps, duration_sec, args.alert_sound)


def main():
    args = parse_args()
    source_str = args.source
    source = int(source_str) if source_str.isdigit() else source_str

    mode = detect_mode(source_str, args.mode)

    print(f"Load model: {args.weights}")
    model = YOLO(args.weights)
    print(f"Mode: {mode} | Source: {source}")

    if not args.no_alert:
        init_alert_sound(args.alert_sound)

    if mode == "image":
        run_image(model, source, args)
    else:
        run_stream(model, source, args)


if __name__ == "__main__":
    main()
