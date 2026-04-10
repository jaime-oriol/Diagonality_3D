"""Render Kane goal 5 vision video. Max quality: 50fps, smoothing=7, dpi=200."""
import sys; sys.path.insert(0, ".")
import pandas as pd, matplotlib, time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.orientation import compute_orientations, add_dynamics
from src.viz.vision_plot import plot_vision_frame, BG

match = "Bayern_Hamburg"
GOAL_F, FPS = 3585218, 50
start_f, end_f = GOAL_F - 20*FPS, GOAL_F + 15*FPS

print(f"Video: {(end_f-start_f)/FPS:.0f}s @ 50fps native (no frame skip)")
print("Loading skeleton...")
skel = pd.read_parquet(f"cache/{match}/skeleton.parquet")
ss = skel[skel["frame_number"].between(start_f - 5, end_f + 5)]; del skel
print(f"  {len(ss):,} skeleton rows")

print("Computing orientations...")
ori = compute_orientations(ss, smooth=True)
dyn = add_dynamics(ori); del ss, ori
print(f"  {len(dyn):,} orientation rows")

ball = pd.read_parquet(f"cache/{match}/ball.parquet")
vframes = sorted(f for f in dyn["frame_number"].unique() if start_f <= f <= end_f)
print(f"  {len(vframes)} render frames (every frame, no skip)")

fig, ax = plt.subplots(figsize=(16, 10.4)); fig.set_facecolor(BG)

def render(i):
    ax.clear()
    f = vframes[i]
    fo = dyn[dyn["frame_number"] == f]
    bf = ball[ball["frame_number"] == f]
    bx = bf.iloc[0]["x"] if len(bf) > 0 else None
    by = bf.iloc[0]["y"] if len(bf) > 0 else None

    plot_vision_frame(
        fo, focus_team=1, focus_jersey=9,  # Kane #9
        ball_x=bx, ball_y=by,
        att_team=1, smoothing=7.0,  # max resolution grid (105*7=735 x 68*7=476)
        ax=ax,
    )

print("Rendering (this will take a while — max quality)...")
t0 = time.time()
ani = FuncAnimation(fig, render, frames=len(vframes), blit=False, interval=20)
ani.save("test/vision_kane_goal5.mp4", fps=50, dpi=200,
         extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                     "-crf", "18"])  # near-lossless quality
plt.close()
print(f"Saved: test/vision_kane_goal5.mp4 ({len(vframes)} frames, {time.time()-t0:.0f}s)")
