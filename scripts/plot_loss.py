#!/usr/bin/env python3
"""plot_loss.py — render the training-loss curve for the paper.

Uses the logged per-epoch MSE (epsilon-prediction) loss from the 9-channel
(T1-conditioned) run, and marks the ~0.006 plateau that every 8-channel
(mask + atlas only) configuration hit — the "informational floor".

Writes training_loss_curve.png if matplotlib is available, otherwise a
dependency-free training_loss_curve.svg (vector, ideal for a paper).
"""

# per-epoch training loss, 9-channel (FLAIR + T1 + mask + 6 atlas) run 52090423
LOSS_9CH = [
    0.0266, 0.0100, 0.0081, 0.0074, 0.0070, 0.0074, 0.0069, 0.0064, 0.0062, 0.0062,
    0.0061, 0.0059, 0.0061, 0.0057, 0.0060, 0.0057, 0.0061, 0.0056, 0.0056, 0.0054,
    0.0061, 0.0054, 0.0054, 0.0055, 0.0055, 0.0053, 0.0053, 0.0054, 0.0051, 0.0054,
    0.0050, 0.0055, 0.0053, 0.0049, 0.0051, 0.0051, 0.0051, 0.0050, 0.0050, 0.0051,
    0.0048, 0.0048, 0.0048, 0.0048, 0.0048, 0.0050, 0.0047, 0.0047, 0.0047, 0.0046,
    0.0048, 0.0047, 0.0049, 0.0046, 0.0046, 0.0044, 0.0044, 0.0046, 0.0046, 0.0046,
    0.0045, 0.0046, 0.0045, 0.0043, 0.0046, 0.0045, 0.0045, 0.0042, 0.0044, 0.0045,
    0.0043, 0.0042, 0.0047, 0.0042, 0.0042, 0.0043, 0.0043, 0.0042, 0.0044, 0.0042,
]
FLOOR_8CH = 0.006     # plateau reached by every mask+atlas-only (8-channel) configuration
YMAX      = 0.012

def with_matplotlib():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(LOSS_9CH)), LOSS_9CH, color="#1f77b4", lw=2, label="9-channel (+ patient T1)")
    ax.axhline(FLOOR_8CH, color="#d62728", ls="--", lw=1.5,
               label="8-channel floor (mask + atlas only) ≈ 0.006")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss (MSE on predicted noise)")
    ax.set_title("Adding the patient T1 lowers the plateau ~0.006 → ~0.0042")
    ax.set_ylim(0, YMAX); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("training_loss_curve.png", dpi=130)
    print("saved training_loss_curve.png")

def with_svg():
    W, H, L, R, T, B = 820, 500, 70, 30, 50, 60      # canvas + margins
    pw, ph = W - L - R, H - T - B
    n = len(LOSS_9CH)
    def X(e): return L + e / (n - 1) * pw
    def Y(v): return T + (1 - v / YMAX) * ph
    pts = " ".join(f"{X(e):.1f},{Y(v):.1f}" for e, v in enumerate(LOSS_9CH))
    yticks = [0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012]
    xticks = [0, 20, 40, 60, 79]
    grid = "".join(f'<line x1="{L}" y1="{Y(t):.1f}" x2="{L+pw}" y2="{Y(t):.1f}" stroke="#e5e5e5"/>'
                   f'<text x="{L-8}" y="{Y(t)+4:.1f}" text-anchor="end" font-size="12" fill="#555">{t:.3f}</text>'
                   for t in yticks)
    grid += "".join(f'<text x="{X(t):.1f}" y="{T+ph+20:.1f}" text-anchor="middle" font-size="12" fill="#555">{t}</text>'
                    for t in xticks)
    floor_y = Y(FLOOR_8CH)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">
<rect width="{W}" height="{H}" fill="white"/>
{grid}
<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="#333"/>
<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="#333"/>
<line x1="{L}" y1="{floor_y:.1f}" x2="{L+pw}" y2="{floor_y:.1f}" stroke="#d62728" stroke-width="1.5" stroke-dasharray="6 4"/>
<text x="{L+pw-6}" y="{floor_y-6:.1f}" text-anchor="end" font-size="12" fill="#d62728">8-channel floor (mask + atlas) ≈ 0.006</text>
<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="2"/>
<text x="{L+pw-6}" y="{Y(0.0042)+16:.1f}" text-anchor="end" font-size="12" fill="#1f77b4">9-channel (+ patient T1) → ~0.0042</text>
<text x="{W/2:.0f}" y="24" text-anchor="middle" font-size="15" fill="#222">Training loss: patient T1 breaks the informational floor</text>
<text x="{W/2:.0f}" y="{H-14}" text-anchor="middle" font-size="13" fill="#333">epoch</text>
<text x="18" y="{T+ph/2:.0f}" text-anchor="middle" font-size="13" fill="#333" transform="rotate(-90 18 {T+ph/2:.0f})">training loss (MSE on predicted noise)</text>
</svg>'''
    with open("training_loss_curve.svg", "w", encoding="utf-8") as f:
        f.write(svg)
    print("saved training_loss_curve.svg")

if __name__ == "__main__":
    try:
        with_matplotlib()
    except ImportError:
        with_svg()
