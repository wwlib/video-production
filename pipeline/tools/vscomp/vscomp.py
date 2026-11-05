#!/usr/bin/env python3
"""vscomp - composite VS Code screen plates onto monitors in AI-rendered shots.

Subcommands (run with /workspace/.venv-vscomp/bin/python):
  probe  SHOT            extract ref/mid/last frames + gridded zoom sheets + auto quad candidates
  track  SHOT            track every monitor in configs/shot_XX.json -> tracks/shot_XX.json + debug overlay
  comp   SHOT            composite -> shot_XX/shot_XX_clip_24fps_grain_vscode.mp4
  run    SHOT [SHOT..]   track + comp
  status                 show which shots have configs / are stale vs. newest clip
Options: --force (ignore stale-source guard), --frames N (only first N frames, for quick tests)
"""
import cv2, numpy as np, json, os, sys, glob, subprocess, argparse, hashlib, time

ROOT = os.environ.get('VSCOMP_ROOT', '/workspace/poc/shots')   # VSCOMP_ROOT/VSCOMP_CFGDIR: run another shot tree (e.g. jen_intro) with its own configs
HERE = os.path.dirname(os.path.abspath(__file__))
SCREENS = os.environ.get('SCREENS', os.path.join(os.environ.get('CDV_DATA', os.path.expanduser('~/video-data/code-de-vie')), 'plates'))
_W = os.environ.get('VSCOMP_CFGDIR')
CFG = f'{_W}/configs' if _W else f'{HERE}/configs'; TRK = f'{_W}/tracks' if _W else f'{HERE}/tracks'; DBG = f'{_W}/debug' if _W else f'{HERE}/debug'
FPS = 24
for _d in (CFG, TRK, DBG):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------------------- io
def newest_clip(shot):
    """Newest *_clip_24fps_grain.mp4 (not _vscode) in shot_XX/. Falls back to _old_intro/shot_XX*/ if the
    live folder is missing (e.g. moved away while a re-render is in flight)."""
    pats = [f'{ROOT}/shot_{shot:02d}/*_clip_24fps_grain.mp4']
    c = [p for p in glob.glob(pats[0]) if not p.endswith('_vscode.mp4')]
    if not c:
        c = [p for p in glob.glob(f'{ROOT}/_old_intro/shot_{shot:02d}*/*_clip_24fps_grain.mp4')]
    if not c: raise SystemExit(f'no clip for shot {shot}')
    return max(c, key=os.path.getmtime)

def out_path(src):
    return src[:-4] + '_vscode.mp4'

def fingerprint(p):
    st = os.stat(p)
    h = hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
    return dict(path=p, size=st.st_size, mtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)), md5=h)

def read_frames(p, n=None):
    cap = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = cap.read()
        if not ok or (n and len(fr) >= n): break
        fr.append(f)
    return fr

class Plate:
    def __init__(self, name, offset=0, src_rect=None, stack=None):
        self.stack = stack  # optional list of [y0,y1] row bands (after src_rect) stacked into a taller virtual plate
        p = name if os.path.isabs(name) else f'{SCREENS}/{name}'
        self.video = p.endswith('.mp4'); self.offset = offset
        self.frames = read_frames(p) if self.video else [cv2.imread(p)]
        if self.frames[0] is None: raise SystemExit(f'plate not found: {p}')
        H, W = self.frames[0].shape[:2]
        self.rect = src_rect or [0, 0, W, H]  # x0,y0,x1,y1 in plate px
    def get(self, t):
        if not self.video: f = self.frames[0]
        else: f = self.frames[min(len(self.frames) - 1, max(0, self.offset + t))]
        x0, y0, x1, y1 = self.rect
        f = f[y0:y1, x0:x1]
        if self.stack: f = np.vstack([f[a:b] for a, b in self.stack])
        return f

def load_cfg(shot):
    p = f'{CFG}/shot_{shot:02d}.json'
    if not os.path.exists(p): raise SystemExit(f'no config {p} (run probe and write one)')
    return json.load(open(p))

def check_source(cfg, shot, force):
    src = newest_clip(shot)
    fp = cfg.get('source', {})
    cur = fingerprint(src)
    if fp.get('md5') != cur['md5']:
        msg = (f'shot {shot}: config was authored for {fp.get("path")} (md5 {fp.get("md5")}, {fp.get("mtime")}),\n'
               f'  but newest clip is {src} (md5 {cur["md5"]}, {cur["mtime"]}).\n'
               f'  Re-run probe, re-check corners in the config, then set "source" (or pass --force / run `accept`).')
        if not force: raise SystemExit(msg)
        print('WARNING', msg)
    return src

# ----------------------------------------------------------------------------- geometry
def grow_quad(q, g):
    q = np.asarray(q, np.float64); c = q.mean(0)
    d = q - c; n = np.linalg.norm(d, axis=1, keepdims=True)
    return q + d / n * g * 1.41421

def poly_mask(shape, poly, grow=0):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 255)
    if grow > 0: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return m

def gsmooth(x, sigma):
    """gaussian smooth along axis 0, edge-reflect."""
    if sigma <= 0: return x
    r = int(3 * sigma + 0.5); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    xp = np.concatenate([x[r:0:-1], x, x[-2:-r - 2:-1]], 0) if len(x) > r + 1 else np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode='edge')
    out = np.zeros_like(x, dtype=np.float64)
    for i, w in enumerate(k): out += w * xp[i:i + len(x)]
    return out

def prep_gray(f):
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    return cv2.GaussianBlur(g, (3, 3), 0)

# ----------------------------------------------------------------------------- tracking
def estimate(model, a, b):
    if len(a) < 6: return None, 0
    if model == 'translation':
        d = np.median(b - a, 0); inl = np.linalg.norm(b - a - d, axis=1) < 1.5
        if inl.sum() >= 4: d = np.median((b - a)[inl], 0)
        return np.array([[1, 0, d[0]], [0, 1, d[1]], [0, 0, 1]], np.float64), int(inl.sum())
    if model == 'homography':
        H, inl = cv2.findHomography(a, b, cv2.RANSAC, 1.5, maxIters=4000, confidence=0.999)
    else:
        fn = cv2.estimateAffinePartial2D if model == 'similarity' else cv2.estimateAffine2D
        M, inl = fn(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=4000)
        H = None if M is None else np.vstack([M, [0, 0, 1]])
    if H is None: return None, 0
    return H, int(inl.sum())

def track_monitor(frames, mon, ref):
    tr = mon.get('track', {}); mode = tr.get('mode', 'klt'); model = tr.get('model', 'homography')
    quad = np.asarray(mon['quad'], np.float64); N = len(frames)
    info = dict(mode=mode, model=model)
    if mode == 'static':
        return np.repeat(quad[None], N, 0), info
    if mode == 'keys':   # hand-keyed corners {frame: quad}, linearly interpolated (moving/deforming screens, e.g. a closing lid)
        K = sorted((int(k), np.asarray(v, np.float64)) for k, v in tr['keys'].items())
        ks = np.array([k for k, _ in K], np.float64); Q = np.stack([q for _, q in K])
        t = np.arange(N, dtype=np.float64)
        C = np.stack([[np.interp(t, ks, Q[:, c, d]) for d in range(2)] for c in range(4)])   # 4,2,N
        C = np.transpose(C, (2, 0, 1))
        if tr.get('smooth'): C = gsmooth(C, tr['smooth'])
        info.update(keys=[int(k) for k in ks], frames=N)
        return C, info
    G = [prep_gray(f) for f in frames]
    shape = frames[0].shape
    tpoly = tr.get('poly') or grow_quad(quad, tr.get('grow', 20)).tolist()
    mask = poly_mask(shape, tpoly)
    for ex in tr.get('exclude', []): mask[poly_mask(shape, ex) > 0] = 0
    if tr.get('screen_only') : mask &= poly_mask(shape, quad)
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    Hs = [None] * N; Hs[ref] = np.eye(3); inl_log = [0] * N
    def seed(gray, H):
        m = cv2.warpPerspective(mask, H, (shape[1], shape[0]), flags=cv2.INTER_NEAREST)
        p = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.003, minDistance=4, mask=m, blockSize=5)
        if p is None: return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        p = p.reshape(-1, 2).astype(np.float32)
        r = cv2.perspectiveTransform(p[None].astype(np.float64), np.linalg.inv(H))[0].astype(np.float32)
        return p, r
    for direction in (1, -1):
        cur, refp = seed(G[ref], np.eye(3)); n0 = max(len(cur), 1); H = np.eye(3)
        t = ref
        while 0 <= t + direction < N:
            nt = t + direction
            if len(cur) >= 4:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(G[t], G[nt], cur.reshape(-1, 1, 2), None, **lk)
                back, st2, _ = cv2.calcOpticalFlowPyrLK(G[nt], G[t], nxt, None, **lk)
                fb = np.linalg.norm(back.reshape(-1, 2) - cur, axis=1)
                ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.7)
                cur, refp = nxt.reshape(-1, 2)[ok], refp[ok]
            Hn, ninl = estimate(model, refp, cur)
            if Hn is not None: H = Hn
            Hs[nt] = H; inl_log[nt] = ninl
            if len(cur) < 0.5 * n0:
                c2, r2 = seed(G[nt], H)
                # drop new seeds that duplicate existing points
                if len(cur):
                    d = np.min(np.linalg.norm(c2[:, None] - cur[None], axis=2), axis=1) if len(c2) else []
                    keep = np.asarray(d) > 3
                    c2, r2 = c2[keep], r2[keep]
                cur = np.vstack([cur, c2]).astype(np.float32); refp = np.vstack([refp, r2]).astype(np.float32)
            t = nt
    # ECC refinement against the reference frame (drift-free), init from KLT
    ecc_ok = 0
    if tr.get('ecc', True):
        tmpl = cv2.GaussianBlur(G[ref], (5, 5), 0).astype(np.float32)
        emask = mask.copy()
        for i in range(N):
            if i == ref: continue
            W = Hs[i].astype(np.float32).copy()
            mot = {'homography': cv2.MOTION_HOMOGRAPHY, 'translation': cv2.MOTION_TRANSLATION}.get(model, cv2.MOTION_AFFINE)
            if mot != cv2.MOTION_HOMOGRAPHY: W = W[:2].copy()
            try:
                cc, W2 = cv2.findTransformECC(tmpl, cv2.GaussianBlur(G[i], (5, 5), 0).astype(np.float32), W, mot,
                                              (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), emask, 5)
            except cv2.error:
                continue
            H2 = W2 if W2.shape[0] == 3 else np.vstack([W2, [0, 0, 1]])
            q1 = cv2.perspectiveTransform(quad[None], Hs[i])[0]; q2 = cv2.perspectiveTransform(quad[None], H2.astype(np.float64))[0]
            if cc > tr.get('ecc_min_cc', 0.6) and np.abs(q1 - q2).max() < tr.get('ecc_max_jump', 6):
                Hs[i] = H2.astype(np.float64); ecc_ok += 1
    corners = np.stack([cv2.perspectiveTransform(quad[None], H)[0] for H in Hs])
    raw = corners.copy()
    corners = gsmooth(corners, tr.get('smooth', 1.5))
    jit = np.abs(raw - corners).max()
    info.update(ecc_refined=ecc_ok, frames=N, min_inliers=int(min(x for i, x in enumerate(inl_log) if i != ref) if N > 1 else 0),
                motion_px=float(np.abs(corners - corners[ref]).max()), smooth_residual_px=float(jit))
    return corners, info

# ----------------------------------------------------------------------------- look
def screen_stats(frames, corners, idx):
    """median colour / bright level / grain of the ORIGINAL screen interior."""
    cols, whites, noise = [], [], []
    for i in idx:
        f = frames[i]; m = poly_mask(f.shape, grow_quad(corners[i], -4)) > 0
        if m.sum() < 50: continue
        px = f[m].astype(np.float32)
        cols.append(np.median(px, 0)); lum = px @ np.array([0.114, 0.587, 0.299])
        whites.append(np.percentile(lum, 99))
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hp = (g - cv2.GaussianBlur(g, (0, 0), 1.5))[m]
        noise.append(1.4826 * np.median(np.abs(hp - np.median(hp))))
    return np.median(cols, 0), float(np.median(whites)), float(np.median(noise))

def frame_grain(frames, idx):
    vals = []
    for i in idx:
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
        flat = cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 2), cv2.CV_32F)), (0, 0), 3) < 0.6
        v = hp[flat & (g > 15) & (g < 120)]
        if v.size > 500: vals.append(1.4826 * np.median(np.abs(v - np.median(v))))
    return float(np.median(vals)) if vals else 2.0

def render_plate(img, look, stats):
    """plate (uint8 BGR) -> emissive screen image (float32 0..255, still in plate space)."""
    p = img.astype(np.float32)
    col, white, _ = stats
    lum = p @ np.array([0.114, 0.587, 0.299], np.float32)
    pw = max(np.percentile(lum, 99.5), 1)
    tw = look.get('white') or float(np.clip(white * look.get('exposure', 1.0), 90, 235))
    gain = tw / pw
    tint = (col / max(col.mean(), 1)) ** look.get('tint', 0.35)
    out = p * gain * tint[None, None]
    out += np.asarray(col, np.float32)[None, None] * look.get('lift', 0.25)   # backlight / veiling glare
    h, w = out.shape[:2]
    v = look.get('vignette', 0.12)
    if v:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx / w - .5) * 2) ** 2 * .5 + ((yy / h - .5) * 2) ** 2 * .5
        out *= (1 - v * r2)[..., None]
    return out

def warp_into(frame_shape, img, quad, ss=2):
    """supersampled perspective warp of img onto quad; returns (rgb float32, alpha float32)."""
    H, W = frame_shape[:2]; q = np.asarray(quad, np.float64) * ss
    ew = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    eh = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    tw, th = max(int(ew), 8), max(int(eh), 8)
    small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    M = cv2.getPerspectiveTransform(src, q.astype(np.float32))
    rgb = cv2.warpPerspective(small, M, (W * ss, H * ss), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.warpPerspective(np.ones((th, tw), np.float32), M, (W * ss, H * ss), flags=cv2.INTER_LINEAR, borderValue=0)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA); a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    return rgb, a

# ----------------------------------------------------------------------------- occlusion
def occlusion_masks(frames, corners, occ, N):
    """holdout matte (1 = keep original pixel, i.e. something is in front of the screen) per frame.
    occ = {"keys": [ {mode, thr, ...}, ... ], "polys": [...], "dilate": 5, "feather": 1.2, "min_area": 60}
    key modes (evaluated inside the tracked screen quad, OR-ed together):
      chroma     : YCrCb chroma distance from the screen's median chroma > thr  (skin, coloured objects).
                   Robust to the AI screen content changing, since that content is ~neutral blue-grey.
      luma_below : luma < thr   (dark hair / head silhouette against a bright screen)
      luma_above : luma > thr   (bright hand / sleeve against a dark screen)
      diff       : |luma - temporal median of the screen| > thr. Catches anything, BUT also flags the
                   original AI screen content changing (e.g. a dialog box vanishing) -> only use when the
                   original screen content is static.
    hull: true -> fill the convex hull of each key component >= hull_min px (screen space 400x225; default 400),
            computed after blanking a hull_margin px border ring (default 14) so the bezel isn't hulled.
    polys: [{frames:[a,b], poly:[[x,y],..]}] manual holdouts in frame coords (rotoscope fallback)."""
    keys = list(occ.get('keys', []))
    if occ.get('mode', 'none') != 'none': keys.append(dict(occ))   # legacy single-key form
    shp = frames[0].shape; H, W = shp[:2]
    out = [np.zeros((H, W), np.float32) for _ in range(N)]
    if keys:
        Sw, Sh = 400, 225; dst = np.float32([[0, 0], [Sw, 0], [Sw, Sh], [0, Sh]])
        unw, Ms, val = [], [], []
        ones = np.full((H, W), 255, np.uint8)
        for i in range(N):
            M = cv2.getPerspectiveTransform(np.float32(corners[i]), dst); Ms.append(M)
            unw.append(cv2.cvtColor(cv2.warpPerspective(frames[i], M, (Sw, Sh)), cv2.COLOR_BGR2YCrCb).astype(np.float32))
            # screen-space pixels that are actually inside the frame (quads may extend off-frame)
            val.append(cv2.erode(cv2.warpPerspective(ones, M, (Sw, Sh), flags=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8)) > 0)
        stack = np.stack(unw); V = np.stack(val)
        clean = np.median(stack, 0)
        medc = np.median(stack[..., 1:][V].reshape(-1, 2), 0) if V.any() else np.array([128, 128], np.float32)
        for i in range(N):
            m = np.zeros((Sh, Sw), bool); Y = unw[i][..., 0]
            for k in keys:
                md, thr = k['mode'], k.get('thr', 28)
                if md == 'chroma': m |= np.abs(unw[i][..., 1:] - medc).sum(-1) > thr
                elif md == 'luma_below': m |= Y < thr
                elif md == 'luma_above': m |= Y > thr
                elif md == 'diff': m |= np.abs(Y - clean[..., 0]) > thr
            m &= val[i]
            m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(m)
            for c in range(1, n):
                if st[c, cv2.CC_STAT_AREA] < occ.get('min_area', 60): m[lab == c] = 0
            if occ.get('hull'):   # fill convex hull of big components (shaded hood/head that only partly passes the keys)
                # ignore a ring at the quad border first: the dark bezel there forms a frame-shaped
                # component whose hull would be the whole screen
                hm = int(occ.get('hull_margin', 14)); mi = m.copy(); mi[:hm] = 0; mi[-hm:] = 0; mi[:, :hm] = 0; mi[:, -hm:] = 0
                n, lab, st, _ = cv2.connectedComponentsWithStats(mi)
                for c in range(1, n):
                    if st[c, cv2.CC_STAT_AREA] >= occ.get('hull_min', 400):
                        pts = cv2.findNonZero((lab == c).astype(np.uint8))
                        cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            fm = cv2.warpPerspective(m, np.linalg.inv(Ms[i]), (W, H), flags=cv2.INTER_LINEAR)
            fm = (fm > 64).astype(np.uint8) * 255
            d = occ.get('dilate', 5)   # dilate in FRAME space so the matte can spill past the quad border
            if d: fm = cv2.dilate(fm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
            fm = fm.astype(np.float32) / 255
            out[i] = np.maximum(out[i], fm)
    for hp in occ.get('polys', []):
        a, b = hp['frames']
        for i in range(max(0, a), min(N, b + 1)): out[i] = np.maximum(out[i], poly_mask(shp, hp['poly']).astype(np.float32) / 255)
    if occ.get('matte'):   # external per-frame foreground matte (person segmentation video, white = foreground in front of screen)
        cap = cv2.VideoCapture(occ['matte']); thr = occ.get('matte_thr', 0.5); md = occ.get('matte_dilate', 3)
        for i in range(N):
            ok, mf = cap.read()
            if not ok: break
            mm = cv2.resize(cv2.cvtColor(mf, cv2.COLOR_BGR2GRAY), (W, H)).astype(np.float32) / 255
            mm = np.clip((mm - thr) * 4 + 0.5, 0, 1)
            if md: mm = cv2.dilate(mm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (md, md)))
            out[i] = mm if occ.get('matte_only') else np.maximum(out[i], mm)
    feather = occ.get('feather', 1.2)
    return [cv2.GaussianBlur(o, (0, 0), feather) if feather else o for o in out]

# ----------------------------------------------------------------------------- comp
def composite(frames, cfg, tracks, seed=7):
    N = len(frames); H, W = frames[0].shape[:2]
    rng = np.random.default_rng(seed)
    grain_std = cfg.get('grain', None) or frame_grain(frames, range(0, N, max(1, N // 8)))
    out = [f.astype(np.float32) for f in frames]
    for mon in cfg['monitors']:
        if mon.get('skip'): continue
        C = np.asarray(tracks[mon['name']]['corners'])[:N]
        look = mon.get('look', {})
        plate = Plate(mon['plate'], mon.get('plate_offset', 0), mon.get('src_rect'), mon.get('src_stack'))
        stats = screen_stats(frames, C, range(0, N, max(1, N // 10)))
        if 'white' in look and look['white'] is None: look.pop('white')
        Cg = np.stack([grow_quad(c, look.get('grow', 1.5) + 2) for c in C])   # key over the grown quad so no plate border crosses a hand
        occ = occlusion_masks(frames, Cg, mon.get('occlusion', {}), N)
        # holdout: other (nearer) tracked screens, e.g. a laptop lid in front of a desktop monitor
        for ho in mon.get('holdout', []):
            HC = np.asarray(tracks[ho['name']]['corners'])[:N]
            for i in range(N):
                hm = poly_mask(frames[0].shape, grow_quad(HC[i], ho.get('grow', 4)).tolist()).astype(np.float32) / 255
                occ[i] = np.maximum(occ[i], cv2.GaussianBlur(hm, (0, 0), 0.8))
        # visibility envelope: "visible": [first, last] fully-on frames, "fade_in"/"fade_out" in frames
        vis = np.ones(N, np.float32)
        if 'visible' in mon:
            v0, v1 = mon['visible']; fi = max(1, mon.get('fade_in', 1)); fo = max(1, mon.get('fade_out', 1))
            for i in range(N):
                if i < v0: vis[i] = max(0.0, 1 - (v0 - i) / fi)
                elif i > v1: vis[i] = max(0.0, 1 - (i - v1) / fo)
        cache = {}
        for i in range(N):
            if vis[i] <= 0: continue
            q = grow_quad(C[i], look.get('grow', 1.5))
            key = plate.offset + i if plate.video else 0
            if key not in cache:
                cache.clear(); cache[key] = render_plate(plate.get(i), look, stats)
            rgb, a = warp_into(frames[0].shape, cache[key], q)
            soft = look.get('soften', 0.6)
            if soft: rgb = cv2.GaussianBlur(rgb, (0, 0), soft)
            ef = look.get('edge_feather', 0.6)
            if ef: a = cv2.GaussianBlur(a, (0, 0), ef)
            # grain (matched to the plate's measured grain, a touch of chroma)
            g = rng.normal(0, 1, (H, W)).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 0.55) * (grain_std * look.get('grain', 1.0) / 0.55)
            gc = cv2.GaussianBlur(rng.normal(0, 1, (H, W, 3)).astype(np.float32), (0, 0), 0.8) * grain_std * 0.25
            rgb = rgb + g[..., None] + gc
            a = a * (1 - occ[i]) * vis[i]
            base = out[i]
            comp = base * (1 - a[..., None]) + rgb * a[..., None]
            glow = look.get('glow', 0.18)
            if glow:
                src = rgb * a[..., None]
                bl = cv2.GaussianBlur(src, (0, 0), look.get('glow_sigma', 7)) * glow
                bl2 = cv2.GaussianBlur(src, (0, 0), 1.6) * glow * 0.6     # small halation on bright text
                comp = comp + bl * ((1 - occ[i]) * vis[i])[..., None] + bl2 * a[..., None]
            out[i] = comp
    return [np.clip(o, 0, 255).astype(np.uint8) for o in out]

def encode(frames, dst, fps=FPS):
    H, W = frames[0].shape[:2]
    p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                          '-c:v', 'libx264', '-crf', '12', '-preset', 'slow', '-tune', 'grain', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', dst],
                         stdin=subprocess.PIPE)
    for f in frames: p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    if p.returncode: raise SystemExit('ffmpeg failed')

# ----------------------------------------------------------------------------- debug
def overlay(frames, cfg, tracks, dst_mp4=None, dst_png=None, idx=None):
    vis = []
    for i, f in enumerate(frames):
        v = f.copy()
        for mon in cfg['monitors']:
            if mon['name'] not in tracks: continue
            q = np.asarray(tracks[mon['name']]['corners'][i])
            cv2.polylines(v, [np.round(q * 4).astype(np.int32)], True, (0, 255, 255) if not mon.get('skip') else (0, 0, 255), 1, cv2.LINE_AA, shift=2)
            cv2.putText(v, mon['name'], tuple(np.int32(np.clip(q.mean(0), 10, 800))), 0, 0.4, (0, 255, 255), 1)
        cv2.putText(v, f'{i}', (5, 15), 0, 0.45, (255, 255, 255), 1)
        vis.append(v)
    if dst_mp4: encode(vis, dst_mp4)
    if dst_png is not None:
        idx = idx or [0, len(frames) // 2, len(frames) - 1]
        cv2.imwrite(dst_png, np.hstack([vis[i] for i in idx]))

def gridsheet(img, x0, y0, x1, y1, sc, dst):
    pad = 200
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(40, 0, 40))
    c = big[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    c = np.clip(((c / 255.0) ** 0.6) * 255, 0, 255).astype(np.uint8)
    c = cv2.resize(c, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    for x in range((x0 // 10) * 10, x1, 10):
        X = (x - x0) * sc
        if X < 0: continue
        cv2.line(c, (X, 0), (X, c.shape[0]), (0, 0, 255) if x % 50 == 0 else (0, 90, 0), 1)
        if x % 50 == 0: cv2.putText(c, str(x), (X + 2, 12), 0, 0.4, (0, 255, 255), 1)
    for y in range((y0 // 10) * 10, y1, 10):
        Y = (y - y0) * sc
        if Y < 0: continue
        cv2.line(c, (0, Y), (c.shape[1], Y), (0, 0, 255) if y % 50 == 0 else (0, 90, 0), 1)
        if y % 50 == 0: cv2.putText(c, str(y), (2, Y - 2), 0, 0.4, (0, 255, 255), 1)
    cv2.imwrite(dst, c)

def auto_quads(img, thr=None):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    thr = thr or max(25, np.percentile(g, 60) * 1.6)
    m = cv2.morphologyEx((g > thr).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for c in sorted(cs, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < 2000: continue
        ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2).tolist()
        res.append(dict(area=float(cv2.contourArea(c)), poly=ap))
    return res

# ----------------------------------------------------------------------------- commands
def cmd_probe(shot, a):
    src = newest_clip(shot); fr = read_frames(src); N = len(fr)
    d = f'{DBG}/shot_{shot:02d}_probe'; os.makedirs(d, exist_ok=True)
    for i in [0, N // 4, N // 2, 3 * N // 4, N - 1]: cv2.imwrite(f'{d}/f{i:03d}.png', fr[i])
    H, W = fr[0].shape[:2]
    gridsheet(fr[0], -20, -20, W + 20, H + 20, 2, f'{d}/grid_f000_full.png')
    q = auto_quads(fr[0])
    json.dump(dict(source=fingerprint(src), frames=N, size=[W, H], auto_quads_f0=q), open(f'{d}/probe.json', 'w'), indent=1)
    print(json.dumps(dict(source=src, frames=N, auto_quads_f0=q), indent=1)); print('wrote', d)

def geom_hash(cfg):
    g = dict(ref=cfg.get('ref_frame', 0), m=[[m['name'], m.get('quad'), m.get('track')] for m in cfg['monitors']])
    return hashlib.md5(json.dumps(g, sort_keys=True).encode()).hexdigest()[:12]

def cmd_track(shot, a):
    cfg = load_cfg(shot); src = check_source(cfg, shot, a.force)
    fr = read_frames(src, a.frames); ref = cfg.get('ref_frame', 0)
    tracks = {}
    for mon in cfg['monitors']:
        C, info = track_monitor(fr, mon, ref)
        tracks[mon['name']] = dict(corners=np.round(C, 3).tolist(), info=info)
        print(f"shot {shot} {mon['name']}: {info}")
    json.dump(dict(source=fingerprint(src), geom=geom_hash(cfg), tracks=tracks), open(f'{TRK}/shot_{shot:02d}.json', 'w'))
    overlay(fr, cfg, tracks, f'{DBG}/shot_{shot:02d}_track.mp4', f'{DBG}/shot_{shot:02d}_track.png')
    return tracks

def cmd_comp(shot, a):
    cfg = load_cfg(shot); src = check_source(cfg, shot, a.force)
    tp = f'{TRK}/shot_{shot:02d}.json'
    T = json.load(open(tp)) if os.path.exists(tp) else None
    if not T or T['source']['md5'] != fingerprint(src)['md5'] or T.get('geom') != geom_hash(cfg) or (a.frames and len(next(iter(T['tracks'].values()))['corners']) < a.frames):
        print('tracks missing/stale -> tracking'); tracks = cmd_track(shot, a)
    else: tracks = T['tracks']
    fr = read_frames(src, a.frames)
    if not [m for m in cfg['monitors'] if not m.get('skip')]:
        print(f'shot {shot}: no monitors to composite (config says skip) - no _vscode file written'); return
    res = composite(fr, cfg, tracks)
    dst = out_path(src); encode(res, dst)
    N = len(res); mid = N // 2
    cv2.imwrite(f'{DBG}/shot_{shot:02d}_comp_mid.png', np.hstack([fr[mid], res[mid]]))
    print('wrote', dst)

def cmd_status(a):
    for p in sorted(glob.glob(f'{CFG}/shot_*.json')):
        cfg = json.load(open(p)); s = cfg['shot']
        try: src = newest_clip(s)
        except SystemExit as e: print(s, e); continue
        fresh = cfg.get('source', {}).get('md5') == fingerprint(src)['md5']
        o = out_path(src)
        print(f"shot {s:02d}  config:{'OK' if fresh else 'STALE (re-check corners)'}  src:{os.path.relpath(src, ROOT)}  "
              f"vscode:{'yes' if os.path.exists(o) and os.path.getmtime(o) > os.path.getmtime(src) else 'no'}  "
              f"monitors:{[m['name'] + ('(skip)' if m.get('skip') else '') for m in cfg['monitors']]}")

def cmd_accept(shot, a):
    p = f'{CFG}/shot_{shot:02d}.json'; cfg = json.load(open(p))
    cfg['source'] = fingerprint(newest_clip(shot)); json.dump(cfg, open(p, 'w'), indent=1); print('accepted', cfg['source'])

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('shots', nargs='*', type=int)
    ap.add_argument('--force', action='store_true'); ap.add_argument('--frames', type=int, default=None)
    a = ap.parse_args()
    if a.cmd == 'status': cmd_status(a); sys.exit()
    for s in a.shots:
        if a.cmd == 'run': cmd_track(s, a); cmd_comp(s, a)
        else: globals()[f'cmd_{a.cmd}'](s, a)
