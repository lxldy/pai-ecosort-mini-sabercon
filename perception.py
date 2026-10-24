"""Small supervised RGB classifier. Only NumPy; no network or API keys."""
from pathlib import Path
import json
import numpy as np

LABELS = ['RED', 'BLUE', 'YELLOW']
COLORS = [[0.88, 0.16, 0.24], [0.05, 0.66, 0.86], [0.95, 0.70, 0.06]]

def feature(pixels):
    rgb = np.median(np.asarray(pixels).reshape(-1, 3), axis=0) / 255.0
    return rgb / max(rgb.sum(), 1e-8)

class ColorAI:
    """Learned softmax regression + distance-based abstention.

    All weights, class centers, and the radius are estimated from training data.
    Softmax scores are not calibrated probabilities of real-world correctness.
    """
    def fit(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        self.mean = x.mean(0)
        self.std = np.maximum(x.std(0), 0.05)
        z = (x - self.mean) / self.std
        self.w = np.zeros((3, 3)); self.b = np.zeros(3)
        target = np.eye(3)[y]
        self.loss = []
        for step in range(1800):
            logits = z @ self.w + self.b
            logits -= logits.max(1, keepdims=True)
            probs = np.exp(logits); probs /= probs.sum(1, keepdims=True)
            grad = (probs - target) / len(x)
            self.w -= 0.06 * (z.T @ grad + 0.001 * self.w)
            self.b -= 0.06 * grad.sum(0)
            if step % 100 == 0:
                self.loss.append(float(-np.log(probs[np.arange(len(y)), y]+1e-10).mean()))
        self.centers = np.array([x[y == k].mean(0) for k in range(3)])
        distances = np.linalg.norm(x-self.centers[y], axis=1)
        self.radius = float(max(0.075, np.quantile(distances, 0.99)*1.4))
        return self

    def predict(self, f):
        f = np.asarray(f)
        logits = (f-self.mean)/self.std @ self.w + self.b
        probs = np.exp(logits-logits.max()); probs /= probs.sum()
        cls = int(probs.argmax())
        distance = float(np.linalg.norm(f-self.centers[cls]))
        reject = distance > self.radius or probs[cls] < 0.8
        return {'class': 'CHECK' if reject else LABELS[cls],
                'raw_class': LABELS[cls], 'score': float(probs[cls]),
                'distance': distance, 'radius': self.radius,
                'scores': probs.tolist()}

    def save(self, path):
        np.savez(path, w=self.w, b=self.b, mean=self.mean, std=self.std,
                 centers=self.centers, radius=self.radius, loss=self.loss)

    @classmethod
    def load(cls, path):
        ai = cls()
        for key, value in np.load(path).items():
            setattr(ai, key, value)
        ai.radius = float(ai.radius)
        return ai

def components(mask):
    """4-connected RGB foreground blobs; does not read simulator IDs."""
    mask = mask.copy()
    h, w = mask.shape
    blobs = []
    for v, u in zip(*np.where(mask)):
        if not mask[v, u]:
            continue
        stack = [(int(v), int(u))]; mask[v, u] = False; points = []
        while stack:
            y, x = stack.pop(); points.append((y, x))
            for yy, xx in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx]:
                    mask[yy, xx] = False; stack.append((yy, xx))
        if len(points) >= 12:
            blobs.append(np.array(points))
    return blobs

def detect(rgb, depth, view, projection, ai, roi=(0.36, 0.82, -0.34, 0.32)):
    """RGB connected components + RGB-D unprojection into world coordinates.

    Color features decide the bin. No ground-truth label or body ID enters here.
    A calibrated workspace ROI excludes the orange robot and colored bin walls.
    """
    h, w = depth.shape
    c = rgb.astype(float)
    hi = c.max(2); lo = c.min(2)
    mask = (hi > 55) & ((hi-lo)/np.maximum(hi,1) > 0.45)
    inv = np.linalg.inv(np.array(projection).reshape(4,4,order='F') @
                        np.array(view).reshape(4,4,order='F'))
    def world(u, v):
        pt = inv @ [2*(u+0.5)/w-1, 1-2*(v+0.5)/h, 2*float(depth[v,u])-1, 1]
        return pt[:3]/pt[3]
    # Crop via projected world positions to avoid colored items outside the source tray.
    vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    ndc = np.stack([2*(uu+.5)/w-1, 1-2*(vv+.5)/h, 2*depth-1, np.ones_like(depth)],-1)
    ws = ndc @ inv.T; ws = ws[...,:3]/ws[...,3:]
    mask &= ((ws[...,0] > roi[0]) & (ws[...,0] < roi[1]) &
             (ws[...,1] > roi[2]) & (ws[...,1] < roi[3]) &
             (ws[...,2] > 0.025) & (ws[...,2] < 0.12))
    detections = []
    for blob in components(mask):
        vs, us = blob[:,0], blob[:,1]
        u, v = int(np.median(us)), int(np.median(vs))
        center = world(u,v)
        result = ai.predict(feature(rgb[vs,us]))
        result.update(xyz=center.tolist(), pixel=[u,v],
                      bbox=[int(us.min()),int(vs.min()),int(us.max()),int(vs.max())])
        detections.append(result)
    return sorted(detections, key=lambda d: (d['xyz'][1], d['xyz'][0]))
