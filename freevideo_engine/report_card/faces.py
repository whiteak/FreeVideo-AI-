"""Face scores of a take against cast card pictures, run as its own process in any Python that has insightface.

The Report Card starts this file with the Python it finds (Eclipse's own when insightface is installed there, else
FREEVIDEO_FACE_PYTHON, else C:\\LocalAI\\facecheck), so ComfyUI's process never loads onnxruntime. It reads one JSON
object on stdin and writes one on stdout; it imports nothing from Eclipse.

stdin:  {"video": path, "refs": {name: [picture paths]}, "step": seconds}
stdout: {"fps", "frames", "width", "height", "refs": {name: pictures used}, "samples": [{"t", "faces": [...]}]}
Each face: {"name" (closest card, or null without cards), "score" (cosine, buffalo_l), "box": [x0, y0, x1, y1]}.
"""
import json
import sys


def main():
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis

    job = json.load(sys.stdin)
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    refs, used = {}, {}
    for name, paths in (job.get('refs') or {}).items():
        vectors = []
        for path in paths:
            image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            faces = app.get(image) if image is not None else []
            if faces:
                vectors.append(max(faces, key=lambda f: f.bbox[2] - f.bbox[0]).normed_embedding)
                used.setdefault(name, []).append(path)
        if vectors:
            mean = np.mean(vectors, 0)
            refs[name] = mean / np.linalg.norm(mean)

    capture = cv2.VideoCapture(job['video'])
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(float(job.get('step') or 1.) * fps))
    samples, width, height = [], 0, 0
    for index in range(0, frames, step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if not ok:
            break
        height, width = image.shape[:2]
        row = []
        for face in app.get(image):
            scores = {name: float(face.normed_embedding @ vector) for name, vector in refs.items()}
            name = max(scores, key=scores.get) if scores else None
            row.append(dict(name=name, score=round(scores[name], 3) if name else None,
                            box=[round(float(v)) for v in face.bbox]))
        samples.append(dict(t=round(index / fps, 3), faces=row))
    json.dump(dict(fps=fps, frames=frames, width=width, height=height, refs=used, samples=samples), sys.stdout)


if __name__ == '__main__':
    main()
