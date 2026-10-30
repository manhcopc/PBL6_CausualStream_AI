"""Separate, offline Full075 montage demo. No training or cloud submission."""
import argparse
from bisect import bisect_left, bisect_right
import hashlib
import html
import json
import math
from pathlib import Path
import time

WEIGHTS_SHA = '3e77f8dec8b0b51a13799fb5d20ec48f977b2ed24ddc995bbf02dba95fb96e10'
PARAMETERS = 2_127_532_032
PROMPT = 'Describe only the currently visible physical action in one short English sentence.'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def verify_export(root, manifest):
    root = Path(root).resolve()
    files = manifest.get('files', {})
    if files.get('model.safetensors') != WEIGHTS_SHA:
        raise ValueError('Exact selected Full075 export required')
    checked = []
    for name, expected in files.items():
        if not name or '/' in name or '\\' in name or name in ('.', '..'):
            raise ValueError('Unsafe asset name')
        p = root / name
        if p.is_symlink() or not p.is_file() or digest(p) != expected:
            raise ValueError('Missing/corrupt Full075 asset: ' + name)
        checked.append(dict(name=name, bytes=p.stat().st_size, sha256=expected))
    return checked


def validate_edl(edl):
    if not edl:
        raise ValueError('Empty edit list')
    for row in edl:
        if (not isinstance(row.get('source'), str) or not row['source']
                or any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) for k in ('start', 'duration'))
                or row['start'] < 0 or row['duration'] <= 0):
            raise ValueError('Invalid edit decision')
    total = sum(r['duration'] for r in edl)
    if not 120 <= total <= 180:
        raise ValueError('Demo must be between 120 and 180 seconds')
    return total


def placement(requested, cuda_available):
    if requested == 'cpu':
        return 'cpu', 'float32'
    if requested != 'cuda':
        raise ValueError('Only explicit cuda or cpu mode is supported')
    if not cuda_available:
        raise RuntimeError('CUDA requested but unavailable; choose --device cpu explicitly')
    return 'cuda:0', 'float16'


def grid_from_frames(frames):
    import cv2
    import numpy as np
    if len(frames) != 4 or any(f is None or f.ndim != 3 or f.shape[2] != 3 for f in frames):
        raise ValueError('Exactly four decoded BGR images required')
    tiles=[]
    for i, frame in enumerate(frames):
        h,w=frame.shape[:2]
        scale=min(336/w,336/h)
        nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
        tile=np.zeros((336,336,3),np.uint8)
        x,y=(336-nw)//2,(336-nh)//2
        tile[y:y+nh,x:x+nw]=cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_AREA)
        cv2.putText(tile,f'[{i+1}]',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        tiles.append(tile)
    return np.concatenate([np.concatenate(tiles[:2],axis=1),np.concatenate(tiles[2:],axis=1)],axis=0)


def windows(frames, stride=3):
    pts = [r['pts'] for r in frames]
    if (any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in pts)
            or any(b <= a for a, b in zip(pts, pts[1:])) or stride <= 0):
        raise ValueError('Strict finite increasing native PTS required')
    if len(pts) < 4 or pts[-1] - pts[0] < 1.5:
        return []
    endpoints = []
    target = pts[0] + 1.5
    while target <= pts[-1] + 1e-9:
        endpoint = pts[min(bisect_left(pts, target - 1e-9), len(pts) - 1)]
        if not endpoints or endpoint > endpoints[-1]:
            endpoints.append(endpoint)
        target += stride
    if endpoints[-1] != pts[-1]:
        endpoints.append(pts[-1])
    result = []
    for end in endpoints:
        indices = [max(0, bisect_right(pts, end - offset + 1e-8) - 1) for offset in (1.5, 1., .5, 0.)]
        if len(set(indices)) != 4:
            raise ValueError('Four distinct real frames required')
        chosen = [frames[i] for i in indices]
        result.append(dict(index=len(result), end=end, timestamps=[r['pts'] for r in chosen],
                           frames=[r['path'] for r in chosen]))
    return result


def checked_results(rows, duration):
    ends = [r['end'] for r in rows]
    if (any(not math.isfinite(t) or t < 0 or t > duration for t in ends)
            or any(b <= a for a, b in zip(ends, ends[1:]))):
        raise ValueError('Invalid prediction endpoints')


def subtitles(rows, duration):
    checked_results(rows, duration)
    def clock(t):
        ms = round(t * 1000)
        return f'{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}'
    lines = []
    for i, row in enumerate(rows):
        end = rows[i + 1]['end'] if i + 1 < len(rows) else duration
        if end > row['end']:
            lines.append(f"{i+1}\n{clock(row['end'])} --> {clock(end)}\n{row['raw_text'].strip() or '[empty caption]'}\n")
    return '\n'.join(lines)


def viewer(rows, duration):
    checked_results(rows, duration)
    payload = json.dumps(rows, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Full075 · Demo cảnh trong nhà</title><style>
*{box-sizing:border-box}body{margin:0;background:#10151d;color:#e6edf7;font:16px system-ui}main{max-width:1120px;margin:auto;padding:28px}
h1{font-size:28px;margin:0 0 8px}p{color:#b6c4d8;line-height:1.6}video{width:100%;max-height:68vh;background:black;border-radius:12px}
#caption{padding:18px 22px;min-height:90px;background:#1d2b3e;border-radius:12px;font-size:22px;margin-top:12px}button,a{color:#87d7cf}
button{background:#182536;border:1px solid #41607f;border-radius:8px;padding:9px;margin:4px;cursor:pointer}details{margin-top:20px}
.row{display:flex;gap:15px;padding:9px;border-bottom:1px solid #28364a}.row span:first-child{min-width:85px;color:#93b4d6}a{margin-right:18px}
</style><main><h1>Full075 · Demo nhiều cảnh trong nhà</h1>
<p>Checkpoint epoch 2 · Caption tiếng Anh nguyên văn từ AI · Xử lý offline, phát lại đồng bộ theo thời điểm quan sát.
Video ghép nhiều nguồn; đây là demo, chưa phải kiểm định accuracy hay tốc độ live.</p>
<video id="video" src="demo.mp4" controls preload="metadata" playsinline></video>
<div id="caption">Bấm phát để xem caption của AI.</div><p id="stamp"></p>
<p><a href="demo-captioned.mp4">Video có phụ đề</a><a href="captions.json">Kết quả JSON</a><a href="CREDITS.md">Nguồn &amp; giấy phép</a><a href="GUIDE.md">Hướng dẫn gửi bạn</a></p>
<div id="scenes"></div><details><summary>Tất cả caption (có thể tua tới từng mốc)</summary><div id="rows"></div></details>
<p>Ô caption rỗng được hiển thị [empty caption]; không tự suy ra cảnh không có hành động. Timestamp là PTS trên clip ghép.
Caption bắt đầu sau khi đã quan sát đủ bốn ảnh; thời gian chờ GPU thực nằm trong JSON.</p></main>
<script>const rows=PAYLOAD;const v=document.getElementById('video'),c=document.getElementById('caption'),s=document.getElementById('stamp');
function sync(){const t=v.currentTime;const seen=rows.filter(r => r.end <= t);const r=seen.at(-1);c.textContent=r?(r.raw_text||'[empty caption]'):'Đang chờ đủ bốn ảnh đầu tiên…';s.textContent=r?`Quan sát đến ${r.end.toFixed(2)} giây · ${r.generated_tokens??'?'} token`:'';}
v.addEventListener('timeupdate',sync);v.addEventListener('seeked',sync);v.addEventListener('loadedmetadata',sync);
for(const r of rows){const el=document.createElement('div');el.className='row';const b=document.createElement('button');b.textContent=r.end.toFixed(2)+' s';b.onclick=()=>{v.currentTime=r.end;sync()};const text=document.createElement('span');text.textContent=r.raw_text||'[empty caption]';el.append(b,text);document.getElementById('rows').append(el);}
</script></html>'''.replace('PAYLOAD', payload)


def sample(video, output):
    import cv2
    output = Path(output)
    output.mkdir(exist_ok=False)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('Cannot decode montage')
    frames, next_target, previous, count = [], 0., -1., 0
    last = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(pts) or pts <= previous:
                raise ValueError('Decoder did not return strictly increasing native PTS')
            previous, count = pts, count + 1
            last = frame
            if pts + 1e-8 >= next_target:
                path = f'frame-{len(frames):04}.png'
                if not cv2.imwrite(str(output / path), frame):
                    raise OSError('Frame write failed')
                frames.append(dict(pts=pts, path=path))
                next_target += .5
        if last is not None and frames[-1]['pts'] != previous:
            path = f'frame-{len(frames):04}.png'
            if not cv2.imwrite(str(output / path), last):
                raise OSError('Last frame write failed')
            frames.append(dict(pts=previous, path=path))
    finally:
        cap.release()
    rows = windows(frames)
    if not rows:
        raise ValueError('Montage has no inference windows')
    write(output / 'sampling.json', dict(video_sha256=digest(video), decoded_frames=count,
        last_native_pts=previous, reached_eof=True, stride_sec=3, rows=rows))
    return rows


def infer(model_root, samples, output, requested_device='cuda'):
    """One bounded CUDA batch; predictions never influence scene selection."""
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration
    samples, output, model_root = Path(samples), Path(output), Path(model_root)
    output.mkdir(exist_ok=False)
    began = time.perf_counter()
    manifest = json.loads((model_root / 'full-manifest.json').read_text(encoding='utf-8'))
    assets = verify_export(model_root, manifest)
    device, dtype_name = placement(requested_device, torch.cuda.is_available())
    is_cuda = device.startswith('cuda')
    if is_cuda:
        torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    dtype = getattr(torch, dtype_name)
    processor = AutoProcessor.from_pretrained(model_root, local_files_only=True, trust_remote_code=False,
                                             min_pixels=262144, max_pixels=655360)
    loaded = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_root, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, dtype=dtype,
        attn_implementation='sdpa', device_map={'': device}).eval().requires_grad_(False)
    model.generation_config = GenerationConfig.from_pretrained(model_root, local_files_only=True)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError('Unexpected model size')
    if is_cuda:
        torch.cuda.synchronize()
    write(output / 'load.json', dict(model='full075', weights_sha256=WEIGHTS_SHA, assets=assets,
        parameters=PARAMETERS, dtype=dtype_name, frozen=True, device=torch.cuda.get_device_name(0) if is_cuda else 'cpu',
        verification_processor_sec=loaded-began, model_load_sec=time.perf_counter()-loaded,
        torch=torch.__version__, transformers=__import__('transformers').__version__))
    sampling = json.loads((samples / 'sampling.json').read_text(encoding='utf-8'))
    with (output / 'captions.jsonl').open('x', encoding='utf-8') as stream:
        for row in sampling['rows']:
            started = time.perf_counter()
            frames = [cv2.imread(str(samples / f)) for f in row['frames']]
            if any(f is None for f in frames):
                raise ValueError('Missing sampled frame')
            grid = grid_from_frames(frames)
            text = PROMPT + '\nFrame timestamps: ' + ', '.join(f'{t:.6f}s' for t in row['timestamps'])
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': Image.fromarray(grid[:, :, ::-1])}, {'type': 'text', 'text': text}]}]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors='pt')
            inputs.pop('token_type_ids', None)
            inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device) for k,v in inputs.items()}
            length = inputs['input_ids'].shape[1]
            if length > 591:
                raise ValueError('Input exceeds unchanged 591-token limit')
            if is_cuda:
                torch.cuda.synchronize()
            generation = time.perf_counter()
            with torch.inference_mode():
                tokens = model.generate(**inputs, do_sample=False, max_new_tokens=64, use_cache=True)
            if is_cuda:
                torch.cuda.synchronize()
            result = dict(**row, model='full075', weights_sha256=WEIGHTS_SHA,
                raw_text=processor.batch_decode(tokens[:, length:].cpu(), skip_special_tokens=True)[0],
                input_tokens=length, generated_tokens=tokens.shape[1]-length,
                sequence_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
                processor_sec=generation-started, generation_sec=time.perf_counter()-generation,
                total_sec=time.perf_counter()-started)
            stream.write(json.dumps(result, ensure_ascii=False)+'\n'); stream.flush()
            print(json.dumps({k:result[k] for k in ('index','end','raw_text','total_sec')}), flush=True)
    write(output / 'complete.json', dict(status='COMPLETE', model='full075', windows=len(sampling['rows']),
        elapsed_sec=time.perf_counter()-began, vram_peak_allocated_bytes=torch.cuda.max_memory_allocated() if is_cuda else 0,
        vram_peak_reserved_bytes=torch.cuda.max_memory_reserved() if is_cuda else 0, training=False,
        independent_accuracy=False, realtime=False, cloud_requests=0))
    del model
    if is_cuda:
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('sample'); s.add_argument('video'); s.add_argument('output')
    s = sub.add_parser('infer'); s.add_argument('model'); s.add_argument('samples'); s.add_argument('output')
    s.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args = p.parse_args()
    if args.command == 'sample':
        rows = sample(args.video, args.output); print('sampled windows', len(rows))
    else:
        infer(args.model, args.samples, args.output,args.device)


if __name__ == '__main__':
    main()
