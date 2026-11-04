# Xynz 3D Studio — handler RunPod Serverless: foto → forma 3D (TRELLIS.2, só geometria, sem textura).
# Entrada: { image_url, upload_url, upload_sig, quality: '1536'|'1024'|'512', faces: int, seed: int }
# Saída: envia o GLB para upload_url (servidor Xynz) e devolve { ok, faces, verts, secs }.
import os, io, json, time, tempfile, traceback
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ.setdefault('SPARSE_ATTN_BACKEND', 'xformers')
import requests, numpy as np, torch, trimesh
from PIL import Image
import runpod
from huggingface_hub import snapshot_download
from trellis2.pipelines import Trellis2ImageTo3DPipeline

T0 = time.time()

def load_pipeline():
    tok = os.environ.get('HF_TOKEN') or None
    root = snapshot_download('microsoft/TRELLIS.2-4B', token=tok, allow_patterns=['pipeline.json', '*.md', 'ckpts/ss_*', 'ckpts/shape_*', 'ckpts/slat_flow_img2shape_*'])
    cfg = json.load(open(os.path.join(root, 'pipeline.json')))
    a = cfg['args']
    for k in ['tex_slat_decoder', 'tex_slat_flow_model_512', 'tex_slat_flow_model_1024']:
        a['models'].pop(k, None)                        # textura não é usada: menos memória e início mais rápido
    a['rembg_model'] = {'name': 'BiRefNet', 'args': {'model_name': 'ZhengPeng7/BiRefNet'}}  # MIT (uso comercial livre)
    a['low_vram'] = False
    local = tempfile.mkdtemp(prefix='xynz-')
    for name in os.listdir(root):
        os.symlink(os.path.join(root, name), os.path.join(local, name))
    os.remove(os.path.join(local, 'pipeline.json'))
    json.dump(cfg, open(os.path.join(local, 'pipeline.json'), 'w'))
    p = Trellis2ImageTo3DPipeline.from_pretrained(local)
    p.low_vram = False
    p.cuda()
    return p

PIPE, LOAD_ERR = None, ''
def ensure():
    global PIPE, LOAD_ERR
    if PIPE is None:
        try:
            PIPE = load_pipeline(); LOAD_ERR = ''
            print(f'[xynz] modelo carregado em {time.time() - T0:.1f}s', flush=True)
        except Exception as e:
            LOAD_ERR = str(e)[:400]; traceback.print_exc()
            if 'dinov3' in LOAD_ERR.lower() or 'gated' in LOAD_ERR.lower() or '403' in LOAD_ERR:
                try:
                    from huggingface_hub import snapshot_download as sd
                    sd('facebook/dinov3-vitl16-pretrain-lvd1689m', token=os.environ.get('HF_TOKEN') or None)
                    PIPE = load_pipeline(); LOAD_ERR = ''
                except Exception as e2:
                    LOAD_ERR = 'DINOv3 ainda sem acesso (aprovação da Meta no Hugging Face): ' + str(e2)[:200]
    return PIPE is not None
ensure()

@torch.no_grad()
def make_shape(img: Image.Image, quality: str, seed: int):
    p = PIPE
    img = p.preprocess_image(img)
    torch.manual_seed(seed)
    c512 = p.get_cond([img], 512)
    if quality == '512':
        coords = p.sample_sparse_structure(c512, 32, 1, {})
        slat = p.sample_shape_slat(c512, p.models['shape_slat_flow_model_512'], coords, {})
        res = 512
    else:
        c1024 = p.get_cond([img], 1024)
        coords = p.sample_sparse_structure(c512, 32, 1, {})
        hr = 1536 if quality == '1536' else 1024
        slat, res = p.sample_shape_slat_cascade(c512, c1024, p.models['shape_slat_flow_model_512'],
                                                p.models['shape_slat_flow_model_1024'], 512, hr, coords, {}, 49152)
    torch.cuda.empty_cache()
    meshes, _ = p.decode_shape_slat(slat, res)
    m = meshes[0]
    m.fill_holes()
    return m

def handler(job):
    t = time.time()
    try:
        inp = job.get('input') or {}
        if not ensure():
            return {'ok': False, 'error': LOAD_ERR, 'boot_secs': round(time.time() - T0, 1)}
        if inp.get('warmup'):
            return {'ok': True, 'warm': True, 'gpu': torch.cuda.get_device_name(0), 'boot_secs': round(time.time() - T0, 1)}
        r = requests.get(inp['image_url'], timeout=60); r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        img = img.convert('RGBA') if img.mode in ('RGBA', 'LA', 'P') else img.convert('RGB')
        quality = str(inp.get('quality', '1536'))
        faces = int(inp.get('faces', 400000))
        m = make_shape(img, quality, int(inp.get('seed', 42)))
        if m.faces.shape[0] > faces:
            m.simplify(faces)
        v = m.vertices.detach().cpu().numpy().astype(np.float32)
        f = m.faces.detach().cpu().numpy().astype(np.int64)
        v[:, 1], v[:, 2] = v[:, 2].copy(), -v[:, 1].copy()      # Z para cima → Y para cima (padrão GLB)
        glb = trimesh.Trimesh(v, f, process=False).export(file_type='glb')
        up = requests.post(inp['upload_url'], headers={'X-Xynz-Sig': inp.get('upload_sig', '')},
                           files={'file': ('modelo.glb', glb, 'model/gltf-binary')}, timeout=120)
        ok = up.status_code == 200 and up.json().get('ok')
        return {'ok': bool(ok), 'faces': int(f.shape[0]), 'verts': int(v.shape[0]), 'secs': round(time.time() - t, 1),
                'upload': up.text[:300] if not ok else ''}
    except Exception as e:
        traceback.print_exc()
        return {'ok': False, 'error': str(e)[:500], 'secs': round(time.time() - t, 1)}
    finally:
        torch.cuda.empty_cache()

runpod.serverless.start({'handler': handler})
