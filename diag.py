# Diagnóstico do build: importa cada parte da IA e mostra o erro real (não trava o build).
import importlib, traceback, sys
sys.path.insert(0, '/app/TRELLIS.2')
mods = ['trellis2.modules.sparse', 'trellis2.models.sparse_structure_vae', 'trellis2.models.sparse_structure_flow',
        'trellis2.models.structured_latent_flow', 'trellis2.models.sc_vaes.fdg_vae', 'trellis2.models.sc_vaes.sparse_unet_vae',
        'trellis2.pipelines.rembg', 'trellis2.modules.image_feature_extractor', 'trellis2.pipelines']
for m in mods:
    try:
        importlib.import_module(m); print('[diag] OK', m, flush=True)
    except Exception:
        print('[diag] FALHOU', m, flush=True); traceback.print_exc(); sys.stderr.flush()
