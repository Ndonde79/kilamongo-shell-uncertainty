# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""2 m shell masks from the DeepRBF workflow fields (inside = field >= iso; small bodies are already removed in the field)."""
import json, sys, numpy as np
K = sys.argv[1] if len(sys.argv) > 1 else '.'
out = {}
for m in ['indicator', 'signed_distance', 'contacts']:
    f = np.load(f'{K}/out/Kilamongo_Au_0.3_{m}_field.npy'); r = json.load(open(f'{K}/out/Kilamongo_Au_0.3_{m}_report.json'))
    iso = float(r['wireframe']['iso']); mask = f >= iso
    out[m] = np.packbits(mask.ravel()); out[m + '_shape'] = np.array(f.shape); out[m + '_iso'] = iso
    print(m, f.shape, 'iso', iso, 'volume_m3', int(mask.sum() * 8))
np.savez_compressed(f'{K}/shell_masks_2m.npz', **out)
