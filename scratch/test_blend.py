import torch

def test_blend():
    patch_size = 128
    scale = 4
    ps_hr = patch_size * scale
    
    # 2D Hann window
    wy = torch.hann_window(ps_hr, periodic=False)
    wx = torch.hann_window(ps_hr, periodic=False)
    window = torch.outer(wy, wx).unsqueeze(0).clamp(min=1e-3)
    
    H, W = 135, 236
    stride = 32 # 75% overlap / high density
    pad_h = max(0, patch_size - H)
    pad_w = max(0, patch_size - W)
    
    Hp = H + pad_h
    Wp = W + pad_w
    
    out = torch.zeros((1, Hp * scale, Wp * scale))
    weights = torch.zeros((1, Hp * scale, Wp * scale))
    
    y_steps = list(range(0, Hp, stride))
    if (Hp - patch_size) not in y_steps:
        y_steps.append(Hp - patch_size)
    x_steps = list(range(0, Wp, stride))
    if (Wp - patch_size) not in x_steps:
        x_steps.append(Wp - patch_size)
        
    for y in y_steps:
        for x in x_steps:
            y0 = min(y, Hp - patch_size)
            x0 = min(x, Wp - patch_size)
            oy, ox = y0 * scale, x0 * scale
            pred = torch.ones((1, ps_hr, ps_hr))
            out[:, oy:oy+ps_hr, ox:ox+ps_hr] += pred * window
            weights[:, oy:oy+ps_hr, ox:ox+ps_hr] += window
            
    res = out / weights.clamp_min(1e-6)
    res = res[:, :H*scale, :W*scale]
    
    diff_from_flat = (res - 1.0).abs().max().item()
    print(f"Max deviation on flat test: {diff_from_flat:.8f}")
    assert diff_from_flat < 1e-5, f"Diff is {diff_from_flat}"
    print("SUCCESS: 100% seamless reconstruction!")

test_blend()
