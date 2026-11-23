with open('scratch/build_1_5meter_experiments.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the internal docstring in Cell 9
old_doc = '''    """
    Photographic Tone-Mapping & Clarity Restoration:
    1. Percentile Calibration (1.5% to 98.5% bounds).
    2. Gamma Curve (gamma=1.45): Pulls down overexposed midtones, restoring deep shadows,
       road contrast, and building distinction (eliminates washed-out bright look).
    3. LAB Luminance Edge Acuity: Restores razor-sharp building outlines and streets
       without altering true physical color hue.
    """'''

new_doc = '''    # Photographic Tone-Mapping & Clarity Restoration:
    # 1. Percentile Calibration (1.5% to 98.5% bounds).
    # 2. Gamma Curve (gamma=1.45): Pulls down overexposed midtones, restoring deep shadows,
    #    road contrast, and building distinction (eliminates washed-out bright look).
    # 3. LAB Luminance Edge Acuity: Restores razor-sharp building outlines and streets
    #    without altering true physical color hue.'''

if old_doc in text:
    text = text.replace(old_doc, new_doc)
    print("Replaced Cell 9 internal docstring!")
else:
    print("Docstring not found directly, checking...")

with open('scratch/build_1_5meter_experiments.py', 'w', encoding='utf-8') as f:
    f.write(text)

