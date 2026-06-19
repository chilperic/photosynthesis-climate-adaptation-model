#!/usr/bin/env python
"""List bundled parameter presets."""
from photosynthesis_adaptation.parameters import describe_available_presets

if __name__ == "__main__":
    for filename, presets in describe_available_presets().items():
        print(filename)
        for preset in presets:
            print(f"  - {preset}")
