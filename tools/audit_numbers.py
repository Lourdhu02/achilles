import os
import re
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
os.environ["LABS_IMPL"] = "solution"

from labs._impl import load
mod = load('labs/05_transformer/solution.py')

def count_params(vocab_size, d_model, n_layer, n_head, n_kv_head=None, tie_embeddings=True):
    if n_kv_head is None: n_kv_head = n_head
    config = mod.GPTConfig(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head, tie_embeddings=tie_embeddings)
    model = mod.GPT(config)
    return sum(p.numel() for p in model.parameters())

def audit():
    setup_file = root / "SETUP.md"
    readme_file = root / "labs" / "05_transformer" / "README.md"
    scaling_file = root / "labs" / "08_scaling_laws" / "README.md"
    
    cpu_params = count_params(256, 128, 4, 4)
    gpu8_params = count_params(256, 384, 6, 6)
    gpu24_params = count_params(256, 768, 12, 12)
    s1_gqa_params = count_params(256, 384, 6, 6, 2)
    
    # Process SETUP.md
    if setup_file.exists():
        content = setup_file.read_text(encoding="utf-8")
        content = re.sub(r'about 0\.9M parameters', f'{cpu_params:,} parameters', content)
        content = re.sub(r'about 11M parameters', f'{gpu8_params:,} parameters', content)
        content = re.sub(r'about 85M parameters', f'{gpu24_params:,} parameters', content)
        setup_file.write_text(content, encoding="utf-8")

    # Process labs/05_transformer/README.md
    if readme_file.exists():
        content = readme_file.read_text(encoding="utf-8")
        content = re.sub(r'10\.72M', f'{gpu8_params:,}', content)
        content = re.sub(r'0\.89M parameters', f'{cpu_params:,} parameters', content)
        content = re.sub(r'9\.54M', f'{s1_gqa_params:,}', content)
        readme_file.write_text(content, encoding="utf-8")

    # Process labs/08_scaling_laws/README.md
    if scaling_file.exists():
        content = scaling_file.read_text(encoding="utf-8")
        content = re.sub(r'0\.89M', f'{count_params(256, 128, 4, 2):,}', content)
        content = re.sub(r'2\.71M', f'{count_params(256, 192, 6, 3):,}', content)
        content = re.sub(r'6\.49M', f'{count_params(256, 256, 8, 4):,}', content)
        content = re.sub(r'14\.3M', f'{count_params(256, 384, 8, 6):,}', content)
        content = re.sub(r'32\.3M', f'{count_params(256, 512, 10, 8):,}', content)
        scaling_file.write_text(content, encoding="utf-8")

    print("Audit complete.")

if __name__ == "__main__":
    audit()
