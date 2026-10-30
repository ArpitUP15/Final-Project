import io, zipfile
import pandas as pd
from PIL import Image
from app.agents.multimodal_analysis_agent import analyze_upload


def _read_dataset(filename, content):
    if filename.lower().endswith('.txt'):
        return pd.DataFrame({'text': [line.strip() for line in content.decode().splitlines() if line.strip()]})
    return pd.read_csv(io.BytesIO(content))

def _image_bytes():
    b = io.BytesIO(); Image.new('RGB', (12, 9), (20, 80, 120)).save(b, format='PNG'); return b.getvalue()

def _zip(entries):
    b = io.BytesIO()
    with zipfile.ZipFile(b, 'w') as z:
        for name, content in entries.items(): z.writestr(name, content)
    return b.getvalue()

def test_router_detects_image_zip_and_profiles_images():
    profile = analyze_upload('images.zip', _zip({'train/cat/a.png': _image_bytes(), 'train/dog/b.png': _image_bytes()}), _read_dataset)
    assert profile['modality'] == 'image'
    assert profile['image_count'] == 2
    assert profile['task_type'] == 'classification'

def test_router_detects_text_corpus():
    profile = analyze_upload('reviews.txt', b'Great product and fast delivery\nNot good at all\n', _read_dataset)
    assert profile['modality'] == 'text'
    assert profile['non_empty_documents'] == 2

def test_router_detects_tabular_dataset():
    data = b'age,city,label\n20,Delhi,no\n31,Meerut,yes\n'
    profile = analyze_upload('people.csv', data, _read_dataset)
    assert profile['modality'] == 'tabular'
    assert profile['shape']['rows'] == 2

