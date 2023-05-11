"""
Formula OCR

Pix2Tex:
    Code: https://github.com/lukas-blecher/LaTeX-OCR
    Docs: https://pix2tex.readthedocs.io/en/latest/pix2tex.html#pix2tex-api-package
"""

# Imports
import asyncio
from pix2tex.api import app
from pix2tex.cli import LatexOCR

# Load Model
app.model = LatexOCR()

# Main Functions
def Pix2Tex_Image2Latex(I_bytes, **params):
    '''
    Image Bytes to Latex Code using Pix2Tex Module
    '''
    # Init Coroutine
    converter = app.predict_from_bytes(I_bytes)
    # Run
    # loop = asyncio.get_event_loop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    latex_code = loop.run_until_complete(asyncio.gather(converter))
    loop.close()
    
    return latex_code

# Main Vars
PIX2TEX_FUNCS = {
    "convert_image_to_latex": Pix2Tex_Image2Latex
}

# RunCode