"""
Right-click the beard image in Claude → 'Copy Image', then run this script.
It reads from your clipboard and saves it as effects/assets/beard.png automatically.
"""
import sys
from pathlib import Path

try:
    from PIL import ImageGrab
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
    from PIL import ImageGrab

import tkinter as tk
from tkinter import messagebox

DEST = Path(__file__).parent / "effects" / "assets" / "beard.png"

root = tk.Tk()
root.withdraw()

img = ImageGrab.grabclipboard()

if img is None:
    messagebox.showerror(
        "No image found",
        "No image in clipboard.\n\n"
        "Right-click the beard image in Claude → 'Copy Image'\n"
        "Then run this script again."
    )
    print("ERROR: No image in clipboard.")
else:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(DEST), "PNG")
    messagebox.showinfo(
        "Done!",
        f"Beard image saved!\n\n{DEST}\n\nNow run:  python main.py"
    )
    print(f"Saved to: {DEST}")

root.destroy()
