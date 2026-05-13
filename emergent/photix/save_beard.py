"""Run this once to pick your beard PNG and copy it to the assets folder."""
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

DEST = Path(__file__).parent / "effects" / "assets" / "beard.png"

root = tk.Tk()
root.withdraw()

src = filedialog.askopenfilename(
    title="Select your beard PNG image",
    filetypes=[("PNG files", "*.png"), ("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")],
)

if src:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, DEST)
    messagebox.showinfo("Done!", f"Beard saved!\n{DEST}\n\nNow run: python main.py")
    print(f"Saved: {DEST}")
else:
    print("No file selected.")

root.destroy()
