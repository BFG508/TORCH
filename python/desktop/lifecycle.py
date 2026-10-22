"""Release owned Tcl resources on the desktop thread."""

import tkinter as tk
from tkinter import font

from PIL import ImageTk


def owned_resources(root, canvas):
    """Inspect widget attributes and containers, excluding arrays."""
    pending, seen, resources, widgets = [root, canvas], set(), [], []
    while pending:
        obj = pending.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(
            obj, (tk.Variable, tk.Image, ImageTk.PhotoImage, font.Font)
        ):
            resources.append(obj)
        elif isinstance(obj, tk.Misc) or obj is canvas:
            if isinstance(obj, tk.Misc):
                widgets.append(obj)
            pending.extend(vars(obj).values())
        elif isinstance(obj, dict):
            pending.extend(obj.values())
        elif isinstance(obj, (list, tuple, set)):
            pending.extend(obj)
    return resources, widgets


def release(resources, widgets):
    """Dispose Tcl handles before numerical-thread garbage collection.

    Tk variables and images have finalizers but no public close method.
    Retire each handle while its owning thread is available.
    Widgets have already been destroyed when this function is called.
    """
    for obj in resources:
        try:
            obj.__del__()
        except tk.TclError:
            pass
        if isinstance(obj, tk.Variable):
            obj._tk = obj._root = None
        elif isinstance(obj, ImageTk.PhotoImage):
            image = obj._PhotoImage__photo
            image.name, image.tk, obj.tk = None, None, None
        elif isinstance(obj, tk.Image):
            obj.name, obj.tk = None, None
        else:
            obj.delete_font, obj._tk = False, None
            obj._call = obj._split = None
    for widget in widgets:
        widget.tk = None
