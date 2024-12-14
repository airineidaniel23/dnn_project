import tkinter as tk
from tkinter import filedialog, colorchooser
from PIL import Image, ImageDraw, ImageFilter, ImageTk

def create_chain_link_fence_overlay(
    width: int,
    height: int,
    fence_color: tuple = (128, 128, 128),
    link_spacing: int = 40,
    link_thickness: int = 3,
    fence_blur: float = 1.0,
    fence_opacity: float = 0.5
) -> Image.Image:
    """
    Creates a synthetic chain-link fence overlay (RGBA) of size (width x height).
    This revised version expands the diagonal range coverage so large link_spacing
    won't leave corners uncovered.
    
    :param width: Width of the image.
    :param height: Height of the image.
    :param fence_color: The RGB color of the fence lines, e.g. (128, 128, 128).
    :param link_spacing: The distance in pixels between fence lines.
    :param link_thickness: The thickness of the fence lines.
    :param fence_blur: Gaussian blur radius for the fence overlay.
    :param fence_opacity: Overall opacity [0.0 (invisible) .. 1.0 (fully opaque)].
    :return: A PIL Image in RGBA mode with the chain-link fence drawn.
    """
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # We expand our range so that even if link_spacing is large, the diagonal lines
    # still run off the edges of the image. Without this, big spacing could cause
    # partial coverage or blank corners.
    # Example:  use -2*height to (width + 2*height) instead of just -height to (width+height).

    # ----- Diagonal lines (bottom-left to top-right) -----
    # Each line runs from (start_x, 0) to (start_x - height, height).
    # We shift start_x over a wider range to ensure full coverage.
    for start_x in range(-2 * height, width + 2 * height, link_spacing):
        draw.line([(start_x, 0), (start_x - height, height)],
                  fill=fence_color, width=link_thickness)

    # ----- Diagonal lines (top-left to bottom-right) -----
    # Each line runs from (start_x, 0) to (start_x + height, height).
    for start_x in range(-2 * height, width + 2 * height, link_spacing):
        draw.line([(start_x, 0), (start_x + height, height)],
                  fill=fence_color, width=link_thickness)

    # Apply optional Gaussian blur to soften the fence lines
    if fence_blur > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(fence_blur))

    # Adjust alpha (transparency) across the overlay
    alpha_value = int(fence_opacity * 255)
    r, g, b, a = overlay.split()
    # For any non-zero alpha from the draw lines, set it to alpha_value
    a = a.point(lambda p: alpha_value if p > 0 else 0)
    overlay = Image.merge("RGBA", (r, g, b, a))

    return overlay

class ChainLinkFenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chain-Link Fence Overlay GUI")

        # Default parameters
        self.fence_color = (128, 128, 128)
        self.link_spacing = 40
        self.link_thickness = 3
        self.fence_blur = 1.0
        self.fence_opacity = 0.5

        self.base_image = None
        self.preview_image = None
        self.tk_preview = None

        # --- Top Frame: Load Image Button, Color Chooser Button ---
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, pady=5)

        load_button = tk.Button(top_frame, text="Load Image", command=self.load_image)
        load_button.pack(side=tk.LEFT, padx=5)

        color_button = tk.Button(top_frame, text="Fence Color", command=self.choose_color)
        color_button.pack(side=tk.LEFT, padx=5)

        # --- Middle Frame: Sliders ---
        slider_frame = tk.Frame(root)
        slider_frame.pack(side=tk.TOP, pady=5)

        self.spacing_scale = tk.Scale(slider_frame, from_=5, to=200, orient=tk.HORIZONTAL,
                                      label="Link Spacing", command=self.update_preview)
        self.spacing_scale.set(self.link_spacing)
        self.spacing_scale.pack(fill=tk.X, padx=5, pady=2)

        self.thickness_scale = tk.Scale(slider_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                                        label="Link Thickness", command=self.update_preview)
        self.thickness_scale.set(self.link_thickness)
        self.thickness_scale.pack(fill=tk.X, padx=5, pady=2)

        self.blur_scale = tk.Scale(slider_frame, from_=0, to=10, resolution=0.5,
                                   orient=tk.HORIZONTAL, label="Fence Blur",
                                   command=self.update_preview)
        self.blur_scale.set(self.fence_blur)
        self.blur_scale.pack(fill=tk.X, padx=5, pady=2)

        self.opacity_scale = tk.Scale(slider_frame, from_=0, to=1, resolution=0.1,
                                      orient=tk.HORIZONTAL, label="Fence Opacity",
                                      command=self.update_preview)
        self.opacity_scale.set(self.fence_opacity)
        self.opacity_scale.pack(fill=tk.X, padx=5, pady=2)

        # --- Preview Canvas / Label ---
        self.preview_label = tk.Label(root, text="No image loaded", bg="gray")
        self.preview_label.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=5, pady=5)

    def load_image(self):
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        if not filename:
            return
        
        self.base_image = Image.open(filename).convert("RGBA")
        self.update_preview()

    def choose_color(self):
        color_code = colorchooser.askcolor(title="Choose fence color", initialcolor=self.fence_color)
        if color_code and color_code[0]:
            # color_code[0] is (R, G, B)
            self.fence_color = tuple(map(int, color_code[0]))  # Convert floats to ints
            self.update_preview()

    def update_preview(self, *args):
        if self.base_image is None:
            return
        
        # Fetch slider values
        self.link_spacing = self.spacing_scale.get()
        self.link_thickness = self.thickness_scale.get()
        self.fence_blur = self.blur_scale.get()
        self.fence_opacity = self.opacity_scale.get()

        # Generate fence overlay
        width, height = self.base_image.size
        fence_overlay = create_chain_link_fence_overlay(
            width, height,
            fence_color=self.fence_color,
            link_spacing=self.link_spacing,
            link_thickness=self.link_thickness,
            fence_blur=self.fence_blur,
            fence_opacity=self.fence_opacity
        )

        # Composite fence with original
        composite_image = Image.alpha_composite(self.base_image, fence_overlay)

        # Convert to PhotoImage for Tkinter display
        self.preview_image = composite_image
        self.tk_preview = ImageTk.PhotoImage(self.preview_image)
        
        self.preview_label.config(image=self.tk_preview)
        self.preview_label.image = self.tk_preview  # keep a reference so it’s not garbage-collected


def main():
    root = tk.Tk()
    app = ChainLinkFenceGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
