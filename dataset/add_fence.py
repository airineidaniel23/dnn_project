import os
import sys
from PIL import Image, ImageDraw, ImageFilter

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

def main():
    # User-defined parameters
    fence_color = (150, 150, 150)  # Light gray
    link_spacing = 59             # Increase for sparser fence
    link_thickness = 1             # Increase for thicker lines
    fence_blur = 0.5              # Increase blur for a more defocused fence
    fence_opacity = 0.6            # 0.0=transparent, 1.0=fully opaque

    rawframes_dir = "rawframes/1/"
    target_dir = "fenced/1/"
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Gather all image filenames in rawframes directory
    for filename in os.listdir(rawframes_dir):
        input_path = os.path.join(rawframes_dir, filename)
        
        # Skip non-image files; you can expand checks if needed
        if not (filename.lower().endswith(".png") or 
                filename.lower().endswith(".jpg") or 
                filename.lower().endswith(".jpeg")):
            continue
        
        # Open the original image
        with Image.open(input_path).convert("RGBA") as base_image:
            width, height = base_image.size
            
            # Create the fence overlay
            fence_overlay = create_chain_link_fence_overlay(
                width,
                height,
                fence_color=fence_color,
                link_spacing=link_spacing,
                link_thickness=link_thickness,
                fence_blur=fence_blur,
                fence_opacity=fence_opacity
            )
            
            # Composite the fence overlay onto the original image
            composite_image = Image.alpha_composite(base_image, fence_overlay)
            
            # Save result to target folder with the same filename
            output_path = os.path.join(target_dir, filename)
            composite_image.convert("RGB").save(output_path)  # Convert to RGB or keep RGBA if you want

    print("Synthetic fence overlay generation complete!")

if __name__ == "__main__":
    main()
