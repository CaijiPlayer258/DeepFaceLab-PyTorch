from typing import List

import numpy as np
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
# Assuming these relative imports are correct for the project structure
from ..gui.from_np import QImage_from_np
from .QXWidget import QXWidget


class QXFixedLayeredImages(QXWidget):
    """
    A widget to show multiple stacked images in fixed area.

    It's implied by sizeHint and QRect requirements that fwidth and height
    are integers. Images are scaled to fit this area while preserving
    their aspect ratio.
    """
    def __init__(self, fwidth: int, height: int):
        super().__init__()
        self._fwidth = fwidth
        self._height = height
        self._qp = QPainter() # QPainter instance stored as a member
        self._images: List[tuple] = [] # Stores (image_object, numpy_array_ref_if_any)

    def clear_images(self):
        self._images = []
        self.update()

    def add_image(self, image_data, name=None): # 'name' parameter is unused
        """
        Adds an image to be displayed.
        image_data can be QImage, QPixmap, or a NumPy array (uint8).
        """
        saved_ref = None # To keep a reference to the np array if image_data is one

        if not isinstance(image_data, (QImage, QPixmap)):
            if isinstance(image_data, np.ndarray):
                saved_ref = image_data
                image_data = QImage_from_np(image_data) # Convert np array to QImage
            else:
                raise ValueError(f'Unsupported type of image_data: {image_data.__class__}')

        self._images.append((image_data, saved_ref))
        self.update() # Trigger a repaint

    def sizeHint(self):
        # QSize constructor expects integer width and height.
        return QSize(self._fwidth, self._height)

    def paintEvent(self, event: QPaintEvent): # Added type hint for event
        super().paintEvent(event)

        qp = self._qp
        qp.begin(self)
        try:
            qp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            w = self._fwidth
            h = self._height

            # If the widget has no valid dimensions, cannot draw.
            if w <= 0 or h <= 0:
                return

            w_half = w / 2.0
            h_half = h / 2.0
            # Aspect ratio of the target drawing area (widget_aspect_ratio)
            # Ensure float division for 'a'
            widget_aspect_ratio = w / float(h)

            for image_object, _ in self._images:
                img_size = image_object.size()
                img_w = img_size.width()
                img_h = img_size.height()

                # If the image has no valid dimensions, skip drawing it.
                if img_w <= 0 or img_h <= 0:
                    continue

                # Aspect ratio of the current image (image_aspect_ratio)
                # Ensure float division for 'ap'
                image_aspect_ratio = img_w / float(img_h)

                target_rect: QRect # Define type for clarity

                # Determine the rectangle (target_rect) within the widget to draw the image,
                # scaling it to fit while maintaining its aspect ratio.
                if image_aspect_ratio > widget_aspect_ratio:
                    # Image is wider relative to the widget area (e.g., 16:9 image on 4:3 area).
                    # Fit to widget width 'w'. This will result in letterboxing (empty space above/below).
                    # Calculate scaled height (fitted_height).
                    # Original formula: fitted_height = h * (widget_aspect_ratio / image_aspect_ratio)
                    # Simplified: fitted_height = w / image_aspect_ratio
                    fitted_height = w / image_aspect_ratio
                    target_rect = QRect(
                        0,  # X-coordinate
                        int(h_half - fitted_height / 2.0),  # Y-coordinate (centered)
                        w,  # Width (fixed to widget width)
                        int(fitted_height)  # Scaled height
                    )
                elif image_aspect_ratio < widget_aspect_ratio:
                    # Image is taller relative to the widget area (e.g., 3:4 image on 16:9 area).
                    # Fit to widget height 'h'. This will result in pillarboxing (empty space left/right).
                    # Calculate scaled width (fitted_width).
                    # Original formula: fitted_width = w * (image_aspect_ratio / widget_aspect_ratio)
                    # Simplified: fitted_width = h * image_aspect_ratio
                    fitted_width = h * image_aspect_ratio
                    target_rect = QRect(
                        int(w_half - fitted_width / 2.0),  # X-coordinate (centered)
                        0,  # Y-coordinate
                        int(fitted_width),  # Scaled width
                        h   # Height (fixed to widget height)
                    )
                else: # image_aspect_ratio is effectively equal to widget_aspect_ratio
                    # Aspect ratios match. Scale image to fill the entire widget area.
                    target_rect = QRect(0, 0, w, h)

                # Draw the image (QImage or QPixmap) into the calculated target_rect
                if isinstance(image_object, QImage):
                    qp.drawImage(target_rect, image_object, image_object.rect())
                elif isinstance(image_object, QPixmap):
                    qp.drawPixmap(target_rect, image_object, image_object.rect())
        finally:
            qp.end() # Always end painting session