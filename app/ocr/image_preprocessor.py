"""OpenCV image preprocessing pipeline with deskew, denoising, contrast and thresholding."""

from typing import Tuple, List
import cv2
import numpy as np

from app.core.exceptions import PreprocessingError
from app.core.logging import get_logger
from app.ocr.quality_checks import assess_image_quality, detect_skew_angle

logger = get_logger("image_preprocessor")


class ImagePreprocessor:
    """Production-grade OpenCV document image enhancement."""

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR/RGB image to single-channel 8-bit grayscale."""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """Apply Non-Local Means Denoising to remove background scanner noise."""
        return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    def deskew(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by negative of detected angle around center."""
        if abs(angle) < 0.5:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive Otsu thresholding for crisp text segmentation."""
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def sharpen(self, gray: np.ndarray) -> np.ndarray:
        """Sharpen slightly blurred characters using an unsharp mask kernel."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(gray, -1, kernel)

    def preprocess(
        self,
        image: np.ndarray,
        apply_binarization: bool = True
    ) -> Tuple[np.ndarray, List[str], dict]:
        """Execute full preprocessing pipeline on input image array."""
        try:
            warnings, metrics = assess_image_quality(image)

            # 0. Automatically upscale small low-res images for legible character heights
            h, w = image.shape[:2]
            if max(h, w) < 1600:
                scale = max(1.0, min(3.5, 1800.0 / max(h, w)))
                if scale > 1.1:
                    image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    warnings.append("RESOLUTION_UPSCALED")

            # 1. Convert to grayscale
            gray = self.to_grayscale(image)

            # 2. Deskew if skew angle is non-negligible
            angle = metrics.get("skew_angle", 0.0)
            if abs(angle) >= 0.5:
                gray = self.deskew(gray, angle)
                warnings.append("ROTATION_CORRECTED")

            # 3. Enhance local contrast
            contrast_enhanced = self.enhance_contrast(gray)

            # 4. Remove scanner/photo noise
            denoised = self.denoise(contrast_enhanced)

            # 5. Sharpen text contours
            sharpened = self.sharpen(denoised)

            # 6. Optional Otsu binarization
            final_img = self.binarize(sharpened) if apply_binarization else sharpened

            return final_img, warnings, metrics
        except Exception as exc:
            logger.error(f"Image preprocessing failed: {exc}")
            raise PreprocessingError(f"Image enhancement error: {exc}")

    def preprocess_for_vision(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, List[str], dict]:
        """
        Enhance image while strictly preserving 3-channel RGB/BGR color:
        - Upscales small/low-res documents (< 1600 px) using high-fidelity bicubic interpolation
        - Corrects subtle tilt/skew while leaving color channels intact
        - Leaves color stamps, colored text, signatures, and inks pristine for multimodal vision
        """
        try:
            warnings, metrics = assess_image_quality(image)
            enhanced = image.copy()

            # 1. Upscale low-res captures (< 1600 px)
            h, w = enhanced.shape[:2]
            if max(h, w) < 1600:
                scale = max(1.0, min(3.5, 1800.0 / max(h, w)))
                if scale > 1.1:
                    enhanced = cv2.resize(enhanced, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    warnings.append("RESOLUTION_UPSCALED")

            # 2. Deskew color image if tilt detected
            angle = metrics.get("skew_angle", 0.0)
            if 0.5 <= abs(angle) <= 15.0:
                enhanced = self.deskew(enhanced, angle)
                warnings.append("ROTATION_CORRECTED")

            return enhanced, warnings, metrics
        except Exception as exc:
            logger.warning(f"Vision color preprocessing failed, returning original: {exc}")
            return image, [], {}


    @staticmethod
    def to_bytes(image: np.ndarray, ext: str = ".png") -> bytes:
        """Encode OpenCV image array to compressed bytes."""
        success, encoded = cv2.imencode(ext, image)
        if not success:
            raise PreprocessingError(f"Failed encoding image to {ext}")
        return encoded.tobytes()

    @staticmethod
    def from_bytes(data: bytes) -> np.ndarray:
        """Decode image byte buffer into OpenCV numpy array."""
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise PreprocessingError("Could not decode image from byte buffer.")
        return img


_preprocessor_instance = ImagePreprocessor()


def get_image_preprocessor() -> ImagePreprocessor:
    """Retrieve singleton ImagePreprocessor instance."""
    return _preprocessor_instance
