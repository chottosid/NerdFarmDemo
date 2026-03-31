"""Enhanced vision-based document processing using GPT-4o Vision API.

Features:
- Image quality assessment (blur, contrast, resolution)
- Auto-enhancement for low-quality images
- Unclear text marking with [unclear: ...]
- Handwriting detection
- Confidence scoring per extraction
"""

import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from openai import AsyncOpenAI

from app.config import get_settings
from .schemas import Page

logger = logging.getLogger(__name__)


@dataclass
class ImageQuality:
    """Quality metrics for an image."""
    resolution: tuple[int, int]
    blur_score: float  # 0-1, lower = more blurry
    contrast_score: float  # 0-1, lower = low contrast
    is_low_quality: bool
    issues: list[str]


class ImageQualityAssessor:
    """Assess and enhance image quality for vision processing."""

    MIN_RESOLUTION = (800, 800)
    MIN_BLUR_SCORE = 0.1  # Below this = too blurry
    MIN_CONTRAST_SCORE = 0.2  # Below this = too low contrast

    def assess(self, image: Image.Image) -> ImageQuality:
        """Assess image quality metrics.

        Args:
            image: PIL Image to assess

        Returns:
            ImageQuality with metrics
        """
        # Convert to grayscale numpy array for analysis
        gray = np.array(image.convert('L'))

        # Resolution check
        resolution = image.size

        # Blur detection using Laplacian variance
        blur_score = self._calculate_blur_score(gray)

        # Contrast calculation
        contrast_score = self._calculate_contrast(gray)

        # Determine if low quality
        issues = []
        is_low_quality = False

        if resolution[0] < self.MIN_RESOLUTION[0] or resolution[1] < self.MIN_RESOLUTION[1]:
            issues.append(f"Low resolution: {resolution[0]}x{resolution[1]}")
            is_low_quality = True

        if blur_score < self.MIN_BLUR_SCORE:
            issues.append(f"Blurry image: score={blur_score:.3f}")
            is_low_quality = True

        if contrast_score < self.MIN_CONTRAST_SCORE:
            issues.append(f"Low contrast: score={contrast_score:.3f}")
            is_low_quality = True

        return ImageQuality(
            resolution=resolution,
            blur_score=blur_score,
            contrast_score=contrast_score,
            is_low_quality=is_low_quality,
            issues=issues,
        )

    def enhance(self, image: Image.Image, quality: ImageQuality | None = None) -> Image.Image:
        """Apply enhancements to improve image quality.

        Args:
            image: PIL Image to enhance
            quality: Pre-computed quality metrics (optional)

        Returns:
            Enhanced image
        """
        if quality is None:
            quality = self.assess(image)

        if not quality.is_low_quality:
            return image

        enhanced = image.copy()

        # Denoise if blurry
        if quality.blur_score < 0.2:
            enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
            logger.debug("Applied denoising filter")

        # Enhance contrast if low
        if quality.contrast_score < 0.4:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.5)  # 50% more contrast
            logger.debug("Enhanced contrast")

        # Sharpen if needed
        if quality.blur_score < 0.3:
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            logger.debug("Applied sharpening")

        return enhanced

    def _calculate_blur_score(self, gray_array: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance.

        Higher score = sharper image.
        """
        try:
            # Laplacian for edge detection
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

            # Simple convolution
            from scipy import ndimage
            edges = ndimage.convolve(gray_array.astype(float), laplacian)
            variance = np.var(edges)

            # Normalize to 0-1 range (empirically, good images have variance > 500)
            return min(variance / 1000, 1.0)
        except ImportError:
            # Fallback if scipy not available
            return 0.5  # Assume medium quality
        except Exception:
            return 0.5

    def _calculate_contrast(self, gray_array: np.ndarray) -> float:
        """Calculate contrast score based on pixel value spread.

        Higher score = better contrast.
        """
        min_val = np.min(gray_array)
        max_val = np.max(gray_array)
        contrast_range = max_val - min_val

        # Normalize to 0-1 (255 = max possible contrast)
        return contrast_range / 255


# Enhanced vision extraction prompt with unclear text marking
VISION_EXTRACTION_PROMPT = """Analyze this legal document image and extract all information.

IMPORTANT INSTRUCTIONS FOR UNCLEAR TEXT:
- If text is difficult to read due to handwriting, low quality, blur, or damage, mark it as: [unclear: your best guess]
- Example: "Signed by [unclear: John Sm?th]" or "Amount: [unclear: $2?,000]"
- Be honest about uncertainty - it's better to mark unclear than to hallucinate
- For completely illegible sections, use: [unclear: illegible text in section X]

Return ONLY a valid JSON object with this exact structure:
{
  "full_text": "complete text content preserving structure. Use [unclear: ...] for uncertain text",
  "tables": [
    {
      "markdown": "table rendered as markdown",
      "caption": "table caption if present, null otherwise"
    }
  ],
  "signatures": [
    {
      "description": "description of whose signature or 'signature detected'",
      "location": "bottom right, bottom left, etc."
    }
  ],
  "stamps_seals": [
    {
      "description": "notary seal, court stamp, etc.",
      "location": "where on page"
    }
  ],
  "document_type": "one of: contract, notice, complaint, motion, agreement, memo, brief, letter, other",
  "parties": [
    {
      "name": "full name of party",
      "role": "plaintiff, defendant, petitioner, respondent, counsel, witness, signatory, other"
    }
  ],
  "dates": [
    {
      "date": "YYYY-MM-DD format if possible",
      "context": "filing date, execution date, deadline, hearing date, etc."
    }
  ],
  "amounts": [
    {
      "amount": "numeric value",
      "context": "settlement amount, filing fee, damages, etc."
    }
  ],
  "case_ids": [
    {
      "type": "case_number, docket_id, file_number",
      "value": "the identifier value"
    }
  ],
  "key_terms": ["important legal terms or phrases from document"],
  "handwriting_detected": true/false,
  "handwritten_sections": ["list of sections that appear to be handwritten"],
  "overall_clarity": "clear, partial, or poor",
  "unclear_sections": ["list of sections marked with [unclear: ...]"],
  "confidence": 0.95
}

Important:
- Extract ALL text, do not summarize
- Preserve document structure (headers, sections, paragraphs)
- Mark uncertain text with [unclear: ...] format
- Detect if any handwritten content is present
- Convert tables to markdown format
- Describe signatures and stamps even if you cannot read text in them
- Use null for fields you cannot find
- Be precise with dates and amounts
- Set confidence lower if image quality is poor"""


class VisionProcessor:
    """Process documents using vision models with quality assessment.

    Features:
    - Image quality assessment before processing
    - Auto-enhancement for low-quality images
    - Unclear text marking
    - Handwriting detection
    """

    def __init__(self, model: str | None = None):
        """Initialize vision processor.

        Args:
            model: Vision model to use (defaults to gpt-4o-mini for cost efficiency)
        """
        settings = get_settings()
        self.model = model or settings.vision_model
        self.client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
        self.max_tokens = settings.vision_max_tokens
        self.quality_assessor = ImageQualityAssessor()

    async def process_image(
        self,
        image: Image.Image,
        page_num: int = 1,
    ) -> tuple[Page, dict[str, Any]]:
        """Process a single image with vision model.

        Args:
            image: PIL Image to process
            page_num: Page number for tracking

        Returns:
            Tuple of (Page object, raw extraction dict)
        """
        # Assess image quality
        quality = self.quality_assessor.assess(image)

        if quality.is_low_quality:
            logger.warning(
                f"Low quality image detected for page {page_num}: {', '.join(quality.issues)}"
            )
            # Enhance the image
            image = self.quality_assessor.enhance(image, quality)

        # Upscale if needed
        image = self._upscale_if_needed(image)

        # Convert image to base64
        image_base64 = self._image_to_base64(image)

        # Call vision model
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": VISION_EXTRACTION_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Extract all information from this legal document.",
                        },
                    ],
                },
            ],
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        # Parse response
        extraction = json.loads(response.choices[0].message.content)

        # Extract clarity info
        overall_clarity = extraction.get("overall_clarity", "clear")
        unclear_sections = extraction.get("unclear_sections", [])
        has_unclear = overall_clarity != "clear" or len(unclear_sections) > 0

        # Adjust confidence based on quality
        confidence = extraction.get("confidence", 0.9)
        if quality.is_low_quality:
            confidence = min(confidence, 0.85)  # Cap confidence for low-quality inputs

        # Create Page object
        page = Page(
            page_num=page_num,
            text=extraction.get("full_text", ""),
            confidence=confidence,
            has_unclear=has_unclear,
        )

        # Add quality and clarity info to extraction
        extraction["_image_quality"] = {
            "resolution": quality.resolution,
            "blur_score": quality.blur_score,
            "contrast_score": quality.contrast_score,
            "was_enhanced": quality.is_low_quality,
            "issues": quality.issues,
        }

        return page, extraction

    async def process_pdf_page(
        self,
        pdf_page_image: Image.Image,
        page_num: int,
    ) -> tuple[Page, dict[str, Any]]:
        """Process a single PDF page image.

        Args:
            pdf_page_image: PIL Image of the PDF page
            page_num: Page number

        Returns:
            Tuple of (Page object, raw extraction dict)
        """
        return await self.process_image(pdf_page_image, page_num)

    def _image_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        """Convert PIL Image to base64 string.

        Args:
            image: PIL Image
            format: Image format (JPEG, PNG)

        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode in ("RGBA", "P", "L"):
            image = image.convert("RGB")

        image.save(buffer, format=format, quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _upscale_if_needed(self, image: Image.Image, min_width: int = 1500) -> Image.Image:
        """Upscale image if it's too small for good vision processing.

        Args:
            image: PIL Image
            min_width: Minimum width in pixels

        Returns:
            Upscaled image (or original if already large enough)
        """
        if image.width < min_width:
            scale = min_width / image.width
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)
            logger.debug(f"Upscaled image from {image.width} to {new_size[0]} pixels wide")
        return image


# Convenience function for direct file processing
async def extract_from_file(file_path: str, page_num: int = 1) -> tuple[Page, dict[str, Any]]:
    """Extract document content from an image file.

    Args:
        file_path: Path to image file
        page_num: Page number for tracking

    Returns:
        Tuple of (Page object, raw extraction dict)
    """
    processor = VisionProcessor()
    image = Image.open(file_path)
    return await processor.process_image(image, page_num)
