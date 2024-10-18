import numpy as np
from typing import List, Tuple, Optional, Dict
import mindspore.dataset.vision.transforms as CV
from mindspore.dataset.vision import Inter
from mindspore.common.tensor import Tensor

def _apply_op(
        img: Tensor, op_name: str, magnitude: float, interpolation: Inter, fill: Optional[List[float]]
) -> Tensor:
    if op_name == "ShearX":
        img = CV.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[np.degrees(np.arctan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        img = CV.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, np.degrees(np.arctan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = CV.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = CV.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = CV.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = CV.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = CV.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = CV.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = CV.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = CV.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = CV.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = CV.auto_contrast(img)
    elif op_name == "Equalize":
        img = CV.equalize(img)
    elif op_name == "Invert":
        img = CV.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


class RandAugment:
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (Inter): Desired interpolation enum defined by
            :class:`mindspore.dataset.vision.Inter`. Default is ``Inter.LINEAR``.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
            self,
            num_ops: int = 2,
            magnitude: int = 9,
            num_magnitude_bins: int = 31,
            interpolation: Inter = Inter.LINEAR,
            fill: Optional[List[float]] = None,
    ) -> None:
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (Tensor(np.array(0.0)), False),
            "ShearX": (Tensor(np.linspace(0.0, 0.3, num_bins)), True),
            "ShearY": (Tensor(np.linspace(0.0, 0.3, num_bins)), True),
            "TranslateX": (Tensor(np.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins)), True),
            "TranslateY": (Tensor(np.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins)), True),
            "Rotate": (Tensor(np.linspace(0.0, 30.0, num_bins)), True),
            "Brightness": (Tensor(np.linspace(0.0, 0.9, num_bins)), True),
            "Contrast": (Tensor(np.linspace(0.0, 0.9, num_bins)), True),
            "Sharpness": (Tensor(np.linspace(0.0, 0.9, num_bins)), True),
            "Posterize": (8 - (Tensor(np.arange(num_bins)) / ((num_bins - 1) / 4)).round().astype(int), False),
            "AutoContrast": (Tensor(np.array(0.0)), False),
            "Equalize": (Tensor(np.array(0.0)), False),
        }

    def forward (self, img: Tensor) -> Tensor:
        fill = self.fill
        image_shape = img.shape
        channels, height, width = image_shape[-3:]
        #channels, height, width = CV   #  F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

            # 数据增强
        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(np.random.randint(len(op_meta)))
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].asnumpy()) if magnitudes.ndim > 0 else 0.0
            if signed and int(np.random.randint(2)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img