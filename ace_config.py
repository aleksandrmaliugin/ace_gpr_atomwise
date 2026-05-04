import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class ACEConfig:
    elements: tuple[str, ...]
    mindist: float
    shells: tuple[float, ...]
    max_order: int = 2

    def __post_init__(self):
        shells_dict = {
            f"{self.shells[i] * self.mindist:.2f}_mindist": (
                self.shells[i] * self.mindist,
                self.shells[i + 1] * self.mindist,
            )
            for i in range(len(self.shells) - 1)
        }
        object.__setattr__(self, "shells_dict", shells_dict)

    def to_dict(self):
        return {
            "elements": self.elements,
            "mindist": self.mindist,
            "shells": self.shells,
            "max_order": self.max_order,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            elements=tuple(d["elements"]),
            mindist=float(d["mindist"]),
            shells=tuple(d["shells"]),
            max_order=int(d["max_order"]),
        )