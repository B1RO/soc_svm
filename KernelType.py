from enum import Enum


class KernelType(Enum):
    LINEAR = "linear"
    POLYNOMIAL = 'polynomial'
    GAUSSIAN = "gaussian"