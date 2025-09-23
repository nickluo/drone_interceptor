import numpy as np

class EMAFilter:
    """
    指数移动平均滤波器 (Exponential Moving Average Filter)
    """
    def __init__(self, n: int, channels: int = 1):
        if n <= 0:
            raise ValueError("n must be a positive integer")
        if channels <= 0:
            raise ValueError("channels must be a positive integer")
        self.alpha = 2.0 / (n + 1)
        self.data = np.zeros(channels)
        self.initialized = np.full(channels, False)
        self.channels = channels
    
    def filter_array(self, values: np.ndarray) -> np.ndarray:
        if values.shape[0] != self.channels:
            raise ValueError(f"Input array size {values.shape[0]} does not match filter channels {self.channels}")
        
        for i in range(self.channels):
            if not self.initialized[i]:
                self.data[i] = values[i]
                self.initialized[i] = True
            else:
                self.data[i] = self.alpha * values[i] + (1 - self.alpha) * self.data[i]
        return self.data.copy()
