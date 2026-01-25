import torch
import torch.nn as nn
import math
from typing import Tuple

def cache_init(cache_interval, max_order, num_steps=None,
               enable_first_enhance=False, first_enhance_steps=3, 
               enable_tailing_enhance=False, tailing_enhance_steps=1, 
               low_freqs_order=0, high_freqs_order=2):
    cache_dic = {}
    cache_dic['counter']= 0
    cache_dic['current_step'] = 0
    cache_dic['cache_interval']= cache_interval
    cache_dic['max_order'] = max_order
    cache_dic['num_steps'] = num_steps

    # enhance related utils
    
    # first enhance: fully compute first some steps, enhancing contour infos
    cache_dic['enable_first_enhance'] = enable_first_enhance
    cache_dic['first_enhance_steps'] = first_enhance_steps

    # tailing enhance: fully compute the last 1 steps, enhancing details
    cache_dic['enable_tailing_enhance'] = enable_tailing_enhance
    cache_dic['tailing_enhance_steps'] = tailing_enhance_steps

    # freqs related utils
    cache_dic['low_freqs_order'] = low_freqs_order
    cache_dic['high_freqs_order'] = high_freqs_order

    # features for training-aware cache, here we don't use these
    cache_dic['enable_force_control']= False 
    cache_dic['force_compute']=False
    return cache_dic

class TaylorCacheContainer(nn.Module):
    def __init__(self, max_order):
        super().__init__()
        self.max_order = max_order
        # 逐个注册buffer
        for i in range(max_order + 1):
            self.register_buffer(f"derivative_{i}", None, persistent=False)
            self.register_buffer(f"temp_derivative_{i}", None, persistent=False)
    
    def get_derivative(self, order):
        return getattr(self, f"derivative_{order}")
    
    def set_derivative(self, order, tensor):
        setattr(self, f"derivative_{order}", tensor)

    def set_temp_derivative(self, order, tensor):
        setattr(self, f"temp_derivative_{order}", tensor)

    def get_temp_derivative(self, order):
        return getattr(self, f"temp_derivative_{order}")
    
    def clear_temp_derivative(self):
        for i in range(self.max_order + 1):
            setattr(self, f"temp_derivative_{i}", None)

    def move_temp_to_derivative(self):
        for i in range(self.max_order + 1):
            if self.get_temp_derivative(i) is not None:
                setattr(self, f"derivative_{i}", self.get_temp_derivative(i))
            else:
                break
        self.clear_temp_derivative()

    def get_all_derivatives(self):
        return [getattr(self, f"derivative_{i}") for i in range(self.max_order + 1)]

    def get_all_filled_derivatives(self):
        return [self.get_derivative(i) for i in range(self.max_order + 1) if self.get_derivative(i) is not None]

    def taylor_formula(self, distance):
        output = 0
        for i in range(len(self.get_all_filled_derivatives())):
            output += (1 / math.factorial(i)) * self.get_derivative(i) * (distance ** i)
        return output
    
    def derivatives_computation(self, x, distance):
        '''
        x: tensor, the new x_0
        distance: int, the distance between the current step and the last full computation step
        '''
        self.set_temp_derivative(0, x)
        for i in range(self.max_order):
            if self.get_derivative(i) is not None:
                self.set_temp_derivative(i+1, (self.get_temp_derivative(i) - self.get_derivative(i)) / distance)
            else:
                break
        self.move_temp_to_derivative()

    def clear_derivatives(self):
        for i in range(self.max_order + 1):
            setattr(self, f"derivative_{i}", None)
            setattr(self, f"temp_derivative_{i}", None)


@torch.compile
def decomposition_FFT(x: torch.Tensor, cutoff_ratio: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast Fourier Transform frequency domain decomposition
    
    Args:
        x: Input tensor [B, H*W, D]
        cutoff_ratio: Cutoff frequency ratio (0~0.5)
        
    Returns:
        Tuple of (low_freq, high_freq) tensors with same dtype as input
    """
    orig_dtype = x.dtype
    device = x.device

    x_fp32 = x.to(torch.float32)  # Convert to fp32 for FFT compatibility

    B, HW, D = x_fp32.shape
    freq = torch.fft.fft(x_fp32, dim=1)  # FFT on spatial dimension

    freqs = torch.fft.fftfreq(HW, d=1.0, device=device)
    cutoff = cutoff_ratio * freqs.abs().max()

    # Create frequency masks
    low_mask = freqs.abs() <= cutoff
    high_mask = ~low_mask

    low_mask = low_mask[None, :, None]  # Broadcast to (B, HW, D)
    high_mask = high_mask[None, :, None]

    low_freq_complex  = freq * low_mask
    high_freq_complex = freq * high_mask

    # IFFT and take real part
    low_fp32  = torch.fft.ifft(low_freq_complex,  dim=1).real
    high_fp32 = torch.fft.ifft(high_freq_complex, dim=1).real

    low  = low_fp32.to(device=device, dtype=orig_dtype)
    high = high_fp32.to(device=device, dtype=orig_dtype)

    return low, high

@torch.compile
def reconstruction(low_freq: torch.Tensor, high_freq: torch.Tensor) -> torch.Tensor:
    return low_freq + high_freq

class CacheWithFreqsContainer(nn.Module):
    def __init__(self, max_order):
        super().__init__()
        self.max_order = max_order
        # 逐个注册buffer
        for i in range(max_order + 1):
            self.register_buffer(f"derivative_{i}_low_freqs", None, persistent=False)
            self.register_buffer(f"derivative_{i}_high_freqs", None, persistent=False)
            self.register_buffer(f"temp_derivative_{i}_low_freqs", None, persistent=False)
            self.register_buffer(f"temp_derivative_{i}_high_freqs", None, persistent=False)
    
    def get_derivative(self, order, freqs):
        return getattr(self, f"derivative_{order}_{freqs}")
    
    def set_derivative(self, order, freqs, tensor):
        setattr(self, f"derivative_{order}_{freqs}", tensor)

    def set_temp_derivative(self, order, freqs, tensor):
        setattr(self, f"temp_derivative_{order}_{freqs}", tensor)

    def get_temp_derivative(self, order, freqs):
        return getattr(self, f"temp_derivative_{order}_{freqs}")
    
    def move_temp_to_derivative(self):
        for i in range(self.max_order + 1):
            if self.get_temp_derivative(i, "low_freqs") is not None:
                setattr(self, f"derivative_{i}_low_freqs", self.get_temp_derivative(i, "low_freqs"))
            if self.get_temp_derivative(i, "high_freqs") is not None:
                setattr(self, f"derivative_{i}_high_freqs", self.get_temp_derivative(i, "high_freqs"))
            else:
                break
        self.clear_temp_derivative()

    def get_all_filled_derivatives(self, freqs):
        return [
            self.get_derivative(i, freqs)
            for i in range(self.max_order + 1)
            if self.get_derivative(i, freqs) is not None
        ]

    def taylor_formula(self, distance):
        low_freqs_output = 0
        high_freqs_output = 0
        for i in range(len(self.get_all_filled_derivatives("low_freqs"))):
            low_freqs_output += (1 / math.factorial(i)) * self.get_derivative(i, "low_freqs") * (distance ** i)
        for i in range(len(self.get_all_filled_derivatives("high_freqs"))):
            high_freqs_output += (1 / math.factorial(i)) * self.get_derivative(i, "high_freqs") * (distance ** i)
        return reconstruction(low_freqs_output, high_freqs_output)
    
    def hermite_formula(self, distance):
        low_freqs_output = 0
        high_freqs_output = 0
        for i in range(len(self.get_all_filled_derivatives("low_freqs"))):
            low_freqs_output += (1 / math.factorial(i)) * self.get_derivative(i, "low_freqs") * (distance ** i)
        for i in range(len(self.get_all_filled_derivatives("high_freqs"))):
            high_freqs_output += (1 / math.factorial(i)) * self.get_derivative(i, "high_freqs") * (distance ** i)
        return reconstruction(low_freqs_output, high_freqs_output)

    def derivatives_computation(self, x, distance, low_freqs_order, high_freqs_order):
        '''
        x: tensor, the new x_0
        distance: int, the distance between the current step and the last full computation step
        '''
        x_low, x_high = decomposition_FFT(x, cutoff_ratio=0.1)
        self.set_temp_derivative(0, "low_freqs", x_low)
        self.set_temp_derivative(0, "high_freqs", x_high)
        for i in range(low_freqs_order):
            if self.get_derivative(i, "low_freqs") is not None:
                diff = (self.get_temp_derivative(i, "low_freqs") -
                        self.get_derivative(i, "low_freqs")) / distance
                self.set_temp_derivative(i+1, "low_freqs", diff)
        for i in range(high_freqs_order):
            if self.get_derivative(i, "high_freqs") is not None:
                diff = (self.get_temp_derivative(i, "high_freqs") -
                        self.get_derivative(i, "high_freqs")) / distance
                self.set_temp_derivative(i+1, "high_freqs", diff)
        self.move_temp_to_derivative()
        
    def clear_temp_derivative(self):
        for i in range(self.max_order + 1):
            setattr(self, f"temp_derivative_{i}_low_freqs", None)
            setattr(self, f"temp_derivative_{i}_high_freqs", None)

    def clear_derivatives(self):
        for i in range(self.max_order + 1):
            setattr(self, f"derivative_{i}_low_freqs", None)
            setattr(self, f"derivative_{i}_high_freqs", None)
            setattr(self, f"temp_derivative_{i}_low_freqs", None)
            setattr(self, f"temp_derivative_{i}_high_freqs", None)