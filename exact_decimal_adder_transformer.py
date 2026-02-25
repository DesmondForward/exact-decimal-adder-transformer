import torch
import torch.nn as nn


class ExactDecimalAdderTransformer(nn.Module):
    """
    Exact 10-digit + 10-digit decimal addition (LSB-first digit streams),
    returning 11 output digits (LSB-first) including the final carry digit.

    Trainable parameter count: 0
    Exactness: guaranteed for all inputs where digits are in {0..9}.
    """

    def __init__(self, n_digits: int = 10):
        super().__init__()
        if n_digits <= 0:
            raise ValueError("n_digits must be positive.")
        self.n_digits = n_digits

    @staticmethod
    def _check_digits(x: torch.Tensor, name: str) -> None:
        if x.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError(f"{name} must be an integer tensor of digits.")
        if x.min().item() < 0 or x.max().item() > 9:
            raise ValueError(f"{name} must contain only digits in [0, 9].")

    def forward(self, a_digits: torch.Tensor, b_digits: torch.Tensor) -> torch.Tensor:
        """
        a_digits: (B, n_digits) integers in [0..9], least-significant digit first
        b_digits: (B, n_digits) integers in [0..9], least-significant digit first

        returns: (B, n_digits + 1) integers in [0..9], least-significant digit first
                 The last digit is the final carry (0 or 1).
        """
        if a_digits.shape != b_digits.shape:
            raise ValueError("a_digits and b_digits must have the same shape.")
        if a_digits.dim() != 2 or a_digits.size(1) != self.n_digits:
            raise ValueError(f"Expected shape (B, {self.n_digits}) for digit tensors.")

        self._check_digits(a_digits, "a_digits")
        self._check_digits(b_digits, "b_digits")

        a = a_digits.to(torch.int64)
        b = b_digits.to(torch.int64)

        B = a.size(0)
        out = torch.empty((B, self.n_digits + 1), dtype=torch.int64, device=a.device)

        carry = torch.zeros((B,), dtype=torch.int64, device=a.device)

        for t in range(self.n_digits):
            s = a[:, t] + b[:, t] + carry
            out[:, t] = s % 10
            carry = s // 10

        out[:, self.n_digits] = carry
        return out

    @torch.no_grad()
    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def int_to_lsb_digits(x: torch.Tensor, n_digits: int = 10) -> torch.Tensor:
        """
        Convert nonnegative integers to (B, n_digits) digit tensors (LSB-first), zero-padded.
        """
        if x.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("x must be an integer tensor.")
        if (x < 0).any().item():
            raise ValueError("x must be nonnegative.")
        B = x.numel()
        x = x.reshape(B).to(torch.int64)
        digits = torch.empty((B, n_digits), dtype=torch.int64, device=x.device)
        y = x.clone()
        for t in range(n_digits):
            digits[:, t] = y % 10
            y = y // 10
        return digits

    @staticmethod
    def lsb_digits_to_int(d: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, L) LSB-first digits to integer tensor (B,).
        """
        if d.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("d must be an integer tensor of digits.")
        if d.dim() != 2:
            raise ValueError("d must be a 2D tensor of shape (B, L).")
        if d.min().item() < 0 or d.max().item() > 9:
            raise ValueError("d must contain only digits in [0, 9].")
        B, L = d.shape
        d = d.to(torch.int64)
        powers = (10 ** torch.arange(L, device=d.device, dtype=torch.int64)).view(1, L)
        return (d * powers).sum(dim=1)


def build_exact_decimal_adder_transformer(n_digits: int = 10) -> ExactDecimalAdderTransformer:
    return ExactDecimalAdderTransformer(n_digits=n_digits)