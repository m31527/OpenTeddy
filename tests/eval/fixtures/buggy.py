def median(nums):
    """Median of a non-empty list of numbers."""
    s = sorted(nums)
    return s[len(s) // 2]
