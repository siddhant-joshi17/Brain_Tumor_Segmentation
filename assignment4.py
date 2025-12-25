import math

def is_prime(n):
    # 1. Prime numbers must be greater than 1
    if n <= 1:
        return False
    
    # 2. Check for divisors from 2 up to sqrt(n)
    # We use int(math.sqrt(n)) + 1 to include the root itself in the range
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            # If divisible, it's a composite number
            return False
            
    # 3. If no divisors were found, it's prime
    return True

# Test cases
test_cases = [
    (2, True),        # Smallest prime number
    (3, True),        # Small prime number
    (4, False),       # First composite number
    (17, True),       # Prime number
    (19, True),       # Prime number
    (20, False),      # Composite number
    (1, False),       # Not a prime number
    (0, False),       # Not a prime number
    (-5, False),      # Negative numbers are not prime
    (97, True)        # Large prime number
]

# Iterate through test cases
for i, (input_num, expected) in enumerate(test_cases):
    print(f"Test Case {i + 1}: Input = {input_num}")
    result = is_prime(input_num)
    print(f"  Output: {result}")
    print(f"  Expected Output: {expected}")
    print(f"  Pass: {result == expected}")
    print("-" * 30)