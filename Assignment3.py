def is_palindrome_iterative(s):
    # An empty string or single character is a palindrome
    if len(s) <= 1:
        return True
    
    # Initialize pointers
    left = 0
    right = len(s) - 1
    
    while left < right:
        # If characters don't match, it's not a palindrome
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
        
    return True

def is_palindrome_recursive(s):
    # Base Case: Empty string or single character
    if len(s) <= 1:
        return True
    
    # Recursive Step: Check outer characters
    if s[0] == s[-1]:
        # Recurse on the string excluding the first and last characters
        return is_palindrome_recursive(s[1:-1])
    
    return False

# Test cases
test_cases = [
    ("radar", True),      # Simple palindrome
    ("boommoob", True),   # Simple palindrome
    ("world", False),     # Not a palindrome
    ("a", True),          # Single character
    ("", True),           # Empty string
    ("RadAr", False),     # Case-sensitive test
]

# Iterate over test cases and print results
for i, (input_str, expected) in enumerate(test_cases):
    print(f"Test Case {i + 1}: Input = '{input_str}'")
    result_iterative = is_palindrome_iterative(input_str)
    result_recursive = is_palindrome_recursive(input_str)
    print(f"  Iterative Output: {result_iterative}")
    print(f"  Recursive Output: {result_recursive}")
    print(f"  Expected Output: {expected}")
    print(f"  Iterative Pass: {result_iterative == expected}")
    print(f"  Recursive Pass: {result_recursive == expected}")
    print("-" * 30)