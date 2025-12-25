def print_spiral_order(matrix):
    if not matrix or not matrix[0]:
        return []

    result = []
    
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        
        for i in range(left, right + 1):
            result.append(matrix[top][i])
        top += 1 

       
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1 

        
        if top <= bottom:
            for i in range(right, left - 1, -1):
                result.append(matrix[bottom][i])
            bottom -= 1

        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1 

    return result


matrix1 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Test case 2: 4x4 Matrix
matrix2 = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

# Test case 3: 5x5 Matrix
matrix3 = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
]

# Output results for each test case
print("Test Case 1 Output:", print_spiral_order(matrix1))
print("Test Case 2 Output:", print_spiral_order(matrix2))
print("Test Case 3 Output:", print_spiral_order(matrix3))