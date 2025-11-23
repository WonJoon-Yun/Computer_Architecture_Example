def dot_product(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def matrix_multiplication(a, b):
    result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result

def matrix_addition(a, b):
    result = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            result[i][j] = a[i][j] + b[i][j]
    return result

def matrix_subtraction(a, b):
    result = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            result[i][j] = a[i][j] - b[i][j]
    return result

def matrix_multiplication(a, b):
    result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result

def matrix_addition(a, b):
    result = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            result[i][j] = a[i][j] + b[i][j]
    return result

def matrix_subtraction(a, b):
    result = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            result[i][j] = a[i][j] - b[i][j]
    return result

def multiply(a, b):
    result = 0
    for i in range(b):
        result += a
    return result

def subtract(a, b):
    result = 0
    for i in range(b):
        result -= a
    return result

def divide(a, b):
    result = 0
    while a >= b:
        a -= b
        result += 1
    return result

def power(a, b):
    result = 1
    for i in range(b):
        result *= a
    return result

def modulo(a, b):
    result = 0
    while a >= b:
        a -= b
        result += 1
    return result

def leq(a, b):
    return a <= b

def geq(a, b):
    return a >= b

def greater(a, b):
    return a > b

def less(a, b):
    return a < b

def equal(a, b):
    return a == b

def not_equal(a, b):
    return a != b
