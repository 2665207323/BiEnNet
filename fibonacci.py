def fibonacci_recursive(n):
    """递归方式求第 n 个斐波那契数（从 0 开始，F(0)=0, F(1)=1）"""
    if n < 0:
        raise ValueError("n 必须为非负整数")
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_iterative(n):
    """迭代方式求第 n 个斐波那契数"""
    if n < 0:
        raise ValueError("n 必须为非负整数")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def fibonacci_sequence(count):
    """生成前 count 个斐波那契数列"""
    if count <= 0:
        return []
    a, b = 0, 1
    result = []
    for _ in range(count):
        result.append(a)
        a, b = b, a + b
    return result


if __name__ == "__main__":
    n = 10
    print(f"斐波那契数列前 {n} 项：")
    print(fibonacci_sequence(n))

    print(f"\n第 {n} 个斐波那契数（迭代）：{fibonacci_iterative(n)}")
    print(f"第 {n} 个斐波那契数（递归）：{fibonacci_recursive(n)}")
