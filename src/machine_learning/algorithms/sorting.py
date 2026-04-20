"""
排序算法实现
"""

from typing import List


def bubble_sort(arr: List) -> List:
    """
    冒泡排序算法
    
    时间复杂度: O(n^2)
    空间复杂度: O(1)
    
    Args:
        arr: 待排序列表
        
    Returns:
        排序后的列表
    """
    n = len(arr)
    arr = arr.copy()  # 避免修改原列表
    
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j + 1] < arr[j]:
                arr[j + 1], arr[j] = arr[j], arr[j + 1]
                swapped = True
        if not swapped:  # 如果本趟没有交换，说明已经有序
            break
    return arr


def merge(left: List, right: List) -> List:
    """
    合并两个有序数组
    
    Args:
        left: 左侧有序数组
        right: 右侧有序数组
        
    Returns:
        合并后的有序数组
    """
    res = []
    i, j = 0, 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    
    # 添加剩余的元素
    res.extend(left[i:])
    res.extend(right[j:])
    return res


def merge_sort(arr: List) -> List:
    """
    归并排序算法
    
    时间复杂度: O(n log n)
    空间复杂度: O(n)
    
    Args:
        arr: 待排序列表
        
    Returns:
        排序后的列表
    """
    n = len(arr)
    if n <= 1:
        return arr.copy()
    
    mid = n // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
