import sys
import random
import time

sys.setrecursionlimit(1000000)  # 增加递归深度


class Sort:
    def __init__(self, n):
        self.len = n
        self.arr = [0] * n
        self.random_data()

    def random_data(self):
        for i in range(self.len):
            self.arr[i] = random.randint(0, 99)

    def partition(self, left, right):  #快排学习新方法
        arr = self.arr
        k = i = left
        random_pos = random.randint(left, right)  # 如何避免陷入最坏时间复杂度
        arr[random_pos], arr[right] = arr[right], arr[random_pos]
        for i in range(left, right):
            if arr[i] < arr[right]:  # 某个位置的值小于分割值，就拿它和k指向的位置交换
                arr[i], arr[k] = arr[k], arr[i]
                k += 1
        arr[k], arr[right] = arr[right], arr[k]
        return k

    def partition_2(self, left, right):  #快排第二种考研方法
        arr = self.arr
        k = arr[left]
        while left < right:
            while left < right and arr[right] >= k:
                right -= 1
            arr[left] = arr[right]
            while left < right and arr[left] <= k:
                left += 1
            arr[right] = arr[left]
        arr[left] = k
        return left

    def quick_sort(self, left, right):
        if left < right:
            pivot = self.partition_2(left, right)
            self.quick_sort(left, pivot - 1)
            self.quick_sort(pivot + 1, right)


if __name__ == '__main__':
    my_sort = Sort(10)
    print(my_sort.arr)
    my_sort.quick_sort(0, 9)
    print(my_sort.arr)
