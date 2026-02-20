import pytest
import random
from 二叉树 import Sort

class TestRandomData:
    def test_random_data_length(self):
        """测试生成的数组长度是否正确"""
        n = 10
        s = Sort(n)
        assert len(s.arr) == n

    def test_random_data_range(self):
        """测试生成的数字是否在0-99范围内"""
        n = 100
        s = Sort(n)
        for num in s.arr:
            assert 0 <= num <= 99

    def test_random_data_uniqueness(self):
        """测试生成的数组是否随机(至少有一个元素不同)"""
        n = 100
        s1 = Sort(n)
        s2 = Sort(n)
        assert s1.arr != s2.arr  # 由于随机性，两个数组应该不同

    def test_random_data_edge_case_empty(self):
        """测试长度为0的边界情况"""
        n = 0
        s = Sort(n)
        assert len(s.arr) == 0

    def test_random_data_negative_length(self):
        """测试负长度的情况"""
        with pytest.raises(ValueError):
            Sort(-1)