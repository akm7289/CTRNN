import math


class Solution(object):
    def binarySearch(self, num, numbers, startIndex, endIndex):
        while (startIndex <= endIndex):
            mid = (endIndex + startIndex) // 2
            if num == numbers[mid]:
                return mid
            elif numbers[mid] > num:
                endIndex = mid - 1
            else:
                startIndex = mid + 1
        return -1

    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        for index, value in enumerate(numbers):
            remeaningIndex = self.binarySearch(target - value, numbers, index+1, len(numbers)-1)
            if remeaningIndex >= 0:
                return [index + 1, remeaningIndex + 1]
        return [0, 0]


if __name__=="__main__":
    print(Solution().twoSum([5,25,75]


,100))