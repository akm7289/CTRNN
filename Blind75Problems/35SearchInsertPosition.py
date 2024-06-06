class Solution(object):
    def binarySearch(self, nums, target, startIndex, endIndex):
        middle = (startIndex + endIndex) // 2
        if startIndex < endIndex:
            if nums[middle] == target:
                return middle
            elif nums[middle] > target:
                endIndex = middle - 1
            elif nums[middle] < target:
                startIndex = middle + 1
            return self.binarySearch(nums, target, startIndex, endIndex)
        if startIndex >= endIndex:
            if startIndex>=len(nums):
                return len(nums)
            if startIndex>endIndex:
                return startIndex
            if nums[startIndex] == target:
                return middle
            else:
                if nums[startIndex] > target:
                    return startIndex
                else:
                    return startIndex + 1

    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        return self.binarySearch(nums, target, 0, len(nums))

if __name__=="__main__":
    print(Solution().searchInsert([1,3,5],4))
