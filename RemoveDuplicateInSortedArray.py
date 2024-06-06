class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        arr = []
        if not nums or len(nums) <= 1:
            return nums
        arr.append(nums[0])
        for i in range(0, len(nums), 1):
            if arr[-1] == nums[i]:
                continue
            arr.append(nums[i])
        nums = arr
        return len(arr)


if __name__=='__main__':
    print(Solution().removeDuplicates([1,1,2]))