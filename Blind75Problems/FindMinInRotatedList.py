class Solution(object):
    def findtheMinRec(self, nums, left, right):
        if left < 0 or right >= len(nums):
            return -1
        if left == right:
            return nums[left]
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            return self.findtheMinRec(nums, mid + 1, right)
        else:
            return self.findtheMinRec(nums, left, mid )


    def findMin(self, nums):
        return self.findtheMinRec(nums, 0, len(nums) - 1)
        """
        :type nums: List[int]
        :rtype: int
        """
if __name__=="__main__":
    print(Solution().findMin([5,1,2,3,4]))