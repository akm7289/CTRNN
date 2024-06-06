class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        if sum(nums) < target:
            return 0
        leftPointer = 0
        rightPointer = -1
        sum_ = 0

        optimalValue = len(nums)
        while (leftPointer < len(nums)):

            if sum_ < target:
                rightPointer += 1
                if rightPointer < len(nums):
                    sum_ += nums[rightPointer]
                else:
                    sum_ = sum_ - nums[leftPointer]
                    leftPointer += 1


            else:
                if rightPointer - leftPointer+1 < optimalValue:
                    optimalValue = rightPointer - leftPointer+1
                sum_ = sum_ - nums[leftPointer]
                leftPointer += 1

        return optimalValue

if __name__=='__main__':
    print(Solution().minSubArrayLen(7,[2,3,1,2,4,3]))



