class Solution(object):
    def productExceptSelf(self, nums):
        product = 1
        zeros = set()
        arr = [0] * len(nums)
        for i in range(len(nums)):
            if nums[i] == 0:
                zeros.add(i)
            else:
                product *= nums[i]
        if len(zeros) > 1:
            return [0] * len(nums)
        for j in range(len(nums)):
            if len(zeros) > 0:
                if nums[j] == 0:
                    arr[j] = product
                else:
                    arr[j] = 0
            else:
                arr[j] = product // nums[j]
        return arr

        """
        :type nums: List[int]
        :rtype: List[int]
        """
if __name__=="__main__":
    print(Solution().productExceptSelf([1,2,3,4]))