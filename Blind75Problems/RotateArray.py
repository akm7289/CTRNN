class Solution(object):
    def rotate(self, nums, k):

        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        # arrlength=len(nums)
        # k=k%arrlength
        # for i in range(len(nums)):
        #     results[(k+i)%arrlength]=nums[i]
        # for i in range(arrlength):
        #     nums[i]=results[i]
        # return results

        arrlength = len(nums)
        k = k % arrlength

        for i in range(arrlength // 2):
            nums[i], nums[-(i + 1)] = nums[-(i + 1)], nums[i]
        for j in range(k // 2):
            nums[j], nums[k - j-1] = nums[k - j-1], nums[j]
        for i in range(k, k+(arrlength - k) // 2):
            nums[i], nums[k-i-1] = nums[k-i-1], nums[i]

        return nums
if __name__=='__main__':
    print(Solution().rotate([1,2,3,4,5,6,7]
,3))

