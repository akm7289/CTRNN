class Solution(object):
    def productExceptSelf(self, nums):
        prefix_mul=[1]*(len(nums))
        postfix_mul=[1]* (len(nums))
        for i in range(len(nums)-1):
            prefix_mul[i+1]=prefix_mul[i]*nums[i]
        for i in range(len(nums)-1,0,-1):
            postfix_mul[i-1]=postfix_mul[i]*nums[i]
        result=[1]*(len(nums))
        for i in range(len(nums)):
            result[i]=prefix_mul[i]*postfix_mul[i]

        return result
if __name__=='__main__':
    print(Solution().productExceptSelf([1,2,3,4]))