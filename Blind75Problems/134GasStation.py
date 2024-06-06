class Solution(object):
    def findMaxSubArray(self, differenceArray):
        minIndexGlobal = 0
        maxIndex = 0
        minIndex=0
        maxSum = 0
        tmpMaxSum = 0
        for i in range(len(differenceArray)):
            tmpMaxSum = tmpMaxSum + differenceArray[i]
            if tmpMaxSum > maxSum:
                maxSum = maxSum
                maxIndex = i
                minIndexGlobal = minIndex
            elif tmpMaxSum < 0:
                tmpMaxSum = 0
                minIndex = i+1
        return minIndexGlobal

    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        if sum(gas) >= sum(cost):
            differenceArray = [0] * len(gas) * 2
            for i in range(len(cost)):
                differenceArray[i] = gas[i] - cost[i]
                differenceArray[i + len(gas)] = gas[i] - cost[i]

            Index = self.findMaxSubArray(differenceArray)

            return Index % len(gas)
        else:
            return -1

if __name__=='__main__':
    print(Solution().canCompleteCircuit([5,1,2,3,4],[4,4,1,5,1]))
