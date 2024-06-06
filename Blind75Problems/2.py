class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        maxDiffernce = 0

        minLst = [0] * len(prices)
        maxLst = [0] * len(prices)
        if prices == None or len(prices) < 1:
            return 0
        min_ = prices[0]
        max_ = prices[-1]
        for index, value in enumerate(prices):
            if min_ > prices[index]:
                min_ = prices[index]
            if max_ < prices[(index+1) * -1]:
                max_ = prices[(index+1) * -1]
            maxLst[(index+1) * -1] = max_
            minLst[index] = min_

        for i in range(len(prices)):
            if (maxLst[i] - minLst[i]) > maxDiffernce:
                maxDiffernce = maxLst[i] - minLst[i]

        return maxDiffernce
if __name__=="__main__":
    print(Solution().maxProfit([7,1,5,3,6,4]))