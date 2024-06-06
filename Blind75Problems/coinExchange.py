import sys

sys.maxint=100000000000


class Solution(object):
    def breadthFirstSearch(self, coins, remainingAmount, bestCoinExchangeMap, number_of_coins):
        if remainingAmount in bestCoinExchangeMap:
            return number_of_coins + bestCoinExchangeMap[remainingAmount]


        result = [sys.maxint] * len(coins)
        for index, value in enumerate(coins):
            if remainingAmount%value==0:
                result[index]=number_of_coins+remainingAmount//value
                bestCoinExchangeMap[remainingAmount]=remainingAmount//value
                break

            if remainingAmount - value in bestCoinExchangeMap:
                result[index] = number_of_coins + bestCoinExchangeMap[remainingAmount - value]
            if remainingAmount - value == 0:
                result[index] = number_of_coins + 1
            elif remainingAmount - value < 0:
                result[index] = sys.maxint
            else:
                result[index] = self.breadthFirstSearch(coins, remainingAmount - value, bestCoinExchangeMap,
                                                        number_of_coins + 1)
        bestCoinExchangeMap[remainingAmount] = min(result)
        return bestCoinExchangeMap[remainingAmount]

    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if coins == None or len(coins) == 0:
            return -1
        if amount == 0:
            return 0
        bestCoinExchangeMap = {}
        maxcoin = len(coins)
        coins.sort(reverse=True)
        for i in coins:
            bestCoinExchangeMap[i] = 1

        result = self.breadthFirstSearch(coins, amount, bestCoinExchangeMap, 0)
        if result == sys.maxint:
            return -1
        else:
            return result


if __name__=="__main__":
    print(Solution().coinChange([406,435,260,178,55]

,2924))