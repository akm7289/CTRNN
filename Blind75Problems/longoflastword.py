class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s or len(s)==0:
            return 0
        index=len(s)-1
        while index>0 and s[index]==' ':
            index-=1
        if index<0:
            return 0
        counter=0
        for i in range(index,-1,-1):
            if s[i]!=' ':
                counter+=1
            else:
                break
        return counter

if __name__=='__main__':
    print(Solution().lengthOfLastWord("a"))