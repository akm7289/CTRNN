class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        cols = set()
        rows = set()
        if matrix==None or len(matrix)<1:
            return matrix
        n,m=len(matrix),len(matrix[0])
        for i in range(n):
            for j in range(m):
                if matrix[i][j] == 0:
                    cols.add(j)
                    rows.add(i)
        for col in cols:
            for i in range(n):
                matrix[i][col] = 0

        for row in rows:
            for j in range(m):
                matrix[row][j] = 0
        return matrix
if __name__=="__main__":
    print(Solution().setZeroes([[0,1,2,0],[3,4,5,2],[1,3,1,5]]))