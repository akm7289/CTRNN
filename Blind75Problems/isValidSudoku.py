class Solution(object):
    def checkEachRow(self, board):
        setn = set()
        for i in range(len(board)):

            setn.clear()
            for j in range(len(board[0])):
                if board[i][j] == ".":
                    continue
                elif board[i][j] in setn:
                    return False
                else:
                    setn.add(board[i][j])
        return True

    def checkEachColumn(self, board):
        setn= set()
        for i in range(len(board[0])):
            setn.clear()
            for j in range(len(board)):
                if board[j][i] == ".":
                    continue
                elif board[j][i] in setn:
                    return False
                else:
                    setn.add(board[j][i])
        return True

    def checkEachBox(self, board):
        setn = set()
        for k in range(3):
            for m in range(3):
                setn.clear()
                for i in range(3):

                    for j in range(3):

                        if board[k * 3 + i][m * 3 + j] == ".":
                            continue
                        elif board[k * 3 + i][m * 3 + j] in setn:
                            return False
                        else:
                            setn.add(board[k * 3 + i][m * 3 + j])
        return True

    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        print(self.checkEachBox(board))
        return self.checkEachColumn(board) and self.checkEachRow(board) and self.checkEachBox(board)
if __name__=='__main__':
    print(Solution().isValidSudoku([[".",".",".",".","5",".",".","1","."],[".","4",".","3",".",".",".",".","."],[".",".",".",".",".","3",".",".","1"],["8",".",".",".",".",".",".","2","."],[".",".","2",".","7",".",".",".","."],[".","1","5",".",".",".",".",".","."],[".",".",".",".",".","2",".",".","."],[".","2",".","9",".",".",".",".","."],[".",".","4",".",".",".",".",".","."]]))