class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def recursiveBuilTree(self, preorder, inorder, indexOfRoot,i,j):
        root = TreeNode(inorder[indexOfRoot])
        index = -1
        for i in range(i, indexOfRoot, 1):
            if preorder.index(inorder[i]) > index:
                index = preorder.index(inorder[i])
            else:
                break
        index=inorder.index(preorder[index])
        if indexOfRoot-index<=1:
            root.left=TreeNode(inorder[index])
        else:
            root.left = self.recursiveBuilTree(preorder, inorder, index)
        index = -1
        for i in range(indexOfRoot, j, 1):
            if preorder.index(inorder[i]) > index:
                index = preorder.index(inorder[i])
            else:
                break
        index = inorder.index(preorder[index])
        if abs(indexOfRoot - index) <= 1:
            root.right = TreeNode(inorder[index])
        else:
            root.right = self.recursiveBuilTree(preorder, inorder, index)
        return root

    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if preorder is None or len(preorder) == 0:
            return None
        indexOfRoot = inorder.index(preorder[0])
        resutl=self.recursiveBuilTree(preorder, inorder, indexOfRoot,0,len(inorder))
        return resutl


if __name__=='__main__':
    print(Solution().buildTree(inorder=[9,3,15,20,7],preorder=[3,9,20,15,7]))