class Solution(object):
    def wordPattern(self, pattern, s):
        """
        :type pattern: str
        :type s: str
        :rtype: bool
        """
        letters = list(pattern)
        words = s.split(' ')
        if len(letters) != len(words):
            return False
        lettersWordsMapping = {}
        wordsLettersMapping = {}
        for letter, word in zip(letters, words):
            if letter in lettersWordsMapping:
                if word != lettersWordsMapping[letter]:
                    return False
                else:
                    continue
            elif word in wordsLettersMapping:
                if letter != wordsLettersMapping[word]:
                    return False
                else:
                    continue
            else:
                lettersWordsMapping[letter] = word
                wordsLettersMapping[word] = letter
        return True
if __name__ =='__main__':
    print(Solution().wordPattern('abba',"dog cat cat dog"
))