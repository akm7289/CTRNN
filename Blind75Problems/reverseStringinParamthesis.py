def solution(inputString):
    result = []
    stack = []
    removedIndices = set
    for i in range(len(inputString)):
        if inputString[i] == '(':
            stack.append(['(', i])
        elif inputString[i] == ')' and len(stack) and stack[-1][0] == '(':
            result.append([stack[-1][1], i])
            stack.pop()
    print(result)
    for i in result:
        inputString = inputString[0:i[0]+1] + inputString[i[1]-1:i[0]:-1] + inputString[i[1]:]

    print(inputString)
    resutlStr = []
    for i in inputString:
        if i == ')' or i == '(':
            continue
        resutlStr.append(i)

    return ''.join(resutlStr)

solution("foo(bar)baz(blim)")