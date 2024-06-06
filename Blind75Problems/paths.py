def solution(path):
    stack = []
    path = path.replace('//', '/')
    strings = path.split('/')
    for i in strings:
        if i == '..' and len(stack) > 0:
            stack.pop()
        elif i == '.':
            continue
        else:
            stack.append(i)
    result = '/'.join(stack)
    if path.endswith('/') and not result.endswith('/'):
        result += '/'

    return result

print(solution("/home/a/./x/../b//c/"))