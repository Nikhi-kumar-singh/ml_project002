def outer_function(n):
    def inner_function(n):
        if n<=0 : return n
        print(n)
        return inner_function1(n-2)
    
    def inner_function1(n):
        print(n)
        return inner_function(n+1)

    return inner_function(n)

print(outer_function(10))  # Output: Hello, Alice!
