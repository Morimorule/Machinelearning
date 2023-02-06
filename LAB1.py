for i in range(8):
 if i<4 :
    print ((i+1) * "x")
 else:
    print((9-i) * "x")

somme = "n45as29@#8ss6"

sum = 0
for char in somme:
    if char.isdigit():
        sum =sum + int(char)
print("the sum is{0}".format(sum)) #this format was suggested by my IDE

def int_to_binary(integer):
    binary_string = ''
    while(integer > 0):
        digit = integer % 2
        binary_string += str(digit)
        integer = integer // 2
    binary_string = binary_string[::-1]
    return binary_string

print(int_to_binary(47))

"""def fibonaci(digit):
    lst=[]
    fibo=0
    if n<=0:
        return n
    elif n==1:
        return 1
    else:
        return fibo