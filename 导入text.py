A = zero((a,a),dtype = float)

file = open(name, "r", encoding='utf-8', errors='ignore')
lines = file.readlines()
A_row = 0
for line in lines:
    items = line.split('|')
    list = items.strip('\n').split('')
    A[A_row:] = list[0:a]
    A_row+=1

print(A)