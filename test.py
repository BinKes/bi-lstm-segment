s = open('msr_training.txt', 'r', encoding='cp936').read()
s = s.split('\r\n')

s=s[:10]
print(s)