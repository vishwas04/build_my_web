# d={"(13,97,116,348)": [["Results", [2, 0]], ["Sivevar", [0, 0]], ["Proyile", [1, 0]], ["Lalenden", [3, 0]]], "(3,466,367,509)": [["is", [0, 4]], ["Lonact", [0, 3]], ["link", [0, 2]], ["Endban", [0, 0]], ["link", [0, 1]]], "(128,100,343,311)": [["slideshow", [0, 0]]], "(20,15,290,86)": [["Pey", [0, 3]], ["About", [0, 2]], ["Narvan", [0, 0]], ["dogin", [0, 1]]]}
# for i in d:
#     sd=sorted(d[i] , key=lambda k: [k[1][0], k[1][1]])
#     d[i]=sd
# print(d)
from spellchecker import SpellChecker
p=["navbar"]
x="mawter"
spell = SpellChecker()
sx=spell.correction(x)
def per(a,b):
    r="text"
    p=0
    for o in a:
        print("vade")
        if(len(o)<=len(b)):
            off=len(b)-len(o)+1
            for i in range(off):
                c=0
                for match in range(len(o)):
                    if(o[match]==b[i+match]):
                        c=c+1
                if(c/len(o)>p and c>=2):
                    p=c/len(o)
                    r=o
        else:
            off=len(o)-len(b)+1
            for i in range(off):
                c=0
                for match in range(len(b)):
                    if(o[i+match]==b[match]):
                        c=c+1
                print(c)
                if(c/len(o)>p and c>=2):
                    p=c/len(o)
                    r=o

        
    return (r,p)

print(per(p,x))
print(per(p,sx))