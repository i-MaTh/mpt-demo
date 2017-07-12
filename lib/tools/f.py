with open("create_net.py",'r') as f:
    lines=f.readlines()

lstall=[]
for line in lines:
    lst=line.strip().split()
    newlst=[]
    for w in lst:
        if "net." in w:
            l=w.split("net.")
            if ","==w[-1] or ")"==w[-1]:
                s=l[0]+"net[prefix+'"+l[1][:-1]+"']"+l[1][-1]
            else:
                s=l[0]+"net[prefix+'"+l[1][:-1]+"']"
            newlst.append(s)
        else:
            newlst.append(w)
    lstall.append(' '.join(newlst))
print '\n'.join(lstall)

