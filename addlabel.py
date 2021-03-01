import os

c=[]


def creat_label(label_file):
    filedir=label_file
    for line in open(filedir):
        a=list(line)
        b=a[0]
        c.append(b)
    return c
creat_label('C:/Users/lenovo/Desktop/ballballle.txt')

def merge_file(signal_folder,target_file):
    meragefiledir=signal_folder
    filenames=os.listdir(meragefiledir)
    filenames.sort(key=lambda x: int(x[10:-4]))
#  filenames.sort(key)
    file=open(target_file,'w')

    i=0
    for filename in filenames:
        filepath=meragefiledir+'/'
        filepath=filepath+filename

        for line in open(filepath):
            file.writelines(c[i]+','+line.strip('\r\n')+'\n')
            print(i)
            i+=1

    file.close()
merge_file('C:/Users/lenovo/Desktop/signal rest','C:/Users/lenovo/Desktop/signal-800.csv')