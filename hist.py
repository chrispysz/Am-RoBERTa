import matplotlib.pyplot as plt


with open("./data/PRoBERTa/1/trn_prepared_shuffled.csv", 'r', encoding="utf8") as in_f:
    filedata = in_f.read()
    split_filedata = filedata.split('\n')
    sizes =[]

    for line in split_filedata:
        split_line = line.split(',')[0]
        if len(split_line)>0:
            sizes.append(len(split_line))


plt.hist(sizes, bins=[23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])
plt.grid(True)
plt.xlim(22, 40)
plt.show()