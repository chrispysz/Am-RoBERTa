with open("./data/f0/red_full.txt", 'r', encoding="utf8") as in_f:
    with open("./data/f0/red_full_processed.txt", 'w') as out_f:
        filedata = in_f.read()
        split_filedata = filedata.split('\n')

        label_lines=0
        line_counter=0

        for line in split_filedata:
            if line.startswith('>'):
                label_lines+=1

            if label_lines == 2 and not line.startswith('>'):
                out_f.write(line + '\n')
                label_lines = 0
            elif label_lines == 2:
                out_f.write(line + '\n')
            elif label_lines == 1 and not line.startswith('>'):
                label_lines = 0


            line_counter+=1

            if line_counter % 100000 == 0:
                print(str(line_counter/len(split_filedata)*100)+"%")  

