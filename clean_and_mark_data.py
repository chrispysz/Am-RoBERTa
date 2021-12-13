import pandas as pd

data_fold = 'variable_length_pdbfull/f6'


def clean_and_mark(data_id, data_type, data_fold):
    raw_path = './data/'+data_fold+'/'+data_id+'_'+data_type+'.txt'
    processed_txt_path = './data/'+data_fold+'/'+data_id+'_'+data_type+'_prepared.txt'
    processed_csv_path = './data/'+data_fold+'/'+data_id+'_'+data_type+'_prepared.csv'

    with open(raw_path, 'r', encoding="utf8") as in_f:
        with open(processed_txt_path, 'w') as out_f:
            print('Preprocessing: '+data_id+'_'+data_type+'...')
            filedata = in_f.read()

            filedata = filedata.replace('▁', '')
            filedata = filedata.replace(' ', '')
            filedata = filedata.replace('\n\n', '\n')
            split_filedata = filedata.split('\n')

            i = 0
            for line in split_filedata:
                i += 1
                if(line != '' and not ">" in line):
                    if (data_id == 'neg'):
                        line = line + ',0'
                    else:
                        line = line + ',1'
                    out_f.write(line+'\n')
    print('Saving as .csv...')
    ftc = pd.read_csv(processed_txt_path)
    ftc.to_csv(processed_csv_path, index=False)
    print('Finished\n--------------------')


clean_and_mark('neg', 'trn', data_fold)
clean_and_mark('neg', 'val', data_fold)
#clean_and_mark('neg', 'tst', data_fold)
#clean_and_mark('bass', 'trn', data_fold)
#clean_and_mark('bass', 'val', data_fold)
#clean_and_mark('hets', 'tst', data_fold)