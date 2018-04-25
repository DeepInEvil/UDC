import os
desc_dir = '/data/dchaudhu/UDC/unix_description/'
files = os.listdir(desc_dir)
unix_desc = open('/data/dchaudhu/UDC/ubuntu_data/desc.csv', 'w')


def get_lines(file_c):
    desc = []
    index_desc = file_c.index('DESCRIPTION\n')
    i = index_desc + 1
    while i < len(file_c):
        line = file_c[i].replace('\n', '')
        i = i + 1
        if line.isupper():
            break
        else:
            desc.append(line)
    return ' '.join([line for line in desc])


for file in files:
    command = file.split('_')[0]
    with open(desc_dir+file, 'r') as f:
        man = f.readlines()
    desc = get_lines(man)
    unix_desc.write(command + '\t' + desc)
    unix_desc.write('\n')

unix_desc.close()
