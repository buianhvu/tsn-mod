file_in = "/media/data/vuba/tsn/mmaction/data/nucla/annotations/nucla_rgb_val_split_1.txt"
file_out = "/media/data/vuba/tsn/mmaction/data/nucla/annotations/testlist01.txt"

in_f = open(file_in)
out_f = open(file_out, "w")

lines = in_f.readlines()
for line in lines:
    segs = line.split(" ")
    name = segs[0].split('/')[-1]
    sub_folder = name.split('_')[0]
    class_ind = sub_folder.split('a')[-1]
    class_ind = int(class_ind)
    # print(name + " " + sub_folder)
    text = sub_folder + "/" + name + ".avi" + ' ' + str(class_ind) + "\n"
    print(text)
    out_f.write(text)

in_f.close
out_f.close

