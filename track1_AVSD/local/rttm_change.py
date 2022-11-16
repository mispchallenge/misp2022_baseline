import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--rttm_train', metavar='PATH', required=True,
                        help='rttm_train_path.')
parser.add_argument('--rttm_train_ch', metavar='PATH', required=True,
                        help='rttm_train_ch.')
parser.add_argument('--audio_type', metavar='PATH', required=True,
                        help='audio_type.')
args = parser.parse_args()
f = open(args.rttm_train, "r")
cf = open(args.rttm_train_ch, "w")

line = f.readline()
while line:
    line_1 = line.split()
    for i in range(0, 6):
        line_2 = line_1[0]+" "+line_1[1]+"_"+args.audio_type+"_ch"+str(i)+" "+line_1[2]+" "+line_1[3]+" "+line_1[4]+" "+line_1[5]+" "+line_1[6]+" "+line_1[7]+" "+line_1[8]+" "+line_1[9]+"\n"
        cf.write(str(line_2))
    line = f.readline()
f.close()
cf.close()