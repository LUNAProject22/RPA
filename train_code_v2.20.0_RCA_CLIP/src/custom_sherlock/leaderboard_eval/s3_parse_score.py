import argparse
import numpy as np

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--f', default='', type=str)
	args = parser.parse_args()
	with open(args.f, "r") as f:
		lines = f.readlines()
	im2txt = []
	txt2im = []
	p_at_1 = []
	model_corr = ""
	loc_metric = ""
	for line in lines:
		if "Model corr" in line:
			model_corr = line
		if "im2txt" in line:
			score = float(line.replace("im2txt: ", ""))
			im2txt.append(score)
		if "txt2im" in line:
			score = float(line.replace("txt2im: ", ""))
			txt2im.append(score)
		if "p_at_1" in line:
			score = float(line.replace("p_at_1: ", ""))
			p_at_1.append(score)
		if "bbox" in line:
			gt_box = line.strip()
	print("Model corr: {}".format(model_corr.strip()))
	print("im2txt 22 Means: {}".format(np.array(im2txt).mean()))
	print("txt2im 22 Means: {}".format(np.array(txt2im).mean()))
	print("p_at_1 22 Means: {}".format(np.array(p_at_1).mean()))
	print("{}".format(gt_box.strip()))


	print("Process Finished")

if __name__ == "__main__":
	main()
