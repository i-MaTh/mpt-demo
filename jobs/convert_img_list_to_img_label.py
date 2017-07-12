import sys

if __name__ == '__main__':
	if len(sys.argv) == 1:
		print "python .. img_paths"
		sys.exit(0)

	img_list = sys.argv[1]
	with open(img_list, 'r')as f:
		lines = f.readlines()

	suffix = lines[0].strip().split('.')[-1]
	for img_path in lines:
		img_path=img_path.strip()
		anno_path = img_path[:-len(suffix)] + "h5"
		print img_path, anno_path
