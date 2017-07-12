import cv2
import sys
import os

if len(sys.argv) == 1:
	print "python .. img_label[dir] resized_height"
	sys.exit(0)

img_label = sys.argv[1]
if "." in sys.argv[2]:
	new_h = float(sys.argv[2])
	assert new_h < 1 and new_h > 0
else:
	new_h = int(sys.argv[2])

if os.path.isdir(img_label):
	for img in sorted(os.listdir(img_label)):
		if "jpg" not in img and "png" not in img:
			continue
		img_path = img_label + "/" + img
		img = cv2.imread(img_path)
		h, w = img.shape[:2]
		if new_h == -1:
			new_w, new_h = w, h
		if new_h < 1 and new_h > 0:
			new_w = int(w * new_h)
			new_h = int(h * new_h)
		else:
			new_w = int(w * new_h / h)
                new_img = cv2.resize(img, (new_w, new_h))
		cv2.imshow("resized img", new_img)
                key=cv2.waitKey()
		if key == -1 or key== 100:
                        sys.exit(0)
else:
	with open(img_label, 'r') as f:
		try:
			while True:
				img_path = f.next().strip().split()[0]
				print img_path
				img = cv2.imread(img_path)
				h, w = img.shape[:2]
				if new_h == -1:
					new_w, new_h = w, h
				if new_h < 1 and new_h > 0:
					new_w = int(w * new_h)
					new_h = int(h * new_h)
				else:
					new_w = int(w * new_h / h)
				new_img = cv2.resize(img, (new_w, new_h))
				cv2.imshow("resized img", new_img)
                                key= cv2.waitKey()
                                if key==-1 or key== 100:
                                        sys.exit(0)
		except StopIteration:
			pass
