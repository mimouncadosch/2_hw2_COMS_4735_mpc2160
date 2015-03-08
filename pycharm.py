import cv2
import numpy as np

img = cv2.imread("../Images/i01.ppm")
cv2.imshow("img", img)
cv2.waitKey(0)

mat = np.zeros((200, 300, 3))
mat = np.uint8(mat)

mat[0:60,0:89,:] = img
# print type(mat)
# print type(img)

cv2.imshow("mat", mat)
cv2.waitKey(0)


# row = 0
# for col in xrange(0, 8):
# 	if col > 0 and col%4 == 0:
# 		row += 1
	
# 	print row*60, (row+1)*60, (col%4)*89, ((col%4)+1)*89

# # row = 0
# # for col in xrange(0, 8):
# #     # Image has index (i+1)
# #     # idx = parse_id(img_list[col])
# #     # img = cv2.imread("../Images/i" + idx + ".ppm")

# #     # print row*60, (row+1)*60, (col*89), (col+1)*89
# #     print col
# #     print row*60, (row+1)*60, (col%4)*89, ((col+1)%4)*89
# #     # montage[row*60:(row+1)*60, (col%4)*89:((col+1)%4)*89,:] = img
# #     if col > 0 and col%4 == 0:
# #         row += 1
# """
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# """

# #l = np.matrix([[-10,50, 2],[30,40, -50]])
# #print l.shape
# #step = 2
# #rng = (np.max(l) - np.min(l))/50
# """
# hist, bins = np.histogram(l, bins=10)

# width = (bins[1] - bins[0]) * 0.5
# center = (bins[:-1] + bins[1:]) / 2

# #plt.bar(center, hist, align='center', width=width, color='m')
# #plt.show()
# print hist.shape
# print bins.size
# """
# """
# cv2.imshow("l", l)
# cv2.waitKey(0)
# hist = cv2.calcHist([l], [0], None, [3], [0, 3])
# print hist
# """
# #hist1 = cv2.calcHist([img1], [0, 1, 2], mask, [b, b, b], [0, 256, 0, 256, 0, 256])
# # hist = np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
# # plt.figure()
# # plt.plot(hist)
# # plt.show()


# # img1 = cv2.imread("/Users/mimoun/haim.jpeg")
# # bw = np.zeros((img1.shape[0], img1.shape[1]), dtype='uint8')
# #
# # c = cv2.split(img1)
# #
# # cv2.add(c[0], c[1], bw)
# # cv2.add(bw, c[2], bw)
# # bw = bw / 3
# #
# # kernel = np.matrix([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
# #
# # #kernel = np.matrix([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
# #
# # bw1 = cv2.filter2D(bw, -1, kernel)
# #
# # bw2 = cv2.Laplacian(bw, -1)
# #
# # cv2.imshow("bw", bw1)
# # cv2.waitKey(0)
# #
# # cv2.imshow("bw", bw2)
# # cv2.waitKey(0)


# #img1 = cv2.imread("/Users/mimoun/haim.jpeg", 0)
# #cv2.waitKey(0)
# # img2 = cv2.imread("../Images/i02.ppm")
# # img3 = cv2.imread("../Images/i03.ppm")
# #
# # row = np.hstack((img1, img2))
# # row = np.hstack((row, img3))
# # cv2.imshow("row", row)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# # a = np.zeros((20, 89, 3))
# # cv2.putText(a, "01", (0, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,(255,255,255),0)
# # cv2.imshow("a", a)
# # cv2.waitKey(0)
