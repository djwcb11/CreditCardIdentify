import cv2 as cv

def SortContours(Contours,method="left-to-right"):
    reverse=False
    i=0
    if method=="right-to-left" or method=="bottom-to-top":
        reverse=True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    BoundingBoxes=[cv.boundingRect(c) for c in Contours]
    (Contours,BoundingBoxes)=zip(*sorted(zip(Contours,BoundingBoxes),key=lambda b:b[1][i],reverse=reverse))
    return Contours,BoundingBoxes


def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv.resize(image, dim, interpolation=inter)
    return resized

