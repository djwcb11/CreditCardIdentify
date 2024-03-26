from imutils import contours
import cv2 as cv
import numpy as np

import MyUtils
def cv_show(name,img):
   cv.imshow(name, img)
   cv.waitKey(0)
   cv.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    FIRST_NUMBER={
        "3": "American Express",
        "4": "Visa",
        "5": "MasterCard",
        "6": "Discover Card"
    }

    #读入模板图像，转为灰度-》转为二值
    Template=cv.imread("images/ocr_a_reference.png")
    GrayTemplate=cv.cvtColor(Template,cv.COLOR_BGR2GRAY)#转为灰度图
    BinTemplate=cv.threshold(GrayTemplate,127,255,cv.THRESH_BINARY_INV)[1]
    cv_show("BinTemplate",BinTemplate)

    #画出模板图像的轮廓（外轮廓）
    t,contours1,hierarchy=cv.findContours(BinTemplate.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(Template,contours1,-1,(0,0,255),2)
    cv_show("DrawingTemplate",Template)

    #对轮廓进行排序,将数字与轮廓对应
    RefCons=MyUtils.SortContours(contours1,"left-to-right")[0]
    digits={}
    for(i,c) in enumerate(RefCons):
        (x,y,w,h)=cv.boundingRect(c)

        roi = BinTemplate[y:y+h,x:x+h]#感兴趣区域
        roi=cv.resize(roi,(57,88))
        digits[i]=roi


    #模板处理完毕，对图像进行处理
    #先读取并转为灰度图
    Img = cv.imread("images/credit_card_01.png")
    Img=MyUtils.resize(Img,width=300)
    GrayImg = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)  # 转为灰度图

    #通过礼帽操作，突出明亮区域
    RectKernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
    SqKernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    TopHat=cv.morphologyEx(GrayImg,cv.MORPH_TOPHAT,RectKernel)
    cv_show("TopHat",TopHat)

    #Sobel算子，做边缘检测
    GradX=cv.Sobel(TopHat,cv.CV_32F,1,0,ksize=-1)
    GradX=np.absolute(GradX)
    (MinVal,MaxVal)=(np.min(GradX),np.max(GradX))
    GradX=(255*(GradX-MinVal)/(MaxVal-MinVal))
    GradX=GradX.astype("uint8")


    #GradY = cv.Sobel(TopHat, cv.CV_32F, 0, 1)
    #取绝对值

    #AbsGradY=cv.convertScaleAbs(GradY)
    #线性混合
    #Dst=cv.addWeighted(AbsGradX,0.5,AbsGradY,0.5,0)
    #cv_show("dst",Dst)
    cv_show("GradX",GradX)

    #通过闭操作，将数字连在一起
    GradX=cv.morphologyEx(GradX,cv.MORPH_CLOSE,RectKernel)
    cv_show("1",GradX)

    #转成二值
    BinImage=cv.threshold(GradX,0,255,cv.THRESH_OTSU)[1]
    cv_show("BinImage",BinImage)

    #不够模糊，再闭操作一下
    BinImage=cv.morphologyEx(BinImage,cv.MORPH_CLOSE,SqKernel)
    cv_show("BinImage",BinImage)

    #计算轮廓
    t,ImageContours,ImageHierarchy=cv.findContours(BinImage.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    CurImage=Img.copy()
    cv.drawContours(CurImage,ImageContours,-1,(0,0,255),2)
    cv_show("CurImg",CurImage)

    #找出信用卡中数字部分，四个一组共四组
    locs=[]
    for(i,c) in enumerate(ImageContours):
        (x, y, w, h) = cv.boundingRect(c)
        ar = w / float(h)
        #print(ar)

        #保留数字组
        if ar>2.5 and ar<4.0:
            if (w>40 and w<55) and(h>10 and h<20):
                locs.append((x,y,w,h))

    locs=sorted(locs,key=lambda x:x[0])

    #提取每一个数字组
    for (i,(gX,gY,gW,gH)) in enumerate(locs):
        groupOutput = []
        group=GrayImg[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]

        group = cv.threshold(group, 0, 255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        cv_show("group", group)
        group_, digitCnts, hierarchy = cv.findContours(group.copy(), cv.RETR_EXTERNAL,
                                                            cv.CHAIN_APPROX_SIMPLE)
        digitCnts = contours.sort_contours(digitCnts,
                                               method="left-to-right")[0]

        #计算每一个数字组中的每一个数
        for c in digitCnts:
                # 找到当前数值的轮廓，resize成合适的的大小
            (x, y, w, h) = cv.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv.resize(roi, (57, 88))
            cv_show('roi', roi)
            # 在模板中计算每一个得分
            scores = []
            for (digit, digitROI) in digits.items():
                result = cv.matchTemplate(roi, digitROI,cv.TM_CCOEFF)

                (_,score,_,_)=cv.minMaxLoc(result)
                scores.append(score)
            print(scores)
            # 得到最合适的数字
            groupOutput.append(str(np.argmax(scores)))

        #画出来
        cv.rectangle(Img, (gX - 5, gY - 5),
                      (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv.putText(Img, "".join(groupOutput), (gX, gY - 15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    cv_show("Finally", Img)







    


