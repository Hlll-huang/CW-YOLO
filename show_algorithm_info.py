import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"]=["Times New Roman"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

if __name__ == '__main__':

    #准备绘制数据
    #x = ["Mon", "Tues", "Wed", "Thur", "Fri","Sat","Sun"]
    #y = [20, 40, 35, 55, 42, 80, 50]

    # ASFF
    x1 = [16.7, 18.5, 22.0, 34.0]
    y1 = [38.1, 40.6, 42.4, 43.9]

    # EfficientDet
    x2 = [10.2, 13.5, 17.7, 29.0]
    y2 = [33.8, 39.6, 43.0, 45.8]

    # yolov4
    x3 = [10.4, 12.0, 16.1]
    y3 = [41.2, 43.0, 43.5]

    # PP-yolov2
    x4 = [14.5, 19.9]
    y4 = [49.5, 50.3]

    # yolov5s
    x5 = [8.7, 11.1, 13.7, 16.0]
    y5 = [36.7, 44.5, 48.2, 50.4]

    # yolox
    x6 = [9.8, 12.3]
    y6 = [39.6, 46.4]

    # yolois
    x7 = [9.4, 11.7, 14.2, 16.6]
    y7 = [39.7, 46.5, 49.3, 50.7]

    # "g" 表示红色，marksize用来设置'D'菱形的大小
    plt.plot(x1, y1, "k-.", marker='*', markersize=8, label="YOLOv3+ASFF*")
    plt.plot(x2, y2, "m-.", marker='p', markersize=8, label="EfficientDet")
    plt.plot(x3, y3, "b-.", marker='v', markersize=8, label="YOLOv4")
    plt.plot(x4, y4, "g-.", marker='>', markersize=8, label="PP-YOLOv2")
    plt.plot(x5, y5, "y-.", marker='<', markersize=8, label="YOLOv5")
    plt.plot(x6, y6, "c-.", marker='^', markersize=8, label="YOLOX")
    plt.plot(x7, y7, "r-.", marker='o', markersize=8, label="FF-YOLO(ours)")


    #绘制坐标轴标签
    plt.xlabel("inference times(ms)", fontname="Times New Roman")
    plt.ylabel("map(%)", fontname="Times New Roman")
    #plt.title("Ladder diagram of algorithm performance")
    plt.title("MS COCO Object Detection", fontname="Times New Roman", fontweight="bold")
    #显示图例
    plt.legend(loc="lower right")
    #调用 text()在图像上绘制注释文本
    #x1、y1表示文本所处坐标位置，ha参数控制水平对齐方式, va控制垂直对齐方式，str(y1)表示要绘制的文本
    """for x1, y1 in zip(x1, y1):
        plt.text(x1, y1, str(y1), ha='center', va='bottom', fontsize=10)
    for x2, y2 in zip(x2, y2):
        plt.text(x2, y2, str(y2), ha='center', va='bottom', fontsize=10)"""
    #保存图片
    plt.savefig(r"runs/algorithm-performance/3.jpg")
    plt.show()