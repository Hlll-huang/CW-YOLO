import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"]=["Times New Roman"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

if __name__ == '__main__':

    #准备绘制数据
    #x = ["Mon", "Tues", "Wed", "Thur", "Fri","Sat","Sun"]
    #y = [20, 40, 35, 55, 42, 80, 50]
    """# yolov5
    x1 = [8.7, 11.1, 13.7, 16.0]
    y1 = [36.7, 44.5, 48.2, 50.4]
    # yolox
    x2 = [9.8, 12.3, 14.5, 17.3]
    y2 = [39.6, 46.4, 50.0, 51.2]"""

    # ASFF
    x1 = ["mAP", "AP50", "AP60", "AP70", "AP80", "AP90"]
    y1 = [54.46, 78.52, 73.97, 66.00, 51.58, 22.90]
    y2 = [54.52, 78.82, 74.34, 66.05, 51.38, 23.11]
    y3 = [54.25, 78.95, 74.42, 66.45, 51.51, 21.13]



    # "g" 表示红色，marksize用来设置'D'菱形的大小
    plt.plot(x1, y1, "k-.", marker='*', markersize=8, label="beta=1")
    plt.plot(x1, y2, "m-.", marker='p', markersize=8, label="beta=0.9")
    plt.plot(x1, y3, "b-.", marker='v', markersize=8, label="beta=0.8")
    #plt.plot(x4, y4, "g-.", marker='>', markersize=8, label="PP-YOLOv2")
    #plt.plot(x5, y5, "y-.", marker='<', markersize=8, label="YOLOv5")
    #plt.plot(x6, y6, "c-.", marker='^', markersize=8, label="YOLOX")
    #plt.plot(x7, y7, "r-.", marker='o', markersize=8, label="FF-YOLO(ours)")


    #绘制坐标轴标签
    #plt.xlabel("inference times(ms)", fontname="Times New Roman")
    plt.ylabel("value(%)", fontname="Times New Roman")
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
    plt.savefig(r"runs/algorithm-performance/2.jpg")
    plt.show()