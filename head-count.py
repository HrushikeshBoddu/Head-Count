from tkinter import *
from tkinter import filedialog
from PIL import Image
import torchvision.transforms as T
import torchvision
import cv2
import torch

main = Tk()
main.title("People Counting System Based on Head Detection using Faster R-CNN")
main.geometry("800x600")
main.config(bg='LightSteelBlue1')

# Loading the Faster R-CNN model to count human heads from images and videos
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def get_prediction(img_path, threshold=0.8):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_classes = [int(i) for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_scores = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_scores.index(x) for x in pred_scores if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]
    head_count = sum([1 for i in pred_classes if i == 1])
    return head_count

def countFromImages():
    filename = filedialog.askopenfilename(initialdir=".", title="Select an Image", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    if filename:
        text.insert(END, f"{filename} loaded\n")
        pathlabel.config(text=f"{filename} loaded")
        head_count = get_prediction(filename, 0.8)
        img = cv2.imread(filename)
        cv2.putText(img, f"Total Head: {head_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)
        cv2.imshow("Image Output", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def countFromVideo():
    filename = filedialog.askopenfilename(initialdir=".", title="Select a Video", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
    if filename:
        text.insert(END, f"{filename} loaded\n")
        pathlabel.config(text=f"{filename} loaded")
        video = cv2.VideoCapture(filename)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            head_count = get_prediction("temp_frame.jpg", 0.8)
            cv2.putText(frame, f"Total Head: {head_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)
            cv2.imshow("Video Output", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

font = ('times', 14, 'bold')
title = Label(main, text='People Counting System Based on Head Detection using Faster R-CNN', bg='DarkGoldenrod1', fg='black', font=font, height=3, width=120)
title.place(x=5, y=5)

font1 = ('times', 13, 'bold')
imageButton = Button(main, text="People Counting from Images", command=countFromImages, font=font1)
imageButton.place(x=50, y=100)

pathlabel = Label(main, bg='brown', fg='white', font=font1)
pathlabel.place(x=480, y=100)

videoButton = Button(main, text="People Counting from Video", command=countFromVideo, font=font1)
videoButton.place(x=50, y=150)

text = Text(main, height=10, width=150, font=('times', 12, 'bold'))
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=200)

main.mainloop()
