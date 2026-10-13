from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'C:\Users\cly\Desktop\UD-LDB\runs\DUO re0.2\weights\best.pt')
    model.val(data=r'C:\Users\cly\Desktop\UD-LDB\ultralytics\cfg\datasets\A_my_data.yaml',
              split='val',
              workers=0,
              imgsz=640,
              batch=16,
              project='runs/val',
              name='exp',
              save=True,   
              )
