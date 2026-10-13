from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'C:\Users\cly\Desktop\UD-LDB\ultralytics\cfg\models\v8\yolov8.yaml')

    model.train(data=r'C:\Users\cly\Desktop\UD-LDB\ultralytics\cfg\datasets\A_my_data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,  
                batch=4,
                close_mosaic=0,
                workers=0,
                seed=0,
                device='',
                optimizer='SGD', 
                amp=True, 
                # amp=False,
                project='runs',
                name='exp',
                # resume='True'
                )
