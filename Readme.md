## Yolo-Nano in Tensorflow 2.1.0

## Statement
This is a YoloNano implatement in tensorflow 2.1.0.

Origin paper:

## Train
1.Use original yolo train set like,the train images(xxx.jpg) and the label files(xxx.txt) are in the same dir.

The label format is as follow:

```
classes_id x_center y_center width height ##box1 range[0.0-1.0]
classes_id x_center y_center width height ##box2 range[0.0-1.0]
```

Pay attention that the box data should be Normalized.

2.Create a train.txt which contains your all train images with full path,such like:

```
/home/user/coco/train/1.jpg
/home/user/coco/train/2.jpg
/home/user/coco/train/3.jpg
```

Pay attention that the picture name should correspond to the label name.

3.Replace `train_path` in train.py with your own train.txt path and modify the `anchors` with your own data.

4.Run`python3 train.py`.

That's all,so easy isn't it?

## Test

1.Modify the lines below in test.py

```python
anchors = np.array([your anchors data,shape(9,2)],dtype='float32')

test_model.load_weights('your model.h5 here')
img = cv2.imread('your test image here')
```
2.Run `python3 test.py`


