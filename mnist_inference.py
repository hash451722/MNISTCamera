import numpy as np
import cv2
import torch

from model import Net


def camara():
    camera_id = 0 
    window_width = 800
    window_height = 600

    cap = cv2.VideoCapture( camera_id )

    # set window size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

    # Detection range
    size = 50
    x1 = int(window_width/2-size)
    x2 = int(window_width/2+size)
    y1 = int(window_height/2-size)
    y2 = int(window_height/2+size)

    # Load trained network
    model = Net()
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    model.eval()

    while True:
        _, frame = cap.read()

        img = frame[y1 : y2, x1 : x2]  # Extraction of detection area
        img = preprocess_image(img)

        pred = model(img)
        pred_number = pred.argmax().item()
        prediction = "Prediction: {}".format(pred_number)
        # print(pred)
        print(prediction)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        frame = draw_graph(frame, pred, pred_number)
        cv2.imshow('frame', frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
    img = cv2.bitwise_not(img)  # reversal
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU) # digitize
    img = cv2.resize(img,(28, 28))
    img = img/255  # 0-255 -> 0.0-1.0
    img = img[np.newaxis, np.newaxis, :, :]  # (28,28) => (1, 1, 28, 28)

    img = torch.tensor(img, dtype=torch.float32)
    return img


def draw_graph(img, pred, pred_number):
    hmax = 130  # px
    w = 28  # px
    interval = 45  # px

    width_window = img.shape[1]
    x0 = int((width_window - (interval*9+w))/2)

    pred_list = pred.to('cpu').detach().numpy().copy()[0]
    probability_list = np.power( np.e, pred_list )

    for num, probability in enumerate(probability_list):
        if pred_number == num:
            col = (0, 0, 255)
        else:
            col = (255, 255, 255)

        img = cv2.rectangle(
            img,
            (x0 + num*interval, 140 - int(hmax*probability)),
            (x0 + num*interval + w, 140),
            color=col,
            thickness=-1
        )

        img = cv2.putText(
            img,
            str(num),
            (x0 + num*interval, 180),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.4,
            color=col,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return img


if __name__ == '__main__':
    camara()
