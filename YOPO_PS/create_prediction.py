from model import YOPO_PS

if __name__ == '__main__':
    hej = YOPO_PS('/home/eiman/data/outputs/detect/train22/weights/best.pt')
    hej.predict('/home/eiman/data/all/multiclass/test/images/new_test_004.JPG', save=True, show_labels=False, save_txt=True, max_det=1500)
    #hej.val()