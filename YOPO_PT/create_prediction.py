from model import YOPO_PT

if __name__ == '__main__':
    predictor = YOPO_PT('/home/aleksispi/Projects/master-theses/cbd_2023/runs/mc_pt/mc_pt_05/weights/best.pt')
    predictor.predict('/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/multiclass/test/images/new_test_015.JPG', save=True, show_labels=True, save_txt=True, max_det=1500, conf=0.097)
    #predictor.val(max_det=1500, split='test')