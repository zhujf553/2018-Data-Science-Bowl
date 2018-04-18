import utils
import model as modellib
from tqdm import tqdm
import numpy as np
    

class my_validation():
    def __init__(self, val_images, val_masks, inference_config):
        self.mAPs = []
        self.val_images = val_images
        self.val_masks = val_masks
        self.inference_config = inference_config
    
    def validate(self, MODEL_DIR, path=None, logs={}):
    
        model2 = modellib.MaskRCNN(mode="inference", 
                          config=self.inference_config,
                          model_dir=MODEL_DIR)
        if path is not None:
            model_path = path
        else:
            model_path = model2.find_last()[1]
        assert model_path != "", "Provide path to trained weights"
        model2.load_weights(model_path, by_name=True)

        APs = []
        for ix in tqdm(range(len(self.val_images))):
            true_masks = self.val_masks[ix]
            results = model2.detect([self.val_images[ix]], verbose=0)
            r = results[0]
            pred_masks = r['masks']
            AP = utils.compute_map_nuclei(true_masks, pred_masks)
            APs.append(AP)
        mAP = np.mean(APs)
        print("============   mAP = " + str(mAP))
        self.mAPs.append(mAP)
        return mAP
    
    def plot(self):
        plt.plot(self.mAPs)
        plt.ylabel('score')
        plt.show()
