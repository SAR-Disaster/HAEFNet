import os
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from torch.utils.data import Dataset


class MultiModalRSDataset(Dataset):
           

    def __init__(
        self,
        root_dir,
        file_list,
        modalities,
        transform=None,
        stage="train",
        target_size=None,
        require_label=True,
    ):
      
        self.root_dir = root_dir
        self.modalities = modalities
        self.transform = transform
        self.stage = stage
        self.target_size = target_size
        self.require_label = False if stage == "test" else require_label

                                        
        self.datalist = self._load_ids(file_list)

                                  
        self._verify_dataset()

    def _load_ids(self, file_list):
        if isinstance(file_list, str):
            with open(file_list, "r") as f:
                return [line.strip() for line in f if line.strip()]
        if isinstance(file_list, (list, tuple)):
            return [str(x).strip() for x in file_list if str(x).strip()]
        raise ValueError("file_list must be a path string or a list/tuple of IDs.")

    def _verify_dataset(self):
                                                                    
        valid_samples = []
        for file_id in self.datalist:
            is_valid = True
                      
            for modality in self.modalities:
                filepath = os.path.join(self.root_dir, modality, f"{file_id}.tif")
                if not os.path.exists(filepath):
                    print(f"Warning: {filepath} not found, skipping sample {file_id}")
                    is_valid = False
                    break
                try:
                    with rasterio.open(filepath) as src:
                        if modality == "rgb" and src.count < 3:
                            print(f"Warning: {filepath} has {src.count} bands, expected 3")
                            is_valid = False
                except Exception as e:
                    print(f"Warning: Failed to read {filepath}: {e}")
                    is_valid = False
                    break

                             
            if self.require_label:
                label_path = os.path.join(self.root_dir, "label", f"{file_id}.tif")
                if not os.path.exists(label_path):
                    print(f"Warning: {label_path} not found, skipping {file_id}")
                    is_valid = False

            if is_valid:
                valid_samples.append(file_id)

        self.datalist = valid_samples

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        file_id = self.datalist[idx]
        images = []
        meta = {"file_id": file_id, "modalities": {}}

        try:
            for modality in self.modalities:
                filepath = os.path.join(self.root_dir, modality, f"{file_id}.tif")
                img = self.read_image(filepath, modality)
                if img is None:
                    raise ValueError(f"Failed to load {modality} for {file_id}")
                images.append(img)
                meta["modalities"][modality] = filepath

            label = None
            if self.require_label:
                label_path = os.path.join(self.root_dir, "label", f"{file_id}.tif")
                with rasterio.open(label_path) as src:
                    label_data = src.read(1)                     

                                     
                    label_np = np.array(label_data, dtype=np.int64, copy=True)

                              
                    label_np = np.ascontiguousarray(label_np)

                                                      
                    label = torch.tensor(label_np, dtype=torch.long)
                    meta["label"] = label_path

                            
            if self.target_size is not None:
                h_t, w_t = self.target_size
                resized = []
                for img in images:
                    resized.append(
                        F.interpolate(img.unsqueeze(0), size=(h_t, w_t), mode="bilinear", align_corners=False).squeeze(
                            0
                        )
                    )
                images = resized
                if label is not None:
                    label = (
                        F.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=(h_t, w_t), mode="nearest")
                        .squeeze(0)
                        .squeeze(0)
                        .long()
                    )

                    
            if self.transform:
                sample = {m: img for m, img in zip(self.modalities, images)}
                if label is not None:
                    sample["label"] = label
                sample = self.transform(sample)
                images = [sample[m] for m in self.modalities]
                label = sample.get("label", None)

                                  
            if label is None:
                label = torch.tensor(0, dtype=torch.long)
            return images, label, meta
        except Exception as e:
            print(f"Error processing sample {file_id}: {str(e)}")
            raise e

    def read_image(self, filepath, modality):
                   
        with rasterio.open(filepath) as src:
                      
            num_bands = src.count

            if modality == "rgb":
                if num_bands >= 3:
                                 
                    img = src.read([1, 2, 3])                        
                else:
                                
                    img = src.read(1)         
            elif modality == "insar_vel" or modality == "insar_phase" or modality == "dem":
                img = src.read(1)                                         
            else:
                img = src.read(1)             

                                      
            img = np.array(img, dtype=np.float32, copy=True)
            img = np.ascontiguousarray(img)
                                                 

                                              
            img = torch.tensor(img, dtype=torch.float32)

                               
            if img.dim() == 2:                                    
                img = img.unsqueeze(0)
                img = torch.cat([img, img, img], dim=0)
            elif img.dim() == 3 and img.shape[0] == 1:                         
                img = torch.cat([img, img, img], dim=0)

            return img

    def set_stage(self, stage):
                            
        self.stage = stage
