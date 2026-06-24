import torch
import json
import urllib.request
import numpy as np

from scipy.ndimage import distance_transform_edt
from concurrent.futures import ThreadPoolExecutor, as_completed
from hdf5storage import loadmat
from pathlib import Path
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from network_tools import get_masks
from network_tools import scale_tensor
from network import MS_Net

def preprocess_data(data, num_scales=2):
  """
  Transforms array to different scales as tensors and returns them as list. 
  """
  # Convert data into torch tensor
  data = torch.as_tensor(data, dtype=torch.float32)

  # Add dimensions to torch tensor
  data = data[None, None, :, :, :]

  # Compute data on different scales
  ds_data = []
  ds_data.append(data)
  for i in range( num_scales-1 ):
    ds_data.append( scale_tensor( ds_data[-1], scale_factor=1/2, mode='nearest') )

  data = ds_data[::-1] # Returns the reversed list (smallest images first)

  return data

def calc_loss(y_pred, y, loss_f, log):

    loss = 0
    y_var = y[-1].var()
    for scale, [y_hat,yi] in enumerate(zip(y_pred, y)):
        loss_scale = loss_f(y_hat,yi)#/y_var
        loss += loss_scale
        prefix = f'scale_{scale}_'
        log[prefix + 'loss'] =  loss_scale.item()

    log['loss'] = loss.item()

    return loss, log

class EarlyStopper:
    """
    Breaks if validation loss does not change anymore, to prevent overfitting. 
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class PorousMediaDataset(Dataset):
    """
    Dataset specified to data at https://web.corral.tacc.utexas.edu/digitalporousmedia/DRP-372/.
    Images and simulations in root_dir.
    In init step for all images the euclidiean distance is calculated.
    Those and their corresponding simulations are preprocessed (scaled).
    Tasks are calculated and everything is loaded to RAM. 
    """
    def __init__(self, root_dir: str, scales: int, transform=None):
        self.root_dir = Path(root_dir)
        self.scales = scales
        self.transform = transform

        self.image_paths = []
        self.sim_image_paths = []
        self.edist_scales = []
        self.masks = []
        self.sim_scales = []

        for dir_path in sorted(self.root_dir.iterdir()):
            if not dir_path.is_dir():
                continue

            simulation_path = dir_path / "LBM.mat"
            image_path = dir_path / f"{dir_path.name}.mat"

            if simulation_path.exists() and image_path.exists():
                self.sim_image_paths.append(str(simulation_path))
                self.image_paths.append(str(image_path))
                
        max_sim = 1e-8
        for sim_path in self.sim_image_paths:
            simulation = loadmat(sim_path)['uz']
            current_max = np.max(np.abs(simulation))
            if current_max > max_sim:
                max_sim = current_max
        
        temp_images = []
        temp_edist = []
        temp_masks = []
        temp_sims = []

        for img_path, sim_path in zip(self.image_paths, self.sim_image_paths):
            image = loadmat(img_path)['bin']
            simulation = loadmat(sim_path)['uz']

            image = image.astype(np.float32)
            image = -1 * image + 1
            edist = distance_transform_edt(image)
            edist = edist / np.amax(edist)
            edist_scale = preprocess_data(edist, self.scales)
            simulation = simulation / max_sim
            sim_scale = preprocess_data(simulation, self.scales)
            masks = get_masks(edist_scale[-1], self.scales)

            temp_images.append(torch.tensor(image, dtype=torch.float32))
            temp_edist.append([e.detach().clone() for e in edist_scale])
            temp_masks.append([m.detach().clone() for m in masks])
            temp_sims.append([s.detach().clone() for s in sim_scale])

        self.images = torch.stack(temp_images)
        num_scales = len(temp_masks[0])
        self.edist_scales = [
            torch.stack([sample[s] for sample in temp_edist]) for s in range(num_scales)
        ]
        self.masks = [
            torch.stack([sample[s] for sample in temp_masks]) for s in range(num_scales)
        ]
        self.sim_scales = [
            torch.stack([sample[s] for sample in temp_sims]) for s in range(num_scales)
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.images[idx]
        edist_sample = [scale[idx] for scale in self.edist_scales]
        mask_sample = [scale[idx] for scale in self.masks]
        simulation = [scale[idx] for scale in self.sim_scales]

        if self.transform is not None:
            image = self.transform(image)

        return image, edist_sample, mask_sample, simulation

def download_data(root_dir: str, max_workers: int = 5):
    """
    Downloads all 256x256x256 images with their LBM simulations from "https://web.corral.tacc.utexas.edu/digitalporousmedia/DRP-372/" (11.3 GB). 
    Metadata file "DRP-372_metadata.json" necessary. 
    """

    base_url = "https://web.corral.tacc.utexas.edu/digitalporousmedia/DRP-372/"
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)

    with open('DRP-372_metadata.json') as f:
        metadata = json.load(f)

    filtered_data = [node for node in metadata["nodes"] if node.get("label") and "Single Phase" in node["label"] and "480" not in node["label"]]

    download_tasks = []

    for node in filtered_data:
        value = node.get("value", {})
        try:
            name = value["name"].split(" ")[2]
        except (KeyError, IndexError):
            continue

        lbm_file_obj = next((f for f in value.get("fileObjs", []) if f.get("name") == "LBM.mat"), None)
        bin_image = next((node for node in metadata["nodes"] if node.get("label") and name == node["label"]), None)
        if not lbm_file_obj or not bin_image:
            continue

        bin_value = bin_image.get("value", {})
        mat_filename = f"{name}.mat"
        mat_file_obj = next((f for f in bin_value.get("fileObjs", []) if f.get("name") == mat_filename), None)

        if mat_file_obj:
            target_dir = root_path / name
            target_dir.mkdir(exist_ok=True)

            safe_path_lbm = lbm_file_obj["path"].replace(" ", "%20")
            download_tasks.append((f'{base_url}{safe_path_lbm}', target_dir / "LBM.mat"))

            download_tasks.append((f'{base_url}{mat_file_obj["path"]}', target_dir / mat_filename))

    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, url, path): path for url, path in download_tasks}

        for future in as_completed(futures):
            completed_count += 1
            status = future.result()
            print(f"[{completed_count}] {status}")


def download_file(url: str, target_path: Path) -> str:
    """
    Download single file to obtain concurrency. 
    """
    if target_path.exists():
        return f"Skipped (already exists): {target_path.name}"

    try:
        urllib.request.urlretrieve(url, target_path)
        return f"Success: {target_path.name}"
    except Exception as e:
        return f"ERROR downloading {target_path.name}: {e}"

def train(model: MS_Net,
          train_dataloader, val_dataloader,
          optimizer, scheduler, num_epochs, loss_function, accumulation_steps, early_stopper: EarlyStopper=None):
    """
    Training of MS_Net with mini batch gradient descent. 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    loss_logs = []
    val_loss_logs = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for batch_idx, (_, image_scales, masks, simulation) in enumerate(train_dataloader):

            x_sample = [scale.squeeze(0).to(device) for scale in image_scales]
            mask_sample = [scale.squeeze(0).to(device) for scale in masks]
            y_sample = [scale.squeeze(0).to(device) for scale in simulation]

            y_hat = model(x_sample, mask_sample)

            batch_loss, _ = calc_loss(y_hat, y_sample, loss_function, {})
            batch_loss.backward()
            
            epoch_loss += batch_loss.item()            

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

        # scheduler.step(epoch_loss)

        if early_stopper is not None:
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_idx, (_, image_scales, masks, simulation) in enumerate(val_dataloader):
                    x_sample = [scale.squeeze(0).to(device) for scale in image_scales]
                    mask_sample = [scale.squeeze(0).to(device) for scale in masks]
                    y_sample = [scale.squeeze(0).to(device) for scale in simulation]

                    y_hat = model(x_sample, mask_sample)
                    loss, _ = calc_loss(y_hat, y_sample, loss_function, {})
                    val_loss += loss.item()

            arg_val_loss = val_loss / len(val_dataloader)
            if early_stopper.early_stop(arg_val_loss):
                torch.save(model.state_dict(), 'savedModels/model_weights.pth')
                break

        if epoch % 100 == 0:
            loss_logs.append(epoch_loss)

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_idx, (_, image_scales, masks, simulation) in enumerate(val_dataloader):
                    x_sample = [scale.squeeze(0).to(device) for scale in image_scales]
                    mask_sample = [scale.squeeze(0).to(device) for scale in masks]
                    y_sample = [scale.squeeze(0).to(device) for scale in simulation]

                    y_hat = model(x_sample, mask_sample)
                    loss, _ = calc_loss(y_hat, y_sample, loss_function, {})
                    val_loss += loss.item()

            arg_val_loss = val_loss / len(val_dataloader)
            val_loss_logs.append(arg_val_loss)
            print(f'Epoch {epoch + 1}/{epochs} | Val Loss: {arg_val_loss}')

    torch.save(model.state_dict(), 'savedModels/model_weights.pth')
    return loss_logs, val_loss_logs

if __name__ == '__main__':
    net = MS_Net(
                num_scales   := 4,   # num of trainable convNets
                num_features  = 1,   # input features (Euclidean distance, etc)
                num_filters   = 2,   # num of kernels on each layer of the finest model (most expensive)
                summary       = False # print the model summary
    )
    # download_data("data")     

    data = PorousMediaDataset("data", num_scales)
    train_data, test_data = torch.utils.data.random_split(data, [0.8, 0.2])

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    learning_rate = 1e-4
    epochs = 2500
    EFFECTIVE_BATCH_SIZE = 4
    optimizer = Adam(net.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, threshold=1e-6, min_lr=1e-6, patience=25)
    early_stopper = EarlyStopper(5, 1e-4)
    loss_f = nn.MSELoss()
    loss_logs, val_logs = train(net, train_loader, test_loader, optimizer, scheduler, epochs, loss_f, EFFECTIVE_BATCH_SIZE, early_stopper)