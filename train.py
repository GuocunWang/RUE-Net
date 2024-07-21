import sys
from torch.nn import Module
import torchvision
from torchvision import transforms
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import wandb
import argparse
from dataclasses import dataclass
from tqdm.autonotebook import tqdm, trange
from dataloader import RUE_Net_DataSet
from tensorboardX import SummaryWriter
from metrics_calculation import *
from RUE_Net_att_2h import *
from RUE_Net_loss import *
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


__all__ = [
    "Trainer",
    "setup",
    "training",
]


@dataclass
class Trainer:
    model: Module
    opt: torch.optim.Optimizer
    loss: Module
    warmup_epochs: int = 5

    def save_best_ssim_weights(self, config):
        torch.save(self.model, config.snapshots_folder + 'best_ssim_weights.ckpt')

    
    def save_best_psnr_weights(self, config):
        torch.save(self.model, config.snapshots_folder + 'best_psnr_weights.ckpt')


    @torch.enable_grad()
    def train(self, train_dataloader, config, test_dataloader, writer, cos_scheduler, warmup_scheduler):
        device = config.device
        primary_loss_lst = []
        vgg_loss_lst = []
        ssim_loss_lst = []
        total_loss_lst = []
        train_lr_lst = []
        loss0_lst = []
        loss2_lst = []
 

        best_weights = [0, 0]
        
        
        for epoch in trange(0, config.num_epochs, desc=f"[Full Loop]", leave=True):
            primary_loss_tmp = 0
            vgg_loss_tmp = 0
            total_loss_tmp = 0
            ssim_loss_tmp = 0
            loss0_tmp = 0
            loss2_tmp = 0
            
            batch_count = 0
            total_batches = len(train_dataloader)
            self.model.train()
            for inp, label, gray, gx, gy, _ in tqdm(train_dataloader, desc="[Train]", leave=False):
                inp = inp.to(device)
                label = label.to(device)
                gray = gray.to(device)
                gx = gx.to(device)
                gy = gy.to(device)

                self.opt.zero_grad()
                out, out2 = self.model(inp, gray, gx, gy)
                
                # Resize label to half its original size
                original_size = label.shape[-2:]  # Get the original height and width
                new_size = (original_size[0] // 2, original_size[1] // 2)
                resized_label = F.interpolate(label, size=new_size, mode='bilinear', align_corners=False)
                
                
                loss0, mse_loss0, vgg_loss0, ssim_loss0 = self.loss(out, label)
                loss2, mse_loss2, vgg_loss2, ssim_loss2 = self.loss(out2, resized_label)
    
                           
                loss = loss0 + 0.3*loss2
                mse_loss = mse_loss0 + 0.3*mse_loss2
                vgg_loss = vgg_loss0 + 0.3*vgg_loss2
                ssim_loss = ssim_loss0 + 0.3*ssim_loss2
                loss.backward()
                self.opt.step()
                
                
                loss0_tmp += loss0.item()
                loss2_tmp += loss2.item()
                primary_loss_tmp += mse_loss.item()
                ssim_loss_tmp += ssim_loss.item()
                vgg_loss_tmp += vgg_loss.item()
                total_loss_tmp += loss.item()
                batch_count += 1

            if epoch < self.warmup_epochs:
                warmup_scheduler.step()
                train_lr_lst.append(self.opt.param_groups[0]['lr'])
            else:
                cos_scheduler.step()
                train_lr_lst.append(self.opt.param_groups[0]['lr'])
            
            
            loss0_lst.append(loss0_tmp/len(train_dataloader))
            loss2_lst.append(loss2_tmp/len(train_dataloader))
            total_loss_lst.append(total_loss_tmp/len(train_dataloader))
            vgg_loss_lst.append(vgg_loss_tmp/len(train_dataloader))
            primary_loss_lst.append(primary_loss_tmp/len(train_dataloader))
            ssim_loss_lst.append(ssim_loss_tmp/len(train_dataloader))

            
            if (config.test == True) & (epoch % config.eval_steps == 0):
                SSIM, PSNR = self.eval(config, test_dataloader, self.model)
                writer.add_scalar("[Test] SSIM", np.mean(SSIM), epoch)
                writer.add_scalar("[Test] PSNR", np.mean(PSNR), epoch)
                avg_ssim = np.mean(SSIM)
                avg_psnr = np.mean(PSNR)
                if avg_ssim > best_weights[0]:
                    best_weights[0] = avg_ssim
                    self.save_best_ssim_weights(config)
                    
                if avg_psnr > best_weights[1]:
                    best_weights[1] = avg_psnr
                    self.save_best_psnr_weights(config)
                    
            writer.add_scalar("[Train] Loss0", loss0_lst[epoch], epoch)
            writer.add_scalar("[Train] Loss2", loss2_lst[epoch], epoch)
            writer.add_scalar("[Train] Total Loss", total_loss_lst[epoch], epoch)
            writer.add_scalar("[Train] Primary Loss", primary_loss_lst[epoch], epoch)
            writer.add_scalar("[Train] ssim Loss", ssim_loss_lst[epoch], epoch)
            writer.add_scalar("[Train] VGG Loss", vgg_loss_lst[epoch], epoch)
            writer.add_scalar("Train Learning Rate", train_lr_lst[epoch], epoch)

            if epoch % config.print_freq == 0:
                print('epoch:[{}]/[{}], image loss:{}, MSE / L1 loss:{}, VGG loss:{}'.format(epoch,config.num_epochs,str(total_loss_lst[epoch]),str(primary_loss_lst[epoch]),str(vgg_loss_lst[epoch])))


            if not os.path.exists(config.snapshots_folder):
                os.mkdir(config.snapshots_folder)

            if epoch % config.snapshot_freq == 0:
                torch.save(self.model, config.snapshots_folder + 'model_epoch_{}.ckpt'.format(epoch))

    @torch.no_grad()
    def eval(self, config, test_dataloader, test_model):
        test_model.eval()
        for img, _, gray, gx, gy, name in test_dataloader:
            with torch.no_grad():
                device = config.device
                img = img.to(device)
                gray = gray.to(device)
                gx = gx.to(device)
                gy = gy.to(device)
                generate_img, _ = test_model(img, gray, gx, gy)
                torchvision.utils.save_image(generate_img, config.output_images_path + name[0])
        SSIM_measures, PSNR_measures = calculate_metrics_ssim_psnr(config.output_images_path,config.GTr_test_images_path)
        # UIQM_measures = calculate_UIQM(config.output_images_path)
        # return UIQM_measures, SSIM_measures, PSNR_measures
        return SSIM_measures, PSNR_measures

def setup(config):
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"
        
    log_folder_name = f"{config.model_name}"
    writer = SummaryWriter(log_dir=config.snapshots_folder)
    model = RUE_Net_att().to(config.device)
    transform = transforms.Compose([transforms.Resize((config.resize,config.resize)),transforms.ToTensor()])
    train_dataset = RUE_Net_DataSet(config.input_images_path,config.label_images_path,transform, True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = config.train_batch_size, shuffle=True)
    print("Train Dataset Reading Completed.")

    loss = RUENet_loss()
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=config.num_epochs - Trainer.warmup_epochs, eta_min=1e-5)
    warmup_scheduler = lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: epoch / Trainer.warmup_epochs)
    trainer = Trainer(model, opt, loss)

    if config.test:
        test_dataset = RUE_Net_DataSet(config.test_images_path, None, transform, False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)
        print("Test Dataset Reading Completed.")
        return train_dataloader, test_dataloader, model, trainer, writer, scheduler, warmup_scheduler
    return train_dataloader, None, model, trainer, writer, scheduler, warmup_scheduler

def training(config):
    ds_train, ds_test, model, trainer, writer, scheduler, warmup_scheduler = setup(config)
    trainer.train(ds_train, config, ds_test, writer, scheduler, warmup_scheduler)
    writer.close()
    print("==================")
    print("Training complete!")
    print("==================")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--input_images_path', type=str, default="",help='path of input images(underwater images)')
    parser.add_argument('--label_images_path', type=str, default="",help='path of label images(clear images)')
    
    parser.add_argument('--test_images_path', type=str, default="",help='path of input images(underwater images) for testing')
    parser.add_argument('--GTr_test_images_path', type = str, default="", help='path of input ground truth images(underwater images) for testing')
   
    parser.add_argument('--test', default=True)
    parser.add_argument('--lr', type=float, default=0.0002)#0.0002
    parser.add_argument('--step_size',type=int,default=50,help="Period of learning rate decay") #50
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=1,help="default : 1")
    parser.add_argument('--test_batch_size', type=int, default=1,help="default : 1")
    parser.add_argument('--resize', type=int, default=256,help="resize images, default:resize images to 256*256")
    parser.add_argument('--cuda_id', type=int, default=0,help="id of cuda device,default:0")
    parser.add_argument('--print_freq', type=int, default=1)    
    parser.add_argument('--snapshot_freq', type=int, default=5)
    parser.add_argument('--snapshots_folder', type=str, default="./snapshots/RUE_Net/")
    parser.add_argument('--output_images_path', type=str, default="./data/RUE_Net/")
    parser.add_argument('--model_name', type=str, default="RUE_Net")
    parser.add_argument('--eval_steps', type=int, default=1)

    config = parser.parse_args()
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.output_images_path):
        os.mkdir(config.output_images_path)
    training(config)