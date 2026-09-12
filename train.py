import os
import torch
import configparser
import pandas as pd
import torch.nn.functional as F
from model.flowz import FlowMatcher
from model.layers.EMA import EMA
from utils.Dataset import XYDataset
from utils.utils import set_seed
from transformers import EsmTokenizer, EsmModel
from timm.scheduler.cosine_lr import CosineLRScheduler
from colorama import Fore, Style
from tqdm import tqdm
from torch import device
from torch.cuda import is_available

def main():
    device = torch.device("cuda:0" if is_available() else "cpu")
    conf = configparser.ConfigParser()
    conf.read('config.ini')
    conf_dict = dict(conf.items('FlowTP_conf'))
    set_seed(int(conf_dict['seed']))

    if not os.path.exists('save_model'):
        os.mkdir('save_model')

    tokenizer = EsmTokenizer.from_pretrained(conf_dict['denoiser_esm_model_name'])
    esm2_model = EsmModel.from_pretrained(
        conf_dict['denoiser_esm_model_name'],
        add_pooling_layer=True
    ).to(device)
    esm2_model.eval()

    train_dataset = XYDataset(pd.read_csv('data/split_v2/train.csv'))
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(conf_dict['diffusion_batch_size']),
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,
        num_workers=8
    )

    val_dataset = XYDataset(pd.read_csv('data/split_v2/val.csv'))
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1024,
        pin_memory=True,
        persistent_workers=True,
        num_workers=8
    )

    flow_mlp = [int(i) for i in conf_dict['denoiser_mlp'].split(',')]
    flow_matcher = FlowMatcher(
        conf_dict['denoiser_esm_model_name'],
        int(conf_dict['denoiser_embedding']),
        flow_mlp
    ).to(device)

    optimizer = torch.optim.AdamW(
        flow_matcher.parameters(),
        lr=float(conf_dict['denoiser_lr']),
        weight_decay=float(conf_dict['denoiser_weight_decay'])
    )

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=200_000,
        lr_min=float(conf_dict['min_denoiser_lr']),
        warmup_lr_init=1e-8,
        warmup_t=10_000,
        cycle_limit=1,
        t_in_epochs=False
    )

    ema = EMA(flow_matcher, 0.99)
    ema.register()

    print(
        "The total number of iteration steps is "
        + Fore.LIGHTGREEN_EX
        + "%s" % int(conf_dict['denoiser_epoch'])
        + Style.RESET_ALL
        + "."
    )

    epoch = 0
    x_steps = 1

    if os.path.exists("save_model/checkpoint_flow.pth"):
        checkpoint = torch.load("save_model/checkpoint_flow.pth")
        flow_matcher.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        ema = checkpoint['ema']
        epoch = checkpoint['epoch'] + 1
        x_steps = checkpoint['x_steps']
        print(f"Continuing from epoch {epoch}")
        del checkpoint

    pbar = tqdm(total=len(train_data_loader), miniters=0)
    end_train = False

    for i in range(epoch, int(conf_dict['denoiser_epoch'])):
        if i % 100 == 0:
            set_seed(int(conf_dict['seed']) + i)


        flow_matcher.train()

        for index, datas in enumerate(train_data_loader):
            seq_encode = tokenizer(
                datas['sequences'],
                max_length=int(conf_dict['max_length']) + 2,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            seq_ids_list = seq_encode['input_ids'].to(device)
            train_attention_mask = seq_encode['attention_mask'].to(device)

            with torch.no_grad():
                x1 = esm2_model(
                    seq_ids_list,
                    train_attention_mask
                ).last_hidden_state

            x0 = torch.randn_like(x1, device=device)

            t1 = torch.rand((x1.shape[0],), device=device)
            t2 = (t1 + torch.rand_like(t1) * (1 - t1) * 0.5).clamp(0, 1)
            t = 0.5 * (t1 + t2)

            x_t1 = (1 - t1.view(-1, 1, 1)) * x0 + t1.view(-1, 1, 1) * x1
            x_t2 = (1 - t2.view(-1, 1, 1)) * x0 + t2.view(-1, 1, 1) * x1
            x_t = 0.5 * (x_t1 + x_t2)

            target_v = x1 - x0

            optimizer.zero_grad()

            y_class = torch.tensor(datas['label_class']).to(device)
            y_activity = torch.tensor(datas['label_activity']).to(device)

            pred_v = flow_matcher(
                x_t,
                t,
                y_class=y_class,
                y_activity=y_activity,
                attention_mask=train_attention_mask
            )

            flow_loss = F.mse_loss(pred_v, target_v)
            flow_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                flow_matcher.parameters(),
                max_norm=float(conf_dict['denoiser_clip_grad'])
            )

            optimizer.step()

            if x_steps % 20 == 0:
                pbar.set_description_str(
                    "epoch:%s   " % (epoch + 1)
                    + Fore.LIGHTRED_EX
                    + "flow_loss:%.4f" % flow_loss.item()
                    + Style.RESET_ALL
                )


            ema.update()
            x_steps += 1
            pbar.update()
            scheduler.step_update(x_steps)

        pbar.reset()
        pbar.clear()

        val_flow_loss_list = []

        with torch.no_grad():
            ema.apply_shadow()
            flow_matcher.eval()

            for val_datas in val_data_loader:
                val_seq_encode = tokenizer(
                    val_datas['sequences'],
                    max_length=int(conf_dict['max_length']) + 2,
                    truncation=True,
                    padding='max_length',
                    return_tensors="pt"
                )

                val_seq_ids_list = val_seq_encode['input_ids'].to(device)
                val_attention_mask = val_seq_encode['attention_mask'].to(device)

                val_x1 = esm2_model(
                    val_seq_ids_list,
                    val_attention_mask
                ).last_hidden_state

                val_t = torch.rand((val_x1.shape[0],), device=device)
                val_x0 = torch.randn_like(val_x1, device=device)
                val_x_t = (1 - val_t.view(-1, 1, 1)) * val_x0 + val_t.view(-1, 1, 1) * val_x1
                val_target_v = val_x1 - val_x0

                val_y_class = torch.tensor(val_datas['label_class']).to(device)
                val_y_activity = torch.tensor(val_datas['label_activity']).to(device)

                val_pred_v = flow_matcher(
                    val_x_t,
                    val_t,
                    y_class=val_y_class,
                    y_activity=val_y_activity,
                    attention_mask=val_attention_mask
                )

                val_flow_loss_list.append(
                    F.mse_loss(val_pred_v, val_target_v)
                )

            val_flow_loss = torch.stack(val_flow_loss_list).mean()


            if (epoch + 1) % 20 == 0:
                torch.save(
                    flow_matcher.state_dict(),
                    "save_model/flow_matcher_model.pkl"
                )

            state = {
                'model_state_dict': flow_matcher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema': ema,
                'epoch': epoch,
                'x_steps': x_steps
            }

            torch.save(state, "save_model/checkpoint_flow.pth")
            epoch += 1
            ema.restore()

            if end_train:
                break

    ema.apply_shadow()
    torch.save(
        flow_matcher.state_dict(),
        "save_model/flow_matcher_model.pkl"
    )


if __name__ == '__main__':
    main()
