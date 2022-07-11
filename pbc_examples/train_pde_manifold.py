import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pbc_examples.data.ns_fno import NSDataset
from pbc_examples.modules.separation_embedding_modulation_2d_integrate_samepx import \
    SeparationEmbeddingMod2dIntegrateSamePX

from pbc_examples.quick_train import SeparationExperiment


class SeparationExperimentFNO(SeparationExperiment):
    def __init__(self, num_samples, param_dim=None, x_param_dim=None, t_param_dim=None, prefix=None, separate_params=True):
        pl.LightningModule.__init__(self)
        self.prefix = prefix
        self.separate_params = separate_params
        self.model = SeparationEmbeddingMod2dIntegrateSamePX(num_samples)


    def forward(self, input):
        output = self.model(input['x'], input['y'], input['t'], self._get_aux_input(input))
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, weight_decay=1e-8)
        # emb_param_list = ['ex.weight', 'et.weight', 'et0.weight', 'ey.weight']
        # emb_params = [p for n, p in self.model.named_parameters() if n in emb_param_list]
        # other_params = [p for n, p in self.model.named_parameters() if n not in emb_param_list]
        # optimizer = torch.optim.Adam([
        #     {'params': emb_params, 'lr':0.002, 'weight_decay':0},
        #     {'params': other_params, 'lr':0.002, 'weight_decay':0},
        # ], lr=0.002)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        input = train_batch
        output = self.model(input['x'], input['y'], input['t'], self._get_aux_input(input))
        # print(output['u_pred'].shape, input['u'].shape)
        loss = F.mse_loss(output['u_pred'], input['u'])
        return loss

    def _plot_solutions(self, input, output, max_plots=None):
        for batch_idx, out in enumerate(output['u_pred']):
            if max_plots is not None and batch_idx > max_plots:
                break
            u = out
            u_target = input['u'][batch_idx]
            plt.figure(figsize=(len(u), 3))
            for t, ut in enumerate(u):
                plt.subplot(2, len(u), t + 1)
                plt.title('output')
                ims = ut.squeeze().cpu()
                plt.title('output')
                plt.imshow(ims)
            plt.suptitle(batch_idx, fontsize=42)
            # plt.show()

            for tt, utt in enumerate(u_target):
                plt.subplot(2, len(u), len(u) + tt + 1)
                plt.title('output')
                ims = utt.squeeze().cpu()
                plt.title('output')
                plt.imshow(ims)
            plt.tight_layout()
            plt.suptitle(batch_idx, fontsize=42)
            plt.show()

            # if t < gt_img.shape[1]:
            #     plt.subplot(2, ntime_res, ntime_res + t + 1)
            #     plt.title('gt')
            #     plt.imshow(gt_img[b, t, :, :, 0].numpy())


    def validation_step(self, val_batch, batch_idx):
        input = val_batch
        # if isinstance(self.model, EmbeddingModule):
        #     self._plot_swapped_embeddings(input, 2, 12)
        #     plt.show()
        #     self._plot_swapped_embeddings(input, 12, 2)
        #     plt.show()
        x= input['t']
        x = torch.cat([x, x[:, 1:] + x[:, [-1]]], 1)
        x = torch.cat([x, x[:, 1:] + x[:, [-1]]], 1)
        output = self.model(input['x'], input['y'], x, self._get_aux_input(input))
        self._plot_solutions(input, output, max_plots=4)
        plt.close()
        if 'zt' in output and 'px' in output:
            self._plot_latents(input, output)
            plt.show(bbox_inches='tight')

    def val_dataloader(self):
        return val_loader



if __name__ == '__main__':
    import sys
    prefix = sys.argv[1]
    separate_params = False
    n = 100
    batch_size = 10
    timelen = 4
    dataset = NSDataset(n=n,  timelen=timelen,)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=True)

    val_dataset = NSDataset(n=n, timelen=timelen,)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=1)

    model = SeparationExperimentFNO(len(dataset), prefix=prefix)

    # training
    trainer = pl.Trainer(gpus=1, num_sanity_val_steps=0, limit_val_batches=1,
                         # limit_train_batches=6, limit_val_batches=6,
                         check_val_every_n_epoch=10, max_epochs=600, log_every_n_steps=1000)
    trainer.fit(model, train_loader)
    trainer.validate(model)
