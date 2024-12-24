"""
包含对齐网络的model代码
还有训练代码

"""
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch.utils.data import Dataset


class DiffusionPrior(nn.Module):

    def __init__(
            self,
            embed_dim=1024,
            cond_dim=42,
            hidden_dim=1024,
            layers_per_block=4,
            time_embed_dim=512,
            act_fn=nn.SiLU,
            dropout=0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # 1. time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)
        self.time_embedding = TimestepEmbedding(
            time_embed_dim,
            hidden_dim,
        )

        # 2. conditional embedding
        self.cond_embedding = nn.Linear(cond_dim, hidden_dim)

        # 3. prior mlp

        # 3.1 input
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
        )

        # 3.2 hidden
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    act_fn(),
                    nn.Dropout(dropout),
                )
                for _ in range(layers_per_block)
            ]
        )

        # 3.3 output
        self.output_layer = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, t, c=None):
        # x (batch_size, embed_dim)
        # t (batch_size, )
        # c (batch_size, cond_dim)

        # 1. time embedding
        t = self.time_proj(t)  # (batch_size, time_embed_dim)
        t = self.time_embedding(t)  # (batch_size, hidden_dim)

        # 2. conditional embedding
        c = self.cond_embedding(c) if c is not None else 0  # (batch_size, hidden_dim)

        # 3. prior mlp

        # 3.1 input
        x = self.input_layer(x)

        # 3.2 hidden
        for layer in self.hidden_layers:
            x = x + t + c
            x = layer(x) + x

        # 3.3 output
        x = self.output_layer(x)

        return x


class DiffusionPriorUNet(nn.Module):

    def __init__(
            self,
            embed_dim=1024,
            cond_dim=42,
            hidden_dim=[1024, 512, 256, 128, 64],
            time_embed_dim=512,
            act_fn=nn.SiLU,
            dropout=0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # 1. time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)

        # 2. conditional embedding
        # to 3.2, 3,3

        # 3. prior mlp

        # 3.1 input
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            act_fn(),
        )

        # 3.2 hidden encoder
        self.num_layers = len(hidden_dim)
        self.encode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(
                time_embed_dim,
                hidden_dim[i],
            ) for i in range(self.num_layers - 1)]
        )  # d_0, ..., d_{n-1}
        self.encode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers - 1)]
        )
        self.encode_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i + 1]),
                nn.LayerNorm(hidden_dim[i + 1]),
                act_fn(),
                nn.Dropout(dropout),
            ) for i in range(self.num_layers - 1)]
        )

        # 3.3 hidden decoder
        self.decode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(
                time_embed_dim,
                hidden_dim[i],
            ) for i in range(self.num_layers - 1, 0, -1)]
        )  # d_{n}, ..., d_1
        self.decode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers - 1, 0, -1)]
        )
        self.decode_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i - 1]),
                nn.LayerNorm(hidden_dim[i - 1]),
                act_fn(),
                nn.Dropout(dropout),
            ) for i in range(self.num_layers - 1, 0, -1)]
        )

        # 3.4 output
        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)

    def forward(self, x, t, c=None):
        # x (batch_size, embed_dim)
        # t (batch_size, )
        # c (batch_size, cond_dim)

        # 1. time embedding
        t = self.time_proj(t)  # (batch_size, time_embed_dim)

        # 2. conditional embedding
        # to 3.2, 3.3

        # 3. prior mlp

        # 3.1 input
        x = self.input_layer(x)

        # 3.2 hidden encoder
        hidden_activations = []
        for i in range(self.num_layers - 1):
            hidden_activations.append(x)
            t_emb = self.encode_time_embedding[i](t)
            c_emb = self.encode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.encode_layers[i](x)

        # 3.3 hidden decoder
        for i in range(self.num_layers - 1):
            t_emb = self.decode_time_embedding[i](t)
            c_emb = self.decode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.decode_layers[i](x)
            x += hidden_activations[-1 - i]

        # 3.4 output
        x = self.output_layer(x)

        return x


class EmbeddingDataset(Dataset):

    def __init__(self, c_embeddings, h_embeddings):
        self.c_embeddings = c_embeddings
        self.h_embeddings = h_embeddings

    def __len__(self):
        return len(self.c_embeddings)

    def __getitem__(self, idx):
        return {
            "c_embedding": self.c_embeddings[idx],
            "h_embedding": self.h_embeddings[idx]
        }


class EmbeddingDatasetVICE(Dataset):
    def __init__(self, path_data):
        image_features_dict = torch.load(os.path.join(path_data, 'openclip_emb/image_features.pt'))
        self.embedding_vise = torch.load(os.path.join(path_data, 'variables/embedding_vise.pt'))
        self.image_features = image_features_dict['image_features']
        self.labels = image_features_dict['labels']
        self.label2index = image_features_dict['l2i']

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        idx_c = self.label2index[self.labels[idx]]
        return {
            "c_embedding": self.embedding_vise[idx_c],
            "h_embedding": self.image_features[idx]
        }


# Copied from diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler.add_noise
def add_noise_with_sigma(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
) -> torch.FloatTensor:
    # Make sure sigmas and timesteps have the same device and dtype as original_samples
    sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
    if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
        # mps does not support float64
        schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
        timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
    else:
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

    step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)

    noisy_samples = original_samples + noise * sigma
    return noisy_samples, sigma


# diffusion pipe
class Pipe:

    def __init__(self, diffusion_prior=None, scheduler=None, device='cuda', modality=None):
        self.diffusion_prior = diffusion_prior.to(device)

        if scheduler is None:
            from diffusers.schedulers import DDPMScheduler
            self.scheduler = DDPMScheduler()
            # self.scheduler.add_noise_with_sigma = add_noise_with_sigma.__get__(self.scheduler)
        else:
            self.scheduler = scheduler

        self.modality = modality
        self.device = device

    def _select_features(self, img_features, depth_features, text_features):
        if self.modality == 'image':
            return img_features
        elif self.modality == 'depth':
            return depth_features
        elif self.modality == 'text':
            return text_features
        else:
            raise ValueError(f"wanted mode: image / depth / text, your mode: {self.modality}")

    def train(self, dataloader,
              num_epochs=10, learning_rate=1e-4,
              eeg_model=None, modality_model=None,
              testdataloader=None, subject=None,
              save_path=None, early_stop_patience = 50):

        device = self.device
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.diffusion_prior.parameters(), lr=learning_rate)
        from diffusers.optimization import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(dataloader) * num_epochs),
        )

        num_train_timesteps = self.scheduler.config.num_train_timesteps

        best_loss = 1000
        best_sim = -100
        early_stopping_step = 0

        for epoch in range(num_epochs):
            if early_stopping_step == early_stop_patience:
                print(f"已经有 {early_stopping_step} 次没有提升，终止训练！")
                break

            self.diffusion_prior.train()
            eeg_model.eval() #设定成eval?
            eval_loss_sum = 0
            loss_sum = 0
            sim = 0
            count = 0
            for x, label, text, text_features, img, img_features, depth_features, aug_img_features, img_index, index \
                    in dataloader:
                count += 1
                x = x.to(device)

                ##### * 10 FOR keeping basic observable ####
                selected_features = (self._select_features(img_features, depth_features, text_features) * 10.0).to(device)
                ##### * 10 FOR keeping basic observable ####

                c_embeds = eeg_model(x).type(torch.float32).to(device)
                h_embeds = selected_features.type(torch.float32).to(device)
                N = h_embeds.shape[0]

                # 1. randomly replecing c_embeds to None
                if torch.rand(1) < 0.1:
                    c_embeds = None

                # 2. Generate noisy embeddings as input
                noise = torch.randn_like(h_embeds) #其是服从正态分布的

                # 3. sample timestep
                timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)

                # 4. add noise to h_embedding
                perturbed_h_embeds = self.scheduler.add_noise(
                    h_embeds,
                    noise,
                    timesteps
                )  # (batch_size, embed_dim), (batch_size, )

                # 5. predict noise
                noise_pre = self.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)

                # 6. loss function weighted by sigma
                loss = criterion(noise_pre, noise)  # (batch_size,)
                loss = (loss).mean()

                # 7. update parameters
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)
                lr_scheduler.step()
                optimizer.step()

                loss_sum += loss.item()

            # lr_scheduler.step(loss)

            # 添加评估代码
            self.diffusion_prior.eval()
            with torch.no_grad():
                for x, label, text, text_features, img, img_features, depth_features, aug_img_features, img_index, index \
                        in testdataloader:
                    count += 1
                    x = x.to(device)
                    selected_features = self._select_features(img_features, depth_features, text_features).to(device)
                    c_embeds = eeg_model(x).type(torch.float32).to(device)
                    h_embeds = selected_features.type(torch.float32).to(device)
                    N = h_embeds.shape[0]

                    # 评估余弦相似度
                    # output_embedding = []
                    # for idx in range(200):
                    #     output_embedding.append(self.generate(c_embeds=c_embeds[idx:idx + 1],
                    #                                           num_inference_steps=50,
                    #                                           guidance_scale=5.0))
                    # output_embedding = torch.stack(output_embedding, dim=0).squeeze(1)

                    output_embedding = self.generate(c_embeds=c_embeds,
                                                     num_inference_steps=50,
                                                     guidance_scale=5.0)
                    sim = F.cosine_similarity(output_embedding / 10, h_embeds[:200], dim=1).mean().item()

                    # 1. randomly replecing c_embeds to None
                    if torch.rand(1) < 0.1:
                        c_embeds = None

                    # 2. Generate noisy embeddings as input
                    noise = torch.randn_like(h_embeds)

                    # 3. sample timestep
                    timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)

                    # 4. add noise to h_embedding
                    perturbed_h_embeds = self.scheduler.add_noise(
                        h_embeds,
                        noise,
                        timesteps
                    )  # (batch_size, embed_dim), (batch_size, )

                    # 5. predict noise
                    noise_pre = self.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)

                    # 6. loss function weighted by sigma
                    eval_loss = criterion(noise_pre, noise)  # (batch_size,)
                    eval_loss = (eval_loss).mean()

                    eval_loss_sum += eval_loss.item()


            loss_epoch = loss_sum / len(dataloader)
            eval_loss_epoch = eval_loss_sum / len(testdataloader)
            print(f'epoch: {epoch}, train_loss: {loss_epoch}, eval_loss: {eval_loss_epoch}, sim: {sim}')

            # save best model
            if best_sim < sim:
                model_name =  f"model_best.pth"
                best_sim = sim
                early_stopping_step = 0
                torch.save(self.diffusion_prior.state_dict(), os.path.join(save_path, model_name))
                print(f'Epoch {epoch}: cosine improved ({sim:.6f}). Model saved.')
            else:
                early_stopping_step += 1

    def generate(
            self,
            c_embeds=None,
            num_inference_steps=50,
            timesteps=None,
            guidance_scale=5.0,
            generator=None
    ):
        # c_embeds (batch_size, cond_dim)
        self.diffusion_prior.eval()
        N = c_embeds.shape[0] if c_embeds is not None else 1

        # 1. Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, timesteps)

        # 2. Prepare c_embeds
        if c_embeds is not None:
            c_embeds = c_embeds.to(self.device)

        # 3. Prepare noise
        h_t = torch.randn(N, self.diffusion_prior.embed_dim, generator=generator, device=self.device)

        # 4. denoising loop
        for _, t in enumerate(timesteps):
            t = torch.ones(h_t.shape[0], dtype=torch.float, device=self.device) * t
            # 4.1 noise prediction
            if guidance_scale == 0 or c_embeds is None:
                noise_pred = self.diffusion_prior(h_t, t)
            else:
                noise_pred_cond = self.diffusion_prior(h_t, t, c_embeds)
                noise_pred_uncond = self.diffusion_prior(h_t, t)
                # perform classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 4.2 compute the previous noisy sample h_t -> h_{t-1}
            h_t_output = []
            if noise_pred.shape[0] > 1:
                for idx in range(noise_pred.shape[0]):
                    h_t_output.append(self.scheduler.step(noise_pred[idx:idx+1], t[idx:idx+1].long().item(), h_t[idx:idx+1], generator=generator).prev_sample)
                h_t = torch.cat(h_t_output)
            else:
                h_t = self.scheduler.step(noise_pred, t.long().item(), h_t, generator=generator).prev_sample

        return h_t


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    output_embedding = torch.randn(100, 1024)
    h_embeds = torch.randn(100, 1024)
    sim = F.cosine_similarity(output_embedding, h_embeds, dim=1).mean().item()

    # 1. test prior
    prior = DiffusionPriorUNet(cond_dim=1024)
    pipe = Pipe(prior, device='cpu', modality="image")

    x = torch.randn(20, 1024)
    t = torch.randint(0, 1000, (20,))
    c = torch.randn(20, 1024)
    pipe.generate(c_embeds=x,
                  num_inference_steps=50,
                  guidance_scale=5.0)
    y = prior(x, t, c)
    print(y.shape)