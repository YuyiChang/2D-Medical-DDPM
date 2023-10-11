# %%
import sys

root_dir = '/workspace'
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


# %% create diffusion
from diffusion.Create_diffusion import create_gaussian_diffusion
from diffusion.resampler import UniformSampler

# These three parameters: training steps number, learning variance or not (using improved DDPM or original DDPM), and inference 
# timesteps number (only effective when using improved DDPM)
diffusion_steps=1000
learn_sigma=True
timestep_respacing=[50]

# Don't toch these parameters, they are irrelant to the image synthesis
sigma_small=False
class_cond=False
noise_schedule='linear'
use_kl=False
predict_xstart=False
rescale_timesteps=True
rescale_learned_sigmas=True
use_checkpoint=False

diffusion = create_gaussian_diffusion(
    steps=diffusion_steps,
    learn_sigma=learn_sigma,
    sigma_small=sigma_small,
    noise_schedule=noise_schedule,
    use_kl=use_kl,
    predict_xstart=predict_xstart,
    rescale_timesteps=rescale_timesteps,
    rescale_learned_sigmas=rescale_learned_sigmas,
    timestep_respacing=timestep_respacing,
)
schedule_sampler = UniformSampler(diffusion)


# %%

# Here are the dataloader hyper-parameters, including the batch size,
# image size, image spacing, and color channel (usually 1 for medical images)
BATCH_SIZE_TRAIN = 8*1
image_size = 256
img_size = (image_size,image_size)
spacing = (1,1)
channels = 1

# %% create network

num_channels=128
channel_mult = (1, 1, 2, 2, 4, 4)
attention_resolutions="64,32,16,8"
num_heads=[4,4,4,8,16,16]
window_size = [[4,4],[4,4],[4,4],[8,8],[8,8],[4,4]]
num_res_blocks = [2,2,1,1,1,1]
sample_kernel=([2,2],[2,2],[2,2],[2,2],[2,2]),


attention_ds = []
for res in attention_resolutions.split(","):
    attention_ds.append(int(res))
class_cond = False
use_scale_shift_norm=True
resblock_updown = False

from network.Diffusion_model_transformer import SwinVITModel
model = SwinVITModel(
        image_size=(image_size,image_size),
        in_channels=1,
        model_channels=num_channels,
        out_channels=2,
        sample_kernel=sample_kernel,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=0,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=False,
        use_fp16=False,
        num_heads=num_heads,
        window_size = window_size,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=False,
    )

# %%
