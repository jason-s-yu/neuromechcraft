# neuromechcraft

> MineRL x NeuroMechFly

## NeuroMechFly (NMF)

[NeuroMechFly (NMF) v2](https://neuromechfly.org/) is a fruit fly model which incorporates a compound eye vision system within a fly-mimicing brain control loop. Eyes are simulated by two cameras covering about 270 degrees FoV combined, with an overlap of about 17 [degrees](https://neuromechfly.org/tutorials/vision_basics.html#retina-simulation).

The model maps the vision from these simulated cameras into a ommatidia input array, approximately 721 ommatidia per eye, each with two color channels (green and blue sensitive receptors). The library essentially outputs a representation of what a fly would see based on the input. We then have the ability to control (with custom decision policies) based on the sensory information.

## MineRL Camera Data -> NMF Data Pipeline

[MineRL](https://github.com/minerllabs/minerl) is a library which provides an OpenAI gym like interface to Minecraft. The observation space provided by the library is a dict `obs` with a `pov` key containing an RGB image `(640x360)` (HxW) pixels representing the simulated view of the agent. This includes the inventory, interface, etc.

## Test Scripts

In [src/](src/), you can find some benchmarking scripts.

- `0_sample_render.py` renders 600 frames worth of video
  - on WSL2 Win 11 Ubuntu 22.04.5 with GPU rendering (RTX 3090) confirmed with `nvidia-smi`, approx 20-45% GPU usage. Elapsed duration about 22-25 seconds.
  - avg frame time is about 30-40 ms.
  - video is rendered to a 60 fps timeline
