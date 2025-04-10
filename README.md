# neuromechcraft

> MineRL x NeuroMechFly

## Requirements / Env Setup

I have tested on two versions of MineRL: v1.0 and v0.4.4.

### MineRL 1.0

- Python 3.11.11
- Java JDK 8: tested on 8.0.442-tem
  - suggest installing and using [sdkman](https://sdkman.io/install/)

A basic setup script is provided in [setup_env.sh](setup_env.sh). It relies on `pyenv` and `pyenv-virtualenv` and creates a virtual env for this project, and installs the required dependencies.

If you have `sdkman` installed, it will recognize the correct sdk version from `.sdkmanrc`. You may need to `sdk install java 8.0.442-tem`.

NB: I couldn't find `8.0.442-tem` on Mac `sdkman`, so any equivalent redistribution on version 8 will work (corretto, etc.)

### MineRL 0.4.4

- Java JDK 8: tested on 8.0.442-tem
  - suggest installing and using [sdkman](https://sdkman.io/install/)
- (suggested) create a separate virtual env
- Requires Python 3.10.16 (python 10) in order to support gym 0.19
  - Python 3.11 has a newer version of `setuptools` which is incompatible with the 0.19 installation
  - EDIT: actualy 3.11 might work, but I tested with 3.10. Feel free to use a later version and see if it works
- `pip install pip install setuptools==65.5.0 pip==21`; `pip install wheel==0.38.0` [Reference](https://stackoverflow.com/a/77205046)
- `pip install matplotlib numpy flygym pyglet`

Then, you will need to modify the MineRL repository. [Reference](https://github.com/minerllabs/minerl/issues/744).

1. Clone this repository
2. Clone `git clone git@github.com:minerllabs/minerl.git`
3. Checkout v0.4.4 `git checkout v0.4.4`
4. Modify `minerl/Malmo/Minecraft/build.gradle`:
   1. Add the lines to the `buildscript`

      ```gradle
      maven {
          url 'file:path/to/neuromechcraft/MixinGradle-dcfaf61'
      }
      ```

   2. Adjust the `classpath` under `buildscript`

      ```gradle
      classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61'){ // 0.6
            // Because forgegradle requires 6.0 (not -debug-all) while mixingradle depends on 5.0
            // and putting mixin right here will place it before forge in the class loader
            exclude group: 'org.ow2.asm', module: 'asm-debug-all'
        }
      ```

5. `pip install path/to/minerl_repository`

## NeuroMechFly (NMF)

[NeuroMechFly (NMF) v2](https://neuromechfly.org/) is a fruit fly model which incorporates a compound eye vision system within a fly-mimicing brain control loop. Eyes are simulated by two cameras covering about 270 degrees FoV combined, with an overlap of about 17 [degrees](https://neuromechfly.org/tutorials/vision_basics.html#retina-simulation).

The model maps the vision from these simulated cameras into a ommatidia input array, approximately 721 ommatidia per eye, each with two color channels (green and blue sensitive receptors). The library essentially outputs a representation of what a fly would see based on the input. We then have the ability to control (with custom decision policies) based on the sensory information.

## MineRL Camera Data -> NMF Data Pipeline

[MineRL](https://github.com/minerllabs/minerl) is a library which provides an OpenAI gym like interface to Minecraft. The observation space provided by the library is a dict `obs` with a `pov` key containing an RGB image `(640x360)` (HxW) pixels representing the simulated view of the agent. This includes the inventory, interface, etc.

## Test Scripts

In [src/](src/), you can find some benchmarking scripts.

- `0_singlecam_rendering_minerl.py` renders 600 frames worth of video straight from Minecraft
  - on WSL2 Win 11 Ubuntu 22.04.5 with GPU rendering (RTX 3090) confirmed with `nvidia-smi`, approx 20-45% GPU usage. Elapsed duration about 22-25 seconds.
  - avg frame time is about 30-40 ms == about 28-30 fps
  - video is rendered to a 60 fps timeline
- `1_singlecam_with_pipeline.py ACTION` has the agent move one step forward and jump, before rendering to output
  - we also pipe the raw rgb data to NMF/flygym and show the fly representation in video
  - `ACTION` defaults to `forward`, but you can pass `random` or `forward` to tell the agent to either do a random action at each turn, or constantly go forward.
