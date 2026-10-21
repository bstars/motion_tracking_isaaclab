# General Motion Tracking

This part trains a general motion tracking policy. Following are super related works

- BeyondMimic [short paper](../literatures/BeyondMimic_Short.pdf), [long paper](../literatures/BeyondMimic_Long.pdf), [Github repository](https://github.com/HybridRobotics/whole_body_tracking/tree/main)

- Sonic [short paper](../literatures/sonic.pdf), [Github repository](https://github.com/NVlabs/GR00T-WholeBodyControl)

- My previous motion tracking project [Github repository](https://github.com/bstars/MotionTracking/tree/main)


## 1. Observation

##### Proprioception
- Robot joint pos $q$
- Robot joint velocity $\dot{q}$
- Robot root angular velocity in root frame
- Robot gravity vector in root frame
- Previous action

#### Reference Motion (Command)
- Reference joint pos (Single or multiple frames)
- Reference joint velocity (Single or multiple frames)
- Reference root gravity vector (Single or multiple frames)
- Relative position from robot anchor to reference anchor (Optional)
- Relative orientation from robot anchor to renference anchor, represented by the last 2 columns of rotation matrix (Optional)

#### Priviliged Information
- Robot root linear velocity
- The relative position and orientation (orientation represented by the last 2 columns of the rotation matrix) from the anchor body to the reference track bodies (except for the anchor body itself). For each body, the relative position is 3-dim, orientation is represented by the last 2 columns of rotation matrix, thus 6-dim.

#### Actor Observation
- Proprioception (noised)
- Reference motion

#### Critic Observation
- Proprioception
- Reference motion
- Priviliged Information

## 2. Termination
- Anchor projected gravity diff more than 0.8 in z axis
- Anchor z coordinate diff more than 0.25
- End effector z coordinate diff more than 0.25

## 3. Adaptive Motion Sampling

The multi-motion command uses a SONIC-style adaptive curriculum. Instead of
sampling all frames uniformly, it increases the probability of sampling motion
segments where the policy frequently fails.

### 3.1 Motion bins

Each motion clip is divided into fixed-size temporal bins:

```text
motion 0: [0, 50), [50, 100), [100, end)
motion 1: [0, 50), [50, 100), ...
```

Every bin stores its motion ID, start frame, and end frame. Only valid command
start frames are included, so all `num_future_frames` reference frames remain
inside the selected motion clip.

The sampler maintains the following statistics for each bin $i$:

- $E_i$: number of episode exposures
- $F_i$: number of early tracking failures
- $r_i = F_i / E_i$: failure-based difficulty score

Both counters are initialized with `init_num_failures`, normally 1. This
pseudo-count gives every bin a nonzero initial score and allows all bins to be
sampled before real failure statistics are available.

When an episode ends, its final local motion frame determines which bin receives
the update. Every completed episode increases $E_i$. Only an early tracking
termination increases $F_i$. Normal environment timeouts and reaching the end
of a motion clip are treated as successful completion rather than failure.

Following SONIC, the exposure increment is normalized by the bin length:

$$
E_i \leftarrow E_i + \frac{n_i}{L_i},
$$

where $n_i$ is the number of completed environments assigned to the bin and
$L_i$ is its length. Consequently, $r_i$ is a difficulty score and is not
necessarily a probability bounded by one.

### 3.2 Sampling probabilities

The failure scores are first clipped to prevent one outlier bin from dominating
training:

$$
\hat r_i = \min\left(r_i,\; \bar r C\right),
$$

where $\bar r$ is the mean failure score and $C$ is
`adp_samp_failure_rate_max_over_mean`.

The clipped scores are normalized into a failure-based distribution:

$$
p_i^{\mathrm{failure}} =
\frac{\hat r_i}{\sum_j \hat r_j}.
$$

To preserve exploration of easier segments, this distribution is mixed with a
uniform distribution:

$$
p_i = (1-u)p_i^{\mathrm{failure}} + u\frac{1}{N},
$$

where $u$ is `uniform_sampling_rate` and $N$ is the number of bins. For
example, $u=0.1$ allocates 90% of sampling probability according to observed
difficulty and 10% uniformly.

When `sequence_length_agnostic=True`, a bin is also weighted by its length and
the number of bins in its motion:

$$
w_i = \frac{L_i / \bar L}{B_{m(i)}},
$$

where $B_{m(i)}$ is the number of bins belonging to the bin's motion. The final
distribution is

$$
p_i^{\mathrm{final}} = \frac{p_i w_i}{\sum_j p_j w_j}.
$$

This reduces the tendency to oversample long motion clips merely because they
contain more bins.

### 3.3 Sampling before failures

At reset, the sampler first draws a bin from $p_i^{\mathrm{final}}$, then
uniformly draws a frame inside that bin. If `pre_failure_sample_window` is
positive, a random offset is subtracted from the sampled frame:

```python
offset = torch.randint(pre_failure_sample_window)
start_frame = max(sampled_frame - offset, 0)
```

This starts the policy before the difficult state, giving it time to learn the
actions that lead into and recover from the failure region.

SONIC also supports optional backward failure-rate decay:

$$
d_i = r_i + \gamma d_{i+1},
$$

where propagation stops at motion boundaries. This is controlled by
`use_failure_rate_decay` and `decay_gamma`. It is normally unnecessary when a
pre-failure sampling window is used.

### 3.4 Configuration

```python
adaptive_sampling={
    "enable": True,
    "bin_size": 50,
    "sequence_length_agnostic": True,
    "init_num_failures": 1,
    "uniform_sampling_rate": 0.1,
    "pre_failure_sample_window": 200,
    "use_failure_rate_decay": False,
    "decay_gamma": 0.8,
    "adp_samp_failure_rate_max_over_mean": 50.0,
}
```

Setting `adaptive_sampling=None` disables the curriculum and restores uniform
motion and frame sampling.

The implementation is in [`command_multi.py`](./command_multi.py):

- `MotionLoader._init_adaptive_sampling`: constructs the motion bins and priors
- `MotionLoader.update_adaptive_sampling`: records episode and failure outcomes
- `MotionLoader._compute_adaptive_sampling_probabilities`: computes the sampling distribution
- `MotionLoader.sample_motion_ids_and_time_steps`: samples a bin and start frame
- `MultiMotionCommand._resample_command`: integrates statistics and sampling with environment resets
