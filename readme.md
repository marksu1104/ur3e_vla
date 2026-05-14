### Terminal A (openvla) — VLA server:

```bash
conda activate openvla
cd ~/IsaacLab/ur3e_vla
python vla_server.py 
# 等到看到 "[VLA Server] Ready."
```

### Terminal B (env_isaaclab_ros2) — Sim:

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
conda activate env_isaaclab_ros2
cd ~/IsaacLab
./isaaclab.sh -p ./ur3e_vla/sim_vla.py \
    --target mug \
    --instruction "pick up the red mug" \
    --vla-server http://localhost:8000 \
    --action-scale 0.1 \
    --max-steps 0
```

### Collect data:

```bash
cd ~/ros2_jazzy_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_jazzy_ws/install/setup.bash
conda activate env_isaaclab_ros2
cd ~/IsaacLab
./isaaclab.sh -p ur3e_vla/sim_collect.py \
    --target mug \
    --episodes 5 \    
    --output-dir outputs/test
```
