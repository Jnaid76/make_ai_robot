# GPT API Integration and Path Planning Pipeline Setup Guide

This guide explains how to set up OpenAI API integration and the GPT-to-path-planning pipeline.

## Step 1: Set Up OpenAI API Key

### Option A: Using .env file (Recommended)

1. Copy the example file:
   ```bash
   cd /Users/zahra/craip_2025f_g4
   cp .env.example .env
   ```

2. Edit `.env` and add your OpenAI API key:
   ```bash
   # Open .env file
   nano .env
   # or
   gedit .env
   ```

3. Replace `your-api-key-here` with your actual API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

### Option B: Using bashrc (Alternative)

If you prefer to use `~/.bashrc` instead:

1. Add to `~/.bashrc`:
   ```bash
   export OPENAI_API_KEY='sk-your-actual-api-key-here'
   ```

2. Source it:
   ```bash
   source ~/.bashrc
   ```

3. Update `config/command_handler_config.yaml`:
   ```yaml
   OPENAI_API_KEY: bashrc  # Change from 'env' to 'bashrc'
   ```

## Step 2: Install Dependencies

1. Activate your conda environment:
   ```bash
   conda activate language_command_handler
   ```

2. Install/update dependencies:
   ```bash
   cd /Users/zahra/craip_2025f_g4/src/language_command_handler
   pip install -r requirements.txt
   ```

   This will install `python-dotenv` if not already installed.

## Step 3: Build the Package

```bash
cd /Users/zahra/craip_2025f_g4
colcon build --packages-select language_command_handler
source install/setup.bash
```

## Step 4: Test the Setup

### Test 1: Basic Language Commands

1. Start Gazebo simulation:
   ```bash
   ros2 launch go1_simulation go1.gazebo.launch.py use_gt_pose:=true
   ```

2. Start robot controller (in another terminal):
   ```bash
   ros2 run unitree_guide2 junior_ctrl
   # Press: 1 → 2 → 5 (to get robot in move_base mode)
   ```

3. Start path tracker (in another terminal):
   ```bash
   ros2 launch path_tracker path_tracker_launch.py
   ```

4. Start language command handler (in another terminal):
   ```bash
   source ~/.bashrc  # If using conda
   ros2 launch language_command_handler start_command_handler.launch.py
   ```

5. Test simple commands:
   ```bash
   # Test stop
   ros2 service call /language_command custom_interfaces/srv/LanguageCommand "{command: 'stop'}"

   # Test forward movement
   ros2 service call /language_command custom_interfaces/srv/LanguageCommand "{command: 'go forward'}"

   # Test backward movement
   ros2 service call /language_command custom_interfaces/srv/LanguageCommand "{command: 'go back'}"
   ```

### Test 2: Navigation with Coordinates

Test navigation to a specific position:

```bash
# Navigate to position (x=2.0, y=1.0, yaw=0.0)
ros2 service call /language_command custom_interfaces/srv/LanguageCommand "{command: 'go to position x=2.0, y=1.0, yaw=0.0'}"

# Navigate to position (x=-1.0, y=2.0, yaw=1.57)
ros2 service call /language_command custom_interfaces/srv/LanguageCommand "{command: 'navigate to coordinates -1.0 2.0 1.57'}"

# Navigate without yaw (defaults to 0.0)
ros2 service call /language_command custom_interfaces/srv/LanguageCommand "{command: 'move to x=3.0, y=1.5'}"
```

## How It Works

### Pipeline Flow

1. **User sends language command** via ROS2 service:
   ```
   /language_command service
   ```

2. **Language Command Handler** receives the command and:
   - Loads OpenAI API key from `.env` file or environment variable
   - Calls GPT-4o with the command and available actions
   - GPT extracts coordinates and selects appropriate action

3. **GPT Response** format:
   - Simple actions: `go_front.py`, `stop.py`, etc.
   - Navigation actions: `navigate_to_goal.launch.py_x2.0_y1.0_yaw0.0`

4. **Action Execution**:
   - Parses action name and parameters
   - Starts the appropriate ROS2 node/launch file
   - For navigation: starts `navigate_to_goal` node with goal coordinates

5. **Path Planning**:
   - `navigate_to_goal` node subscribes to `/go1_pose`
   - Generates smooth path using cubic Hermite splines
   - Publishes path to `/local_path` topic

6. **Path Tracking**:
   - `path_tracker` subscribes to `/local_path`
   - Uses MPPI algorithm to follow the path
   - Publishes velocity commands to `/cmd_vel`

### Supported Commands

**Simple Movement:**
- "go forward", "move forward", "go ahead" → `go_front.py`
- "go back", "move backward", "back up" → `go_back.launch.py`
- "stop", "freeze", "halt" → `stop.py`

**Navigation with Coordinates:**
- "go to x=2.0, y=1.0" → `navigate_to_goal.launch.py_x2.0_y1.0_yaw0.0`
- "navigate to position 3.5, 2.0, yaw=1.57" → `navigate_to_goal.launch.py_x3.5_y2.0_yaw1.57`
- "move to coordinates -1.0 2.0 0.0" → `navigate_to_goal.launch.py_x-1.0_y2.0_yaw0.0`

**Note:** GPT is smart enough to extract coordinates from various natural language formats.

## Troubleshooting

### API Key Not Found
- Check that `.env` file exists in project root
- Verify API key format: `OPENAI_API_KEY=sk-...`
- Check file permissions: `ls -la .env`

### GPT Not Understanding Commands
- Check the prompt in `config/command_handler_config.yaml`
- Look at GPT response in the language_command_handler logs
- Try more explicit commands: "go to position x=2.0, y=1.0"

### Navigation Not Working
- Verify path tracker is running: `ros2 launch path_tracker path_tracker_launch.py`
- Check robot pose is being published: `ros2 topic echo /go1_pose`
- Verify path is being published: `ros2 topic echo /local_path`

### Robot Not Moving
- Ensure robot is in "move_base" mode (press 5 in junior_ctrl terminal)
- Check `/cmd_vel` topic: `ros2 topic echo /cmd_vel`
- Verify robot controller is running

## Next Steps

1. **Add Named Locations**: Update the GPT prompt to map location names (e.g., "toilet", "nurse station") to coordinates
2. **Improve Path Planning**: Add collision avoidance to the path planner
3. **Add More Actions**: Create additional nodes for specific missions (find food, detect objects, etc.)

## Files Modified/Created

- `.env.example` - Template for API key
- `.gitignore` - Added `.env` to ignore list
- `src/language_command_handler/requirements.txt` - Added `python-dotenv`
- `src/language_command_handler/language_command_handler/language_command_handler.py` - Added .env loading and parameter parsing
- `src/language_command_handler/language_command_handler/navigate_to_goal.py` - New navigation node
- `src/language_command_handler/launch/navigate_to_goal.launch.py` - New launch file
- `src/language_command_handler/config/command_handler_config.yaml` - Updated prompt and actions
- `src/language_command_handler/CMakeLists.txt` - Added navigate_to_goal.py to install

