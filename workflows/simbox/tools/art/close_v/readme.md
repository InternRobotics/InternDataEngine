# Annotation Documentation

We provide an optimized and simplified annotation pipeline that removes many redundancies. No need to rename base_link, contact_link, etc. Keep the original hierarchy and naming as much as possible.

## 🗂️ File Information

| Configuration | Example | Description |
|---------------|---------|-------------|
| **DIR** | `YOUR_PATH_TO_DIR/usd` | Directory where USD files are stored |
| **USD_NAME** | `microwave_0.usd` | Scene description file name |
| **INSTANCE_NAME** | `microwave7265` | Model identifier in the scene. You can name it yourself, preferably matching the generated file name |

## 🔧 Model Structure Configuration

| Component | Example | Description |
|-----------|---------|-------------|
| **link0_initial_prim_path** | `/root/group_18` | Absolute path in Isaac Sim for the "door" that interacts with the gripper. Check in the original USD |
| **base_initial_prim_path** | `/root/group_0` | Absolute path in Isaac Sim for the microwave base. Check in the original USD |
| **revolute_joint_initial_prim_path** | `/root/group_18/RevoluteJoint` | Absolute path in Isaac Sim for the revolute joint that opens/closes the microwave. Check in the original USD |
| **Joint Index** | `0` | Joint number, default is 0 |

## 🧭 Axis Configuration

| Axis Type | Example | Description | Visualization |
|-----------|---------|-------------|---------------|
| **LINK0_ROT_AXIS** | `y` | In the local coordinate system of the rotating joint, the axis direction pointing vertically upward | ![LINK0_ROT_AXIS Example](LINK0_ROT_AXIS.jpg) |
| **BASE_FRONT_AXIS** | `z` | In the local coordinate system of the microwave base link, the axis direction facing the door | ![BASE_FRONT_AXIS Example](BASE_FRONT_AXIS.jpg) |
| **LINK0_CONTACT_AXIS** | `-y` | In the local coordinate system of the contact link, the axis direction pointing vertically downward | ![LINK0_CONTACT_AXIS Example](LINK0_CONTACT_AXIS.jpg) |

## 📏 Physical Parameters

| Parameter | Example | Description |
|-----------|---------|-------------|
| **SCALED_VOLUME** | `0.02` | Default value 0.02 for microwave-like objects |

---

# Point Annotation

| Point Type | Description | Visualization |
|------------|-------------|---------------|
| First Point (articulated_object_head) | `Desired base position where the gripper contacts the microwave door` | ![First Point Diagram](head.jpg) |
| Second Point (articulated_object_tail) | `The line direction from the first point should be perpendicular to the microwave door's rotation axis` | ![Second Point Diagram](tail.jpg) |

---