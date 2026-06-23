from pathlib import Path
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat_shapes
from .poe_fkine import poe_fk


# resolve a urdf path
def _resolve_urdf_path(urdf_name: str) -> Path:
    """Resolve a URDF path from either a filename or a relative/absolute path."""
    candidate = Path(urdf_name)
    if candidate.is_file():
        return candidate.resolve()
    current_dir = Path(__file__).resolve().parent
    search_paths = [
        current_dir / "urdf" / urdf_name,
        current_dir / urdf_name,
        Path.cwd() / "urdf" / urdf_name,
        Path.cwd() / urdf_name,
    ]
    for path in search_paths:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Could not find URDF '{urdf_name}'. Looked in: {search_paths}")

# add robot in Meshcat
def add_robot(urdf_name="myarm_300_pi_thorgripper.urdf", scale_factor=0.001):
    urdf_path = _resolve_urdf_path(urdf_name)
    urdf_dir = urdf_path.parent
    current_dir = Path(__file__).resolve().parent

    # Let Pinocchio resolve relative mesh paths against the URDF folder.
    package_dirs = [str(urdf_dir), str(current_dir)]

    # load model
    model, collision_model, visual_model = pin.buildModelsFromUrdf(str(urdf_path), package_dirs)
    data = model.createData()

    # scale only visuals that are still at unit scale.
    # The gripper meshes already have their own 0.001 scale in the URDF;
    # multiplying them again would make them effectively invisible.
    for v in visual_model.geometryObjects:
        mesh_scale = np.asarray(v.meshScale, dtype=float)
        if np.allclose(mesh_scale, np.ones(3)):
            v.meshScale = mesh_scale * scale_factor
        else:
            v.meshScale = mesh_scale
        v.meshColor = np.array([1.0, 1.0, 1.0, 1.0])

    # create MeshCat visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open = False)
    viz.loadViewerModel()

    print(f"Loaded URDF: {urdf_path.name} | nq={model.nq}, nv={model.nv}, visuals={len(visual_model.geometryObjects)}")

    return viz, model, data

# add box in Meshcat
def add_box(viz, T, size_xyz, color = 0xff0000, name = "box"):
    handle = viz.viewer[name]
    handle.set_object(g.Box(size_xyz), g.MeshLambertMaterial(color = color))
    handle.set_transform(T)

# add sphere in Meshcat
def add_sphere(viz, T, radius, color = 0xff0000, name = "sphere"):
    handle = viz.viewer[name]
    handle.set_object(g.Sphere(radius), g.MeshLambertMaterial(color = color))
    handle.set_transform(T)

# add frame in Meshcat
def add_frame(viz, frame_name, axis_length = 0.05, axis_thickness = 0.005, opacity = 1.0, origin_radius = 0.01):
    frame_handle = viz.viewer[frame_name]
    meshcat_shapes.frame(
        frame_handle,
        axis_length = axis_length,
        axis_thickness = axis_thickness,
        opacity = opacity,
        origin_radius = origin_radius,
    )

# update frame in MeshCat
def update_frame(viz, frame_name, T):
    viz.viewer[frame_name].set_transform(T)

# set robot's base
def set_Tws(viz, Tws):
    viz.viewer["pinocchio"].set_transform(Tws)

# pad or truncate q so it matches the Pinocchio model configuration size
def coerce_q_for_model(q, model):
    """Pad or truncate q so it matches the Pinocchio model configuration size."""
    q = np.asarray(q, dtype=float).reshape(-1)
    q_model = np.zeros(model.nq)
    n = min(q.size, model.nq)
    q_model[:n] = q[:n]
    return q_model

# key listener handler
def key_listener_handler(key_callback_map):
    """
    Terminal-based key handler.
    Usage:
        key_listener_handler({
            "": on_enter,
            " ": on_space,
            "q": on_quit,
        })
    Notes:
        Press ENTER after typing a key.
        Empty input "" corresponds to just pressing ENTER.
    """
    print("Keyboard controls:")
    print("  ENTER : next/default action")
    print("  q     : quit")
    while True:
        key = input("> ")
        callback = key_callback_map.get(key)
        if callback is not None:
            callback()
        elif key.lower() in ["q", "quit", "exit"]:
            break
        else:
            print(f"No callback assigned for key: {repr(key)}")

# update robot's configuration
def send_q(viz, model, data, q):
    q_model = coerce_q_for_model(q, model)
    viz.display(q_model)
    pin.forwardKinematics(model, data, q_model)
    pin.updateFramePlacements(model, data)
    return q_model

# send next pose to be visualized
def send_next_pose(viz, model, data, qd_list, Twbd_list, frame_in, pose_index):
    q_new = qd_list[pose_index]
    send_q(viz, model, data, q_new)
    Twb = poe_fk(q_new, frame_in, "world")
    update_frame(viz, "bb", Twb)
    update_frame(viz, "bb_d", Twbd_list[pose_index])
    print(f"Switching pose {pose_index + 1}: q = {np.round(q_new, 3)}")
