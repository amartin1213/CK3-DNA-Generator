"""Full attribute->render pipeline: applies blendshapes (shape keys) AND
guessed bone-pose transforms for coarse (non-"bs") morph genes, then renders
front+profile views matching the AGOT reference-video framing.

Run with:
    blender --background --python ck3_render_full.py -- <json_attr_file> <out_png_prefix>

<json_attr_file> is a JSON dict of {attribute_name: value_0_to_255}.
Attributes matching a blendshapes/male_bs_<attr>_max.mesh / _min.mesh file
are applied as shape keys; everything else is looked up in BONE_MAP below
and applied as a guessed pose-bone translation/rotation.
"""
import json
import math
import os
import sys

import bpy
import mathutils

from bl_ext.user_default.io_pdx_mesh.pdx_blender import blender_import_export as bie  # noqa: E402

HEAD_DIR = (
    r"D:\SteamLibrary\steamapps\common\Crusader Kings III\game\gfx\models\portraits\male_head"
)
BASE_MESH = os.path.join(HEAD_DIR, "male_head.mesh")
BLENDSHAPES_DIR = os.path.join(HEAD_DIR, "blendshapes")

# Guessed bone-driven coarse gene -> list of (bone_name, axis, sign, scale_units)
# axis is local bone-space axis (0=X,1=Y,2=Z). scale_units is the translation
# magnitude (in mesh units, same scale as bone head coords ~tens) applied at
# full +1.0 offset (DNA value 255 -> offset +1.0, value 0 -> offset -1.0).
BONE_MAP = {
    "chin_height": [("bn_h_chin_l_side", 2, -1, 0.9), ("bn_h_chin_r_side", 2, -1, 0.9)],
    "chin_width": [("bn_h_chin_l_side", 0, +1, 1.2), ("bn_h_chin_r_side", 0, -1, 1.2)],
    "chin_forward": [("bn_h_chin_l_side", 1, -1, 1.0), ("bn_h_chin_r_side", 1, -1, 1.0)],
    "jaw_height": [("bn_h_jaw_low", 2, -1, 6.0)],
    "jaw_width": [("bn_h_jaw_l_side", 0, +1, 4.0), ("bn_h_jaw_r_side", 0, -1, 4.0)],
    "jaw_forward": [("bn_h_jaw_low", 1, -1, 4.0)],
    "eye_depth": [("bn_h_eye_l_main", 1, +1, 2.5), ("bn_h_eye_r_main", 1, +1, 2.5)],
    "eye_distance": [("bn_h_eye_l_main", 0, +1, 2.0), ("bn_h_eye_r_main", 0, -1, 2.0)],
    "eye_height": [("bn_h_eye_l_main", 2, +1, 2.0), ("bn_h_eye_r_main", 2, -1, 2.0)],

    # --- Cheek (bn_h_cheek_l/r_main = primary, _high = upper cheek, _side = lower/face_lower child) ---
    "cheek_forward": [("bn_h_cheek_l_main", 1, -1, 1.5), ("bn_h_cheek_r_main", 1, -1, 1.5),
                       ("bn_h_cheek_l_side", 1, -1, 1.5), ("bn_h_cheek_r_side", 1, -1, 1.5)],
    "cheek_height": [("bn_h_cheek_l_high", 2, +1, 1.5), ("bn_h_cheek_r_high", 2, +1, 1.5)],
    "cheek_width": [("bn_h_cheek_l_main", 0, +1, 1.5), ("bn_h_cheek_r_main", 0, -1, 1.5),
                     ("bn_h_cheek_l_side", 0, +1, 1.5), ("bn_h_cheek_r_side", 0, -1, 1.5)],

    # --- Forehead ---
    "forehead_height": [("bn_h_forehead_main", 2, +1, 1.5), ("bn_h_forehead_mid", 2, +1, 1.5)],
    "forehead_width": [("bn_h_forehead_l_side", 0, +1, 1.5), ("bn_h_forehead_r_side", 0, -1, 1.5)],
    "forehead_angle": [("bn_h_forehead_main", 1, -1, 1.5)],
    "forehead_roundness": [("bn_h_forehead_mid", 1, -1, 1.5)],
    "forehead_brow_height": [("bn_h_forehead_l_browInner", 2, +1, 1.0), ("bn_h_forehead_r_browInner", 2, +1, 1.0),
                              ("bn_h_forehead_l_browOuter", 2, +1, 1.0), ("bn_h_forehead_r_browOuter", 2, +1, 1.0)],

    # --- Head shape (approximate; no dedicated "top" bone found, reuse bn_h_skull) ---
    "head_height": [("bn_h_head_mid", 2, +1, 1.5)],
    "head_width": [("bn_h_midhead_l_side", 0, +1, 1.5), ("bn_h_midhead_r_side", 0, -1, 1.5)],
    "head_profile": [("bn_h_head_mid", 1, -1, 1.5)],
    "head_top_height": [("bn_h_skull", 2, +1, 1.5)],
    "head_top_width": [("bn_h_skull", 0, +1, 1.5)],

    # --- Mouth ---
    "mouth_corner_depth": [("bn_h_mouth_l_corner", 1, -1, 1.5), ("bn_h_mouth_r_corner", 1, -1, 1.5)],
    "mouth_corner_height": [("bn_h_mouth_l_corner", 2, +1, 1.5), ("bn_h_mouth_r_corner", 2, +1, 1.5)],
    "mouth_forward": [("bn_h_mouth_upperLip", 1, -1, 1.8), ("bn_h_mouth_lowerLip", 1, -1, 1.8),
                       ("bn_h_mouth_l_upperLip", 1, -1, 1.8), ("bn_h_mouth_r_upperLip", 1, -1, 1.8),
                       ("bn_h_mouth_l_lowerLip", 1, -1, 1.8), ("bn_h_mouth_r_lowerLip", 1, -1, 1.8)],
    "mouth_height": [("bn_h_mouth_upperLip", 2, +1, 1.8), ("bn_h_mouth_lowerLip", 2, +1, 1.8)],
    "mouth_lower_lip_size": [("bn_h_mouth_lowerLip", 2, +1, 1.2), ("bn_h_mouth_l_lowerLip", 2, +1, 1.2),
                              ("bn_h_mouth_r_lowerLip", 2, +1, 1.2)],
    "mouth_upper_lip_size": [("bn_h_mouth_upperLip", 2, -1, 1.2), ("bn_h_mouth_l_upperLip", 2, -1, 1.2),
                              ("bn_h_mouth_r_upperLip", 2, -1, 1.2)],
    "mouth_open": [("bn_h_mouth_lowerLip", 2, -1, 2.5), ("bn_h_mouth_l_lowerLip", 2, -1, 2.5),
                   ("bn_h_mouth_r_lowerLip", 2, -1, 2.5)],
    "mouth_width": [("bn_h_mouth_l_corner", 0, +1, 2.0), ("bn_h_mouth_r_corner", 0, -1, 2.0)],

    # --- Neck ---
    "neck_length": [("bn_h_neck_1", 2, +1, 1.5), ("bn_h_neck_2", 2, +1, 1.5)],
    "neck_width": [("bn_h_neck_l_side", 0, +1, 1.5), ("bn_h_neck_r_side", 0, -1, 1.5)],
}

# Rotation-driven coarse genes: no matching blendshape file exists for these names, and
# the bones involved (eye "_rotate"/eyelid bones, brow bones) are natural rotation hinges
# rather than translation targets. Entries: gene -> [(bone_name, euler_axis, sign, max_radians)]
ROTATION_MAP = {
    "eye_angle": [("bn_h_eye_l_rotate", 2, +1, 0.35), ("bn_h_eye_r_rotate", 2, -1, 0.35)],
    "eye_shut": [
        ("bn_h_eye_l_lidUpper", 0, -1, 0.6), ("bn_h_eye_r_lidUpper", 0, -1, 0.6),
        ("bn_h_eye_l_lidLower", 0, +1, 0.3), ("bn_h_eye_r_lidLower", 0, +1, 0.3),
    ],
    "eyebrows_fullness": [
        ("bn_h_forehead_l_browInner", 0, -1, 0.25), ("bn_h_forehead_r_browInner", 0, -1, 0.25),
        ("bn_h_forehead_l_browOuter", 0, -1, 0.25), ("bn_h_forehead_r_browOuter", 0, -1, 0.25),
    ],
    "eyebrows_shape": [
        ("bn_h_forehead_l_browInner", 1, +1, 0.25), ("bn_h_forehead_r_browInner", 1, -1, 0.25),
        ("bn_h_forehead_l_browOuter", 1, -1, 0.25), ("bn_h_forehead_r_browOuter", 1, +1, 0.25),
    ],
}


def offset_from_value(v):
    """Map a 0-255 UI value to a signed -1..+1 offset (128 = neutral)."""
    return (v - 128.0) / 127.0


def find_blendshape_file(attr):
    for suffix in ("_max", "_min"):
        path = os.path.join(BLENDSHAPES_DIR, f"male_bs_{attr}{suffix}.mesh")
        if os.path.exists(path):
            return path, suffix
    return None, None


# The base mesh's embedded material references a placeholder texture name
# ("neutral_DefaultMaterial_BaseColor...") that does not exist on disk --
# the real game resolves the actual skin diffuse/normal at runtime based on
# ethnicity/age DNA genes, which this static-mesh import can't replicate.
# Fall back to one of the real ethnicity-variant textures shipped alongside
# the mesh so renders have actual skin tone/shading instead of a flat
# placeholder color (needed for dlib face detection on synthetic renders).
FALLBACK_DIFFUSE = os.path.join(HEAD_DIR, "male_head_02_diffuse.dds")
FALLBACK_NORMAL = os.path.join(HEAD_DIR, "male_head_02_normal.dds")

# Non-color scalar/vector inputs (roughness, specular/metallic masks, and the
# small "detail decal" normal overlays like decal_male_head_chin_def_normal)
# must NOT receive the face-photo fallback: the decal nodes reconstruct their
# target value via Separate Color -> Combine Color chains rather than a
# direct link into a Normal Map node, so the old "feeds a Normal Map node"
# check never caught them, and they ended up with the real skin-tone photo
# plugged in as a roughness/bump source -- its natural contrast (eyebrows,
# lip/shadow edges, etc.) was read back as wild per-pixel roughness values or
# fake normal-map bumps, producing dark blotchy "fake shading" patches on
# every render regardless of pose. Classify by the (very descriptive)
# placeholder image/node name instead of trying to trace the node graph.
_FLAT_NORMAL_IMG_NAME = "ck3dna_flat_normal_fallback"
_FLAT_GRAY_IMG_NAME = "ck3dna_flat_gray_fallback"


def _flat_fallback_image(name, rgba):
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.new(name, width=4, height=4, alpha=False)
        img.pixels = list(rgba) * (4 * 4)
    img.colorspace_settings.name = "Non-Color"
    return img


def fix_skin_texture(mesh_obj):
    """Replaces io_pdx_mesh's placeholder (missing-file) texture nodes with
    appropriate fallbacks across EVERY material slot on `mesh_obj` -- not
    just the primary skin material. The head mesh also carries small
    "detail decal" materials (e.g. a chin-definition normal overlay) whose
    source textures are likewise absent on disk; left alone, their tiny
    placeholder images render as visible colored/shading blotches.
    """
    for mat_slot in mesh_obj.material_slots:
        mat = mat_slot.material
        if mat is None or not mat.use_nodes:
            continue
        for node in list(mat.node_tree.nodes):
            if node.type != "TEX_IMAGE":
                continue
            img = node.image
            # Real textures are large; the io_pdx_mesh placeholder for a
            # missing file is a tiny (or 0x0) stand-in -- only replace that.
            if img is not None and img.size[0] > 64:
                continue
            name = (img.name if img is not None else node.name).lower()
            if "normal" in name:
                # Flat "pointing straight up" normal so decal detail layers
                # (reconstructed via Separate/Combine Color, not a direct
                # Normal Map link) contribute zero extra bump instead of
                # baking the stand-in face photo's contrast into fake bumps.
                node.image = _flat_fallback_image(_FLAT_NORMAL_IMG_NAME, (0.5, 0.5, 1.0, 1.0))
                continue
            if any(tok in name for tok in ("roughness", "specular", "metallic", "gloss", "occlusion")):
                # Flat mid-gray scalar so these non-color inputs don't get
                # random per-pixel values from the face photo's RGB content.
                node.image = _flat_fallback_image(_FLAT_GRAY_IMG_NAME, (0.5, 0.5, 0.5, 1.0))
                continue
            # Base color / diffuse / albedo and anything else: real skin photo.
            if not os.path.exists(FALLBACK_DIFFUSE):
                continue
            node.image = bpy.data.images.load(FALLBACK_DIFFUSE, check_existing=True)


EYE_DIFFUSE = os.path.join(HEAD_DIR, "male_eyes_diffuse.dds")


def fix_eye_texture(mesh_obj):
    for mat_slot in mesh_obj.material_slots:
        mat = mat_slot.material
        if mat is None or not mat.use_nodes:
            continue
        principled = mat.node_tree.nodes.get("Principled BSDF")
        if principled is None:
            continue
        socket = principled.inputs.get("Base Color")
        if socket is None or not socket.is_linked:
            continue
        node = socket.links[0].from_node
        if node.type != "TEX_IMAGE":
            continue
        img = node.image
        if img is not None and (img.size[0] > 64 or not os.path.exists(EYE_DIFFUSE)):
            continue
        if not os.path.exists(EYE_DIFFUSE):
            continue
        node.image = bpy.data.images.load(EYE_DIFFUSE, check_existing=True)


def setup_camera_and_light(mesh_obj, name_suffix=""):
    verts = [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    cx, cy, cz = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2
    depth = max(ys) - min(ys)

    cam_data = bpy.data.cameras.new(f"Cam{name_suffix}")
    cam_obj = bpy.data.objects.new(f"Cam{name_suffix}", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = (cx, cy - depth * 3.5, cz)
    cam_obj.rotation_euler = (math.radians(90), 0, 0)
    bpy.context.scene.camera = cam_obj

    light_data = bpy.data.lights.new(f"Sun{name_suffix}", type="SUN")
    light_data.energy = 3.0
    # A default SUN light's tiny angular size casts razor-hard shadows; on
    # this low-poly head mesh those hard shadows fall in natural facial
    # creases (nasolabial fold, under-eye, mouth corners, jaw/chin) and read
    # as dark blotchy "fake shading" patches that have nothing to do with
    # texture placeholders. Widen the angle to soften shadows substantially.
    light_data.angle = math.radians(25)
    light_obj = bpy.data.objects.new(f"Sun{name_suffix}", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (cx + depth, cy - depth, cz + depth)

    fill_data = bpy.data.lights.new(f"Fill{name_suffix}", type="SUN")
    fill_data.energy = 2.0
    fill_data.angle = math.radians(25)
    fill_obj = bpy.data.objects.new(f"Fill{name_suffix}", fill_data)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.location = (cx - depth, cy - depth * 2.0, cz)
    fill_obj.rotation_euler = (math.radians(70), 0, math.radians(-30))

    world = bpy.data.worlds.new("World") if not bpy.context.scene.world else bpy.context.scene.world
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[1].default_value = 0.6
    return cam_obj


def render(path):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    attr_file, out_prefix = argv[0], argv[1]
    with open(attr_file, encoding="utf-8") as f:
        attrs = json.load(f)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bie.import_meshfile(BASE_MESH, imp_mesh=True, imp_skel=True, imp_locs=False, join_materials=True)
    except AttributeError as exc:
        print(f"WARN: material setup failed (ignored): {exc}")

    mesh_obj = next(o for o in bpy.data.objects if o.type == "MESH")
    rig_obj = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    fix_skin_texture(mesh_obj)

    # Eyeballs are a separate rigged mesh (skinned to the same bn_h_eye_*
    # bones already in the armature), not part of the head mesh -- import
    # it into the same scene (reusing the existing skeleton) so rendered
    # faces have real eyes instead of empty sockets.
    eye_mesh_path = os.path.join(HEAD_DIR, "male_eyes.mesh")
    if os.path.exists(eye_mesh_path):
        try:
            bie.import_meshfile(eye_mesh_path, imp_mesh=True, imp_skel=False, imp_locs=False, join_materials=True)
        except AttributeError as exc:
            print(f"WARN: eye material setup failed (ignored): {exc}")
        for obj in bpy.data.objects:
            if obj.type == "MESH" and obj is not mesh_obj:
                fix_eye_texture(obj)

    applied_bs, applied_bone = [], []

    # --- Blendshapes ---
    for attr, value in list(attrs.items()):
        bs_path, bs_suffix = find_blendshape_file(attr)
        if bs_path is None:
            continue
        offset = offset_from_value(value)
        # "_max" mesh is the positive-direction shape key (only meaningful for offset > 0);
        # "_min" mesh is the negative-direction shape key (only meaningful for offset < 0),
        # and its key value should be the positive magnitude of that negative offset.
        if bs_suffix == "_max":
            key_value = max(0.0, offset)
        else:
            key_value = max(0.0, -offset)
        if key_value <= 0.0:
            applied_bs.append((attr, value))
            continue
        try:
            pre_import_objs = set(bpy.data.objects.keys())
            bie.import_meshfile(bs_path, imp_mesh=True, imp_skel=False, imp_locs=False, join_materials=False)
        except AttributeError as exc:
            print(f"WARN: bs material setup failed (ignored): {exc}")
            continue
        new_objs = [bpy.data.objects[name] for name in bpy.data.objects.keys() if name not in pre_import_objs]
        bs_objs = [o for o in new_objs if o.type == "MESH"]
        if not bs_objs:
            print(f"WARN: bs import for '{attr}' created no new mesh object -- skipped")
            continue
        bs_obj = bs_objs[-1]
        bpy.ops.object.select_all(action="DESELECT")
        bs_obj.select_set(True)
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.join_shapes()
        key = mesh_obj.data.shape_keys.key_blocks[-1]
        key.value = key_value
        bpy.data.objects.remove(bs_obj, do_unlink=True)
        applied_bs.append((attr, value))

    # --- Bone transforms ---
    bpy.context.view_layer.objects.active = rig_obj
    bpy.ops.object.mode_set(mode="POSE")
    for attr, value in attrs.items():
        mapping = BONE_MAP.get(attr)
        if not mapping:
            continue
        offset = offset_from_value(value)
        for bone_name, axis, sign, scale in mapping:
            pbone = rig_obj.pose.bones.get(bone_name)
            if pbone is None:
                print(f"WARN: bone {bone_name} not found")
                continue
            delta = [0.0, 0.0, 0.0]
            delta[axis] = sign * offset * scale
            pbone.location = mathutils.Vector(pbone.location) + mathutils.Vector(delta)
        applied_bone.append((attr, value))

    # --- Bone rotations (eye gaze/eyelid/eyebrow hinge genes) ---
    applied_rot = []
    for attr, value in attrs.items():
        mapping = ROTATION_MAP.get(attr)
        if not mapping:
            continue
        offset = offset_from_value(value)
        for bone_name, axis, sign, max_rad in mapping:
            pbone = rig_obj.pose.bones.get(bone_name)
            if pbone is None:
                print(f"WARN: bone {bone_name} not found")
                continue
            pbone.rotation_mode = "XYZ"
            euler = list(pbone.rotation_euler)
            euler[axis] += sign * offset * max_rad
            pbone.rotation_euler = euler
        applied_rot.append((attr, value))
    bpy.ops.object.mode_set(mode="OBJECT")

    print(f"Applied blendshapes: {applied_bs}")
    print(f"Applied bone transforms: {applied_bone}")
    print(f"Applied bone rotations: {applied_rot}")
    unmatched = [
        a for a in attrs
        if a not in dict(applied_bs) and a not in dict(applied_bone) and a not in dict(applied_rot)
    ]
    if unmatched:
        print(f"WARN: no mapping found for: {unmatched}")

    setup_camera_and_light(mesh_obj, "Front")
    render(f"{out_prefix}_front.png")

    cam = bpy.context.scene.camera
    verts = [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]
    cy = (min(v.y for v in verts) + max(v.y for v in verts)) / 2
    cz = (min(v.z for v in verts) + max(v.z for v in verts)) / 2
    depth = max(v.y for v in verts) - min(v.y for v in verts)
    cam.location = (depth * 3.5, cy, cz)
    cam.rotation_euler = (math.radians(90), 0, math.radians(90))
    render(f"{out_prefix}_profile.png")


main()
