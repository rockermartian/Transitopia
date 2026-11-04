# SPDX-License-Identifier: GPL-3.0-or-later
"""
Bridge Blender -> Cinema 4D (simulation Rigid Body)
===================================================

Envoie les rigid bodies de la scène Blender à Cinema 4D (c4dpy, sans
interface), fait calculer la simulation par le système Simulation de C4D
(tags Rigid Body / Collider), puis réapplique le résultat sous forme de
keyframes sur les objets Blender existants.

Conversion de repère : Blender (Z en haut, droitier, mètres) <-> Cinema 4D
(Y en haut, gaucher, centimètres) par échange Y/Z et facteur d'unité.
"""

bl_info = {
    "name": "Bridge Cinema 4D (Rigid Body)",
    "author": "Claude",
    "version": (2, 3, 0),
    "blender": (4, 2, 0),
    "location": "Vue 3D > Barre latérale (N) > Bridge C4D",
    "description": "Fait calculer la simulation rigid body par Cinema 4D et "
                   "réapplique le résultat sur les objets Blender",
    "category": "Physics",
}

import array
import glob
import math
import json
import os
import subprocess
import tempfile
import time
import zlib

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)
from mathutils import Matrix, Quaternion, Vector

JOB_ENV = "C4D_BRIDGE_JOB"
ACTION_PREFIX = "C4D_"
_LINEAR = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value

_proc = None          # processus c4dpy en cours
_proc_job = None      # dossier de travail associé
_proc_scene = None    # nom de la scène Blender

RB_SHAPE_MAP = {
    'CONVEX_HULL': "CONVEX_HULLS",
    'MESH': "TRIANGLE_MESH",
    'SPHERE': "SPHERE",
    'BOX': "BOX",
}


# -----------------------------------------------------------------------------
# Conversion de repère
# -----------------------------------------------------------------------------

def _p(v):
    return (v[0], v[2], v[1])


def unit_factor(scene):
    """Centimètres C4D par unité Blender."""
    return 100.0 * scene.unit_settings.scale_length


def to_c4d_matrix(M, k):
    """Matrice monde Blender -> liste [off, v1, v2, v3] dans le repère C4D."""
    R = M.to_3x3()
    c0, c1, c2 = R.col[0], R.col[1], R.col[2]
    t = M.translation
    off = [x * k for x in _p(t)]
    return off + list(_p(c0)) + list(_p(c2)) + list(_p(c1))


def from_c4d_matrix(m, k):
    """[off, v1, v2, v3] C4D -> Matrix monde Blender."""
    off, v1, v2, v3 = m[0:3], m[3:6], m[6:9], m[9:12]
    c0, c1, c2 = _p(v1), _p(v3), _p(v2)
    t = [x / k for x in _p(off)]
    return Matrix(((c0[0], c1[0], c2[0], t[0]),
                   (c0[1], c1[1], c2[1], t[1]),
                   (c0[2], c1[2], c2[2], t[2]),
                   (0.0, 0.0, 0.0, 1.0)))


def force_matrix(M, k):
    """Matrice d'une force : l'axe Z local C4D doit suivre l'axe Z Blender."""
    R = M.to_3x3().normalized()
    x, y, z = R.col[0], R.col[1], R.col[2]
    off = [v * k for v in _p(M.translation)]
    v1 = list(_p(x))
    v2 = [-c for c in _p(y)]
    v3 = list(_p(z))
    return off + v1 + v2 + v3


# -----------------------------------------------------------------------------
# Collecte de la scène
# -----------------------------------------------------------------------------

def rb_objects(scene):
    rbw = scene.rigidbody_world
    if rbw is None or rbw.collection is None:
        return []
    names = set(o.name for o in scene.objects)
    return [o for o in rbw.collection.objects
            if o.type == 'MESH' and o.rigid_body is not None and o.name in names]


def body_type(o):
    rb = o.rigid_body
    if rb.type == 'ACTIVE' and rb.enabled and not rb.kinematic:
        return "ACTIVE"
    return "PASSIVE"


def _own_anim(idb):
    ad = getattr(idb, "animation_data", None)
    if ad is None:
        return False
    if ad.action is not None and not ad.action.name.startswith(ACTION_PREFIX):
        return True
    return len(ad.drivers) > 0 or len(ad.nla_tracks) > 0


def is_animated(o, depth=0):
    """Objet qui bouge réellement (keyframes, drivers, contraintes, parent animé)."""
    if o is None or depth > 16:
        return False
    if _own_anim(o) or len(o.constraints) > 0:
        return True
    return is_animated(o.parent, depth + 1)


def _geom_key(o):
    """Clé de géométrie sans évaluer le mesh : objets partageant le même mesh et
    la même pile de modificateurs -> même géométrie de collision."""
    mods = []
    for m in o.modifiers:
        props = []
        for p in m.bl_rna.properties:
            if p.type in {'POINTER', 'COLLECTION'} or p.identifier in {
                    "rna_type", "is_active", "show_expanded", "persistent_uid",
                    "is_override_data", "execution_time"}:
                continue
            try:
                v = getattr(m, p.identifier)
            except Exception:  # noqa
                continue
            props.append((p.identifier, tuple(v) if hasattr(v, "__len__") and not isinstance(v, str) else v))
        ng = getattr(m, "node_group", None)
        mods.append((m.type, m.name, ng.name_full if ng else "", props))
    return (o.data.name_full, zlib.crc32(repr(mods).encode()))


class _PausedDistributions:
    """Suspend la synchronisation auto de l'add-on « Distribution sur points »
    (s'il est installé) pendant l'export / l'import."""

    def __enter__(self):
        self.saved = []
        for c in bpy.data.collections:
            st = getattr(c, "ptd", None)
            if st is not None and st.is_output and st.live:
                self.saved.append(c.name)
                st.live = False
        return self

    def __exit__(self, *exc):
        for name in self.saved:
            c = bpy.data.collections.get(name)
            if c is not None:
                c.ptd.live = True
        return False


def mesh_payload(o, depsgraph, k):
    ev = o.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        n = len(me.vertices)
        co = [0.0] * (3 * n)
        me.vertices.foreach_get("co", co)
        nt = len(me.loop_triangles)
        tri = [0] * (3 * nt)
        me.loop_triangles.foreach_get("vertices", tri)
    finally:
        ev.to_mesh_clear()
    import numpy as np
    pts = (np.asarray(co, dtype=np.float64).reshape(-1, 3)[:, (0, 2, 1)] * k).round(5)
    # échange Y/Z = symétrie : on inverse l'ordre des sommets des triangles
    tris = np.asarray(tri, dtype=np.int64).reshape(-1, 3)[:, (0, 2, 1)]
    return {"points": pts.ravel().tolist(), "tris": tris.ravel().tolist()}


def sample_surface(o, depsgraph, count, seed=0):
    """Points répartis uniformément (pondérés par l'aire) sur la surface du mesh
    évalué, en coordonnées locales de l'objet."""
    import numpy as np
    ev = o.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        n, nt = len(me.vertices), len(me.loop_triangles)
        if nt == 0:
            return []
        co = np.empty(3 * n)
        me.vertices.foreach_get("co", co)
        tri = np.empty(3 * nt, dtype=np.int64)
        me.loop_triangles.foreach_get("vertices", tri)
    finally:
        ev.to_mesh_clear()
    co = co.reshape(-1, 3)
    a, b_, c = (co[tri.reshape(-1, 3)[:, i]] for i in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b_ - a, c - a), axis=1)
    if area.sum() <= 0:
        return []
    rng = np.random.default_rng(seed)
    pick = rng.choice(nt, size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    pts = a[pick] + (b_[pick] - a[pick]) * u[:, None] + (c[pick] - a[pick]) * v[:, None]
    return [Vector(p) for p in pts]


def axis_column(scene, s, k, cache, name, origin, axis, strength, common,
                auto=True, length=2.0, min_length=0.0, parent=None, points=None,
                warn=True):
    """Colonne d'attracteurs C4D (atténuation 1/d²) le long d'un axe, qui imite
    une attraction vers cet axe. Pour que la force soit dirigée vers l'axe sans
    composante verticale (écrasement en anneau), la colonne doit être bien plus
    longue que la distance des cailloux à l'axe : sa longueur est calculée
    automatiquement à partir de la position des cailloux."""
    axis = axis.normalized()
    key = ("_line", name)
    if key not in cache:
        rel = [b.matrix_world.translation - origin
               for b in rb_objects(scene) if body_type(b) == "ACTIVE"]
        span = max((abs(v.dot(axis)) for v in rel), default=1.0)
        radii = sorted((v - axis * v.dot(axis)).length for v in rel) or [1.0]
        r_ref = max(radii[len(radii) // 2], 1e-3)
        if auto:
            L = max(min_length, 2.0 * 3.0 * max(span, r_ref))
            n = max(s.line_attract_points, min(400, int(math.ceil(L / (r_ref / 4.0))) + 1))
        else:
            L, n = length, (points or s.line_attract_points)
        cache[key] = (L, n, r_ref)
        if warn and L < 4.0 * max(span, r_ref):
            cache.setdefault("_warnings", []).append(
                f"Ligne « {name} » courte ({L:.2f}) par rapport aux cailloux "
                f"(rayon {r_ref:.2f}, demi-hauteur {span:.2f}) : risque "
                f"d'écrasement, active « Longueur auto »")
    L, n, r_ref = cache[key]
    # Intensité par point : densité linéaire λ telle que la force d'une ligne
    # infinie (2λ/r) égale celle d'un point de même intensité totale (S/r²) à la
    # distance typique des cailloux -> même effet quelle que soit la longueur.
    step = L / (n - 1)
    per_point = strength * step / (2.0 * r_ref)
    out = []
    for i in range(n):
        wp = origin + axis * (L * (i / (n - 1) - 0.5))
        f = dict(common, name=f"{name}#{i:03d}", source=name, strength=per_point,
                 falloff="INVERSE_SQUARE", matrix=force_matrix(Matrix.Translation(wp), k))
        if parent:
            f["parent"] = parent
        out.append(f)
    return out


def orbit_axis(scene, s):
    """Axe de la rotation initiale : celui du verre, sinon du premier vortex,
    sinon l'axe Z du monde."""
    if s.attract_object is not None:
        origin, axis, _h = glass_axis(s.attract_object)
        return origin, axis
    for o in scene.objects:
        if o.field and o.field.type == 'VORTEX':
            return o.matrix_world.translation.copy(), o.matrix_world.to_3x3().col[2].normalized()
    return Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))


def glass_axis(o):
    """Axe du verre : axe Z local passant par le centre de sa boîte englobante."""
    from mathutils import Vector as V
    bb = [V(c) for c in o.bound_box]
    c_local = sum(bb, V()) / 8.0
    mw = o.matrix_world
    zs = [(mw @ p) for p in bb]
    axis = mw.to_3x3().col[2].normalized()
    height = max(p.dot(axis) for p in zs) - min(p.dot(axis) for p in zs)
    return mw @ c_local, axis, height


def collect_forces(scene, s, k, cache=None):
    """Forces Blender -> forces C4D. `cache` conserve les points échantillonnés
    sur les meshes attracteurs (mêmes points à chaque image)."""
    rbw = scene.rigidbody_world
    w = rbw.effector_weights
    pool = w.collection.all_objects if w.collection else scene.objects
    forces, skipped = [], []
    if cache is None:
        cache = {}
    for o in pool:
        fld = o.field
        if fld is None or fld.type == 'NONE':
            continue
        if fld.type == 'VORTEX':
            strength = fld.strength * w.vortex * w.all * s.vortex_multiplier
            if abs(strength) < 1e-9 and not _own_anim(o) and not _own_anim(scene):
                continue  # tourbillon nul : rien à envoyer
            if s.invert_vortex:
                strength = -strength
            mw = o.matrix_world
            if strength < 0.0:
                # C4D refuse une vitesse de rotation négative (ramenée à 0) : on
                # inverse le sens en retournant l'objet Rotation (180° autour de X).
                strength = -strength
                mw = mw @ Matrix.Rotation(math.pi, 4, 'X')
            forces.append({"kind": "ROTATION", "name": o.name, "source": o.name,
                           "strength": strength, "matrix": force_matrix(mw, k)})
        elif fld.type == 'FORCE' and s.attract_object is not None:
            cache.setdefault("_ignored_force", set()).add(o.name)
        elif fld.type == 'FORCE':
            # Blender : force négative = attraction ; C4D : positive = attraction
            strength = -fld.strength * w.force * w.all * s.force_multiplier
            common = {"kind": "ATTRACTOR", "speed_limit": s.attractor_speed_limit,
                      "min_distance": s.attract_min_distance * k}
            if o.type == 'MESH' and fld.shape in {'SURFACE', 'POINTS'} and s.mesh_attract_points > 0:
                # Attraction vers un mesh : attracteurs répartis sur sa surface
                if o.name not in cache:
                    dg = bpy.context.evaluated_depsgraph_get()
                    cache[o.name] = sample_surface(o, dg, s.mesh_attract_points)
                pts = cache[o.name]
                for i, p in enumerate(pts):
                    wp = o.matrix_world @ p
                    forces.append(dict(common, name=f"{o.name}#{i:03d}", source=o.name, group="MESH",
                                       strength=strength / len(pts), falloff="INVERSE_SQUARE",
                                       matrix=force_matrix(Matrix.Translation(wp), k)))
            elif fld.shape == 'LINE' and s.line_attract_points > 1:
                mw = o.matrix_world
                forces += axis_column(scene, s, k, cache, o.name, mw.translation,
                                      mw.to_3x3().col[2], strength, common,
                                      auto=s.line_length_auto, length=s.line_length)
            else:
                forces.append(dict(common, name=o.name, source=o.name, strength=strength,
                                   matrix=force_matrix(o.matrix_world, k)))
        else:
            skipped.append(f"{o.name} ({fld.type})")
    g = s.attract_object
    if g is not None and g.type == 'MESH':
        origin, axis, height = glass_axis(g)
        common = {"kind": "ATTRACTOR", "speed_limit": s.attractor_speed_limit,
                  "min_distance": s.attract_min_distance * k}
        L = max(height * s.attract_column_scale, 1e-3)
        n = max(2, s.attract_column_points)
        forces += axis_column(scene, s, k, cache, f"Attraction {g.name}", origin, axis,
                              s.attract_strength, common, auto=False, length=L,
                              points=n, parent=g.name, warn=False)
    return forces, skipped


def default_job_dir(scene):
    s = scene.c4d_bridge
    if s.job_dir:
        return bpy.path.abspath(s.job_dir)
    if bpy.data.filepath:
        base = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
        return os.path.join(os.path.dirname(bpy.data.filepath), f"c4d_bridge_{base}")
    return os.path.join(bpy.app.tempdir or os.path.expanduser("~"), "c4d_bridge")


# -----------------------------------------------------------------------------
# Orbite guidée : trajectoires cibles suivies par les rigid bodies dans C4D
# -----------------------------------------------------------------------------

def _read_fcurves(idb):
    ad = getattr(idb, "animation_data", None)
    if ad is None or ad.action is None:
        return []
    act = ad.action
    if hasattr(act, "slots"):
        from bpy_extras import anim_utils
        if ad.action_slot is None:
            return []
        cb = anim_utils.action_get_channelbag_for_slot(act, ad.action_slot)
        return list(cb.fcurves) if cb else []
    return list(act.fcurves)


def anim_values(idb, path, default, frames):
    """Valeurs d'une propriété (animée ou non) pour chaque image."""
    for fc in _read_fcurves(idb):
        if fc.data_path == path:
            return [fc.evaluate(f) for f in frames]
    return [default] * len(frames)


def glass_profile(g, bins=96):
    """Rayon extérieur du verre selon la hauteur, en coordonnées locales :
    (hauteurs, rayons, centre local de l'axe)."""
    import numpy as np
    dg = bpy.context.evaluated_depsgraph_get()
    ev = g.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        co = np.empty(3 * len(me.vertices))
        me.vertices.foreach_get("co", co)
    finally:
        ev.to_mesh_clear()
    co = co.reshape(-1, 3)
    bb = np.array([list(c) for c in g.bound_box])
    center = bb.mean(0)
    rel = co - center
    h = rel[:, 2]
    r = np.hypot(rel[:, 0], rel[:, 1])
    edges = np.linspace(h.min(), h.max(), bins + 1)
    idx = np.clip(np.digitize(h, edges) - 1, 0, bins - 1)
    rmax = np.full(bins, np.nan)
    for i in range(bins):
        sel = r[idx == i]
        if sel.size:
            rmax[i] = sel.max()
    hc = 0.5 * (edges[:-1] + edges[1:])
    ok = ~np.isnan(rmax)
    rmax = np.interp(hc, hc[ok], rmax[ok])
    return hc, rmax, center


def sample_matrices(scene, objs, f0, f1):
    """Matrices monde de quelques objets à chaque image (simulation Blender et
    synchronisation de la distribution suspendues)."""
    out = {o.name: [] for o in objs}
    rbw = scene.rigidbody_world
    was = rbw.enabled if rbw else None
    back = scene.frame_current
    if rbw:
        rbw.enabled = False
    try:
        with _PausedDistributions():
            animated = any(is_animated(o) for o in objs)
            for f in range(f0, f1 + 1):
                if animated or f == f0:
                    scene.frame_set(f)
                for o in objs:
                    out[o.name].append(o.matrix_world.copy() if (animated or f == f0)
                                       else out[o.name][0])
    finally:
        if rbw:
            rbw.enabled = was
        scene.frame_set(back)
    return out


def tilt_settings(s, G0):
    """(inclinaison, direction) des orbites. Si un empty d'axe est choisi, elles
    sont lues sur son axe Z, exprimé dans le repère du verre :
    axe = Rz(direction) · Rx(inclinaison) · Z."""
    e = s.guided_axis_object
    if e is None:
        return s.guided_tilt, s.guided_tilt_direction
    n = (G0.to_3x3().normalized().inverted() @ e.matrix_world.to_3x3().col[2]).normalized()
    tilt = math.acos(max(-1.0, min(1.0, n.z)))
    direction = math.atan2(n.x, -n.y) if tilt > 1e-6 else 0.0
    return tilt, direction


def axis_vector(tilt, direction):
    """Axe de rotation (repère du verre) correspondant à une inclinaison/direction."""
    return Vector((math.sin(direction) * math.sin(tilt),
                   -math.cos(direction) * math.sin(tilt), math.cos(tilt)))


def guided_targets(scene, s, stones, G0, f0, f1):
    """Tourbillon de cailloux autour du verre, axe FIXE (position du verre à la
    première image : le verre peut ensuite sortir de la spirale).

    Chaque caillou a sa propre vitesse, ondule en hauteur, « respire » contre la
    paroi et arrive au contact à son propre rythme (réglage Variété). Les cailloux
    au-dessus ou en dessous du verre sont ramenés dans sa hauteur.
    Renvoie (positions cibles monde [images, cailloux, 3],
             vitesses initiales monde [cailloux, 3], rotations initiales [cailloux, 4]).
    """
    import numpy as np
    g = s.attract_object
    rng = np.random.default_rng(s.guided_seed)
    frames = list(range(f0, f1 + 1))
    nf, n = len(frames), len(stones)
    fps = scene.render.fps / scene.render.fps_base
    V = s.guided_variety
    speed = np.array(anim_values(scene, "c4d_bridge.guided_speed", s.guided_speed, frames))
    hs, rs, c = glass_profile(g)
    G = np.array(G0)
    Ginv = np.linalg.inv(G)
    gscale = float(np.mean(list(G0.to_scale()))) or 1.0
    P = np.array([list(o.matrix_world.translation) + [1.0] for o in stones])
    q = (Ginv @ P.T).T[:, :3] - c
    r0 = np.hypot(q[:, 0], q[:, 1])
    th0 = np.arctan2(q[:, 1], q[:, 0])
    h0 = q[:, 2]
    half = np.array([max(o.dimensions) * 0.5 for o in stones]) / gscale
    lo, hi = hs[0] + half, hs[-1] - half
    h_t = np.clip(h0, lo, np.maximum(lo, hi))
    H = hs[-1] - hs[0]

    # paramètres propres à chaque caillou
    k_speed = 1.0 + 0.5 * V * rng.uniform(-1, 1, n)
    wob_a = 0.06 * V * H * rng.uniform(0.3, 1.0, n)
    wob_t = rng.uniform(2.0, 5.0, n)
    wob_p = rng.uniform(0, 2 * np.pi, n)
    br_a = 0.6 * V * half * rng.uniform(0.0, 1.0, n)
    br_t = rng.uniform(1.5, 4.0, n)
    br_p = rng.uniform(0, 2 * np.pi, n)
    delay = 0.4 * V * s.guided_approach * rng.uniform(0, 1, n)

    # Orbites inclinées : le caillou monte d'un côté du verre et descend de
    # l'autre (hauteur = centre + A·sin(angle - direction)). A dépend de
    # l'inclinaison et du rayon de contact ; le centre est recalé pour que toute
    # l'orbite reste dans la hauteur du verre (ensemble centré).
    base_tilt, base_dir = tilt_settings(s, G0)
    tilt = base_tilt * (1.0 + 0.3 * V * rng.uniform(-1, 1, n))
    if s.guided_tilt_random:
        tilt_dir = rng.uniform(0, 2 * np.pi, n)
    else:
        tilt_dir = np.full(n, base_dir)
    r_contact = np.interp(h_t, hs, rs) + half
    span = np.maximum(hi - lo, 0.0)
    amp = np.minimum(r_contact * np.tan(np.clip(tilt, 0.0, np.radians(80))), 0.5 * span)
    h_c = np.clip(h_t, lo + amp, np.maximum(lo + amp, hi - amp))

    t = np.arange(nf)[:, None] / fps                                  # (images, 1)
    turns = np.concatenate([[0.0], np.cumsum(2 * np.pi * speed[:-1] / fps)])[:, None]
    theta = th0[None, :] + turns * k_speed[None, :]
    x = np.clip((np.arange(nf)[:, None] - delay[None, :]) / max(s.guided_approach, 1), 0, 1)
    a = x * x * (3 - 2 * x)                                            # approche douce
    h_orbit = h_c + amp * np.sin(theta - tilt_dir)
    h = h0 + (h_orbit - h0) * a + wob_a * np.sin(2 * np.pi * t / wob_t + wob_p) * a
    rc = (np.interp(h, hs, rs) + half + s.guided_margin / gscale
          + br_a * 0.5 * (1 + np.sin(2 * np.pi * t / br_t + br_p)))
    R = r0 + (rc - r0) * a
    local = np.stack([c[0] + R * np.cos(theta), c[1] + R * np.sin(theta), c[2] + h,
                      np.ones_like(R)], -1)
    world = np.einsum("ij,fnj->fni", G, local)[..., :3]

    # vitesse de départ = vitesse de l'orbite (pas d'à-coup), rotation propre aléatoire
    vel = (world[1] - world[0]) * fps if nf > 1 else np.zeros((n, 3))
    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    spin = V * rng.uniform(90.0, 360.0, n)                               # degrés / s
    return world, vel, np.concatenate([axes, spin[:, None]], 1)


def export_job(context, job):
    """Écrit scene.json et le script C4D dans `job`. Renvoie (nb corps, avertissements)."""
    scene = context.scene
    s = scene.c4d_bridge
    rbw = scene.rigidbody_world
    objs = rb_objects(scene)
    if not objs:
        raise RuntimeError("Aucun rigid body dans la scène (Rigid Body World vide)")
    clear_animation(scene, objs, restore_world=False)

    k = unit_factor(scene)
    f0, f1 = scene.frame_start, scene.frame_end
    fps = scene.render.fps / scene.render.fps_base
    warnings = []
    if abs(fps - round(fps)) > 1e-3:
        warnings.append(f"Cadence {fps:.3f} i/s arrondie à {round(fps)} i/s dans C4D")

    frame_back = scene.frame_current
    was_enabled = rbw.enabled
    rbw.enabled = False  # positions de départ sans simulation Blender
    wm = context.window_manager
    wm.progress_begin(0, len(objs))
    pause = _PausedDistributions().__enter__()
    try:
        scene.frame_set(f0)
        depsgraph = context.evaluated_depsgraph_get()
        geoms, geom_index, bodies = [], {}, []
        for n_obj, o in enumerate(objs):
            if n_obj % 50 == 0:
                wm.progress_update(n_obj)
            key = _geom_key(o)
            if key not in geom_index:
                geom_index[key] = len(geoms)
                geoms.append(mesh_payload(o, depsgraph, k))
            rb = o.rigid_body
            btype = body_type(o)
            bodies.append({
                "name": o.name,
                "geom": geom_index[key],
                "type": btype,
                "shape": RB_SHAPE_MAP.get(rb.collision_shape, "CONVEX_HULLS"),
                "mass": rb.mass,
                "friction": rb.friction,
                "bounce": rb.restitution,
                "matrix": to_c4d_matrix(o.matrix_world, k),
                "anim_src": btype == "PASSIVE" and is_animated(o),
            })
            # friction / rebond animés (clés sur le Rigid Body) -> valeur par image
            frames = list(range(f0, f1 + 1))
            for path, key in (("rigid_body.friction", "anim_friction"),
                              ("rigid_body.restitution", "anim_bounce")):
                vals = anim_values(o, path, None, frames)
                if vals[0] is not None and max(vals) - min(vals) > 1e-9:
                    bodies[-1][key] = [round(v, 5) for v in vals]
                    bodies[-1]["friction" if key == "anim_friction" else "bounce"] = vals[0]
        anim_bodies = [b for b in bodies if b.pop("anim_src")]
        n_mat = sum(1 for b in bodies if "anim_friction" in b or "anim_bounce" in b)
        if n_mat:
            warnings.append(f"Friction / rebond animés transmis pour {n_mat} objet(s)")
        if s.mode == 'FORCES' and abs(s.orbit_speed) > 1e-9:
            # Vitesse initiale tangentielle : chaque caillou part en rotation autour
            # de l'axe (comme un disque qui tourne : vitesse = ω × distance à l'axe)
            o_origin, o_axis = orbit_axis(scene, s)
            omega = 2.0 * math.pi * s.orbit_speed
            n_init = 0
            for b in bodies:
                if b["type"] != "ACTIVE":
                    continue
                p = bpy.data.objects[b["name"]].matrix_world.translation - o_origin
                radial = p - o_axis * p.dot(o_axis)
                tang = o_axis.cross(radial)
                if tang.length < 1e-9:
                    continue
                dirv = tang.normalized() * (1.0 if omega > 0 else -1.0)
                b["init_vel"] = list(_p(dirv)) + [abs(omega) * radial.length * k]
                n_init += 1
            warnings.append(f"Rotation initiale : {s.orbit_speed:+.2f} tour/s donnée à {n_init} cailloux")
        fcache = {}
        forces, skipped = (([], []) if s.mode == 'ORBIT'
                           else collect_forces(scene, s, k, fcache))  # image de départ
        if s.mode == 'ORBIT':
            if s.attract_object is None:
                raise RuntimeError("Choisis le verre dans le panneau (champ « Verre »)")
            forces, skipped = [], []
            stones = [bpy.data.objects[b["name"]] for b in bodies if b["type"] == "ACTIVE"]
            G0 = sample_matrices(scene, [s.attract_object], f0, f0)[s.attract_object.name][0]
            tg, vel, ang = guided_targets(scene, s, stones, G0, f0, f1)
            i_st = 0
            for b in bodies:
                if b["type"] != "ACTIVE":
                    continue
                pts = tg[:, i_st, :]
                b["follow"] = [round(v, 3) for p in pts for v in (p[0] * k, p[2] * k, p[1] * k)]
                v = Vector(vel[i_st])
                if v.length > 1e-9:
                    b["init_vel"] = list(_p(v.normalized())) + [v.length * k]
                if ang[i_st, 3] > 0:
                    b["init_ang"] = list(_p(ang[i_st, :3])) + [float(ang[i_st, 3])]
                i_st += 1
            warnings.append(f"Tourbillon : {i_st} cailloux autour de « {s.attract_object.name} » "
                            f"(axe fixe)")
        if skipped:
            warnings.append("Forces non transférées : " + ", ".join(skipped))
        warnings += fcache.get("_warnings", [])
        if fcache.get("_ignored_force"):
            warnings.append("Champs Force Blender ignorés (attraction par le verre) : "
                            + ", ".join(sorted(fcache["_ignored_force"])))
        for key, val in fcache.items():
            if isinstance(key, tuple) and key[0] == "_line":
                what = ("Attraction vers le verre" if key[1].startswith("Attraction ")
                        else f"Ligne « {key[1]} »")
                warnings.append(f"{what} : {val[1]} attracteurs sur {val[0]:.2f} le long de l'axe")
        # Forces animées : keyframes / drivers sur la force (intensité, position,
        # rotation) ou sur la scène (poids des effecteurs, multiplicateurs du bridge)
        forces_anim = bool(forces) and (_own_anim(scene) or any(
            is_animated(bpy.data.objects[n]) for n in {f["source"] for f in forces}
            if n in bpy.data.objects))
        for b in anim_bodies:
            b["anim"] = []
        f_str = [[] for _ in forces]
        f_mat = [[] for _ in forces]
        if anim_bodies or forces_anim:
            for f in range(f0, f1 + 1):
                scene.frame_set(f)
                for b in anim_bodies:
                    b["anim"].append(to_c4d_matrix(bpy.data.objects[b["name"]].matrix_world, k))
                if forces_anim:
                    cur = {fc["name"]: fc for fc in collect_forces(scene, s, k, fcache)[0]}
                    for i, fc in enumerate(forces):
                        now = cur.get(fc["name"], fc)
                        f_str[i].append(now["strength"])
                        f_mat[i].append(now["matrix"])
        for i, fc in enumerate(forces):
            if f_str[i] and max(f_str[i]) - min(f_str[i]) > 1e-9:
                fc["anim_strength"] = f_str[i]
            if fc.get("parent"):
                continue  # rangés sous le verre dans C4D : ils le suivent d'eux-mêmes
            if f_mat[i] and any(abs(a - c) > 1e-6 for m in f_mat[i] for a, c in zip(m, f_mat[i][0])):
                fc["anim_matrix"] = f_mat[i]
        n_anim = len({fc["source"] for fc in forces if "anim_strength" in fc or "anim_matrix" in fc})
        if n_anim:
            warnings.append(f"{n_anim} force(s) animée(s) transmise(s) image par image")
        n_mesh = len({fc["source"] for fc in forces if fc.get("group") == "MESH"})
        if n_mesh:
            warnings.append(f"{n_mesh} attracteur(s) mesh → {s.mesh_attract_points} points chacun")
    finally:
        rbw.enabled = was_enabled
        scene.frame_set(frame_back)
        pause.__exit__(None, None, None)
        wm.progress_end()

    g = scene.gravity
    gravity = 0.0
    if scene.use_gravity:
        gravity = g.length * rbw.effector_weights.gravity * rbw.effector_weights.all * k
        if g.length > 1e-6 and g.normalized().z > -0.99:
            warnings.append("Gravité non verticale : seule son intensité est transmise")

    cfg = {
        "fps": round(fps),
        "frame_start": f0,
        "frame_end": f1,
        "unit_factor": k,
        "follow": {"strength": s.guided_follow, "damping": s.guided_damping,
                   "rot_strength": 0.0},
        "simulation": {"substeps": s.substeps, "iterations": s.iterations,
                       "gravity": gravity, "gpu": s.use_gpu, "damping": s.sim_damping},
        "geometries": geoms,
        "bodies": bodies,
        "forces": forces,
        "save_c4d": s.save_c4d,
        "blender_scene": scene.name,
    }
    os.makedirs(job, exist_ok=True)
    for fn in ("result.bin", "result.json", "progress.json", "c4d_log.txt"):
        p = os.path.join(job, fn)
        if os.path.exists(p):
            os.remove(p)
    with open(os.path.join(job, "scene.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    with open(os.path.join(job, "c4d_bridge_runner.py"), "w", encoding="utf-8") as fh:
        fh.write(RUNNER_SOURCE)
    n_active = sum(1 for b in bodies if b["type"] == "ACTIVE")
    return n_active, len(bodies) - n_active, len(geoms), warnings


# -----------------------------------------------------------------------------
# Import du résultat / nettoyage
# -----------------------------------------------------------------------------

def _fcurves_for(obj, action):
    """Collection de F-curves compatible Blender 4.2 -> 5.x (actions à slots)."""
    ad = obj.animation_data
    if hasattr(action, "slots"):
        from bpy_extras import anim_utils
        slot = ad.action_slot
        if slot is None:
            slot = action.slots.new('OBJECT', obj.name)
            ad.action_slot = slot
        return anim_utils.action_ensure_channelbag_for_slot(action, slot).fcurves
    return action.fcurves


def _set_curve(fcurves, path, index, frames, values):
    fc = fcurves.new(path, index=index)
    n = len(frames)
    fc.keyframe_points.add(n)
    co = [0.0] * (2 * n)
    co[0::2] = frames
    co[1::2] = values
    fc.keyframe_points.foreach_set("co", co)
    fc.keyframe_points.foreach_set("interpolation", [_LINEAR] * n)
    fc.update()


def clear_animation(scene, objs=None, restore_world=True):
    """Retire l'animation importée de C4D et restaure les objets."""
    if objs is None:
        objs = [o for o in scene.objects if "c4db_rest" in o]
    for o in objs:
        if "c4db_rest" not in o:
            continue
        ad = o.animation_data
        act = ad.action if ad else None
        if act is not None and act.name.startswith(ACTION_PREFIX):
            ad.action = None
            if act.users == 0:
                bpy.data.actions.remove(act)
        prev = o.get("c4db_prev_action", "")
        if prev and prev in bpy.data.actions and ad is not None:
            ad.action = bpy.data.actions[prev]
        o.rotation_mode = o.get("c4db_rotmode", 'XYZ')
        rest = o["c4db_rest"]
        o.matrix_basis = Matrix([rest[0:4], rest[4:8], rest[8:12], rest[12:16]])
        for key in ("c4db_rest", "c4db_prev_action", "c4db_rotmode"):
            if key in o:
                del o[key]
    if restore_world and "c4db_rbw_enabled" in scene:
        if scene.rigidbody_world:
            scene.rigidbody_world.enabled = bool(scene["c4db_rbw_enabled"])
        del scene["c4db_rbw_enabled"]


def import_result(context, job):
    scene = context.scene
    s = scene.c4d_bridge
    with open(os.path.join(job, "result.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(job, "scene.json")) as fh:
        k = json.load(fh)["unit_factor"]
    names = meta["bodies"]
    f0, f1 = meta["frame_start"], meta["frame_end"]
    nf, nb = f1 - f0 + 1, len(names)
    data = array.array("d")
    with open(os.path.join(job, "result.bin"), "rb") as fh:
        data.fromfile(fh, nf * nb * 12)
    frames = list(range(f0, f1 + 1))
    missing = 0
    with _PausedDistributions():
        missing = _apply_result(scene, names, data, nf, nb, frames, k)
    rbw = scene.rigidbody_world
    if s.disable_blender_sim and rbw is not None:
        if "c4db_rbw_enabled" not in scene:
            scene["c4db_rbw_enabled"] = rbw.enabled
        rbw.enabled = False
    scene.frame_set(scene.frame_current)
    return nb - missing, missing, meta


def _apply_result(scene, names, data, nf, nb, frames, k):
    missing = 0
    for j, name in enumerate(names):
        o = bpy.data.objects.get(name)
        if o is None:
            missing += 1
            continue
        clear_animation(scene, [o], restore_world=False)
        rest = [v for row in o.matrix_basis for v in row]
        ad = o.animation_data_create()
        o["c4db_rest"] = rest
        o["c4db_rotmode"] = o.rotation_mode
        o["c4db_prev_action"] = ad.action.name if ad.action else ""
        parent_inv = None
        if o.parent is not None:
            parent_inv = (o.parent.matrix_world @ o.matrix_parent_inverse).inverted_safe()
        locs = [[], [], []]
        rots = [[], [], [], []]
        prev_q = None
        scale = None
        for f in range(nf):
            base = (f * nb + j) * 12
            M = from_c4d_matrix(data[base:base + 12], k)
            if parent_inv is not None:
                M = parent_inv @ M
            loc, q, sc = M.decompose()
            if scale is None:
                scale = sc
            if prev_q is not None and prev_q.dot(q) < 0.0:
                q.negate()
            prev_q = q
            for i in range(3):
                locs[i].append(loc[i])
            for i in range(4):
                rots[i].append(q[i])
        o.rotation_mode = 'QUATERNION'
        o.scale = scale
        act = bpy.data.actions.new(ACTION_PREFIX + o.name)
        ad.action = act
        fcs = _fcurves_for(o, act)
        for i in range(3):
            _set_curve(fcs, "location", i, frames, locs[i])
        for i in range(4):
            _set_curve(fcs, "rotation_quaternion", i, frames, rots[i])
    return missing


# -----------------------------------------------------------------------------
# Lancement de c4dpy et suivi
# -----------------------------------------------------------------------------

def launcher_dir():
    """Dossier au chemin 100 % ASCII pour le script lancé par c4dpy : c4dpy
    convertit le chemin du script en URL et échoue sur les accents."""
    cands = [os.path.join(tempfile.gettempdir(), "c4d_bridge"),
             os.path.join(os.environ.get("PUBLIC", r"C:\Users\Public"), "c4d_bridge"),
             os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "c4d_bridge")]
    for d in cands:
        if not d.isascii():
            continue
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, "probe.tmp")
            with open(probe, "w") as fh:
                fh.write("ok")
            os.remove(probe)
            return d
        except OSError:
            continue
    return None


def write_launcher(job):
    """Écrit le script C4D et job.txt (chemin du dossier d'échange, UTF-8) dans
    un dossier ASCII. Renvoie le chemin du script à lancer."""
    d = launcher_dir()
    if d is None:
        if not job.isascii():
            raise RuntimeError("Aucun dossier sans accent disponible pour lancer c4dpy : "
                               "choisis un dossier d'échange sans accent")
        d = job
    script = os.path.join(d, "c4d_bridge_runner.py")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(RUNNER_SOURCE)
    with open(os.path.join(d, "job.txt"), "w", encoding="utf-8") as fh:
        fh.write(job)
    return script


def find_c4dpy():
    pats = [r"C:\Program Files\Maxon*\c4dpy.exe",
            r"C:\Program Files\Maxon*\*\c4dpy.exe",
            "/Applications/Maxon Cinema 4D*/c4dpy.app/Contents/MacOS/c4dpy"]
    found = []
    for p in pats:
        found += glob.glob(p)
    found.sort(reverse=True)  # version la plus récente d'abord
    return found[0] if found else ""


def prefs():
    return bpy.context.preferences.addons[__name__].preferences


def _redraw():
    wm = bpy.context.window_manager
    for w in wm.windows:
        for a in w.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()


def _log_tail(job, n=6):
    p = os.path.join(job, "c4d_log.txt")
    if not os.path.exists(p):
        return ""
    with open(p, encoding="utf-8", errors="replace") as fh:
        lines = [l.strip() for l in fh if l.strip()]
    return " | ".join(lines[-n:])


def _poll_process():
    global _proc, _proc_job, _proc_scene
    scene = bpy.data.scenes.get(_proc_scene) if _proc_scene else None
    if _proc is None or scene is None:
        return None
    s = scene.c4d_bridge
    try:
        with open(os.path.join(_proc_job, "progress.json"), "rb") as fh:
            st = json.loads(fh.read().decode("utf-8"))
        if st.get("total"):
            s.progress = st.get("frame", 0) / st["total"]
            if st.get("state") == "simulating":
                s.status = f"Simulation C4D : image {st['frame']} / {st['total']}"
            elif st.get("state") == "building":
                s.status = "Construction de la scène dans C4D…"
    except (OSError, ValueError):
        pass
    code = _proc.poll()
    if code is None:
        _redraw()
        return 0.5
    job = _proc_job
    _proc = None
    s.running = False
    if os.path.exists(os.path.join(job, "result.bin")):
        try:
            with bpy.context.temp_override(scene=scene):
                n, missing, meta = import_result(bpy.context, job)
            s.status = (f"Terminé : {n} objets animés"
                        f" (simulation C4D {meta.get('sim_seconds', 0):.1f} s)")
            if not meta.get("moved", True):
                s.status += " — attention, aucun objet n'a bougé"
            if "ATTENTION : cette version de Cinema 4D ne propose pas d'atténuation" in _log_tail(job, 10000):
                s.status += " — ATTENTION : attracteurs sans atténuation dans C4D (voir c4d_log.txt)"
            if missing:
                s.status += f", {missing} introuvable(s)"
        except Exception as e:  # noqa
            s.status = f"Erreur à l'import : {e}"
    else:
        tail = _log_tail(job)
        s.status = f"Échec (code {code}). {tail or 'Voir la console de c4dpy.'}"
    s.progress = 0.0
    _redraw()
    return None


# -----------------------------------------------------------------------------
# Propriétés
# -----------------------------------------------------------------------------

class C4DB_Preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    c4dpy_path: StringProperty(
        name="c4dpy", subtype='FILE_PATH',
        description="Chemin de c4dpy.exe (dossier d'installation de Cinema 4D)")

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "c4dpy_path")
        col.operator(C4DB_OT_detect.bl_idname, icon='VIEWZOOM')
        col.label(text="Premier lancement : ouvrir c4dpy.exe une fois à la main "
                       "pour se connecter au compte Maxon.", icon='INFO')


class C4DB_Settings(bpy.types.PropertyGroup):
    substeps: IntProperty(name="Sous-pas", default=10, min=1, max=200,
                          description="Sous-pas par image de la scène Simulation C4D")
    iterations: IntProperty(name="Itérations", default=10, min=1, max=500,
                            description="Itérations du solveur C4D")
    use_gpu: BoolProperty(name="Calcul GPU", default=False,
                          description="Utilise le GPU pour la simulation C4D")
    vortex_multiplier: FloatProperty(
        name="Tourbillon ×", default=20.0, soft_min=0.0, soft_max=500.0,
        description="Intensité Blender (Vortex) × ce facteur = vitesse de l'objet "
                    "Rotation C4D. Les deux logiciels n'ont pas les mêmes unités : "
                    "à ajuster par essais")
    invert_vortex: BoolProperty(
        name="Inverser le sens", default=False,
        description="Inverse le sens de rotation du tourbillon dans C4D")
    force_multiplier: FloatProperty(
        name="Attraction ×", default=20.0, soft_min=0.0, soft_max=500.0,
        description="Intensité Blender (Force, négative = attire) × ce facteur = "
                    "intensité de l'Attracteur C4D")
    attract_object: PointerProperty(
        name="Verre attracteur", type=bpy.types.Object,
        poll=lambda self, o: o.type == 'MESH',
        description="Objet (le verre) vers lequel les cailloux sont attirés. Le bridge "
                    "crée dans C4D un groupe d'attracteurs le long de son axe, enfant du "
                    "verre : il le suit quand il bouge. Les champs Force de Blender sont "
                    "alors ignorés")
    attract_strength: FloatProperty(
        name="Intensité d'attraction", default=500.0, soft_min=-5000.0, soft_max=5000.0,
        description="Positive = attire vers l'axe du verre, négative = repousse. "
                    "Animable (touche I) : passe en négatif quand le verre avance")
    attract_column_scale: FloatProperty(
        name="Longueur de la colonne", default=1.0, min=0.1, soft_max=5.0,
        subtype='FACTOR',
        description="Longueur de la colonne d'attracteurs, en multiple de la hauteur du "
                    "verre (1 = hauteur du verre). Plus longue : les cailloux gardent mieux "
                    "leur hauteur mais ceux au-dessus et en dessous du verre restent "
                    "attirés vers l'axe. Plus courte : ils sont ramenés vers le verre")
    attract_column_points: IntProperty(
        name="Points de la colonne", default=41, min=2, max=400,
        description="Nombre d'attracteurs le long de la colonne")
    mode: EnumProperty(
        name="Mode",
        items=[('ORBIT', "Tourbillon", "Les cailloux tournent autour du verre en suivant "
                                       "des trajectoires (avec collisions). Recommandé"),
               ('FORCES', "Champs de force", "Mode avancé : vortex et attracteurs")],
        default='ORBIT')
    guided_speed: FloatProperty(
        name="Vitesse (tours/s)", default=0.3, soft_min=-2.0, soft_max=2.0,
        precision=2, step=1,
        description="Vitesse moyenne de rotation des cailloux autour du verre, en tours "
                    "par seconde (négatif = autre sens). Animable (touche I)")
    guided_approach: IntProperty(
        name="Approche (images)", default=40, min=1, soft_max=250,
        description="Durée pour que les cailloux viennent recouvrir le verre")
    guided_variety: FloatProperty(
        name="Variété", default=0.6, min=0.0, max=1.0, subtype='FACTOR',
        description="Désordre du mouvement : vitesses différentes, ondulations en "
                    "hauteur, arrivées décalées, cailloux qui tournent sur eux-mêmes. "
                    "0 = mouvement mécanique")
    guided_margin: FloatProperty(
        name="Pression sur le verre", default=0.0, soft_min=-0.5, soft_max=0.5,
        subtype='DISTANCE',
        description="Négatif : les cibles sont légèrement dans le verre, les cailloux "
                    "appuient dessus et frottent davantage. Positif : couche plus lâche")
    guided_tilt: FloatProperty(
        name="Inclinaison", default=0.4363, min=0.0, max=1.3963, subtype='ANGLE',
        description="Inclinaison des orbites par rapport à l'horizontale : les "
                    "cailloux montent d'un côté du verre et descendent de l'autre. "
                    "0 = rotation horizontale")
    guided_tilt_random: BoolProperty(
        name="Inclinaisons variées", default=True,
        description="Chaque caillou a sa propre direction de diagonale. Décoché : tous "
                    "partagent la même (disque penché)")
    guided_tilt_direction: FloatProperty(
        name="Direction", default=0.0, subtype='ANGLE',
        description="Direction commune de l'inclinaison autour du verre (si les "
                    "inclinaisons ne sont pas variées)")
    guided_axis_object: PointerProperty(
        name="Axe (empty)", type=bpy.types.Object,
        poll=lambda self, o: o.type == 'EMPTY',
        description="Empty dont l'axe Z (la flèche) donne l'axe de rotation des "
                    "cailloux : il remplace les réglages Inclinaison et Direction")
    guided_seed: IntProperty(name="Graine", default=0, min=0,
                             description="Change le tirage aléatoire du mouvement")
    guided_follow: FloatProperty(
        name="Force de suivi", default=10.0, min=0.0, soft_max=100.0,
        description="« Suivre la position » du tag Rigid Body de C4D. Plus faible : "
                    "mouvement plus libre et chaotique ; plus fort : trajectoires respectées")
    guided_damping: FloatProperty(
        name="Amortissement du suivi", default=0.0, min=0.0, soft_max=100.0,
        description="Évite les oscillations autour des trajectoires (0 = défaut C4D)")
    orbit_speed: FloatProperty(
        name="Rotation initiale", default=0.0, soft_min=-2.0, soft_max=2.0,
        precision=3, step=1,
        description="Vitesse de rotation donnée aux cailloux au départ, en tours par "
                    "seconde, autour de l'axe du verre (négatif = autre sens). Contrairement "
                    "au tourbillon, elle ne s'emballe pas : l'attraction retient les "
                    "cailloux en orbite. 0 = désactivé")
    mesh_attract_points: IntProperty(
        name="Points par mesh attracteur", default=64, min=0, max=1000,
        description="Un champ Force posé sur un mesh (forme Surface ou Chaque point) "
                    "devient autant d'attracteurs C4D répartis sur sa surface. Plus de "
                    "points = attraction plus fidèle à la forme")
    line_attract_points: IntProperty(
        name="Points sur la ligne", default=24, min=2, max=500,
        description="Champ Force de forme Ligne : nombre minimal d'attracteurs C4D le "
                    "long de son axe Z (en mode auto, augmenté pour que les points soient "
                    "assez rapprochés)")
    line_length_auto: BoolProperty(
        name="Longueur auto", default=True,
        description="Calcule la longueur de la ligne (6 fois la plus grande des deux "
                    "valeurs : demi-hauteur des cailloux, distance des cailloux à l'axe) et "
                    "le nombre de points. Indispensable pour éviter l'écrasement en anneau")
    line_length: FloatProperty(
        name="Longueur de la ligne", default=2.0, min=0.0, subtype='DISTANCE',
        description="Longueur de la colonne d'attracteurs (centrée sur la force). "
                    "Prends un peu plus que la hauteur occupée par les cailloux")
    sim_damping: FloatProperty(
        name="Amortissement", default=0.0, min=0.0, soft_max=1.0,
        description="Amortissement global de la simulation C4D : freine les cailloux "
                    "pour une rotation régulière au lieu d'une accélération sans fin "
                    "(0 = valeur C4D par défaut)")
    attract_min_distance: FloatProperty(
        name="Distance min. d'attraction", default=0.0, min=0.0, subtype='DISTANCE',
        description="Rayon autour de chaque attracteur où il n'agit plus : évite les "
                    "accélérations explosives au contact (0 = valeur C4D par défaut)")
    attractor_speed_limit: FloatProperty(
        name="Vitesse max attracteur", default=0.0, min=0.0,
        description="Limite de vitesse de l'attracteur C4D (0 = valeur par défaut "
                    "de C4D)")
    save_c4d: BoolProperty(
        name="Garder la scène .c4d", default=True,
        description="Enregistre la scène construite pour pouvoir l'ouvrir dans C4D")
    disable_blender_sim: BoolProperty(
        name="Couper la simulation Blender", default=True,
        description="Désactive le Rigid Body World de Blender après l'import, "
                    "sinon il recalculerait par-dessus le résultat de C4D")
    job_dir: StringProperty(
        name="Dossier d'échange", subtype='DIR_PATH',
        description="Vide = dossier « c4d_bridge_<fichier> » à côté du .blend")
    running: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})
    progress: FloatProperty(default=0.0, options={'HIDDEN', 'SKIP_SAVE'})
    status: StringProperty(default="", options={'SKIP_SAVE'})


# -----------------------------------------------------------------------------
# Opérateurs
# -----------------------------------------------------------------------------

class C4DB_OT_detect(bpy.types.Operator):
    bl_idname = "c4dbridge.detect"
    bl_label = "Détecter c4dpy"
    bl_description = "Cherche c4dpy dans les dossiers d'installation de Maxon"

    def execute(self, context):
        p = find_c4dpy()
        if not p:
            self.report({'WARNING'}, "c4dpy introuvable : indique le chemin à la main")
            return {'CANCELLED'}
        prefs().c4dpy_path = p
        self.report({'INFO'}, f"c4dpy trouvé : {p}")
        return {'FINISHED'}


class C4DB_OT_simulate(bpy.types.Operator):
    bl_idname = "c4dbridge.simulate"
    bl_label = "Simuler dans Cinema 4D"
    bl_description = ("Envoie la scène à Cinema 4D (en arrière-plan), calcule la "
                      "simulation et réapplique le résultat")

    @classmethod
    def poll(cls, context):
        return _proc is None

    def execute(self, context):
        global _proc, _proc_job, _proc_scene
        exe = bpy.path.abspath(prefs().c4dpy_path) or find_c4dpy()
        if not exe or not os.path.isfile(exe):
            self.report({'ERROR'}, "c4dpy introuvable : renseigne son chemin dans "
                                   "les préférences de l'add-on")
            return {'CANCELLED'}
        prefs().c4dpy_path = exe
        scene = context.scene
        s = scene.c4d_bridge
        job = default_job_dir(scene)
        t_exp = time.time()
        try:
            na, npas, ng, warnings = export_job(context, job)
        except Exception as e:  # noqa
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        t_exp = time.time() - t_exp
        for w in warnings:
            self.report({'WARNING'}, w)
        try:
            script = write_launcher(job)
        except Exception as e:  # noqa
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        env = os.environ.copy()
        env[JOB_ENV] = job
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0
        try:
            _proc = subprocess.Popen([exe, script], env=env,
                                     cwd=os.path.dirname(script), creationflags=flags)
        except OSError as e:
            self.report({'ERROR'}, f"Impossible de lancer c4dpy : {e}")
            return {'CANCELLED'}
        _proc_job, _proc_scene = job, scene.name
        s.running, s.progress = True, 0.0
        s.status = (f"Démarrage de C4D… ({na} corps actifs, {npas} colliders, "
                    f"{ng} géométries, export {t_exp:.1f} s)")
        bpy.app.timers.register(_poll_process, first_interval=1.0)
        return {'FINISHED'}


class C4DB_OT_cancel(bpy.types.Operator):
    bl_idname = "c4dbridge.cancel"
    bl_label = "Annuler"
    bl_description = "Arrête la simulation en cours dans Cinema 4D"

    def execute(self, context):
        global _proc
        if _proc is not None:
            _proc.kill()
            _proc = None
        s = context.scene.c4d_bridge
        s.running, s.progress, s.status = False, 0.0, "Simulation annulée"
        return {'FINISHED'}


class C4DB_OT_export(bpy.types.Operator):
    bl_idname = "c4dbridge.export"
    bl_label = "Exporter seulement"
    bl_description = ("Prépare le dossier d'échange pour lancer le script à la main "
                      "dans C4D (Extensions > Gestionnaire de scripts)")

    def execute(self, context):
        job = default_job_dir(context.scene)
        try:
            na, npas, ng, warnings = export_job(context, job)
        except Exception as e:  # noqa
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        for w in warnings:
            self.report({'WARNING'}, w)
        context.scene.c4d_bridge.status = f"Exporté dans {job}"
        self.report({'INFO'}, f"{na} corps exportés dans {job}")
        return {'FINISHED'}


class C4DB_OT_import(bpy.types.Operator):
    bl_idname = "c4dbridge.import_result"
    bl_label = "Importer le résultat"
    bl_description = "Applique le dernier résultat calculé par Cinema 4D"

    def execute(self, context):
        job = default_job_dir(context.scene)
        if not os.path.exists(os.path.join(job, "result.bin")):
            self.report({'ERROR'}, f"Aucun résultat dans {job}")
            return {'CANCELLED'}
        n, missing, _meta = import_result(context, job)
        context.scene.c4d_bridge.status = f"Importé : {n} objets animés"
        return {'FINISHED'}


class C4DB_OT_make_axis(bpy.types.Operator):
    bl_idname = "c4dbridge.make_axis"
    bl_label = "Créer l'empty d'axe"
    bl_description = ("Crée une flèche au centre du verre, orientée selon l'inclinaison "
                      "actuelle. Tourne-la ensuite pour régler l'inclinaison (rotation X) "
                      "et la direction (rotation Z)")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        s = scene.c4d_bridge
        g = s.attract_object
        if g is None:
            self.report({'ERROR'}, "Choisis d'abord le verre")
            return {'CANCELLED'}
        f0 = scene.frame_start
        G0 = sample_matrices(scene, [g], f0, f0)[g.name][0]
        origin, _axis, height = glass_axis(g)
        tilt, direction = s.guided_tilt, s.guided_tilt_direction
        Rg = G0.to_3x3().normalized()
        # orientation : Z de l'empty = axe des orbites (rotation X puis Z, repère du verre)
        local = (Matrix.Rotation(direction, 3, 'Z') @ Matrix.Rotation(tilt, 3, 'X'))
        e = bpy.data.objects.new("Axe tourbillon", None)
        e.empty_display_type = 'SINGLE_ARROW'
        e.empty_display_size = max(height * 0.6, 0.1)
        e.show_in_front = True
        e.matrix_world = Matrix.Translation(origin) @ (Rg @ local).to_4x4()
        (g.users_collection[0] if g.users_collection else scene.collection).objects.link(e)
        s.guided_axis_object = e
        for o in context.selected_objects:
            o.select_set(False)
        e.select_set(True)
        context.view_layer.objects.active = e
        self.report({'INFO'}, "Tourne l'empty : rotation X = inclinaison, rotation Z = direction")
        return {'FINISHED'}


class C4DB_OT_preview_orbit(bpy.types.Operator):
    bl_idname = "c4dbridge.preview_orbit"
    bl_label = "Aperçu des trajectoires"
    bl_description = ("Anime les cailloux sur leurs trajectoires cibles dans Blender, "
                      "sans physique, pour régler vitesse et écart avant de simuler")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        s = scene.c4d_bridge
        if s.attract_object is None:
            self.report({'ERROR'}, "Choisis le verre dans « Verre »")
            return {'CANCELLED'}
        stones = [o for o in rb_objects(scene) if body_type(o) == "ACTIVE"]
        clear_animation(scene, stones, restore_world=False)
        f0, f1 = scene.frame_start, scene.frame_end
        G0 = sample_matrices(scene, [s.attract_object], f0, f0)[s.attract_object.name][0]
        tg, _vel, _ang = guided_targets(scene, s, stones, G0, f0, f1)
        k = unit_factor(scene)
        nf, nb = tg.shape[0], tg.shape[1]
        data = array.array("d")
        rest = [o.matrix_world.copy() for o in stones]
        for f in range(nf):
            for j, o in enumerate(stones):
                M = rest[j].copy()
                M.translation = Vector(tg[f, j])
                data.extend(to_c4d_matrix(M, k))
        with _PausedDistributions():
            _apply_result(scene, [o.name for o in stones], data, nf, nb,
                          list(range(f0, f1 + 1)), k)
        rbw = scene.rigidbody_world
        if rbw is not None:
            if "c4db_rbw_enabled" not in scene:
                scene["c4db_rbw_enabled"] = rbw.enabled
            rbw.enabled = False
        scene.frame_set(scene.frame_current)
        s.status = "Aperçu des trajectoires (sans physique) — « Effacer » pour revenir"
        return {'FINISHED'}


class C4DB_OT_clear(bpy.types.Operator):
    bl_idname = "c4dbridge.clear"
    bl_label = "Effacer la simulation C4D"
    bl_description = ("Retire l'animation importée, remet les objets à leur position "
                      "de départ et réactive la simulation Blender")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        clear_animation(context.scene)
        context.scene.c4d_bridge.status = "Animation C4D retirée"
        return {'FINISHED'}


class C4DB_OT_open_folder(bpy.types.Operator):
    bl_idname = "c4dbridge.open_folder"
    bl_label = "Ouvrir le dossier"
    bl_description = "Ouvre le dossier d'échange (scène .c4d, journal)"

    def execute(self, context):
        job = default_job_dir(context.scene)
        os.makedirs(job, exist_ok=True)
        bpy.ops.wm.path_open(filepath=job)
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------

class C4DB_PT_panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Bridge C4D"
    bl_label = "Simulation Cinema 4D"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        s = scene.c4d_bridge
        objs = rb_objects(scene)
        na = sum(1 for o in objs if body_type(o) == "ACTIVE")
        layout.label(text=f"{na} corps actifs, {len(objs) - na} colliders",
                     icon='PHYSICS')
        if not prefs().c4dpy_path:
            box = layout.box()
            box.label(text="Chemin de c4dpy non défini", icon='ERROR')
            box.operator(C4DB_OT_detect.bl_idname, icon='VIEWZOOM')

        if s.running:
            if hasattr(layout, "progress"):
                layout.progress(factor=s.progress, type='BAR', text=s.status)
            else:
                layout.label(text=s.status)
            layout.operator(C4DB_OT_cancel.bl_idname, icon='CANCEL')
        else:
            row = layout.row()
            row.scale_y = 1.6
            row.operator(C4DB_OT_simulate.bl_idname, icon='PLAY')
            if s.status:
                layout.label(text=s.status, icon='INFO')

        box = layout.box()
        box.row().prop(s, "mode", expand=True)
        box.prop(s, "attract_object", text="Verre")
        if s.mode == 'ORBIT':
            col = box.column(align=True)
            col.prop(s, "guided_speed")
            col.prop(s, "guided_approach")
            col.prop(s, "guided_variety")
            col.prop(s, "guided_margin")
            col = box.column(align=True)
            col.prop(s, "guided_axis_object", text="Axe (empty)")
            if s.guided_axis_object is None:
                col.prop(s, "guided_tilt")
                if not s.guided_tilt_random:
                    col.prop(s, "guided_tilt_direction")
                col.operator(C4DB_OT_make_axis.bl_idname, icon='EMPTY_SINGLE_ARROW')
            elif s.attract_object is not None:
                G0 = s.attract_object.matrix_world
                t, d = tilt_settings(s, G0)
                col.label(text=f"Inclinaison {math.degrees(t):.0f}°, direction "
                               f"{math.degrees(d) % 360:.0f}° (lues sur l'empty)",
                          icon='EMPTY_SINGLE_ARROW')
            col.prop(s, "guided_tilt_random")
            col = box.column(align=True)
            col.prop(s, "guided_seed")
            box.label(text="Axe fixe : anime le verre pour qu'il en sorte", icon='INFO')
            box.operator(C4DB_OT_preview_orbit.bl_idname, icon='PLAY')
        else:
            if s.attract_object is not None:
                col = box.column(align=True)
                col.prop(s, "attract_strength")
                col.prop(s, "attract_column_scale")
                col.prop(s, "attract_column_points")
                col.prop(s, "attract_min_distance")
            col = box.column(align=True)
            col.label(text="Rotation")
            col.prop(s, "orbit_speed", text="Rotation initiale (tours/s)")
            col.prop(s, "vortex_multiplier", text="Tourbillon × (Vortex Blender)")
            col.prop(s, "invert_vortex")
            if s.attract_object is None:
                col = box.column(align=True)
                col.label(text="Champs Force Blender → Attracteurs C4D")
                col.prop(s, "force_multiplier")
                col.prop(s, "attractor_speed_limit")
                col.prop(s, "mesh_attract_points")
                col.prop(s, "line_attract_points")
                col.prop(s, "line_length_auto")
                sub = col.row()
                sub.enabled = not s.line_length_auto
                sub.prop(s, "line_length")
                col.prop(s, "attract_min_distance")

        box = layout.box()
        box.label(text="Simulation C4D", icon='SETTINGS')
        col = box.column(align=True)
        col.prop(s, "substeps")
        col.prop(s, "iterations")
        if s.mode == 'ORBIT':
            col = box.column(align=True)
            col.prop(s, "guided_follow")
            col.prop(s, "guided_damping")
        box.prop(s, "sim_damping")
        box.prop(s, "use_gpu")

        box = layout.box()
        box.label(text="Résultat", icon='ACTION')
        box.prop(s, "disable_blender_sim")
        box.operator(C4DB_OT_clear.bl_idname, icon='TRASH')
        row = box.row(align=True)
        row.operator(C4DB_OT_import.bl_idname, icon='IMPORT')
        row.operator(C4DB_OT_export.bl_idname, icon='EXPORT')

        box = layout.box()
        box.prop(s, "job_dir")
        box.prop(s, "save_c4d")
        box.operator(C4DB_OT_open_folder.bl_idname, icon='FILE_FOLDER')


# -----------------------------------------------------------------------------
# Enregistrement
# -----------------------------------------------------------------------------

classes = (C4DB_Preferences, C4DB_Settings, C4DB_OT_detect, C4DB_OT_simulate,
           C4DB_OT_preview_orbit, C4DB_OT_make_axis,
           C4DB_OT_cancel, C4DB_OT_export, C4DB_OT_import, C4DB_OT_clear,
           C4DB_OT_open_folder, C4DB_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.c4d_bridge = PointerProperty(type=C4DB_Settings)


def unregister():
    global _proc
    if _proc is not None:
        _proc.kill()
        _proc = None
    if bpy.app.timers.is_registered(_poll_process):
        bpy.app.timers.unregister(_poll_process)
    del bpy.types.Scene.c4d_bridge
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


# -----------------------------------------------------------------------------
# Script exécuté par Cinema 4D (écrit dans le dossier d'échange à chaque envoi)
# -----------------------------------------------------------------------------

RUNNER_SOURCE = r'''# -*- coding: utf-8 -*-
# Bridge Blender -> Cinema 4D : simulation Rigid Body
# ---------------------------------------------------
# Ce script est exécuté par Cinema 4D (c4dpy en arrière-plan, ou Gestionnaire
# de scripts). Il lit le dossier de travail écrit par l'add-on Blender
# (scene.json), reconstruit la scène avec les tags Rigid Body / Collider du
# système Simulation, calcule la simulation image par image et écrit les
# matrices de chaque corps dans result.bin.
#
# Toutes les coordonnées reçues sont déjà dans le repère de Cinema 4D
# (Y vers le haut, centimètres) : la conversion est faite côté Blender.

import array
import json
import os
import sys
import time
import traceback

import c4d

JOB_ENV = "C4D_BRIDGE_JOB"
LOG = []


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    LOG.append(line)
    print(line)
    sys.stdout.flush()


def const(name):
    """Constante c4d par son nom, None si absente dans cette version."""
    return getattr(c4d, name, None)


def setp(node, name, value):
    """Fixe un paramètre par nom de constante, sans planter si la constante
    ou le paramètre n'existent pas dans cette version de Cinema 4D."""
    pid = const(name)
    if pid is None:
        log("  (paramètre absent dans cette version : %s)" % name)
        return False
    try:
        node[pid] = value
        return True
    except Exception as e:  # noqa
        log("  (échec %s : %s)" % (name, e))
        return False


def vec(v):
    return c4d.Vector(float(v[0]), float(v[1]), float(v[2]))


def mat(m):
    # m = [off(3), v1(3), v2(3), v3(3)]
    return c4d.Matrix(vec(m[0:3]), vec(m[3:6]), vec(m[6:9]), vec(m[9:12]))


def safe_replace(src, dst, tries=40, wait=0.25):
    """os.replace avec nouvelles tentatives : sous Windows, le remplacement
    échoue (accès refusé) si Blender, un antivirus ou OneDrive lit le fichier
    au même instant."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return True
        except PermissionError:
            time.sleep(wait)
    return False


def write_status(job, **kw):
    """Fichier de progression lu par Blender. Ne doit JAMAIS interrompre la
    simulation : en cas d'échec, l'information est simplement perdue."""
    path = os.path.join(job, "progress.json")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(kw, f)
        if not safe_replace(tmp, path, tries=8, wait=0.05):
            with open(path, "w") as f:  # dernier recours : écriture directe
                json.dump(kw, f)
    except OSError:
        pass


def build_geometry(g):
    pts = g["points"]
    tris = g["tris"]
    n = len(pts) // 3
    obj = c4d.PolygonObject(n, len(tris) // 3)
    obj.SetAllPoints([c4d.Vector(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2])
                      for i in range(n)])
    for i in range(len(tris) // 3):
        a, b, c = tris[3 * i], tris[3 * i + 1], tris[3 * i + 2]
        obj.SetPolygon(i, c4d.CPolygon(a, b, c))
    obj.Message(c4d.MSG_UPDATE)
    return obj


RB_SHAPES = {
    "CONVEX_HULLS": "RIGIDBODY_PBD_COLLISION_SHAPES_CONVEX_HULLS",
    "TRIANGLE_MESH": "RIGIDBODY_PBD_COLLISION_SHAPES_TRIANGLE_MESH",
    "SPHERE": "RIGIDBODY_PBD_COLLISION_SHAPES_SPHERE",
    "BOX": "RIGIDBODY_PBD_COLLISION_SHAPES_BOX",
    "AUTO": "RIGIDBODY_PBD_COLLISION_SHAPES_AUTO",
}
COLLIDER_SHAPES = {
    "CONVEX_HULLS": "CONVEX_COLLIDER_CONVEX_HULLS",
    "TRIANGLE_MESH": "CONVEX_COLLIDER_TRIANGLE_MESH",
    "SPHERE": "CONVEX_COLLIDER_SPHERE",
    "BOX": "CONVEX_COLLIDER_BOX",
    "AUTO": "CONVEX_COLLIDER_TRIANGLE_MESH",
}


SH_SIMULATION = 1057220  # ID du scene hook de simulation (forum Maxon)


def get_default_sim_scene(doc):
    """Scène de simulation par défaut du document (Paramètres du projet)."""
    try:
        hook = doc.FindSceneHook(getattr(c4d, "SHsimulation", SH_SIMULATION))
        if hook is None:
            return None
        for data in hook.GetBranchInfo():
            if data.get("id") == c4d.Obase:
                head = data.get("head")
                return head.GetFirst() if head is not None else None
    except Exception as e:  # noqa
        log("Accès à la scène de simulation par défaut impossible : %s" % e)
    return None


def configure_sim_scene(node, sim, f0, f1):
    setp(node, "PBDSCENE_USEDOCUMENT_RANGE", True)
    setp(node, "PBDSCENE_SIMULATION_FROM", f0)
    setp(node, "PBDSCENE_SIMULATION_TO", f1)
    setp(node, "PBDSCENE_SUBSTEPS", int(sim["substeps"]))
    setp(node, "PBDSCENE_ITERATIONS", int(sim["iterations"]))
    if sim.get("damping"):
        setp(node, "PBDSCENE_DAMPING", float(sim["damping"]))
    # Dans C4D la gravité est négative (vers le bas) : -981 cm/s² par défaut.
    g = abs(float(sim["gravity"]))
    setp(node, "PBDSCENE_DEFAULTGRAVITY", -g if g > 0.0 else 0.0)
    if sim.get("gpu"):
        gpu = const("PBDSCENE_DEVICE_GPU")
        if gpu is not None:
            setp(node, "PBDSCENE_DEVICE", gpu)


def add_rigid_body(obj, b):
    tag = obj.MakeTag(c4d.Trigidbody)
    shape = const(RB_SHAPES.get(b["shape"], RB_SHAPES["CONVEX_HULLS"]))
    if shape is not None:
        setp(tag, "RIGIDBODY_PBD_COLLISION_SHAPES", shape)
    use_mass = const("RIGIDBODY_PBD_USE_CUSTOM_MASS")
    if use_mass is not None:
        setp(tag, "RIGIDBODY_PBD_MASS_SELECTION", use_mass)
    setp(tag, "RIGIDBODY_PBD_MASS", float(b["mass"]))
    setp(tag, "RIGIDBODY_PBD_FRICTION", float(b["friction"]))
    setp(tag, "RIGIDBODY_PBD_BOUNCINESS", float(b["bounce"]))
    iv = b.get("init_vel")
    if iv:
        # vitesse initiale tangentielle (rotation de départ autour du verre)
        setp(tag, "RIGIDBODY_PBD_CUSTOM_INITIAL_VELOCITY", True)
        setp(tag, "RIGIDBODY_PBD_INITIAL_VELOCITY_IN_LOCALSPACE", False)
        setp(tag, "RIGIDBODY_PBD_INITIAL_LINEAR_VELOCITY", vec(iv[0:3]))
        setp(tag, "RIGIDBODY_PBD_INITIAL_LINEAR_VELOCITY_STRENGTH", float(iv[3]))
    ia = b.get("init_ang")
    if ia:
        # rotation propre de départ (les cailloux roulent au lieu de glisser)
        setp(tag, "RIGIDBODY_PBD_CUSTOM_INITIAL_ANGULAR_VELOCITY", True)
        setp(tag, "RIGIDBODY_PBD_INITIAL_ANGULAR_VELOCITY_IN_LOCALSPACE", False)
        setp(tag, "RIGIDBODY_PBD_INITIAL_ANGULAR_VELOCITY", vec(ia[0:3]))
        setp(tag, "RIGIDBODY_PBD_INITIAL_ANGULAR_VELOCITY_STRENGTH", float(ia[3]))
    dump_once(tag, "Tag Rigid Body")
    return tag


def key_position(obj, pts, f0, fps):
    """Clés de position (une par image) : trajectoire cible pour « Suivre la position »."""
    n = len(pts) // 3
    for comp, vid in enumerate((c4d.VECTOR_X, c4d.VECTOR_Y, c4d.VECTOR_Z)):
        did = c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_POSITION, c4d.DTYPE_VECTOR, 0),
                         c4d.DescLevel(vid, c4d.DTYPE_REAL, 0))
        track = c4d.CTrack(obj, did)
        obj.InsertTrackSorted(track)
        curve = track.GetCurve()
        for i in range(n):
            key = curve.AddKey(c4d.BaseTime(f0 + i, fps))["key"]
            key.SetValue(curve, float(pts[3 * i + comp]))


def setup_follow(tag, follow):
    setp(tag, "RIGIDBODY_PBD_FORCES_FOLLOW_POSITION_STRENGTH", float(follow["strength"]))
    if follow.get("damping"):
        setp(tag, "RIGIDBODY_PBD_FORCES_FOLLOW_POSITION_DAMPING", float(follow["damping"]))
    setp(tag, "RIGIDBODY_PBD_FORCES_FOLLOW_ROTATION_STRENGTH", float(follow.get("rot_strength", 0.0)))


def key_param(node, pname, vals, f0, fps, label):
    """Piste d'animation d'un paramètre réel (pour la scène .c4d enregistrée)."""
    try:
        track = c4d.CTrack(node, c4d.DescID(c4d.DescLevel(const(pname), c4d.DTYPE_REAL, 0)))
        node.InsertTrackSorted(track)
        curve = track.GetCurve()
        for i, v in enumerate(vals):
            key = curve.AddKey(c4d.BaseTime(f0 + i, fps))["key"]
            key.SetValue(curve, float(v))
    except Exception as e:  # noqa
        log("  (piste %s non créée pour %s : %s)" % (pname, label, e))


def add_collider(obj, b):
    tag = obj.MakeTag(c4d.Tcollider)
    shape = const(COLLIDER_SHAPES.get(b["shape"], "CONVEX_COLLIDER_TRIANGLE_MESH"))
    if shape is not None:
        setp(tag, "CONVEX_COLLIDER", shape)
    setp(tag, "COLLIDER_FRICTION", float(b["friction"]))
    setp(tag, "COLLIDER_BOUNCE", float(b["bounce"]))
    both = const("CLOTH_PBD_COLLIDER_SIDE_BOTH")
    if both is not None:
        setp(tag, "CLOTH_PBD_COLLIDER_SIDE", both)
    return tag


STRENGTH_PARAM = {"ROTATION": "ROTATIONOBJECT_STRENGTH",
                  "ATTRACTOR": "ATTRACTOROBJECT_STRENGTH"}


def key_strength(op, f, f0, fps):
    """Crée une piste d'animation de l'intensité dans C4D, pour que la scène
    .c4d enregistrée contienne aussi l'animation (visible dans la timeline)."""
    try:
        pid = const(STRENGTH_PARAM[f["kind"]])
        track = c4d.CTrack(op, c4d.DescID(c4d.DescLevel(pid, c4d.DTYPE_REAL, 0)))
        op.InsertTrackSorted(track)
        curve = track.GetCurve()
        for k, v in enumerate(f["anim_strength"]):
            key = curve.AddKey(c4d.BaseTime(f0 + k, fps))["key"]
            key.SetValue(curve, float(v))
    except Exception as e:  # noqa
        log("  (piste d'animation non créée pour %s : %s)" % (f["name"], e))


_DUMPED = set()
_FALLOFF_REPORTED = []


def describe(op):
    """Paramètres de la description d'un objet : (ident, nom, id, valeur, options)."""
    out = []
    try:
        desc = op.GetDescription(c4d.DESCFLAGS_DESC_NONE)
    except Exception as e:  # noqa
        log("  (description illisible : %s)" % e)
        return out
    for item in desc:
        try:
            bc, paramid = item[0], item[1]
            ident = bc[c4d.DESC_IDENT] or ""
            name = bc[c4d.DESC_NAME] or ""
            pid = paramid[0].id
            cycle = {}
            cyc = bc.GetContainer(c4d.DESC_CYCLE)
            if cyc is not None:
                for k, v in cyc:
                    cycle[k] = v
            try:
                val = op[pid]
            except Exception:  # noqa
                val = None
            try:
                lo, hi = bc[c4d.DESC_MIN], bc[c4d.DESC_MAX]
                if lo is not None or hi is not None:
                    name = "%s, %s..%s" % (name, lo, hi)
            except Exception:  # noqa
                pass
            out.append((ident, name, pid, val, cycle))
        except Exception:  # noqa
            continue
    return out


def dump_once(op, label):
    """Écrit une fois dans le journal tous les paramètres d'un type d'objet."""
    if label in _DUMPED:
        return
    _DUMPED.add(label)
    log("Paramètres disponibles (%s) :" % label)
    for ident, name, pid, val, cycle in describe(op):
        if not ident and not name:
            continue
        line = "  %s [%s] id=%s = %r" % (ident, name, pid, val)
        if cycle:
            line += "  options=%r" % (cycle,)
        log(line)


def set_inverse_square(op):
    """Règle l'atténuation de l'attracteur sur « inverse du carré de la distance »
    si cette version de C4D propose ce réglage. Renvoie une description ou None."""
    for ident, name, pid, val, cycle in describe(op):
        key = (ident + " " + name).upper()
        if not cycle or not any(w in key for w in ("FALLOFF", "ATTÉNUATION", "ATTENUATION", "DÉCROISSANCE")):
            continue
        best = None
        if ident:
            for cname in dir(c4d):
                if cname.startswith(ident + "_") and ("SQUARE" in cname or "SQR" in cname):
                    best = getattr(c4d, cname)
                    break
        if best is None:
            for v, label in cycle.items():
                low = str(label).lower()
                if "squar" in low or "carr" in low or "²" in low:
                    best = v
                    break
        if best is not None:
            op[pid] = best
            return "%s = %s" % (ident or name, cycle.get(best, best))
    return None


def add_force(doc, parent, f):
    kind = f["kind"]
    if kind == "ROTATION":
        op = c4d.BaseObject(c4d.Orotation)
        setp(op, "ROTATIONOBJECT_STRENGTH", float(f["strength"]))
        dump_once(op, "Rotation")
    elif kind == "ATTRACTOR":
        op = c4d.BaseObject(c4d.Oattractor)
        setp(op, "ATTRACTOROBJECT_STRENGTH", float(f["strength"]))
        if f.get("speed_limit"):
            setp(op, "ATTRACTOROBJECT_SPEEDLIMIT", float(f["speed_limit"]))
        if f.get("min_distance"):
            setp(op, "ATTRACTOROBJECT_MIN_DISTANCE", float(f["min_distance"]))
        dump_once(op, "Attracteur")
        if f.get("falloff") == "INVERSE_SQUARE":
            done = set_inverse_square(op)
            if not _FALLOFF_REPORTED:
                _FALLOFF_REPORTED.append(done)
                if done:
                    log("Atténuation des attracteurs : %s" % done)
                else:
                    log("ATTENTION : cette version de Cinema 4D ne propose pas d'atténuation "
                        "pour l'attracteur. Les attracteurs répartis (ligne / surface) "
                        "agissent alors comme un seul point en leur centre : les cailloux "
                        "s'écrasent en anneau.")
    else:
        log("Force ignorée (type non géré) : %s" % kind)
        return None
    acc = const("PARTICLEBASE_MODE_ACCELERATION")
    if acc is not None:
        setp(op, "PARTICLEBASE_MODE", acc)
    op.SetName(f["name"])
    doc.InsertObject(op, parent)
    op.SetMg(mat(f["matrix"]))
    return op


def run(job):
    t0 = time.time()
    with open(os.path.join(job, "scene.json"), "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    fps = int(round(cfg["fps"]))
    f0, f1 = int(cfg["frame_start"]), int(cfg["frame_end"])
    nframes = f1 - f0 + 1
    bodies = cfg["bodies"]
    log("Scène : %d corps, %d géométries, %d forces, images %d-%d à %d i/s"
        % (len(bodies), len(cfg["geometries"]), len(cfg["forces"]), f0, f1, fps))
    write_status(job, state="building", frame=0, total=nframes)

    doc = c4d.documents.BaseDocument()
    doc.SetFps(fps)
    doc.SetMinTime(c4d.BaseTime(f0, fps))
    doc.SetMaxTime(c4d.BaseTime(f1, fps))
    doc.SetLoopMinTime(c4d.BaseTime(f0, fps))
    doc.SetLoopMaxTime(c4d.BaseTime(f1, fps))

    # --- Scène de simulation --------------------------------------------
    # Les réglages sont appliqués à notre objet Simulation Scene ET à la scène
    # de simulation par défaut du projet (rangée dans un « scene hook », hors
    # de la liste des objets), pour qu'aucun corps n'échappe aux réglages.
    sim = cfg["simulation"]
    scene = c4d.BaseObject(c4d.Osimulationscene)
    scene.SetName("Simulation Bridge")
    doc.InsertObject(scene)
    sim_nodes = [("Simulation Bridge", scene)]
    default_scene = get_default_sim_scene(doc)
    if default_scene is not None:
        sim_nodes.append(("scène par défaut du projet", default_scene))
    else:
        log("Scène de simulation par défaut introuvable (réglages appliqués à "
            "l'objet Simulation Bridge seulement)")
    for label, node in sim_nodes:
        configure_sim_scene(node, sim, f0, f1)
        try:
            log("Gravité %s : %s" % (label, node[c4d.PBDSCENE_DEFAULTGRAVITY]))
        except Exception:  # noqa
            pass
    dump_once(scene, "Scène de simulation")

    # --- Géométries prototypes ------------------------------------------
    protos = [build_geometry(g) for g in cfg["geometries"]]

    # --- Corps ----------------------------------------------------------
    elements = c4d.InExcludeData()
    objs = []
    animated = []
    animated_params = []
    n_follow = 0
    for i, b in enumerate(bodies):
        o = protos[b["geom"]].GetClone()
        o.SetName("b%05d_%s" % (i, b["name"]))
        doc.InsertObject(o, scene, objs[-1] if objs else None)
        o.SetMg(mat(b["matrix"]))
        if b["type"] == "ACTIVE":
            tag = add_rigid_body(o, b)
            if b.get("follow"):
                key_position(o, b["follow"], f0, fps)
                setup_follow(tag, cfg["follow"])
                n_follow += 1
        else:
            tag = add_collider(o, b)
            if b.get("anim"):
                animated.append((o, b["anim"]))
        if b.get("anim_friction") or b.get("anim_bounce"):
            active = b["type"] == "ACTIVE"
            prm = {"anim_friction": "RIGIDBODY_PBD_FRICTION" if active else "COLLIDER_FRICTION",
                   "anim_bounce": "RIGIDBODY_PBD_BOUNCINESS" if active else "COLLIDER_BOUNCE"}
            for key, pname in prm.items():
                if b.get(key):
                    animated_params.append((tag, pname, b[key]))
                    key_param(tag, pname, b[key], f0, fps, b["name"])
        elements.InsertObject(o, 1)
        objs.append(o)
    setp(scene, "PBDSCENE_ELEMENTS", elements)
    if animated_params:
        log("Paramètres animés : " + ", ".join(sorted({p for _t, p, _v in animated_params})))
    if n_follow:
        log("Orbite guidée : %d cailloux suivent leur trajectoire (force %s)"
            % (n_follow, cfg["follow"]["strength"]))

    # --- Forces ---------------------------------------------------------
    forces = c4d.InExcludeData()
    animated_forces = []
    by_name = {b["name"]: objs[i] for i, b in enumerate(bodies)}
    groups = {}
    for f in cfg["forces"]:
        parent = scene
        pname = f.get("parent")
        if pname and pname in by_name:
            if pname not in groups:
                null = c4d.BaseObject(c4d.Onull)
                null.SetName(f["source"])
                doc.InsertObject(null, by_name[pname])
                null.SetMg(by_name[pname].GetMg())
                groups[pname] = null
            parent = groups[pname]
        op = add_force(doc, parent, f)
        if op is not None:
            forces.InsertObject(op, 1)
            if f.get("anim_strength") or f.get("anim_matrix"):
                animated_forces.append((op, f))
                if f.get("anim_strength"):
                    key_strength(op, f, f0, fps)
    if animated_forces:
        log("Forces animées : " + ", ".join(sorted({f["source"] for _op, f in animated_forces})))
    for pname, null in groups.items():
        fs = [f for f in cfg["forces"] if f.get("parent") == pname]
        log("Groupe « %s » : %d attracteurs, enfant de %s" % (fs[0]["source"], len(fs), pname))
    for _label, node in sim_nodes:
        setp(node, "PBDSCENE_FORCES", forces)

    if cfg.get("save_c4d"):
        path = os.path.join(job, "scene_bridge.c4d")
        try:
            c4d.documents.SaveDocument(doc, path, c4d.SAVEDOCUMENTFLAGS_NONE,
                                       c4d.FORMAT_C4DEXPORT)
            log("Scène C4D enregistrée : " + path)
        except Exception as e:  # noqa
            log("Impossible d'enregistrer la scène C4D : %s" % e)

    # --- Simulation -----------------------------------------------------
    active_idx = [i for i, b in enumerate(bodies) if b["type"] == "ACTIVE"]
    out = open(os.path.join(job, "result.bin.part"), "wb")
    t_sim = time.time()
    first_mg = [objs[i].GetMg() for i in active_idx]
    moved = False
    for k in range(nframes):
        frame = f0 + k
        for o, anim in animated:
            o.SetMg(mat(anim[k]))
        for tag, pname, vals in animated_params:
            setp(tag, pname, float(vals[k]))
        for op, f in animated_forces:
            if f.get("anim_strength"):
                setp(op, STRENGTH_PARAM[f["kind"]], float(f["anim_strength"][k]))
            if f.get("anim_matrix"):
                op.SetMg(mat(f["anim_matrix"][k]))
        doc.SetTime(c4d.BaseTime(frame, fps))
        if not doc.ExecutePasses(None, True, True, True, c4d.BUILDFLAGS_NONE):
            raise RuntimeError("ExecutePasses a échoué à l'image %d" % frame)
        buf = array.array("d")
        for j, i in enumerate(active_idx):
            mg = objs[i].GetMg()
            buf.extend((mg.off.x, mg.off.y, mg.off.z,
                        mg.v1.x, mg.v1.y, mg.v1.z,
                        mg.v2.x, mg.v2.y, mg.v2.z,
                        mg.v3.x, mg.v3.y, mg.v3.z))
            if not moved and k > 0 and mg != first_mg[j]:
                moved = True
        buf.tofile(out)
        if k % 5 == 0 or k == nframes - 1:
            el = time.time() - t_sim
            write_status(job, state="simulating", frame=k + 1, total=nframes,
                         elapsed=el)
            log("Image %d / %d  (%.1f s)" % (k + 1, nframes, el))
    out.close()
    if not moved and active_idx and nframes > 1:
        log("ATTENTION : aucun corps n'a bougé pendant la simulation.")
    if not safe_replace(os.path.join(job, "result.bin.part"), os.path.join(job, "result.bin")):
        raise RuntimeError("Impossible de finaliser result.bin (fichier verrouillé par "
                           "un autre programme : antivirus, OneDrive ?)")
    with open(os.path.join(job, "result.json"), "w") as fh:
        json.dump({"frame_start": f0, "frame_end": f1,
                   "bodies": [bodies[i]["name"] for i in active_idx],
                   "moved": moved,
                   "sim_seconds": time.time() - t_sim,
                   "total_seconds": time.time() - t0}, fh)
    write_status(job, state="done", frame=nframes, total=nframes,
                 elapsed=time.time() - t_sim)
    log("Terminé en %.1f s (simulation %.1f s)" % (time.time() - t0, time.time() - t_sim))


def find_job():
    job = os.environ.get(JOB_ENV)
    if job and os.path.isfile(os.path.join(job, "scene.json")):
        return job
    # Chemin transmis en UTF-8 dans job.txt, à côté de ce script (évite les
    # problèmes d'accents dans les arguments de la ligne de commande).
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "job.txt"), "r", encoding="utf-8") as fh:
            job = fh.read().strip()
        if os.path.isfile(os.path.join(job, "scene.json")):
            return job
    except Exception:  # noqa
        pass
    if len(sys.argv) > 1 and os.path.isfile(os.path.join(sys.argv[-1], "scene.json")):
        return sys.argv[-1]
    # Mode manuel dans Cinema 4D (Gestionnaire de scripts)
    try:
        path = c4d.storage.LoadDialog(title="Choisir scene.json (export Blender)")
    except Exception:  # noqa
        path = None
    if path:
        return os.path.dirname(path)
    return None


def main():
    job = find_job()
    if not job:
        print("Aucun dossier de travail trouvé (scene.json).")
        return
    try:
        run(job)
    except Exception as e:  # noqa
        log("ERREUR : %s" % e)
        log(traceback.format_exc())
        try:
            write_status(job, state="error", message=str(e))
        except Exception:  # noqa
            pass
    finally:
        try:
            with open(os.path.join(job, "c4d_log.txt"), "a", encoding="utf-8") as fh:
                fh.write("\n".join(LOG) + "\n")
        except Exception:  # noqa
            pass


if __name__ == "__main__":
    main()
'''
