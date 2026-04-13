from __future__ import annotations

import numpy as np
import warp as wp
import newton


@wp.kernel
def _transform_mesh_points(
    points: wp.array(dtype=wp.vec3),
    scale: wp.vec3,
    xform: wp.transform,
    output_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    output_points[tid] = wp.transform_point(xform, wp.cw_mul(points[tid], scale))


class CpuRobotMeshRenderer:
    """Fallback mesh renderer for Newton models when CPU GL instancing is unavailable."""

    def __init__(self, viewer, model):
        self.viewer = viewer
        self.model = model
        self.entries = []

        for batch_index, batch in enumerate(viewer._shape_instances.values()):
            if int(batch.geo_type) != int(newton.GeoType.MESH):
                continue

            if len(batch.model_shapes) == 0:
                continue

            shape_index = int(batch.model_shapes[0])
            mesh = model.shape_source[shape_index]
            if mesh is None or mesh.vertices is None or mesh.indices is None:
                continue

            points_np = np.asarray(mesh.vertices, dtype=np.float32)
            indices_np = np.asarray(mesh.indices, dtype=np.int32).reshape(-1)

            base_points = wp.array(points_np, dtype=wp.vec3, device="cpu")
            indices = wp.array(indices_np, dtype=wp.int32, device="cpu")
            output_points = [wp.zeros(len(base_points), dtype=wp.vec3, device="cpu") for _ in batch.model_shapes]
            names = [f"/cpu_robot_mesh/{batch_index}_{instance_index}" for instance_index in range(len(batch.model_shapes))]

            self.entries.append(
                {
                    "batch": batch,
                    "base_points": base_points,
                    "indices": indices,
                    "output_points": output_points,
                    "names": names,
                }
            )

    def draw(self, state) -> None:
        for entry in self.entries:
            batch = entry["batch"]
            visible = self.viewer._should_show_shape(batch.flags, batch.static)

            batch.update(state, world_offsets=self.viewer.world_offsets)
            world_xforms = batch.world_xforms.numpy()
            scales = batch.scales.numpy()

            for instance_index, name in enumerate(entry["names"]):
                xform = wp.transform(*world_xforms[instance_index])
                scale = wp.vec3(*scales[instance_index])
                output_points = entry["output_points"][instance_index]

                wp.launch(
                    _transform_mesh_points,
                    dim=len(entry["base_points"]),
                    inputs=[entry["base_points"], scale, xform],
                    outputs=[output_points],
                    device="cpu",
                    record_tape=False,
                )

                self.viewer.log_mesh(name, output_points, entry["indices"], hidden=not visible)
