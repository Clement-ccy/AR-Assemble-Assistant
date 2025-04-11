import bpy
import os
import json
import datetime
from bpy_extras.object_utils import world_to_camera_view

bl_info = {
    "name": "Auto BBox Annotator",
    "author": "Chen Chengye",
    "version": (1, 0),
    "blender": (4, 1, 0),
    "location": "Render Properties > BBox Annotation",
    "description": "Automatically generate bounding box annotations for rendered images",
    "warning": "",
    "category": "Render",
}

class BBoxAnnotationProperties(bpy.types.PropertyGroup):
    output_path: bpy.props.StringProperty(
        name="Output Directory",
        subtype='DIR_PATH',
        default="//annotations/"
    )
    format: bpy.props.EnumProperty(
        name="Format",
        items=[('COCO', 'COCO', 'COCO JSON format'),
               ('YOLO', 'YOLO', 'YOLO TXT format')],
        default='COCO'
    )
    min_visibility: bpy.props.FloatProperty(
        name="Min Visibility",
        description="Minimum visible ratio",
        default=0.8,
        min=0.0,
        max=1.0
    )

def get_2d_bbox(obj, camera):
    scene = bpy.context.scene
    render = scene.render
    
    # Get all mesh vertices in world coordinates
    matrix = obj.matrix_world
    mesh = obj.data
    vertices = [matrix @ v.co for v in mesh.vertices]

    # Project vertices to camera view
    coords_2d = []
    for v in vertices:
        co_ndc = world_to_camera_view(scene, camera, v)
        coords_2d.append((co_ndc.x, co_ndc.y))

    # Filter points outside view
    valid_coords = [(x, y) for x, y in coords_2d 
                   if (0.0 < x < 1.0 and 0.0 < y < 1.0)]
    
    if not valid_coords:
        return None

    # Calculate bbox
    min_x = min([x for x, y in valid_coords])
    max_x = max([x for x, y in valid_coords])
    min_y = min([y for x, y in valid_coords])
    max_y = max([y for x, y in valid_coords])

    # Convert to pixel coordinates
    width = render.resolution_x
    height = render.resolution_y
    
    return [
        min_x * width,
        (1 - max_y) * height,  # Blender Y-axis is inverted
        (max_x - min_x) * width,
        (max_y - min_y) * height
    ]

def save_annotation(context, filepath):
    props = context.scene.bbox_annotator_props
    camera = context.scene.camera
    annotations = []
    
    for obj in context.scene.objects:
        if obj.type != 'MESH' or obj.hide_render:
            continue
            
        bbox = get_2d_bbox(obj, camera)
        if not bbox:
            continue
            
        # COCO format
        if props.format == 'COCO':
            annotations.append({
                "image_id": os.path.basename(filepath),
                "category_id": 0,  # Modify for multi-class
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
        # YOLO format
        elif props.format == 'YOLO':
            width = context.scene.render.resolution_x
            height = context.scene.render.resolution_y
            x_center = (bbox[0] + bbox[2]/2) / width
            y_center = (bbox[1] + bbox[3]/2) / height
            w = bbox[2] / width
            h = bbox[3] / height
            
            annotations.append(f"0 {x_center} {y_center} {w} {h}")

    # Save to file
    if props.format == 'COCO':
        with open(filepath, 'w') as f:
            json.dump({"annotations": annotations}, f)
    else:
        with open(filepath, 'w') as f:
            f.write("\n".join(annotations))

class BBOX_PT_AnnotationPanel(bpy.types.Panel):
    bl_label = "BBox Annotation"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.bbox_annotator_props
        
        # 使用新式列布局
        col = layout.column(heading="Settings", align=True)
        col.prop(props, "output_path", text="Folder")
        col.prop(props, "format", text="Format")
        
        row = col.row(align=True)
        row.prop(props, "min_visibility", slider=True)
        
        layout.separator()
        layout.operator("render.render_with_annotation", 
                       icon='RENDER_STILL', 
                       text="Render with BBox")

class RenderWithAnnotation(bpy.types.Operator):
    bl_idname = "render.render_with_annotation"
    bl_label = "Render with Annotation"
    
    def execute(self, context):
        print("Operator triggered")  # 确认操作符被调用
        props = context.scene.bbox_annotator_props
        original_path = bpy.context.scene.render.filepath
        try:
            # 生成唯一文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"render_{timestamp}"
            
            # 设置完整渲染路径
            output_dir = bpy.path.abspath(props.output_path)
            render_path = os.path.join(output_dir, filename)
            
            # 显式指定文件格式
            bpy.context.scene.render.filepath = f"{render_path}.png"
            
            # 执行渲染
            bpy.ops.render.render(write_still=True)
            
            # 保存标注（使用相同基础路径）
            annotation_path = f"{render_path}.{'json' if props.format == 'COCO' else 'txt'}"
            save_annotation(context, annotation_path)
            print(f"Saved annotation to: {annotation_path}")  # 添加调试输出
            
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
            
        finally:
            bpy.context.scene.render.filepath = original_path
            
        return {'FINISHED'}

class BBOX_PT_AnnotationPanel(bpy.types.Panel):
    bl_label = "BBox Annotation"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.bbox_annotator_props
        
        layout.prop(props, "output_path")
        layout.prop(props, "format")
        layout.prop(props, "min_visibility")
        layout.operator("render.render_with_annotation")

def register():
    bpy.utils.register_class(BBoxAnnotationProperties)
    bpy.utils.register_class(RenderWithAnnotation)
    bpy.utils.register_class(BBOX_PT_AnnotationPanel)
    # 使用新式场景属性附加方式
    if not hasattr(bpy.types.Scene, "bbox_annotator_props"):
        bpy.types.Scene.bbox_annotator_props = bpy.props.PointerProperty(
            type=BBoxAnnotationProperties
        )

def unregister():
    bpy.utils.unregister_class(BBoxAnnotationProperties)
    bpy.utils.unregister_class(RenderWithAnnotation)
    bpy.utils.unregister_class(BBOX_PT_AnnotationPanel)
    # 清理场景属性
    if hasattr(bpy.types.Scene, "bbox_annotator_props"):
        del bpy.types.Scene.bbox_annotator_props

if __name__ == "__main__":
    register()
