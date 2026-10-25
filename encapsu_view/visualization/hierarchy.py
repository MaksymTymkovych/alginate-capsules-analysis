import json
import sys
import numpy as np
import matplotlib.pyplot as plt

from anytree import Node, RenderTree, ContStyle, AsciiStyle


from encapsu_view.analysis.processed_data import ProcessedData

def create_tree_from_processed_data(processed_data: ProcessedData) -> Node:
    root = Node("Image", 
                image_shape=processed_data.image_shape, 
                scale=processed_data.scale,
                completeness_ratio=processed_data.get_completeness_ratio())
    
    for capsule in processed_data.capsules:
        capsule_node = Node(f"Capsule {capsule.id}", 
                           parent=root,
                           type="capsule",
                           ellipse_center=capsule.ellipse.center,
                           ellipse_axes=capsule.ellipse.axes,
                           ellipse_angle=capsule.ellipse.angle,
                           ellipse_area=capsule.ellipse.area,
                           is_complete=capsule.is_complete)
        
        for inner_obj in capsule.inner_objects:
            Node(f"Inner Object {inner_obj.id}", 
                 parent=capsule_node,
                 type="inner_object",
                 area=inner_obj.area,
                 centroid=inner_obj.centroid,
                 bbox=inner_obj.bbox)
    
    return root

def format_tree(processed_data: ProcessedData, unicode: bool = True) -> str:

    root = create_tree_from_processed_data(processed_data)
    if unicode:
        icons = {"tree": "🌳 ", "image": "📷 ", "capsule": "💊 ", "inner": "🔹 "}
        style = ContStyle()
    else:
        icons = {"tree": "", "image": "", "capsule": "", "inner": ""}
        style = AsciiStyle()

    lines = [f"{icons['tree']}Tree Structure of Capsules:", "=" * 50]

    for pre, fill, node in RenderTree(root, style=style):
        if node.name == "Image":
            lines.append(f"{pre}{icons['image']}{node.name}")
            lines.append(f"{pre}   Shape: {node.image_shape}")
            lines.append(f"{pre}   Scale: {node.scale}")
            lines.append(f"{pre}   Completeness: {node.completeness_ratio:.2f}")
        elif "Capsule" in node.name:
            lines.append(f"{pre}{icons['capsule']}{node.name}")
            lines.append(f"{pre}   Center: ({node.ellipse_center[0]:.1f}, {node.ellipse_center[1]:.1f})")
            lines.append(f"{pre}   Axes: ({node.ellipse_axes[0]:.1f}, {node.ellipse_axes[1]:.1f})")
            lines.append(f"{pre}   Area: {node.ellipse_area:.1f}")
            lines.append(f"{pre}   Complete: {node.is_complete}")
        else:
            lines.append(f"{pre}{icons['inner']}{node.name}")
            lines.append(f"{pre}   Area: {node.area:.1f}")
            lines.append(f"{pre}   Center: ({node.centroid[0]:.1f}, {node.centroid[1]:.1f})")

    return "\n".join(lines)


def _stdout_supports_unicode() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or ""
    return encoding.lower().replace("-", "") in ("utf8", "utf16", "utf32")


def visualize_tree(processed_data: ProcessedData):
    # Windows consoles often use cp1251/cp866, which cannot encode emoji or box-drawing characters
    print(format_tree(processed_data, unicode=_stdout_supports_unicode()))


def plot_capsule_hierarchy(processed_data: ProcessedData, show: bool = True) -> plt.Figure:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    

    areas = [capsule.ellipse.area for capsule in processed_data.capsules]
    inner_areas = [sum(inner.area for inner in capsule.inner_objects) for capsule in processed_data.capsules]
    
    x_pos = np.arange(len(areas))
    width = 0.35
    
    ax1.bar(x_pos - width/2, areas, width, label='Ellipse Area', alpha=0.7)
    ax1.bar(x_pos + width/2, inner_areas, width, label='Total Inner Area', alpha=0.7)
    ax1.set_xlabel('Capsule Index')
    ax1.set_ylabel('Area')
    ax1.set_title('Area Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    aspect_ratios = [max(capsule.ellipse.axes) / min(capsule.ellipse.axes) 
                    for capsule in processed_data.capsules if min(capsule.ellipse.axes) > 0]
    
    ax2.hist(aspect_ratios, bins=15, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Aspect Ratio (major/minor axis)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Ellipse Aspect Ratio Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if show:
        plt.show()

    return fig

def export_to_json(processed_data: ProcessedData, filename: str = "capsules_data.json"):
    data_dict = processed_data.to_dict()
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Data was exported to {filename}")
    return data_dict