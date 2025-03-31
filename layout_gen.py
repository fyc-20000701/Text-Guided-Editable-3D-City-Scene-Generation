import json
import random
import os
# 设置 PROJ_LIB 环境变量
os.environ['PROJ_LIB'] = r"E:\Anaconda\Library\share\proj"
os.environ['rasterio_LIB'] = r"E:\Anaconda"
os.environ['rasterio_LIB2'] = r"E:\Anaconda\Lib\site-packages"

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import shapefile
from shapely.geometry import Polygon, LineString
from noise import pnoise2  # Perlin 噪声生成
import numpy as np
from PIL import Image
import pyproj
from scipy.ndimage import gaussian_filter
from rasterio.crs import CRS
from rasterio import Affine
from rasterio import open as rio_open

def load_json(json_path):
    with open(json_path, 'r') as f:
        layout_data = json.load(f)
    return layout_data

def initialize_size_presets():
    """
    初始化不同语义类型的节点尺寸预设方案。
    """
    return {
        'building': [(20, 20), (40, 40), (60, 60), (20, 40), (40, 60), (40, 20), (60, 40)],  # 建筑的预设尺寸
        'water': [(20, 20), (40, 40), (60, 60), (80, 80), (20, 40), (40, 60), (40, 20), (60, 40)],  # 水体的预设尺寸
        'green': [(20, 20), (40, 40), (60, 60), (80, 80), (20, 40), (40, 60), (40, 20), (60, 40)]  # 绿色区域的预设尺寸
    }

def select_size_for_entity(entity, size_presets, max_width, max_height):
    """
    从语义类型的预设中随机选择一个尺寸，并确保尺寸不超过指定的最大值。
    """
    valid_sizes = [size for size in size_presets.get(entity, [(20, 20)])
                   if size[0] <= max_width and size[1] <= max_height]

    # 如果没有有效尺寸，则选择最接近的一个
    if not valid_sizes:
        valid_sizes = [(min(max_width, 20), min(max_height, 20))]

    return random.choice(valid_sizes)

def initialize_region_division(root_width, root_height):
    """
    初始化根节点的区域划分，确保中间值与根节点尺寸一致。
    """
    options = [20, 40, 60]
    # options.remove(root_width)  # 确保中间值与根节点尺寸一致
    options_y = [20, 40, 60]
    options_y.remove(root_height)

    return {
        "x": [random.choice(options), root_width, random.choice(options)],
        "y": [random.choice(options_y), root_height, random.choice(options_y)]
    }

def adjust_region_division_for_child(parent_division, parent_direction):
    """
    根据父节点的区域划分调整子节点的区域划分，子节点的中心区域应与父节点的分区一致。
    """
    # 根据不同方向调整子节点区域划分
    if parent_direction == "east":
        child_x_division = [parent_division['x'][1], parent_division['x'][2], random.choice([20, 40, 60, 80])]
        child_y_division = parent_division['y']
    elif parent_direction == "west":
        child_x_division = [random.choice([20, 40, 60, 80]), parent_division['x'][0], parent_division['x'][1]]
        child_y_division = parent_division['y']
    elif parent_direction == "north":
        child_y_division = [random.choice([20, 40, 60, 80]), parent_division['y'][0], parent_division['y'][1]]
        child_x_division = parent_division['x']
    elif parent_direction == "south":
        child_y_division = [parent_division['y'][1], parent_division['y'][2], random.choice([20, 40, 60, 80])]
        child_x_division = parent_division['x']
    elif parent_direction == "northeast":
        child_x_division = [parent_division['x'][2], random.choice([20, 40, 60, 80]), random.choice([20, 40, 60, 80])]
        child_y_division = [random.choice([20, 40, 60, 80]), parent_division['y'][0], random.choice([20, 40, 60, 80])]
    elif parent_direction == "northwest":
        child_x_division = [random.choice([20, 40, 60, 80]), parent_division['x'][0], random.choice([20, 40, 60, 80])]
        child_y_division = [random.choice([20, 40, 60, 80]), parent_division['y'][0], random.choice([20, 40, 60, 80])]
    elif parent_direction == "southeast":
        child_x_division = [parent_division['x'][1], parent_division['x'][2], random.choice([20, 40, 60, 80])]
        child_y_division = [parent_division['y'][2], random.choice([20, 40, 60, 80]), random.choice([20, 40, 60, 80])]
    elif parent_direction == "southwest":
        child_x_division = [random.choice([20, 40, 60, 80]), parent_division['x'][0], parent_division['x'][1]]
        child_y_division = [parent_division['y'][1], parent_division['y'][2], random.choice([20, 40, 60, 80])]
    else:
        child_x_division = parent_division['x']
        child_y_division = parent_division['y']

    return {
        "x": child_x_division,
        "y": child_y_division
    }

def generate_layout(layout_data, size_presets, gap=7.5):
    """
    使用广度优先遍历生成布局，避免重叠。
    """
    size = 512
    bounding_boxes = {}
    directions = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"]

    # 根节点的初始位置和大小
    root_node = next(node for node in layout_data if node['relation'] == 0)
    root_position = (size / 2, size / 2)
    root_size = select_size_for_entity(root_node['entity'], size_presets, size, size)
    region_division = initialize_region_division(root_size[0], root_size[1])
    bounding_boxes[root_node['num']] = {
        "position": root_position,
        "size": root_size,
        "occupied": 0b00000000,
        "region_division": region_division
    }

    # 使用广度优先遍历来生成布局
    queue = deque([root_node])

    while queue:
        current_node = queue.popleft()
        parent_info = bounding_boxes[current_node['num']]
        parent_position = parent_info['position']
        parent_region_division = parent_info['region_division']

        # 查找当前节点的所有子节点
        for node in layout_data:
            if node['relation'] == current_node['num']:
                # 子节点的中心区域应与父节点的分区一致
                child_region_division = adjust_region_division_for_child(parent_region_division, node['position'])

                # 确定子节点的位置
                direction = node['position']
                if direction in directions:
                    i = directions.index(direction)
                    if not (parent_info['occupied'] & (1 << i)):
                        # 获取子节点的大小
                        child_size = select_size_for_entity(node['entity'], size_presets,
                                                            parent_info['size'][0], parent_info['size'][1])

                        # 计算子节点的位置
                        half_parent_width = parent_info['size'][0] / 2
                        half_parent_height = parent_info['size'][1] / 2
                        half_child_width = child_size[0] / 2
                        half_child_height = child_size[1] / 2

                        if direction == "east":
                            new_position = (parent_position[0] + half_parent_width + half_child_width + gap, parent_position[1])
                        elif direction == "west":
                            new_position = (parent_position[0] - half_parent_width - half_child_width - gap, parent_position[1])
                        elif direction == "north":
                            new_position = (parent_position[0], parent_position[1] + half_parent_height + half_child_height + gap)
                        elif direction == "south":
                            new_position = (parent_position[0], parent_position[1] - half_parent_height - half_child_height - gap)
                        elif direction == "northeast":
                            new_position = (parent_position[0] + half_parent_width + half_child_width + gap,
                                            parent_position[1] + half_parent_height + half_child_height + gap)
                        elif direction == "northwest":
                            new_position = (parent_position[0] - half_parent_width - half_child_width - gap,
                                            parent_position[1] + half_parent_height + half_child_height + gap)
                        elif direction == "southeast":
                            new_position = (parent_position[0] + half_parent_width + half_child_width + gap,
                                            parent_position[1] - half_parent_height - half_child_height - gap)
                        elif direction == "southwest":
                            new_position = (parent_position[0] - half_parent_width - half_child_width - gap,
                                            parent_position[1] - half_parent_height - half_child_height - gap)

                        # 更新占用情况，确保没有重叠
                        parent_info['occupied'] |= (1 << i)
                        bounding_boxes[node['num']] = {
                            "position": new_position,
                            "size": child_size,  # 子节点的区域大小
                            "occupied": 0b00000000,
                            "region_division": child_region_division
                        }
                        queue.append(node)

    return bounding_boxes


def draw_layout(bounding_boxes, layout_data, gap=7.5, road_width=2.5):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.invert_yaxis()  # 反转 y 轴以匹配 GeoTIFF 文件的方向
    ax.axis('off')

    colors = {'building': 'yellow', 'green': 'green', 'water': 'blue'}

    for num, info in bounding_boxes.items():
        entity = next(item for item in layout_data if item["num"] == num)
        shape = entity['shape']
        color = colors.get(entity['entity'], 'grey')
        x, y = info["position"]
        width, height = info["size"]

        # 绘制形状，填充到整个区域
        if shape == 'rec':
            rect = patches.Rectangle((x - width / 2, y - height / 2), width, height, linewidth=0, facecolor=color)
            ax.add_patch(rect)
        elif shape == 'ellipse':
            ellipse = patches.Ellipse((x, y), width, height, linewidth=0, facecolor=color)
            ax.add_patch(ellipse)

    # 生成道路
    roads, road_areas = generate_roads(bounding_boxes, gap, road_width)
    for road in roads:
        line = LineString(road)
        x, y = line.xy
        ax.plot(x, y, color='red', linewidth=road_width)

        # 保存图像时确保尺寸一致
    plt.savefig('layout_final.png', bbox_inches='tight', pad_inches=0, dpi=100)  # 调整 DPI 确保 600x600 像素
    #plt.show()

    # 保存到shapefile文件
    save_shapefiles(bounding_boxes, layout_data, roads)



def generate_roads(bounding_boxes, gap=7.5, road_width=2.5):
    """
    生成建筑之间的道路，确保相邻的建筑之间有一条道路，并使道路的中轴线位于区块边框外多半个道路宽度的位置。
    """
    roads = []
    road_areas = []  # 存储道路区域信息
    horizontal_roads = {}
    vertical_roads = {}

    # 道路偏移量，确保道路中轴线位于边框外多半个道路宽度
    offset = road_width / 2

    for num, info in bounding_boxes.items():
        position = info["position"]
        width, height = info["size"]
        x, y = position

        # 水平道路
        road_start_x = x - width / 2 - gap / 2 - offset
        road_end_x = x + width / 2 + gap / 2 + offset
        road_y1 = y - height / 2 - gap / 2 - offset
        road_y2 = y + height / 2 + gap / 2 + offset

        if road_y1 in horizontal_roads:
            horizontal_roads[road_y1].append((road_start_x, road_end_x))
        else:
            horizontal_roads[road_y1] = [(road_start_x, road_end_x)]

        if road_y2 in horizontal_roads:
            horizontal_roads[road_y2].append((road_start_x, road_end_x))
        else:
            horizontal_roads[road_y2] = [(road_start_x, road_end_x)]

        # 垂直道路
        road_start_y = y - height / 2 - gap / 2 - offset
        road_end_y = y + height / 2 + gap / 2 + offset
        road_x1 = x - width / 2 - gap / 2 - offset
        road_x2 = x + width / 2 + gap / 2 + offset

        if road_x1 in vertical_roads:
            vertical_roads[road_x1].append((road_start_y, road_end_y))
        else:
            vertical_roads[road_x1] = [(road_start_y, road_end_y)]

        if road_x2 in vertical_roads:
            vertical_roads[road_x2].append((road_start_y, road_end_y))
        else:
            vertical_roads[road_x2] = [(road_start_y, road_end_y)]

    # 处理重叠的道路
    for y, x_ranges in horizontal_roads.items():
        x_ranges.sort()
        merged_range = x_ranges[0]
        for start, end in x_ranges[1:]:
            if start <= merged_range[1]:
                merged_range = (merged_range[0], max(merged_range[1], end))
            else:
                roads.append([(merged_range[0], y), (merged_range[1], y)])
                road_areas.append({
                    "position": ((merged_range[0] + merged_range[1]) / 2, y),
                    "size": (merged_range[1] - merged_range[0], road_width)
                })
                merged_range = (start, end)
        roads.append([(merged_range[0], y), (merged_range[1], y)])
        road_areas.append({
            "position": ((merged_range[0] + merged_range[1]) / 2, y),
            "size": (merged_range[1] - merged_range[0], road_width)
        })

    for x, y_ranges in vertical_roads.items():
        y_ranges.sort()
        merged_range = y_ranges[0]
        for start, end in y_ranges[1:]:
            if start <= merged_range[1]:
                merged_range = (merged_range[0], max(merged_range[1], end))
            else:
                roads.append([(x, merged_range[0]), (x, merged_range[1])])
                road_areas.append({
                    "position": (x, (merged_range[0] + merged_range[1]) / 2),
                    "size": (road_width, merged_range[1] - merged_range[0])
                })
                merged_range = (start, end)
        roads.append([(x, merged_range[0]), (x, merged_range[1])])
        road_areas.append({
            "position": (x, (merged_range[0] + merged_range[1]) / 2),
            "size": (road_width, merged_range[1] - merged_range[0])
        })

    return roads, road_areas

def save_shapefiles(bounding_boxes, layout_data, roads):
    """
    按照不同语义类型保存shapefiles，包含building, water, green, road（线元素）。
    """
    # 创建输出目录
    output_dir = "shp"
    os.makedirs(output_dir, exist_ok=True)

    # 面元素文件
    shp_building = shapefile.Writer(os.path.join(output_dir, 'building.shp'), shapeType=shapefile.POLYGON)
    shp_water = shapefile.Writer(os.path.join(output_dir, 'water.shp'), shapeType=shapefile.POLYGON)
    shp_green = shapefile.Writer(os.path.join(output_dir, 'green.shp'), shapeType=shapefile.POLYGON)

    # 添加字段
    shp_building.field('ID', 'N')
    shp_water.field('ID', 'N')
    shp_green.field('ID', 'N')

    # 道路线元素文件
    shp_road = shapefile.Writer(os.path.join(output_dir, 'road.shp'), shapeType=shapefile.POLYLINE)
    shp_road.field('ID', 'N')

    # 定义中心点
    center_x, center_y = 512 / 2, 512 / 2

    # 保存每个形状到shapefile
    for num, info in bounding_boxes.items():
        entity = next(item for item in layout_data if item["num"] == num)
        x, y = info["position"]
        width, height = info["size"]

        # 调整坐标，使其以中心点为原点
        x -= center_x
        y = -(y - center_y) # 反转 Y 坐标

        polygon = Polygon([
            (x - width / 2, y - height / 2),
            (x + width / 2, y - height / 2),
            (x + width / 2, y + height / 2),
            (x - width / 2, y + height / 2)
        ])

        # 根据不同语义类型保存
        if entity['entity'] == 'building':
            shp_building.poly([list(polygon.exterior.coords)])
            shp_building.record(num)
        elif entity['entity'] == 'water':
            shp_water.poly([list(polygon.exterior.coords)])
            shp_water.record(num)
        elif entity['entity'] == 'green':
            shp_green.poly([list(polygon.exterior.coords)])
            shp_green.record(num)

    # 保存道路线元素
    for i, road in enumerate(roads):
        # 调整道路坐标，使其以中心点为原点，并反转 Y 坐标
        adjusted_road = [(x - center_x, -(y - center_y)) for (x, y) in road]
        shp_road.line([adjusted_road])
        shp_road.record(i)

    # 关闭文件
    shp_building.close()
    shp_water.close()
    shp_green.close()
    shp_road.close()

    # 为每个shapefile创建.prj文件
    prj_content = pyproj.CRS.from_epsg(4326).to_wkt()
    for shp_name in ['building', 'water', 'green', 'road']:
        with open(os.path.join(output_dir, f'{shp_name}.prj'), 'w') as prj_file:
            prj_file.write(prj_content)



def generate_height_map(bounding_boxes, road_areas, layout_size=512, preset_params=None):
    """
    基于多套Perlin噪声参数生成自然高度图，并应用语义约束。
    """
    # 定义预设的Perlin噪声参数
    if preset_params is None:
        preset_params = [
            {"scale": 50, "octaves": 2, "persistence": 0.3, "lacunarity": 1.5},
            {"scale": 80, "octaves": 4, "persistence": 0.5, "lacunarity": 2.0},
            {"scale": 100, "octaves": 3, "persistence": 0.6, "lacunarity": 2.5},
            {"scale": 120, "octaves": 5, "persistence": 0.7, "lacunarity": 3.0},
            {"scale": 90, "octaves": 2, "persistence": 0.4, "lacunarity": 1.8},
            {"scale": 70, "octaves": 3, "persistence": 0.8, "lacunarity": 2.2},
            {"scale": 110, "octaves": 6, "persistence": 0.3, "lacunarity": 3.5},
            {"scale": 60, "octaves": 4, "persistence": 0.9, "lacunarity": 2.3},
            {"scale": 130, "octaves": 7, "persistence": 0.5, "lacunarity": 2.7},
            {"scale": 40, "octaves": 1, "persistence": 0.2, "lacunarity": 1.2},
            {"scale": 75, "octaves": 5, "persistence": 0.6, "lacunarity": 2.8},
            {"scale": 140, "octaves": 6, "persistence": 0.4, "lacunarity": 3.2},
            {"scale": 85, "octaves": 4, "persistence": 0.7, "lacunarity": 2.9},
            {"scale": 95, "octaves": 3, "persistence": 0.8, "lacunarity": 2.1},
            {"scale": 120, "octaves": 8, "persistence": 0.6, "lacunarity": 3.0},
        ]

    # 随机选择一个参数配置
    noise_params = random.choice(preset_params)

    # 初始化高度图
    height_map = np.zeros((layout_size, layout_size))

    # 生成Perlin噪声基础图
    for i in range(layout_size):
        for j in range(layout_size):
            height_map[i][j] = pnoise2(i / noise_params["scale"],
                                       j / noise_params["scale"],
                                       octaves=noise_params["octaves"],
                                       persistence=noise_params["persistence"],
                                       lacunarity=noise_params["lacunarity"])

    # 归一化高度图
    height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min())

    # 应用语义约束
    for num, info in bounding_boxes.items():
        entity = next(item for item in layout_data if item["num"] == num)
        x, y = int(info["position"][0]), int(info["position"][1])
        width, height = int(info["size"][0]), int(info["size"][1])

        if entity['entity'] == 'building':
            height_map[y - height // 2:y + height // 2, x - width // 2:x + width // 2] = 0.2
        elif entity['entity'] == 'water':
            height_map[y - height // 2:y + height // 2, x - width // 2:x + width // 2] *= 0
        elif entity['entity'] == 'green':
            height_map[y - height // 2:y + height // 2, x - width // 2:x + width // 2] *= 0.5

    # 调整道路高度，使其与周围齐平
    buffer = 25  # 附近区域范围
    for road in road_areas:
        x, y = int(road["position"][0]), int(road["position"][1])
        width, height = int(road["size"][0]), int(road["size"][1])

        # 计算周围区域的平均高度
        surrounding_heights = height_map[max(0, y - height // 2 - buffer):min(layout_size, y + height // 2 + buffer),
                                         max(0, x - width // 2 - buffer):min(layout_size, x + width // 2 + buffer)]
        avg_height = np.mean(surrounding_heights[surrounding_heights > 0])  # 计算非零区域的平均高度

        # 将道路区域高度设置为周围平均高度
        height_map[y - height // 2:y + height // 2, x - width // 2:x + width // 2] = avg_height

        # 平滑过渡范围
        smooth_range = 60
        for i in range(max(0, y - height // 2 - buffer), min(layout_size, y + height // 2 + buffer)):
            for j in range(max(0, x - width // 2 - buffer), min(layout_size, x + width // 2 + buffer)):
                if height_map[i, j] != avg_height:  # 确保不是道路本身
                    distance_to_edge = min(abs(i - (y - height // 2)), abs(i - (y + height // 2)),
                                           abs(j - (x - width // 2)), abs(j - (x + width // 2)))
                    if distance_to_edge < smooth_range:
                        # 使用线性插值平滑过渡高度值
                        factor = distance_to_edge / smooth_range
                        height_map[i, j] = factor * height_map[i, j] + (1 - factor) * avg_height

    # 应用高斯模糊以平滑过渡区域
    height_map = gaussian_filter(height_map, sigma=5)

    return height_map



def generate_georeferenced_height_map(height_map, output_path="height_map_geotiff.tif", minX=0, maxY=512, maxX=512, minY=0):
    """
    使用 rasterio 将生成的高度图保存为带有地理参考信息的 GeoTIFF 文件。
    """
    # 归一化高度图并转换为 uint8 类型
    normalized_height_map = ((height_map - height_map.min()) / (height_map.max() - height_map.min()) * 255).astype(np.uint8)

    # 使用 Pillow 打开高度图
    height_map_img = Image.fromarray(normalized_height_map)

    # 保存为临时 PNG 文件
    temp_png_path = "temp_height_map.png"
    height_map_img.save(temp_png_path)

    # 定义仿射变换矩阵 (Affine Transformation Matrix)
    transform = Affine.translation(minX, minY) * Affine.scale((maxX - minX) / height_map.shape[1], (maxY - minY) / height_map.shape[0])

    # 使用 rasterio 保存 GeoTIFF（确保坐标系一致）
    with rio_open(output_path, 'w', driver='GTiff', height=height_map.shape[0], width=height_map.shape[1],
                  count=1, dtype=height_map.dtype, crs=CRS.from_epsg(4326), transform=transform) as dst:
        dst.write(height_map, 1)

    print(f"地理参考高度图已生成并保存为 {output_path}")


def plot_height_map(height_map):
    """
    绘制生成的高度图，并保存为无轴的png图像。
    """
    plt.figure(figsize=(6, 6))
    # 使用 origin='upper' 确保图像方向一致
    plt.imshow(height_map, cmap='terrain', origin='upper')
    plt.axis('off')  # 移除坐标轴
    plt.gca().set_position([0, 0, 1, 1])  # 去除所有边距
    plt.savefig('height_map.png', bbox_inches='tight', pad_inches=0)
    #plt.show()


if __name__ == "__main__":
    layout_data = load_json('layout.json')
    size_presets = initialize_size_presets()
    bounding_boxes = generate_layout(layout_data, size_presets)
    road_width = 5

    # 生成布局并返回道路信息
    roads, road_areas = generate_roads(bounding_boxes, gap=7.5, road_width=road_width)
    draw_layout(bounding_boxes, layout_data, road_width)  # 可调节的道路线宽
    print("布局已生成并保存为layout_final.png")

    # 生成高度图，使用road_areas来调整道路高度
    height_map = generate_height_map(bounding_boxes, road_areas)
    generate_georeferenced_height_map(height_map)
    plot_height_map(height_map)
    print("高度图已生成并保存为height_map.png")
    input("Press Enter to continue...")


