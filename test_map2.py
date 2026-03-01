import serial
import struct
import threading
import time
from datetime import datetime
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from collections import deque
import os

from crc8 import crc8
from crc_utils import crc_table

np.set_printoptions(suppress=True)

class PoseGraphNode:
    """Узел графа поз: хранит позу робота и связанный скан."""
    def __init__(self, pose, scan_angles, scan_ranges, timestamp):
        self.pose = pose  # (x, y, theta)
        self.scan_angles = scan_angles.copy() if scan_angles is not None else None
        self.scan_ranges = scan_ranges.copy() if scan_ranges is not None else None
        self.timestamp = timestamp
        self.id = None

class PoseGraphEdge:
    """Ребро графа поз: связь между двумя узлами."""
    def __init__(self, from_id, to_id, rel_pose, information_matrix=None):
        self.from_id = from_id
        self.to_id = to_id
        self.rel_pose = rel_pose  # (dx, dy, dtheta)
        # Матрица информации вес ребра. По умолчанию диагональная.
        if information_matrix is None:
            # Аналог infoMat = [1 0 0 1 0 1] из статьи
            self.information_matrix = np.diag([1.0, 1.0, 1.0])
        else:
            self.information_matrix = information_matrix

class PoseGraph:
    """ Граф поз для SLAM."""
    def __init__(self):
        self.nodes = [] # Список узлов
        self.edges = [] # Список ребер
        self.next_node_id = 0

    def add_node(self, pose, scan_angles=None, scan_ranges=None, timestamp=None):
        """Добавляет новый узел в граф."""
        node = PoseGraphNode(pose, scan_angles, scan_ranges, timestamp)
        node.id = self.next_node_id
        self.nodes.append(node)
        self.next_node_id += 1
        return node.id

    def add_edge(self, from_id, to_id, rel_pose, information_matrix=None):
        """Добавляет ребро между существующими узлами."""
        if from_id >= len(self.nodes) or to_id >= len(self.nodes):
            raise ValueError("Invalid node ID for edge")
        edge = PoseGraphEdge(from_id, to_id, rel_pose, information_matrix)
        self.edges.append(edge)

    def get_node_pose(self, node_id):
        """Возвращает позу узла."""
        if node_id < len(self.nodes):
            return self.nodes[node_id].pose
        return None

    def set_node_pose(self, node_id, pose):
        """Обновляет позу узла."""
        if node_id < len(self.nodes):
            self.nodes[node_id].pose = pose

    def get_num_nodes(self):
        return len(self.nodes)

    def optimize(self, num_iterations=50):
        """ Оптимизация графа поз методом Гаусса-Зейделяю. """
        if len(self.nodes) < 2:
            return
        for iteration in range(num_iterations):
            max_change = 0.0
            # Проходим по всем узлам (кроме первого, который закреплен)
            for i in range(1, len(self.nodes)):
                # Собираем все связи, в которых участвует узел i
                connected_edges = []
                for edge in self.edges:
                    if edge.from_id == i or edge.to_id == i:
                        connected_edges.append(edge)

                if not connected_edges:
                    continue

                # Вычисляем новую позу как среднее по всем связям
                new_x, new_y, new_theta = 0.0, 0.0, 0.0
                weight_sum = 0.0

                for edge in connected_edges:
                    # Определяем соседний узел и знак преобразования
                    if edge.from_id == i:
                        neighbor_id = edge.to_id
                        rel = edge.rel_pose
                        neighbor_pose = self.get_node_pose(neighbor_id)
                        if neighbor_pose is None:
                            continue
                        # Вычисляем предполагаемую позу узла i из соседа и rel_pose
                        suggested_x = neighbor_pose[0] - rel[0] * np.cos(neighbor_pose[2]) + rel[1] * np.sin(neighbor_pose[2])
                        suggested_y = neighbor_pose[1] - rel[0] * np.sin(neighbor_pose[2]) - rel[1] * np.cos(neighbor_pose[2])
                        suggested_theta = neighbor_pose[2] - rel[2]
                    else:  # edge.to_id == i
                        neighbor_id = edge.from_id
                        rel = edge.rel_pose
                        neighbor_pose = self.get_node_pose(neighbor_id)
                        if neighbor_pose is None:
                            continue
                        suggested_x = neighbor_pose[0] + rel[0] * np.cos(neighbor_pose[2]) - rel[1] * np.sin(neighbor_pose[2])
                        suggested_y = neighbor_pose[1] + rel[0] * np.sin(neighbor_pose[2]) + rel[1] * np.cos(neighbor_pose[2])
                        suggested_theta = neighbor_pose[2] + rel[2]

                    # Вес ребра из информационной матрицы
                    weight = np.trace(edge.information_matrix) / 3.0

                    new_x += suggested_x * weight
                    new_y += suggested_y * weight
                    new_theta += suggested_theta * weight
                    weight_sum += weight

                if weight_sum > 0:
                    new_x /= weight_sum
                    new_y /= weight_sum
                    new_theta /= weight_sum

                    # Нормализация угла
                    new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))

                    current_pose = self.get_node_pose(i)
                    change = np.sqrt((new_x - current_pose[0])**2 +
                                    (new_y - current_pose[1])**2 +
                                    (new_theta - current_pose[2])**2)
                    max_change = max(max_change, change)

                    self.set_node_pose(i, (new_x, new_y, new_theta))

            if max_change < 1e-6:
                break

def match_scans(scan1_angles, scan1_ranges, scan2_angles, scan2_ranges,
                max_range=8.0, resolution=0.1):
    """
    Сопоставление двух сканов для оценки относительного положения.
    Возвращает: (dx, dy, dtheta) - относительное положение scan2 относительно scan1.
    """
    if scan1_angles is None or scan2_angles is None:
        return (0, 0, 0)

    # Преобразуем в декартовы координаты
    x1 = scan1_ranges * np.cos(scan1_angles)
    y1 = scan1_ranges * np.sin(scan1_angles)
    x2 = scan2_ranges * np.cos(scan2_angles)
    y2 = scan2_ranges * np.sin(scan2_angles)

    # Фильтруем точки с некорректной дальностью
    valid1 = (scan1_ranges > 0.1) & (scan1_ranges < max_range)
    valid2 = (scan2_ranges > 0.1) & (scan2_ranges < max_range)

    if not np.any(valid1) or not np.any(valid2):
        return (0, 0, 0)

    x1 = x1[valid1]
    y1 = y1[valid1]
    x2 = x2[valid2]
    y2 = y2[valid2]

    # Поиск соответствий по ближайшим точкам
    best_score = -1
    best_pose = (0, 0, 0)

    # Дискретный поиск по сетке
    for dx in np.arange(-1.0, 1.0, resolution):
        for dy in np.arange(-1.0, 1.0, resolution):
            for dtheta in np.arange(-0.5, 0.5, 0.1):
                # Трансформируем второй скан
                x2_trans = x2 * np.cos(dtheta) - y2 * np.sin(dtheta) + dx
                y2_trans = x2 * np.sin(dtheta) + y2 * np.cos(dtheta) + dy

                # Считаем количество точек, которые совпадают
                score = 0
                for j in range(len(x2_trans)):
                    # Ищем ближайшую точку в первом скане
                    dist2 = (x1 - x2_trans[j])**2 + (y1 - y2_trans[j])**2
                    if np.min(dist2) < 0.1**2:  # порог 10 см
                        score += 1

                if score > best_score:
                    best_score = score
                    best_pose = (dx, dy, dtheta)

    return best_pose


#FASTSLAM С ГРАФОМ ПОЗ

class Particle:
    """Частица для FastSLAM."""
    def __init__(self, x=0.0, y=0.0, theta=0.0, map_size=15.0, grid_size=600):
        self.x = x
        self.y = y
        self.theta = theta
        self.map_size = map_size
        self.grid_size = grid_size
        self.resolution = map_size / grid_size
        self.map = np.zeros((grid_size, grid_size))
        self.log_odds = np.zeros((grid_size, grid_size))
        self.weight = 1.0
        self.laser_max_range = 8.0
        self.laser_min_range = 0.1

        # Граф поз для этой частицы
        self.pose_graph = PoseGraph()
        # Добавляем начальный узел
        self.pose_graph.add_node((x, y, theta))
        self.last_node_id = 0

    def copy(self):
        new = Particle(self.x, self.y, self.theta, self.map_size, self.grid_size)
        new.map = self.map.copy()
        new.log_odds = self.log_odds.copy()
        new.weight = self.weight
        new.pose_graph = self.pose_graph  # Разделяем граф
        new.last_node_id = self.last_node_id
        return new

    def move(self, dx, dy, dtheta, noise):
        """Обновление положения с шумом."""
        dx_noisy = dx + random.gauss(0, noise[0])
        dy_noisy = dy + random.gauss(0, noise[1])
        dtheta_noisy = dtheta + random.gauss(0, noise[2])

        self.x += dx_noisy * np.cos(self.theta) - dy_noisy * np.sin(self.theta)
        self.y += dx_noisy * np.sin(self.theta) + dy_noisy * np.cos(self.theta)
        self.theta += dtheta_noisy
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        return self

    def add_scan_node(self, scan_angles, scan_ranges, timestamp):
        """Добавляет новый узел в граф поз с текущим сканом."""
        current_pose = (self.x, self.y, self.theta)
        new_id = self.pose_graph.add_node(current_pose, scan_angles, scan_ranges, timestamp)

        # Добавляем ребро от предыдущего узла
        if hasattr(self, 'last_node_id'):
            prev_pose = self.pose_graph.get_node_pose(self.last_node_id)
            if prev_pose:
                # Вычисляем относительное перемещение
                dx = self.x - prev_pose[0]
                dy = self.y - prev_pose[1]
                dtheta = self.theta - prev_pose[2]
                dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
                self.pose_graph.add_edge(self.last_node_id, new_id, (dx, dy, dtheta))

        self.last_node_id = new_id
        return new_id

    def detect_loop_closure(self, current_scan_angles, current_scan_ranges,
                            search_radius=2.0, threshold=0.7):
        """
        Обнаружение замыкания цикла.
        """
        if len(self.pose_graph.nodes) < 10:
            return None, None

        best_match = None
        best_pose = None
        best_score = threshold

        # Ищем среди предыдущих узлов
        for i, node in enumerate(self.pose_graph.nodes[:-5]):
            if node.scan_angles is None:
                continue

            # Вычисляем расстояние до текущей позиции
            dist = np.sqrt((self.x - node.pose[0])**2 + (self.y - node.pose[1])**2)

            # Если узел в радиусе поиска
            if dist < search_radius:
                # Сопоставляем текущий скан со сканом узла
                rel_pose = match_scans(
                    current_scan_angles, current_scan_ranges,
                    node.scan_angles, node.scan_ranges,
                    self.laser_max_range, self.resolution
                )

                # Оцениваем качество совпадения по величине перемещения
                score = 1.0 / (1.0 + abs(rel_pose[0]) + abs(rel_pose[1]) + abs(rel_pose[2]))

                if score > best_score:
                    best_score = score
                    best_match = i
                    best_pose = rel_pose

        if best_match is not None:
            return best_match, best_pose
        return None, None

    def update_map(self, scan_angles, scan_ranges):
        """Обновление карты частицы."""
        for rel_angle, z in zip(scan_angles, scan_ranges):
            if z < self.laser_min_range or z > self.laser_max_range:
                continue

            global_angle = self.theta + rel_angle
            dx = np.cos(global_angle)
            dy = np.sin(global_angle)

            step_size = self.resolution
            max_steps = int(min(z, self.laser_max_range) / step_size)

            l_occupied = 0.5
            l_free = -0.2
            l_min = -10.0
            l_max = 10.0

            for step in range(1, max_steps + 1):
                check_x = self.x + step * step_size * dx
                check_y = self.y + step * step_size * dy

                grid_x = int((check_x + self.map_size/2) / self.resolution)
                grid_y = int((check_y + self.map_size/2) / self.resolution)

                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    if step < max_steps:
                        self.log_odds[grid_x, grid_y] += l_free
                    else:
                        self.log_odds[grid_x, grid_y] += l_occupied

                    self.log_odds[grid_x, grid_y] = np.clip(
                        self.log_odds[grid_x, grid_y], l_min, l_max
                    )

        self.map = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def get_position(self):
        return (self.x, self.y, self.theta)


class FastSLAM:
    """FastSLAM с графом поз и обнаружением циклов."""
    def __init__(self, num_particles=100, map_size=15.0, grid_size=600):
        self.num_particles = num_particles
        self.map_size = map_size
        self.grid_size = grid_size
        self.resolution = map_size / grid_size
        self.motion_noise = [0.02, 0.02, 0.01]
        self.particles = []
        self.best_particle = None
        self.best_map = None
        self.best_trajectory = []
        self.step_count = 0

        # Параметры для обнаружения циклов
        self.loop_closure_threshold = 0.7
        self.loop_closure_search_radius = 2.0

        # Инициализация частиц
        for i in range(num_particles):
            x = random.gauss(0, 0.05)
            y = random.gauss(0, 0.05)
            theta = random.gauss(0, 0.02)
            self.particles.append(Particle(x, y, theta, map_size, grid_size))

        self.best_particle = self.particles[0]
        self.best_map = self.best_particle.map

    def predict(self, dx, dy, dtheta):
        """Обновление положения всех частиц."""
        for p in self.particles:
            p.move(dx, dy, dtheta, self.motion_noise)

    def add_scan_to_graph(self, scan_angles, scan_ranges, timestamp):
        """Добавляет текущий скан в граф поз каждой частицы."""
        for p in self.particles:
            p.add_scan_node(scan_angles, scan_ranges, timestamp)

    def detect_and_add_loop_closures(self, scan_angles, scan_ranges):
        """Обнаруживает замыкания цикла и добавляет ребра в граф."""
        for p in self.particles:
            match_id, rel_pose = p.detect_loop_closure(
                scan_angles, scan_ranges,
                self.loop_closure_search_radius,
                self.loop_closure_threshold
            )

            if match_id is not None and rel_pose is not None:
                # Добавляем ребро замыкания цикла в граф
                current_id = p.last_node_id
                p.pose_graph.add_edge(current_id, match_id, rel_pose)
                print(f"🔄 Обнаружено замыкание цикла: узел {current_id} -> {match_id}")

    def optimize_pose_graphs(self):
        """Оптимизирует графы поз всех частиц."""
        for p in self.particles:
            p.pose_graph.optimize()
            # Обновляем текущую позицию из оптимизированного графа
            current_pose = p.pose_graph.get_node_pose(p.last_node_id)
            if current_pose:
                p.x, p.y, p.theta = current_pose

    def update_maps(self, scan_angles, scan_ranges):
        """Обновление карт всех частиц."""
        for p in self.particles:
            p.update_map(scan_angles, scan_ranges)

    def update_weights(self, scan_angles, scan_ranges):
        """Обновление весов частиц."""
        # Пока оставляем равные веса и берем первую частицу как лучшую
        for p in self.particles:
            p.weight = 1.0 / len(self.particles)

        self.best_particle = self.particles[0]
        self.best_map = self.best_particle.map
        self.best_trajectory.append(self.best_particle.get_position())

    def step(self, dx, dy, dtheta, scan_angles, scan_ranges, timestamp):
        """Один шаг алгоритма FastSLAM с графом поз."""
        self.step_count += 1
        # Предсказание
        self.predict(dx, dy, dtheta)

        if scan_angles is not None and len(scan_angles) > 0:
            # Добавляем скан в граф
            self.add_scan_to_graph(scan_angles, scan_ranges, timestamp)

            # Обнаружение циклов (каждые 20 шагов)
            if self.step_count % 20 == 0 and self.step_count > 30:
                self.detect_and_add_loop_closures(scan_angles, scan_ranges)

            # Оптимизация графа (каждые 50 шагов)
            if self.step_count % 50 == 0:
                self.optimize_pose_graphs()

            # Обновление карт
            self.update_maps(scan_angles, scan_ranges)

            # Обновление весов
            self.update_weights(scan_angles, scan_ranges)

    def get_map(self):
        return self.best_map

    def get_position(self):
        if self.best_particle:
            return self.best_particle.get_position()
        return (0, 0, 0)

    def save_map(self, filename):
        """Сохраняет карту и граф поз лучшей частицы."""
        if self.best_map is not None:
            np.save(filename, self.best_map)
            print(f"Карта сохранена как {filename}")

            # Сохраняем граф поз лучшей частицы
            if hasattr(self.best_particle, 'pose_graph'):
                graph_file = filename.replace('.npy', '_graph.npy')
                # Сохраняем только позы узлов для простоты
                graph_data = np.array([node.pose for node in self.best_particle.pose_graph.nodes])
                np.save(graph_file, graph_data)
                print(f"Граф поз сохранен как {graph_file}")

            # Визуализация
            plt.figure(figsize=(10, 8))
            plt.imshow(self.best_map.T, origin='lower', cmap='hot', vmin=0, vmax=1)
            plt.colorbar(label='Вероятность занятости')
            plt.title('Карта FastSLAM с графом поз')
            plt.xlabel('X (пиксели)')
            plt.ylabel('Y (пиксели)')
            plt.savefig(filename.replace('.npy', '.png'), dpi=300)
            plt.close()

    def save_trajectory(self, filename):
        np.save(filename, np.array(self.best_trajectory))
        print(f"Траектория сохранена как {filename}")


class SLAMVisualizer:
    """Визуализация карты и графа поз."""
    def __init__(self, fastslam, map_size=15.0):
        self.fastslam = fastslam
        self.map_size = map_size
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.running = False
        self.thread = None
        self.last_save_time = time.time()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._visualization_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.fig:
            plt.close(self.fig)

    def _visualization_loop(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 7))

        while self.running:
            self.ax1.clear()
            self.ax2.clear()

            best_map = self.fastslam.get_map()
            best_particle = self.fastslam.best_particle

            # Левая панель: карта
            if best_map is not None:
                im = self.ax1.imshow(best_map.T, origin='lower', cmap='hot',
                                    vmin=0, vmax=1,
                                    extent=[-self.map_size/2, self.map_size/2,
                                           -self.map_size/2, self.map_size/2])
                plt.colorbar(im, ax=self.ax1, fraction=0.046, pad=0.04)

                # Траектория
                if len(self.fastslam.best_trajectory) > 1:
                    traj = np.array(self.fastslam.best_trajectory)
                    self.ax1.plot(traj[:, 0], traj[:, 1], 'g-', linewidth=2, alpha=0.7)

                # Текущее положение
                x, y, theta = self.fastslam.get_position()
                self.ax1.plot(x, y, 'ro', markersize=8)
                self.ax1.arrow(x, y, 0.3*np.cos(theta), 0.3*np.sin(theta),
                              head_width=0.15, head_length=0.15, fc='r', ec='r')

            self.ax1.set_title(f'FastSLAM карта (шаг {self.fastslam.step_count})')
            self.ax1.set_xlabel('X (м)')
            self.ax1.set_ylabel('Y (м)')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.set_aspect('equal')

            
            if best_particle and hasattr(best_particle, 'pose_graph'):
                pg = best_particle.pose_graph

                for node in pg.nodes:
                    self.ax2.plot(node.pose[0], node.pose[1], 'b.', markersize=5)

                for edge in pg.edges:
                    if abs(edge.from_id - edge.to_id) == 1:
                        from_pose = pg.get_node_pose(edge.from_id)
                        to_pose = pg.get_node_pose(edge.to_id)
                        if from_pose and to_pose:
                            self.ax2.plot([from_pose[0], to_pose[0]],
                                         [from_pose[1], to_pose[1]], 'c-', alpha=0.5)

                for edge in pg.edges:
                    if abs(edge.from_id - edge.to_id) > 1:
                        from_pose = pg.get_node_pose(edge.from_id)
                        to_pose = pg.get_node_pose(edge.to_id)
                        if from_pose and to_pose:
                            self.ax2.plot([from_pose[0], to_pose[0]],
                                         [from_pose[1], to_pose[1]], 'r-', linewidth=2, alpha=0.7)

                # Текущий узел
                if hasattr(best_particle, 'last_node_id'):
                    current_pose = pg.get_node_pose(best_particle.last_node_id)
                    if current_pose:
                        self.ax2.plot(current_pose[0], current_pose[1], 'r*', markersize=12)

                self.ax2.set_title(f'Граф поз (узлов: {pg.get_num_nodes()})')
                self.ax2.set_xlabel('X (м)')
                self.ax2.set_ylabel('Y (м)')
                self.ax2.grid(True, alpha=0.3)
                self.ax2.set_aspect('equal')

            # Автосохранение каждые 30 секунд
            if time.time() - self.last_save_time > 30:
                self.fastslam.save_map(f"autosave_map_{int(time.time())}.npy")
                self.last_save_time = time.time()

            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)

        plt.ioff()


def compute_odometry(robot_info, prev_th_l, prev_th_r, wheel_radius, wheel_base):
    """Вычисление перемещения из данных одометрии."""
    if robot_info["th_l"] is None or robot_info["th_r"] is None:
        return 0, 0, 0, prev_th_l, prev_th_r

    d_left = wheel_radius * (robot_info["th_l"] - prev_th_l)
    d_right = wheel_radius * (robot_info["th_r"] - prev_th_r)

    if abs(d_left) > 0.1 or abs(d_right) > 0.1:
        return 0, 0, 0, robot_info["th_l"], robot_info["th_r"]

    d_center = (d_right + d_left) / 2.0
    d_theta = (d_right - d_left) / wheel_base

    return d_center, 0, d_theta, robot_info["th_l"], robot_info["th_r"]

class LIDAR:
    def __init__(self, serial_port, baudrate):
        self.PACKET_LENGTH = 49
        self.POINT_PER_PACK = 12
        self.serial_conn = serial.Serial(serial_port, baudrate=baudrate, timeout=1)

    def calculate_crc8(self, data):
        crc = 0x00
        for byte in data:
            crc = crc_table[(crc ^ byte) & 0xFF]
        return crc

    def parse_packet(self, packet):
        if len(packet) != self.PACKET_LENGTH:
            return

        if packet[0] != 0x54 or packet[1] != 0x2C:
            return

        received_crc = packet[self.PACKET_LENGTH - 3]
        calculated_crc = self.calculate_crc8(packet[: self.PACKET_LENGTH - 3])
        if received_crc != calculated_crc:
            return

        header, ver_len, speed, start_angle = struct.unpack("<BBHH", packet[:6])
        end_angle, timestamp = struct.unpack("<HH", packet[42:46])

        distances = np.zeros((self.POINT_PER_PACK,))
        intensities = np.zeros((self.POINT_PER_PACK,))
        angles = np.zeros((self.POINT_PER_PACK,))

        start_angle = (start_angle % 36000) / 100.0
        end_angle = (end_angle % 36000) / 100.0

        angle_diff = (end_angle - start_angle + 360.0) % 360.0
        angle_increment = angle_diff / 11

        offset = 6
        for i in range(self.POINT_PER_PACK):
            distance, intensity = struct.unpack("<HB", packet[offset: offset + 3])
            distances[i] = distance
            intensities[i] = intensity
            angles[i] = (start_angle + i * angle_increment) % 360.0
            offset += 3

        return angles, distances

    def read_lidar_data(self):
        packet = self.serial_conn.read_until(b"\x54\x2C")
        if len(packet) != (self.PACKET_LENGTH - 2):
            return
        packet = bytes(b"\x54\x2C") + packet
        if len(packet) == self.PACKET_LENGTH:
            if polar_coord := self.parse_packet(packet):
                info = np.stack(polar_coord).T
                return info

    def close_serial_connection(self):
        self.serial_conn.close()


def read_packet(ser):
    chunk = ser.read_until(b"\x7E")
    if len(chunk) != 17:
        return
    values = struct.unpack("<ffffx", chunk)
    return values


def motion_control(lidar_info, side="right", max_speed=5, safe_dist=0.5):
    theta = lidar_info["theta"]
    length = lidar_info["length"]

    if theta is None or length is None:
        return 0, 0

    theta = np.array(theta)
    length = np.array(length)

    forward_idx = (theta > -15*math.pi/180) & (theta < 15*math.pi/180)
    left_idx = (theta > 75*math.pi/180) & (theta < 105*math.pi/180)
    right_idx = (theta > -105*math.pi/180) & (theta < -75*math.pi/180)

    dist_forward = np.min(length[forward_idx]) if np.any(forward_idx) else 10
    dist_left = np.min(length[left_idx]) if np.any(left_idx) else 10
    dist_right = np.min(length[right_idx]) if np.any(right_idx) else 10

    wl_star = max_speed
    wr_star = max_speed

    if side == "right":
        if dist_right > safe_dist:
            wl_star = max_speed
            wr_star = max_speed * 0.8
        else:
            wl_star = max_speed * 0.8
            wr_star = max_speed
    else:
        if dist_left > safe_dist:
            wl_star = max_speed * 0.8
            wr_star = max_speed
        else:
            wl_star = max_speed
            wr_star = max_speed * 0.8

    if dist_forward < safe_dist:
        if side == "right":
            wl_star = max_speed * 0.8
            wr_star = -max_speed * 0.8
        else:
            wl_star = -max_speed * 0.8
            wr_star = max_speed * 0.8

    return wl_star, wr_star


def robot_process(info: dict):
    T = 1/50
    with serial.Serial(
        "/dev/ttyAMA0", 115200, bytesize=8,
        stopbits=serial.STOPBITS_ONE, timeout=1000
    ) as ser:
        while info["state"]:
            t1 = time.time()

            values = read_packet(ser)
            if values is None:
                continue

            info["th_l"], info["th_r"], info["w_l"], info["w_r"] = values
            info["time"] = time.time()

            buf = struct.pack("<ff", info["wl_star"], info["wr_star"])
            crc = crc8()
            crc.update(buf)
            buf = b"\x7E" + buf + crc.digest()
            ser.write(buf)
            ser.flush()

            dtime = T - (time.time() - t1)
            if dtime > 0:
                time.sleep(dtime)


def lidar_process(info: dict):
    lidar = LIDAR(serial_port="/dev/ttyUSB0", baudrate=230400)
    prev_end_angle = 0
    end_angle = 0
    laser_scan = []
    while info["state"]:
        data = lidar.read_lidar_data()
        if data is None:
            continue

        prev_end_angle = end_angle
        end_angle = data[-1, 0]

        if not (prev_end_angle > 180 and end_angle < 180):
            laser_scan.append(data)
            continue

        laser_scan.append(data)
        scan_array = np.vstack(laser_scan)
        laser_scan = []

        th = scan_array[:, 0] * np.pi / 180
        l = scan_array[:, 1] / 1000

        info["theta"] = th
        info["length"] = l
        info["time"] = time.time()


if __name__ == "__main__":
    robot_info = {
        "state": True, "th_l": None, "th_r": None,
        "w_l": None, "w_r": None,
        "wl_star": 0, "wr_star": 0, "time": time.time()
    }
    lidar_info = {"state": True, "theta": None, "length": None, "time": time.time()}

    fastslam = FastSLAM(num_particles=100, map_size=15.0, grid_size=600)
    visualizer = SLAMVisualizer(fastslam, map_size=15.0)

    robot_thread = threading.Thread(target=robot_process, args=(robot_info,))
    lidar_thread = threading.Thread(target=lidar_process, args=(lidar_info,))

    robot_thread.start()
    lidar_thread.start()

    print("Waiting for data...")
    while robot_info["th_l"] is None or lidar_info["theta"] is None:
        time.sleep(0.1)

    print("Start moving with Pose Graph SLAM")

    wheel_radius = 0.021
    wheel_base = 0.128
    prev_th_l = robot_info["th_l"]
    prev_th_r = robot_info["th_r"]

    visualizer.start()

    try:
        T = 1/50
        scan_counter = 0

        while True:
            t1 = time.time()

            wl_star, wr_star = motion_control(lidar_info, side="left", max_speed=5, safe_dist=0.5)
            robot_info["wl_star"] = wl_star
            robot_info["wr_star"] = wr_star

            #ОБНОВЛЕНИЕ FASTSLAM
            if lidar_info["theta"] is not None and len(lidar_info["theta"]) > 0:
                scan_counter += 1
                if scan_counter % 2 == 0:
                    dx, dy, dtheta, prev_th_l, prev_th_r = compute_odometry(
                        robot_info, prev_th_l, prev_th_r, wheel_radius, wheel_base)

                    fastslam.step(dx, dy, dtheta,
                                 lidar_info["theta"],
                                 lidar_info["length"],
                                 lidar_info["time"])

            dtime = T - (time.time() - t1)
            if dtime > 0:
                time.sleep(dtime)

    except KeyboardInterrupt:
        print("\nStopping robot")
        robot_info["wl_star"] = 0
        robot_info["wr_star"] = 0
        time.sleep(1)
        robot_info["state"] = False
        lidar_info["state"] = False

        visualizer.stop()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"slam_results_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        fastslam.save_map(f"{save_dir}/map.npy")
        fastslam.save_trajectory(f"{save_dir}/trajectory.npy")

        print(f"Результаты сохранены в папку: {save_dir}")

    finally:
        robot_thread.join()
        lidar_thread.join()
        print("All threads finished")
